"""阶段 2 · 第 3 课：run_once 和断点续跑的离线测试，不调用 API。

运行：uv run python -m stage2.test_persist
"""
import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import main as stage1
from stage2 import persist
from stage2.agent import build_graph
from stage2.persist import Ledger, has_side_effects, is_unfinished, run_once
from stage2.test_agent import ai
from stage2.test_basic import ScriptedModel, call

stage1.TOOL_LATENCY = 0
CALL = call("restart_service", {"host": "web-01", "service": "nginx"}, "c1")


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


class Killed(BaseException):
    """模拟进程被杀：继承 BaseException，ToolNode 的错误处理（只管 Exception）拦不住它。"""


class Executor:
    """假的工具执行函数，记录被调用了几次。"""

    def __init__(self, fail: BaseException | None = None, status: str = "success"):
        self.count, self.fail, self.status = 0, fail, status

    async def __call__(self) -> ToolMessage:
        self.count += 1
        if self.fail:
            raise self.fail
        return ToolMessage("nginx 已重启", tool_call_id=CALL["id"], name=CALL["name"], status=self.status)


async def attempt(ledger, key, execute):
    try:
        return await run_once(ledger, key, CALL, execute), None
    except BaseException as e:  # noqa: BLE001
        return None, e


async def main() -> None:
    results = []

    # ---------- 工具分类 ----------
    print("-- 工具分类 --")
    results.append(check("只读工具和 record_finding 不需要记账",
                         not any(has_side_effects(n) for n in ("get_logs", "check_disk", "record_finding"))))
    results.append(check("restart_service 需要记账", has_side_effects("restart_service")))
    results.append(check("没登记过的新工具：默认按有副作用处理", has_side_effects("delete_pod")))

    # ---------- run_once ----------
    print("\n-- run_once --")
    ledger, ex = Ledger(":memory:"), Executor()
    msg, err = await attempt(ledger, "t:c1", ex)
    ok = err is None and isinstance(msg, ToolMessage)
    results.append(check("第一次：执行，返回工具结果",
                         ok and ex.count == 1 and msg.content == "nginx 已重启", f"{err!r} 执行 {ex.count} 次"))
    entry = ledger.get("t:c1")
    results.append(check("执行后账本里记下了结果", bool(entry and entry.done and entry.content == "nginx 已重启"),
                         f"账本: {entry}"))

    msg, err = await attempt(ledger, "t:c1", ex)
    ok = err is None and isinstance(msg, ToolMessage)
    results.append(check("同一个 key 再来一次：不执行，返回账本里的结果，配对不变",
                         ok and ex.count == 1 and msg.content == "nginx 已重启" and msg.tool_call_id == "c1",
                         f"{err!r} 执行 {ex.count} 次"))

    msg, err = await attempt(ledger, "t:c2", ex)
    results.append(check("不同的 key：正常执行", err is None and ex.count == 2, f"{err!r} 执行 {ex.count} 次"))

    ledger, ex = Ledger(":memory:"), Executor(status="error")
    await attempt(ledger, "t:c1", ex)
    msg, err = await attempt(ledger, "t:c1", ex)
    results.append(check("工具返回的是 error：再来一次仍然不执行，status 也照原样返回",
                         err is None and isinstance(msg, ToolMessage) and ex.count == 1 and msg.status == "error",
                         f"{err!r} 执行 {ex.count} 次 status={getattr(msg, 'status', None)}"))

    # 执行到一半进程没了：账本里只有"开始"，没有结果
    ledger, ex = Ledger(":memory:"), Executor(fail=Killed())
    _, err = await attempt(ledger, "t:c1", ex)
    results.append(check("执行中途被杀：异常原样抛出（run_once 不吞掉它）", isinstance(err, Killed), f"实际: {err!r}"))
    ex2 = Executor()
    msg, err = await attempt(ledger, "t:c1", ex2)
    ok = err is None and isinstance(msg, ToolMessage)
    results.append(check("恢复后同一个 key：结果未知，不再执行",
                         ok and ex2.count == 0, f"{err!r} 执行 {ex2.count} 次"))
    results.append(check("……并且返回 error 状态，内容告诉模型该怎么办",
                         ok and msg.status == "error" and len(msg.content) >= 10 and msg.tool_call_id == "c1",
                         f"status={getattr(msg, 'status', None)} content={getattr(msg, 'content', None)!r}"))

    # ---------- 图：断点续跑 ----------
    print("\n-- 断点续跑 --")
    persist.RESTARTS.clear()
    killed = {"armed": True}
    real_check_disk = stage1.check_disk

    async def flaky_check_disk(host: str) -> dict:
        await asyncio.sleep(0.05)  # 让 restart_service 先执行完
        if killed["armed"]:
            killed["armed"] = False
            raise Killed()
        return await real_check_disk(host)

    stage1.check_disk = flaky_check_disk
    model = ScriptedModel([
        ai([call("restart_service", {"host": "web-01", "service": "nginx"}, "r1"),
            call("check_disk", {"host": "web-01"}, "d1")]),
        ai(text="已重启，磁盘 91%"),
    ])
    try:
        graph = build_graph(model, InMemorySaver(), Ledger(":memory:"))
        config = {"configurable": {"thread_id": "t1"}}
        try:
            # 第 4 课之后：重启要先过审批，图会停在 approve 节点；批准后才进入 tools
            await graph.ainvoke({"messages": [HumanMessage("重启 nginx 并看磁盘")]}, config)
            await graph.ainvoke(Command(resume={"r1": {"action": "approve"}}), config)
            crashed = False
        except Killed:
            crashed = True
        snapshot = await graph.aget_state(config)
        results.append(check("tools 节点中途被杀：快照停在 tools，重启已经执行了 1 次",
                             crashed and snapshot.next == ("tools",) and len(persist.RESTARTS) == 1,
                             f"crashed={crashed} next={snapshot.next} 重启 {len(persist.RESTARTS)} 次"))

        out = await graph.ainvoke(None, config)  # None：从断点继续
        tools = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("恢复后跑完：模型一共只被调用 2 次（第一次的回复没有重新生成）",
                             len(model.calls) == 2 and out["messages"][-1].content == "已重启，磁盘 91%",
                             f"模型调用 {len(model.calls)} 次"))
        results.append(check("恢复后重启没有再执行一次，模型拿到的是第一次的结果",
                             len(persist.RESTARTS) == 1 and "已重启" in tools["r1"].content,
                             f"重启 {len(persist.RESTARTS)} 次，r1={tools['r1'].content!r}"))
        results.append(check("只读的 check_disk 重新执行了，拿到结果", "91" in tools["d1"].content, tools["d1"].content))
    except BaseException as e:  # noqa: BLE001
        results.append(check("断点续跑", False, f"{type(e).__name__}: {str(e)[:200]}"))
    finally:
        stage1.check_disk = real_check_disk

    # 真实运行时遇到的情况：agent 节点的结果已保存，但下一个 checkpoint 还没写，进程就没了
    class KillingSaver(InMemorySaver):
        armed = True

        async def aput(self, config, checkpoint, metadata, new_versions):
            if self.armed and metadata.get("step") == 3:  # agent 节点之后的那个 checkpoint
                KillingSaver.armed = False
                raise Killed()  # 这时 agent 的结果（writes）已经保存了，只差这个 checkpoint
            return await super().aput(config, checkpoint, metadata, new_versions)

    model = ScriptedModel([ai([call("check_disk", {"host": "web-01"}, "d1")]), ai(text="磁盘 91%")])
    try:
        graph, config = build_graph(model, KillingSaver()), {"configurable": {"thread_id": "t2"}}
        try:
            # durability="sync"：每个 checkpoint 写完才执行下一步。默认是 "async"（后台写，图不等它），
            # 那样这里的"被杀"要等图跑完才会被发现，没法稳定复现
            await graph.ainvoke({"messages": [HumanMessage("看磁盘")]}, config, durability="sync")
        except Killed:
            pass
        snapshot = await graph.aget_state(config)
        results.append(check("节点结果已保存、下一个 checkpoint 没写：snapshot.next 为空，但仍被识别为没跑完",
                             snapshot.next == () and is_unfinished(snapshot),
                             f"next={snapshot.next} tasks={[t.name for t in snapshot.tasks]}"))
        out = await graph.ainvoke(None, config)
        done = await graph.aget_state(config)
        results.append(check("从这里恢复：已保存的模型回复不重新生成，跑完后识别为已完成",
                             len(model.calls) == 2 and out["messages"][-1].content == "磁盘 91%" and not is_unfinished(done),
                             f"模型调用 {len(model.calls)} 次，is_unfinished={is_unfinished(done)}"))
    except BaseException as e:  # noqa: BLE001
        results.append(check("checkpoint 没写完时的恢复", False, f"{type(e).__name__}: {str(e)[:200]}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
