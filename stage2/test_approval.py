"""阶段 2 · 第 4 课：审批的离线测试，不调用 API。

运行：uv run python -m stage2.test_approval
"""
import asyncio
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from loguru import logger

import main as stage1
from stage2 import persist
from stage2.agent import build_graph
from stage2.approval import resolve
from stage2.persist import Ledger
from stage2.test_agent import ai
from stage2.test_basic import ScriptedModel, call

stage1.TOOL_LATENCY = 0
logger.remove()

RESTART = {"host": "web-01", "service": "nginx"}
P1, P2 = call("restart_service", RESTART, "p1"), call("restart_service", {"host": "db-01", "service": "mysql"}, "p2")
APPROVE = {"action": "approve"}


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def try_resolve(pending, decisions):
    try:
        out = resolve(pending, decisions)
        return (out, None) if isinstance(out, dict) else (None, f"返回的不是 dict: {out!r}")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


async def main() -> None:
    results = []

    # ---------- resolve ----------
    print("-- resolve --")
    out, err = try_resolve([P1, P2], {"p1": APPROVE, "p2": APPROVE})
    results.append(check("全部批准：返回 {}", err is None and out == {}, err or f"实际: {out}"))

    out, err = try_resolve([P1, P2], {"p1": APPROVE, "p2": {"action": "reject", "reason": "现在是业务高峰"}})
    results.append(check("批准一个、拒绝一个：只有被拒绝的出现在结果里",
                         err is None and list(out) == ["p2"], err or f"实际: {out}"))
    results.append(check("拒绝说明里带着人给的原因", err is None and "现在是业务高峰" in str(out.get("p2", "")),
                         err or f"实际: {out}"))
    # 说明是给模型看的：不能只有原因原文，还要告诉模型这个操作没执行、接下来别重试。
    # 具体措辞由你决定，这里只检查"原因之外还有一段固定的指引"：两种拒绝场景的说明里应该有同一段指引
    with_reason = str(out.get("p2", "")) if not err else ""
    results.append(check("拒绝说明不只是原因原文，还有给模型的指引",
                         len(with_reason) > len("现在是业务高峰") + 10, err or f"实际: {with_reason!r}"))

    out, err = try_resolve([P1], {"p1": {"action": "reject"}})
    no_reason = str(out.get("p1", "")) if not err else ""
    results.append(check("拒绝但没写原因：不报错，有默认原因，也有同样的指引",
                         err is None and len(no_reason) >= 20, err or f"实际: {no_reason!r}"))
    results.append(check("没写原因时用的是默认原因，不是空着（不能等于把原因挖掉后的样子）",
                         bool(no_reason) and no_reason != with_reason.replace("现在是业务高峰", ""),
                         f"没原因: {no_reason!r}"))
    results.append(check("说明里没有未写完的占位（如 ……、...、{reason}）",
                         bool(with_reason) and not any(x in with_reason + no_reason for x in ("……", "...", "{reason}")),
                         f"有原因: {with_reason!r}"))
    common = SequenceMatcher(None, with_reason, no_reason).find_longest_match()
    results.append(check("有原因、没原因两种说明里有同一段固定的指引（至少 6 个字相同）",
                         common.size >= 6, f"有原因: {with_reason!r}\n   没原因: {no_reason!r}"))

    # 下面几条是这个函数的重点：答复不完整、格式不对时，倒向哪一边
    out, err = try_resolve([P1, P2], {"p1": APPROVE})
    results.append(check("答复里漏了一个调用：漏掉的按拒绝处理", err is None and list(out) == ["p2"], err or f"实际: {out}"))

    for label, bad in [("yes", "yes"), ("approved", "approved"), ("APPROVE", "APPROVE"), ("空字符串", "")]:
        out, err = try_resolve([P1], {"p1": {"action": bad}})
        results.append(check(f"action 写成 {label}（不是约定的 approve）：按拒绝处理",
                             err is None and "p1" in out, err or f"实际: {out}"))

    for label, bad in [("None", None), ("字符串 'approve'", "approve"), ("True", True), ("列表", [APPROVE])]:
        out, err = try_resolve([P1, P2], bad)
        results.append(check(f"整个答复是 {label}（不是约定的 dict）：不报错，全部按拒绝处理",
                             err is None and sorted(out) == ["p1", "p2"], err or f"实际: {out}"))

    out, err = try_resolve([P1], {"p1": "approve"})
    results.append(check("某个调用的答复不是 dict：不报错，按拒绝处理", err is None and "p1" in out, err or f"实际: {out}"))

    out, err = try_resolve([P1], {"p1": APPROVE, "zzz": {"action": "reject", "reason": "x"}})
    results.append(check("答复里多出来的 id：忽略，结果里只有 pending 里的调用",
                         err is None and out == {}, err or f"实际: {out}"))

    # ---------- 图 ----------
    print("\n-- 图 --")

    def script(extra_reply="已处理"):
        return ScriptedModel([
            ai([call("restart_service", RESTART, "r1"), call("check_disk", {"host": "web-01"}, "d1")]),
            ai(text=extra_reply),
        ])

    # 1. 有副作用的调用：图停下来，什么都没执行
    try:
        persist.RESTARTS.clear()
        model, config = script(), {"configurable": {"thread_id": "t1"}}
        graph = build_graph(model, InMemorySaver(), Ledger(":memory:"))
        out = await graph.ainvoke({"messages": [HumanMessage("重启 nginx 并看磁盘")]}, config)
        asked = out.get("__interrupt__", [])
        calls = asked[0].value["calls"] if asked else []
        results.append(check("遇到 restart_service：图暂停，交出待审批的调用（只有重启，没有只读的 check_disk）",
                             [c["name"] for c in calls] == ["restart_service"] and calls[0]["args"] == RESTART,
                             f"__interrupt__: {asked}"))
        tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        results.append(check("等待审批期间：重启没执行，只读工具也没执行（同一步的调用一起等）",
                             len(persist.RESTARTS) == 0 and not tools,
                             f"重启 {len(persist.RESTARTS)} 次，ToolMessage {len(tools)} 条"))
        snapshot = await graph.aget_state(config)
        results.append(check("快照里能看到在等什么（进程可以退出，之后再查）",
                             bool(snapshot.interrupts) and snapshot.next == ("approve",), f"next={snapshot.next}"))

        # 2. 批准
        out = await graph.ainvoke(Command(resume={"r1": APPROVE}), config)
        tools = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("批准后：重启执行了 1 次，只读工具也执行了，模型给出结论",
                             len(persist.RESTARTS) == 1 and "已重启" in tools["r1"].content and "91" in tools["d1"].content
                             and out["messages"][-1].content == "已处理" and len(model.calls) == 2,
                             f"重启 {len(persist.RESTARTS)} 次，模型调用 {len(model.calls)} 次"))
    except Exception as e:
        results.append(check("暂停与批准", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 3. 拒绝
    try:
        persist.RESTARTS.clear()
        model, config = script("好的，先不重启"), {"configurable": {"thread_id": "t2"}}
        graph = build_graph(model, InMemorySaver(), Ledger(":memory:"))
        await graph.ainvoke({"messages": [HumanMessage("重启 nginx 并看磁盘")]}, config)
        out = await graph.ainvoke(Command(resume={"r1": {"action": "reject", "reason": "现在是业务高峰"}}), config)
        tools = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("拒绝后：重启没执行，模型拿到 error 状态的 ToolMessage，里面有拒绝原因",
                             len(persist.RESTARTS) == 0 and tools["r1"].status == "error"
                             and "现在是业务高峰" in tools["r1"].content,
                             f"重启 {len(persist.RESTARTS)} 次，r1={tools['r1'].content!r}"))
        results.append(check("拒绝后：同一步里的只读工具照常执行，图继续走到结束",
                             "91" in tools["d1"].content and out["messages"][-1].content == "好的，先不重启"))
        # 模型第二次请求时看到的消息里，每个 tool_call 都有配对
        seen = [m.tool_call_id for m in model.calls[-1] if isinstance(m, ToolMessage)]
        results.append(check("拒绝后配对仍然完整", sorted(seen) == ["d1", "r1"], f"实际: {seen}"))
    except Exception as e:
        results.append(check("拒绝", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 4. 用 None 继续（当成崩溃恢复）：不能绕过审批
    try:
        persist.RESTARTS.clear()
        model, config = script(), {"configurable": {"thread_id": "t3"}}
        graph = build_graph(model, InMemorySaver(), Ledger(":memory:"))
        await graph.ainvoke({"messages": [HumanMessage("重启 nginx 并看磁盘")]}, config)
        out = await graph.ainvoke(None, config)
        results.append(check("不带答复直接继续：仍然停在审批，重启没有执行",
                             bool(out.get("__interrupt__")) and len(persist.RESTARTS) == 0,
                             f"重启 {len(persist.RESTARTS)} 次"))
    except Exception as e:
        results.append(check("不带答复直接继续", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 5. 只有只读工具：不打扰人
    try:
        model = ScriptedModel([ai([call("check_disk", {"host": "web-01"}, "d1")]), ai(text="91%")])
        out = await build_graph(model, InMemorySaver()).ainvoke({"messages": [HumanMessage("看磁盘")]},
                                                                {"configurable": {"thread_id": "t4"}})
        results.append(check("只有只读工具时不需要审批，直接跑完",
                             "__interrupt__" not in out and out["messages"][-1].content == "91%"))
    except Exception as e:
        results.append(check("只读工具不需要审批", False, f"{type(e).__name__}: {str(e)[:200]}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
