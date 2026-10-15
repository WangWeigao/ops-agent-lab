"""阶段 2 · 第 4 课：Human-in-the-loop —— 有副作用的操作执行前，暂停等人审批

图：... → guard → approve → tools → ...

实验结论：
  - interrupt() 让节点中止并保存状态；恢复时节点"从头重新执行"，再次走到 interrupt() 时直接返回人的答复
    → interrupt() 之前的代码会执行两次，不能有副作用
  - 恢复用 Command(resume=答复)；传 None 只会重跑一遍，停在同一处
  - 审批不能放在 tools 节点内部：批准后整个节点重跑，已经执行过的只读工具会再执行一次
    → 做成 tools 之前的独立节点，审批结果写进 State 的 rejected，由 tools 的包装函数读取

运行：
    uv run python -m stage2.approval 重启-web01 --ask "web-01 的 nginx 卡死了，重启一下"
    uv run python -m stage2.approval 重启-web01                 # 查看在等什么
    uv run python -m stage2.approval 重启-web01 --approve       # 批准
    uv run python -m stage2.approval 重启-web01 --reject "现在是业务高峰，晚上再重启"
"""
import argparse
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt
from loguru import logger

import main as stage1
from stage2.persist import CHECKPOINT_DB, Ledger, has_side_effects, is_unfinished


def needs_approval(calls: list[dict], blocked: dict) -> list[dict]:
    """这一步里需要人审批的调用：有副作用，并且没有被重复检测拦下（被拦下的本来就不会执行）。"""
    return [c for c in calls if has_side_effects(c["name"]) and c["id"] not in blocked]


def resolve(pending: list[dict], decisions) -> dict[str, str]:
    """把人的答复变成"哪些调用被拒绝"。

    参数：
      pending    需要审批的调用，每个是 {"name": ..., "args": ..., "id": ...}
      decisions  人的答复。约定的格式是
                     {tool_call_id: {"action": "approve"}}
                     {tool_call_id: {"action": "reject", "reason": "现在是业务高峰"}}
                 但它来自图的外面（命令行、网页、聊天机器人），可能不完整、可能格式不对、可能是 None

    返回：
      {tool_call_id: 给模型看的拒绝说明}。被批准的调用不出现在结果里；全部批准时返回 {}。
      只有 pending 里的调用才可能出现在结果里。
    """
    result = {}
    REJECT_TEMPLATE = "操作未执行：审批未通过（{reason}）。不要重试这个操作；把情况告诉用户，继续做不需要审批的排查。"
    DEFAULT_REASON = "用户未明确批准"
    for call in pending:
        tool_call_id = call["id"]
        if not isinstance(decisions, dict):
            result[tool_call_id] = REJECT_TEMPLATE.format(reason=DEFAULT_REASON)
            continue

        decision = decisions.get(tool_call_id, {})

        if not isinstance(decision, dict):
            result[tool_call_id] = REJECT_TEMPLATE.format(reason=DEFAULT_REASON)
            continue

        action = decision.get("action", "")
        if action == "approve":
            continue
        else:
            reason = decision.get("reason")
            if not reason:
                result[tool_call_id] = REJECT_TEMPLATE.format(reason=DEFAULT_REASON)
            else:
                result[tool_call_id] = REJECT_TEMPLATE.format(reason=reason)

    return result


def approve_node(state: dict) -> dict:
    last = state["messages"][-1]
    pending = needs_approval(last.tool_calls, state.get("blocked") or {})
    if not pending:
        return {"rejected": {}}  # 每一步都要写：清掉上一步留下的拒绝记录

    # 注意：这一行之前的代码在恢复时会再执行一遍，所以上面只做计算，不写日志、不发通知。
    # interrupt 的参数是给人看的内容，会出现在调用方拿到的 __interrupt__ 里，也会随 checkpoint 保存。
    decisions = interrupt({"calls": [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in pending]})

    rejected = resolve(pending, decisions)
    for call in pending:
        if call["id"] in rejected:
            logger.warning(f"  ✗ 审批拒绝 {call['name']} {call['args']}")
        else:
            logger.info(f"  ✓ 审批通过 {call['name']} {call['args']}")
    return {"rejected": rejected}


# ---------- 入口 ----------
def show_pending(thread_id: str, interrupts) -> str:
    lines = [f"thread {thread_id} 在等待审批："]
    for item in interrupts:
        for call in item.value["calls"]:
            lines.append(f"  - {call['name']} {call['args']}")
    lines.append("批准：加 --approve；拒绝：加 --reject \"原因\"")
    return "\n".join(lines)


async def run(thread_id: str, ask: str | None, approve: bool, reject: str | None) -> str:
    from langchain_deepseek import ChatDeepSeek
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from stage2.agent import RECURSION_LIMIT, TOOLS, build_graph

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        graph = build_graph(ChatDeepSeek(model=stage1.MODEL).bind_tools(TOOLS), saver, Ledger())
        snapshot = await graph.aget_state(config)

        # 三种状态要分清：在等人、上次崩溃了、已经跑完
        if snapshot.interrupts:
            if not approve and reject is None:
                return show_pending(thread_id, snapshot.interrupts)
            decision = {"action": "approve"} if approve else {"action": "reject", "reason": reject}
            calls = [c for item in snapshot.interrupts for c in item.value["calls"]]
            graph_input = Command(resume={c["id"]: decision for c in calls})
            logger.info(f"thread {thread_id}：{'批准' if approve else '拒绝'} {len(calls)} 个操作，继续执行")
        elif is_unfinished(snapshot):
            logger.warning(f"thread {thread_id} 上次停在 {[t.name for t in snapshot.tasks]}，从断点继续")
            graph_input = None
        elif ask:
            graph_input = {"messages": [HumanMessage(ask)]}
        else:
            return f"thread {thread_id} 没有在执行的任务，用 --ask 提问"

        async for update in graph.astream(graph_input, config, stream_mode="updates"):
            for node in update:
                if not node.startswith("__"):
                    logger.info(f"[{node}] 完成")

        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            return show_pending(thread_id, snapshot.interrupts) + "\n（进程现在退出，状态已保存）"
        return snapshot.values["messages"][-1].content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("thread_id")
    parser.add_argument("--ask")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reject", metavar="原因")
    args = parser.parse_args()
    print(asyncio.run(run(args.thread_id, args.ask, args.approve, args.reject)))
