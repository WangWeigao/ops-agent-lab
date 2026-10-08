"""阶段 2 · 第 2 课（下）：重复调用检测做成节点

对照阶段 1 的 guard.py：
    RepeatGuard.counts（进程内存里）  → State 的 call_counts 字段
    主循环里的 guard.check(...)       → guard 节点，放在 agent 和 tools 之间
    WARN：不执行，回一条提醒           → guard 把这些调用的 id 写进 State 的 blocked，
                                        tools 节点的包装函数读到后不执行、直接返回提醒
    STOP：return 中止任务              → guard 给每个调用补一条 ToolMessage（保持配对），
                                        再加一条说明中止的 AIMessage，条件边走向 END

为什么 WARN 不能由 guard 节点直接回 ToolMessage：
    ToolNode 总是执行最后一条 AIMessage 的全部 tool_calls，不管其中某个是否已经有了结果，
    提前回答会得到两条 tool_call_id 相同的 ToolMessage。
"""
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from guard import STOP_AT, WARN_AT, call_key  # 复用阶段 1 的归一规则和阈值


def count_key(name: str, args: dict) -> str:
    """call_key 返回 tuple，而 State 要能序列化进 checkpoint，dict 的 key 用字符串。"""
    return ":".join(call_key(name, args))


def warn_text(name: str, n: int) -> str:
    return (f"重复查询：这是第 {n} 次以相同目标调用 {name}，未执行。之前的结果仍然有效，"
            f"请查看 [排查笔记] 或已有的工具结果。信息足够就直接给出结论，再次重复将中止任务。")


def guard_node(state: dict) -> dict:
    calls = state["messages"][-1].tool_calls
    counts = dict(state.get("call_counts") or {})  # 复制一份再改，不原地修改 State
    blocked, stop = {}, None

    # 按顺序逐个判断：同一步里重复两次的调用，第一个执行，第二个被拦下
    for call in calls:
        key = count_key(call["name"], call["args"])
        counts[key] = n = counts.get(key, 0) + 1
        if n >= STOP_AT:
            stop = stop or (call["name"], n)
        elif n >= WARN_AT:
            blocked[call["id"]] = n
            logger.warning(f"  ! 重复调用 {call['name']} 第 {n} 次，不执行，返回提醒")

    if stop is None:
        return {"call_counts": counts, "blocked": blocked}

    name, n = stop
    logger.warning(f"  ✗ 重复调用 {name} 第 {n} 次，中止任务")
    # 中止前，给这一步的每个调用都补一条 ToolMessage：
    # 这个 thread 之后可能继续对话，留下没有配对的 tool_calls，下一次请求会 400
    return {
        "call_counts": counts,
        "blocked": {},
        "messages": [
            *(ToolMessage("任务已中止，未执行", tool_call_id=c["id"], name=c["name"], status="error") for c in calls),
            AIMessage(f"检测到重复调用（{name} 第 {n} 次），任务中止，请人工介入"),
        ],
    }
