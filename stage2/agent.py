"""阶段 2 · 第 2 课：compress 节点 + 官方 ToolNode + 笔记 + 重复检测

图：START → begin_turn → compress → agent ─(有 tool_calls)→ guard ─(放行)→ tools → compress → ...
                          └─(没有)→ END        └─(重复第 3 次，中止)→ END

对照阶段 1：
    compress()                    → compress 节点（stage2/compress.py）
    run_tool 的 try/except         → ToolNode + 自定义错误处理函数 tool_error_message
    run_concurrently 的超时和限流   → awrap_tool_call 给每次工具调用包一层
    Notebook                      → State 的 notes 字段 + record_finding 工具（stage2/state.py）
    RepeatGuard                   → State 的 call_counts 字段 + guard 节点（stage2/repeat.py）

第 3 课：持久化的 checkpoint、断点续跑、写操作的幂等，见 stage2/persist.py

运行：uv run python -m stage2.agent
"""
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from loguru import logger

import concurrency
import main as stage1
from stage2 import basic
from stage2.compress import compress_node
from stage2.persist import Ledger, has_side_effects, restart_service, run_once
from stage2.repeat import guard_node, warn_text
from stage2.state import AgentState, record_finding, render_notes

TOOLS = [*basic.TOOLS, record_finding, restart_service]
SYSTEM_PROMPT = ("你是运维助手。需要数据时调用工具，不要编造。"
                 "每得到一个关键发现，就用 record_finding 记录（写明依据），尽量和其他工具在同一轮并行调用。"
                 "较早的工具结果可能被省略，这时以 [排查笔记] 为准，不要重复查询。")
# 现在一轮要经过 4 个节点（compress → agent → guard → tools），recursion_limit 数的是节点执行次数。
# 第 1 课的 25 在这里只够 6 轮，要跟着图的结构一起调整。
MAX_ROUNDS = 12
RECURSION_LIMIT = MAX_ROUNDS * 4 + 3  # 加上入口的 begin_turn 和最后一次 compress → agent


def make_agent_node(model):
    async def agent(state: AgentState) -> dict:
        request = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        # 笔记和 system prompt 一样：只出现在这一次请求里，不写进 State 的 messages。
        # 放在末尾：前面的内容不变，前缀缓存不受影响；也离模型的"注意力"最近。
        if notes := render_notes(state.get("notes") or []):
            request.append(HumanMessage(notes))
        return {"messages": [await model.ainvoke(request)]}

    return agent


def tool_error_message(e: Exception) -> str:
    """工具抛异常时，生成给模型看的错误信息（ToolNode 的 handle_tool_errors）。

    参数的类型注解决定捕获范围：注解成 Exception，所有异常都会交给这个函数，图不会崩溃。

    要求（阶段 1 第 1 课的"两层错误信息"）：
      - 完整的异常信息写进日志，给人排查
      - 返回给模型的：异常类型名 + 截断后的信息，不要把超长或敏感的原文整个交出去
      - 异常信息为空时（如 TimeoutError()）也要返回有意义的内容
      - 本函数自身不能抛异常
    """
    logger.opt(exception=e).error("工具执行异常")

    return f"{type(e).__name__}: {str(e)[:200]}"


def make_tools_node(ledger: Ledger | None = None) -> ToolNode:
    """官方 ToolNode，加上超时、并发上限和日志。传入 ledger 时，写操作保证最多执行一次（第 3 课）。"""
    # 信号量放在闭包里：每个图一份，不同的图（不同的任务）互不影响
    sem = asyncio.Semaphore(concurrency.MAX_CONCURRENCY)

    async def guarded_call(request: ToolCallRequest, handler):
        """给每一次工具调用包一层。handler(request) 才是真正执行工具。"""
        call = request.tool_call
        # guard 节点判定为重复的调用：不执行，直接返回提醒。判定结果是从 State 里读的
        if n := (request.state.get("blocked") or {}).get(call["id"]):
            return ToolMessage(warn_text(call["name"], n), tool_call_id=call["id"], name=call["name"])
        logger.info(f"  → {call['name']} {call['args']}")
        t0 = time.perf_counter()

        async def execute():
            async with sem:  # 和阶段 1 一样：先排队拿到名额，再开始计时
                try:
                    return await asyncio.wait_for(handler(request), concurrency.TOOL_TIMEOUT)
                except asyncio.TimeoutError:
                    # 超时发生在这一层，ToolNode 的错误处理管不到，要自己返回 ToolMessage 来保持配对
                    return ToolMessage(f"工具执行超时（超过 {concurrency.TOOL_TIMEOUT}s）",
                                       tool_call_id=call["id"], name=call["name"], status="error")

        if ledger is not None and has_side_effects(call["name"]):
            # 第 3 课：有副作用的操作先查账。tool_call_id 是模型生成的，同一次调用在恢复前后不变
            thread_id = request.runtime.config.get("configurable", {}).get("thread_id", "")
            result = await run_once(ledger, f"{thread_id}:{call['id']}", call, execute)
        else:
            result = await execute()
        # 工具返回的可能是 ToolMessage，也可能是 Command（record_finding），后者的消息在 update 里
        msg = result if isinstance(result, ToolMessage) else result.update["messages"][-1]
        log = logger.warning if msg.status == "error" else logger.info
        log(f"  ← {call['name']} {msg.status} {time.perf_counter() - t0:.2f}s | {' '.join(str(msg.content).split())[:80]}")
        return result

    return ToolNode(TOOLS, handle_tool_errors=tool_error_message, awrap_tool_call=guarded_call)


def has_tool_calls(state: AgentState) -> bool:
    """最后一条消息是带 tool_calls 的 AIMessage。两条条件边共用这一个判断：
    agent 之后：模型要调工具 → guard；guard 之后：没有中止（最后一条还是那条 AIMessage）→ tools。"""
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def begin_turn(state: AgentState) -> dict:
    """每轮用户提问开始时，清空重复检测的计数。

    call_counts 放进 State 后会跟着 thread 一直保存。不清空的话，用户第二天在同一个 thread 里说
    "索引加完了，再看看 web-01 的日志"，这次合理的查询会被当成重复拦下。
    放在入口：START 只在新的输入到来时经过一次；崩溃后恢复不会重新经过，一轮之内的计数不受影响。
    """
    return {"call_counts": {}, "blocked": {}}


def build_graph(model, checkpointer=None, ledger: Ledger | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("begin_turn", begin_turn)
    builder.add_node("compress", compress_node)
    builder.add_node("agent", make_agent_node(model))
    builder.add_node("guard", guard_node)
    builder.add_node("tools", make_tools_node(ledger))
    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "compress")
    builder.add_edge("compress", "agent")
    builder.add_conditional_edges("agent", has_tool_calls, {True: "guard", False: END})
    builder.add_conditional_edges("guard", has_tool_calls, {True: "tools", False: END})
    builder.add_edge("tools", "compress")  # 工具执行完先经过压缩，再回到模型
    return builder.compile(checkpointer=checkpointer)


async def run(question: str) -> str:
    from langchain_deepseek import ChatDeepSeek

    graph = build_graph(ChatDeepSeek(model=stage1.MODEL).bind_tools(TOOLS))
    final, step = None, 0
    async for update in graph.astream({"messages": [HumanMessage(question)]},
                                      config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, delta in update.items():
            msgs = delta.get("messages", []) if isinstance(delta, dict) else []
            if node == "compress":
                if msgs:
                    logger.info(f"[compress] 把 {len(msgs)} 条较早的工具结果替换成占位文字")
            elif node == "tools":
                # 有工具返回 Command 时，这个节点的更新是一个列表（每个工具一份），否则是一个 dict
                parts = delta if isinstance(delta, list) else [delta]
                if n := sum(len(p.get("notes", [])) for p in parts if isinstance(p, dict)):
                    logger.info(f"[tools] 新增 {n} 条笔记")
            elif node == "guard":
                final = msgs[-1] if msgs else final  # 中止时，最后一条是 guard 写的说明
            elif node == "agent":
                step += 1
                final = msgs[-1]
                usage = final.usage_metadata or {}
                cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
                logger.info(f"[agent] 第 {step} 次 | prompt {usage.get('input_tokens')}（缓存命中 {cached}）"
                            f" | {len(final.tool_calls)} 个工具调用")
    return final.content


if __name__ == "__main__":
    print(asyncio.run(run("web-01 上的订单接口最近很慢，帮我排查根因。先看 web-01 的日志和磁盘，再根据线索看相关主机的日志。")))
