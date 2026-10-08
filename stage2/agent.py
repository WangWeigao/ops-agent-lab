"""阶段 2 · 第 2 课（上）：compress 节点 + 官方 ToolNode

图：START → compress → agent ─(有 tool_calls)→ tools → compress → ...
                          └─(没有)→ END

对照阶段 1：
    compress()                    → compress 节点（stage2/compress.py）
    run_tool 的 try/except         → ToolNode + 自定义错误处理函数 tool_error_message
    run_concurrently 的超时和限流   → awrap_tool_call 给每次工具调用包一层

运行：uv run python -m stage2.agent
"""
import asyncio
import time

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from loguru import logger

import concurrency
import main as stage1
from stage2.basic import RECURSION_LIMIT, TOOLS, make_agent_node
from stage2.compress import compress_node, last_prompt_tokens


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


def make_tools_node() -> ToolNode:
    """官方 ToolNode，加上超时、并发上限和日志。"""
    # 信号量放在闭包里：每个图一份，不同的图（不同的任务）互不影响
    sem = asyncio.Semaphore(concurrency.MAX_CONCURRENCY)

    async def guarded_call(request: ToolCallRequest, handler):
        """给每一次工具调用包一层。handler(request) 才是真正执行工具。"""
        call = request.tool_call
        logger.info(f"  → {call['name']} {call['args']}")
        t0 = time.perf_counter()
        async with sem:  # 和阶段 1 一样：先排队拿到名额，再开始计时
            try:
                result = await asyncio.wait_for(handler(request), concurrency.TOOL_TIMEOUT)
            except asyncio.TimeoutError:
                # 超时发生在这一层，ToolNode 的错误处理管不到，要自己返回 ToolMessage 来保持配对
                result = ToolMessage(f"工具执行超时（超过 {concurrency.TOOL_TIMEOUT}s）",
                                     tool_call_id=call["id"], name=call["name"], status="error")
        status = getattr(result, "status", "?")
        log = logger.warning if status == "error" else logger.info
        log(f"  ← {call['name']} {status} {time.perf_counter() - t0:.2f}s | "
            f"{' '.join(str(getattr(result, 'content', result)).split())[:80]}")
        return result

    return ToolNode(TOOLS, handle_tool_errors=tool_error_message, awrap_tool_call=guarded_call)


def route(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


def build_graph(model, checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("compress", compress_node)
    builder.add_node("agent", make_agent_node(model))
    builder.add_node("tools", make_tools_node())
    builder.add_edge(START, "compress")
    builder.add_edge("compress", "agent")
    builder.add_conditional_edges("agent", route, ["tools", END])
    builder.add_edge("tools", "compress")  # 工具执行完先经过压缩，再回到模型
    return builder.compile(checkpointer=checkpointer)


async def run(question: str) -> str:
    from langchain_deepseek import ChatDeepSeek

    graph = build_graph(ChatDeepSeek(model=stage1.MODEL).bind_tools(TOOLS))
    final, step = None, 0
    async for update in graph.astream({"messages": [HumanMessage(question)]},
                                      config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, delta in update.items():
            msgs = (delta or {}).get("messages", [])
            if node == "compress":
                if msgs:
                    logger.info(f"[compress] 把 {len(msgs)} 条较早的工具结果替换成占位文字")
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
