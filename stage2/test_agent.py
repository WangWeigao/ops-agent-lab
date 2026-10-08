"""阶段 2 · 第 2 课（上）：compress + ToolNode 图的离线测试，不调用 API。

运行：uv run python -m stage2.test_agent
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import concurrency
import main as stage1
from context import CONTEXT_THRESHOLD, PLACEHOLDER
from stage2.agent import build_graph, tool_error_message
from stage2.test_basic import ScriptedModel, call

stage1.TOOL_LATENCY = 0


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def ai(tool_calls: list | None = None, text: str = "", tokens: int = 100) -> AIMessage:
    return AIMessage(text, tool_calls=tool_calls or [],
                     usage_metadata={"input_tokens": tokens, "output_tokens": 10, "total_tokens": tokens + 10})


async def main() -> None:
    results = []

    # ---------- tool_error_message ----------
    print("-- tool_error_message --")
    try:
        msg = tool_error_message(ValueError("unknown host: cache-01"))
        results.append(check("包含异常类型名和信息", "ValueError" in msg and "cache-01" in msg, repr(msg)))
        long_msg = tool_error_message(RuntimeError("x" * 5000))
        results.append(check("超长的异常信息被截断", len(long_msg) < 400, f"长度 {len(long_msg)}"))
        empty = tool_error_message(TimeoutError())
        results.append(check("异常信息为空时仍有类型名", "TimeoutError" in empty, repr(empty)))
    except Exception as e:
        results.append(check("tool_error_message 自身不抛异常", False, f"{type(e).__name__}: {e}"))

    # ---------- 图 ----------
    print("\n-- 图 --")
    # 1. 工具抛异常：图不崩溃，错误作为 ToolMessage 返回给模型，其他工具不受影响
    model = ScriptedModel([
        ai([call("check_disk", {"host": "cache-01"}, "a"), call("check_disk", {"host": "web-01"}, "b")]),
        ai(text="cache-01 不存在"),
    ])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("查")]})
        tools = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("工具抛异常时图不崩溃，返回 error 状态的 ToolMessage",
                             tools["a"].status == "error" and tools["b"].status == "success"
                             and out["messages"][-1].content == "cache-01 不存在",
                             f"a={tools['a'].status} b={tools['b'].status}"))
    except Exception as e:
        results.append(check("工具抛异常时图不崩溃", False, f"{type(e).__name__}: {e}"))

    # 2. 超时：慢的工具返回错误，不拖住整个图
    stage1.TOOL_LATENCY, old_timeout = 1.0, concurrency.TOOL_TIMEOUT
    concurrency.TOOL_TIMEOUT = 0.2
    model = ScriptedModel([ai([call("check_disk", {"host": "web-01"}, "a")]), ai(text="超时了")])
    try:
        t0 = asyncio.get_running_loop().time()
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("查")]})
        elapsed = asyncio.get_running_loop().time() - t0
        tool = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        results.append(check("工具超时返回 error，tool_call_id 配对仍在",
                             tool.status == "error" and tool.tool_call_id == "a" and elapsed < 0.8,
                             f"status={tool.status} 耗时 {elapsed:.2f}s"))
    except Exception as e:
        results.append(check("工具超时返回 error", False, f"{type(e).__name__}: {e}"))
    finally:
        stage1.TOOL_LATENCY, concurrency.TOOL_TIMEOUT = 0, old_timeout

    # 3. 压缩在图里生效：4 轮工具调用，prompt 一直超阈值
    over = CONTEXT_THRESHOLD + 1

    def script() -> ScriptedModel:
        return ScriptedModel([
            ai([call("get_logs", {"host": "web-01", "lines": 2}, "a"),
                call("check_disk", {"host": "web-01"}, "b")], tokens=over),
            ai([call("get_logs", {"host": "db-01", "lines": 2}, "c")], tokens=over),
            ai([call("check_disk", {"host": "db-01"}, "d")], tokens=over),
            ai([call("get_logs", {"host": "web-01", "lines": 1}, "e")], tokens=over),
            ai(text="根因是缺索引", tokens=over),
        ])

    try:
        nodes = []
        async for update in build_graph(script()).astream({"messages": [HumanMessage("排查")]},
                                                          stream_mode="updates"):
            nodes += list(update)
        results.append(check("节点顺序：compress → agent → tools → compress → ...",
                             nodes[:5] == ["compress", "agent", "tools", "compress", "agent"], f"实际: {nodes[:6]}"))

        model = script()
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("排查")]})
        msgs = out["messages"]
        contents = {m.tool_call_id: m.content for m in msgs if isinstance(m, ToolMessage)}
        results.append(check("消息总数正确（1 + 4 轮 AI + 5 条 Tool + 最终 AI = 11），没有多出追加的副本",
                             len(msgs) == 11, f"实际 {len(msgs)} 条"))
        results.append(check("较早的工具结果（a b c）被替换，最近 2 步（d e）保留",
                             all(contents[k] == PLACEHOLDER for k in "abc")
                             and all(contents[k] != PLACEHOLDER for k in "de"), f"实际: {contents}"))
        # 最后一次调用模型时，它看到的 a b c 已经是占位文字
        seen = {m.tool_call_id: m.content for m in model.calls[-1] if isinstance(m, ToolMessage)}
        results.append(check("模型最后一次请求看到的是压缩后的内容",
                             all(seen.get(k) == PLACEHOLDER for k in "abc"), f"模型看到: {seen}"))
    except Exception as e:
        results.append(check("压缩在图里生效", False, f"{type(e).__name__}: {e}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
