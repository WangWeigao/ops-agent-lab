"""阶段 2 · 第 2 课：完整的图（compress + ToolNode + 笔记 + 重复检测）的离线测试，不调用 API。

运行：uv run python -m stage2.test_agent
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import concurrency
import main as stage1
from context import CONTEXT_THRESHOLD, PLACEHOLDER
from memory import NOTES_PREFIX
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
            ai([call("check_disk", {"host": "db-02"}, "e")], tokens=over),
            ai(text="根因是缺索引", tokens=over),
        ])

    try:
        nodes = []
        async for update in build_graph(script()).astream({"messages": [HumanMessage("排查")]},
                                                          stream_mode="updates"):
            nodes += list(update)
        results.append(check("节点顺序：begin_turn → compress → agent → guard → tools → compress → ...",
                             nodes[:7] == ["begin_turn", "compress", "agent", "guard", "tools", "compress", "agent"],
                             f"实际: {nodes[:8]}"))

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

    # ---------- 笔记 ----------
    print("\n-- 笔记 --")

    def note(n: int, id_: str) -> dict:
        return call("record_finding", {"finding": f"发现{n}", "evidence": f"依据{n}", "host": "web-01", "kind": "状态"}, id_)

    # 4. 同一步里并行记两条笔记 + 一个普通工具
    model = ScriptedModel([
        ai([note(1, "a"), call("check_disk", {"host": "web-01"}, "b"), note(2, "c")]),
        ai([note(3, "d")]),
        ai(text="结论"),
    ])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("排查")]})
        results.append(check("并行的 record_finding 都写进了 State 的 notes，顺序不变",
                             [n["finding"] for n in out.get("notes", [])] == ["发现1", "发现2", "发现3"],
                             f"实际: {out.get('notes')}"))
        tools = [m.tool_call_id for m in out["messages"] if isinstance(m, ToolMessage)]
        results.append(check("record_finding 也有配对的 ToolMessage", tools == ["a", "b", "c", "d"], f"实际: {tools}"))
        results.append(check("第一次请求没有笔记消息",
                             not any(str(m.content).startswith(NOTES_PREFIX) for m in model.calls[0])))  # system prompt 里也提到了笔记，所以用 startswith
        last = model.calls[-1][-1]
        results.append(check("之后的请求末尾带着全部笔记",
                             isinstance(last, HumanMessage) and all(f"发现{i}" in last.content for i in (1, 2, 3)),
                             f"最后一条: {type(last).__name__} {str(last.content)[:80]!r}"))
        results.append(check("笔记消息没有写进 State 的 messages（只存在于每次请求里）",
                             not any(str(m.content).startswith(NOTES_PREFIX) for m in out["messages"])))
    except Exception as e:
        results.append(check("笔记写进 State 并出现在请求里", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 5. 笔记随 checkpoint 保存：同一个 thread 的第二轮对话还能看到
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        model = ScriptedModel([ai([note(1, "a")]), ai(text="第一轮结论"), ai(text="第二轮结论")])
        graph, config = build_graph(model, InMemorySaver()), {"configurable": {"thread_id": "t1"}}
        await graph.ainvoke({"messages": [HumanMessage("排查")]}, config)
        await graph.ainvoke({"messages": [HumanMessage("再看看")]}, config)
        results.append(check("同一个 thread 的下一轮对话仍然带着上一轮的笔记",
                             "发现1" in str(model.calls[-1][-1].content), f"最后一条: {model.calls[-1][-1]!r}"[:200]))
    except Exception as e:
        results.append(check("笔记随 checkpoint 保存", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # ---------- 重复检测 ----------
    print("\n-- 重复检测 --")
    disk = {"host": "web-01"}

    # 6. 第 2 次：不执行，返回提醒；其他调用正常执行
    model = ScriptedModel([
        ai([call("check_disk", disk, "a")]),
        ai([call("check_disk", disk, "b"), call("check_disk", {"host": "db-01"}, "c")]),
        ai(text="结论"),
    ])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("查")]})
        tools = {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("第 2 次相同查询不执行，返回提醒；同一步里的其他调用正常执行",
                             "重复查询" in tools["b"] and "usage_percent" in tools["a"] and "usage_percent" in tools["c"],
                             f"实际: {tools}"))
        results.append(check("计数保存在 State 的 call_counts 里",
                             sorted(out["call_counts"].values()) == [1, 2], f"实际: {out.get('call_counts')}"))
    except Exception as e:
        results.append(check("第 2 次相同查询返回提醒", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 7. get_logs 只换 lines 也算重复；同一步里重复两次：第一个执行，第二个被拦下
    model = ScriptedModel([
        ai([call("get_logs", {"host": "web-01", "lines": 2}, "a"), call("get_logs", {"host": "web-01", "lines": 5}, "b")]),
        ai(text="结论"),
    ])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("查")]})
        tools = {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("同一步里的重复调用：第一个执行，第二个被拦下",
                             "重复查询" not in tools["a"] and "重复查询" in tools["b"], f"实际: {tools}"))
    except Exception as e:
        results.append(check("同一步里的重复调用", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 8. 第 3 次：中止。图正常结束（不是抛异常），配对完整
    model = ScriptedModel([ai([call("check_disk", disk, "x")])], repeat_last=True)
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("查")]}, config={"recursion_limit": 50})
        msgs = out["messages"]
        results.append(check("第 3 次相同查询：任务中止，模型只被调用了 3 次",
                             len(model.calls) == 3 and "中止" in msgs[-1].content and not msgs[-1].tool_calls,
                             f"模型调用 {len(model.calls)} 次，最后一条: {msgs[-1].content!r}"))
        asked = [c["id"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls]
        answered = [m.tool_call_id for m in msgs if isinstance(m, ToolMessage)]
        results.append(check("中止后每个 tool_call 仍有配对的 ToolMessage", len(asked) == len(answered) == 3,
                             f"tool_calls {len(asked)} 个，ToolMessage {len(answered)} 条"))
    except Exception as e:
        results.append(check("第 3 次相同查询中止任务", False, f"{type(e).__name__}: {str(e)[:200]}"))

    # 9. 计数只在一轮提问内有效：同一个 thread 的下一轮提问，可以重新查同一台主机
    model = ScriptedModel([ai([call("check_disk", disk, "a")]), ai(text="91%"),
                           ai([call("check_disk", disk, "b")]), ai(text="清理后正常")])
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        graph, config = build_graph(model, InMemorySaver()), {"configurable": {"thread_id": "t2"}}
        await graph.ainvoke({"messages": [HumanMessage("web-01 磁盘怎么样")]}, config)
        out = await graph.ainvoke({"messages": [HumanMessage("清理完了，再看看")]}, config)
        tools = {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}
        results.append(check("同一个 thread 的下一轮提问：计数清零，相同查询正常执行",
                             "usage_percent" in tools["b"], f"实际: {tools['b'][:60]}"))
    except Exception as e:
        results.append(check("下一轮提问计数清零", False, f"{type(e).__name__}: {str(e)[:200]}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
