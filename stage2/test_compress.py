"""阶段 2 · 第 2 课（上）：compress_node 的离线测试，不调用 API。

运行：uv run python -m stage2.test_compress
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from context import CONTEXT_THRESHOLD, PLACEHOLDER
from stage2.compress import compress_node


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def usage(tokens: int) -> dict:
    return {"input_tokens": tokens, "output_tokens": 10, "total_tokens": tokens + 10}


def step(n: int, call_ids: list[str], tokens: int) -> list:
    ai = AIMessage("", id=f"ai{n}", usage_metadata=usage(tokens),
                   tool_calls=[{"name": "get_logs", "args": {"host": "web-01"}, "id": c, "type": "tool_call"}
                               for c in call_ids])
    tools = [ToolMessage(f"step{n}-{c} 的长日志...", tool_call_id=c, id=f"tool-{c}") for c in call_ids]
    return [ai, *tools]


def build(tokens: int, steps: int = 4) -> list:
    """4 步：step1 并行调用两个工具（a b），step2（c），step3（d），step4（e f）。"""
    plan = [(1, ["a", "b"]), (2, ["c"]), (3, ["d"]), (4, ["e", "f"])][:steps]
    msgs = [HumanMessage("web-01 为什么慢？", id="human")]
    for n, ids in plan:
        msgs += step(n, ids, tokens)
    return msgs


def run(msgs: list):
    """调用节点，再用 add_messages 把返回值合并进 State，模拟图的行为。"""
    try:
        update = compress_node({"messages": msgs})
        merged = add_messages(list(msgs), update.get("messages", []))
        return update, merged, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def contents(msgs: list) -> dict:
    return {m.tool_call_id: m.content for m in msgs if isinstance(m, ToolMessage)}


def main() -> None:
    results = []

    # 1. 未超阈值
    update, merged, err = run(build(CONTEXT_THRESHOLD))
    results.append(check("未超阈值时不做修改，返回 {}", err is None and not update.get("messages"), err or f"返回: {update}"))

    # 2. 超过阈值
    original = build(CONTEXT_THRESHOLD + 1)
    update, merged, err = run(original)
    if err:
        for name in ("消息条数和顺序不变", "较早的 a b c 被替换", "最近 2 步 d e f 保留", "只返回要改的 3 条", "配对未破坏"):
            results.append(check(name, False, err))
    else:
        results.append(check("合并后消息条数和顺序不变（原地替换，不是追加）",
                             [m.id for m in merged] == [m.id for m in original],
                             f"原 {len(original)} 条，合并后 {len(merged)} 条: {[m.id for m in merged]}"))
        c = contents(merged)
        old = {k: c.get(k) for k in "abc"}
        results.append(check("较早的 step1、step2（a b c）被替换成占位文字",
                             all(v == PLACEHOLDER for v in old.values()), f"实际: {old}"))
        recent = {k: c.get(k) for k in "def"}
        results.append(check("最近 2 步 step3、step4（d e f）完整保留",
                             all(v and v != PLACEHOLDER for v in recent.values()), f"实际: {recent}"))
        returned = update.get("messages", [])
        results.append(check("只返回要改的 3 条，不返回完整列表",
                             sorted(m.tool_call_id for m in returned) == ["a", "b", "c"],
                             f"返回了 {len(returned)} 条: {[getattr(m, 'tool_call_id', type(m).__name__) for m in returned]}"))
        results.append(check("tool_call_id 配对未破坏",
                             [m.tool_call_id for m in merged if isinstance(m, ToolMessage)] == list("abcdef")))

        # 3. 再压缩一次：已经是占位文字的不用再返回
        update2, _, err2 = run(merged)
        results.append(check("已经是占位文字的不再重复返回",
                             err2 is None and not update2.get("messages"), err2 or f"返回了 {len(update2.get('messages', []))} 条"))

    # 4. 步数不够
    for steps in (1, 2):
        update, _, err = run(build(CONTEXT_THRESHOLD + 1, steps=steps))
        results.append(check(f"只有 {steps} 步时不报错、不修改", err is None and not update.get("messages"), err or ""))

    # 5. 还没调用过模型
    update, _, err = run([HumanMessage("你好", id="human")])
    results.append(check("还没调用过模型时不报错、不修改", err is None and not update.get("messages"), err or ""))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
