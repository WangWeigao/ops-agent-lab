"""manage_context 的离线测试，不调用 API。

运行：uv run python test_context.py
"""
import copy

from context import CONTEXT_THRESHOLD, PLACEHOLDER, manage_context


def step(n: int, call_ids: list[str]) -> list[dict]:
    """构造一步：一条带 tool_calls 的 assistant 消息 + 对应的 tool 消息。"""
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": "get_logs", "arguments": '{"host": "web-01"}'}}
            for cid in call_ids
        ],
    }
    tools = [{"role": "tool", "tool_call_id": cid, "content": f"step{n}-{cid} 的长日志..."}
             for cid in call_ids]
    return [assistant, *tools]


def build_messages() -> list[dict]:
    # 4 步，第 1 步是一次调用两个工具的并行调用
    return [
        {"role": "system", "content": "你是运维助手。"},
        {"role": "user", "content": "web-01 为什么慢？"},
        *step(1, ["a", "b"]),
        *step(2, ["c"]),
        *step(3, ["d"]),
        *step(4, ["e", "f"]),
    ]


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def tool_contents(msgs: list[dict]) -> dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}


def main() -> None:
    results = []

    # 1. 未超阈值：不做任何修改
    original = build_messages()
    out = manage_context(copy.deepcopy(original), CONTEXT_THRESHOLD - 1)
    results.append(check("未超阈值时原样返回", out == original))

    # 2. 超过阈值
    out = manage_context(copy.deepcopy(original), CONTEXT_THRESHOLD + 1)
    contents = tool_contents(out)

    results.append(check("消息条数不变", len(out) == len(original),
                         f"期望 {len(original)} 条，实际 {len(out)} 条"))
    results.append(check("system 和 user 消息保留", out[:2] == original[:2]))

    old = {k: contents.get(k) for k in "abc"}
    results.append(check("较早的 step1、step2（a b c）的结果被替换",
                         all(v == PLACEHOLDER for v in old.values()), f"实际: {old}"))

    recent = {k: contents.get(k) for k in "def"}
    expected = {"d": "step3-d 的长日志...", "e": "step4-e 的长日志...", "f": "step4-f 的长日志..."}
    results.append(check("最近 2 步 step3、step4（d e f）的结果完整保留",
                         recent == expected, f"实际: {recent}"))

    ids_before = [m.get("tool_call_id") for m in original]
    ids_after = [m.get("tool_call_id") for m in out]
    results.append(check("tool_call_id 顺序和配对未被破坏", ids_before == ids_after))

    assistants_before = [m for m in original if m["role"] == "assistant"]
    assistants_after = [m for m in out if m["role"] == "assistant"]
    results.append(check("assistant 消息未被修改", assistants_before == assistants_after))

    # 3. 步数不足 KEEP_RECENT_STEPS 时，什么都不该被替换
    short = build_messages()[:6]  # system + user + step1 + step2
    out = manage_context(copy.deepcopy(short), CONTEXT_THRESHOLD + 1)
    results.append(check("只有 2 步时不替换任何结果", out == short))

    # 4. 步数少于 KEEP_RECENT_STEPS：比如第一步就拉回一份超大日志，下一轮就超阈值了
    for n_steps, msgs in [(0, build_messages()[:2]), (1, build_messages()[:5])]:
        try:
            out = manage_context(copy.deepcopy(msgs), CONTEXT_THRESHOLD + 1)
            results.append(check(f"只有 {n_steps} 步时不报错、不替换", out == msgs))
        except Exception as e:
            results.append(check(f"只有 {n_steps} 步时不报错、不替换", False, f"{type(e).__name__}: {e}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
