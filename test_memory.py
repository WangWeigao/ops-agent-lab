"""Notebook / with_notes 的离线测试，不调用 API。

运行：uv run python test_memory.py
"""
import copy

from memory import MAX_NOTES, NOTES_PREFIX, Notebook, with_notes


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def build_messages() -> list[dict]:
    return [
        {"role": "system", "content": "你是运维助手。"},
        {"role": "user", "content": "web-01 为什么慢？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_logs", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "check_disk", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "日志..."},
        {"role": "tool", "tool_call_id": "b", "content": "91%"},
    ]


def pairing_ok(msgs: list[dict]) -> bool:
    """每条带 tool_calls 的 assistant 后面，必须紧跟它全部 id 的 tool 消息。"""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "tool":
            return False  # 游离的 tool 消息
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = {tc["id"] for tc in m["tool_calls"]}
            following = []
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "tool":
                following.append(msgs[j]["tool_call_id"])
                j += 1
            if set(following) != ids:
                return False
            i = j
            continue
        i += 1
    return True


def main() -> None:
    results = []

    # Notebook
    nb = Notebook()
    results.append(check("空笔记本 render 返回 None", nb.render() is None))
    for i in range(MAX_NOTES + 3):
        nb.add(f"发现{i}", "证据")
    results.append(check(f"超过 {MAX_NOTES} 条时丢弃最早的",
                         len(nb.notes) == MAX_NOTES and nb.notes[0].finding == "发现3"))

    # with_notes：空笔记
    msgs = build_messages()
    results.append(check("笔记为空时原样返回", with_notes(msgs, Notebook()) == msgs))

    # with_notes：有笔记
    nb = Notebook()
    nb.add("web-01 磁盘 91%", "check_disk web-01")
    msgs = build_messages()
    frozen = copy.deepcopy(msgs)
    out = with_notes(msgs, nb)

    note_msgs = [m for m in out if m["role"] == "user" and str(m["content"]).startswith(NOTES_PREFIX)]
    results.append(check("笔记消息恰好出现一次", len(note_msgs) == 1, f"实际 {len(note_msgs)} 次"))
    results.append(check("笔记内容来自 render()", note_msgs == [{"role": "user", "content": nb.render()}]))
    results.append(check("不修改传入的 messages", msgs == frozen))
    results.append(check("system 仍在第一条", out[:1] == frozen[:1]))
    results.append(check("其余消息顺序不变", [m for m in out if m not in note_msgs] == frozen))
    results.append(check("tool_call_id 配对未被破坏", pairing_ok(out)))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
