"""阶段 2 · 第 2 课（下）：notes 字段的 reducer（merge_notes）的离线测试。

运行：uv run python -m stage2.test_state
"""
from memory import MAX_NOTES
from stage2.state import merge_notes


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def note(n: int) -> dict:
    return {"finding": f"发现{n}", "evidence": f"依据{n}", "host": "web-01", "kind": "状态"}


def names(notes) -> list:
    return [n["finding"] for n in notes] if isinstance(notes, list) else notes


def run(old: list, new: list):
    try:
        return merge_notes(old, new), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    results = []

    out, err = run([], [note(1)])
    results.append(check("第一条笔记：已有笔记为空", err is None and names(out) == ["发现1"], err or f"实际: {names(out)}"))

    out, err = run([note(1), note(2)], [note(3)])
    results.append(check("追加在后面，顺序不变", err is None and names(out) == ["发现1", "发现2", "发现3"],
                         err or f"实际: {names(out)}"))

    # 模拟同一步里并行的 3 次 record_finding：框架会连续调用 3 次 reducer
    out, err = [note(0)], None
    for i in (1, 2, 3):
        out, err = run(out, [note(i)]) if err is None else (out, err)
    results.append(check("连续调用 3 次（并行的 3 个 record_finding），3 条都在",
                         err is None and names(out) == ["发现0", "发现1", "发现2", "发现3"], err or f"实际: {names(out)}"))

    old, new = [note(1)], [note(2)]
    out, err = run(old, new)
    results.append(check("不修改传入的 old 和 new，返回新列表",
                         err is None and names(old) == ["发现1"] and names(new) == ["发现2"] and out is not old,
                         err or f"old={names(old)} new={names(new)}"))

    full = [note(i) for i in range(MAX_NOTES)]
    out, err = run(full, [note(100), note(101)])
    results.append(check(f"超过 MAX_NOTES（{MAX_NOTES}）时丢弃最早的，保留最新的",
                         err is None and len(out) == MAX_NOTES and names(out)[0] == "发现2" and names(out)[-1] == "发现101",
                         err or f"共 {len(out)} 条，第一条 {names(out)[:1]}，最后一条 {names(out)[-1:]}"))

    out, err = run([note(1)], [])
    results.append(check("没有新笔记时，已有笔记不变", err is None and names(out) == ["发现1"], err or f"实际: {names(out)}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
