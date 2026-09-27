"""call_key / RepeatGuard 的离线测试，不调用 API。

运行：uv run python test_guard.py
"""
from guard import RepeatGuard, Verdict, call_key


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


class _Err:
    """call_key 抛异常时的占位值：互不相等、不可哈希，保证相关测试一定失败。"""
    __hash__ = None

    def __init__(self, e: Exception) -> None:
        self.text = f"💥 {type(e).__name__}: {e}"

    def __repr__(self) -> str:
        return self.text


def ok(*keys) -> bool:
    return not any(isinstance(k, _Err) for k in keys)


def safe_key(name: str, args: dict):
    try:
        return call_key(name, args)
    except Exception as e:
        return _Err(e)


def main() -> None:
    results = []

    a = safe_key("get_logs", {"host": "web-01", "lines": 80})
    b = safe_key("get_logs", {"host": "web-01", "lines": 40})
    c = safe_key("get_logs", {"host": "web-01"})
    results.append(check("get_logs 同一主机、lines 不同 → 同一查询", a == b == c, f"{a} / {b} / {c}"))

    d = safe_key("get_logs", {"host": "db-01", "lines": 80})
    results.append(check("get_logs 不同主机 → 不同查询", ok(a, d) and a != d, f"{a} / {d}"))

    e = safe_key("check_disk", {"host": "web-01"})
    results.append(check("不同工具、同一主机 → 不同查询", ok(e, a) and e != a, f"{e} / {a}"))

    f = safe_key("some_tool", {"x": 1, "y": 2})
    g = safe_key("some_tool", {"y": 2, "x": 1})
    h = safe_key("some_tool", {"x": 1, "y": 3})
    results.append(check("其他工具：参数顺序无关", f == g, f"{f} / {g}"))
    results.append(check("其他工具：任一参数不同 → 不同查询", ok(f, h) and f != h, f"{f} / {h}"))

    try:
        hash(a), hash(f)
        results.append(check("返回值可哈希", True))
    except TypeError as ex:
        results.append(check("返回值可哈希", False, str(ex)))

    # RepeatGuard：复现上次的失忆循环，第 2 次提醒、第 3 次中止
    guard = RepeatGuard()
    expected = [Verdict.OK, Verdict.WARN, Verdict.STOP]
    try:
        verdicts = [guard.check("get_logs", f'{{"host": "web-01", "lines": {n}}}')[0]
                    for n in (80, 60, 40)]
        detail = f"实际: {[v.value for v in verdicts]}"
    except Exception as ex:
        verdicts, detail = None, f"{type(ex).__name__}: {ex}"
    results.append(check("RepeatGuard：同一主机换 lines 查 3 次 → OK / WARN / STOP",
                         verdicts == expected, detail))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
