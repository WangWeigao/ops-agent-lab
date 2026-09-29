"""长期记忆的离线测试，不调用 API。

运行：uv run python test_long_term.py
"""
import tempfile
from pathlib import Path

from long_term import VOLATILE_TTL_HOURS, Memory, MemoryStore, format_memory
from memory import STABLE, VOLATILE, Note

HOUR = 3600
NOW = 1_800_000_000.0


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def fmt(m: Memory) -> str:
    try:
        return format_memory(m, NOW)
    except Exception as e:
        return f"💥 {type(e).__name__}: {e}"


def main() -> None:
    results = []

    # ---------- format_memory ----------
    print("-- format_memory --")
    fresh = Memory("web-01", "磁盘 91%", "check_disk web-01", VOLATILE, NOW - 1 * HOUR)
    stale = Memory("web-01", "磁盘 91%", "check_disk web-01", VOLATILE, NOW - (VOLATILE_TTL_HOURS + 48) * HOUR)
    stable_old = Memory("db-01", "orders.user_id 缺索引", "get_logs db-01", STABLE, NOW - 30 * 24 * HOUR)

    f_fresh, f_stale, f_stable = fmt(fresh), fmt(stale), fmt(stable_old)
    results.append(check("包含 finding 和 evidence",
                         "磁盘 91%" in f_fresh and "check_disk web-01" in f_fresh, f_fresh))
    results.append(check("新旧不同的记忆，输出不同（能看出多旧）",
                         not f_fresh.startswith("💥") and f_fresh != f_stale, f"{f_fresh}\n   {f_stale}"))
    results.append(check("过期的状态类记忆提示重新验证", "验证" in f_stale, f_stale))
    results.append(check("结构类记忆即使很旧，也不提示重新验证",
                         not f_stable.startswith("💥") and "验证" not in f_stable, f_stable))

    # ---------- MemoryStore（不依赖 format_memory）----------
    print("\n-- MemoryStore --")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.json"
        store = MemoryStore(path)
        n = store.save_notes([
            Note("磁盘 91%", "check_disk web-01", "web-01", VOLATILE),
            Note("缺索引", "get_logs db-01", "db-01", STABLE),
            Note("没有主机的笔记", "?", "", VOLATILE),
        ], now=NOW - 2 * HOUR)
        store.save_notes([Note("磁盘 95%", "check_disk web-01", "web-01", VOLATILE)], now=NOW)
        results.append(check("没有 host 的笔记不写入", n == 2, f"实际写入 {n} 条"))

        reloaded = MemoryStore(path)
        hits = reloaded.recall("web-01")
        results.append(check("重新加载后仍能读到（跨会话）", len(reloaded.memories) == 3))
        results.append(check("按主机过滤、最新在前", [h.finding for h in hits] == ["磁盘 95%", "磁盘 91%"],
                             f"实际: {[h.finding for h in hits]}"))
        results.append(check("没有记录时返回明确提示", "没有历史记录" in reloaded.recall_text("cache-01", NOW)))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
