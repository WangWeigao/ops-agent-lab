"""run_concurrently 的离线测试，不调用 API。

运行：uv run python test_concurrency.py
"""
import asyncio
import json
import time

from concurrency import run_concurrently


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


def sleeper(seconds: float, value: str):
    async def job() -> str:
        await asyncio.sleep(seconds)
        return value
    return job


def failing():
    async def job() -> str:
        await asyncio.sleep(0.05)
        raise RuntimeError("ssh: connection refused")
    return job


def is_error(result) -> bool:
    try:
        return json.loads(result).get("error") is True
    except Exception:
        return False


async def timed(jobs, **kwargs):
    t0 = time.perf_counter()
    try:
        out = await run_concurrently(jobs, **kwargs)
    except Exception as e:
        out = f"💥 {type(e).__name__}: {e}"
    return out, time.perf_counter() - t0


async def main() -> None:
    results = []

    out, _ = await timed([sleeper(0.3, "a"), sleeper(0.1, "b"), sleeper(0.2, "c")])
    results.append(check("结果顺序和 jobs 一致（与完成顺序无关）", out == ["a", "b", "c"], f"实际: {out}"))

    out, t = await timed([sleeper(0.2, str(i)) for i in range(4)], limit=4)
    results.append(check("4 个 0.2s 的 job 并发执行，总耗时约 0.2s",
                         out == ["0", "1", "2", "3"] and t < 0.35, f"耗时 {t:.2f}s，结果 {out}"))

    out, t = await timed([sleeper(0.2, str(i)) for i in range(4)], limit=2)
    results.append(check("limit=2 时分两批执行，总耗时约 0.4s",
                         out == ["0", "1", "2", "3"] and 0.35 <= t < 0.6, f"耗时 {t:.2f}s"))

    out, t = await timed([sleeper(1.0, "slow"), sleeper(0.1, "fast")], timeout=0.3)
    ok = isinstance(out, list) and len(out) == 2
    results.append(check("超时的 job 返回错误，不拖住整批",
                         ok and is_error(out[0]) and out[1] == "fast" and t < 0.6, f"耗时 {t:.2f}s，结果 {out}"))

    out, _ = await timed([failing(), sleeper(0.1, "ok")])
    ok = isinstance(out, list) and len(out) == 2
    results.append(check("抛异常的 job 返回错误，其他 job 正常",
                         ok and is_error(out[0]) and out[1] == "ok", f"结果 {out}"))

    out, _ = await timed([])
    results.append(check("空列表返回空列表", out == [], f"结果 {out}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
