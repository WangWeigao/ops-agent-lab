"""阶段 1 · 第 5 课：并发执行同一步里的多个工具调用"""
import asyncio
import json
from time import time_ns
from typing import Awaitable, Callable

TOOL_TIMEOUT = 10.0  # 单个工具最长等待时间（秒），超时的不能拖住整批
MAX_CONCURRENCY = 4  # 同时执行的工具数上限，比如跳板机 SSH 连接数有限

Job = Callable[[], Awaitable[str]]


def error_result(message: str) -> str:
    """给模型看的错误结果，格式和 run_tool 的错误保持一致。"""
    return json.dumps({"error": True, "message": message}, ensure_ascii=False)


async def run_concurrently(jobs: list[Job], timeout: float = TOOL_TIMEOUT,
                           limit: int = MAX_CONCURRENCY) -> list[str]:
    """并发执行一批工具调用，按原顺序返回结果。

    每个 job 是一个无参数的 async 函数，调用后返回工具结果字符串。

    要求：
      - 返回列表的顺序和 jobs 一致（tool 消息要和 tool_calls 一一对应）
      - 同时执行的数量不超过 limit
      - 单个 job 超过 timeout 秒：该位置返回 error_result(...)，不影响其他 job
      - 单个 job 抛异常：该位置返回 error_result(...)，不影响其他 job
      - 本函数自身永远不抛异常
    """
    sem = asyncio.Semaphore(limit)

    async def run_one(job: Job) -> str:
        """包一层：保证单个 job 无论怎样都返回字符串，不向外抛异常。"""
        async with sem:  # 先排队拿到名额，再开始计时
            try:
                result = await asyncio.wait_for(job(), timeout)
                return result  # 用 asyncio.wait_for 执行 job()，超时时间是 timeout
            except asyncio.TimeoutError:
                return error_result(f"工具执行超时（超过 {timeout}s）")  # error_result(...)，写清楚超时了多久
            except Exception as e:
                return error_result(f"{type(e).__name__}: {e}")  # error_result(...)，带上异常类型名

    return await asyncio.gather(*[run_one(job) for job in jobs])  # 用 asyncio.gather 一起运行所有 run_one(job)，转成 list
