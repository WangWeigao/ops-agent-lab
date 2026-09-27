"""阶段 1 · 第 3 课：重复调用检测"""
import json
from collections import Counter
from enum import Enum

WARN_AT = 2  # 同一查询第 2 次：不执行，提醒模型
STOP_AT = 3  # 同一查询第 3 次：中止任务


class Verdict(Enum):
    OK = "ok"
    WARN = "warn"
    STOP = "stop"


def call_key(name: str, args: dict) -> tuple:
    """把一次工具调用归一成"查询身份"：key 相同，就视为同一个查询。

    要求：
      - 返回值必须可哈希（用作 Counter 的 key）
      - get_logs 查同一台主机，lines 不同也算同一个查询
      - 不同主机、不同工具，都是不同的查询
      - 没有特殊规则的工具：所有参数都参与判断，且与参数顺序无关
    """
    # TODO(human)
    if name == 'get_logs':
        args = {k: v for k, v in args.items() if k != "lines"}
    return name, json.dumps(args, sort_keys=True)


class RepeatGuard:
    """记录一次 run_agent 内每个查询被调用的次数。"""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def check(self, name: str, args_json: str) -> tuple[Verdict, int]:
        try:
            args = json.loads(args_json)
            key = call_key(name, args) if isinstance(args, dict) else (name, args_json)
        except json.JSONDecodeError:
            # 参数本身不合法，交给 run_tool 去报错，这里按原始字符串计数
            key = (name, args_json)

        self.counts[key] += 1
        n = self.counts[key]
        if n >= STOP_AT:
            return Verdict.STOP, n
        if n >= WARN_AT:
            return Verdict.WARN, n
        return Verdict.OK, n
