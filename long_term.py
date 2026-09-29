"""阶段 1 · 第 4 课（下）：长期记忆（跨会话保存笔记，按主机检索）"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from datetime import timedelta

import humanize

from memory import VOLATILE, Note

humanize.i18n.activate("zh_CN")

STORE_PATH = Path("memory_store.json")
VOLATILE_TTL_HOURS = 24  # 状态类记忆超过这个时间，视为可能已过期
RECALL_LIMIT = 10  # 每次最多返回几条，避免记忆多了撑爆上下文

RECALL_MEMORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": "查询某台主机在以往排查中记录的发现。历史记录只是线索，"
                       "状态类信息可能已经变化，下结论前要用工具重新验证。",
        "parameters": {
            "type": "object",
            "properties": {"host": {"type": "string", "description": "主机名，如 web-01"}},
            "required": ["host"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Memory:
    host: str
    finding: str
    evidence: str
    kind: str  # VOLATILE 或 STABLE
    observed_at: float  # Unix 时间戳（秒）


def format_memory(m: Memory, now: float) -> str:
    """把一条记忆格式化成给模型看的一行文字。

    要求：
      - 包含 finding 和 evidence
      - 让模型知道这条记忆有多旧（不同时间的记忆，输出要不同）
      - 状态类记忆超过 VOLATILE_TTL_HOURS：明确提示"需要重新验证"
      - 结构类记忆：不需要这个提示
    """
    delta = now - m.observed_at
    humanize_delta = humanize.naturaltime(timedelta(seconds=delta))  # → "3天前"
    finding = m.finding
    evidence = m.evidence
    if m.kind == VOLATILE and delta / 3600 > VOLATILE_TTL_HOURS:
        return f"{finding}（依据：{evidence}）, 时间: {humanize_delta}. 状态类信息超过 24 小时，可能已变化，需要重新验证"
    else:
        return f"{finding}（依据：{evidence}）, 时间: {humanize_delta}"


class MemoryStore:
    """用一个 JSON 文件保存所有记忆。学习用，生产环境可换成数据库。"""

    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = path
        self.memories: list[Memory] = []
        if path.exists():
            self.memories = [Memory(**d) for d in json.loads(path.read_text(encoding="utf-8"))]

    def save_notes(self, notes: list[Note], now: float | None = None) -> int:
        """任务结束时调用：把本次笔记写入长期记忆。没有 host 的笔记无法检索，跳过。"""
        now = time.time() if now is None else now
        new = [Memory(n.host, n.finding, n.evidence, n.kind, now) for n in notes if n.host]
        if new:
            self.memories.extend(new)
            self.path.write_text(json.dumps([asdict(m) for m in self.memories], ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        return len(new)

    def recall(self, host: str) -> list[Memory]:
        """按主机精确匹配，最新的在前。"""
        hits = [m for m in self.memories if m.host == host]
        return sorted(hits, key=lambda m: m.observed_at, reverse=True)[:RECALL_LIMIT]

    def recall_text(self, host: str, now: float | None = None) -> str:
        """recall_memory 工具的实现。"""
        now = time.time() if now is None else now
        hits = self.recall(host)
        if not hits:
            return f"{host} 没有历史记录"
        return f"{host} 的历史记录（共 {len(hits)} 条，最新在前）：\n" + \
            "\n".join(format_memory(m, now) for m in hits)
