"""阶段 1 · 第 4 课：记忆（Agent 边排查边记笔记）"""
from copy import deepcopy
from dataclasses import dataclass

MAX_NOTES = 20  # 笔记本身也会无限增长，超出后丢弃最早的
NOTES_PREFIX = "[排查笔记]"

# 第 4 课（下）：记忆类型决定过期速度
VOLATILE = "状态"   # 磁盘使用率、连接数等，几小时就可能变
STABLE = "结构"     # 依赖关系、缺索引等，很少变

RECORD_FINDING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_finding",
        "description": "记录一条关键发现。早期工具结果可能被压缩，笔记会一直保留。"
                       "只记录有工具结果支撑的事实，可以和其他工具并行调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "发现，一句话，保留具体数值"},
                "evidence": {"type": "string", "description": "依据：哪个工具、哪台主机、哪条原始结果"},
                "host": {"type": "string", "description": "这条发现属于哪台主机，如 web-01"},
                "kind": {"type": "string", "enum": [VOLATILE, STABLE],
                         "description": f"{VOLATILE}：会随时间变化的指标（磁盘、连接数）；{STABLE}：很少变化的事实（依赖关系、表结构）"},
            },
            "required": ["finding", "evidence", "host", "kind"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Note:
    finding: str
    evidence: str
    host: str = ""
    kind: str = VOLATILE


class Notebook:
    """一次 run_agent 内的笔记。不存进 messages，所以不会被压缩。"""

    def __init__(self) -> None:
        self.notes: list[Note] = []

    def add(self, finding: str, evidence: str, host: str = "", kind: str = VOLATILE) -> dict:
        """record_finding 工具的实现。"""
        self.notes.append(Note(finding, evidence, host, kind))
        dropped = max(0, len(self.notes) - MAX_NOTES)
        self.notes = self.notes[dropped:]
        return {"recorded": True, "total_notes": len(self.notes)}

    def render(self) -> str | None:
        if not self.notes:
            return None
        lines = [f"{i}. {n.finding}（依据：{n.evidence}）" for i, n in enumerate(self.notes, 1)]
        return f"{NOTES_PREFIX}\n" + "\n".join(lines)


def with_notes(messages: list, notebook: Notebook) -> list:
    """生成本次实际发给模型的 messages：在合适的位置放入一条笔记消息。

    要求：
      - 笔记为空时原样返回
      - 返回新列表，不修改传入的 messages（笔记只存在于本次请求，不会累积进历史）
      - 笔记消息：{"role": "user", "content": notebook.render()}，只出现一次
      - 不能破坏 tool_call_id 配对
    """
    if not notebook.notes:
        return messages

    new_messages = deepcopy(messages)
    new_messages.append({"role": "user", "content": notebook.render()})
    return new_messages
