"""阶段 2 · 第 2 课（下）：自定义 State + 笔记

对照阶段 1 的 memory.py：
    Notebook 对象（进程内存里）   → State 的 notes 字段（会被 checkpoint 保存）
    notebook.add(...)            → record_finding 工具返回 Command(update={"notes": [...]})
    self.notes.append + 截断      → notes 字段的 reducer：merge_notes
    with_notes(messages, nb)     → agent 节点里把 render_notes(...) 临时加在请求末尾
"""
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import MessagesState
from langgraph.types import Command

from memory import MAX_NOTES, NOTES_PREFIX, STABLE, VOLATILE  # 复用阶段 1 的配置


def merge_notes(old: list[dict], new: list[dict]) -> list[dict]:
    """notes 字段的 reducer：把一次更新（new）合并进已有的笔记（old），返回合并后的完整列表。

    什么时候被调用：每次有节点/工具返回 {"notes": [...]}，框架就调用一次 merge_notes(当前值, 返回值)。
    模型在同一步里并行调用 3 次 record_finding，就会连续调用 3 次，每次 new 里有 1 条。

    要求：
      - 新笔记追加在已有笔记后面，顺序不变
      - 总数超过 MAX_NOTES 时，丢弃最早的，只保留最新的 MAX_NOTES 条
      - 不修改传入的 old 和 new，返回新列表
    """
    result_notes = old + new
    return result_notes[-MAX_NOTES:]


class AgentState(MessagesState):
    """messages 之外，再放三样"进程重启后不能丢"的东西。"""
    # 多个 record_finding 可能在同一步里各写一次 → 必须有 reducer，否则 InvalidUpdateError
    notes: Annotated[list[dict], merge_notes]
    # 下面两个字段只有 guard 节点会写，每步最多写一次 → 不需要 reducer，新值直接覆盖旧值
    call_counts: dict[str, int]  # 每个查询累计被调用的次数，跨步骤累积
    blocked: dict[str, int]      # 只对当前这一步有效：tool_call_id → 第几次重复。这些调用不执行，只返回提醒


def render_notes(notes: list[dict]) -> str | None:
    if not notes:
        return None
    lines = [f"{i}. {n['finding']}（依据：{n['evidence']}）" for i, n in enumerate(notes, 1)]
    return f"{NOTES_PREFIX}\n" + "\n".join(lines)


@tool
async def record_finding(finding: str, evidence: str, host: str, kind: Literal["状态", "结构"],
                         tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """记录一条关键发现。早期工具结果可能被压缩，笔记会一直保留。只记录有工具结果支撑的事实，可以和其他工具并行调用。

    Args:
        finding: 发现，一句话，保留具体数值
        evidence: 依据：哪个工具、哪台主机、哪条原始结果
        host: 这条发现属于哪台主机，如 web-01
        kind: 状态 = 会随时间变化的指标（磁盘、连接数）；结构 = 很少变化的事实（依赖关系、表结构）
    """
    # tool_call_id 由框架注入，不会出现在给模型的 schema 里。
    # 返回 Command 而不是字符串：除了 ToolMessage（配对用），还要更新 notes 字段。
    note = {"finding": finding, "evidence": evidence, "host": host, "kind": kind}
    return Command(update={
        "notes": [note],  # 只返回新增的这一条，由 merge_notes 合并
        "messages": [ToolMessage("已记录", tool_call_id=tool_call_id, name="record_finding")],
    })


assert {VOLATILE, STABLE} == {"状态", "结构"}  # Literal 里不能用变量，这里防止和阶段 1 的常量不一致
