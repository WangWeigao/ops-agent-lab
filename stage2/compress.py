"""阶段 2 · 第 2 课（上）：compress 节点（策略①：把较早的工具结果替换成占位文字）

对照阶段 1 的 context.manage_context：
    last_prompt_tokens 变量     → 最后一条 AIMessage.usage_metadata["input_tokens"]（已经在 State 里）
    原地修改 messages[i]["content"] → 返回同一个 id 的新 ToolMessage，由 add_messages 原地替换
"""
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import MessagesState

from context import CONTEXT_THRESHOLD, KEEP_RECENT_STEPS, PLACEHOLDER  # 复用阶段 1 的配置


def last_prompt_tokens(messages: list[AnyMessage]) -> int:
    """上一次模型请求的 prompt token 数。还没调用过模型时返回 0。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.usage_metadata:
            return m.usage_metadata["input_tokens"]
    return 0


def compress_node(state: MessagesState) -> dict:
    """策略①：prompt 超过阈值时，把较早步骤的工具结果替换成占位文字。

    一步 = 一条带 tool_calls 的 AIMessage + 紧跟其后的全部 ToolMessage。

    要求：
      - last_prompt_tokens(...) 未超过 CONTEXT_THRESHOLD，或步数不超过 KEEP_RECENT_STEPS：不做任何修改，返回 {}
      - 最近 KEEP_RECENT_STEPS 步完整保留
      - 更早步骤的 ToolMessage：content 换成 PLACEHOLDER，id 和 tool_call_id 保持不变
      - 只返回"要改的那几条"：{"messages": [新的 ToolMessage, ...]}，不要返回完整列表
      - 已经是 PLACEHOLDER 的不用再返回
    """
    # TODO(human)
    return {}
