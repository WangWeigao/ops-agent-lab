"""阶段 1 · 第 2 课：上下文管理（策略①：清理旧工具结果）"""

CONTEXT_THRESHOLD = 1500  # 测试用的低阈值，生产环境要按模型上下文窗口设置
KEEP_RECENT_STEPS = 2     # 最近几步的工具结果完整保留
PLACEHOLDER = "[早期工具结果已省略]"


def manage_context(messages: list, last_prompt_tokens: int) -> list:
    """上一次请求的 prompt_tokens 超过阈值时，把较早步骤的 tool 结果替换为占位文字。

    一步 = 一条带 tool_calls 的 assistant 消息 + 紧跟其后的全部 tool 消息。

    要求：
      - last_prompt_tokens 未超过 CONTEXT_THRESHOLD 时原样返回
      - system 消息永远保留
      - 最近 KEEP_RECENT_STEPS 步完整保留
      - 更早步骤的 tool 消息只替换 content，assistant 消息和 tool_call_id 都不动
    """
    # TODO: 你来实现（约 15 行）
    if last_prompt_tokens < CONTEXT_THRESHOLD:
        return messages

    starts = []
    for idx, message in enumerate(messages):
        role = message['role']
        if role == "assistant" and "tool_calls" in message:
            starts.append(idx)

    if len(starts) < KEEP_RECENT_STEPS:
        return messages
    cut = starts[-KEEP_RECENT_STEPS]

    for idx, _ in enumerate(messages):
        role = messages[idx]['role']
        if idx < cut and role == 'tool':
            messages[idx]['content'] = PLACEHOLDER




    return messages
