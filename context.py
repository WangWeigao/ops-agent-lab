"""阶段 1 · 第 2–3 课：上下文管理（策略①：清理旧工具结果；策略③：摘要压缩）"""
from typing import Callable

CONTEXT_THRESHOLD = 1500  # 测试用的低阈值，生产环境要按模型上下文窗口设置
KEEP_RECENT_STEPS = 2     # 最近几步的工具结果完整保留
PLACEHOLDER = "[早期工具结果已省略]"
SUMMARY_PREFIX = "[早期排查摘要]"


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


def summarize_context(messages: list, last_prompt_tokens: int,
                      summarize: Callable[[list], str]) -> list:
    """策略③：超过阈值时，把较早的整步删掉，换成一条摘要消息。

    - 删除单位是"整步"（assistant + 它的全部 tool 结果），所以不会拆散 tool_call_id 配对
    - 摘要插在被删除的位置（原始 user 消息之后），system 和开头的前缀保持不变，利于缓存
    - summarize 由调用方传入：生产环境调用模型，测试时传一个假函数
    - 返回新列表，不修改传入的 messages
    """
    if last_prompt_tokens <= CONTEXT_THRESHOLD:
        return messages

    starts = [i for i, m in enumerate(messages)
              if m["role"] == "assistant" and m.get("tool_calls")]
    if len(starts) <= KEEP_RECENT_STEPS:
        return messages

    first, cut = starts[0], starts[-KEEP_RECENT_STEPS]
    summary = summarize(messages[first:cut])
    summary_msg = {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"}
    return messages[:first] + [summary_msg] + messages[cut:]
