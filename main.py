"""阶段 1 · 第 1–2 课：最小 Agent 循环 + 上下文管理（同步版，不用任何 Agent 框架）

运行前：
    uv sync
    在项目根目录的 .env 中写入 DEEPSEEK_API_KEY=你的key（.env 已加入 .gitignore）
模型名以 DeepSeek 官方文档当前列出的为准。
"""
import json
import os
from typing import Callable

from loguru import logger
from openai import OpenAI
import jsonschema
from dotenv import load_dotenv

from context import manage_context

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-flash"
MAX_STEPS = 12


# ---------- 工具：实现 + 描述（schema） ----------
def check_disk(host: str) -> dict:
    """模拟数据，先不接真实服务器。"""
    fake = {"web-01": 91, "db-01": 45}
    if host not in fake:
        raise ValueError(f"unknown host: {host}")
    return {"host": host, "usage_percent": fake[host]}


def get_logs(host: str, lines: int = 40) -> str:
    """模拟日志，故意返回较长内容，用来触发上下文压缩。"""
    templates = {
        "web-01": [
            "INFO  nginx upstream response time 1.2s /api/orders",
            "WARN  nginx upstream response time 4.8s /api/orders",
            "ERROR app write /var/log/app/access.log failed: No space left on device",
            "WARN  app db query slow (3.1s): SELECT * FROM orders WHERE user_id=?",
            "INFO  app GC pause 120ms",
        ],
        "db-01": [
            "INFO  mysqld slow query 2.9s: SELECT * FROM orders WHERE user_id=? (rows examined 1.2M)",
            "WARN  mysqld missing index on orders.user_id",
            "INFO  mysqld connections 180/500",
            "INFO  mysqld innodb buffer pool hit rate 97.1%",
        ],
    }
    if host not in templates:
        raise ValueError(f"unknown host: {host}，可用主机: {list(templates)}")
    rows = templates[host]
    return "\n".join(f"2026-09-27 10:{i // 60:02d}:{i % 60:02d} {rows[i % len(rows)]}" for i in range(lines))


TOOLS = {"check_disk": check_disk, "get_logs": get_logs}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_disk",
            "description": "查询指定主机的磁盘使用率",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "主机名，如 web-01"}
                },
                "required": ["host"],
                # 多传的参数在校验阶段就拦住，而不是等到 **kwargs 解包时抛 TypeError
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "获取指定主机最近的应用/数据库日志",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "主机名，如 web-01"},
                    "lines": {"type": "integer", "description": "返回行数，默认 40", "minimum": 1, "maximum": 200},
                },
                "required": ["host"],
                "additionalProperties": False,
            },
        },
    },
]

# 按工具名索引 schema，避免每次都遍历列表
SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}


def validate_args(name: str, args: dict) -> str | None:
    """校验通过返回 None，失败返回给模型看的错误说明。"""
    try:
        jsonschema.validate(args, SCHEMA_BY_NAME[name])
        return None
    except jsonschema.ValidationError as e:
        return f"参数校验失败: {e.message}"


def _format_error_message(message: str) -> str:
    return json.dumps(
        {
            "error": True,
            "message": message
        }, ensure_ascii=False
    )


def run_tool(name: str, args_json: str) -> str:
    # TODO 1: 解析 args_json，调用 TOOLS[name]，把结果转成 JSON 字符串返回。
    # 要求：工具不存在、参数解析失败、工具抛异常，这三种情况都不能让程序崩溃，
    #       而是把错误信息作为字符串返回。
    tool_func: Callable | None = TOOLS.get(name)
    if not tool_func:
        return _format_error_message(f"工具 {name} 不存在，可用工具: {list(TOOLS)}")

    try:
        args_dict = json.loads(args_json)
    except json.JSONDecodeError as e:
        # 已知异常：信息安全且能指出错误位置，可以直接给模型
        return _format_error_message(f"参数不是合法 JSON: {e}")

    if error := validate_args(name, args_dict):
        return _format_error_message(error)

    try:
        result = tool_func(**args_dict)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        # 未知异常：完整堆栈写日志给人排查，给模型的只是类型名 + 截断后的信息
        logger.exception(f"工具 {name} 执行异常")
        return _format_error_message(f"{type(e).__name__}: {str(e)[:200]}")



def run_agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": "你是运维助手。需要数据时调用工具，不要编造。"},
        {"role": "user", "content": user_input},
    ]

    last_prompt_tokens = 0
    for step in range(MAX_STEPS):
        logger.info(f'执行Agent循环---{step}')
        # 第 2 课：每次请求前按上一次的 prompt_tokens 判断是否压缩
        messages = manage_context(messages, last_prompt_tokens)
        # 在压缩之后、请求之前打印，看到的就是实际发给模型的内容
        logger.info(f"messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        last_prompt_tokens = resp.usage.prompt_tokens
        logger.info(f'prompt_tokens: {last_prompt_tokens}')

        msg = resp.choices[0].message
        logger.info(f'执行Agent循环---获取结果1: {msg.model_dump(exclude_none=True)}')

        # TODO 2: 如果 msg 没有 tool_calls，说明模型认为任务完成，返回 msg.content
        if not msg.tool_calls:
            return msg.content

        # TODO 3: 把模型这条消息（含 tool_calls）追加进 messages。
        #         提示：msg.model_dump(exclude_none=True) 可以转成 dict
        messages.append(msg.model_dump(exclude_none=True))

        logger.info(msg.model_dump(exclude_none=True))

        # TODO 4: 遍历 msg.tool_calls，用 run_tool 执行每一个，
        #         再以 {"role": "tool", "tool_call_id": ..., "content": ...} 追加结果
        for tool in msg.tool_calls:
            logger.info(f"tool call: ")
            func_name = tool.function.name
            func_args = tool.function.arguments
            logger.info(f'func_name: {func_name}')
            logger.info(f'func_args: {func_args}')
            content = run_tool(func_name, func_args)
            logger.info(f'tool_call_response: {content}')
            messages.append({
                "role": "tool", "tool_call_id": tool.id, "content": content
            })


    return "达到最大步数，任务未完成"


if __name__ == "__main__":
    # 第 1 课
    # print(run_agent("web-01 和 db-01 的磁盘哪个需要处理？"))
    # print(run_agent("查一下 cache-01 的磁盘"))
    # 第 2 课：需要多步排查、会拉多次日志，用来触发上下文压缩
    print(run_agent("web-01 上的订单接口最近很慢，帮我排查根因。先看 web-01 的日志和磁盘，再根据线索看相关主机的日志。"))
