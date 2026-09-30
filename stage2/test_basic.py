"""阶段 2 · 第 1 课：build_graph 的离线测试，不调用 API。

运行：uv run python -m stage2.test_basic
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import main as stage1
from stage2.basic import build_graph

stage1.TOOL_LATENCY = 0  # 测试不需要模拟 SSH 延迟


class ScriptedModel:
    """按顺序返回预先写好的 AIMessage，代替真实模型。记录每次收到的消息，方便检查。"""

    def __init__(self, replies: list[AIMessage], repeat_last: bool = False):
        self.replies, self.repeat_last, self.calls = list(replies), repeat_last, []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if len(self.replies) > 1 or not self.repeat_last else self.replies[0]
        # 每次返回一个没有 id 的副本，和真实模型一样。
        # add_messages 第一次追加时会把 id 写回原对象；同一个对象再返回，会被当成"更新"而不是"追加"
        return reply.model_copy(update={"id": None})


def call(name: str, args: dict, id_: str) -> dict:
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n   {detail}" if not cond and detail else ""))
    return cond


async def main() -> None:
    results = []

    # 1. 不调工具，直接回答：agent → END
    model = ScriptedModel([AIMessage("磁盘正常")])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("web-01 怎么样")]})
        msgs = out["messages"]
        results.append(check("没有 tool_calls 时直接结束", len(msgs) == 2 and msgs[-1].content == "磁盘正常",
                             f"messages: {[type(m).__name__ for m in msgs]}"))
    except Exception as e:
        results.append(check("没有 tool_calls 时直接结束", False, f"{type(e).__name__}: {e}"))

    # 2. 一轮并行工具调用，再给结论：agent → tools → agent → END
    model = ScriptedModel([
        AIMessage("", tool_calls=[call("check_disk", {"host": "web-01"}, "a"),
                                  call("get_logs", {"host": "web-01", "lines": 3}, "b")]),
        AIMessage("web-01 磁盘 91%"),
    ])
    try:
        out = await build_graph(model).ainvoke({"messages": [HumanMessage("排查 web-01")]})
        msgs = out["messages"]
        kinds = [type(m).__name__ for m in msgs]
        results.append(check("消息顺序：Human → AI(tool_calls) → Tool × 2 → AI",
                             kinds == ["HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage"],
                             f"实际: {kinds}"))
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        results.append(check("tool_call_id 按原顺序配对", [m.tool_call_id for m in tool_msgs] == ["a", "b"]))
        results.append(check("工具真的执行了", "91" in tool_msgs[0].content if tool_msgs else False,
                             f"check_disk 结果: {tool_msgs[0].content if tool_msgs else None}"))
        results.append(check("第二次调用模型时，带上了工具结果（State 累积）",
                             len(model.calls) == 2 and any(isinstance(m, ToolMessage) for m in model.calls[1])))
    except Exception as e:
        for name in ("消息顺序", "配对", "工具执行", "State 累积"):
            results.append(check(name, False, f"{type(e).__name__}: {e}"))

    # 3. 模型一直调工具：recursion_limit 兜底（对应阶段 1 的 MAX_STEPS）
    model = ScriptedModel([AIMessage("", tool_calls=[call("check_disk", {"host": "web-01"}, "x")])],
                          repeat_last=True)
    try:
        await build_graph(model).ainvoke({"messages": [HumanMessage("查")]}, config={"recursion_limit": 6})
        results.append(check("一直调工具时，被 recursion_limit 拦下", False, "没有抛 GraphRecursionError"))
    except GraphRecursionError:
        results.append(check("一直调工具时，被 recursion_limit 拦下", True))
    except Exception as e:
        results.append(check("一直调工具时，被 recursion_limit 拦下", False, f"{type(e).__name__}: {e}"))

    print(f"\n{sum(results)}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
