"""阶段 2 · 第 1 课：用 LangGraph 重写最小 Agent 循环

对照阶段 1 的 main.py：
    messages 列表          → State（MessagesState，messages 字段用 add_messages 追加）
    调用模型那一段          → agent 节点
    run_tool + 并发执行     → tools 节点（本课手写，下节课换成官方 ToolNode）
    if not msg.tool_calls  → 条件边
    for step in range(N)   → 图里的环 + recursion_limit

运行：uv run python -m stage2.basic
"""
import asyncio
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from loguru import logger

import main as stage1  # 复用阶段 1 的模拟工具（get_logs / check_disk，各带 1 秒延迟）

load_dotenv(".env")

SYSTEM_PROMPT = "你是运维助手。需要数据时调用工具，不要编造。"
RECURSION_LIMIT = 25  # 对应阶段 1 的 MAX_STEPS。注意：它数的是"节点执行次数"，一轮 agent + tools 算 2 次


# ---------- 工具：用 @tool 包装，框架从签名和 docstring 自动生成 schema ----------
@tool
async def get_logs(host: str, lines: int = 40) -> str:
    """获取指定主机最近的应用/数据库日志。host 如 web-01，lines 为返回行数（1-200）。"""
    return await stage1.get_logs(host, lines)


@tool
async def check_disk(host: str) -> str:
    """查询指定主机的磁盘使用率。host 如 web-01。"""
    return json.dumps(await stage1.check_disk(host), ensure_ascii=False)


TOOLS = [get_logs, check_disk]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------- 节点：输入整个 State，返回要更新的部分 ----------
def make_agent_node(model):
    """model 由外部传入（依赖注入）：生产用 ChatDeepSeek，测试用脚本化的假模型。"""

    async def agent(state: MessagesState) -> dict:
        # system prompt 不存进 State，每次调用时临时加在最前面
        response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        # 只返回新增的这一条。add_messages 会把它追加到 messages 末尾，相当于阶段 1 的 messages.append
        return {"messages": [response]}

    return agent

async def route_agent_node(state: MessagesState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools_node"
    else:
        return END

async def tools_node(state: MessagesState) -> dict:
    """手写版工具节点：并发执行最后一条 AIMessage 里的全部 tool_calls，按原顺序返回 ToolMessage。"""
    last: AIMessage = state["messages"][-1]

    async def run_one(call: dict) -> ToolMessage:
        tool_fn = TOOLS_BY_NAME.get(call["name"])
        try:
            if tool_fn is None:
                content = f"工具 {call['name']} 不存在，可用工具: {list(TOOLS_BY_NAME)}"
            else:
                # call["args"] 已经是 dict：阶段 1 里自己做的 json.loads，框架替你做了
                content = await tool_fn.ainvoke(call["args"])
        except Exception as e:
            content = f"{type(e).__name__}: {str(e)[:200]}"
        logger.info(f"  ← {call['name']} {str(content)[:60]!r}")
        # tool_call_id 配对：和阶段 1 一样，缺了它下一次请求就会 400
        return ToolMessage(content=str(content), tool_call_id=call["id"], name=call["name"])

    for call in last.tool_calls:
        logger.info(f"  → {call['name']} {call['args']}")
    return {"messages": list(await asyncio.gather(*(run_one(c) for c in last.tool_calls)))}


# ---------- 图 ----------
def build_graph(model):
    """把节点和边组装成图并编译。

    结构：START → agent →(条件边)→ tools → agent → ... →(条件边)→ END

    条件边的规则：最后一条 AIMessage 有 tool_calls → 去 "tools"；没有 → END
    （对应阶段 1 的 `if not msg.tool_calls: return msg.content`）

    节点：make_agent_node(model) 生成 agent 节点；tools_node 是工具节点
    提示：StateGraph(MessagesState)、add_node、add_edge、add_conditional_edges、compile
    """
    builder = StateGraph(MessagesState)
    builder.add_node("agent_node", make_agent_node(model))
    builder.add_node("tools_node", tools_node)
    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", route_agent_node, ['tools_node', END])
    builder.add_edge("tools_node", "agent_node")
    graph = builder.compile()

    return graph



async def run(question: str) -> str:
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(model=stage1.MODEL).bind_tools(TOOLS)
    graph = build_graph(model)
    # stream_mode="updates"：每个节点执行完，产出一次它对 State 的更新，方便观察图是怎么走的
    final = None
    async for update in graph.astream({"messages": [HumanMessage(question)]},
                                      config={"recursion_limit": RECURSION_LIMIT},
                                      stream_mode="updates"):
        for node, delta in update.items():
            msg = delta["messages"][-1]
            logger.info(f"[{node}] 产出 {len(delta['messages'])} 条消息"
                        + (f"，{len(msg.tool_calls)} 个工具调用" if getattr(msg, "tool_calls", None) else ""))
            final = msg
    return final.content


if __name__ == "__main__":
    print(asyncio.run(run("web-01 上的订单接口最近很慢，帮我排查根因。先看 web-01 的日志和磁盘，再根据线索看相关主机的日志。")))
