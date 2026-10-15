"""阶段 2 · 第 3 课：Checkpoint 持久化 + 写操作的幂等

实验结论：checkpoint 的粒度是"节点"。tools 节点并行执行 3 个工具，执行到一半进程被杀，
恢复时整个节点重新执行，已经成功的工具会再执行一次。
只读工具无所谓；写操作（重启服务、发通知）不能执行两次。

State 管不到节点内部的"部分完成"，所以要在 State 之外另外记一本账（Ledger）：
每个写操作执行前登记、执行后记下结果，恢复时先查账。

命令行入口在 stage2/approval.py（第 4 课加了审批后，入口要同时处理"等人审批"这种状态）。
"""
import asyncio
import sqlite3
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from loguru import logger

import main as stage1

CHECKPOINT_DB = "checkpoints.db"
LEDGER_DB = "ledger.db"

# ---------- 一个模拟的写操作 ----------
RESTARTS: list[tuple[str, str]] = []  # 记录"真的被执行了几次"，测试用


@tool
async def restart_service(host: str, service: str) -> str:
    """重启指定主机上的服务。这是写操作，会造成短暂中断，只在确认根因、且重启能解决问题时使用。"""
    await asyncio.sleep(stage1.TOOL_LATENCY)
    RESTARTS.append((host, service))
    return f"{host} 上的 {service} 已重启"


# 没有外部副作用的工具：重新执行一次没有影响，不需要记账（下一课：也不需要审批）。
#   get_logs / check_disk  只读
#   record_finding         只改 State，随节点一起成功或一起作废，不存在"部分完成"
# 写成"安全名单"而不是"危险名单"：新加的工具忘了登记时，按有副作用处理（多一次保护），
# 而不是不经任何保护直接执行。这份名单由人维护，不让模型判断，也不直接采信工具自己的声明。
NO_SIDE_EFFECT_TOOLS = {"get_logs", "check_disk", "record_finding"}


def has_side_effects(tool_name: str) -> bool:
    return tool_name not in NO_SIDE_EFFECT_TOOLS


# ---------- 账本 ----------
@dataclass(frozen=True)
class Entry:
    done: bool     # False：登记了开始，但没有记下结果（执行到一半进程没了，或者执行时抛了异常）
    content: str   # 工具结果（done 为 True 时才有意义）
    status: str    # "success" / "error"


class Ledger:
    """写操作的执行记录，存在 SQLite 里。每次写入立即提交，进程被杀也不会丢。

    用同步的 sqlite3：单条写入很快。真正上生产要换成异步驱动，否则会短暂阻塞事件循环（阶段 1 第 5 课）。
    """

    def __init__(self, path: str = LEDGER_DB) -> None:
        self.db = sqlite3.connect(path, isolation_level=None)  # isolation_level=None：自动提交
        self.db.execute("CREATE TABLE IF NOT EXISTS ledger "
                        "(key TEXT PRIMARY KEY, done INTEGER NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL)")

    def get(self, key: str) -> Entry | None:
        row = self.db.execute("SELECT done, content, status FROM ledger WHERE key = ?", (key,)).fetchone()
        return Entry(bool(row[0]), row[1], row[2]) if row else None

    def start(self, key: str) -> None:
        """登记"开始执行"。key 已存在时抛 sqlite3.IntegrityError。"""
        self.db.execute("INSERT INTO ledger VALUES (?, 0, '', '')", (key,))

    def finish(self, key: str, content: str, status: str = "success") -> None:
        """记下执行结果。"""
        self.db.execute("UPDATE ledger SET done = 1, content = ?, status = ? WHERE key = ?", (content, status, key))


def reply(call: dict, content: str, status: str = "success") -> ToolMessage:
    """不执行工具，直接生成这次调用的 ToolMessage。"""
    return ToolMessage(content, tool_call_id=call["id"], name=call["name"], status=status)


async def run_once(ledger: Ledger, key: str, call: dict, execute) -> ToolMessage:
    """保证一个写操作最多执行一次。

    参数：
      key      这次调用的唯一标识（thread_id + tool_call_id）。同一个 key = 同一次调用
      call     工具调用，{"name": ..., "args": ..., "id": ...}
      execute  await execute() 真正执行工具，返回 ToolMessage（有 .content 和 .status）

    可以用的：
      ledger.get(key)    → None（没执行过）/ Entry(done, content, status)
      ledger.start(key)  登记开始
      ledger.finish(key, content, status)  记下结果
      reply(call, content, status)  不执行工具，直接生成 ToolMessage

    账本里有三种情况，各自该怎么办：
      1. 没有记录
      2. 有记录，done=True
      3. 有记录，done=False —— 上次登记了开始，但没有记下结果
    """
    entry = ledger.get(key)
    if entry is None:
        ledger.start(key)
        tool_message = await execute()
        ledger.finish(key, tool_message.content, tool_message.status)
        return tool_message
    elif entry.done:
        return reply(call, entry.content, entry.status)
    else:
        logger.warning(f"{key}: 上次执行中断，结果未知，本次未执行，需要人工确认")
        return reply(call, "上次执行这个操作时中断，结果未知，本次没有执行。不要重试；"
                           "先用只读工具确认当前状态，无法确认时如实告诉用户需要人工核实。", "error")


# ---------- 入口：新的提问，还是从断点继续 ----------
def is_unfinished(snapshot) -> bool:
    """这个 thread 上次有没有跑完。

    不能用 snapshot.next 判断：next 只列出"还没有结果的任务"。
    如果节点的结果已经落盘、但下一个 checkpoint 还没写进程就没了（checkpoint 默认在后台异步写入，
    这个窗口确实存在），next 是空的，看起来像跑完了，其实没有。tasks 在这种情况下仍然不为空。
    """
    return bool(snapshot.tasks)
