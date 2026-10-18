"""作用：调度在线连接的一次性提醒投递。

当前功能：
- 定时从 SQLite 领取到期提醒，断线未展示的事项重连补发。
- 展示回执提交后加入本连接的待播报队列，文本与语音交付分开。
- 播报前复核取消状态，关闭连接释放未确认租约。
"""

from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo
from src.repositories.reminders import ReminderRepository


def notification_text(row: dict) -> str:
    """以固定文案展示事项，不经过模型推测或改写。"""
    due = datetime.fromisoformat(row["due_at"]).astimezone(ZoneInfo(row["timezone"]))
    return f"提醒（原定 {due:%Y-%m-%d %H:%M}，{row['timezone']}）：{row['title']}。"


class ReminderDelivery:
    """每个 WebSocket 拥有独立投递租约和有界语音等待队列。"""

    def __init__(self, repository: ReminderRepository, user_id: str, owner: str):
        """使用已认证会话身份和连接代次，拒绝客户端指定归属。"""
        self.repository = repository
        self.user_id = user_id
        self.owner = owner
        self.pending: OrderedDict[str, dict] = OrderedDict()
        self.speech: OrderedDict[str, dict] = OrderedDict()

    async def poll(self) -> list[dict]:
        """有限领取到期事项；过期未确认的投递由数据库租约控制重试。"""
        rows = await self.repository.claim(self.user_id, self.owner)
        for row in rows:
            self.pending[row["reminder_id"]] = row
        # 活跃提醒配额为一百；清理已取消或被其他连接接管的旧凭据。
        while len(self.pending) > 100:
            self.pending.popitem(last=False)
        return [{**row, "text": notification_text(row)} for row in rows]

    async def acknowledge(self, reminder_id: str, token: str) -> bool:
        """仅确认本连接实际发出的提醒；重复回执不产生重复语音。"""
        row = self.pending.get(reminder_id)
        if not row or row["delivery_token"] != token:
            return False
        accepted = await self.repository.acknowledge(self.user_id, self.owner, reminder_id, token)
        if accepted:
            self.pending.pop(reminder_id, None)
            self.speech[reminder_id] = row
            while len(self.speech) > 100:
                self.speech.popitem(last=False)
        return accepted

    async def next_speech(self) -> dict | None:
        """取一条仍有效的已展示提醒；语音被打断后不自动重播。"""
        while self.speech:
            reminder_id, row = self.speech.popitem(last=False)
            current = await self.repository.get(self.user_id, reminder_id)
            if current and current["status"] == "delivered":
                return row
        return None

    async def close(self) -> None:
        """释放本连接未展示租约，已展示通知不因断线重复展示。"""
        await self.repository.release(self.owner)
