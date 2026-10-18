"""作用：在现有 SQLite 文件中持久化一次性提醒。

当前功能：
- 按可信用户隔离创建、查询与取消，独立于长期记忆记录。
- 使用事务保存业务写入和幂等回执，重复轮次不创建第二条提醒。
- 以短租约领取到期提醒，只有匹配连接凭据的展示回执才能完成投递。
"""

import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.contracts.reminders import ReminderRecord
from src.repositories.memory import MemoryRepository, token_digest
from src.tools.executor import ToolFailure


def public_record(row) -> dict:
    """仅输出工具契约字段，避免公开用户、幂等键或交付租约。"""
    return ReminderRecord(**{name: row[name] for name in ReminderRecord.model_fields}).model_dump()


class ReminderRepository:
    """复用现有 SQLite 连接与事务设施，提醒使用单独的表。"""

    def __init__(self, database: MemoryRepository):
        """幂等初始化提醒、索引和业务操作回执。"""
        self.database = database
        with closing(database.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL, title TEXT NOT NULL,
                    due_at TEXT NOT NULL, timezone TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
                    delivered_at TEXT, delivery_owner TEXT, delivery_token TEXT,
                    lease_until TEXT);
                CREATE INDEX IF NOT EXISTS reminders_due ON reminders(user_id,status,due_at);
                CREATE TABLE IF NOT EXISTS reminder_operations (
                    user_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY(user_id,operation_id));
            """)
            db.commit()

    async def mutate(self, user_id: str, operation_id: str, request: dict, operation) -> dict:
        """在同一事务内核验身份、幂等键并提交变更与原始回执。"""
        fingerprint = token_digest(json.dumps(request, ensure_ascii=False, sort_keys=True))

        def apply(db):
            """不同内容复用同一操作键时拒绝执行，已提交的操作直接返回回执。"""
            if not db.execute("SELECT 1 FROM memory_users WHERE user_id=?", (user_id,)).fetchone():
                raise ToolFailure("identity_required", "持久化用户身份不可用，请重新连接。")
            previous = db.execute("SELECT * FROM reminder_operations WHERE user_id=? AND operation_id=?", (user_id, operation_id)).fetchone()
            if previous:
                if previous["request_hash"] != fingerprint:
                    raise ToolFailure("operation_conflict", "同一轮次对应不同提醒操作，请先查看提醒列表，再用新轮次明确请求。")
                return json.loads(previous["result"])
            result = operation(db)
            db.execute("INSERT INTO reminder_operations VALUES (?,?,?,?)", (user_id, operation_id, fingerprint, json.dumps(result, ensure_ascii=False)))
            return result

        return await self.database.run(apply)

    async def create(self, context, title: str, due_at: datetime, zone: str, arguments: dict) -> dict:
        """以原始参数计算幂等指纹，相对时间重试不改变首次提交的时间。"""
        def insert(db):
            """限制活跃提醒数量，原子创建用户明确请求的一次性事项。"""
            count = db.execute("SELECT count(*) FROM reminders WHERE user_id=? AND status='pending'", (context.user_id,)).fetchone()[0]
            if count >= 100:
                raise ToolFailure("limit_reached", "待提醒事项已达到 100 条，请先取消不需要的提醒。")
            reminder_id = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO reminders(reminder_id,user_id,source_session_id,title,due_at,timezone,created_at) VALUES (?,?,?,?,?,?,?)",
                       (reminder_id, context.user_id, context.session_id, title, due_at.isoformat(), zone, now))
            return public_record(db.execute("SELECT * FROM reminders WHERE reminder_id=?", (reminder_id,)).fetchone())

        return await self.mutate(context.user_id, context.operation_id, {"action": "create", "arguments": arguments}, insert)

    async def list_pending(self, user_id: str) -> dict:
        """返回全部未完成提醒，上限与创建配额保持一致。"""
        def read(db):
            """读取当前用户仍待投递的记录，到期未展示的提醒仍包含在内。"""
            rows = db.execute("SELECT * FROM reminders WHERE user_id=? AND status='pending' ORDER BY due_at,reminder_id LIMIT 100", (user_id,)).fetchall()
            return {"reminders": [public_record(row) for row in rows]}

        return await self.database.run(read)

    async def cancel(self, context, target: str) -> dict:
        """只取消所属用户的待提醒事项，刚才指向同会话最新创建记录。"""
        def change(db):
            """标题重名时追问标识，不猜测或批量取消。"""
            if target == "latest":
                rows = db.execute("SELECT * FROM reminders WHERE user_id=? AND source_session_id=? ORDER BY created_at DESC LIMIT 1",
                                  (context.user_id, context.session_id)).fetchall()
            else:
                rows = db.execute("SELECT * FROM reminders WHERE user_id=? AND (reminder_id=? OR title=?) ORDER BY created_at DESC LIMIT 2",
                                  (context.user_id, target, target)).fetchall()
            if len(rows) != 1:
                raise ToolFailure("clarification_required", "没有找到唯一对应的提醒，请查看提醒列表并提供要取消的提醒编号。", ["target"])
            row = rows[0]
            if row["status"] == "delivered":
                raise ToolFailure("already_delivered", "这条提醒已经展示，无法撤回已送达的通知。")
            db.execute("UPDATE reminders SET status='cancelled',delivery_owner=NULL,delivery_token=NULL,lease_until=NULL WHERE reminder_id=?", (row["reminder_id"],))
            return public_record(db.execute("SELECT * FROM reminders WHERE reminder_id=?", (row["reminder_id"],)).fetchone())

        return await self.mutate(context.user_id, context.operation_id, {"action": "cancel", "target": target}, change)

    async def claim(self, user_id: str, owner: str, now: datetime | None = None) -> list[dict]:
        """领取最多十条到期通知；无在线连接时记录留在数据库等待补发。"""
        now = now or datetime.now(timezone.utc)

        def take(db):
            """多连接竞争同一事务，只有过期或未领取的提醒可以投递。"""
            rows = db.execute("SELECT * FROM reminders WHERE user_id=? AND status='pending' AND due_at<=? AND (lease_until IS NULL OR lease_until<=?) ORDER BY due_at LIMIT 10",
                              (user_id, now.isoformat(), now.isoformat())).fetchall()
            result = []
            for row in rows:
                token = str(uuid4())
                db.execute("UPDATE reminders SET delivery_owner=?,delivery_token=?,lease_until=? WHERE reminder_id=?",
                           (owner, token, (now + timedelta(seconds=30)).isoformat(), row["reminder_id"]))
                result.append({**public_record(row), "delivery_token": token})
            return result

        return await self.database.run(take)

    async def acknowledge(self, user_id: str, owner: str, reminder_id: str, token: str) -> bool:
        """以用户、连接和租约的组合验证展示回执，过期投递不能确认新投递。"""
        def mark(db):
            """数据库中的 delivered 仅表示已展示，不表示已播音或用户已完成事项。"""
            cursor = db.execute("UPDATE reminders SET status='delivered',delivered_at=?,lease_until=NULL WHERE user_id=? AND reminder_id=? AND status='pending' AND delivery_owner=? AND delivery_token=?",
                                (datetime.now(timezone.utc).isoformat(), user_id, reminder_id, owner, token))
            return cursor.rowcount == 1

        return await self.database.run(mark)

    async def get(self, user_id: str, reminder_id: str) -> dict | None:
        """读取指定提醒的实时状态，排队播报前核查取消。"""
        def read(db):
            """只有可信用户自己的提醒可以被获取。"""
            row = db.execute("SELECT * FROM reminders WHERE user_id=? AND reminder_id=?", (user_id, reminder_id)).fetchone()
            return public_record(row) if row else None

        return await self.database.run(read)

    async def release(self, owner: str) -> None:
        """连接关闭时释放未确认投递，新连接可以立即补发。"""
        def clear(db):
            """只清除指定连接持有的待投递租约，不影响已展示状态。"""
            db.execute("UPDATE reminders SET delivery_owner=NULL,delivery_token=NULL,lease_until=NULL WHERE status='pending' AND delivery_owner=?", (owner,))

        await self.database.run(clear)
