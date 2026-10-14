"""作用：以 SQLite 事务保存长期记忆和浏览器身份。

当前功能：
- 使用散列后的随机凭据关联长期用户，数据库不保存原始凭据。
- 提供按用户隔离的查询、版本更正、物理删除和幂等回执。
- 在线程中执行短事务；取消与版本检查防止迟到提取重新写回已删除数据。
"""

import asyncio
import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4
from contextlib import closing

from src.contracts.memory import MemoryCandidate, MemoryRecord, MemoryResult


def token_digest(token: str) -> str:
    """将不可预测的浏览器凭据转换为数据库比较值。"""
    return hashlib.sha256(token.encode()).hexdigest()


class MemoryRepository:
    """管理一个本地数据库文件，每次操作使用独立连接和有限锁等待。"""

    def __init__(self, path: str):
        """创建存储目录和基础表，启动故障由装配层降级处理。"""
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memory_users (
                    user_id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, persona_id TEXT NOT NULL,
                    kind TEXT NOT NULL, fact_key TEXT NOT NULL, content TEXT NOT NULL,
                    evidence TEXT NOT NULL, source_session_id TEXT NOT NULL, source_turn_id TEXT NOT NULL,
                    version INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    valid_to TEXT, supersedes_id TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS memory_active_key
                    ON memory_records(user_id, persona_id, fact_key) WHERE status='active';
                CREATE INDEX IF NOT EXISTS memory_owner ON memory_records(user_id, status);
                CREATE TABLE IF NOT EXISTS memory_operations (
                    user_id TEXT NOT NULL, operation_id TEXT NOT NULL, result TEXT NOT NULL, request_hash TEXT NOT NULL,
                    PRIMARY KEY(user_id, operation_id));
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_operations)")}
            if "request_hash" not in columns:
                db.execute("ALTER TABLE memory_operations ADD COLUMN request_hash TEXT NOT NULL DEFAULT ''")
            db.commit()

    def connect(self):
        """返回有限等待的连接，关闭时覆盖已删除的数据库页。"""
        db = sqlite3.connect(self.path, timeout=0.25)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        return db

    async def run(self, operation, after_commit=None):
        """执行一个可取消事务；提交前检查取消，已提交的事实不因输出中断回滚。"""
        cancelled = Event()
        loop = asyncio.get_running_loop()

        def execute():
            """在单独线程内持有连接，并在退出时关闭数据库文件。"""
            db = self.connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                result = operation(db)
                if cancelled.is_set():
                    db.rollback()
                    return None
                db.commit()
                if after_commit:
                    loop.call_soon_threadsafe(after_commit, result)
                return result
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

        try:
            return await asyncio.to_thread(execute)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def identity(self, token: str | None, *, create: bool = False):
        """验证浏览器凭据；仅创建会话接口可以签发新身份。"""
        def resolve(db):
            """优先恢复已有用户，未知凭据不能指定任意用户标识。"""
            if token and len(token) <= 128:
                row = db.execute("SELECT user_id FROM memory_users WHERE token_hash=?", (token_digest(token),)).fetchone()
                if row:
                    return row["user_id"], token
            if not create:
                return None
            issued = secrets.token_urlsafe(32)
            user_id = str(uuid4())
            db.execute("INSERT INTO memory_users(user_id,token_hash) VALUES (?,?)", (user_id, token_digest(issued)))
            return user_id, issued
        return await self.run(resolve)

    async def snapshot(self, user_id: str, *, historical: bool = False):
        """读取用户事实及同一事务下的版本号，供后续乐观并发检查。"""
        def read(db):
            """读取有界记录；写入总量上限保证查询不会静默遗漏活跃记录。"""
            owner = db.execute("SELECT revision FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
            if owner is None:
                raise ValueError("长期用户不存在")
            query = "SELECT * FROM memory_records WHERE user_id=? AND persona_id='mira'"
            if not historical:
                query += " AND status='active'"
            rows = db.execute(query + " ORDER BY created_at DESC LIMIT 1000", (user_id,)).fetchall()
            return owner["revision"], [MemoryRecord(**dict(row)) for row in rows]
        return await self.run(read)

    async def commit(self, user_id: str, operation_id: str, candidate: MemoryCandidate,
                     revision: int, session_id: str, turn_id: str, after_commit=None,
                     request_text: str | None = None) -> MemoryResult:
        """原子提交一次操作；更正保留旧版本，删除覆盖同一事实的全部版本。"""
        request_hash = token_digest(request_text if request_text is not None else candidate.model_dump_json())
        def apply(db):
            """核验归属和版本后修改事实，回执不保存正文或原始证据。"""
            previous = db.execute("SELECT result,request_hash FROM memory_operations WHERE user_id=? AND operation_id=?",
                                  (user_id, operation_id)).fetchone()
            if previous:
                if previous["request_hash"] != request_hash:
                    return MemoryResult(status="conflict", message="同一轮次标识对应了不同的记忆请求，请重新提交新轮次。")
                return MemoryResult.model_validate_json(previous["result"])
            owner = db.execute("SELECT revision FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
            if owner is None or owner["revision"] != revision:
                return MemoryResult(status="conflict", message="记忆已在其他请求中发生变化，请重新说明本次修改。")
            if candidate.action == "delete":
                result = self.delete_records(db, user_id, candidate)
            else:
                result = self.write_record(db, user_id, candidate, session_id, turn_id)
            if result.status in {"saved", "deleted"}:
                db.execute("UPDATE memory_users SET revision=revision+1 WHERE user_id=?", (user_id,))
            if result.status in {"saved", "unchanged", "deleted"}:
                db.execute("INSERT INTO memory_operations(user_id,operation_id,result,request_hash) VALUES (?,?,?,?)",
                           (user_id, operation_id, result.model_dump_json(), request_hash))
                db.execute("DELETE FROM memory_operations WHERE user_id=? AND rowid NOT IN "
                           "(SELECT rowid FROM memory_operations WHERE user_id=? ORDER BY rowid DESC LIMIT 256)",
                           (user_id, user_id))
            return result
        return await self.run(apply, after_commit)

    @staticmethod
    def write_record(db, user_id, candidate, session_id, turn_id):
        """去重并为明确更正建立新版本，含糊冲突保留原事实并要求澄清。"""
        old = db.execute("SELECT * FROM memory_records WHERE user_id=? AND persona_id='mira' "
                         "AND fact_key=? AND status='active'", (user_id, candidate.fact_key)).fetchone()
        if old and old["content"] == candidate.content:
            return MemoryResult(status="unchanged", message="这条信息已经保存。", memory_ids=[old["memory_id"]])
        if old and candidate.action != "update":
            return MemoryResult(status="clarify", message="这与已保存的信息不同。请明确说明要把哪条信息改成什么内容。")
        total = db.execute("SELECT count(*) FROM memory_records WHERE user_id=?", (user_id,)).fetchone()[0]
        if total >= 1000:
            return MemoryResult(status="failed", message="记忆存储已达到当前容量上限，请先删除不再需要的记录。")
        now = datetime.now(timezone.utc).isoformat()
        if old:
            db.execute("UPDATE memory_records SET status='inactive',valid_to=? WHERE memory_id=?", (now, old["memory_id"]))
        record = MemoryRecord(memory_id=str(uuid4()), user_id=user_id, kind=candidate.kind,
                              fact_key=candidate.fact_key, content=candidate.content, evidence=candidate.evidence,
                              source_session_id=session_id, source_turn_id=turn_id,
                              version=old["version"] + 1 if old else 1, status="active", created_at=now,
                              supersedes_id=old["memory_id"] if old else None)
        fields = record.model_dump()
        db.execute("INSERT INTO memory_records (" + ",".join(fields) + ") VALUES (" + ",".join("?" for _ in fields) + ")",
                   tuple(fields.values()))
        return MemoryResult(status="saved", message="已更新这条长期记忆。" if old else "已保存这条长期记忆。",
                            memory_ids=[record.memory_id])

    @staticmethod
    def delete_records(db, user_id, candidate):
        """按所有权解析目标，删除同一事实的历史版本并清除旧操作回执。"""
        if candidate.delete_all:
            selected = db.execute("SELECT memory_id FROM memory_records WHERE user_id=?", (user_id,)).fetchall()
            db.execute("DELETE FROM memory_records WHERE user_id=?", (user_id,))
        else:
            if not candidate.target_ids:
                return MemoryResult(status="clarify", message="请明确要忘掉哪条信息。")
            placeholders = ",".join("?" for _ in candidate.target_ids)
            rows = db.execute(f"SELECT fact_key FROM memory_records WHERE user_id=? AND memory_id IN ({placeholders})",
                              (user_id, *candidate.target_ids)).fetchall()
            if len(rows) != len(set(candidate.target_ids)):
                return MemoryResult(status="failed", message="未找到属于当前用户的完整删除目标。")
            keys = list({row["fact_key"] for row in rows})
            placeholders = ",".join("?" for _ in keys)
            selected = db.execute(f"SELECT memory_id FROM memory_records WHERE user_id=? AND fact_key IN ({placeholders})",
                                  (user_id, *keys)).fetchall()
            db.execute(f"DELETE FROM memory_records WHERE user_id=? AND fact_key IN ({placeholders})", (user_id, *keys))
        db.execute("DELETE FROM memory_operations WHERE user_id=?", (user_id,))
        return MemoryResult(status="deleted", message=f"已删除 {len(selected)} 条长期记忆记录（含历史版本），并清空相关会话上下文。",
                            deleted_count=len(selected), cleared=True)
