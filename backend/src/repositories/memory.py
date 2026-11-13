"""作用：以 SQLite 事务保存长期记忆和浏览器身份。

当前功能：
- 使用散列后的随机凭据关联长期用户，数据库不保存原始凭据。
- 提供按用户隔离的查询、版本更正、物理删除和幂等回执。
- 在线程中执行短事务；版本及删除代次阻止迟到提取恢复已删除数据。
- 保存有界整合任务与待确认候选，提供批量原子整合、恢复、重试和过期清理。
- 委托候选存储处理版本绑定、去重及幂等确认，关闭已采纳或拒绝的冲突。
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

from src.contracts.memory import MemoryCandidate, MemoryDecision, MemoryRecord, MemoryResult
from src.repositories.memory_pending import setup_pending, store_pending, list_pending, resolve_pending


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
                CREATE INDEX IF NOT EXISTS memory_owner ON memory_records(user_id, status);
                CREATE TABLE IF NOT EXISTS memory_operations (
                    user_id TEXT NOT NULL, operation_id TEXT NOT NULL, result TEXT NOT NULL, request_hash TEXT NOT NULL,
                    PRIMARY KEY(user_id, operation_id));
                CREATE TABLE IF NOT EXISTS memory_pending (
                    pending_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    candidate TEXT NOT NULL, target_id TEXT, reason TEXT NOT NULL,
                    created_at TEXT NOT NULL, expires_at REAL NOT NULL,
                    UNIQUE(user_id, operation_id));
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload TEXT NOT NULL,
                    epoch INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    attempts INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL,
                    expires_at REAL NOT NULL, UNIQUE(user_id, operation_id));
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_operations)")}
            if "request_hash" not in columns:
                db.execute("ALTER TABLE memory_operations ADD COLUMN request_hash TEXT NOT NULL DEFAULT ''")
            record_columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_records)")}
            for name, declaration in {"memory_type": "TEXT NOT NULL DEFAULT 'fact'",
                                      "domain_scope": "TEXT NOT NULL DEFAULT 'daily'", "expires_at": "TEXT",
                                      "confidence": "REAL NOT NULL DEFAULT 1.0",
                                      "source_kind": "TEXT NOT NULL DEFAULT 'user'",
                                      "source_ref": "TEXT NOT NULL DEFAULT 'user'"}.items():
                if name not in record_columns:
                    db.execute(f"ALTER TABLE memory_records ADD COLUMN {name} {declaration}")
            user_columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_users)")}
            if "epoch" not in user_columns:
                db.execute("ALTER TABLE memory_users ADD COLUMN epoch INTEGER NOT NULL DEFAULT 0")
            db.execute("DROP INDEX IF EXISTS memory_active_key")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS memory_active_scope_key ON memory_records "
                       "(user_id,persona_id,domain_scope,fact_key) WHERE status='active'")
            setup_pending(db)
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
                query += " AND status='active' AND (expires_at IS NULL OR julianday(expires_at)>julianday('now'))"
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
                         "AND domain_scope=? AND fact_key=? AND status='active'",
                         (user_id, candidate.domain_scope, candidate.fact_key)).fetchone()
        expired = bool(old and old["expires_at"] and datetime.fromisoformat(old["expires_at"]) <= datetime.now(timezone.utc))
        if (old and old["content"] == candidate.content and not expired
                and (candidate.kind != "episodic" or old["evidence"] == candidate.evidence)):
            return MemoryResult(status="unchanged", message="这条信息已经保存。", memory_ids=[old["memory_id"]])
        if old and not expired and candidate.action != "update":
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
                              supersedes_id=old["memory_id"] if old else None,
                              memory_type=candidate.memory_type, domain_scope=candidate.domain_scope,
                              expires_at=candidate.expires_at, confidence=candidate.confidence,
                              source_kind=candidate.source_kind, source_ref=candidate.source_ref)
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
            rows = db.execute(f"SELECT fact_key,domain_scope FROM memory_records WHERE user_id=? AND memory_id IN ({placeholders})",
                              (user_id, *candidate.target_ids)).fetchall()
            if len(rows) != len(set(candidate.target_ids)):
                return MemoryResult(status="failed", message="未找到属于当前用户的完整删除目标。")
            selected = []
            for scope, key in {(row["domain_scope"], row["fact_key"]) for row in rows}:
                selected.extend(db.execute("SELECT memory_id FROM memory_records WHERE user_id=? AND domain_scope=? AND fact_key=?",
                                           (user_id, scope, key)).fetchall())
                db.execute("DELETE FROM memory_records WHERE user_id=? AND domain_scope=? AND fact_key=?", (user_id, scope, key))
        # 删除撤销该用户的全部未完成整合，运行中的模型结果也必须再次核对 epoch。
        db.execute("UPDATE memory_users SET epoch=epoch+1 WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM memory_jobs WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM memory_pending WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM memory_operations WHERE user_id=?", (user_id,))
        return MemoryResult(status="deleted", message=f"已删除 {len(selected)} 条长期记忆记录（含历史版本），并清空相关会话上下文。",
                            deleted_count=len(selected), cleared=True)

    async def epoch(self, user_id):
        """读取删除代次，用于约束排队任务和正在运行的语义提取。"""
        def read(db):
            """不存在的用户不能创建或恢复整合任务。"""
            row = db.execute("SELECT epoch FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
            if not row:
                raise ValueError("长期用户不存在")
            return row["epoch"]
        return await self.run(read)

    async def integrate(self, user_id, operation_id, decisions: list[MemoryDecision], revision,
                        epoch, session_id, turn_id, request_text, pending_ttl=604800):
        """在一个事务内整合多个事实；相似度结果不能直接进入写入事务。"""
        request_hash = token_digest(request_text)

        def apply(db):
            """核对幂等、版本与删除代次，任何失败回滚整组操作。"""
            owner = db.execute("SELECT revision,epoch FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
            if not owner or owner["epoch"] != epoch:
                return MemoryResult(status="conflict", message="记忆删除后旧候选已失效，请重新提供需要保存的信息。")
            previous = db.execute("SELECT result,request_hash FROM memory_operations WHERE user_id=? AND operation_id=?",
                                  (user_id, operation_id)).fetchone()
            if previous:
                if previous["request_hash"] != request_hash:
                    return MemoryResult(status="conflict", message="同一轮次对应不同的记忆内容，请重新提交。")
                return MemoryResult.model_validate_json(previous["result"])
            if owner["revision"] != revision:
                return MemoryResult(status="conflict", message="记忆发生并发变化，本次未覆盖，请重新说明。")
            ids, actions, pending_ids = [], [], []
            db.execute("SAVEPOINT memory_batch")
            for index, decision in enumerate(decisions):
                candidate = decision.candidate
                actions.append(decision.action)
                if decision.action == "PENDING":
                    try:
                        pending_id, waiting = store_pending(db, user_id, operation_id + f":{index}", decision,
                                                            session_id, turn_id, epoch, pending_ttl)
                    except ValueError:
                        db.execute("ROLLBACK TO memory_batch")
                        return MemoryResult(status="failed", message="待确认候选已达到容量上限，请先清理。")
                    if waiting:
                        pending_ids.append(pending_id)
                    else:
                        actions[-1] = "SKIP"
                elif decision.action == "SKIP":
                    if decision.target_id:
                        ids.append(decision.target_id)
                else:
                    if decision.action == "UPDATE":
                        target = db.execute("SELECT fact_key,domain_scope FROM memory_records WHERE user_id=? AND memory_id=? AND status='active'",
                                            (user_id, decision.target_id)).fetchone()
                        if not target or target["domain_scope"] != candidate.domain_scope:
                            db.execute("ROLLBACK TO memory_batch")
                            return MemoryResult(status="conflict", message="更正目标已失效，本次未写入。")
                        candidate = candidate.model_copy(update={"action": "update", "fact_key": target["fact_key"]})
                    else:
                        candidate = candidate.model_copy(update={"action": "remember"})
                    result = self.write_record(db, user_id, candidate, session_id, turn_id)
                    if result.status not in {"saved", "unchanged"}:
                        db.execute("ROLLBACK TO memory_batch")
                        return result
                    ids.extend(result.memory_ids)
                    if decision.action == "UPDATE":
                        db.execute("UPDATE memory_pending SET status='stale' WHERE user_id=? AND target_id=? AND status='pending'",
                                   (user_id, decision.target_id))
            changed = any(action in {"ADD", "UPDATE"} for action in actions)
            pending = "PENDING" in actions
            status = "pending" if pending else "saved" if changed else "unchanged"
            message = ("存在冲突，原事实保持有效；请在待确认记忆中选择使用新信息或保留原记录。" if pending else
                       "已保存或更新这些长期记忆。" if changed else "没有需要新增的长期记忆。")
            result = MemoryResult(status=status, message=message, memory_ids=ids, decisions=actions, pending_ids=pending_ids)
            if changed:
                db.execute("UPDATE memory_users SET revision=revision+1 WHERE user_id=?", (user_id,))
            db.execute("INSERT INTO memory_operations VALUES (?,?,?,?)", (user_id, operation_id, result.model_dump_json(), request_hash))
            db.execute("DELETE FROM memory_operations WHERE user_id=? AND rowid NOT IN "
                       "(SELECT rowid FROM memory_operations WHERE user_id=? ORDER BY rowid DESC LIMIT 256)", (user_id, user_id))
            return result
        return await self.run(apply)

    async def pending(self, user_id):
        """只向本人返回尚未过期的候选，不混入有效事实快照。"""
        def read(db):
            """清理过期候选后返回有界列表。"""
            return list_pending(db, user_id)
        return await self.run(read)

    async def resolve_pending(self, user_id, pending_id, request, *, evidence="", after_commit=None):
        """原子确认一个目标版本绑定的候选，只有真实变化才触发派生缓存失效。"""
        def apply(db):
            """复用主库连接和事实版本写入，确认过程不调用模型。"""
            return resolve_pending(db, user_id, pending_id, request, self.write_record, evidence)
        return await self.run(apply, after_commit)

    async def delete_pending(self, user_id, pending_id):
        """按所有权删除一个候选，并撤销可能重新产生该候选的旧后台任务。"""
        def remove(db):
            """目标不存在时不改变其他用户或当前用户的有效事实。"""
            changed = db.execute("DELETE FROM memory_pending WHERE user_id=? AND pending_id=?", (user_id, pending_id)).rowcount
            if changed:
                db.execute("UPDATE memory_users SET epoch=epoch+1,revision=revision+1 WHERE user_id=?", (user_id,))
                db.execute("UPDATE memory_pending SET epoch=epoch+1 WHERE user_id=? AND status='pending'", (user_id,))
                db.execute("DELETE FROM memory_jobs WHERE user_id=?", (user_id,))
                db.execute("DELETE FROM memory_operations WHERE user_id=?", (user_id,))
            return bool(changed)
        return await self.run(remove)

    async def enqueue(self, user_id, operation_id, payload, ttl_seconds=604800, limit=256):
        """持久化有界整合任务；入队回执不表示事实已保存。"""
        def add(db):
            """按用户和轮次去重，记录原始删除代次，拒绝不同正文复用 ID。"""
            now = datetime.now(timezone.utc).timestamp()
            db.execute("DELETE FROM memory_jobs WHERE expires_at<=?", (now,))
            previous = db.execute("SELECT request_hash FROM memory_jobs WHERE user_id=? AND operation_id=?", (user_id, operation_id)).fetchone()
            digest = token_digest(payload["text"])
            if previous:
                return previous["request_hash"] == digest
            owner = db.execute("SELECT epoch FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
            if not owner or db.execute("SELECT count(*) FROM memory_jobs").fetchone()[0] >= limit:
                return False
            db.execute("INSERT INTO memory_jobs(user_id,operation_id,request_hash,payload,epoch,available_at,expires_at) VALUES (?,?,?,?,?,?,?)",
                       (user_id, operation_id, digest, json.dumps(payload, ensure_ascii=False), owner["epoch"], now, now + ttl_seconds))
            return True
        return await self.run(add)

    async def claim_job(self, lease_seconds=120):
        """领取一个任务；进程崩溃留下的运行任务在租约到期后可恢复。"""
        def claim(db):
            """原子领取并增加尝试次数，过期和超过三次的任务直接清理。"""
            now = datetime.now(timezone.utc).timestamp()
            db.execute("DELETE FROM memory_jobs WHERE expires_at<=? OR (attempts>=3 AND available_at<=?)", (now, now))
            db.execute("DELETE FROM memory_pending WHERE expires_at<=?", (now,))
            row = db.execute("SELECT * FROM memory_jobs WHERE available_at<=? ORDER BY job_id LIMIT 1", (now,)).fetchone()
            if not row:
                return None
            db.execute("UPDATE memory_jobs SET status='running',attempts=attempts+1,available_at=? WHERE job_id=?", (now + lease_seconds, row["job_id"]))
            return {**dict(row), "payload": json.loads(row["payload"])}
        return await self.run(claim)

    async def finish_job(self, job_id, *, retry=False):
        """完成任务即清除原文；暂时失败以有限退避重试。"""
        def finish(db):
            """删除结果任务或恢复待领取状态，最多尝试三次。"""
            if retry:
                now = datetime.now(timezone.utc).timestamp()
                db.execute("UPDATE memory_jobs SET status='queued',available_at=? WHERE job_id=?", (now + 5, job_id))
            else:
                db.execute("DELETE FROM memory_jobs WHERE job_id=?", (job_id,))
        return await self.run(finish)
