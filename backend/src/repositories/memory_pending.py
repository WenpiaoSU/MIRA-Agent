"""作用：在记忆主库事务中管理冲突候选的确认生命周期。

当前功能：
- 迁移旧候选，保存目标快照、版本、来源和删除代次。
- 去重待确认及已拒候选，返回有界、可恢复的确认列表。
- 原子采纳或保留旧事实，校验过期和并发变化并保存幂等回执。
"""

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from src.contracts.memory import MemoryCandidate, MemoryResult, PendingMemory


def candidate_hash(candidate, target_id, target_version):
    """同一目标版本和原子正文只产生一条候选，不受措辞外的来源包装影响。"""
    payload = [target_id, target_version, candidate.domain_scope, candidate.fact_key, candidate.content]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def setup_pending(db):
    """补充候选快照字段，旧目标 ID 指向不可变版本，可安全恢复原版本号。"""
    columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_pending)")}
    additions = {"target_version": "INTEGER", "target_content": "TEXT NOT NULL DEFAULT ''",
                 "epoch": "INTEGER NOT NULL DEFAULT 0", "status": "TEXT NOT NULL DEFAULT 'pending'",
                 "candidate_hash": "TEXT NOT NULL DEFAULT ''", "resolution_action": "TEXT",
                 "resolution": "TEXT", "resolved_at": "TEXT", "confirmation_evidence": "TEXT",
                 "source_session_id": "TEXT NOT NULL DEFAULT ''", "source_turn_id": "TEXT NOT NULL DEFAULT ''"}
    for name, declaration in additions.items():
        if name not in columns:
            db.execute(f"ALTER TABLE memory_pending ADD COLUMN {name} {declaration}")
    for row in db.execute("SELECT * FROM memory_pending WHERE candidate_hash=''").fetchall():
        target = db.execute("SELECT * FROM memory_records WHERE user_id=? AND memory_id=?",
                            (row["user_id"], row["target_id"])).fetchone()
        owner = db.execute("SELECT epoch FROM memory_users WHERE user_id=?", (row["user_id"],)).fetchone()
        try:
            candidate = MemoryCandidate.model_validate_json(row["candidate"])
            digest = candidate_hash(candidate, row["target_id"], target["version"] if target else None)
        except ValidationError:
            db.execute("UPDATE memory_pending SET status='stale',candidate_hash=? WHERE pending_id=?", (str(uuid4()), row["pending_id"]))
            continue
        duplicate = db.execute("SELECT pending_id FROM memory_pending WHERE user_id=? AND candidate_hash=?",
                               (row["user_id"], digest)).fetchone()
        if duplicate:
            db.execute("DELETE FROM memory_pending WHERE pending_id=?", (row["pending_id"],))
            continue
        db.execute("UPDATE memory_pending SET target_version=?,target_content=?,epoch=?,candidate_hash=?,status=? WHERE pending_id=?",
                   (target["version"] if target else None, target["content"] if target else "", owner["epoch"] if owner else 0,
                    digest, "pending" if target and target["status"] == "active" else "stale", row["pending_id"]))
        if row["operation_id"].count(":") >= 2:
            original_operation = row["operation_id"].rsplit(":", 1)[0]
            source_session, source_turn = original_operation.split(":", 1)
            db.execute("UPDATE memory_pending SET source_session_id=?,source_turn_id=? WHERE pending_id=?",
                       (source_session, source_turn, row["pending_id"]))
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS memory_pending_candidate ON memory_pending(user_id,candidate_hash) WHERE candidate_hash<>''")


def store_pending(db, user_id, operation_id, decision, session_id, turn_id, epoch, ttl):
    """存入已验证候选及原目标快照；同一被拒候选在保留期内不重复打扰。"""
    now = datetime.now(timezone.utc)
    db.execute("DELETE FROM memory_pending WHERE expires_at<=?", (now.timestamp(),))
    target = db.execute("SELECT * FROM memory_records WHERE user_id=? AND memory_id=? AND status='active'",
                        (user_id, decision.target_id)).fetchone()
    version = target["version"] if target else None
    candidate = decision.candidate
    digest = candidate_hash(candidate, decision.target_id, version)
    previous = db.execute("SELECT pending_id,status FROM memory_pending WHERE user_id=? AND candidate_hash=?", (user_id, digest)).fetchone()
    if previous:
        return previous["pending_id"], previous["status"] == "pending"
    if db.execute("SELECT count(*) FROM memory_pending WHERE user_id=? AND status='pending'", (user_id,)).fetchone()[0] >= 1000:
        raise ValueError("待确认候选已达到容量上限")
    expires = now.timestamp() + ttl
    if candidate.expires_at:
        expires = min(expires, datetime.fromisoformat(candidate.expires_at).timestamp())
    pending_id = str(uuid4())
    db.execute("INSERT INTO memory_pending(pending_id,user_id,operation_id,candidate,target_id,reason,created_at,expires_at,"
               "target_version,target_content,epoch,candidate_hash,source_session_id,source_turn_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (pending_id, user_id, operation_id, candidate.model_dump_json(), decision.target_id, decision.reason,
                now.isoformat(), expires, version, target["content"] if target else "", epoch, digest, session_id, turn_id))
    return pending_id, True


def list_pending(db, user_id):
    """过期或旧目标已变化的候选关闭，不作为事实或继续可采纳的提示返回。"""
    now = datetime.now(timezone.utc).timestamp()
    db.execute("DELETE FROM memory_pending WHERE expires_at<=?", (now,))
    db.execute("UPDATE memory_pending SET status='stale' WHERE user_id=? AND status='pending' AND "
               "(epoch<>(SELECT epoch FROM memory_users WHERE user_id=?) OR NOT EXISTS "
               "(SELECT 1 FROM memory_records r WHERE r.user_id=? AND r.memory_id=memory_pending.target_id "
               "AND r.version=memory_pending.target_version AND r.status='active' "
               "AND (r.expires_at IS NULL OR julianday(r.expires_at)>julianday('now'))))", (user_id, user_id, user_id))
    records = []
    for row in db.execute("SELECT * FROM memory_pending WHERE user_id=? AND status='pending' ORDER BY created_at DESC LIMIT 100", (user_id,)):
        candidate = MemoryCandidate.model_validate_json(row["candidate"])
        question = f"原记录是“{row['target_content']}”；新信息是“{candidate.content}”。是否使用新信息替换原记录？"
        records.append(PendingMemory(pending_id=row["pending_id"], candidate=candidate, target_id=row["target_id"],
            target_version=row["target_version"], target_content=row["target_content"], reason=row["reason"],
            question=question, created_at=row["created_at"], expires_at=row["expires_at"]).model_dump())
    return records


def finish_resolution(db, row, action, status, result, evidence):
    """保存真实回执并限制关闭记录数量，重放不能再次增版本或改变原决定。"""
    db.execute("UPDATE memory_pending SET status=?,resolution_action=?,resolution=?,resolved_at=?,confirmation_evidence=? WHERE pending_id=?",
               (status, action, result.model_dump_json(), datetime.now(timezone.utc).isoformat(), evidence[:1000], row["pending_id"]))
    db.execute("DELETE FROM memory_pending WHERE user_id=? AND status<>'pending' AND pending_id NOT IN "
               "(SELECT pending_id FROM memory_pending WHERE user_id=? AND status<>'pending' ORDER BY resolved_at DESC LIMIT 256)",
               (row["user_id"], row["user_id"]))
    return result


def resolve_pending(db, user_id, pending_id, request, write_record, evidence):
    """在调用方的 SQLite 写事务内确认候选，过期或并发变更不覆盖当前事实。"""
    row = db.execute("SELECT * FROM memory_pending WHERE user_id=? AND pending_id=?", (user_id, pending_id)).fetchone()
    if not row:
        return MemoryResult(status="not_found", message="未找到属于当前用户的待确认记忆，请刷新列表。")
    if request.target_version != row["target_version"]:
        return MemoryResult(status="conflict", message="确认所对应的旧事实版本不匹配，请刷新列表。")
    if row["expires_at"] <= datetime.now(timezone.utc).timestamp():
        return finish_resolution(db, row, request.action, "expired", MemoryResult(status="expired", message="该候选已过期，请重新说明需要记住的信息。"), evidence)
    if row["resolution"]:
        if row["resolution_action"] != request.action:
            return MemoryResult(status="conflict", message="该候选已经以另一项决定处理，请刷新列表。")
        return MemoryResult.model_validate_json(row["resolution"]).model_copy(update={"changed": False})
    if request.action == "keep_old":
        return finish_resolution(db, row, request.action, "kept_old",
            MemoryResult(status="unchanged", message="已保留当前记忆，不采用这条新信息。", decisions=["SKIP"]), evidence)
    owner = db.execute("SELECT epoch FROM memory_users WHERE user_id=?", (user_id,)).fetchone()
    target = db.execute("SELECT * FROM memory_records WHERE user_id=? AND memory_id=? AND status='active'", (user_id, row["target_id"])).fetchone()
    candidate = MemoryCandidate.model_validate_json(row["candidate"])
    if candidate.expires_at and datetime.fromisoformat(candidate.expires_at) <= datetime.now(timezone.utc):
        return finish_resolution(db, row, request.action, "expired",
            MemoryResult(status="expired", message="新信息已超过有效期，未替换原记忆。"), evidence)
    if (row["status"] != "pending" or not owner or owner["epoch"] != row["epoch"] or not target
            or target["version"] != row["target_version"] or target["domain_scope"] != candidate.domain_scope
            or target["kind"] != candidate.kind
            or (target["expires_at"] and datetime.fromisoformat(target["expires_at"]) <= datetime.now(timezone.utc))):
        return finish_resolution(db, row, request.action, "stale",
            MemoryResult(status="conflict", message="原记忆已变化或目标不再适用，本次未覆盖；请刷新后完整说明要更正的信息。"), evidence)
    candidate = candidate.model_copy(update={"action": "update", "fact_key": target["fact_key"]})
    result = write_record(db, user_id, candidate, row["source_session_id"], row["source_turn_id"])
    if result.status not in {"saved", "unchanged"}:
        return result
    changed = result.status == "saved"
    if changed:
        db.execute("UPDATE memory_users SET revision=revision+1 WHERE user_id=?", (user_id,))
    result = result.model_copy(update={"message": "已采用新信息并更新长期记忆，旧版本已保留。", "decisions": ["UPDATE"], "changed": changed})
    return finish_resolution(db, row, request.action, "accepted", result, evidence)
