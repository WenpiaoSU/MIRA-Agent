"""作用：将校验后的异步事件归约为 M08 社会状态。

当前功能：
- 按来源去重、按采集时间防倒退，并验证身份、连接和轮次归属。
- 对视觉与用户情感执行配置化 TTL，记录来源、缺失和降级原因。
- 深拷贝版本快照并生成预算内的 Agent 上下文，不调用模型或数据库。
- 清理重连、身份切换、记忆删除与会话关闭产生的派生状态。
"""

import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.contracts.state import (
    DialogueState, HealthState, InteractionState, Provenance, RelationshipState,
    SocialContext, SocialEvent, SocialState, StateDelta, VisualState,
)


def utc_now() -> datetime:
    """使用带时区的 UTC 时钟，浏览器采集时钟须先由适配器映射。"""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SocialStateConfig:
    """限定感知有效期、迟到窗口、上下文字符数与去重容量。"""
    visual_ttl_ms: int = 3000
    affect_ttl_ms: int = 15000
    max_lateness_ms: int = 5000
    context_max_chars: int = 6000
    dedup_capacity: int = 512

    def __post_init__(self):
        """拒绝无效预算，保证最小上下文仍能包含归属与降级信息。"""
        if min(self.visual_ttl_ms, self.affect_ttl_ms, self.max_lateness_ms) < 0:
            raise ValueError("状态时间预算不能为负数")
        if self.context_max_chars < 1024 or self.dedup_capacity < 1:
            raise ValueError("状态上下文至少 1024 字符，去重容量至少为 1")


@dataclass
class StateEntry:
    """保存当前快照与有界来源序号，不保留完整事件流。"""
    state: SocialState
    seen: OrderedDict[str, None] = field(default_factory=OrderedDict)
    sequences: dict[str, int] = field(default_factory=dict)


class SocialStateService:
    """在事件循环内同步归约，实时停止无需等待锁或外部服务。"""

    def __init__(self, config: SocialStateConfig | None = None, clock: Callable[[], datetime] = utc_now):
        """注入确定性配置与可回放时钟。"""
        self.config = config or SocialStateConfig()
        self.clock = clock
        self.entries: dict[str, StateEntry] = {}

    def bind(self, session_id: str, user_id: str | None, connection_epoch: int) -> None:
        """由可信会话层绑定身份；身份或 epoch 改变时清空旧感知与派生数据。"""
        entry = self.entries.get(session_id)
        if entry and (entry.state.user_id, entry.state.connection_epoch) == (user_id, connection_epoch):
            return
        now = self.clock()
        self.entries[session_id] = StateEntry(SocialState(
            session_id=session_id, user_id=user_id, connection_epoch=connection_epoch,
            version=entry.state.version + 1 if entry else 0, as_of=now, as_of_ms=int(now.timestamp() * 1000),
            interaction=InteractionState(mode="idle", user_speaking=False, robot_playing=False),
            relationship=RelationshipState(user_id=user_id),
            staleness={"visual": "missing", "user_affect": "missing"}))

    def close(self, session_id: str) -> None:
        """释放已关闭会话的状态与去重缓存。"""
        self.entries.pop(session_id, None)

    def emit(self, session_id: str, kind: str, payload=None, *, source: str = "gateway",
             turn_id: str | None = None, generation_id: str | None = None) -> StateDelta:
        """为进程内适配器生成信封；外部模块可构造 SocialEvent 调用 apply。"""
        entry = self.entries[session_id]
        seq = entry.sequences.get(source, -1) + 1
        now = self.clock()
        return self.apply(SocialEvent(event_id=str(uuid4()), type=kind, session_id=session_id,
                                    user_id=entry.state.user_id, connection_epoch=entry.state.connection_epoch,
                                    source=source, seq=seq, captured_at=now, received_at=now,
                                    turn_id=turn_id, generation_id=generation_id, payload=payload))

    def apply(self, event: SocialEvent) -> StateDelta:
        """验证归属、时间和顺序后归约；迟到事件不回写先前返回的快照。"""
        entry = self.entries.get(event.session_id)
        if entry is None:
            return StateDelta(session_id=event.session_id, version=0, accepted=False, reason="unknown_session")
        state = entry.state
        reason = self.rejection(entry, event)
        if reason:
            return StateDelta(session_id=event.session_id, version=state.version, accepted=False, reason=reason)
        entry.sequences[event.source] = event.seq
        entry.seen[event.event_id] = None
        while len(entry.seen) > self.config.dedup_capacity:
            entry.seen.popitem(last=False)
        if event.type == "clock.tick":
            changed = self.expire(entry, self.clock())
            return StateDelta(session_id=event.session_id, version=entry.state.version, accepted=True,
                              changed_domains=changed)
        domain = {"turn.updated": "interaction", "utterance.committed": "utterance",
                  "affect.updated": "user_affect", "robot_affect.confirmed": "robot_affect",
                  "action.updated": "actions", "memory.cleared": "memory"}.get(event.type, event.type.split(".")[0])
        if domain == "health":
            domain = "health:" + event.payload.component
        previous = state.provenance.get(domain)
        if previous and event.captured_at < previous.captured_at:
            return StateDelta(session_id=event.session_id, version=state.version, accepted=False, reason="older_observation")
        updates = self.reduce(state, event, domain)
        if domain in {"visual", "user_affect"}:
            updates["health"] = {**state.health, domain: HealthState(
                component=domain, status="ok", updated_at=event.captured_at)}
        ttl = {"visual": self.config.visual_ttl_ms, "user_affect": self.config.affect_ttl_ms}.get(domain)
        expires_at = event.captured_at + timedelta(milliseconds=ttl) if ttl is not None else None
        if domain == "user_affect" and event.payload.expires_at is not None:
            expires_at = min(expires_at, event.payload.expires_at)
        provenance = dict(updates.pop("provenance", state.provenance))
        provenance[domain] = Provenance(event_id=event.event_id, source=event.source, seq=event.seq,
                                        captured_at=event.captured_at, received_at=event.received_at,
                                        expires_at=expires_at,
                                        turn_id=event.turn_id, generation_id=event.generation_id)
        staleness = dict(updates.pop("staleness", state.staleness))
        staleness[domain] = "fresh"
        # model_copy 的 update 值不会被 deep 参数复制，因此合并后再隔离整个对象。
        entry.state = state.model_copy(update={**updates, "provenance": provenance, "staleness": staleness,
                                               "version": state.version + 1}).model_copy(deep=True)
        expired = self.expire(entry, self.clock())
        return StateDelta(session_id=event.session_id, version=entry.state.version, accepted=True,
                          changed_domains=tuple(dict.fromkeys((domain, *updates, *expired))))

    def rejection(self, entry: StateEntry, event: SocialEvent) -> str | None:
        """拒绝跨身份、旧连接、旧轮次、过晚或未来的观测。"""
        state = entry.state
        if (event.user_id, event.connection_epoch) != (state.user_id, state.connection_epoch):
            return "identity_or_epoch_mismatch"
        if event.event_id in entry.seen or event.seq <= entry.sequences.get(event.source, -1):
            return "duplicate_or_out_of_order"
        if event.source not in entry.sequences and len(entry.sequences) >= 64:
            return "source_limit"
        if event.type == "health.updated" and event.payload.component not in state.health and len(state.health) >= 32:
            return "health_component_limit"
        now = self.clock()
        if event.captured_at > event.received_at or event.received_at > now:
            return "future_observation"
        memory_boundary = state.provenance.get("memory")
        if (memory_boundary and event.type in {"utterance.committed", "intent.updated", "affect.updated",
                                              "dialogue.updated", "relationship.updated", "action.updated"}
                and event.captured_at < memory_boundary.captured_at):
            return "before_memory_boundary"
        if event.type in {"visual.updated", "affect.updated"} and (
                now - event.captured_at).total_seconds() * 1000 > self.config.max_lateness_ms:
            return "too_late"
        if event.type in {"visual.updated", "affect.updated"}:
            domain = "visual" if event.type == "visual.updated" else "user_affect"
            health = state.health.get(domain)
            health_evidence = state.provenance.get("health:" + domain)
            if health and health.status == "unavailable":
                return "modality_unavailable"
            if health_evidence and event.captured_at < health_evidence.captured_at:
                return "before_health_boundary"
        if event.type in {"utterance.committed", "intent.updated", "affect.updated", "robot_affect.confirmed", "action.updated"}:
            interaction = state.interaction
            if (not event.turn_id or not event.generation_id or event.turn_id != interaction.turn_id or
                    event.generation_id != interaction.generation_id):
                return "turn_or_generation_mismatch"
            if event.type in {"intent.updated", "utterance.committed"} and interaction.mode in {"recovering", "disconnected"}:
                return "inactive_generation"
            if interaction.mode in {"recovering", "disconnected"} and (
                    event.type == "robot_affect.confirmed" or
                    event.type == "action.updated" and event.payload.status in {"pending", "running"}):
                return "inactive_generation"
        if event.type == "relationship.updated" and event.payload.user_id != state.user_id:
            return "relationship_owner_mismatch"
        if event.type == "relationship.updated" and event.payload.version <= state.relationship.version:
            return "old_relationship_version"
        if event.type == "robot_affect.confirmed" and event.source != "expression":
            return "untrusted_expression"
        if event.type == "action.updated" and (event.payload.turn_id, event.payload.generation_id) != (
                event.turn_id, event.generation_id):
            return "action_owner_mismatch"
        return None

    @staticmethod
    def reduce(state: SocialState, event: SocialEvent, domain: str) -> dict:
        """按域归约，保持情感与表达以及计划与实际执行事实独立。"""
        payload = event.payload
        if domain == "interaction":
            updates = {"interaction": payload}
            if (payload.turn_id, payload.generation_id) != (state.interaction.turn_id, state.interaction.generation_id):
                cleared = {"utterance", "intent", "user_affect", "actions"}
                updates.update(utterance=None, committed_text=None, intent=None, user_affect=None,
                               pending_actions=(), provenance={key: value for key, value in state.provenance.items()
                                                              if key not in cleared},
                               staleness={**state.staleness, **{key: (
                                   "unavailable" if key in state.health and state.health[key].status == "unavailable"
                                   else "missing") for key in cleared}})
            if payload.mode in {"recovering", "disconnected"}:
                updates["pending_actions"] = ()
                if state.last_action and state.last_action.status in {"pending", "running"}:
                    updates["last_action"] = state.last_action.model_copy(update={"status": "cancelled", "verified": False})
            return updates
        if domain == "utterance":
            return {"utterance": payload, "committed_text": payload.text}
        if domain.startswith("health:"):
            updates = {"health": {**state.health, payload.component: payload.model_copy(
                update={"updated_at": event.captured_at})}}
            if payload.component in {"visual", "user_affect"} and payload.status == "unavailable":
                updates[payload.component] = VisualState() if payload.component == "visual" else None
                updates["staleness"] = {**state.staleness, payload.component: "unavailable"}
            elif payload.component in {"visual", "user_affect"} and state.staleness.get(payload.component) == "unavailable":
                updates["staleness"] = {**state.staleness, payload.component: "missing"}
            return updates
        if domain == "actions":
            pending = [action for action in state.pending_actions if action.action_id != payload.action_id]
            if payload.status in {"pending", "running"}:
                pending.append(payload)
            component = "tools" if payload.kind == "tool" else "playback"
            return {"pending_actions": tuple(pending[-24:]), "last_action": payload,
                    "health": {**state.health, component: HealthState(
                        component=component, status="degraded" if payload.status in {"error", "unknown"} else "ok",
                        reason="execution_failed" if payload.status in {"error", "unknown"} else None,
                        updated_at=event.captured_at)}}
        if domain == "memory":
            cleared = {"utterance", "intent", "user_affect", "actions", "dialogue", "relationship"}
            return {"dialogue": DialogueState(), "relationship": RelationshipState(user_id=state.user_id),
                    "utterance": None, "committed_text": None, "intent": None, "user_affect": None,
                    "pending_actions": (), "last_action": None,
                    "provenance": {key: value for key, value in state.provenance.items() if key not in cleared},
                    "staleness": {**state.staleness, **{key: "missing" for key in cleared}}}
        return {domain: payload}

    def expire(self, entry: StateEntry, now: datetime) -> tuple[str, ...]:
        """读取或时钟事件清除过期感知，只有有效状态改变才递增版本。"""
        state = entry.state
        updates, staleness, changed = {}, dict(state.staleness), []
        health = dict(state.health)
        for domain in ("visual", "user_affect"):
            evidence = state.provenance.get(domain)
            if (evidence and evidence.expires_at is not None and now >= evidence.expires_at
                    and staleness.get(domain) == "fresh"):
                updates[domain] = VisualState() if domain == "visual" else None
                staleness[domain] = "expired"
                health[domain] = HealthState(component=domain, status="degraded", reason="ttl_expired", updated_at=now)
                changed.append(domain)
        if changed:
            entry.state = state.model_copy(update={**updates, "staleness": staleness, "health": health,
                                                   "version": state.version + 1}, deep=True)
        return tuple(changed)

    def snapshot(self, session_id: str) -> SocialState:
        """返回与内部状态隔离的快照，读取时执行 TTL。"""
        entry = self.entries[session_id]
        now = self.clock()
        self.expire(entry, now)
        return entry.state.model_copy(update={"as_of": now, "as_of_ms": int(now.timestamp() * 1000)}, deep=True)

    def to_agent_context(self, snapshot: SocialState) -> SocialContext:
        """在完整 JSON 字符预算内提供可靠观测、执行状态和证据引用。"""
        data = {"as_of": snapshot.as_of.isoformat(),
                "interaction": snapshot.interaction.model_dump(mode="json"),
                "final_text": (snapshot.committed_text or "")[:2000],
                "intent": snapshot.intent.model_dump(mode="json") if snapshot.intent else None,
                "user_affect": snapshot.user_affect.model_dump(mode="json") if snapshot.user_affect else None,
                "visual": snapshot.visual.model_dump(mode="json"),
                "dialogue": snapshot.dialogue.model_dump(mode="json"),
                "relationship": snapshot.relationship.model_dump(mode="json"),
                "robot_affect": snapshot.robot_affect.model_dump(mode="json"),
                "pending_actions": [action.model_dump(mode="json") for action in snapshot.pending_actions[-6:]],
                "last_action": snapshot.last_action.model_dump(mode="json") if snapshot.last_action else None,
                "health": {key: value.model_dump(mode="json") for key, value in snapshot.health.items()},
                "staleness": dict(snapshot.staleness),
                "observations": {key: {"source": item.source, "captured_at": item.captured_at.isoformat(),
                                       "event_id": item.event_id}
                                 for key, item in snapshot.provenance.items()
                                 if key in {"utterance", "intent", "user_affect", "visual"}
                                 and snapshot.staleness.get(key) == "fresh"}}
        evidence = tuple(item.event_id for key, item in snapshot.provenance.items()
                         if snapshot.staleness.get(key) == "fresh")[-16:]
        missing = tuple(key for key in ("visual", "user_affect") if snapshot.staleness.get(key) != "fresh")
        context = SocialContext(session_id=snapshot.session_id, user_id=snapshot.user_id, version=snapshot.version,
                                data=data, evidence_event_ids=evidence, missing_modalities=missing,
                                truncated=len(snapshot.committed_text or "") > 2000)
        # 按优先级移除可选摘要，不截断 JSON；完整最终输入仍作为独立用户消息传入。
        for key in ("health", "observations", "intent", "dialogue", "pending_actions", "last_action", "relationship",
                    "robot_affect", "final_text", "visual", "user_affect"):
            if len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False)) <= self.config.context_max_chars:
                break
            data.pop(key, None)
            context = context.model_copy(update={"data": dict(data), "evidence_event_ids": (), "truncated": True})
        if len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False)) > self.config.context_max_chars:
            context = context.model_copy(update={"data": {"mode": snapshot.interaction.mode},
                                                 "evidence_event_ids": (), "truncated": True})
        return context.model_copy(deep=True)
