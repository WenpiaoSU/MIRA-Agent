"""作用：验证 M08 状态聚合、时效边界和现有网关集成。

当前功能：
- 使用虚拟时钟覆盖乱序、去重、TTL、来源和身份隔离。
- 检查新轮次、取消、播放回执、记忆清理与会话生命周期。
- 检查快照隔离、上下文预算、契约序列化及真实网关冻结路径。
"""

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError

from src.agent.graph import AgentGraph
from src.api.realtime import realtime
from src.contracts.perception import IntentEstimate
from src.contracts.state import (
    ActionState, CommittedUtterance, HealthState, InteractionState, RelationshipState,
    RobotAffectState, SocialEvent, SocialState, UserAffectState, VisualState,
)
from src.gateway.sessions import Session, SessionRegistry
from src.social_state import SocialStateConfig, SocialStateService
from test_p3 import FakeSocket, FakeSpeech, FakeTTS, RecordingAgent


def intent() -> IntentEstimate:
    """构造不依赖模型的有效意图。"""
    return IntentEstimate(intent="chat", slots=[], confidence=0.9, needs_clarification=False,
                          missing_slots=[], control_action=None)


class VirtualClock:
    """提供可以精确推进的 UTC 时钟。"""

    def __init__(self):
        """固定初始时间以避免测试依赖执行主机的时区。"""
        self.now = datetime(2026, 10, 18, tzinfo=timezone.utc)

    def __call__(self):
        """读取当前虚拟时间。"""
        return self.now

    def advance(self, milliseconds):
        """以毫秒推进虚拟时间。"""
        self.now += timedelta(milliseconds=milliseconds)


class SocialStateTests(unittest.TestCase):
    """验证归约契约而非简单映射配置。"""

    def setUp(self):
        """为每个场景创建独立状态和当前轮次。"""
        self.clock = VirtualClock()
        self.service = SocialStateService(SocialStateConfig(visual_ttl_ms=1000, affect_ttl_ms=2000), self.clock)
        self.service.bind("s", "u", 1)
        self.service.emit("s", "turn.updated", InteractionState(
            mode="thinking", user_speaking=False, robot_playing=False, turn_id="t", generation_id="g"), source="turn")
        self.seq = 0

    def event(self, kind="visual.updated", payload=None, **updates):
        """构造可覆盖归属、采集时间和来源的外部事件。"""
        self.seq += 1
        values = dict(event_id=f"e{self.seq}", type=kind, session_id="s", user_id="u", connection_epoch=1,
                      source="perception", seq=self.seq, captured_at=self.clock(), received_at=self.clock(),
                      turn_id="t", generation_id="g",
                      payload=payload if payload is not None else VisualState(presence="present", active_track_id="track"))
        values.update(updates)
        return SocialEvent(**values)

    def test_ttl_expires_at_capture_boundary_without_mutating_old_snapshot(self):
        """延迟到达不能延长有效期，读取与时钟事件均能执行到期清理。"""
        event = self.event(captured_at=self.clock() - timedelta(milliseconds=500))
        self.assertTrue(self.service.apply(event).accepted)
        frozen = self.service.snapshot("s")
        self.clock.advance(500)
        expired = self.service.snapshot("s")
        self.assertEqual(expired.visual.presence, "unknown")
        self.assertEqual(expired.staleness["visual"], "expired")
        self.assertEqual(expired.health["visual"].reason, "ttl_expired")
        self.assertEqual(expired.version, frozen.version + 1)
        self.assertEqual(frozen.visual.active_track_id, "track")
        self.assertEqual(self.service.snapshot("s").version, expired.version)
        self.service.emit("s", "affect.updated", UserAffectState(emotion="sad", confidence=0.5),
                          source="affect", turn_id="t", generation_id="g")
        self.clock.advance(2000)
        delta = self.service.emit("s", "clock.tick")
        self.assertIn("user_affect", delta.changed_domains)
        self.assertIsNone(self.service.snapshot("s").user_affect)

    def test_dedup_source_order_and_capture_order(self):
        """同源倒序与跨源更早观测均不能回退当前状态。"""
        first = self.event(seq=5)
        self.assertTrue(self.service.apply(first).accepted)
        self.assertEqual(self.service.apply(first).reason, "duplicate_or_out_of_order")
        self.assertEqual(self.service.apply(self.event(seq=4)).reason, "duplicate_or_out_of_order")
        older = self.event(source="other", captured_at=self.clock() - timedelta(milliseconds=1))
        self.assertEqual(self.service.apply(older).reason, "older_observation")
        self.clock.advance(10)
        self.assertTrue(self.service.apply(self.event(seq=6, payload=VisualState(presence="absent"))).accepted)
        self.assertEqual(self.service.snapshot("s").visual.presence, "absent")

    def test_future_and_excessively_late_observations(self):
        """未知未来时间与超过容忍窗口的观测显式拒绝。"""
        self.assertEqual(self.service.apply(self.event(captured_at=self.clock() + timedelta(seconds=1))).reason,
                         "future_observation")
        self.assertEqual(self.service.apply(self.event(captured_at=self.clock() - timedelta(seconds=6))).reason, "too_late")

    def test_identity_epoch_and_session_isolation(self):
        """旧连接与错误用户不能更新状态，可信身份切换清除先前输入。"""
        self.service.emit("s", "utterance.committed", CommittedUtterance(text="私人内容", source="text"),
                          turn_id="t", generation_id="g")
        self.assertEqual(self.service.apply(self.event(user_id="other")).reason, "identity_or_epoch_mismatch")
        self.assertEqual(self.service.apply(self.event(connection_epoch=0)).reason, "identity_or_epoch_mismatch")
        self.assertEqual(self.service.apply(self.event(session_id="missing")).reason, "unknown_session")
        before = self.service.snapshot("s")
        self.service.bind("s", "other", 2)
        changed = self.service.snapshot("s")
        self.assertGreater(changed.version, before.version)
        self.assertIsNone(changed.committed_text)
        self.assertEqual(changed.relationship.user_id, "other")
        self.assertEqual(changed.staleness["visual"], "missing")

    def test_camera_shutdown_blocks_late_observations_until_reenabled(self):
        """摄像头关闭立即移除目标，迟到视觉不能恢复被撤销的感知。"""
        self.service.apply(self.event())
        self.clock.advance(10)
        self.service.emit("s", "health.updated", HealthState(component="visual", status="unavailable", reason="camera_off"))
        state = self.service.snapshot("s")
        self.assertEqual(state.visual.presence, "unknown")
        self.assertEqual(state.staleness["visual"], "unavailable")
        self.assertEqual(self.service.apply(self.event()).reason, "modality_unavailable")
        self.service.emit("s", "health.updated", HealthState(component="visual", status="ok"))
        self.assertEqual(self.service.apply(self.event(captured_at=self.clock() - timedelta(milliseconds=5))).reason,
                         "before_health_boundary")
        self.assertTrue(self.service.apply(self.event()).accepted)

    def test_new_turn_clears_intent_affect_and_old_evidence(self):
        """跨轮不得沿用情绪或意图，旧轮结果也不能污染下一轮。"""
        self.service.emit("s", "intent.updated", intent(), turn_id="t", generation_id="g")
        self.service.emit("s", "affect.updated", UserAffectState(emotion="sad"), turn_id="t", generation_id="g")
        self.service.emit("s", "turn.updated", InteractionState(
            mode="thinking", user_speaking=False, robot_playing=False, turn_id="t2", generation_id="g2"))
        state = self.service.snapshot("s")
        self.assertIsNone(state.intent)
        self.assertIsNone(state.user_affect)
        self.assertNotIn("intent", state.provenance)
        self.assertEqual(self.service.apply(self.event("intent.updated", intent())).reason, "turn_or_generation_mismatch")

    def test_affect_does_not_set_robot_expression(self):
        """用户情感与角色表达独立，角色状态仅接收可信 M16 确认事件。"""
        self.service.apply(self.event("affect.updated", UserAffectState(emotion="sad", confidence=0.7)))
        self.assertIsNone(self.service.snapshot("s").robot_affect.expression)
        cue = RobotAffectState(expression="concerned", intensity=0.5, behavior_plan_id="p")
        self.assertEqual(self.service.apply(self.event("robot_affect.confirmed", cue)).reason, "untrusted_expression")
        self.assertTrue(self.service.apply(self.event("robot_affect.confirmed", cue, source="expression")).accepted)
        self.assertEqual(self.service.snapshot("s").robot_affect.expression, "concerned")

    def test_relationship_ownership_and_version(self):
        """关系汇总必须匹配主体，旧版本不能覆盖新版本。"""
        foreign = RelationshipState(user_id="other", version=1)
        self.assertEqual(self.service.apply(self.event("relationship.updated", foreign)).reason, "relationship_owner_mismatch")
        relation = RelationshipState(user_id="u", version=2, familiarity=0.3)
        self.assertTrue(self.service.apply(self.event("relationship.updated", relation)).accepted)
        self.assertEqual(self.service.apply(self.event("relationship.updated", relation)).reason, "old_relationship_version")

    def test_affect_modality_weights_windows_and_provider_expiry(self):
        """拒绝非法融合权重及倒退窗口，提供者更短的有效期优先。"""
        with self.assertRaises(ValidationError):
            UserAffectState(modalities={"audio": {"available": False, "weight": 1}})
        with self.assertRaises(ValidationError):
            UserAffectState(modalities={"audio": {"available": True, "weight": 0.3}})
        with self.assertRaises(ValidationError):
            UserAffectState(window_start_ms=20, window_end_ms=10)
        affect = UserAffectState(emotion="sad", confidence=0.7, affect_distribution={"sad": 1.0},
                                modalities={"text": {"available": True, "quality": 0.8, "weight": 1}},
                                window_start_ms=10, window_end_ms=20,
                                expires_at=self.clock() + timedelta(milliseconds=50))
        self.assertTrue(self.service.apply(self.event("affect.updated", affect)).accepted)
        self.clock.advance(50)
        self.assertIsNone(self.service.snapshot("s").user_affect)

    def test_actions_require_binding_and_cancel_with_generation(self):
        """执行摘要须匹配轮次，取消不遗留运行中的行动。"""
        action = ActionState(action_id="call", kind="tool", status="running", generation_id="g", turn_id="t")
        self.assertEqual(self.service.apply(self.event("action.updated", action.model_copy(update={"turn_id": "wrong"}))).reason,
                         "action_owner_mismatch")
        self.assertTrue(self.service.apply(self.event("action.updated", action)).accepted)
        self.service.emit("s", "turn.updated", InteractionState(
            mode="recovering", user_speaking=False, robot_playing=False, turn_id="t", generation_id="g"))
        state = self.service.snapshot("s")
        self.assertFalse(state.pending_actions)
        self.assertEqual(state.last_action.status, "cancelled")
        self.assertEqual(self.service.apply(self.event("intent.updated", intent())).reason, "inactive_generation")

    def test_nested_snapshot_and_input_containers_are_isolated(self):
        """修改外部输入或已返回容器都不影响后续快照。"""
        affect = UserAffectState(emotion="sad", affect_distribution={"sad": 1.0})
        self.service.apply(self.event("affect.updated", affect))
        affect.affect_distribution.clear()
        snapshot = self.service.snapshot("s")
        snapshot.user_affect.affect_distribution.clear()
        snapshot.provenance.clear()
        current = self.service.snapshot("s")
        self.assertEqual(current.user_affect.affect_distribution, {"sad": 1.0})
        self.assertIn("user_affect", current.provenance)
        self.assertEqual(SocialState.model_validate_json(current.model_dump_json()), current)

    def test_contract_rejects_partial_malformed_probabilities_and_naive_time(self):
        """partial 不属于状态事件，非法概率和无时区采集时间拒绝入库。"""
        with self.assertRaises(ValidationError):
            self.event(kind="asr.partial")
        with self.assertRaises(ValidationError):
            UserAffectState(affect_distribution={"sad": 0.8})
        with self.assertRaises(ValidationError):
            VisualState(presence="present", face_center=(2, 0))
        with self.assertRaises(ValidationError):
            self.event(captured_at=datetime(2026, 10, 18))
        event = self.event()
        self.assertEqual(SocialEvent.model_validate_json(event.model_dump_json()), event)

    def test_context_budget_and_missing_modalities(self):
        """完整 JSON 不超过最小预算，保留降级标记并显式说明裁剪。"""
        self.service.emit("s", "utterance.committed", CommittedUtterance(text="长输入" * 3000, source="text"),
                          turn_id="t", generation_id="g")
        bounded = SocialStateService(SocialStateConfig(context_max_chars=1024))
        context = bounded.to_agent_context(self.service.snapshot("s"))
        self.assertLessEqual(len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False)), 1024)
        self.assertTrue(context.truncated)
        self.assertEqual(set(context.missing_modalities), {"visual", "user_affect"})

    def test_dedup_cache_is_bounded(self):
        """高频感知不累积完整事件流，序号水位阻止被淘汰事件重新生效。"""
        service = SocialStateService(SocialStateConfig(dedup_capacity=4), self.clock)
        service.bind("s", "u", 1)
        first = self.event()
        service.apply(first)
        for _ in range(10):
            service.apply(self.event())
        self.assertEqual(len(service.entries["s"].seen), 4)
        self.assertEqual(service.apply(first).reason, "duplicate_or_out_of_order")

    def test_memory_boundary_rejects_delayed_derived_data(self):
        """删除前采集的结果不能在同一当前轮次重新建立派生上下文。"""
        delayed = self.event("intent.updated", intent())
        self.clock.advance(10)
        self.service.emit("s", "memory.cleared")
        self.assertEqual(self.service.apply(delayed).reason, "before_memory_boundary")
        self.assertIsNone(self.service.snapshot("s").intent)

    def test_health_capacity_and_context_fallback_remain_bounded(self):
        """来源和健康域有界，最小预算在大量降级信息下仍然有效。"""
        for index in range(32):
            result = self.service.emit("s", "health.updated", HealthState(
                component=str(index) + "x" * 70, status="degraded", reason="原因" * 200))
            self.assertTrue(result.accepted)
        self.assertEqual(self.service.emit("s", "health.updated", HealthState(
            component="overflow", status="degraded")).reason, "health_component_limit")
        bounded = SocialStateService(SocialStateConfig(context_max_chars=1024))
        context = bounded.to_agent_context(self.service.snapshot("s"))
        self.assertLessEqual(len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False)), 1024)


class SessionSocialTests(unittest.TestCase):
    """验证既有会话账本投影，避免复制一套轮次状态机。"""

    def test_session_projection_playback_and_late_receipt(self):
        """开始生成不是播放，旧代次终态回执只更新原交付账本。"""
        session = Session()
        session.begin_generation("g", "t", "完整输入", speech_enabled=True, source="speech")
        state = session.social_snapshot(intent=intent())
        self.assertEqual(state.utterance.source, "speech")
        self.assertFalse(state.interaction.robot_playing)
        session.register_segment("g", {"segment_id": "a", "text": "回复"})
        session.register_audio("a", 1000)
        self.assertEqual(session.social_snapshot().pending_actions[0].status, "pending")
        session.record_playback("a", "playback.started", 0, 1000, "g")
        self.assertTrue(session.social_snapshot().interaction.robot_playing)
        session.cancel_generation()
        session.begin_generation("g2", "t2", "新输入")
        session.record_playback("a", "playback.stopped", 400, 1000, "g")
        new = session.social_snapshot()
        self.assertEqual(new.committed_text, "新输入")
        self.assertEqual(new.interaction.generation_id, "g2")
        self.assertFalse(new.pending_actions)
        self.assertIsNone(new.intent)

    def test_registry_cleanup_and_memory_deletion_clear_derived_state(self):
        """记忆删除清理 M08 派生输入和证据，关闭会话释放状态。"""
        registry = SessionRegistry()
        session = registry.create("u")
        other = registry.create("other")
        session.begin_generation("g", "t", "删除的事实")
        session.social_snapshot(intent=intent())
        registry.forget_memories("u")
        snapshot = registry.social_state.snapshot(session.session_id)
        self.assertIsNone(snapshot.committed_text)
        self.assertIsNone(snapshot.intent)
        self.assertNotIn("utterance", snapshot.provenance)
        self.assertEqual(registry.social_state.snapshot(other.session_id).user_id, "other")
        registry.delete(session.session_id)
        session.sync_social_state()
        self.assertNotIn(session.session_id, registry.social_state.entries)
        with self.assertRaises(ValueError):
            session.begin_generation("late", "late", "迟到输入")

    def test_prompt_uses_compact_state_without_raw_state_duplication(self):
        """图执行保留完整意图，模型只读紧凑观测并兼容带时区时间。"""
        session = Session()
        session.begin_generation("g", "t", "你好")
        snapshot = session.social_snapshot(intent=intent())
        compact = session.social_state.to_agent_context(snapshot)
        state = {"user_text": "你好", "history": [], "recent_avatar_cues": [],
                 "runtime_context": {"social_state": snapshot.model_dump(), "social_context": compact.model_dump()}}
        prompt = AgentGraph.prepare_context(state)["messages"][0]["content"]
        self.assertIn('"social_context"', prompt)
        self.assertNotIn('"social_state"', prompt)


class SocialGatewayTests(unittest.IsolatedAsyncioTestCase):
    """通过生产 WebSocket 函数检查冻结状态与来源。"""

    async def test_gateway_freezes_compact_version_and_records_only_metadata(self):
        """最终输入进入聚合器，Agent 得到同一版本摘要，轨迹不包含用户正文。"""
        registry = SessionRegistry()
        session = registry.create()
        agent = RecordingAgent()
        app = SimpleNamespace(state=SimpleNamespace(
            sessions=registry, agent=agent, tts=FakeTTS(), speech_service=FakeSpeech(),
            intent_service=SimpleNamespace(classify=AsyncMock(return_value=intent())),
            settings=SimpleNamespace(llm_ready=True, tts_ready=False, stop_ack_timeout_ms=500)))
        socket = FakeSocket(app, session)
        task = asyncio.create_task(realtime(socket, session.session_id))
        try:
            await socket.until("session.status")
            await socket.send("input.text", {"text": "本轮私人输入"}, "t")
            await socket.until("response.completed")
            runtime = agent.calls[0][1]
            self.assertEqual(runtime["social_context"]["version"], runtime["social_state"]["version"])
            self.assertEqual(runtime["social_context"]["data"]["final_text"], "本轮私人输入")
            self.assertEqual(set(runtime["social_context"]["missing_modalities"]), {"visual", "user_affect"})
            self.assertNotIn("本轮私人输入", json.dumps(session.trace.report(), ensure_ascii=False))
            frozen = [event for event in session.trace.events if event["type"] == "social_state.frozen"]
            self.assertEqual(frozen[0]["payload"]["state_version"], runtime["social_context"]["version"])
        finally:
            await socket.close()
            await asyncio.wait_for(task, 3)


if __name__ == "__main__":
    unittest.main()
