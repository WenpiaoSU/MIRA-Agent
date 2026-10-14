"""作用：验证记忆系统的持久化、隔离、交付事实和删除生命周期。

当前功能：
- 在临时 SQLite 中验证真实事务、版本更正、幂等和用户隔离。
- 运行生产 LangGraph 与 Checkpointer，确认跨轮状态和旧代次写入边界。
- 使用本地模型替身及 HTTP 接口检查跨会话查询、取消和浏览器归属。
"""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from src.agent.graph import AgentGraph
from src.api import memories, sessions
from src.api.realtime import realtime
from src.contracts.memory import MemoryCandidate
from src.contracts.perception import IntentEstimate
from src.gateway.identity import owns_session
from src.gateway.sessions import SessionRegistry
from src.memory.service import MemoryService
from src.repositories.memory import MemoryRepository
from test_p3 import FakeSocket, FakeSpeech, FakeTTS


class MemoryProvider:
    """提供明确候选或回答文本，并记录实际送入模型的数据。"""

    def __init__(self):
        """初始化候选、调用记录与取消控制。"""
        self.candidate = MemoryCandidate(action="remember", fact_key="user.reply_style",
                                         content="我喜欢简短回答", evidence="我喜欢简短回答")
        self.messages = []
        self.started = asyncio.Event()
        self.block = False

    async def stream(self, messages):
        """按系统提示区分提取与回答，测试不访问任何网络模型。"""
        self.messages.append(messages)
        if "你负责提取一个长期记忆操作" in messages[0]["content"]:
            self.started.set()
            if self.block:
                await asyncio.sleep(20)
            yield self.candidate.model_dump_json()
        else:
            yield '{"expression":"neutral","gaze":"center","gesture":null,"intensity":0.6}\n根据已有记录回答。'


def intent(name):
    """构造通过生产契约校验的意图，隔离本测试不涉及的分类质量。"""
    return IntentEstimate(intent=name, slots=[], confidence=.9, needs_clarification=False,
                          missing_slots=[], control_action=None)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    """验证真实存储和编排，而不是仅检查模型替身输出。"""

    async def asyncSetUp(self):
        """每项测试使用独立数据库、用户、图与会话注册表。"""
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "memory.sqlite3")
        self.repo = MemoryRepository(self.path)
        self.user, self.token = await self.repo.identity(None, create=True)
        self.provider = MemoryProvider()
        self.service = MemoryService(self.repo, self.provider)
        self.agent = AgentGraph(self.provider, memory=self.service)
        self.sessions = SessionRegistry()
        self.sessions.forget_checkpoint = self.agent.working.forget
        self.sessions.invalidate_checkpoint = self.agent.working.invalidate
        self.service.on_delete = self.sessions.forget_memories
        self.session = self.sessions.create(self.user)
        self.count = 0

    async def asyncTearDown(self):
        """等待短事务清理后删除临时数据库。"""
        self.temp.cleanup()

    async def update(self, text="我喜欢简短回答", turn="t1"):
        """通过生产服务进行一次有截止时间的操作。"""
        return await self.service.update(text, self.user, self.session.session_id, turn, turn,
                                         datetime.now(timezone.utc) + timedelta(seconds=10))

    async def turn(self, text, name="chat", session=None):
        """模拟网关提交与展示回执，保留实际 Checkpointer 数据。"""
        session = session or self.session
        self.count += 1
        generation = f"g{self.count}"
        session.begin_generation(generation, generation, text)
        history = await self.agent.prepare_turn(session, generation)
        runtime = {"social_state": session.social_snapshot(text, intent(name)).model_dump(), "turn_id": generation}
        events = [event async for event in self.agent.stream(text, history, [], generation, runtime)]
        for event in events:
            if event["type"] == "response.segment":
                session.register_segment(generation, event["payload"])
                session.displayed(generation, event["payload"]["segment_id"])
        await self.agent.refresh_delivery(session, generation)
        return events

    async def test_sqlite_reopen_identity_and_user_isolation(self):
        """重建存储对象后事实与身份仍存在，其他用户看不到记录。"""
        result = await self.update()
        self.assertEqual(result.status, "saved")
        reopened = MemoryRepository(self.path)
        self.assertEqual((await reopened.identity(self.token))[0], self.user)
        _, rows = await reopened.snapshot(self.user)
        self.assertEqual(rows[0].content, "我喜欢简短回答")
        self.assertEqual(rows[0].source_turn_id, "t1")
        other, _ = await reopened.identity(None, create=True)
        self.assertEqual((await reopened.snapshot(other))[1], [])

    async def test_versions_dedup_conflict_and_stale_commit_after_delete(self):
        """重复调用不增版本，明确更正保留历史，删除后旧提取不能恢复信息。"""
        saved = await self.update()
        self.assertEqual((await self.update()).memory_ids, saved.memory_ids)
        self.provider.candidate = MemoryCandidate(action="remember", fact_key="user.reply_style",
                                                   content="我喜欢详细回答", evidence="我喜欢详细回答")
        self.assertEqual((await self.update("我喜欢详细回答", "t2")).status, "clarify")
        self.provider.candidate.action = "update"
        self.assertEqual((await self.update("我现在改为：我喜欢详细回答", "t3")).status, "saved")
        revision, versions = await self.repo.snapshot(self.user, historical=True)
        self.assertEqual({row.status for row in versions}, {"active", "inactive"})
        active = next(row for row in versions if row.status == "active")
        self.assertEqual(active.version, 2)
        removed = await self.repo.commit(self.user, "delete", MemoryCandidate(action="delete", target_ids=[active.memory_id]),
                                         revision, "s", "t")
        self.assertEqual(removed.deleted_count, 2)
        late = await self.repo.commit(self.user, "late", self.provider.candidate, revision, "s", "t")
        self.assertEqual(late.status, "conflict")
        self.assertEqual((await self.repo.snapshot(self.user, historical=True))[1], [])

    async def test_extraction_evidence_gate_and_cancel(self):
        """模型编造正文或使用不在本轮的证据不能保存，取消提取不会产生写入。"""
        self.provider.candidate.content = "用户在北京"
        self.assertEqual((await self.update()).status, "failed")
        self.provider.candidate.content = "我喜欢简短回答"
        self.provider.block = True
        self.provider.started.clear()
        task = asyncio.create_task(self.update(turn="cancel"))
        await self.provider.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((await self.repo.snapshot(self.user))[1], [])

    async def test_graph_cross_session_recall_and_state_reset(self):
        """写入后另一个会话可检索，上一轮的固定确认不能覆盖下一轮普通回答。"""
        first = await self.turn("我喜欢简短回答", "memory_update")
        self.assertIn("已保存", first[-1]["payload"]["text"])
        self.assertEqual(len(self.provider.messages), 1)
        second = await self.turn("解释一下光合作用")
        self.assertEqual(second[-1]["payload"]["text"], "根据已有记录回答。")
        self.assertIn("我喜欢简短回答", self.provider.messages[-1][0]["content"])
        other = self.sessions.create(self.user)
        await self.turn("你记得我的回答风格吗", "memory_query", other)
        self.assertIn("我喜欢简短回答", self.provider.messages[-1][0]["content"])
        self.assertIn("source_turn_id", self.provider.messages[-1][0]["content"])

    async def test_checkpoint_is_memory_source_and_never_assumes_speech(self):
        """清空实时账本后检查点仍提供历史，未展示或未播放正文不冒充交付。"""
        await self.turn("我正在看一本关于树木的书")
        self.session.outputs.clear()
        self.session.begin_generation("next", "next", "它说什么？")
        history = await self.agent.prepare_turn(self.session, "next")
        self.assertTrue(any("树木" in message["content"] for message in history))
        self.assertTrue(any("已有记录" in message["content"] for message in history))
        self.session.register_segment("next", {"segment_id": "not_seen", "text": "未交付秘密", "index": 1})
        await self.agent.refresh_delivery(self.session, "next")
        working = self.agent.working.read(self.session.session_id)
        self.assertNotIn("未交付秘密", json.dumps(working, ensure_ascii=False))
        isolated = self.sessions.create()
        isolated.begin_generation("isolated", "t", "你好")
        self.assertEqual(await self.agent.prepare_turn(isolated, "isolated"), [])

    async def test_old_checkpoint_write_and_unbounded_growth_are_blocked(self):
        """新轮次撤销旧配置写入权，多轮执行不会保留所有过去快照。"""
        await self.turn("第一轮")
        old = self.agent.working.config(self.session.session_id, "g1")
        await self.turn("第二轮")
        await self.agent.graph.aupdate_state(old, {"working_turns": [{"generation_id": "bad", "user": "旧污染"}]}, as_node="generate_response")
        self.assertNotIn("旧污染", json.dumps(self.agent.working.read(self.session.session_id), ensure_ascii=False))
        for index in range(12):
            await self.turn(f"第 {index} 轮")
        history = self.agent.working.read(self.session.session_id)
        self.assertLessEqual(len(history), 12)
        checkpoints = list(self.agent.working.checkpointer.list({"configurable": {"thread_id": self.session.session_id}}))
        self.assertLess(len(checkpoints), 15)

    async def test_expired_dialogue_is_not_rehydrated_from_output_ledger(self):
        """过期交付记录不能因仍在播放账本里而重新进入短期记忆。"""
        await self.turn("过期事实")
        self.session.outputs["g1"].created_at -= 7200
        self.agent.working.touched[self.session.session_id] -= 7200
        self.session.begin_generation("new", "new", "现在呢")
        history = await self.agent.prepare_turn(self.session, "new")
        self.assertEqual(history, [])

    async def test_stop_immediately_revokes_checkpoint_write_without_losing_input(self):
        """停止后即使没有新轮次，旧节点也不能更新检查点；已提交输入仍保留。"""
        await self.turn("需要保留的输入")
        config = self.agent.working.config(self.session.session_id, "g1")
        self.session.cancel_generation("user_stop")
        await self.agent.graph.aupdate_state(config, {"working_turns": []}, as_node="generate_response")
        self.assertIn("需要保留的输入", json.dumps(self.agent.working.read(self.session.session_id), ensure_ascii=False))

    async def test_delete_clears_all_user_context_and_prevents_resurrection(self):
        """删除清理当前用户全部会话缓存和版本，其他用户的检查点保持独立。"""
        await self.turn("我喜欢简短回答", "memory_update")
        other = self.sessions.create(self.user)
        await self.turn("我也在这个窗口聊简短回答", session=other)
        stranger = self.sessions.create("other-user")
        await self.turn("另一个人的信息", session=stranger)
        self.provider.candidate = MemoryCandidate(action="delete", delete_all=True, evidence="忘掉我的全部记忆")
        result = await self.turn("忘掉我的全部记忆", "memory_update")
        self.assertIn("已删除", result[-1]["payload"]["text"])
        self.assertEqual((await self.repo.snapshot(self.user, historical=True))[1], [])
        self.assertEqual(self.agent.working.read(other.session_id), [])
        self.assertEqual(other.context_history(), [])
        self.assertTrue(self.agent.working.read(stranger.session_id))
        await self.turn("之前聊过什么", "memory_query")
        prompt = json.dumps(self.provider.messages[-1], ensure_ascii=False)
        self.assertNotIn("我喜欢简短回答", prompt)

    async def test_database_failure_does_not_claim_saved(self):
        """数据库提交失败返回明确失败，不产生虚假的保存确认。"""
        self.repo.commit = AsyncMock(side_effect=sqlite3.OperationalError("locked"))
        result = await self.update()
        self.assertEqual(result.status, "failed")
        self.assertIn("未确认完成", result.message)

    async def test_failed_read_is_distinct_from_empty_memory(self):
        """明确查询遇到数据库故障时报告无法核实，普通交流继续使用短期上下文。"""
        self.repo.snapshot = AsyncMock(side_effect=sqlite3.OperationalError("locked"))
        queried = await self.turn("你记得我的偏好吗", "memory_query")
        self.assertIn("无法读取", queried[-1]["payload"]["text"])
        chatted = await self.turn("你好")
        self.assertEqual(chatted[-1]["payload"]["text"], "根据已有记录回答。")

    async def test_same_operation_id_cannot_confirm_different_request(self):
        """重复轮次只有原请求可复用回执，不同正文必须作为新请求处理。"""
        await self.update()
        self.provider.candidate = MemoryCandidate(action="remember", fact_key="user.home_city",
                                                   content="我住在杭州", evidence="我住在杭州")
        result = await self.update("我住在杭州")
        self.assertEqual(result.status, "conflict")
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 1)

    async def test_gateway_delete_notification_precedes_reply_and_recall(self):
        """实际 WebSocket 网关完成保存、删除和下一轮查询，且清理事件先于确认正文。"""
        classifier = SimpleNamespace(classify=AsyncMock(return_value=intent("memory_update")))
        app = SimpleNamespace(state=SimpleNamespace(
            sessions=self.sessions, agent=self.agent, memory=self.service, intent_service=classifier,
            settings=SimpleNamespace(llm_ready=True, tts_ready=False, stop_ack_timeout_ms=100),
            tts=FakeTTS(), speech_service=FakeSpeech()))
        socket = FakeSocket(app, self.session)
        task = asyncio.create_task(realtime(socket, self.session.session_id))
        try:
            await socket.until("session.status")
            await socket.send("input.text", {"text": "我喜欢简短回答"}, "save")
            saved = await socket.until("response.completed")
            self.assertIn("已保存", saved["payload"]["text"])
            await socket.until("turn.delivered")
            self.provider.candidate = MemoryCandidate(action="delete", delete_all=True, evidence="忘掉我的全部记忆")
            await socket.send("input.text", {"text": "忘掉我的全部记忆"}, "delete")
            await socket.until("memory.cleared")
            deleted = await socket.until("response.completed")
            self.assertIn("已删除", deleted["payload"]["text"])
            await socket.until("turn.delivered")
            classifier.classify.return_value = intent("memory_query")
            await socket.send("input.text", {"text": "你记得我的回答偏好吗"}, "query")
            answer = await socket.until("response.completed")
            self.assertIn("没有找到", answer["payload"]["text"])
        finally:
            await socket.close()
            await asyncio.wait_for(task, 3)

    async def test_cookie_identity_and_http_access_isolation(self):
        """同一浏览器新会话保持用户归属，另一浏览器不能查看、删除或接管其会话。"""
        app = FastAPI()
        app.state.memory = self.service
        app.state.sessions = self.sessions
        app.state.speech_service = SimpleNamespace(ready=False)
        app.include_router(sessions.router)
        app.include_router(memories.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as first:
            a = (await first.post("/v1/sessions")).json()
            b = (await first.post("/v1/sessions")).json()
            self.assertEqual(a["user_id"], b["user_id"])
            self.assertIn("HttpOnly", (await first.post("/v1/sessions")).headers["set-cookie"])
            owned = self.sessions.get(a["session_id"])
            self.assertTrue(owns_session(SimpleNamespace(cookies=first.cookies), owned))
            revision, _ = await self.repo.snapshot(a["user_id"])
            saved = await self.repo.commit(a["user_id"], "save", self.provider.candidate, revision, "s", "t")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as second:
                await second.post("/v1/sessions")
                self.assertEqual((await second.get("/v1/memories")).json()["records"], [])
                self.assertEqual((await second.delete("/v1/memories/" + saved.memory_ids[0])).status_code, 404)
                self.assertEqual((await second.get("/v1/sessions/" + a["session_id"] + "/trace")).status_code, 404)
                self.assertFalse(owns_session(SimpleNamespace(cookies=second.cookies), owned))
            self.assertEqual(len((await first.get("/v1/memories")).json()["records"]), 1)
            self.assertEqual((await first.delete("/v1/memories")).json()["status"], "deleted")
