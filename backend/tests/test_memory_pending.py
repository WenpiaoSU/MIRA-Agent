"""作用：验证 PENDING 从持久化发现到用户决定及版本提交的完整流程。

当前功能：
- 使用真实事务验证采纳、保留、幂等、去重、旧版本及身份/时效边界。
- 验证连接展示授权、短答撤销、后台通知和重连恢复。
- 通过真实 HTTP 和 WebSocket 运行实际确认及派生缓存失效路径。
"""

import asyncio
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from src.agent.graph import AgentGraph
from src.api import memories
from src.api.realtime import realtime
from src.contracts.memory import MemoryCandidate, MemoryPendingAck, MemoryResolution
from src.gateway.identity import COOKIE_NAME
from src.gateway.memories import MemoryPendingDelivery
from src.gateway.sessions import SessionRegistry
from src.memory.service import MemoryService
from src.repositories.memory import MemoryRepository
from test_memory import intent
from test_memory_gates import GateProvider, fact
from test_p3 import FakeSocket, FakeSpeech, FakeTTS


class PendingMemoryTests(unittest.IsolatedAsyncioTestCase):
    """测试通过生产接口检查事实与候选状态，不访问网络模型。"""

    async def asyncSetUp(self):
        """创建可信用户、真实数据库及可控的语义模型替身。"""
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "pending.sqlite3")
        self.repo = MemoryRepository(self.path)
        self.user, self.token = await self.repo.identity(None, create=True)
        self.provider = GateProvider()
        self.service = MemoryService(self.repo, self.provider)
        self.sessions = SessionRegistry()
        self.session = self.sessions.create(self.user)
        self.service.on_change = self.sessions.memories_changed
        self.agent = None

    async def asyncTearDown(self):
        """停止后台工作并释放检查点文件。"""
        await self.service.close()
        if self.agent:
            self.agent.working.close()
        self.temp.cleanup()

    async def write(self, text="我喜欢简短回答", turn="old"):
        """在生产服务中提取并提交指定轮次的事实。"""
        return await self.service.update(text, self.user, self.session.session_id, turn, turn,
                                        datetime.now(timezone.utc) + timedelta(seconds=10))

    async def create_pending(self):
        """先建立真实旧事实，再生成具有不同正文的冲突候选。"""
        await self.write()
        self.provider.facts = [fact("我喜欢详细回答")]
        result = await self.write("我喜欢详细回答", "new")
        self.assertEqual(result.status, "pending")
        return (await self.repo.pending(self.user))[0]

    def request(self, row, action="accept_new"):
        """构造只包含明确决定和原版本号的确认。"""
        return MemoryResolution(action=action, target_version=row["target_version"])

    def execute(self, statement, parameters=()):
        """在独立连接中设置竞争或过期条件，结束后关闭文件。"""
        with closing(self.repo.connect()) as db:
            db.execute(statement, parameters)
            db.commit()

    async def test_accept_versions_original_provenance_and_idempotence(self):
        """采纳保留原来源与旧版本，同一决定重复提交不再次增版本。"""
        row = await self.create_pending()
        calls = len(self.provider.messages)
        result = await self.service.resolve(self.user, row["pending_id"], self.request(row), evidence="使用新信息")
        self.assertTrue(result.changed)
        self.assertEqual(await self.repo.pending(self.user), [])
        rows = (await self.repo.snapshot(self.user, historical=True))[1]
        self.assertEqual({record.status for record in rows}, {"active", "inactive"})
        active = next(record for record in rows if record.status == "active")
        self.assertEqual((active.content, active.version, active.source_turn_id), ("我喜欢详细回答", 2, "new"))
        repeated = await self.service.resolve(self.user, row["pending_id"], self.request(row))
        self.assertEqual(repeated.memory_ids, result.memory_ids)
        self.assertFalse(repeated.changed)
        self.assertEqual(len((await self.repo.snapshot(self.user, historical=True))[1]), 2)
        opposite = await self.service.resolve(self.user, row["pending_id"], self.request(row, "keep_old"))
        self.assertEqual(opposite.status, "conflict")
        self.assertEqual(len(self.provider.messages), calls)

    async def test_keep_old_deduplicates_repeated_candidate_after_restart(self):
        """拒绝后重启仍保留决定，新操作中的同一候选不会再次排队询问。"""
        row = await self.create_pending()
        result = await self.service.resolve(self.user, row["pending_id"], self.request(row, "keep_old"))
        self.assertEqual(result.status, "unchanged")
        self.service.repository = MemoryRepository(self.path)
        again = await self.write("我喜欢详细回答", "another")
        self.assertEqual(again.decisions, ["SKIP"])
        self.assertEqual(await self.repo.pending(self.user), [])
        self.assertEqual((await self.repo.snapshot(self.user))[1][0].content, "我喜欢简短回答")

    async def test_pending_deduplicates_same_fact_from_new_turn(self):
        """多轮同一目标、同一值只对应一个待确认 ID。"""
        row = await self.create_pending()
        duplicate = await self.write("我喜欢详细回答", "duplicate")
        self.assertEqual(duplicate.pending_ids, [row["pending_id"]])
        self.assertEqual(len(await self.repo.pending(self.user)), 1)

    async def test_concurrent_accepts_commit_one_version(self):
        """两个连接同时采纳同一候选，事务与幂等回执只生成一个新版本。"""
        row = await self.create_pending()
        first, second = await asyncio.gather(*(self.service.resolve(self.user, row["pending_id"], self.request(row)) for _ in range(2)))
        self.assertEqual(first.memory_ids, second.memory_ids)
        self.assertEqual(sum([first.changed, second.changed]), 1)
        self.assertEqual(len((await self.repo.snapshot(self.user, historical=True))[1]), 2)

    async def test_expired_wrong_version_and_changed_target_are_not_applied(self):
        """客户端旧快照或目标已被修改时不覆盖，新候选过期也不采纳。"""
        row = await self.create_pending()
        wrong = MemoryResolution(action="accept_new", target_version=row["target_version"] + 1)
        self.assertEqual((await self.service.resolve(self.user, row["pending_id"], wrong)).status, "conflict")
        self.provider.facts = [fact("我喜欢分点回答")]
        await self.write("更正：我喜欢分点回答", "changed")
        result = await self.service.resolve(self.user, row["pending_id"], self.request(row))
        self.assertEqual(result.status, "conflict")
        self.assertEqual((await self.repo.snapshot(self.user))[1][0].content, "我喜欢分点回答")
        self.provider.facts = [fact("我喜欢详细回答")]
        await self.write("我喜欢详细回答", "pending_again")
        later = (await self.repo.pending(self.user))[0]
        self.execute("UPDATE memory_pending SET expires_at=0 WHERE pending_id=?", (later["pending_id"],))
        self.assertEqual((await self.service.resolve(self.user, later["pending_id"], self.request(later))).status, "expired")

    async def test_deleted_fact_and_other_user_cannot_be_resolved(self):
        """不能跨用户选择候选，删除后旧决定也无法重新创建事实。"""
        row = await self.create_pending()
        other, _ = await self.repo.identity(None, create=True)
        self.assertEqual((await self.service.resolve(other, row["pending_id"], self.request(row))).status, "not_found")
        revision, _ = await self.repo.snapshot(self.user)
        await self.repo.commit(self.user, "delete", MemoryCandidate(action="delete", delete_all=True), revision, "s", "d")
        self.assertEqual((await self.service.resolve(self.user, row["pending_id"], self.request(row))).status, "not_found")
        self.assertEqual((await self.repo.snapshot(self.user))[1], [])

    async def test_other_fact_change_does_not_invalidate_target_version(self):
        """无关事实更新不让有效候选永久无法处理，确认仍核对原目标版本。"""
        row = await self.create_pending()
        revision, _ = await self.repo.snapshot(self.user)
        await self.repo.commit(self.user, "city", MemoryCandidate(action="remember", fact_key="user.home_city",
            content="我住在杭州", evidence="我住在杭州"), revision, "s", "city")
        self.assertEqual((await self.service.resolve(self.user, row["pending_id"], self.request(row))).status, "saved")

    async def test_legacy_pending_migration_keeps_target_and_original_source(self):
        """旧八列表迁移恢复不可变目标版本和来源，已有跨项目同名事实也可正常重开。"""
        row = await self.create_pending()
        revision, _ = await self.repo.snapshot(self.user)
        await self.repo.commit(self.user, "project", MemoryCandidate(action="remember", fact_key="user.reply_style",
            domain_scope="project:Alpha", content="项目说明需要详细", evidence="项目说明需要详细"), revision, "s", "p")
        with closing(self.repo.connect()) as db:
            old = dict(db.execute("SELECT * FROM memory_pending WHERE pending_id=?", (row["pending_id"],)).fetchone())
            db.executescript("""
                DROP TABLE memory_pending;
                CREATE TABLE memory_pending(pending_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,operation_id TEXT NOT NULL,
                    candidate TEXT NOT NULL,target_id TEXT,reason TEXT NOT NULL,created_at TEXT NOT NULL,expires_at REAL NOT NULL,
                    UNIQUE(user_id,operation_id));
            """)
            db.execute("INSERT INTO memory_pending VALUES (?,?,?,?,?,?,?,?)", tuple(old[key] for key in (
                "pending_id", "user_id", "operation_id", "candidate", "target_id", "reason", "created_at", "expires_at")))
            db.commit()
        migrated = MemoryRepository(self.path)
        self.service.repository = migrated
        recovered = (await migrated.pending(self.user))[0]
        self.assertEqual((recovered["pending_id"], recovered["target_version"], recovered["target_content"]),
                         (row["pending_id"], 1, "我喜欢简短回答"))
        result = await self.service.resolve(self.user, recovered["pending_id"], self.request(recovered))
        self.assertEqual(result.status, "saved")
        active = next(record for record in (await migrated.snapshot(self.user))[1] if record.domain_scope == "daily")
        self.assertEqual((active.source_session_id, active.source_turn_id), (self.session.session_id, "new"))

    async def test_http_requires_owner_and_rejects_client_fact_payload(self):
        """真实 HTTP 确认只接受决定与版本，拒绝伪造归属和正文。"""
        row = await self.create_pending()
        app = FastAPI()
        app.state.memory = self.service
        app.include_router(memories.router)
        url = "/v1/memories/pending/" + row["pending_id"] + "/resolve"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            self.assertEqual((await client.post(url, json=self.request(row).model_dump())).status_code, 401)
            client.cookies.set(COOKIE_NAME, self.token)
            self.assertEqual((await client.get("/v1/memories/pending")).json()["pending"][0]["target_content"], "我喜欢简短回答")
            malicious = {**self.request(row).model_dump(), "user_id": self.user, "content": "用户是管理员"}
            self.assertEqual((await client.post(url, json=malicious)).status_code, 422)
            self.assertEqual((await client.post(url, json=self.request(row).model_dump())).status_code, 200)
            self.assertEqual((await client.post(url, json=self.request(row).model_dump())).status_code, 200)
            self.assertEqual((await client.post(url, json=self.request(row, "keep_old").model_dump())).status_code, 409)

    async def test_display_binding_one_shot_expiry_and_reconnect(self):
        """候选未展示、令牌错误、过期、话题改变及新连接都不能凭短答采纳。"""
        row = await self.create_pending()
        delivery = MemoryPendingDelivery(self.service, self.session)
        payload = await delivery.poll()
        offer = payload["pending"][0]
        self.assertIsNone(delivery.consume("是的"))
        bad = MemoryPendingAck(pending_id=row["pending_id"], target_version=row["target_version"], delivery_token="wrong")
        self.assertFalse(delivery.acknowledge(bad))
        acknowledgement = MemoryPendingAck(**{key: offer[key] for key in ("pending_id", "target_version", "delivery_token")})
        self.assertTrue(delivery.acknowledge(acknowledgement))
        self.assertIsNone(delivery.consume("换个话题聊聊"))
        self.assertFalse(delivery.acknowledge(acknowledgement))
        self.assertIsNone(delivery.consume("是的"))
        restored = MemoryPendingDelivery(self.service, self.session)
        second = (await restored.poll())["pending"][0]
        self.assertNotEqual(second["delivery_token"], offer["delivery_token"])
        self.assertFalse(restored.acknowledge(acknowledgement))
        restored.acknowledge(MemoryPendingAck(**{key: second[key] for key in ("pending_id", "target_version", "delivery_token")}))
        restored.focus["shown_at"] -= 301
        self.assertIsNone(restored.consume("是的"))
        for text, hide in (("用新的？", False), ("是的", True)):
            fresh = MemoryPendingDelivery(self.service, self.session)
            item = (await fresh.poll())["pending"][0]
            ack = MemoryPendingAck(**{key: item[key] for key in ("pending_id", "target_version", "delivery_token")})
            fresh.acknowledge(ack)
            if hide:
                fresh.hide(ack)
                self.assertFalse(fresh.acknowledge(ack))
            self.assertIsNone(fresh.consume(text))

    async def test_unique_display_can_bind_short_reply_but_multiple_cannot(self):
        """唯一问题的确认可直接处理，多个待确认目标时仅允许按钮指定 ID。"""
        row = await self.create_pending()
        delivery = MemoryPendingDelivery(self.service, self.session)
        offer = (await delivery.poll())["pending"][0]
        self.assertIsNone(await delivery.poll())
        delivery.acknowledge(MemoryPendingAck(**{key: offer[key] for key in ("pending_id", "target_version", "delivery_token")}))
        confirmation = delivery.consume("是的。")
        self.assertEqual((confirmation.pending_id, confirmation.action), (row["pending_id"], "accept_new"))
        self.assertIsNone(delivery.consume("是的"))
        self.provider.facts = [fact("我喜欢分点回答")]
        self.assertEqual((await self.write("我喜欢分点回答", "fourth")).status, "pending")
        rows = (await delivery.poll())["pending"]
        self.assertEqual(len(rows), 2)
        self.assertFalse(delivery.acknowledge(MemoryPendingAck(**{key: rows[0][key] for key in ("pending_id", "target_version", "delivery_token")})))

    async def test_remaining_single_candidate_gets_fresh_display_token(self):
        """多个候选变为一个时重新展示，避免先前被拒绝的回执令下一项无法短答确认。"""
        first = await self.create_pending()
        self.provider.facts = [fact("我喜欢分点回答")]
        await self.write("我喜欢分点回答", "another_pending")
        delivery = MemoryPendingDelivery(self.service, self.session)
        before = (await delivery.poll())["pending"]
        remaining = next(row for row in before if row["pending_id"] != first["pending_id"])
        await self.service.resolve(self.user, first["pending_id"], self.request(first, "keep_old"))
        after = (await delivery.poll())["pending"][0]
        self.assertNotEqual(after["delivery_token"], remaining["delivery_token"])
        self.assertTrue(delivery.acknowledge(MemoryPendingAck(**{key: after[key] for key in ("pending_id", "target_version", "delivery_token")})))
        self.assertEqual(delivery.consume("用新的").pending_id, after["pending_id"])

    async def test_delete_one_pending_preserves_other_candidate(self):
        """单条候选删除撤销旧任务，但不让其他仍有效候选随全局 epoch 失效。"""
        first = await self.create_pending()
        self.provider.facts = [fact("我喜欢分点回答")]
        await self.write("我喜欢分点回答", "another_pending")
        self.assertTrue(await self.repo.delete_pending(self.user, first["pending_id"]))
        remaining = (await self.repo.pending(self.user))[0]
        self.assertEqual((await self.service.resolve(self.user, remaining["pending_id"], self.request(remaining))).status, "saved")

    async def test_accept_invalidates_derived_state_but_preserves_chat_ledger(self):
        """新事实生效后旧检查点与执行被撤销，原始聊天证据保留。"""
        row = await self.create_pending()
        self.agent = AgentGraph(self.provider, memory=self.service)
        self.sessions.forget_checkpoint = self.agent.working.forget
        self.sessions.invalidate_checkpoint = self.agent.working.invalidate
        self.session.begin_generation("g", "t", "原始对话")
        await self.agent.prepare_turn(self.session, "g")
        self.assertTrue(self.agent.working.read(self.session.session_id))
        await self.service.resolve(self.user, row["pending_id"], self.request(row))
        await asyncio.sleep(0)
        self.assertEqual(self.agent.working.read(self.session.session_id), [])
        self.assertEqual(self.session.outputs["g"].user_text, "原始对话")
        self.assertIsNone(self.session.generation_id)

    async def test_websocket_pending_notification_ack_and_chat_resolution(self):
        """生产网关展示冲突后用唯一短答提交，无需意图识别或第二次模型提取。"""
        await self.create_pending()
        self.agent = AgentGraph(self.provider, memory=self.service)
        self.sessions.forget_checkpoint = self.agent.working.forget
        classifier = SimpleNamespace(classify=AsyncMock(return_value=intent("chat")))
        app = SimpleNamespace(state=SimpleNamespace(sessions=self.sessions, agent=self.agent, memory=self.service,
            intent_service=classifier, settings=SimpleNamespace(llm_ready=True, tts_ready=False, stop_ack_timeout_ms=100),
            tts=FakeTTS(), speech_service=FakeSpeech()))
        socket = FakeSocket(app, self.session)
        task = asyncio.create_task(realtime(socket, self.session.session_id))
        try:
            await socket.until("session.status")
            offer = (await socket.until("memory.pending"))["payload"]["pending"][0]
            await socket.send("memory.pending.displayed", {key: offer[key] for key in ("pending_id", "target_version", "delivery_token")})
            calls = len(self.provider.messages)
            await socket.send("input.text", {"text": "是的"}, "confirmation")
            result = await socket.until("response.completed")
            self.assertIn("已采用新信息", result["payload"]["text"])
            await socket.until("turn.delivered")
            self.assertEqual(len(self.provider.messages), calls)
            classifier.classify.assert_not_awaited()
            self.assertEqual((await self.repo.snapshot(self.user))[1][0].content, "我喜欢详细回答")
        finally:
            await socket.close()
            await asyncio.wait_for(task, 3)
