"""作用：验证一次性提醒的事务、时间解释、Agent 规划和实际投递闭环。

当前功能：
- 使用临时 SQLite 验证用户隔离、持久化、幂等、取消和租约恢复。
- 验证中文相对时间、跨日、歧义和夏令时拒绝策略。
- 通过真实分类、LangGraph、任务规划和 WebSocket 协程验证集成行为。
- 区分已提交但未确认的写入、已展示通知及可打断的语音输出。
"""

import asyncio
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from threading import Event
import httpx
from fastapi import FastAPI

from src.agent.graph import AgentGraph
from src.agent.planner import TaskPlanner
from src.api.realtime import realtime
from src.api import sessions as sessions_api
from src.contracts.tools import ToolCallRequest, ToolContext, PendingClarification
from src.contracts.agent import PlanDraft
from src.gateway.identity import COOKIE_NAME
from src.gateway.reminders import ReminderDelivery
from src.gateway.sessions import SessionRegistry
from src.intent.service import IntentService
from src.repositories.memory import MemoryRepository, token_digest
from src.repositories.reminders import ReminderRepository
from src.tools.builtin.reminder_time import resolve_due_at
from src.tools.builtin.reminders import register_reminder_tools, CREATE_REMINDER_SPEC
from src.tools.builtin.time import TIME_SPEC, CurrentTimeTool
from src.tools.builtin.datetime import DATETIME_SPEC, convert_datetime
from src.tools.executor import ToolExecutor, ToolFailure
from src.tools.registry import ToolRegistry
from test_intent import estimate
from test_planning import ScenarioProvider
from test_p3 import FakeSocket, FakeSpeech, FakeTTS


class ReminderFixture(unittest.IsolatedAsyncioTestCase):
    """为存储与集成测试提供隔离数据库和真实内置处理函数。"""

    async def asyncSetUp(self):
        """每个场景创建一个临时用户和 SQLite 文件。"""
        self.temp = tempfile.TemporaryDirectory()
        self.database = MemoryRepository(str(Path(self.temp.name) / "test.sqlite3"))
        self.user_id, self.token = await self.database.identity(None, create=True)
        self.repository = ReminderRepository(self.database)
        self.registry = ToolRegistry()
        register_reminder_tools(self.registry, self.repository, "Asia/Shanghai")
        self.executor = ToolExecutor(self.registry)
        self.reference = datetime.now(timezone.utc)
        self.context = ToolContext(self.user_id, "session", "session:turn", "十分钟后提醒我休息", self.reference)

    async def asyncTearDown(self):
        """所有线程事务退出后删除临时数据库。"""
        self.temp.cleanup()

    async def invoke(self, name="create_reminder", arguments=None, context=None, executor=None):
        """通过生产执行器调用工具，所有截止时间使用带时区的服务端时钟。"""
        call = ToolCallRequest(call_id="call", tool_name=name, arguments=arguments if arguments is not None else {"title": "休息", "due_at": "十分钟后"},
                               deadline_at=datetime.now(timezone.utc) + timedelta(seconds=3))
        return await (executor or self.executor).execute(call, context or self.context)

    async def seed_due(self):
        """模拟模型处理期间已到期的提醒，后续仍应可靠投递。"""
        context = replace(self.context, request_text="一秒后提醒我休息", reference_at=self.reference - timedelta(seconds=10))
        result = await self.invoke(arguments={"title": "休息", "due_at": "一秒后"}, context=context)
        self.assertTrue(result.verified)
        return result.data


class ReminderTimeTests(unittest.TestCase):
    """验证时间解释由确定性代码完成，不依赖模型日期推断。"""

    def test_relative_chinese_and_cross_day(self):
        """分钟、半小时和明早按同一个请求基准计算。"""
        reference = datetime.fromisoformat("2026-10-15T15:55:00+00:00")
        self.assertEqual(resolve_due_at("十分钟后", "Asia/Shanghai", reference).isoformat(), "2026-10-15T16:05:00+00:00")
        self.assertEqual(resolve_due_at("半小时后", "Asia/Shanghai", reference), reference + timedelta(minutes=30))
        self.assertEqual(resolve_due_at("明早八点", "Asia/Shanghai", reference).isoformat(), "2026-10-16T00:00:00+00:00")
        self.assertEqual(resolve_due_at("明天下午三点半", "Asia/Shanghai", reference).isoformat(), "2026-10-16T07:30:00+00:00")
        self.assertEqual(resolve_due_at("2026-10-17 15:00", "Asia/Shanghai", reference).isoformat(), "2026-10-17T07:00:00+00:00")
        self.assertEqual(resolve_due_at("明天09:00", "Asia/Shanghai", reference).isoformat(), "2026-10-16T01:00:00+00:00")

    def test_ambiguous_invalid_past_and_dst_require_clarification(self):
        """不猜上午下午、无效日期、隐式跨天或夏令时重叠。"""
        reference = datetime.fromisoformat("2026-10-15T00:00:00+00:00")
        for text in ("明天三点", "今天早上七点", "零分钟后", "每天八点", "2026年2月30日下午三点", "明天下午十五点"):
            with self.subTest(text=text), self.assertRaises(ToolFailure) as error:
                resolve_due_at(text, "Asia/Shanghai", reference)
            self.assertEqual(error.exception.error.missing_slots, ["due_at"])
        with self.assertRaises(ToolFailure):
            resolve_due_at("2026-11-01凌晨一点半", "America/New_York", reference)


class ReminderStorageTests(ReminderFixture):
    """验证真实事务及执行器对业务副作用的处理。"""

    async def test_create_idempotence_conflict_persistence_and_identity(self):
        """同轮重放返回同一编号，变参冲突，其他用户不能查询或取消。"""
        first, repeated = await self.invoke(), await self.invoke()
        self.assertTrue(first.verified)
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(datetime.fromisoformat(first.data["due_at"]), self.reference + timedelta(minutes=10))
        conflict = await self.invoke(arguments={"title": "我休息", "due_at": "十分钟后"})
        self.assertEqual(conflict.error.code, "operation_conflict")
        reopened = ReminderRepository(MemoryRepository(self.database.path))
        self.assertEqual(len((await reopened.list_pending(self.user_id))["reminders"]), 1)
        other, _ = await self.database.identity(None, create=True)
        self.assertEqual((await reopened.list_pending(other))["reminders"], [])
        target = first.data["reminder_id"]
        context = replace(self.context, user_id=other, operation_id="other", request_text="取消提醒 " + target)
        result = await self.invoke("cancel_reminder", {"target": target}, context)
        self.assertFalse(result.verified)
        self.assertEqual((await reopened.get(self.user_id, target))["status"], "pending")
        request = ToolCallRequest(call_id="missing", tool_name="list_reminders", arguments={}, deadline_at=self.reference + timedelta(seconds=10))
        self.assertEqual((await self.executor.execute(request)).error.code, "identity_required")

    async def test_identity_works_when_long_term_memory_is_disabled(self):
        """提醒可单独启用，关闭长期记忆不会失去浏览器持久身份。"""
        app = FastAPI()
        app.include_router(sessions_api.router)
        app.state.sessions = SessionRegistry()
        app.state.memory = None
        app.state.database = self.database
        app.state.reminders = self.repository
        app.state.speech_service = FakeSpeech()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = (await client.post("/v1/sessions")).json()
            second = (await client.post("/v1/sessions")).json()
            self.assertEqual(first["user_id"], second["user_id"])
            self.assertFalse(first["feature_flags"]["long_term_memory"])
            self.assertTrue(first["feature_flags"]["reminders"])
            self.assertIn(COOKIE_NAME, client.cookies)

    async def test_cancel_latest_is_session_scoped_and_not_sent(self):
        """刚才不会取消另一会话记录，重复取消幂等且取消后不会被调度。"""
        row = await self.seed_due()
        wrong = replace(self.context, session_id="other-session", operation_id="wrong", request_text="取消刚才的提醒")
        self.assertFalse((await self.invoke("cancel_reminder", {"target": "latest"}, wrong)).verified)
        correct = replace(wrong, session_id="session", operation_id="cancel")
        cancelled = await self.invoke("cancel_reminder", {"target": "latest"}, correct)
        self.assertEqual(cancelled.data["status"], "cancelled")
        self.assertEqual((await self.invoke("cancel_reminder", {"target": "latest"}, correct)).data, cancelled.data)
        self.assertEqual(await self.repository.claim(self.user_id, "socket"), [])
        self.assertEqual((await self.repository.get(self.user_id, row["reminder_id"]))["status"], "cancelled")

    async def test_delivery_lease_ack_owner_and_reconnect(self):
        """并发连接只领取一次；未确认的连接关闭后补发，旧凭据不能确认新投递。"""
        row = await self.seed_due()
        left = ReminderDelivery(self.repository, self.user_id, "left")
        right = ReminderDelivery(self.repository, self.user_id, "right")
        a, b = await asyncio.gather(left.poll(), right.poll())
        winner, loser = (left, right) if a else (right, left)
        first = (a or b)[0]
        self.assertEqual(len(a) + len(b), 1)
        self.assertFalse(await loser.acknowledge(row["reminder_id"], first["delivery_token"]))
        await winner.close()
        second = (await loser.poll())[0]
        self.assertNotEqual(first["delivery_token"], second["delivery_token"])
        self.assertFalse(await winner.acknowledge(row["reminder_id"], first["delivery_token"]))
        self.assertTrue(await loser.acknowledge(row["reminder_id"], second["delivery_token"]))
        self.assertFalse(await loser.acknowledge(row["reminder_id"], second["delivery_token"]))
        self.assertIsNotNone(await loser.next_speech())
        self.assertIsNone(await loser.next_speech())
        self.assertEqual(await self.repository.claim(self.user_id, "next"), [])
        self.assertEqual((await self.repository.list_pending(self.user_id))["reminders"], [])

    async def test_expired_lease_after_process_loss_is_reclaimed(self):
        """进程未能释放租约时，过期后新连接补发且旧投递无法覆盖新回执。"""
        row = await self.seed_due()
        first = (await self.repository.claim(self.user_id, "crashed", self.reference))[0]
        self.assertEqual(await self.repository.claim(self.user_id, "next", self.reference + timedelta(seconds=29)), [])
        second = (await self.repository.claim(self.user_id, "next", self.reference + timedelta(seconds=31)))[0]
        self.assertFalse(await self.repository.acknowledge(self.user_id, "crashed", row["reminder_id"], first["delivery_token"]))
        self.assertTrue(await self.repository.acknowledge(self.user_id, "next", row["reminder_id"], second["delivery_token"]))

    async def test_commit_survives_output_timeout_without_duplicate(self):
        """提交后响应超时提示先核查，重试同一操作仍只保留一条记录。"""
        original = self.registry.get("create_reminder").handler
        registry = ToolRegistry()

        async def delayed(arguments, context):
            """提交真实事务后模拟确认过程阻塞。"""
            result = await original(arguments, context)
            await asyncio.sleep(.2)
            return result

        registry.register(CREATE_REMINDER_SPEC, delayed, needs_context=True)
        result = await self.invoke(executor=ToolExecutor(registry, 100))
        self.assertEqual(result.error.code, "write_unconfirmed")
        rows = (await self.repository.list_pending(self.user_id))["reminders"]
        self.assertEqual(len(rows), 1)
        self.assertEqual((await self.invoke()).data["reminder_id"], rows[0]["reminder_id"])

    async def test_cancel_before_commit_rolls_back_transaction(self):
        """写线程尚未提交时取消，释放线程后确认没有迟到记录。"""
        started, release, closed = Event(), Event(), Event()

        def blocked(db):
            """模拟事务持锁后等待，事务设施必须在提交前观察取消。"""
            db.execute("INSERT INTO reminders(reminder_id,user_id,source_session_id,title,due_at,timezone,created_at) VALUES ('late',?,'session','迟到','2099','UTC','now')", (self.user_id,))
            started.set()
            release.wait(2)
            closed.set()

        task = asyncio.create_task(self.database.run(blocked))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.to_thread(closed.wait, 1)
        self.assertIsNone(await self.repository.get(self.user_id, "late"))

    async def test_unrequested_or_invented_write_is_rejected(self):
        """历史和模型槽位不能单独授予新提醒写入权限。"""
        for text in ("查一下天气", "不要提醒我休息", "删除提醒"):
            result = await self.invoke(context=replace(self.context, request_text=text))
            self.assertFalse(result.verified)
        invented = await self.invoke(arguments={"title": "喝水", "due_at": "十分钟后"})
        self.assertEqual(invented.error.missing_slots, ["title"])
        self.assertEqual((await self.repository.list_pending(self.user_id))["reminders"], [])

    async def test_reminder_title_can_contain_cancel_action(self):
        """提醒取消订阅属于新建事项，不误判为取消现有提醒。"""
        context = replace(self.context, request_text="十分钟后提醒我取消订阅")
        result = await self.invoke(arguments={"title": "取消订阅", "due_at": "十分钟后"}, context=context)
        self.assertTrue(result.verified)


class ReminderAgentTests(ReminderFixture):
    """验证意图识别、单次澄清与多工具写入计划。"""

    async def run_turn(self, provider, text, generation="g1", pending=None, plan=None, agent=None):
        """运行真实分类和图，可信身份、轮次和时间从网关等价输入传入。"""
        intent = await IntentService(provider).classify(text, "", self.registry.list_tools(), source="text", pending_clarification=pending)
        runtime = {"social_state": {"user_id": self.user_id, "session_id": "session", "intent": intent.model_dump()},
                   "turn_id": generation, "request_received_at": self.reference.isoformat(),
                   "pending_clarification": pending.model_dump() if pending else None, "active_plan": plan}
        graph = agent or AgentGraph(provider, self.executor)
        return [item async for item in graph.stream(text, [], [], generation, runtime)]

    async def test_single_reminder_clarification_preserves_operation_and_reference(self):
        """上午下午歧义先追问，补参后保存且仍属于原始业务请求。"""
        provider = ScenarioProvider()
        provider.intent = estimate("tool_request", tool_intent="create_reminder", slots=[
            {"name": "title", "value": "休息", "evidence": "休息"}, {"name": "due_at", "value": "明天三点", "evidence": "明天三点"}])
        graph = AgentGraph(provider, self.executor)
        first = await self.run_turn(provider, "明天三点提醒我休息", agent=graph)
        pending = PendingClarification.model_validate([event["payload"]["pending"] for event in first if event["type"] == "tool.pending"][-1])
        self.assertEqual(pending.missing_slots, ["due_at"])
        self.assertEqual(pending.execution_id, "g1")
        provider.intent = estimate("tool_request", tool_intent="create_reminder", continues_pending=True, slots=[
            {"name": "due_at", "value": "明天下午三点", "evidence": "明天下午三点"}])
        second = await self.run_turn(provider, "明天下午三点", "g2", pending=pending, agent=graph)
        self.assertIn("已创建一次性提醒", second[-1]["payload"]["text"])
        self.assertEqual(len((await self.repository.list_pending(self.user_id))["reminders"]), 1)

    async def test_dependent_time_conversion_to_reminder_and_repeat(self):
        """不同工具字段依赖可以驱动提醒写入；重复只复述原结果。"""
        self.registry.register(TIME_SPEC, CurrentTimeTool())
        self.registry.register(DATETIME_SPEC, convert_datetime)
        text = "查询 UTC 当前时间，转换成 Asia/Shanghai 的明天时间，按转换后的时间提醒我出门。"
        draft = {"goal": "按转换后的时间提醒出门", "steps": [
            {"kind": "tool", "tool_name": "get_current_time", "arguments": {"timezone": "UTC"}, "argument_evidence": {"timezone": "UTC"}},
            {"kind": "tool", "tool_name": "convert_datetime", "arguments": {"timezone": "Asia/Shanghai", "days_offset": 1},
             "argument_evidence": {"timezone": "Asia/Shanghai", "days_offset": "明天"}, "bindings": {"iso_time": {"step": 1, "path": "/iso_time"}}},
            {"kind": "tool", "tool_name": "create_reminder", "arguments": {"title": "出门"}, "argument_evidence": {"title": "出门"},
             "bindings": {"due_at": {"step": 2, "path": "/iso_time"}, "timezone": {"step": 2, "path": "/timezone"}}}, {"kind": "respond"}]}
        provider = ScenarioProvider(draft)
        provider.answer = {"text": "已按指定的转换时间安排提醒。", "result_refs": [1, 2, 3]}
        graph = AgentGraph(provider, self.executor)
        events = await self.run_turn(provider, text, agent=graph)
        plan = [event["payload"]["plan"] for event in events if event["type"] == "plan.updated"][-1]
        self.assertEqual(plan["status"], "completed")
        self.assertEqual(plan["steps"][2]["result"]["data"]["status"], "pending")
        self.assertEqual(len((await self.repository.list_pending(self.user_id))["reminders"]), 1)
        runtime = {"social_state": {"user_id": self.user_id, "session_id": "session", "intent": estimate("control", control_action="repeat").model_dump()},
                   "interrupted_plan": plan}
        repeated = [event async for event in graph.stream("重复一遍", [], [], "repeat", runtime)]
        self.assertIn("已创建一次性提醒", repeated[-1]["payload"]["text"])
        self.assertEqual(len((await self.repository.list_pending(self.user_id))["reminders"]), 1)
        with self.assertRaises(ValueError):
            TaskPlanner(provider).validate(PlanDraft.model_validate(draft), "查询 UTC 时间并转换成 Asia/Shanghai 的明天时间", text, self.registry.list_tools())


class ReminderGatewayTests(ReminderFixture):
    """通过生产网关检查离线补发、展示回执、空闲播报和停止。"""

    async def test_due_display_ack_idle_speech_stop_and_reconnect(self):
        """离线到期后连接展示，澄清期间不播报，停止不改变提醒业务状态。"""
        row = await self.seed_due()
        sessions = SessionRegistry()
        session = sessions.create(self.user_id, token_digest(self.token))
        provider = ScenarioProvider()
        state = SimpleNamespace(sessions=sessions, reminders=self.repository, tool_registry=self.registry,
                                agent=AgentGraph(provider, self.executor), intent_service=IntentService(provider),
                                settings=SimpleNamespace(llm_ready=True, tts_ready=True, stop_ack_timeout_ms=2000),
                                tts=FakeTTS(), speech_service=FakeSpeech())
        ws = FakeSocket(SimpleNamespace(state=state), session)
        ws.cookies = {COOKIE_NAME: self.token}
        task = asyncio.create_task(realtime(ws, session.session_id))
        try:
            await ws.until("session.status")
            notice = await ws.until("reminder.due")
            self.assertEqual(notice["payload"]["reminder_id"], row["reminder_id"])
            self.assertEqual((await self.repository.get(self.user_id, row["reminder_id"]))["status"], "pending")
            session.pending_clarification = PendingClarification(original_text="查询", tool_name="get_weather", arguments={}, missing_slots=["location"], question="地点？")
            await ws.send("reminder.displayed", {"reminder_id": row["reminder_id"], "delivery_token": notice["payload"]["delivery_token"]})
            await asyncio.sleep(1.1)
            self.assertTrue(ws.outgoing.empty())
            self.assertEqual((await self.repository.get(self.user_id, row["reminder_id"]))["status"], "delivered")
            session.pending_clarification = None
            speech = await ws.until("reminder.speech")
            chunk = await ws.until("tts.chunk")
            self.assertEqual(speech["generation_id"], chunk["generation_id"])
            self.assertEqual(provider.calls, [])
            await ws.send("control.stop", {"generation_id": chunk["generation_id"]})
            stopped = await ws.until("output.cancel")
            self.assertEqual(stopped["generation_id"], chunk["generation_id"])
        finally:
            await ws.close()
            await asyncio.wait_for(task, 3)
        self.assertEqual(await self.repository.claim(self.user_id, "reconnected"), [])


if __name__ == "__main__":
    unittest.main()
