"""作用：验证分级门控、异步整合与阈值压缩的实际数据边界。

当前功能：
- 通过真实 SQLite 检查多事实原子提交、语义去重、冲突暂存及项目范围。
- 检查后台任务非阻塞、重启恢复、删除代次与可信工具来源。
- 使用生产 Checkpointer 验证落盘、TTL、状态变化和摘要覆盖范围。
"""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time
from contextlib import closing

from src.agent.graph import AgentGraph
from src.contracts.memory import ExtractedFact, MemoryCandidate, MemoryEvidence
from src.memory.gating import eligible, operation
from src.memory.service import MemoryService
from src.memory.working import WorkingMemory
from src.repositories.memory import MemoryRepository
from test_memory import MemoryProvider


def fact(content="我喜欢简短回答", **updates):
    """构造有原文证据的提取结果，允许覆盖安全边界字段。"""
    return ExtractedFact.model_validate({"should_write": True, "memory_type": "preference", "content": content,
        "evidence": content, "confidence": .98, "is_temporary": False, "fact_key": "user.reply_style", **updates})


class GateProvider(MemoryProvider):
    """为提取和压缩提供确定数据，用可释放事件模拟慢模型。"""

    def __init__(self):
        """初始化多事实输出、摘要和模型挂起控制。"""
        super().__init__()
        self.facts = [fact()]
        self.digest = {"goal": "帮我写论文", "constraints": [], "pending": [], "evidence": []}
        self.release = asyncio.Event()

    async def stream(self, messages):
        """按专用 Schema 返回数据，其余分支复用已有实际图测试替身。"""
        prompt = messages[0]["content"]
        if "你负责提取长期原子事实" in prompt:
            self.messages.append(messages)
            self.started.set()
            if self.block:
                await self.release.wait()
            yield json.dumps({"facts": [item.model_dump() for item in self.facts]}, ensure_ascii=False)
        elif "压缩历史为任务工作上下文" in prompt:
            self.messages.append(messages)
            yield json.dumps(self.digest, ensure_ascii=False)
        else:
            async for chunk in super().stream(messages):
                yield chunk


class MemoryGateTests(unittest.IsolatedAsyncioTestCase):
    """验证真实事务与工作器，不依赖外部模型或真实用户数据库。"""

    async def asyncSetUp(self):
        """为每项测试建立临时持久化文件和可信用户。"""
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "memory.sqlite3")
        self.repo = MemoryRepository(self.path)
        self.user, _ = await self.repo.identity(None, create=True)
        self.provider = GateProvider()
        self.service = MemoryService(self.repo, self.provider)

    async def asyncTearDown(self):
        """关闭后台任务后清理临时文件。"""
        await self.service.close()
        self.temp.cleanup()

    async def write(self, text="我喜欢简短回答", turn="t1", **kwargs):
        """调用生产同步整合入口并使用真实截止时间。"""
        return await self.service.update(text, self.user, "session", turn, turn,
            datetime.now(timezone.utc) + timedelta(seconds=10), **kwargs)

    async def test_prefilter_rejects_temporary_uncertain_quoted_and_negative_instructions(self):
        """无记忆价值或缺少实际授权的消息不会发起事实抽取。"""
        texts = ["今天心情不好", "今晚想吃火锅", "我以后可能换工作", "他说：我喜欢简短回答",
                 "不要记住我喜欢简短回答", "如何删除我的全部记忆", "不要删除我的全部记忆"]
        for text in texts:
            with self.subTest(text=text):
                self.assertFalse(eligible(text))
                self.assertIsNone(operation(text))
        self.assertIsNone(await self.service.observe("今天心情不好", self.user, "s", "t", "g", datetime.now(timezone.utc)))
        self.assertEqual(self.provider.messages, [])

    async def test_temporary_and_stable_facts_in_same_message(self):
        """临时情绪不升级，另一分句中有证据的稳定偏好可以保存。"""
        result = await self.write("今天心情不好，我喜欢简短回答")
        self.assertEqual(result.status, "saved")
        self.assertEqual((await self.repo.snapshot(self.user))[1][0].content, "我喜欢简短回答")

    async def test_model_cannot_hide_qualifiers_or_promote_temporary_emotion_as_event(self):
        """证据所在句保留推测限定，模型标成事件也不能保存临时情绪。"""
        for text, proposed in [("记住：我可能喜欢简短回答", fact("喜欢简短回答")),
                               ("记住：我不喜欢简短回答", fact("喜欢简短回答")),
                               ("记住：今天心情不好", fact("今天心情不好", memory_type="event", fact_key="event.mood"))]:
            self.provider.facts = [proposed]
            self.assertEqual((await self.write(text)).status, "rejected")
        self.assertEqual((await self.repo.snapshot(self.user))[1], [])

    async def test_important_events_keep_temporal_evidence_when_content_repeats(self):
        """同一事件表述在不同时间发生，不能只凭正文相同跳过新事件。"""
        self.provider.facts = [fact("完成了重要实验", memory_type="event", fact_key="event.first_experiment",
                                    evidence="今天完成了重要实验")]
        self.assertEqual((await self.write("记住：今天完成了重要实验")).decisions, ["ADD"])
        self.provider.facts = [fact("完成了重要实验", memory_type="event", fact_key="event.second_experiment",
                                    evidence="昨天完成了重要实验")]
        self.provider.relation = "different"
        self.assertEqual((await self.write("记住：昨天完成了重要实验", "another_event")).decisions, ["ADD"])
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 2)

    async def test_multiple_atomic_facts_and_invalid_member_roll_back(self):
        """多事实整组提交；任一事实伪造证据时整组都不写入。"""
        self.provider.facts = [fact(), fact("我住在杭州", memory_type="fact", fact_key="user.home_city")]
        result = await self.write("记住：我喜欢简短回答；我住在杭州")
        self.assertEqual(result.decisions, ["ADD", "ADD"])
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 2)
        self.provider.facts = [fact("我研究机器学习", fact_key="user.research_topic"), fact("我叫张三", fact_key="user.preferred_name")]
        self.assertEqual((await self.write("记住：我研究机器学习", "bad")).status, "failed")
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 2)

    async def test_semantic_duplicate_skips_but_similar_different_fact_adds(self):
        """同义表述去重；高相关但独立的偏好在比较后仍新增。"""
        await self.write()
        self.provider.facts = [fact("我偏好简短回答")]
        self.provider.relation = "same"
        self.assertEqual((await self.write("我偏好简短回答", "paraphrase")).decisions, ["SKIP"])
        self.provider.facts = [fact("我喜欢语音回答", fact_key="user.answer_medium")]
        self.provider.relation = "different"
        self.assertEqual((await self.write("我喜欢语音回答", "different")).decisions, ["ADD"])
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 2)

    async def test_pending_is_not_retrieved_and_explicit_update_versions_old_fact(self):
        """含糊冲突不覆盖事实；明确更正后创建版本并清理关联候选。"""
        await self.write()
        self.provider.facts = [fact("我喜欢详细回答")]
        self.assertEqual((await self.write("我喜欢详细回答", "ambiguous")).decisions, ["PENDING"])
        pending = await self.repo.pending(self.user)
        self.assertEqual(len(pending), 1)
        self.assertEqual((await self.repo.snapshot(self.user))[1][0].content, "我喜欢简短回答")
        self.assertEqual((await self.write("我现在改为：我喜欢详细回答", "confirmed")).decisions, ["UPDATE"])
        self.assertEqual(await self.repo.pending(self.user), [])
        records = (await self.repo.snapshot(self.user, historical=True))[1]
        self.assertEqual({record.version for record in records}, {1, 2})

    async def test_same_key_in_different_projects_keeps_both_decisions(self):
        """项目范围参与事实唯一键和匹配，同名属性不会跨项目覆盖。"""
        for name, value in [("Alpha", "SQLite"), ("Beta", "PostgreSQL")]:
            text = f"{name}项目确定采用{value}"
            self.provider.facts = [fact(text, memory_type="decision", fact_key="user.database", domain_scope=f"project:{name}")]
            self.assertEqual((await self.write(text, name)).decisions, ["ADD"])
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 2)

    async def test_background_does_not_block_normal_reply(self):
        """后台提取挂起时，普通回复仍可从同一 Agent 图交付。"""
        self.provider.block = True
        agent = AgentGraph(self.provider, memory=self.service)
        try:
            events = [event async for event in agent.stream("我喜欢简短回答", [], [], "g", {
                "social_state": {"user_id": self.user, "session_id": "s"}, "turn_id": "t"})]
            self.assertEqual(events[-1]["payload"]["text"], "根据已有记录回答。")
            self.assertTrue(any(event["type"] == "memory.result" and event["payload"]["status"] == "queued" for event in events))
            await asyncio.wait_for(self.provider.started.wait(), 1)
            self.assertEqual((await self.repo.snapshot(self.user))[1], [])
        finally:
            agent.working.close()

    async def test_queue_persists_across_service_restart_and_is_idempotent(self):
        """未执行任务从同一文件恢复，重复领取不能生成多条有效事实。"""
        payload = {"text": "我喜欢简短回答", "session_id": "s", "turn_id": "t", "generation_id": "g"}
        self.assertTrue(await self.repo.enqueue(self.user, "s:t", payload))
        restarted = MemoryService(MemoryRepository(self.path), self.provider)
        self.assertTrue(await restarted.process_once())
        self.assertFalse(await restarted.process_once())
        await self.repo.enqueue(self.user, "s:t", payload)
        await restarted.process_once()
        self.assertEqual(len((await self.repo.snapshot(self.user))[1]), 1)

    async def test_delete_epoch_blocks_already_running_extraction(self):
        """删除发生在模型提取期间，释放旧模型结果后仍无法复活事实。"""
        self.provider.block = True
        await self.repo.enqueue(self.user, "s:t", {"text": "我喜欢简短回答", "session_id": "s", "turn_id": "t", "generation_id": "g"})
        task = asyncio.create_task(self.service.process_once())
        await asyncio.wait_for(self.provider.started.wait(), 1)
        revision, _ = await self.repo.snapshot(self.user)
        await self.repo.commit(self.user, "delete", MemoryCandidate(action="delete", delete_all=True), revision, "s", "delete")
        self.provider.release.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual((await self.repo.snapshot(self.user))[1], [])
        self.assertFalse(await self.service.process_once())

    async def test_only_verified_tool_sources_are_accepted(self):
        """工具证据必须经服务端验证，入库后保留具体调用来源。"""
        self.provider.facts = [fact("实验准确率为92%", source_id="tool:call1", memory_type="fact", fact_key="user.experiment")]
        source = MemoryEvidence(source_id="tool:call1", text="实验准确率为92%")
        self.assertEqual((await self.write("记住这条实验结果", tool_evidence=(source,))).status, "failed")
        source.verified = True
        self.assertEqual((await self.write("记住这条实验结果", "verified", tool_evidence=(source,))).status, "saved")
        record = (await self.repo.snapshot(self.user))[1][0]
        self.assertEqual((record.source_kind, record.source_ref), ("tool", "tool:call1"))

    async def test_queue_limit_expiry_pending_ownership_and_retry_budget(self):
        """队列满时拒绝，过期清理，待确认项按所有权删除，失败不无限重试。"""
        payload = {"text": "我喜欢简短回答", "session_id": "s", "turn_id": "t", "generation_id": "g"}
        self.assertTrue(await self.repo.enqueue(self.user, "s:t", payload, limit=1))
        self.assertFalse(await self.repo.enqueue(self.user, "s:t2", payload, limit=1))
        with closing(self.repo.connect()) as db:
            db.execute("UPDATE memory_jobs SET attempts=3,available_at=0")
            db.commit()
        self.assertIsNone(await self.repo.claim_job())
        await self.write()
        self.provider.facts = [fact("我喜欢详细回答")]
        await self.write("我喜欢详细回答", "pending")
        pending = (await self.repo.pending(self.user))[0]
        other, _ = await self.repo.identity(None, create=True)
        self.assertFalse(await self.repo.delete_pending(other, pending["pending_id"]))
        with closing(self.repo.connect()) as db:
            db.execute("UPDATE memory_pending SET expires_at=0")
            db.commit()
        self.assertEqual(await self.repo.pending(self.user), [])

    async def test_legacy_database_migrates_without_losing_existing_fact(self):
        """旧版表自动补充元数据和任务表，已有正文与版本不受影响。"""
        legacy = str(Path(self.temp.name) / "legacy.sqlite3")
        with closing(sqlite3.connect(legacy)) as db:
            db.executescript("""
                CREATE TABLE memory_users(user_id TEXT PRIMARY KEY,token_hash TEXT UNIQUE NOT NULL,revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE memory_records(memory_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,persona_id TEXT NOT NULL,
                    kind TEXT NOT NULL,fact_key TEXT NOT NULL,content TEXT NOT NULL,evidence TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,source_turn_id TEXT NOT NULL,version INTEGER NOT NULL,
                    status TEXT NOT NULL,created_at TEXT NOT NULL,valid_to TEXT,supersedes_id TEXT);
                CREATE TABLE memory_operations(user_id TEXT NOT NULL,operation_id TEXT NOT NULL,result TEXT NOT NULL,
                    PRIMARY KEY(user_id,operation_id));
                INSERT INTO memory_users VALUES ('legacy','hash',2);
                INSERT INTO memory_records VALUES ('old','legacy','mira','semantic','user.reply_style','我喜欢简短回答',
                    '我喜欢简短回答','session','turn',2,'active','2026-10-01T00:00:00+00:00',NULL,NULL);
            """)
            db.commit()
        migrated = MemoryRepository(legacy)
        revision, records = await migrated.snapshot("legacy")
        self.assertEqual((revision, records[0].version, records[0].content), (2, 2, "我喜欢简短回答"))
        self.assertEqual(records[0].domain_scope, "daily")
        self.assertEqual(await migrated.pending("legacy"), [])


class WorkingGateTests(unittest.IsolatedAsyncioTestCase):
    """验证工作状态门控和生产 SQLite 检查点的可恢复性。"""

    async def test_state_changes_only_for_tasks_constraints_and_verified_results(self):
        """无状态变化不增加版本；新任务清除旧约束，工具结果必须 verified。"""
        first = WorkingMemory.update_task({}, "帮我写论文")
        same = WorkingMemory.update_task(first, "你好")
        self.assertEqual(first, same)
        constrained = WorkingMemory.update_task(same, "必须使用中文")
        self.assertEqual(constrained["version"], first["version"] + 1)
        next_task = WorkingMemory.update_task(constrained, "新任务：帮我做报告")
        self.assertEqual(next_task["constraints"], [])
        self.assertEqual(WorkingMemory.update_task(next_task, tool_results=[{"verified": False, "status": "ok"}]), next_task)
        verified = WorkingMemory.update_task(next_task, tool_results=[{"verified": True, "status": "ok", "call_id": "c", "data": {"value": 1}}])
        self.assertIn('"call_id": "c"', verified["evidence"][0])

    async def test_checkpoint_reopens_and_ttl_survives_restart(self):
        """进程对象重建后同一 thread 可读，持久化 TTL 到期后实际清除。"""
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoints.sqlite3")
            working = WorkingMemory(path=path)
            agent = AgentGraph(GateProvider(), working=working)
            _ = [event async for event in agent.stream("需要恢复的原始输入", [], [], "g", {"social_state": {"session_id": "s"}})]
            working.close()
            restored = WorkingMemory(path=path)
            self.assertEqual(restored.state("s")["user_text"], "需要恢复的原始输入")
            with restored.checkpointer.cursor() as cur:
                cur.execute("UPDATE memory_threads SET touched=0")
            restored.close()
            expired = WorkingMemory(path=path)
            self.assertEqual(expired.read("s"), [])
            self.assertIsNone(expired.checkpointer.get_tuple({"configurable": {"thread_id": "s"}}))
            expired.close()

    async def test_threshold_summary_preserves_sources_and_prevents_rehydration(self):
        """未超阈值没有摘要调用，压缩后旧账本不会再次灌回历史。"""
        working = WorkingMemory(token_threshold=512)
        provider = GateProvider()
        try:
            now = time()
            short = [{"generation_id": "small", "user": "你好", "assistant": "收到", "created_at": now}]
            self.assertEqual((await working.compress(provider, short, {}))[0], short)
            self.assertEqual(provider.messages, [])
            turns = [{"generation_id": str(index), "user": "帮我写论文" + "证据" * 200,
                      "assistant": "已交付" * 30, "created_at": now + index} for index in range(5)]
            kept, digest, cutoff = await working.compress(provider, turns, {})
            self.assertEqual(digest["goal"], "帮我写论文")
            self.assertTrue(cutoff)
            merged = working.merge(kept, turns, cutoff)
            self.assertEqual([turn["generation_id"] for turn in merged], [turn["generation_id"] for turn in kept])
            self.assertLessEqual(working.token_count(kept, digest), 512)
            await working.compress(provider, kept, {"context_digest": digest, "compressed_before": cutoff})
            self.assertEqual(len(provider.messages), 1)
        finally:
            working.close()

    async def test_summary_cannot_introduce_fabricated_fact(self):
        """无证据摘要被丢弃，回退保留确定的任务目标。"""
        working = WorkingMemory(token_threshold=512)
        provider = GateProvider()
        provider.digest["goal"] = "用户在北京工作"
        turns = [{"generation_id": str(index), "user": "原始历史" * 200, "created_at": time() + index} for index in range(4)]
        try:
            _, digest, _ = await working.compress(provider, turns, {"task_memory": {"goal": "帮我写论文"}})
            self.assertEqual(digest["goal"], "帮我写论文")
            self.assertNotIn("北京", json.dumps(digest, ensure_ascii=False))
        finally:
            working.close()
