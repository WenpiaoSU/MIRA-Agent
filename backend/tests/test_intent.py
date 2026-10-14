"""作用：验证结构化意图识别、实际路由和网关取消边界。

当前功能：
- 用固定模型输出检查七类契约、工具参数、文本证据和失败降级。
- 验证确定性能力说明、会话记忆查询及句子级恢复分支。
- 通过生产网关检查慢分类、停止、新轮次与旧结果隔离。
- 检查上下文预算、无 ASR 置信度输入及脱敏轨迹。
"""

import asyncio
import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError

from src.agent.graph import AgentGraph
from src.api.realtime import realtime
from src.contracts.perception import IntentEstimate
from src.contracts.tools import ToolSpec
from src.evaluation.trace import TraceRecorder
from src.gateway.sessions import Session, SessionRegistry
from src.intent.service import IntentService, build_context_summary
from src.providers.llm import LLMError
from test_p3 import FakeSocket, FakeSpeech, FakeTTS, RecordingAgent


def estimate(intent="chat", **updates):
    """生成一份合法的分类结果，允许各场景覆盖参数和标志。"""
    payload = {"intent": intent, "slots": [], "confidence": 0.9, "tool_intent": None,
               "needs_clarification": False, "missing_slots": [], "control_action": None}
    payload.update(updates)
    return IntentEstimate.model_validate(payload)


class ScriptedProvider:
    """模拟 SSE 分块与错误，所有测试不访问外部模型。"""

    def __init__(self, payload=None, *, raw=None, error=None, delay=0):
        """记录指定输出、失败方式及实际请求消息。"""
        self.raw = raw if raw is not None else json.dumps(payload or estimate().model_dump(), ensure_ascii=False)
        self.error = error
        self.delay = delay
        self.messages = []
        self.closed = False

    async def stream(self, messages):
        """拆开 JSON 产出，关闭时记录流已回收。"""
        self.messages.append(messages)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            midpoint = len(self.raw) // 2
            yield self.raw[:midpoint]
            yield self.raw[midpoint:]
        finally:
            self.closed = True


class IntentServiceTests(unittest.IsolatedAsyncioTestCase):
    """检查真实分类服务的契约与有界降级，不以模拟结果声称模型准确率。"""

    async def test_seven_categories_and_input_without_asr_confidence(self):
        """七类 JSON 能进入契约，传给模型的数据不包含 ASR 置信度。"""
        cases = [("chat", "今天好累"), ("question", "为什么植物会停止生长？"),
                 ("tool_request", "查杭州天气"), ("memory_query", "你记得我的研究方向吗"),
                 ("memory_update", "我喜欢简短回答"), ("control", "刚才那段请重新讲讲"), ("other", "那个呢")]
        for name, text in cases:
            with self.subTest(intent=name):
                output = estimate(name, control_action="repeat" if name == "control" else None)
                provider = ScriptedProvider(output.model_dump())
                result = await IntentService(provider).classify(text, "", [], source="speech")
                self.assertEqual(result.intent, name)
                request = json.loads(provider.messages[0][1]["content"])
                self.assertEqual(request["source"], "speech")
                self.assertNotIn("asr_confidence", request)
                self.assertTrue(provider.closed)
        self.assertNotIn("asr_confidence", inspect.signature(IntentService.classify).parameters)

    async def test_trusted_controls_skip_llm_but_quoted_words_do_not(self):
        """完整控制短句不调用模型，引用及否定不会命中快捷路径。"""
        provider = ScriptedProvider(estimate("question").model_dump())
        service = IntentService(provider)
        for text, action in [("停一下。", "stop"), ("取消", "stop"), ("请继续", "resume"),
                             ("重复一遍", "repeat"), ("换个话题", "change_topic")]:
            result = await service.classify(text, "", [], source="text")
            self.assertEqual(result.control_action, action)
        self.assertEqual(provider.messages, [])
        for text in ("为什么会停止生长？", "不要停止", "他说‘停一下’是什么意思？"):
            self.assertEqual((await service.classify(text, "", [], source="text")).intent, "question")
        self.assertEqual(len(provider.messages), 3)

    async def test_empty_input_does_not_call_provider(self):
        """空白最终文本不发起分类调用。"""
        provider = ScriptedProvider()
        result = await IntentService(provider).classify(" \n", "", [], source="speech")
        self.assertEqual(result.intent, "other")
        self.assertEqual(provider.messages, [])

    async def test_single_json_fence_is_accepted_but_mixed_prose_is_rejected(self):
        """兼容完整 JSON 外层围栏，不从解释正文或多个代码块中猜测结果。"""
        payload = estimate().model_dump_json()
        for raw in ("```json\n" + payload + "\n```", "```\n" + payload + "\n```",
                    "```json\r\n" + payload + "\r\n```"):
            result = await IntentService(ScriptedProvider(raw=raw)).classify("你好", "", [], source="text")
            self.assertEqual(result.intent, "chat")
        for raw in ("说明\n```json\n" + payload + "\n```", "```json\n" + payload + "\n```\n说明",
                    "```json\n" + payload + "\n```\n```json\n" + payload + "\n```"):
            result = await IntentService(ScriptedProvider(raw=raw)).classify("你好", "", [], source="text")
            self.assertEqual(result.intent, "other")

    async def test_invalid_json_schema_and_evidence_degrade(self):
        """拒绝字段矛盾、非法类型、伪造证据、超长输出和非 JSON 正文。"""
        valid = estimate().model_dump()
        invalid = ["不是 JSON", "分类结果：```json\n" + json.dumps(valid) + "\n```", "x" * 8193,
                   json.dumps({**valid, "confidence": 1.1}), json.dumps({**valid, "confidence": float("nan")}),
                   json.dumps({**valid, "needs_clarification": "false"}),
                   json.dumps({**valid, "tool_intent": "get_weather"}),
                   json.dumps({**valid, "control_action": "stop"}),
                   json.dumps({**valid, "missing_slots": ["location"]}),
                   json.dumps({**valid, "extra": "不允许额外字段"}),
                   json.dumps({**valid, "slots": [{"name": "location", "value": "北京", "evidence": "北京"}]}),
                   json.dumps({**valid, "slots": [{"name": "location", "value": "杭州"}]})]
        invalid.append(json.dumps({**valid, "slots": [{"name": "number", "value": float("nan"), "evidence": "杭州"}]}))
        for raw in invalid:
            with self.subTest(raw=raw[:60]):
                provider = ScriptedProvider(raw=raw)
                result = await IntentService(provider).classify("查杭州天气", "", [], source="text")
                self.assertEqual(result, IntentEstimate.uncertain())
                self.assertTrue(provider.closed)

    async def test_timeout_model_failure_and_cancel_close_stream(self):
        """超时或模型故障能够澄清，取消不吞掉，HTTP 流始终回收。"""
        for provider in (ScriptedProvider(delay=1), ScriptedProvider(error=LLMError("测试错误"))):
            result = await IntentService(provider, timeout_ms=20).classify("你好", "", [], source="text")
            self.assertEqual(result, IntentEstimate.uncertain())
            self.assertTrue(provider.closed)
        provider = ScriptedProvider(delay=10)
        task = asyncio.create_task(IntentService(provider).classify("你好", "", [], source="text"))
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(provider.closed)

    async def test_tool_catalog_validates_types_and_derives_missing_fields(self):
        """真实 Schema 校验参数类型，并计算模型漏报的必填缺项。"""
        tool = ToolSpec(name="get_weather", description="查询天气", input_schema={
            "type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer", "minimum": 1}},
            "required": ["location"], "additionalProperties": False})
        cases = [(estimate("tool_request", tool_intent="get_weather"), "tool_request", ["location"]),
                 (estimate("tool_request", tool_intent="get_weather", slots=[
                     {"name": "location", "value": "杭州", "evidence": "杭州"}]), "tool_request", []),
                 (estimate("tool_request", tool_intent="get_weather", slots=[
                     {"name": "days", "value": "三", "evidence": "三"}]), "other", []),
                 (estimate("tool_request", tool_intent="get_weather", slots=[
                     {"name": "user_id", "value": "杭州", "evidence": "杭州"}]), "other", []),
                 (estimate("tool_request", tool_intent="unknown_tool"), "other", [])]
        for output, name, missing in cases:
            with self.subTest(output=output):
                service = IntentService(ScriptedProvider(output.model_dump()))
                result = await service.classify("查杭州三天的天气", "", [tool], source="text")
                self.assertEqual(result.intent, name)
                self.assertEqual(result.missing_slots, missing)
                if missing:
                    self.assertTrue(result.needs_clarification)
        result = await IntentService(ScriptedProvider(estimate("tool_request").model_dump())).classify(
            "查杭州天气", "", [], source="text")
        self.assertEqual(result.intent, "tool_request")
        self.assertIsNone(result.tool_intent)
        self.assertFalse(result.needs_clarification)

    async def test_nested_tool_schema_and_invalid_reference_degrade(self):
        """嵌套必填错误和无法解析的本地引用不能被当作合法参数。"""
        for nested in ({"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
                       {"$ref": "#/$defs/not_found"}):
            tool = ToolSpec(name="nested", description="嵌套参数", input_schema={
                "type": "object", "properties": {"options": nested}, "required": ["options"]})
            output = estimate("tool_request", tool_intent="nested", slots=[{"name": "options", "value": {}, "evidence": "选项"}])
            result = await IntentService(ScriptedProvider(output.model_dump())).classify("选项", "", [tool], source="text")
            self.assertEqual(result, IntentEstimate.uncertain())

    async def test_context_evidence_and_nullable_confidence_are_accepted(self):
        """允许可核实的近期指代和空置信度，不用任意阈值替代判断。"""
        result = await IntentService(ScriptedProvider(estimate("memory_update", confidence=None, slots=[
            {"name": "value", "value": "小苏", "evidence": "小苏"}]).model_dump())).classify(
                "以后这样叫我", "用户：小苏", [], source="text")
        self.assertEqual(result.intent, "memory_update")
        self.assertIsNone(result.confidence)


class IntentStateTests(unittest.TestCase):
    """覆盖上下文预算、Schema 安全边界和本轮状态隔离。"""

    def test_context_is_bounded_and_uses_delivered_information(self):
        """保留最近消息及已说事实，丢弃较早大消息和未交付计划。"""
        history = [{"role": "user", "content": "早" * 4000}, {"role": "user", "content": "杭州"},
                   {"role": "assistant", "content": "收到"}]
        summary = build_context_summary(history, {"spoken_text": "完整播完。", "remaining_text": "未播内容。"})
        self.assertLessEqual(len(summary), 3000)
        self.assertIn("杭州", summary)
        self.assertIn("完整播完", summary)
        self.assertNotIn("未播内容", summary)
        self.assertNotIn("早", summary)

    def test_snapshot_does_not_reuse_old_intent_and_trace_redacts_slots(self):
        """本轮快照深拷贝识别结果，后续快照不沿用，轨迹过滤正文。"""
        session = Session()
        intent = estimate("memory_update", slots=[{"name": "value", "value": "私人内容", "evidence": "私人内容"}])
        snapshot = session.social_snapshot("私人内容", intent)
        intent.slots.clear()
        self.assertEqual(snapshot.intent.slots[0].value, "私人内容")
        self.assertIsNone(session.social_snapshot("下一轮").intent)
        trace = TraceRecorder()
        trace.record("intent.classified", "t", {"intent": "memory_update", "elapsed_ms": 5,
                                                "text": "私人内容", "slots": [{"value": "私人内容"}]})
        self.assertNotIn("私人内容", json.dumps(trace.report(), ensure_ascii=False))

    def test_duplicate_slots_and_external_schema_refs_are_rejected(self):
        """不接受冲突参数或需要远程读取的 Schema。"""
        with self.assertRaises(ValidationError):
            estimate(slots=[{"name": "value", "value": "a"}, {"name": "value", "value": "b"}])
        with self.assertRaises(ValidationError):
            ToolSpec(name="remote", description="远程引用", input_schema={
                "type": "object", "properties": {"x": {"$ref": "https://example.com/schema"}}})


class IntentGraphTests(unittest.IsolatedAsyncioTestCase):
    """检查生产 LangGraph 的实际分支及对用户的可见结果。"""

    async def run_graph(self, intent, text="当前请求", history=None, interruption=None):
        """运行真实图并返回事件和模型提示，模型正文使用固定合法格式。"""
        provider = ScriptedProvider(raw='{"expression":"neutral","gaze":"center","gesture":null,"intensity":0.6}\n回答内容。')
        graph = AgentGraph(provider)
        context = {"social_state": {"intent": intent.model_dump()}, "interruption_state": interruption}
        events = [event async for event in graph.stream(text, history or [], [], "g", runtime_context=context)]
        return events, provider

    async def test_chat_and_question_have_different_routes_with_same_output(self):
        """交流和问答采用各自约束，同时保持原有行为与正文事件。"""
        for intent, instruction in [("chat", "交流或分享"), ("question", "知识或建议")]:
            events, provider = await self.run_graph(estimate(intent))
            self.assertIn(instruction, provider.messages[0][0]["content"])
            self.assertEqual([item["type"] for item in events], ["behavior.plan", "response.segment", "response.completed"])

    async def test_unavailable_tools_and_memory_are_deterministic(self):
        """未开放能力不调用回复模型，也不生成操作成功或天气结果。"""
        cases = [(estimate("tool_request"), "外部工具"),
                 (estimate("memory_update"), "未进行持久保存"),
                 (estimate("memory_update", slots=[{"name": "operation", "value": "delete", "evidence": "忘掉"}]), "未进行持久保存或删除"),
                 (estimate("memory_query"), "没有找到可供核实的相关记忆记录"),
                 (IntentEstimate.uncertain(), "请换个说法")]
        for intent, expected in cases:
            with self.subTest(intent=intent):
                events, provider = await self.run_graph(intent)
                self.assertIn(expected, events[-1]["payload"]["text"])
                self.assertEqual(provider.messages, [])

    async def test_memory_query_uses_only_user_history(self):
        """有会话历史时采用用户证据约束，不将模型推测视为长期事实。"""
        _, provider = await self.run_graph(estimate("memory_query"), history=[{"role": "user", "content": "我的方向是 NLP"}])
        self.assertIn("只依据长期记录和近期用户原文", provider.messages[0][0]["content"])
        self.assertEqual(provider.messages[0][1]["content"], "我的方向是 NLP")

    async def test_clarification_missing_fields_and_contextual_other(self):
        """工具未注册时说明能力不可用；其他歧义仍可结合当前文本澄清。"""
        events, provider = await self.run_graph(estimate("tool_request", tool_intent="get_weather",
                                                      needs_clarification=True, missing_slots=["location"]))
        self.assertIn("没有开放", events[-1]["payload"]["text"])
        self.assertEqual(provider.messages, [])
        _, provider = await self.run_graph(estimate("other", needs_clarification=True))
        self.assertIn("只询问一个具体的待确认问题", provider.messages[0][0]["content"])

    async def test_semantic_recovery_and_stop_do_not_replay_tools(self):
        """语义控制采用已有句子级约束；停止静默结束，不生成新语音。"""
        for action in ("resume", "repeat", "change_topic"):
            _, provider = await self.run_graph(estimate("control", control_action=action),
                                               interruption={"resume_allowed": True, "spoken_text": "第一句。"})
            self.assertIn("不执行或重复外部操作", provider.messages[0][0]["content"])
        for action in ("stop", "cancel"):
            events, provider = await self.run_graph(estimate("control", control_action=action))
            self.assertEqual([event["type"] for event in events], ["turn.controlled", "response.completed"])
            self.assertEqual(events[0]["payload"]["action"], action)
            self.assertEqual(events[-1]["payload"]["text"], "")
            self.assertEqual(provider.messages, [])


class IntentGatewayTests(unittest.IsolatedAsyncioTestCase):
    """验证分类任务与真实网关接收、取消和状态冻结之间的关系。"""

    async def asyncSetUp(self):
        """建立完全本地的生产网关任务与双向事件队列。"""
        registry = SessionRegistry()
        self.session = registry.create()
        self.agent = RecordingAgent()
        self.provider = ScriptedProvider()
        self.service = IntentService(self.provider)
        self.app = SimpleNamespace(state=SimpleNamespace(
            sessions=registry, agent=self.agent, intent_service=self.service, tts=FakeTTS(),
            speech_service=FakeSpeech(), settings=SimpleNamespace(llm_ready=True, tts_ready=False)))
        self.ws = FakeSocket(self.app, self.session)
        self.task = asyncio.create_task(realtime(self.ws, self.session.session_id))
        await self.ws.until("session.status")

    async def asyncTearDown(self):
        """关闭连接并确认所有连接子任务都能退出。"""
        await self.ws.close()
        await asyncio.wait_for(self.task, 3)

    async def test_slow_classification_does_not_block_stop(self):
        """分类等待期间直接停止；分类流回收且 Agent 不被调用。"""
        self.provider.delay = 10
        await self.ws.send("input.text", {"text": "解释方法"}, "t1")
        await self.ws.until("turn.state", lambda event: event["payload"]["state"] == "received")
        for _ in range(100):
            if self.provider.messages:
                break
            await asyncio.sleep(.001)
        self.assertTrue(self.provider.messages)
        await self.ws.send("input.text", {"text": "停一下"}, "stop")
        await self.ws.until("output.cancel")
        await self.ws.until("turn.controlled")
        self.assertEqual(self.agent.calls, [])
        self.assertTrue(self.provider.closed)
        self.assertIsNone(self.session.generation_id)

    async def test_late_old_classification_cannot_enter_new_generation(self):
        """故意吞取消的分类返回后，旧结果仍被 generation 检查丢弃。"""
        entered = asyncio.Event()

        async def classify(text, context_summary, tools, *, source, domain_scope="daily"):
            """第一轮模拟吞取消的供应商，第二轮立即返回真实结果。"""
            if text == "旧请求":
                entered.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    return estimate("memory_update")
            return estimate("question")

        self.app.state.intent_service = SimpleNamespace(classify=classify)
        self.session.stop_ack_timeout_ms = 1
        await self.ws.send("input.text", {"text": "旧请求"}, "t1")
        await asyncio.wait_for(entered.wait(), 1)
        await self.ws.send("input.text", {"text": "新请求"}, "t2")
        await self.ws.until("response.completed")
        self.assertEqual([call[0] for call in self.agent.calls], ["新请求"])
        frozen = self.agent.calls[0][1]["social_state"]
        self.assertEqual(frozen["intent"]["intent"], "question")
        classified = [event for event in self.session.trace.events if event["type"] == "intent.classified"]
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0]["payload"]["intent"], "question")

    async def test_semantic_stop_completes_without_audio_or_empty_placeholder(self):
        """真实图静默停止后发送现有控制事件，前端可收起空回复占位。"""
        self.provider.raw = json.dumps(estimate("control", control_action="stop").model_dump())
        self.app.state.agent = AgentGraph(self.provider)
        await self.ws.send("input.text", {"text": "请结束刚才的说明"}, "semantic_stop")
        controlled = await self.ws.until("turn.controlled")
        self.assertEqual(controlled["payload"]["action"], "stop")
        delivered = await self.ws.until("turn.delivered")
        record = self.session.outputs[delivered["generation_id"]]
        self.assertEqual(record.segments, {})
        self.assertEqual(len(self.provider.messages), 1)
        self.assertIsNone(self.session.generation_id)

    async def test_partial_does_not_classify_and_committed_voice_has_source(self):
        """真实语音事件流只有完整轮次会调用分类，并标记 speech 来源。"""
        self.service.classify = AsyncMock(wraps=self.service.classify)
        await self.ws.send("input.audio_start", {"stream_id": "voice", "encoding": "pcm_s16le",
                           "sample_rate": 16000, "channels": 1})
        await self.ws.until("turn.state", lambda event: event["payload"].get("audio_ready"))
        for index in range(32):
            await self.ws.send("input.audio_frame", {"stream_id": "voice", "frame_seq": index,
                               "capture_start_ms": index * 20, "capture_end_ms": (index + 1) * 20})
        await self.ws.until("asr.partial")
        self.assertEqual(self.provider.messages, [])
        self.service.classify.assert_not_awaited()
        await self.ws.send("input.audio_frame", {"stream_id": "voice", "frame_seq": 32,
                           "capture_start_ms": 640, "capture_end_ms": 660})
        await self.ws.until("response.completed")
        # 测试语音流最终文本为“继续”，命中确定性控制，不需要模型。
        self.assertEqual(self.provider.messages, [])
        self.service.classify.assert_awaited_once()
        self.assertEqual(self.service.classify.await_args.kwargs["source"], "speech")
        self.assertEqual(self.agent.calls[0][0], "继续")
        self.assertEqual(self.agent.calls[0][1]["social_state"]["intent"]["control_action"], "resume")


if __name__ == "__main__":
    unittest.main()
