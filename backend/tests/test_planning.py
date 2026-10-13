"""作用：验证不同工具依赖任务与现有 Agent、实时网关的完整闭环。

当前功能：
- 验证查询、时间转换、依赖天气参数和综合回复的串联。
- 验证非法计划、引用、证据、步数、类型和模型失败的有界降级。
- 验证跨轮消歧、跳过成功步骤、部分失败和整轮预算。
- 验证停止、断线、旧结果隔离、结果恢复及 TTS 共用链路。
"""

import asyncio
import copy
import json
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx

from src.agent.graph import AgentGraph
from src.agent.planner import TaskPlanner, PlanError
from src.api.realtime import realtime
from src.contracts.agent import PlanDraft, TaskPlan
from src.contracts.tools import PendingClarification, ToolSpec
from src.gateway.sessions import SessionRegistry
from src.intent.service import IntentService
from src.tools.builtin.datetime import DATETIME_SPEC, convert_datetime
from src.tools.builtin.time import TIME_SPEC, CurrentTimeTool
from src.tools.builtin.weather import WEATHER_SPEC, WeatherTool
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry
from test_intent import estimate
from test_p3 import FakeSocket, FakeSpeech, FakeTTS
from test_tools import WeatherHTTP, request


COMPOUND_TEXT = "按 UTC 当前时间换成 Asia/Shanghai 的明天日期，查杭州, 浙江该日天气，再查天气地点的当前时间并给出出门建议。"


def compound_draft():
    """构造包含不同工具、三层字段依赖与四次调用的真实任务。"""
    return {"goal": "查询转换后的明天天气和当地当前时间，给出出门建议", "steps": [
        {"kind": "tool", "tool_name": "get_current_time", "arguments": {"timezone": "UTC"},
         "argument_evidence": {"timezone": "UTC"}},
        {"kind": "tool", "tool_name": "convert_datetime", "arguments": {"timezone": "Asia/Shanghai", "days_offset": 1},
         "argument_evidence": {"timezone": "Asia/Shanghai", "days_offset": "明天"},
         "bindings": {"iso_time": {"step": 1, "path": "/iso_time"}}},
        {"kind": "tool", "tool_name": "get_weather", "arguments": {"location": "杭州, 浙江"},
         "argument_evidence": {"location": "杭州, 浙江"}, "bindings": {"date": {"step": 2, "path": "/date"}}},
        {"kind": "tool", "tool_name": "get_current_time", "bindings": {"timezone": {"step": 3, "path": "/timezone"}}},
        {"kind": "respond"}]}


class ScenarioProvider:
    """按实际提示阶段返回分类、计划及综合回复，不访问外部模型。"""

    def __init__(self, draft=None):
        """记录模型调用与关闭状态，并允许对子场景注入错误输出。"""
        self.draft = draft or compound_draft()
        self.intent = estimate("tool_request", needs_planning=True)
        self.answer = {"text": "结合查询的预报和当地时间，出门可准备薄外套，并留意天气变化。", "result_refs": [1, 2, 3, 4]}
        self.calls = []
        self.closed = []
        self.delays = {}
        self.raw_plan = None

    async def stream(self, messages):
        """复用同一流式接口，取消或超时时标记资源已关闭。"""
        prompt = messages[0]["content"]
        role = "plan" if "你为用户的实际任务" in prompt else "answer" if "根据用户目标和已验证" in prompt else "intent"
        self.calls.append((role, messages))
        try:
            if self.delays.get(role):
                await asyncio.sleep(self.delays[role])
            payload = self.draft if role == "plan" else self.answer if role == "answer" else self.intent.model_dump()
            raw = self.raw_plan if role == "plan" and self.raw_plan else json.dumps(payload, ensure_ascii=False)
            yield raw[:len(raw) // 2]
            yield raw[len(raw) // 2:]
        finally:
            self.closed.append(role)


def make_registry(http, calls):
    """绑定真实内置函数，记录实际调用名及已解析的参数。"""
    registry = ToolRegistry()
    handlers = [(TIME_SPEC, CurrentTimeTool()), (DATETIME_SPEC, convert_datetime),
                (WEATHER_SPEC, WeatherTool(httpx.MockTransport(http)))]
    for spec, handler in handlers:
        def bind(name, function):
            """创建独立闭包，避免循环中工具名与函数被后续覆盖。"""
            async def invoke(arguments):
                """记录执行器最终传入的参数，随后运行生产工具。"""
                calls.append((name, copy.deepcopy(arguments)))
                return await function(arguments)
            return invoke
        registry.register(spec, bind(spec.name, handler))
    return registry


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    """检查生产规划器和 LangGraph 节点，不以模拟结果声称模型准确率。"""

    async def asyncSetUp(self):
        """构造独立模型、HTTP 替身及真实工具目录。"""
        self.provider = ScenarioProvider()
        self.http = WeatherHTTP()
        self.calls = []
        self.registry = make_registry(self.http, self.calls)
        self.planner = TaskPlanner(self.provider)
        self.executor = ToolExecutor(self.registry)
        self.graph = AgentGraph(self.provider, self.executor, self.planner)

    async def run_graph(self, text=COMPOUND_TEXT, intent=None, runtime=None):
        """运行真实图，返回内部状态与用户输出事件。"""
        context = {"social_state": {"intent": (intent or self.provider.intent).model_dump()}, **(runtime or {})}
        return [event async for event in self.graph.stream(text, [], [], "graph-test", runtime_context=context)]

    async def test_different_tools_with_chained_parameters_and_more_than_two_calls(self):
        """实际执行查询、转换、依赖查询和当地时间，再综合回答。"""
        events = await self.run_graph()
        self.assertEqual([name for name, _ in self.calls], ["get_current_time", "convert_datetime", "get_weather", "get_current_time"])
        plan = TaskPlan.model_validate([event for event in events if event["type"] == "plan.updated"][-1]["payload"]["plan"])
        self.assertEqual(plan.status, "completed")
        self.assertEqual(self.calls[1][1]["iso_time"], plan.steps[0].result.data["iso_time"])
        self.assertEqual(self.calls[2][1]["date"], plan.steps[1].result.data["date"])
        self.assertEqual(self.calls[3][1]["timezone"], plan.steps[2].result.data["timezone"])
        self.assertEqual(Counter(role for role, _ in self.provider.calls), {"plan": 1, "answer": 1})
        self.assertIn("综合回答", events[-1]["payload"]["text"])
        self.assertIn("Open-Meteo", events[-1]["payload"]["text"])
        self.assertEqual(len([event for event in events if event["type"] == "tool.result"]), 4)

    async def test_invalid_plans_are_rejected_before_any_tool(self):
        """未知工具、未来引用、缺证据、错字段和错类型都不能部分执行。"""
        drafts = []
        for update in ("unknown", "future", "missing_path", "bad_type", "evidence", "extra", "duplicate"):
            draft = compound_draft()
            step = draft["steps"][1]
            if update == "unknown": draft["steps"][2]["tool_name"] = "create_reminder"
            elif update == "future": step["bindings"]["iso_time"]["step"] = 3
            elif update == "missing_path": step["bindings"]["iso_time"]["path"] = "/not_found"
            elif update == "bad_type":
                step["arguments"].pop("days_offset")
                step["argument_evidence"].pop("days_offset")
                step["bindings"] = {"days_offset": {"step": 1, "path": "/iso_time"}}
            elif update == "evidence": step["argument_evidence"]["timezone"] = "用户没提供的证据"
            elif update == "extra": step["code"] = "不允许执行任意代码"
            elif update == "duplicate": step["arguments"]["iso_time"] = "UTC"
            drafts.append((update, draft))
        for name, draft in drafts:
            with self.subTest(name=name):
                self.provider.draft = draft
                events = await self.run_graph()
                self.assertEqual(self.calls, [])
                self.assertIn("可靠的执行计划", events[-1]["payload"]["text"])

    async def test_step_limit_rejects_plan_instead_of_truncating_goal(self):
        """超限计划整段拒绝，不截取前几个步骤而遗漏用户目标。"""
        self.graph = AgentGraph(self.provider, self.executor, TaskPlanner(self.provider, max_steps=4))
        events = await self.run_graph()
        self.assertEqual(self.calls, [])
        self.assertIn("减少任务步骤", events[-1]["payload"]["text"])

    async def test_response_refs_and_model_failure_keep_verified_facts(self):
        """综合模型引用非法步骤时保留事实，并将整体任务标为未完成。"""
        self.provider.answer["result_refs"] = [999]
        events = await self.run_graph()
        plan = TaskPlan.model_validate([event for event in events if event["type"] == "plan.updated"][-1]["payload"]["plan"])
        self.assertEqual(plan.status, "failed")
        self.assertIn("暂时无法生成", events[-1]["payload"]["text"])
        self.assertIn("Open-Meteo", events[-1]["payload"]["text"])
        self.assertNotIn(self.provider.answer["text"], events[-1]["payload"]["text"])

    async def test_null_dependency_fails_without_querying_next_tool(self):
        """Schema 有字段但实际为 null 时不把缺失事实传给后续工具。"""
        self.provider.draft = {"goal": "依赖当前天气的预报日期", "steps": [
            {"kind": "tool", "tool_name": "get_weather", "arguments": {"location": "杭州, 浙江"},
             "argument_evidence": {"location": "杭州, 浙江"}},
            {"kind": "tool", "tool_name": "get_weather", "arguments": {"location": "杭州, 浙江"},
             "argument_evidence": {"location": "杭州, 浙江"}, "bindings": {"date": {"step": 1, "path": "/forecast/date"}}},
            {"kind": "respond"}]}
        events = await self.run_graph()
        self.assertEqual(len(self.calls), 1)
        self.assertIn("任务未全部完成", events[-1]["payload"]["text"])
        self.assertFalse(any(role == "answer" for role, _ in self.provider.calls))

    async def test_static_missing_parameter_waits_before_any_tool(self):
        """后续步骤有静态缺项时先补参，避免执行半个已知不完整任务。"""
        del self.provider.draft["steps"][2]["arguments"]["location"]
        del self.provider.draft["steps"][2]["argument_evidence"]["location"]
        events = await self.run_graph()
        self.assertEqual(self.calls, [])
        updated = [event for event in events if event["type"] == "plan.updated"][-1]["payload"]
        self.assertEqual(updated["plan"]["status"], "waiting_input")
        self.assertEqual(updated["pending"]["step_id"], "step_3")

    async def test_planning_timeout_closes_stream_and_does_not_execute(self):
        """规划超时直接澄清，不启动工具或第二次模型修复。"""
        self.provider.delays["plan"] = 10
        self.graph = AgentGraph(self.provider, self.executor, TaskPlanner(self.provider, timeout_ms=20))
        events = await self.run_graph()
        self.assertIn("规划超时", events[-1]["payload"]["text"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.provider.closed, ["plan"])

    async def test_actual_deadline_stops_later_steps(self):
        """所有步骤共享当前轮次剩余预算，预算不足时不再执行后续工具。"""
        deadline = datetime.now(timezone.utc) - timedelta(milliseconds=1)
        events = await self.run_graph(runtime={"generation_deadline_at": deadline.isoformat()})
        self.assertEqual(self.calls, [])
        self.assertIn("超时", events[-1]["payload"]["text"])

    async def test_simple_query_has_no_planner_call(self):
        """已有简单时间查询直接执行，不增加规划或综合模型请求。"""
        events = await self.run_graph("现在几点", estimate("tool_request", tool_intent="get_current_time"))
        self.assertEqual(self.provider.calls, [])
        self.assertIn("当前时间", events[-1]["payload"]["text"])

    async def test_intent_planning_skips_single_tool_missing_fields_and_marks_continuation(self):
        """组合任务不被单工具缺项拦截；计划补参自动绑定原规划路线。"""
        self.provider.intent = estimate("tool_request", tool_intent="get_weather", needs_planning=True)
        service = IntentService(self.provider)
        result = await service.classify(COMPOUND_TEXT, "", self.registry.list_tools(), source="speech")
        self.assertEqual(result.missing_slots, [])
        self.assertTrue(result.needs_planning)
        self.provider.intent = estimate("tool_request", tool_intent="get_weather", continues_pending=True, slots=[
            {"name": "location", "value": "杭州", "evidence": "杭州"}])
        pending = PendingClarification(original_text=COMPOUND_TEXT, tool_name="get_weather", arguments={},
                                       missing_slots=["location"], question="地点？", plan_id="p", step_id="step_3")
        result = await service.classify("杭州", "", self.registry.list_tools(), source="text", pending_clarification=pending)
        self.assertTrue(result.needs_planning)

    async def test_datetime_conversion_is_deterministic_and_rejects_naive_input(self):
        """转换工具依据带偏移时间计算当地日期，无时区输入必须澄清。"""
        result = await self.executor.execute(request("convert_datetime", {
            "iso_time": "2026-10-13T23:30:00-07:00", "timezone": "Asia/Shanghai", "days_offset": 1}))
        self.assertEqual(result.data["date"], "2026-10-15")
        self.assertIn("+08:00", result.data["iso_time"])
        result = await self.executor.execute(request("convert_datetime", {
            "iso_time": "2026-10-13T23:30:00", "timezone": "Asia/Shanghai"}))
        self.assertEqual(result.error.missing_slots, ["iso_time"])

    async def test_other_registered_tools_and_array_references_are_supported(self):
        """规划不写死内置工具组合，并验证数组字段到新工具参数的依赖。"""
        async def values(arguments):
            """返回有明确输出 Schema 的本地测试数据。"""
            return {"values": [2, 7]}

        async def double(arguments):
            """消费前序引用得到的整数并返回计算结果。"""
            return {"answer": arguments["value"] * 2}

        registry = ToolRegistry()
        registry.register(ToolSpec(name="values", description="获取数值", input_schema={"type": "object"},
                                  output_schema={"type": "object", "properties": {"values": {
                                      "type": "array", "items": {"type": "integer"}}}, "required": ["values"]}), values)
        registry.register(ToolSpec(name="double", description="翻倍", input_schema={"type": "object",
                                  "properties": {"value": {"type": "integer"}}, "required": ["value"]},
                                  output_schema={"type": "object", "properties": {"answer": {"type": "integer"}},
                                                 "required": ["answer"]}), double)
        self.provider.draft = {"goal": "获取数字并翻倍", "steps": [
            {"kind": "tool", "tool_name": "values"},
            {"kind": "tool", "tool_name": "double", "bindings": {"value": {"step": 1, "path": "/values/1"}}},
            {"kind": "respond"}]}
        self.provider.answer["result_refs"] = [1, 2]
        self.graph = AgentGraph(self.provider, ToolExecutor(registry))
        events = await self.run_graph("获取数字并翻倍")
        final = [event for event in events if event["type"] == "plan.updated"][-1]["payload"]["plan"]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["steps"][1]["result"]["data"], {"answer": 14})

    async def test_cancel_does_not_backfill_new_results_to_old_generation(self):
        """跨轮任务取消只改变历史快照状态，不把新结果写入旧输出。"""
        events = await self.run_graph()
        completed = TaskPlan.model_validate([event for event in events if event["type"] == "plan.updated"][-1]["payload"]["plan"])
        previous = completed.model_copy(deep=True)
        previous.status = "waiting_input"
        previous.response_text = ""
        for step in previous.steps[2:]:
            step.result = None
            step.status = "pending"
        previous.steps[2].status = "waiting_input"
        current = completed.model_copy(deep=True)
        current.status = "running"
        current.response_text = ""
        current.steps[3].result = None
        current.steps[3].status = "running"
        current.steps[4].status = "pending"
        session = SessionRegistry().create()
        old_record = session.begin_generation("old")
        old_record.task_plan = previous
        new_record = session.begin_generation("new")
        new_record.task_plan = current.model_copy(deep=True)
        session.active_plan = current
        session.cancel_generation("user_stop")
        self.assertIsNone(old_record.task_plan.steps[2].result)
        self.assertTrue(new_record.task_plan.steps[2].result.verified)
        self.assertEqual(old_record.task_plan.status, "cancelled")
        self.assertEqual(new_record.task_plan.status, "cancelled")


class PlanningGatewayTests(unittest.IsolatedAsyncioTestCase):
    """验证真实 WebSocket 网关、分类、规划、步骤和输出之间的状态边界。"""

    async def asyncSetUp(self):
        """装配真实网关与局部网络替身。"""
        sessions = SessionRegistry()
        self.session = sessions.create()
        self.provider = ScenarioProvider()
        self.http = WeatherHTTP()
        self.calls = []
        self.registry = make_registry(self.http, self.calls)
        self.executor = ToolExecutor(self.registry)
        self.settings = SimpleNamespace(llm_ready=True, tts_ready=False, stop_ack_timeout_ms=100)
        self.app = SimpleNamespace(state=SimpleNamespace(
            sessions=sessions, agent=AgentGraph(self.provider, self.executor), tool_registry=self.registry,
            intent_service=IntentService(self.provider), tts=FakeTTS(), speech_service=FakeSpeech(), settings=self.settings))
        self.ws = FakeSocket(self.app, self.session)
        self.task = asyncio.create_task(realtime(self.ws, self.session.session_id))
        await self.ws.until("session.status")

    async def asyncTearDown(self):
        """关闭连接并确认任务执行和网络子任务能够回收。"""
        await self.ws.close()
        await asyncio.wait_for(self.task, 3)

    async def submit(self, text, turn="task"):
        """提交本轮并等待纯文字输出交付，避免测试状态竞争。"""
        await self.ws.send("input.text", {"text": text}, turn)
        response = await self.ws.until("response.completed", lambda event: event.get("turn_id") == turn)
        await self.ws.until("turn.delivered", lambda event: event.get("turn_id") == turn)
        return response

    async def test_gateway_stores_multiple_results_trace_and_repeat_without_reexecution(self):
        """不同工具依赖任务保存全部结果，重复只复述原回答，不重规划。"""
        response = await self.submit(COMPOUND_TEXT)
        record = self.session.outputs[response["generation_id"]]
        self.assertEqual(record.task_plan.status, "completed")
        self.assertEqual(len(record.tool_results), 4)
        self.assertEqual(len(self.session.snapshot_interruption(record).completed_tool_call_ids), 4)
        self.assertIsNone(self.session.active_plan)
        trace = self.session.trace.report()["events"]
        self.assertNotIn("杭州", json.dumps(trace, ensure_ascii=False))
        self.assertEqual(len([event for event in trace if event["type"] == "tool.completed"]), 4)
        self.assertTrue(all(event["payload"].get("step_id") for event in trace if event["type"] == "tool.completed"))
        roles = len(self.provider.calls)
        repeated = await self.submit("重复一遍", "repeat")
        self.assertEqual(repeated["payload"]["text"], response["payload"]["text"])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.provider.calls), roles)

    async def test_domain_clarification_preserves_successful_dependencies(self):
        """第三步地点重名时保留时间和转换结果；补充后只执行剩余步骤。"""
        self.http.locations.append({**self.http.locations[0], "latitude": 30.06, "longitude": 102.19, "admin1": "四川"})
        first = await self.submit(COMPOUND_TEXT, "first")
        self.assertIn("同名地点", first["payload"]["text"])
        original_id = self.session.active_plan.plan_id
        self.assertEqual(self.session.pending_clarification.step_id, "step_3")
        self.provider.intent = estimate("tool_request", tool_intent="get_weather", continues_pending=True, slots=[
            {"name": "location", "value": "杭州, 浙江", "evidence": "杭州, 浙江"}])
        self.http.locations = self.http.locations[:1]
        second = await self.submit("杭州, 浙江", "second")
        record = self.session.outputs[second["generation_id"]]
        self.assertEqual(record.task_plan.plan_id, original_id)
        self.assertEqual(record.task_plan.status, "completed")
        self.assertEqual(Counter(name for name, _ in self.calls), {"get_current_time": 2, "convert_datetime": 1, "get_weather": 2})
        self.assertEqual(Counter(role for role, _ in self.provider.calls), {"intent": 2, "plan": 1, "answer": 1})
        self.assertEqual(len(self.session.snapshot_interruption(record).completed_tool_call_ids), 4)

    async def test_static_later_parameter_can_be_supplemented_before_first_step(self):
        """预检查暂停第三步缺项，补充后仍按第一步开始，保持整个目标。"""
        self.provider.draft["steps"][2].pop("arguments")
        self.provider.draft["steps"][2].pop("argument_evidence")
        await self.submit(COMPOUND_TEXT, "missing")
        self.assertEqual(self.calls, [])
        self.provider.intent = estimate("tool_request", tool_intent="get_weather", continues_pending=True, slots=[
            {"name": "location", "value": "杭州, 浙江", "evidence": "杭州, 浙江"}])
        await self.submit("杭州, 浙江", "filled")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(Counter(role for role, _ in self.provider.calls)["plan"], 1)

    async def test_failure_stops_dependent_tools_and_reports_partial_completion(self):
        """天气失败时不执行依赖当地时区的下一步，不伪报完整任务成功。"""
        self.http.status = 503
        response = await self.submit(COMPOUND_TEXT)
        record = self.session.outputs[response["generation_id"]]
        self.assertEqual(record.task_plan.status, "failed")
        self.assertEqual([name for name, _ in self.calls], ["get_current_time", "convert_datetime", "get_weather"])
        self.assertIn("任务未全部完成", response["payload"]["text"])
        self.assertNotIn("Open-Meteo", response["payload"]["text"])
        self.assertFalse(any(role == "answer" for role, _ in self.provider.calls))

    async def test_explicit_stop_clears_waiting_task_even_after_output_delivered(self):
        """已经追问且文字交付完成时，取消仍终止等待中的计划。"""
        self.provider.draft["steps"][2].pop("arguments")
        self.provider.draft["steps"][2].pop("argument_evidence")
        response = await self.submit(COMPOUND_TEXT)
        await self.ws.send("input.text", {"text": "取消"}, "stop")
        await self.ws.until("turn.controlled")
        self.assertIsNone(self.session.active_plan)
        self.assertIsNone(self.session.pending_clarification)
        self.assertEqual(self.session.outputs[response["generation_id"]].task_plan.status, "cancelled")

    async def test_late_cancelled_tool_cannot_start_next_step_or_replace_new_task(self):
        """供应商吞取消后返回也不能执行后续步骤或污染新任务结果。"""
        started = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()

        async def stubborn(arguments):
            """模拟不可靠查询，迟到数据只能被丢弃。"""
            started.set()
            try:
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return await CurrentTimeTool()(arguments)
            finally:
                closed.set()

        registry = ToolRegistry()
        registry.register(TIME_SPEC, stubborn)
        registry.register(DATETIME_SPEC, convert_datetime)
        registry.register(WEATHER_SPEC, WeatherTool(httpx.MockTransport(self.http)))
        self.app.state.tool_registry = registry
        self.app.state.agent = AgentGraph(self.provider, ToolExecutor(registry))
        await self.ws.send("input.text", {"text": COMPOUND_TEXT}, "old")
        await asyncio.wait_for(started.wait(), 2)
        old = self.session.generation_id
        await self.ws.send("control.stop", {"generation_id": old})
        await self.ws.until("output.cancel")
        self.provider.intent = estimate("tool_request", tool_intent="get_weather")
        await self.submit("查天气", "new")
        release.set()
        await asyncio.wait_for(closed.wait(), 2)
        self.assertEqual(self.session.outputs[old].task_plan.status, "cancelled")
        self.assertEqual(self.session.outputs[old].tool_results, [])
        self.assertEqual(self.session.pending_clarification.tool_name, "get_weather")
        self.assertIsNone(self.session.pending_clarification.plan_id)
        self.assertEqual(self.http.requests, [])

    async def test_completed_plan_uses_tts_pipeline_and_resume_never_runs_tools(self):
        """组合回答经已有语音链路输出；停止后继续复述结果，不重新执行计划。"""
        self.settings.tts_ready = True
        await self.ws.send("input.text", {"text": COMPOUND_TEXT}, "voice-output")
        plan_event = await self.ws.until("behavior.plan")
        audio = await self.ws.until("tts.chunk")
        generation = audio["generation_id"]
        self.assertEqual(plan_event["payload"]["timing"]["anchor"], "speech_start")
        self.assertEqual(len(self.session.outputs[generation].tool_results), 4)
        await self.ws.send("control.stop", {"generation_id": generation})
        cancellation = await self.ws.until("output.cancel")
        await self.ws.send("command.ack", {"generation_id": generation, "command_id": cancellation["payload"]["command_id"],
                                          "status": "completed", "clock_ms": 1000, "stop_elapsed_ms": 1})
        await self.ws.until("interruption.updated")
        count = len(self.provider.calls)
        await self.ws.send("input.text", {"text": "继续"}, "resume")
        response = await self.ws.until("response.completed", lambda event: event.get("turn_id") == "resume")
        self.assertIn("综合回答", response["payload"]["text"])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.provider.calls), count)

    async def test_disconnect_drops_waiting_goal(self):
        """等待用户输入的计划在断线时清理，不通过重连自动续查。"""
        self.provider.draft["steps"][2].pop("arguments")
        self.provider.draft["steps"][2].pop("argument_evidence")
        await self.submit(COMPOUND_TEXT)
        await self.ws.close()
        await asyncio.wait_for(self.task, 3)
        self.assertIsNone(self.session.active_plan)
        self.assertIsNone(self.session.pending_clarification)


if __name__ == "__main__":
    unittest.main()
