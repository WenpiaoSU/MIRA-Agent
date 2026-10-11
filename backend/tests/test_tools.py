"""作用：验证基础工具调用闭环及生产 Agent、网关的状态边界。

当前功能：
- 验证白名单、双向 Schema、总截止时间、取消与结果脱敏。
- 通过本地 HTTP 替身检查天气消歧、预报日期及异常数据。
- 检查多轮参数补充、真实工具目录、重复复述及 TTS 共用输出。
- 检查新任务和停止清理待澄清请求，迟到工具结果不污染会话。
"""

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

from src.agent.graph import AgentGraph
from src.api.realtime import realtime
from src.contracts.tools import PendingClarification, ToolCallRequest, ToolResult, ToolSpec
from src.gateway.sessions import SessionRegistry
from src.intent.service import IntentService
from src.tools.builtin.time import CurrentTimeTool, TIME_SPEC
from src.tools.builtin.weather import WeatherTool, WEATHER_SPEC, forecast_date
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry
from test_intent import ScriptedProvider, estimate
from test_p3 import FakeSocket, FakeSpeech, FakeTTS


def request(name: str, arguments: dict, seconds: float = 5) -> ToolCallRequest:
    """创建带真实 UTC 截止时间的工具请求。"""
    return ToolCallRequest(call_id="call-test", tool_name=name, arguments=arguments,
                           deadline_at=datetime.now(timezone.utc) + timedelta(seconds=seconds))


def tool_intent(name="get_weather", *, continuing=False, **arguments):
    """构造有当前原文证据的工具意图，缺失必填字段由分类器补充。"""
    return estimate("tool_request", tool_intent=name, continues_pending=continuing, slots=[
        {"name": key, "value": value, "evidence": value} for key, value in arguments.items()])


class WeatherHTTP:
    """根据生产请求返回地理和天气 JSON，记录实际查询参数。"""

    def __init__(self):
        """默认提供唯一杭州地点和新鲜、单位明确的当前天气或预报。"""
        self.requests = []
        self.locations = [{"name": "杭州", "latitude": 30.29, "longitude": 120.16,
                           "timezone": "Asia/Shanghai", "country": "中国", "admin1": "浙江"}]
        self.status = 200
        self.stale = False
        self.units = "°C"
        self.code = 2
        self.broken_daily = False

    def __call__(self, incoming: httpx.Request) -> httpx.Response:
        """响应固定官方端点；日期和时间相对本次测试动态生成。"""
        self.requests.append(incoming)
        if self.status != 200:
            return httpx.Response(self.status, json={"secret": "不应展示的供应商正文"})
        if incoming.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": self.locations})
        zone = ZoneInfo("Asia/Shanghai")
        now = datetime.now(zone)
        if "current" in incoming.url.params:
            valid = now - timedelta(hours=2) if self.stale else now
            return httpx.Response(200, json={"timezone": "Asia/Shanghai", "current_units": {"temperature_2m": self.units},
                                           "current": {"time": int(valid.timestamp()), "temperature_2m": 21.5,
                                                       "weather_code": self.code}})
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return httpx.Response(200, json={"timezone": "Asia/Shanghai", "daily_units": {
            "temperature_2m_max": self.units, "temperature_2m_min": self.units}, "daily": {
            "time": [int((midnight + timedelta(days=i)).timestamp()) for i in range(7)],
            "weather_code": [2] * 7, "temperature_2m_max": [23.0] * (6 if self.broken_daily else 7),
            "temperature_2m_min": [18.0] * 7}})


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    """检查执行器的真实校验、超时和取消，不访问网络。"""

    async def test_whitelist_and_input_validation_prevent_calls(self):
        """未知名称、额外参数和过期请求均不能运行处理函数。"""
        calls = []

        async def handler(arguments):
            """记录被允许执行的查询并返回合法结果。"""
            calls.append(arguments)
            return {"value": 1}

        registry = ToolRegistry()
        spec = ToolSpec(name="test", description="测试", input_schema={"type": "object", "additionalProperties": False},
                        output_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]})
        registry.register(spec, handler)
        with self.assertRaises(ValueError):
            registry.register(spec, handler)
        registry.list_tools()[0].input_schema["additionalProperties"] = True
        executor = ToolExecutor(registry)
        for name, arguments, seconds, code in [("unknown", {}, 5, "unavailable"),
                                               ("test", {"extra": True}, 5, "invalid_arguments"),
                                               ("test", {}, -1, "timeout")]:
            result = await executor.execute(request(name, arguments, seconds))
            self.assertEqual(result.error.code, code)
            self.assertFalse(result.verified)
        self.assertEqual(calls, [])
        result = await executor.execute(request("test", {}))
        self.assertTrue(result.verified)
        self.assertEqual(result.data, {"value": 1})

    async def test_invalid_output_and_nonfinite_data_are_never_verified(self):
        """返回结构错误、额外字段或 NaN 时不提供可展示事实。"""
        for output in ({"wrong": 1}, {"value": "错误"}, {"value": float("nan")}):
            async def handler(arguments):
                """返回当前子场景的非法供应商结果。"""
                return output
            registry = ToolRegistry()
            registry.register(ToolSpec(name="test", description="测试", input_schema={"type": "object"},
                                       output_schema={"type": "object", "required": ["value"],
                                                      "properties": {"value": {"type": "number"}}}), handler)
            result = await ToolExecutor(registry).execute(request("test", {}))
            self.assertEqual(result.error.code, "invalid_result")
            self.assertIsNone(result.data)
            self.assertFalse(result.verified)

    async def test_total_timeout_and_cancellation_close_handler(self):
        """截止时间涵盖整个处理函数，取消立即传播且不留下后台任务。"""
        closed = asyncio.Event()
        started = asyncio.Event()

        async def handler(arguments):
            """模拟慢服务，在超时或取消时释放资源。"""
            started.set()
            try:
                await asyncio.sleep(10)
                return {}
            finally:
                closed.set()

        registry = ToolRegistry()
        registry.register(ToolSpec(name="test", description="测试", input_schema={"type": "object"}), handler)
        result = await ToolExecutor(registry, timeout_ms=20).execute(request("test", {}))
        self.assertEqual(result.error.code, "timeout")
        self.assertTrue(closed.is_set())
        closed.clear()
        started.clear()
        task = asyncio.create_task(ToolExecutor(registry).execute(request("test", {})))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())

    async def test_timezone_default_explicit_and_invalid(self):
        """默认及明确时区返回带偏移的时间，无效时区不静默回退。"""
        registry = ToolRegistry()
        registry.register(TIME_SPEC, CurrentTimeTool())
        executor = ToolExecutor(registry)
        for arguments, name in [({}, "Asia/Shanghai"), ({"timezone": "Europe/London"}, "Europe/London")]:
            result = await executor.execute(request("get_current_time", arguments))
            self.assertEqual(result.data["timezone"], name)
            value = datetime.fromisoformat(result.data["iso_time"])
            self.assertLess(abs((datetime.now(timezone.utc) - value).total_seconds()), 3)
        result = await executor.execute(request("get_current_time", {"timezone": "火星时间"}))
        self.assertEqual(result.error.missing_slots, ["timezone"])
        with self.assertRaises(ValidationError):
            ToolCallRequest(call_id="id", tool_name="test", arguments={}, deadline_at=datetime.now())
        with self.assertRaises(ValidationError):
            ToolResult(call_id="id", tool_name="test", status="ok", data={}, verified=False, elapsed_ms=1)


class WeatherTests(unittest.IsolatedAsyncioTestCase):
    """通过本地 HTTP 传输验证单来源天气的领域边界。"""

    async def asyncSetUp(self):
        """装配真实天气函数与独立 HTTP 替身。"""
        self.http = WeatherHTTP()
        registry = ToolRegistry()
        registry.register(WEATHER_SPEC, WeatherTool(httpx.MockTransport(self.http)))
        self.executor = ToolExecutor(registry)

    async def test_current_weather_preserves_resolved_place_units_time_and_source(self):
        """当前天气带地点、温度单位、时区、有效时间与固定来源。"""
        result = await self.executor.execute(request("get_weather", {"location": "杭州，浙江"}))
        self.assertTrue(result.verified)
        self.assertEqual(result.data["location"], "杭州，浙江，中国")
        self.assertEqual(result.data["temperature"], 21.5)
        self.assertEqual(result.data["source"], "Open-Meteo")
        self.assertEqual(result.data["units"], "°C")
        self.assertEqual(result.data["condition"], "局部多云")
        self.assertEqual(result.data["kind"], "current")
        self.assertIn("+08:00", result.data["valid_at"])
        self.assertEqual(self.http.requests[0].url.params["name"], "杭州,浙江")

    async def test_ambiguous_or_missing_location_never_queries_weather(self):
        """重名与未知地点都先澄清，不选择第一项或猜测城市。"""
        self.http.locations.append({**self.http.locations[0], "latitude": 30.06, "longitude": 102.19, "admin1": "四川"})
        for locations, code in [(self.http.locations.copy(), "location_ambiguous"), ([], "location_not_found")]:
            self.http.locations = locations
            self.http.requests.clear()
            result = await self.executor.execute(request("get_weather", {"location": "杭州"}))
            self.assertEqual(result.error.code, code)
            self.assertEqual(result.error.missing_slots, ["location"])
            self.assertEqual(len(self.http.requests), 1)

    async def test_forecast_and_invalid_date(self):
        """按地点时区解释明天；模糊、过去及超范围日期必须补充。"""
        result = await self.executor.execute(request("get_weather", {"location": "杭州, 浙江", "date": "明天"}))
        tomorrow = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
        self.assertEqual(result.data["forecast"]["date"], tomorrow.isoformat())
        self.assertEqual(result.data["kind"], "forecast")
        self.assertIsNone(result.data["temperature"])
        self.assertEqual(result.data["forecast"]["temperature_min"], 18.0)
        for value in ("过几天", "2020-01-01", (tomorrow + timedelta(days=7)).isoformat()):
            result = await self.executor.execute(request("get_weather", {"location": "杭州", "date": value}))
            self.assertEqual(result.error.missing_slots, ["date"])
        self.assertEqual(forecast_date("明天", tomorrow), tomorrow + timedelta(days=1))

    async def test_bad_units_stale_codes_and_partial_forecasts_fail(self):
        """温度单位异常、旧数据、未知代码和预报缺项都不标记已验证。"""
        for name, value, arguments in [("units", "°F", {}), ("stale", True, {}), ("code", 999, {}),
                                        ("broken_daily", True, {"date": "明天"})]:
            with self.subTest(name=name):
                original = getattr(self.http, name)
                setattr(self.http, name, value)
                result = await self.executor.execute(request("get_weather", {"location": "杭州", **arguments}))
                self.assertEqual(result.error.code, "invalid_result")
                self.assertFalse(result.verified)
                self.assertIsNone(result.data)
                setattr(self.http, name, original)

    async def test_http_failure_is_safe_and_has_no_retry(self):
        """限流或服务故障不展示供应商正文，不自动再次查询。"""
        self.http.status = 429
        result = await self.executor.execute(request("get_weather", {"location": "杭州"}))
        self.assertEqual(result.error.code, "service_unavailable")
        self.assertNotIn("secret", result.model_dump_json())
        self.assertEqual(len(self.http.requests), 1)


class ToolGatewayTests(unittest.IsolatedAsyncioTestCase):
    """运行真实分类器、LangGraph 与 WebSocket 网关，隔离外部网络。"""

    async def asyncSetUp(self):
        """启动单会话及真实工具链，默认仅文字输出。"""
        sessions = SessionRegistry()
        self.session = sessions.create()
        self.http = WeatherHTTP()
        self.registry = ToolRegistry()
        self.registry.register(TIME_SPEC, CurrentTimeTool())
        self.registry.register(WEATHER_SPEC, WeatherTool(httpx.MockTransport(self.http)))
        self.executor = ToolExecutor(self.registry)
        self.provider = ScriptedProvider()
        self.settings = SimpleNamespace(llm_ready=True, tts_ready=False, stop_ack_timeout_ms=500)
        self.app = SimpleNamespace(state=SimpleNamespace(
            sessions=sessions, agent=AgentGraph(self.provider, self.executor), tool_registry=self.registry,
            intent_service=IntentService(self.provider), tts=FakeTTS(), speech_service=FakeSpeech(), settings=self.settings))
        self.ws = FakeSocket(self.app, self.session)
        self.task = asyncio.create_task(realtime(self.ws, self.session.session_id))
        await self.ws.until("session.status")

    async def asyncTearDown(self):
        """关闭网关并回收所有连接任务。"""
        await self.ws.close()
        await asyncio.wait_for(self.task, 3)

    async def submit(self, text, intent, turn_id="turn-test"):
        """提供本轮分类 JSON 并等待真实回复交付完成。"""
        self.provider.raw = intent.model_dump_json()
        await self.ws.send("input.text", {"text": text}, turn_id)
        response = await self.ws.until("response.completed")
        await self.ws.until("turn.delivered")
        return response

    async def test_two_turn_clarification_merges_date_and_uses_real_catalog(self):
        """天气缺地点时保留明天参数；补充地点后执行原查询且不再调用回复模型。"""
        first = await self.submit("查明天的天气", tool_intent(date="明天"), "first")
        self.assertIn("城市", first["payload"]["text"])
        self.assertEqual(self.session.pending_clarification.arguments, {"date": "明天"})
        self.assertEqual(self.http.requests, [])
        second = await self.submit("杭州, 浙江", tool_intent(continuing=True, location="杭州, 浙江"), "second")
        self.assertIn("预报", second["payload"]["text"])
        self.assertIsNone(self.session.pending_clarification)
        self.assertEqual(len(self.provider.messages), 2)
        sent = json.loads(self.provider.messages[-1][1]["content"])
        self.assertEqual({spec["name"] for spec in sent["available_tools"]}, {"get_current_time", "get_weather"})
        self.assertTrue(all("output_schema" not in spec for spec in sent["available_tools"]))
        self.assertEqual(sent["pending_clarification"]["original_text"], "查明天的天气")
        record = self.session.outputs[second["generation_id"]]
        self.assertTrue(record.tool_result.verified)
        snapshot = self.session.snapshot_interruption(record)
        self.assertEqual(snapshot.completed_tool_call_ids, [record.tool_result.call_id])
        trace = self.session.trace.report()["events"]
        completed = [item for item in trace if item["type"] == "tool.completed"]
        self.assertEqual(len(completed), 1)
        self.assertNotIn("杭州", json.dumps(trace, ensure_ascii=False))

    async def test_ambiguous_location_keeps_date_for_correction(self):
        """同名地点要求纠正 location，已有日期仍保留在待补充请求中。"""
        self.http.locations.append({**self.http.locations[0], "latitude": 30.06, "longitude": 102.19, "admin1": "四川"})
        first = await self.submit("查杭州明天天气", tool_intent(location="杭州", date="明天"))
        self.assertIn("同名地点", first["payload"]["text"])
        self.assertEqual(self.session.pending_clarification.arguments, {"date": "明天"})
        self.http.locations = self.http.locations[:1]
        second = await self.submit("杭州, 浙江", tool_intent(continuing=True, location="杭州, 浙江"))
        self.assertIn("预报", second["payload"]["text"])

    async def test_new_request_does_not_inherit_old_date_and_repeat_never_requeries(self):
        """独立新天气请求不沿用待补充的明天；重复多次只复述旧结果。"""
        await self.submit("查明天天气", tool_intent(date="明天"))
        response = await self.submit("查杭州, 浙江天气", tool_intent(location="杭州, 浙江"))
        self.assertIn("当前", response["payload"]["text"])
        count = len(self.http.requests)
        for _ in range(2):
            repeated = await self.submit("重复一遍", estimate("control", control_action="repeat"))
            self.assertIn("21.5°C", repeated["payload"]["text"])
        self.assertEqual(len(self.http.requests), count)
        self.assertEqual(len(self.provider.messages), 2)

    async def test_stop_and_change_topic_clear_pending(self):
        """停止快捷路径与换话题清除待补充目标，下一任务不会继承旧槽位。"""
        await self.submit("查明天天气", tool_intent(date="明天"))
        await self.ws.send("input.text", {"text": "取消"}, "stop")
        await self.ws.until("turn.controlled")
        self.assertIsNone(self.session.pending_clarification)
        await self.submit("查天气", tool_intent())
        self.provider.raw = '{"expression":"neutral","gaze":"center","gesture":null,"intensity":0.6}\n可以。'
        await self.ws.send("input.text", {"text": "换个话题"}, "change")
        await self.ws.until("turn.delivered")
        self.assertIsNone(self.session.pending_clarification)

    async def test_late_cancelled_tool_cannot_overwrite_pending_or_emit_output(self):
        """模拟吞取消的慢工具，旧结果仍被 generation 校验过滤。"""
        started = asyncio.Event()
        release = asyncio.Event()

        async def stubborn(arguments):
            """故意吞取消，再迟到返回合法时间数据。"""
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return await CurrentTimeTool()(arguments)

        registry = ToolRegistry()
        registry.register(TIME_SPEC, stubborn)
        registry.register(WEATHER_SPEC, WeatherTool(httpx.MockTransport(self.http)))
        self.app.state.tool_registry = registry
        self.app.state.agent = AgentGraph(self.provider, ToolExecutor(registry))
        self.provider.raw = tool_intent("get_current_time").model_dump_json()
        await self.ws.send("input.text", {"text": "现在几点"}, "old")
        await asyncio.wait_for(started.wait(), 2)
        await self.submit("查明天天气", tool_intent(date="明天"), "new")
        release.set()
        for _ in range(20):
            if all(record.tool_result is None for record in self.session.outputs.values()):
                await asyncio.sleep(.01)
        self.assertEqual(self.session.pending_clarification.arguments, {"date": "明天"})
        self.assertTrue(all(record.tool_result is None for record in self.session.outputs.values()))
        remaining = []
        while not self.ws.outgoing.empty():
            remaining.append(self.ws.outgoing.get_nowait())
        self.assertFalse(any(item["turn_id"] == "old" and item["type"].startswith("response.") for item in remaining))

    async def test_time_result_reuses_tts_pipeline(self):
        """时间事实经同一行为、语段和 TTS 链路输出，不调用回复模型。"""
        self.settings.tts_ready = True
        self.provider.raw = tool_intent("get_current_time").model_dump_json()
        await self.ws.send("input.text", {"text": "现在几点"}, "time")
        plan = await self.ws.until("behavior.plan")
        audio = await self.ws.until("tts.chunk")
        self.assertEqual(plan["payload"]["timing"]["anchor"], "speech_start")
        self.assertTrue(self.session.outputs[audio["generation_id"]].tool_result.verified)
        self.assertEqual(len(self.provider.messages), 1)

    async def test_failed_service_replies_safely_without_llm_facts(self):
        """天气服务失败时只说明未完成，不进入回复模型编造温度。"""
        self.http.status = 503
        response = await self.submit("查杭州天气", tool_intent(location="杭州"))
        self.assertIn("尚未获取结果", response["payload"]["text"])
        self.assertNotIn("°C", response["payload"]["text"])
        self.assertEqual(len(self.provider.messages), 1)
        self.assertFalse(self.session.outputs[response["generation_id"]].tool_result.verified)

    async def test_classifier_failure_preserves_pending_and_false_continuation_is_rejected(self):
        """分类故障保留原任务；没有原任务时不能接受模型声称继续补参。"""
        await self.submit("查明天天气", tool_intent(date="明天"))
        self.provider.raw = "不是 JSON"
        await self.ws.send("input.text", {"text": "杭州"}, "bad-classification")
        await self.ws.until("turn.delivered")
        self.assertEqual(self.session.pending_clarification.arguments, {"date": "明天"})
        intent = tool_intent(continuing=True, location="杭州")
        result = await IntentService(ScriptedProvider(intent.model_dump())).classify(
            "杭州", "", self.registry.list_tools(), source="text")
        self.assertEqual(result.intent, "other")

    async def test_repeat_after_new_tool_result_uses_latest_result(self):
        """已经重复过旧天气后再查询时间，下一次重复应使用最新时间结果。"""
        await self.submit("查杭州天气", tool_intent(location="杭州"))
        await self.submit("重复一遍", estimate("control", control_action="repeat"))
        current = await self.submit("现在几点", tool_intent("get_current_time"))
        repeated = await self.submit("重复一遍", estimate("control", control_action="repeat"))
        self.assertIn(current["payload"]["text"], repeated["payload"]["text"])
        self.assertNotIn("°C", repeated["payload"]["text"])

    async def test_stop_during_tool_execution_cancels_http_and_no_result_is_committed(self):
        """停止按钮在等待天气时立即取消查询，不产生结果或遗留待补充任务。"""
        started = asyncio.Event()
        closed = asyncio.Event()

        async def slow_http(incoming):
            """模拟可取消网络请求，并记录连接任务回收。"""
            started.set()
            try:
                await asyncio.sleep(10)
                return self.http(incoming)
            finally:
                closed.set()

        registry = ToolRegistry()
        registry.register(WEATHER_SPEC, WeatherTool(httpx.MockTransport(slow_http)))
        self.app.state.tool_registry = registry
        self.app.state.agent = AgentGraph(self.provider, ToolExecutor(registry))
        self.provider.raw = tool_intent(location="杭州").model_dump_json()
        await self.ws.send("input.text", {"text": "查杭州天气"}, "slow")
        await asyncio.wait_for(started.wait(), 2)
        old_generation = self.session.generation_id
        await self.ws.send("control.stop", {"generation_id": old_generation})
        await self.ws.until("output.cancel")
        await asyncio.wait_for(closed.wait(), 1)
        self.assertIsNone(self.session.generation_id)
        self.assertIsNone(self.session.pending_clarification)
        self.assertIsNone(self.session.outputs[old_generation].tool_result)

    async def test_resume_after_stop_reuses_completed_tool_fact(self):
        """查询已完成但播报被停止时，继续只复述账本结果，不再次查天气。"""
        self.settings.tts_ready = True
        self.provider.raw = tool_intent(location="杭州").model_dump_json()
        await self.ws.send("input.text", {"text": "查杭州天气"}, "before-stop")
        audio = await self.ws.until("tts.chunk")
        generation = audio["generation_id"]
        calls = len(self.http.requests)
        await self.ws.send("control.stop", {"generation_id": generation})
        cancellation = await self.ws.until("output.cancel")
        await self.ws.send("command.ack", {"generation_id": generation,
                                           "command_id": cancellation["payload"]["command_id"],
                                           "status": "completed", "clock_ms": 1000, "stop_elapsed_ms": 2})
        await self.ws.until("interruption.updated")
        await self.ws.send("input.text", {"text": "继续"}, "resume")
        response = await self.ws.until("response.completed", lambda event: event.get("turn_id") == "resume")
        self.assertIn("21.5°C", response["payload"]["text"])
        self.assertEqual(len(self.http.requests), calls)
        self.assertEqual(len(self.provider.messages), 1)


if __name__ == "__main__":
    unittest.main()
