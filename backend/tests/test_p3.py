"""作用：验证 P3 实时控制、交付账本与异步语音的关键边界。

当前功能：
- 检查打断确认、回声风险、旧回执隔离及句子级续说。
- 验证慢 ASR final 不阻塞 VAD，乱序 final 保持原片段顺序。
- 通过真实网关协程验证持续轮次、取消、停止确认和迟到 TTS。
- 验证输出背压、故障降级、恢复上下文和确定性回放。
"""

import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from src.agent.graph import AgentGraph
from src.agent.recovery import control_action
from src.api.realtime import realtime
from src.evaluation.trace import replay_turn
from src.gateway.sessions import Session, SessionRegistry
from src.output.coordinator import OutputCoordinator
from src.speech.asr import ASRResult, ASRStream
from src.speech.frontend import validate_frame
from src.speech.service import SpeechStream
from src.turn.manager import TurnManager


class TurnTests(unittest.TestCase):
    """验证确定性打断门控，不依赖 ASR 或语言模型。"""

    def test_barge_in_requires_real_playback_and_over_600ms(self):
        """只有实际播放中的持续有效语音超过窗口才确认，且每代只确认一次。"""
        turn = TurnManager()
        turn.configure(True, "browser")
        turn.begin("g")
        self.assertFalse(turn.vad("start", 0, 700, .9))
        turn.playback("g", "s", "started")
        self.assertFalse(turn.vad("start", 1000, 1020, .9))
        self.assertFalse(turn.vad("active", 1020, 1600, .9))
        self.assertTrue(turn.vad("active", 1600, 1620, .9))
        self.assertFalse(turn.vad("active", 1620, 1700, .9))

    def test_echo_clipping_and_missing_aec_never_confirm(self):
        """无 AEC、明确回声风险、削波及短噪声不能累计为插话。"""
        for aec, echo, clipping in [("unavailable", 0, 0), ("browser", .9, 0), ("browser", 0, .5)]:
            turn = TurnManager()
            turn.configure(True, aec)
            turn.begin("g")
            turn.playback("g", "s", "started")
            self.assertFalse(turn.vad("start", 0, 1000, .9, echo, clipping))
        turn = TurnManager()
        turn.configure(True, "browser")
        turn.begin("g")
        turn.playback("g", "s", "started")
        turn.vad("start", 0, 400, .9)
        turn.vad("active", 400, 420, .1)
        self.assertFalse(turn.vad("active", 420, 800, .9))
        turn.playback("g", "s", "paused")
        self.assertFalse(turn.vad("active", 800, 1600, .9))

    def test_replay_uses_recorded_capture_clock(self):
        """回放无需实际等待，重放同样的控制事件得到同一次打断。"""
        events = [
            {"type": "audio.configured", "payload": {"continuous": True, "aec_mode": "browser"}},
            {"type": "generation.started", "payload": {"generation_id": "g"}},
            {"type": "playback.started", "payload": {"generation_id": "g", "segment_id": "s"}},
            {"type": "vad.event", "payload": {"phase": "start", "start_ms": 100, "end_ms": 120, "p_speech": .9}},
            {"type": "vad.event", "payload": {"phase": "active", "start_ms": 120, "end_ms": 740, "p_speech": .9}},
            {"type": "output.cancel", "payload": {"generation_id": "g"}},
            {"type": "output.stop_confirmed", "payload": {"generation_id": "g"}},
        ]
        report = replay_turn(events)
        self.assertEqual(report["confirmed_generations"], ["g"])
        self.assertFalse(report["interaction"]["robot_playing"])


class LedgerTests(unittest.TestCase):
    """验证已生成、已展示、已说出与取消位置不会混淆。"""

    def make_record(self):
        """构造包含一个已播完句子与一个正在播放句子的输出。"""
        session = Session(connection_epoch=1)
        session.begin_generation("g1", "t1", "解释方法", "trace", True)
        session.register_segment("g1", {"segment_id": "s1", "text": "第一句。", "index": 1})
        session.register_segment("g1", {"segment_id": "s2", "text": "第二句。", "index": 2})
        session.register_audio("s1", 1000)
        session.register_audio("s2", 1000)
        session.displayed("g1", "s1")
        session.displayed("g1", "s2")
        session.record_playback("s1", "playback.ended", 1000, 1000)
        session.record_playback("s2", "playback.started", 0, 1000)
        session.record_playback("s2", "playback.progress", 300, 1000)
        return session

    def test_old_final_receipt_updates_snapshot_not_new_turn(self):
        """新代次启动后，旧 stopped 修正快照但不停止新音源。"""
        session = self.make_record()
        session.cancel_generation("barge_in")
        command = session.outputs["g1"].cancel_command_id
        session.begin_generation("g2", "t2", "继续")
        session.turn.playback("g2", "new", "started")
        self.assertTrue(session.record_playback("s2", "playback.stopped", 500, 1000, "g1"))
        self.assertTrue(session.confirm_stop("g1", command))
        self.assertTrue(session.turn.robot_playing)
        self.assertEqual(session.generation_id, "g2")
        self.assertEqual(session.interruption.spoken_text, "第一句。")
        self.assertEqual(session.interruption.remaining_text, "第二句。")
        self.assertEqual(session.interruption.played_audio_ms, 500)
        self.assertEqual(session.interruption.spoken_text_precision, "approximate")
        self.assertTrue(session.interruption.stop_confirmed)
        self.assertFalse(session.confirm_stop("g1", command))
        self.assertFalse(session.record_playback("s2", "playback.progress", 900, 1000, "g1"))

    def test_progress_never_rewinds_and_epoch_isolated(self):
        """倒退进度不回滚位置，错误时长、错误命令及旧 epoch 被拒绝。"""
        session = self.make_record()
        session.record_playback("s2", "playback.progress", 100, 1000)
        self.assertEqual(session.outputs["g1"].segments["s2"].played_ms, 300)
        self.assertFalse(session.record_playback("s2", "playback.progress", 500, 999))
        session.cancel_generation()
        self.assertFalse(session.confirm_stop("g1", "wrong"))
        session.connection_epoch = 2
        self.assertFalse(session.record_playback("s2", "playback.stopped", 500, 1000, "g1"))

    def test_invalid_ended_cannot_advance_position_and_timeout_rejects_receipt(self):
        """无效结束事件不污染交付位置；超时后不接受旧停止回执。"""
        session = self.make_record()
        self.assertFalse(session.record_playback("s2", "playback.ended", 800, 1000))
        self.assertEqual(session.outputs["g1"].segments["s2"].played_ms, 300)
        session.cancel_generation()
        record = session.outputs["g1"]
        record.cancelled_at_ms -= session.stop_ack_timeout_ms + 1
        self.assertFalse(session.record_playback("s2", "playback.stopped", 500, 1000, "g1"))
        self.assertFalse(session.confirm_stop("g1", record.cancel_command_id))

    def test_failure_is_degraded_not_delivered_success(self):
        """音频失败释放等待但保留降级状态，不能声称完整说出。"""
        session = Session(connection_epoch=1)
        record = session.begin_generation("g", "t", "问题", speech_enabled=True)
        session.register_segment("g", {"segment_id": "s", "text": "回答。"})
        session.register_audio("s", 1000)
        session.displayed("g", "s")
        record.response_complete = record.synthesis_complete = True
        self.assertIsNone(session.finish_if_delivered())
        session.record_playback("s", "playback.failed", 0, 1000)
        self.assertEqual(session.finish_if_delivered().status, "degraded")
        self.assertIn("完整播完的句子=无", session.history[-1]["content"])

    def test_cancelled_input_remains_in_history_and_archive_bounded(self):
        """生成中取消不丢用户输入；历史账本不会无限增长。"""
        session = Session()
        session.begin_generation("g", "t", "原始问题")
        session.cancel_generation()
        self.assertEqual(session.history[0]["content"], "原始问题")
        for index in range(30):
            session.begin_generation(str(index), str(index), "问题")
            session.cancel_generation()
        self.assertLessEqual(len(session.outputs), 24)


class FakeVad:
    """为语音并发测试提供预定 VAD 阶段。"""

    def __init__(self, phases):
        """保存每帧的确定性阶段。"""
        self.phases = iter(phases)

    def feed(self, samples):
        """按帧返回活动，忽略测试中的静音波形。"""
        return next(self.phases, [])

    def reset(self):
        """提供与真实模型一致的片段重置接口。"""
        pass


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    """验证真实 SpeechStream 的网络隔离及片段顺序。"""

    async def test_slow_final_does_not_block_next_vad_or_reorder_final(self):
        """两个 final 逆序完成时，VAD 继续且正文仍按原片段顺序提交。"""
        gates = [asyncio.Event(), asyncio.Event()]
        started = [asyncio.Event(), asyncio.Event()]

        class Provider:
            """让转写任务分别等待可控闸门。"""
            count = 0

            async def transcribe(self, pcm):
                """记录启动顺序，模拟独立网络完成时间。"""
                index = self.count
                self.count += 1
                started[index].set()
                await gates[index].wait()
                return ASRResult(["第一段", "第二段"][index])

        provider = Provider()
        vad = FakeVad([[("start", .9)], [("end", .1)], [("start", .9)], [("end", .1)]])
        stream = SpeechStream("s", vad, ASRStream(provider))
        try:
            for index in range(4):
                events = await asyncio.wait_for(stream.feed(validate_frame("s", index, index * 20, (index + 1) * 20, bytes(640))), .5)
                if index == 2:
                    self.assertTrue(any(kind == "vad.event" and payload["phase"] == "start" for kind, payload in events))
            await asyncio.wait_for(started[1].wait(), .5)
            gates[1].set()
            await asyncio.sleep(.01)
            self.assertFalse(any(kind == "asr.final" for kind, _ in await stream.poll()))
            gates[0].set()
            await asyncio.sleep(.01)
            finals = await stream.poll()
            self.assertEqual([payload["text"] for kind, payload in finals if kind == "asr.final"], ["第一段", "第二段"])
            stream.last_end_ms = 1000
            completed = await stream.poll()
            self.assertEqual(completed[-1][1]["text"], "第一段第二段")
            self.assertEqual(await stream.poll(), [])
        finally:
            await stream.close()

    async def test_incomplete_final_extends_eot_but_partial_cannot_submit(self):
        """未完成 final 只有限延长等待，空 final/partial 不能提交用户轮次。"""
        stream = SpeechStream("s", FakeVad([]), ASRStream(SimpleNamespace()))
        try:
            stream.last_valid_ms = 100
            stream.last_end_ms = 900
            self.assertEqual(await stream.poll(), [])
            stream.parts = [("u", "因为")]
            self.assertEqual(await stream.poll(), [])
            stream.last_end_ms = 1500
            events = await stream.poll()
            self.assertEqual(events[-1][0], "turn.completed")
            self.assertEqual(events[-1][1]["text"], "因为")
            self.assertEqual(await stream.poll(), [])
        finally:
            await stream.close()

    async def test_close_cancels_asr_children(self):
        """关闭流时回收未结束 final 请求，不遗留网络子任务。"""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class Provider:
            """模拟永不自行完成的转写请求。"""

            async def transcribe(self, pcm):
                """等待取消并记录子任务确实退出。"""
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        stream = SpeechStream("s", FakeVad([[("start", .9)], [("end", .1)]]), ASRStream(Provider()))
        await stream.feed(validate_frame("s", 0, 0, 20, bytes(640)))
        await stream.feed(validate_frame("s", 1, 20, 40, bytes(640)))
        await asyncio.wait_for(started.wait(), .5)
        await stream.close()
        self.assertTrue(cancelled.is_set())


class FakeSocket:
    """通过队列模拟双向 WebSocket，所有测试读取都有截止时间。"""

    def __init__(self, app, session):
        """绑定应用、会话及网络事件队列。"""
        self.app = app
        self.session = session
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.seq = 0
        self.closed = False

    async def accept(self):
        """模拟连接握手完成。"""
        pass

    async def receive(self):
        """接收测试注入的网络事件。"""
        return await self.incoming.get()

    async def send_json(self, event):
        """记录网关真正发出的 JSON 事件。"""
        await self.outgoing.put(event)

    async def close(self, **kwargs):
        """模拟连接关闭并唤醒网关接收任务。"""
        self.closed = True
        await self.incoming.put({"type": "websocket.disconnect"})

    async def send(self, kind, payload, turn_id=None, epoch=None):
        """注入符合生产契约的客户端事件或二进制音频。"""
        event = {"schema_version": "1.0", "event_id": str(uuid4()), "type": kind, "source": "avatar-web",
                 "session_id": self.session.session_id, "connection_epoch": epoch or self.session.connection_epoch,
                 "seq": self.seq, "occurred_at": datetime.now(timezone.utc).isoformat(),
                 "time_ref": {"clock_id": "browser-performance", "capture_start_ms": 0, "capture_end_ms": 0},
                 "trace_id": "test-trace", "payload": payload}
        self.seq += 1
        if turn_id:
            event["turn_id"] = turn_id
        if kind == "input.audio_frame":
            header = json.dumps(event).encode()
            message = {"type": "websocket.receive", "bytes": len(header).to_bytes(2, "big") + header + bytes(640)}
        else:
            message = {"type": "websocket.receive", "text": json.dumps(event)}
        await self.incoming.put(message)

    async def until(self, kind, predicate=lambda event: True):
        """有界等待目标事件，缺失事件直接使测试失败。"""
        for _ in range(200):
            event = await asyncio.wait_for(self.outgoing.get(), 2)
            if event["type"] == kind and predicate(event):
                return event
        raise AssertionError("目标事件未出现：" + kind)


class RecordingAgent:
    """记录恢复输入并生成固定、合法的稳定语段。"""

    def __init__(self):
        """保存调用及冻结上下文。"""
        self.calls = []

    async def stream(self, text, history, cues, generation_id, runtime_context=None):
        """记录调用并输出一段正文和生成完成事件。"""
        self.calls.append((text, runtime_context, history))
        yield {"type": "response.segment", "payload": {"segment_id": "s1", "index": 1, "text": "解释内容。"}}
        yield {"type": "response.completed", "payload": {"text": "解释内容。", "response_id": "r"}}


class FakeTTS:
    """返回固定音频元数据，网络隔离测试不访问真实供应商。"""

    async def synthesize(self, text, expression):
        """构造可注册和播放反馈的测试音频。"""
        return SimpleNamespace(data=b"RIFFtest", duration_ms=2000, sample_rate=16000, channels=1)


class ActivityStream:
    """提供连续 VAD 和显式 final，检查网关不提前调用 Agent。"""

    async def feed(self, frame):
        """前 32 帧为活动，第 33 帧产生 final 与完整轮次。"""
        if frame.seq == 32:
            return [("vad.event", {"phase": "end", "start_ms": 1640, "end_ms": 1660, "p_speech": .1}),
                    ("asr.final", {"text": "继续", "utterance_id": "u", "revision": 2}),
                    ("turn.completed", {"text": "继续", "reason": "silence"})]
        return [("vad.event", {"phase": "start" if frame.seq == 0 else "active",
                             "start_ms": frame.capture_start_ms, "end_ms": frame.capture_end_ms, "p_speech": .9}),
                ("asr.partial", {"text": "临时假设", "revision": 1, "utterance_id": "u"})]

    async def poll(self):
        """无后台转写时不产生额外事件。"""
        return []

    async def finish(self):
        """测试中显式结束不追加正文。"""
        return []

    async def close(self):
        """提供关闭接口供网关统一回收。"""
        pass


class FakeSpeech:
    """为网关测试声明可用的独立语音服务。"""
    ready = True

    def create_stream(self, stream_id):
        """每次创建独立活动流。"""
        return ActivityStream()


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    """覆盖生产网关的并发、取消及新旧输出交接。"""

    async def asyncSetUp(self):
        """创建无外部请求的应用与网关任务。"""
        registry = SessionRegistry()
        self.session = registry.create()
        self.agent = RecordingAgent()
        self.app = SimpleNamespace(state=SimpleNamespace(
            sessions=registry, agent=self.agent, tts=FakeTTS(), speech_service=FakeSpeech(),
            settings=SimpleNamespace(llm_ready=True, tts_ready=True, stop_ack_timeout_ms=500)))
        self.ws = FakeSocket(self.app, self.session)
        self.task = asyncio.create_task(realtime(self.ws, self.session.session_id))
        await self.ws.until("session.status")

    async def asyncTearDown(self):
        """关闭连接并确认网关及全部子任务可退出。"""
        await self.ws.close()
        await asyncio.wait_for(self.task, 3)

    async def start_reply(self):
        """提交文本并取得实际可播放音频与 generation。"""
        await self.ws.send("input.text", {"text": "解释方法"}, "t1")
        chunk = await self.ws.until("tts.chunk")
        generation = chunk["generation_id"]
        await self.ws.send("response.displayed", {"generation_id": generation, "segment_id": "s1"})
        await self.ws.send("playback.started", {"generation_id": generation, "segment_id": "s1",
                           "played_ms": 0, "duration_ms": 2000, "clock_ms": 1000})
        await self.ws.until("turn.state", lambda event: event["payload"]["state"] == "speaking")
        return generation

    async def test_barge_in_then_final_resume_keeps_capture_and_snapshot(self):
        """播放中持续插话直接停音；partial 不推理，final 使用修正快照续说。"""
        generation = await self.start_reply()
        await self.ws.send("input.audio_start", {"stream_id": "voice", "encoding": "pcm_s16le",
                           "sample_rate": 16000, "channels": 1, "continuous": True, "aec_mode": "browser"})
        await self.ws.until("turn.state", lambda event: event["payload"].get("audio_ready"))
        for index in range(32):
            await self.ws.send("input.audio_frame", {"stream_id": "voice", "frame_seq": index,
                               "capture_start_ms": 1000 + index * 20, "capture_end_ms": 1020 + index * 20})
        cancel = await self.ws.until("output.cancel")
        self.assertEqual(cancel["generation_id"], generation)
        self.assertEqual(cancel["payload"]["reason"], "barge_in")
        self.assertEqual(len(self.agent.calls), 1)
        await self.ws.send("playback.stopped", {"generation_id": generation, "segment_id": "s1",
                           "played_ms": 600, "duration_ms": 2000, "clock_ms": 1650})
        await self.ws.send("command.ack", {"generation_id": generation,
                           "command_id": cancel["payload"]["command_id"], "status": "completed",
                           "clock_ms": 1650, "stop_elapsed_ms": 1, "speech_to_stop_ms": 650})
        await self.ws.until("interruption.updated")
        await self.ws.send("input.audio_frame", {"stream_id": "voice", "frame_seq": 32,
                           "capture_start_ms": 1640, "capture_end_ms": 1660})
        new_chunk = await self.ws.until("tts.chunk")
        self.assertNotEqual(new_chunk["generation_id"], generation)
        self.assertTrue(self.session.turn.continuous)
        text, context, history = self.agent.calls[-1]
        self.assertEqual(text, "继续")
        self.assertTrue(context["interruption_state"]["stop_confirmed"])
        self.assertEqual(context["interruption_state"]["played_audio_ms"], 600)
        self.assertEqual(context["interruption_state"]["remaining_text"], "解释内容。")
        self.assertEqual(self.session.trace.report()["stop_ack_samples"], 1)

    async def test_new_generation_rejects_old_epoch_and_old_audio_start(self):
        """错误 epoch 不触发 Agent，无 AEC 的采集不能覆盖当前回复。"""
        generation = await self.start_reply()
        await self.ws.send("input.text", {"text": "不应执行"}, "bad", epoch=999)
        self.assertEqual((await self.ws.until("error"))["payload"]["code"], "invalid_event")
        self.assertEqual(len(self.agent.calls), 1)
        await self.ws.send("input.audio_start", {"stream_id": "voice", "encoding": "pcm_s16le",
                           "sample_rate": 16000, "channels": 1, "continuous": True, "aec_mode": "unavailable"})
        self.assertEqual((await self.ws.until("error"))["payload"]["code"], "audio_unavailable")
        self.assertEqual(self.session.generation_id, generation)

    async def test_duplicate_stop_is_idempotent_and_receipt_can_follow_new_input(self):
        """旧取消命令重复发送仍用同一 ID，停止回执不释放新一轮。"""
        old = await self.start_reply()
        await self.ws.send("control.stop", {"generation_id": old})
        first = await self.ws.until("output.cancel")
        await self.ws.send("control.stop", {"generation_id": old})
        second = await self.ws.until("output.cancel")
        self.assertEqual(first["payload"]["command_id"], second["payload"]["command_id"])
        await self.ws.send("input.text", {"text": "继续"}, "t2")
        await self.ws.send("playback.stopped", {"generation_id": old, "segment_id": "s1",
                           "played_ms": 500, "duration_ms": 2000, "clock_ms": 1500})
        await self.ws.send("command.ack", {"generation_id": old, "command_id": first["payload"]["command_id"],
                           "status": "completed", "clock_ms": 1500, "stop_elapsed_ms": 1})
        chunk = await self.ws.until("tts.chunk")
        self.assertNotEqual(chunk["generation_id"], old)
        self.assertEqual(self.session.generation_id, chunk["generation_id"])
        self.assertTrue(self.agent.calls[-1][1]["interruption_state"]["stop_confirmed"])

    async def test_stop_timeout_preserves_uncertainty_and_resume_does_not_stall(self):
        """停止回执丢失会产生超时事件，恢复保留不确定标志且能继续新输出。"""
        old = await self.start_reply()
        await self.ws.send("control.stop", {"generation_id": old})
        await self.ws.until("output.cancel")
        timeout = await self.ws.until("output.stop_timeout")
        self.assertEqual(timeout["generation_id"], old)
        self.assertTrue(self.session.outputs[old].stop_timed_out)
        self.assertFalse(self.session.interruption.stop_confirmed)
        await self.ws.send("input.text", {"text": "继续"}, "t2")
        chunk = await self.ws.until("tts.chunk")
        self.assertNotEqual(chunk["generation_id"], old)
        self.assertFalse(self.agent.calls[-1][1]["interruption_state"]["stop_confirmed"])

    async def test_disabled_duplex_falls_back_to_single_turn(self):
        """服务端关闭双工时，即使设备声明 AEC 也只开放单轮采集。"""
        self.app.state.settings.full_duplex_enabled = False
        await self.ws.send("input.audio_start", {"stream_id": "voice", "encoding": "pcm_s16le",
                           "sample_rate": 16000, "channels": 1, "continuous": True, "aec_mode": "browser"})
        ready = await self.ws.until("turn.state", lambda event: event["payload"].get("audio_ready"))
        self.assertFalse(ready["payload"]["continuous"])
        self.assertFalse(self.session.turn.continuous)
        await self.ws.send("input.audio_end", {"stream_id": "voice", "submit": False})
        await self.ws.until("audio.closed")

    async def test_takeover_invalidates_old_connection_and_cleans_old_playback(self):
        """同会话新连接取得所有权后，旧网关退出且旧播放不留在实时状态中。"""
        old = await self.start_reply()
        other = FakeSocket(self.app, self.session)
        task = asyncio.create_task(realtime(other, self.session.session_id))
        try:
            status = await other.until("session.status")
            self.assertEqual(status["connection_epoch"], 2)
            await asyncio.wait_for(self.task, 2)
            self.assertTrue(self.ws.closed)
            self.assertIsNone(self.session.generation_id)
            self.assertFalse(self.session.turn.robot_playing)
            self.assertEqual(self.session.outputs[old].cancel_reason, "takeover")
            self.assertFalse(self.session.confirm_stop(old, self.session.outputs[old].cancel_command_id))
        finally:
            await other.close()
            await asyncio.wait_for(task, 2)

    async def test_late_tts_after_cancel_is_never_sent(self):
        """供应商即使在取消时返回结果，旧音频也无法提交到新代次。"""
        started = asyncio.Event()

        class LateTTS:
            """模拟请求已经完成但取消刚好抵达的竞态。"""

            async def synthesize(self, text, expression):
                """第一次等待取消后仍返回数据，之后立即返回。"""
                if not started.is_set():
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        pass
                return await FakeTTS().synthesize(text, expression)

        self.app.state.tts = LateTTS()
        await self.ws.send("input.text", {"text": "旧问题"}, "t1")
        await asyncio.wait_for(started.wait(), 1)
        old = self.session.generation_id
        await self.ws.send("control.stop", {"generation_id": old})
        await self.ws.until("output.cancel")
        await self.ws.send("input.text", {"text": "新问题"}, "t2")
        chunk = await self.ws.until("tts.chunk")
        self.assertNotEqual(chunk["generation_id"], old)
        self.assertIsNone(self.session.outputs[old].segments["s1"].duration_ms)


class OutputAndAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证播放背压及恢复决策仍使用原有 Agent 图。"""

    async def test_output_credit_waits_for_playback(self):
        """首段耗尽缓冲信用时，后续合成等待真实播放进度。"""
        session = Session()
        session.begin_generation("g", speech_enabled=True)
        for index in range(2):
            session.register_segment("g", {"segment_id": str(index), "text": "句子。"})
        sent = []

        async def send(kind, payload):
            """记录协调器输出，供容量断言使用。"""
            sent.append((kind, payload))

        coordinator = OutputCoordinator(session, "g", FakeTTS(), send, buffer_ms=1000)
        try:
            await coordinator.submit({"segment_id": "0", "text": "句子。"}, "neutral")
            await coordinator.submit({"segment_id": "1", "text": "句子。"}, "neutral")
            await asyncio.sleep(.1)
            self.assertEqual(len(sent), 1)
            session.record_playback("0", "playback.ended", 2000, 2000)
            await asyncio.wait_for(coordinator.finish(), 1)
            self.assertEqual(len(sent), 2)
        finally:
            await coordinator.close()

    async def test_agent_receives_resume_contract_and_no_fake_history(self):
        """真实 LangGraph 根据快照约束续说，不把未生成余文当作既有事实。"""

        class Provider:
            """记录实际发给语言模型的上下文。"""

            async def stream(self, messages):
                """输出合法角色首行及一段文本。"""
                self.messages = messages
                yield '{"expression":"neutral","gaze":"center","gesture":null,"intensity":0.6}\n接着解释。'

        provider = Provider()
        graph = AgentGraph(provider)
        context = {"social_state": {"version": 4}, "interruption_state": {
            "resume_allowed": True, "response_complete": False, "spoken_text": "第一句。",
            "remaining_text": "第二句。", "spoken_text_precision": "approximate", "stop_confirmed": True}}
        events = [event async for event in graph.stream("继续", [], [], "g", runtime_context=context)]
        self.assertIn("未确认停止或标为 approximate", provider.messages[0]["content"])
        self.assertIn("第一句。", provider.messages[0]["content"])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(control_action("为什么会停止生长？"), "respond")
        self.assertEqual(control_action("停一下。"), "stop")

