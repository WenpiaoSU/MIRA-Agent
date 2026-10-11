"""作用：验证 P2 语音帧契约和半双工提交边界。

当前功能：
- 检查音频帧格式与乱序拒绝。
- 确认短暂停顿中的 final 只合并一次，空结果不提交。
- 验证 WebSocket partial 不调用 Agent，轮次完成后才调用。
- 验证语音或文字轮次后的 TTS 语段编码仍可发送。
- 通过本地分类替身隔离外部 LLM 请求，保持提交边界测试确定性。
"""

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from src.main import create_app
from src.contracts.perception import IntentEstimate
from src.speech.frontend import validate_frame
from src.turn.half_duplex import HalfDuplexTurn


class VoiceTurnTests(unittest.TestCase):
    """覆盖语音输入的关键协议和提交约束。"""

    def test_frame_and_turn_rules(self):
        """拒绝错误 PCM，合并短暂停顿并忽略空 final。"""
        self.assertEqual(len(validate_frame("s", 0, 0, 20, bytes(640)).pcm), 640)
        with self.assertRaises(ValueError):
            validate_frame("s", 0, 0, 20, bytes(320))
        turn = HalfDuplexTurn(700)
        turn.speech_started()
        turn.accept_final("你好", 1000)
        self.assertIsNone(turn.complete_if_ready(1500))
        turn.speech_started()
        turn.accept_final("世界", 1800)
        self.assertEqual(turn.complete_if_ready(2500), "你好世界")
        self.assertIsNone(turn.complete_if_ready(3200))
        turn.speech_started()
        turn.accept_final("  ", 3500)
        self.assertIsNone(turn.complete_if_ready(4200))

    def test_partial_never_enters_agent(self):
        """通过真实 WebSocket 网关检查 partial 与轮次提交的隔离。"""
        calls = []

        class FakeStream:
            """提供确定性的 partial、final 与轮次完成事件。"""

            async def feed(self, frame):
                """首帧返回 partial，第二帧确认完整轮次。"""
                if frame.seq == 0:
                    return [("asr.partial", {"text": "你", "utterance_id": "u", "revision": 1, "is_final": False, "confidence": None})]
                return [("asr.final", {"text": "你好", "utterance_id": "u", "revision": 2, "is_final": True, "confidence": None}), ("turn.completed", {"text": "你好", "reason": "silence", "end_ms": 40})]

        class FakeSpeechService:
            """代替 VAD 与 ASR 服务，以便只检查网关调用时机。"""

            ready = True

            def create_stream(self, stream_id):
                """为测试流返回确定性的语音事件。"""
                return FakeStream()

        class FakeAgent:
            """记录被提交到 Agent 的用户文本。"""

            async def stream(self, text, history, cues, generation_id, runtime_context=None):
                """记录输入并输出一条完成事件。"""
                calls.append(text)
                yield {"type": "response.completed", "payload": {"text": "收到"}}

        app = create_app()
        app.state.speech_service = FakeSpeechService()
        app.state.agent = FakeAgent()
        app.state.intent_service.classify = AsyncMock(return_value=IntentEstimate(
            intent="chat", slots=[], confidence=0.9, needs_clarification=False,
            missing_slots=[], control_action=None))
        with TestClient(app) as client:
            session_id = client.post("/v1/sessions").json()["session_id"]
            with client.websocket_connect(f"/v1/sessions/{session_id}/realtime") as ws:
                status = ws.receive_json()
                epoch = status["connection_epoch"]

                def envelope(kind, payload, seq):
                    """构造符合当前连接代次的客户端事件。"""
                    event_id = str(uuid4())
                    return {"schema_version": "1.0", "event_id": event_id, "type": kind, "source": "avatar-web", "session_id": session_id, "connection_epoch": epoch, "seq": seq, "occurred_at": datetime.now(timezone.utc).isoformat(), "time_ref": {"clock_id": "test", "capture_start_ms": 0, "capture_end_ms": 0}, "trace_id": event_id, "payload": payload}

                ws.send_json(envelope("input.audio_start", {"stream_id": "s", "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}, 0))
                self.assertEqual(ws.receive_json()["payload"]["state"], "listening")

                def send_frame(seq):
                    """按协议发送二进制事件头和 640 字节 PCM。"""
                    header = json.dumps(envelope("input.audio_frame", {"stream_id": "s", "frame_seq": seq, "capture_start_ms": seq * 20, "capture_end_ms": (seq + 1) * 20}, seq + 1)).encode()
                    ws.send_bytes(len(header).to_bytes(2, "big") + header + bytes(640))

                send_frame(0)
                self.assertEqual(ws.receive_json()["type"], "audio.received")
                self.assertEqual(ws.receive_json()["type"], "asr.partial")
                self.assertEqual(calls, [])
                send_frame(1)
                self.assertEqual(ws.receive_json()["type"], "asr.final")
                self.assertEqual(ws.receive_json()["type"], "turn.completed")
                self.assertEqual(ws.receive_json()["payload"]["state"], "received")
                while True:
                    event = ws.receive_json()
                    if event["type"] == "turn.state":
                        self.assertEqual(event["payload"]["state"], "thinking")
                        break
                while ws.receive_json()["type"] != "response.completed":
                    pass
                self.assertEqual(calls, ["你好"])

    def test_tts_chunk_after_turn_submission(self):
        """确保轮次提交后 TTS 音频能正确编码并送到 WebSocket。"""

        class FakeAgent:
            """输出一个可合成的完整语段。"""

            async def stream(self, text, history, cues, generation_id, runtime_context=None):
                """返回语段和完整回复。"""
                yield {"type": "response.segment", "payload": {"segment_id": "segment-1", "text": "你好"}}
                yield {"type": "response.completed", "payload": {"text": "你好"}}

        class FakeTTS:
            """返回固定 WAV 字节供编码测试。"""

            async def synthesize(self, text, expression):
                """构造具有时长和采样率的音频结果。"""
                return SimpleNamespace(data=b"RIFFtest", duration_ms=100, sample_rate=16000, channels=1)

        app = create_app()
        app.state.settings = SimpleNamespace(llm_ready=True, tts_ready=True)
        app.state.agent = FakeAgent()
        app.state.tts = FakeTTS()
        app.state.intent_service.classify = AsyncMock(return_value=IntentEstimate(
            intent="chat", slots=[], confidence=0.9, needs_clarification=False,
            missing_slots=[], control_action=None))
        with TestClient(app) as client:
            session_id = client.post("/v1/sessions").json()["session_id"]
            with client.websocket_connect(f"/v1/sessions/{session_id}/realtime") as ws:
                status = ws.receive_json()
                event_id = str(uuid4())
                ws.send_json({"schema_version": "1.0", "event_id": event_id, "type": "input.text", "source": "avatar-web", "session_id": session_id, "connection_epoch": status["connection_epoch"], "seq": 0, "occurred_at": datetime.now(timezone.utc).isoformat(), "time_ref": {"clock_id": "test", "capture_start_ms": 0, "capture_end_ms": 0}, "trace_id": event_id, "turn_id": "turn-1", "payload": {"text": "你好"}})
                received = []
                while not received or received[-1]["type"] != "tts.chunk":
                    received.append(ws.receive_json())
                self.assertIn("response.segment", [item["type"] for item in received])
                self.assertEqual(received[-1]["payload"]["audio_base64"], "UklGRnRlc3Q=")


if __name__ == "__main__":
    unittest.main()
