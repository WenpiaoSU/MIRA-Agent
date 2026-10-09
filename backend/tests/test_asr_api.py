"""作用：验证 ASR API 的请求契约及 partial/final 边界。

当前功能：
- 检查 WAV multipart、Bearer Key、模型名与可选语言。
- 检查 Deepgram 的 Token、WAV 请求体、查询参数与嵌套转写结果。
- 检查 API 错误不会生成虚构转写。
- 检查 partial 可被 final 修订，且 final 使用完整音频。
"""

import asyncio
import unittest
from types import SimpleNamespace

import httpx

from src.speech.asr import ASRError, ASRResult, ASRStream, TranscriptionAPIProvider


class ASRAPITests(unittest.IsolatedAsyncioTestCase):
    """覆盖远端转写适配器及单片段状态。"""

    async def test_multipart_request_and_response(self):
        """确认请求使用 WAV 文件和配置的认证、模型及语言。"""
        captured = {}

        def respond(request):
            """截取本地模拟传输中的请求信息。"""
            captured["authorization"] = request.headers.get("authorization")
            captured["body"] = request.content
            captured["content_type"] = request.headers.get("content-type")
            return httpx.Response(200, json={"text": " 你好 ", "confidence": 0.8})

        settings = SimpleNamespace(asr_ready=True, asr_url="https://asr.example/v1/audio/transcriptions", asr_key="test-key", asr_model_name="speech-model", asr_language="zh")
        provider = TranscriptionAPIProvider(settings, httpx.MockTransport(respond))
        result = await provider.transcribe(bytes(3200))
        self.assertEqual(result, ASRResult("你好", 0.8))
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertIn("multipart/form-data", captured["content_type"])
        self.assertIn(b'name="model"', captured["body"])
        self.assertIn(b"speech-model", captured["body"])
        self.assertIn(b'name="language"', captured["body"])
        self.assertIn(b"RIFF", captured["body"])
        self.assertIn(b"speech.wav", captured["body"])

    async def test_http_failure_is_explicit(self):
        """远端错误应抛出 ASRError，不能伪造成空白 final。"""

        def respond(request):
            """模拟限流响应。"""
            return httpx.Response(429, json={"error": "rate limited"})

        settings = SimpleNamespace(asr_ready=True, asr_url="https://asr.example/v1/audio/transcriptions", asr_key="test-key", asr_model_name="speech-model", asr_language="")
        provider = TranscriptionAPIProvider(settings, httpx.MockTransport(respond))
        with self.assertRaisesRegex(ASRError, "HTTP 429"):
            await provider.transcribe(bytes(3200))

    async def test_deepgram_request_and_response(self):
        """Deepgram API Key 使用 Token，音频直接作为 WAV 请求体发送。"""
        captured = {}

        def respond(request):
            """检查本地模拟请求并返回 Deepgram 结构。"""
            captured["authorization"] = request.headers.get("authorization")
            captured["content_type"] = request.headers.get("content-type")
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"transcript": " 你好 ", "confidence": 0.91}]}]}})

        settings = SimpleNamespace(asr_ready=True, asr_url="https://api.deepgram.com/v1/listen", asr_key="test-key", asr_model_name="nova-3", asr_language="zh", asr_provider="auto")
        result = await TranscriptionAPIProvider(settings, httpx.MockTransport(respond)).transcribe(bytes(3200))
        self.assertEqual(result, ASRResult("你好", 0.91))
        self.assertEqual(captured["authorization"], "Token test-key")
        self.assertEqual(captured["content_type"], "audio/wav")
        self.assertIn("model=nova-3", captured["url"])
        self.assertIn("language=zh", captured["url"])
        self.assertTrue(captured["body"].startswith(b"RIFF"))
        self.assertNotIn(b'multipart/form-data', captured["body"])

    async def test_partial_is_replaced_by_final(self):
        """partial 与 final 分别请求当前快照和完整片段。"""

        class FakeProvider:
            """记录请求音频长度并返回不同修订。"""

            def __init__(self):
                """初始化请求记录。"""
                self.lengths = []

            async def transcribe(self, pcm):
                """按请求顺序返回 partial 与 final。"""
                self.lengths.append(len(pcm))
                return ASRResult("你好" if len(self.lengths) == 1 else "你好世界")

        provider = FakeProvider()
        stream = ASRStream(provider)
        self.assertIsNone(await stream.feed(bytes(48000)))
        await asyncio.sleep(0)
        self.assertEqual(await stream.feed(bytes(640)), ("你好", 1, None))
        self.assertEqual(await stream.feed(bytes(640), is_final=True), ("你好世界", 2, None))
        self.assertEqual(provider.lengths, [48000, 49280])


if __name__ == "__main__":
    unittest.main()
