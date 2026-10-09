"""作用：通过可配置的转写 API 识别 VAD 截取的语音。

当前功能：
- 将 16 kHz 单声道 PCM 封装成 WAV，调用 Audio Transcriptions 或 Deepgram 接口。
- 定期转写当前语音快照以提供可覆盖的 partial。
- 在语音结束时取消过期 partial，并为关闭操作回收子任务。
"""

import asyncio
import io
import json
import wave
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from src.core.config import Settings


SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
PARTIAL_INTERVAL_BYTES = int(BYTES_PER_SECOND * 1.5)
MAX_RESPONSE_BYTES = 64 * 1024


class ASRError(Exception):
    """表示 ASR 接口配置、请求或响应失败。"""


@dataclass(frozen=True)
class ASRResult:
    """保存一次转写的文本和接口实际返回的可选置信度。"""

    text: str
    confidence: float | None = None


def pcm_to_wav(pcm: bytes) -> bytes:
    """将标准 PCM 样本包装为可由转写 API 接收的 WAV 文件。"""
    if not pcm or len(pcm) % 2:
        raise ASRError("没有可转写的有效音频。")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm)
    return output.getvalue()


class TranscriptionAPIProvider:
    """按供应商协议调用转写接口并统一返回文本。"""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        """保存 ASR 配置；可注入 HTTP 传输以验证请求契约。"""
        self.settings = settings
        self.transport = transport

    def _protocol(self) -> str:
        """优先使用显式配置，并识别 Deepgram 官方域名。"""
        provider = getattr(self.settings, "asr_provider", "auto").lower()
        if provider == "auto":
            hostname = urlsplit(self.settings.asr_url).hostname or ""
            return "deepgram" if hostname == "deepgram.com" or hostname.endswith(".deepgram.com") else "transcriptions"
        if provider not in {"deepgram", "transcriptions"}:
            raise ASRError("MIRA_ASR_PROVIDER 只能是 auto、deepgram 或 transcriptions。")
        return provider

    @staticmethod
    def _result_from_response(data: object, protocol: str) -> ASRResult:
        """解析供应商响应，拒绝缺少真实转写文本的结构。"""
        if not isinstance(data, dict):
            raise ASRError("ASR API 响应不是 JSON 对象。")
        if protocol == "deepgram":
            results = data.get("results")
            channels = results.get("channels") if isinstance(results, dict) else None
            channel = channels[0] if isinstance(channels, list) and channels else None
            alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
            transcript = alternatives[0] if isinstance(alternatives, list) and alternatives else None
            if not isinstance(transcript, dict) or not isinstance(transcript.get("transcript"), str):
                raise ASRError("Deepgram 响应缺少 results.channels[0].alternatives[0].transcript。")
            text = transcript["transcript"]
            confidence = transcript.get("confidence")
        else:
            if not isinstance(data.get("text"), str):
                raise ASRError("ASR API 响应缺少 text 字段。")
            text = data["text"]
            confidence = data.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            confidence = None
        return ASRResult(text=text.strip(), confidence=confidence)

    async def transcribe(self, pcm: bytes) -> ASRResult:
        """按选定协议发送 WAV，限制响应体并解析转写。"""
        if not self.settings.asr_ready:
            raise ASRError("请填写 MIRA_ASR_URL、MIRA_ASR_KEY 和 MIRA_ASR_MODEL_NAME。")
        protocol = self._protocol()
        wav = pcm_to_wav(pcm)
        if protocol == "deepgram":
            params = {"model": self.settings.asr_model_name}
            if self.settings.asr_language:
                params["language"] = self.settings.asr_language
            request_options = {
                "headers": {"Authorization": f"Token {self.settings.asr_key}", "Content-Type": "audio/wav"},
                "params": params,
                "content": wav,
            }
        else:
            fields = {"model": self.settings.asr_model_name, "response_format": "json"}
            if self.settings.asr_language:
                fields["language"] = self.settings.asr_language
            request_options = {
                "headers": {"Authorization": f"Bearer {self.settings.asr_key}"},
                "data": fields,
                "files": {"file": ("speech.wav", wav, "audio/wav")},
            }
        timeout = httpx.Timeout(connect=10, read=20, write=10, pool=10)
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
                async with client.stream(
                    "POST", self.settings.asr_url, **request_options,
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ASRError("ASR API 响应超过 64 KB 上限。")
        except httpx.HTTPStatusError as exc:
            raise ASRError(f"ASR API 返回 HTTP {exc.response.status_code}。") from exc
        except httpx.RequestError as exc:
            raise ASRError("无法连接 ASR API，请检查 URL 与网络。") from exc
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ASRError("ASR API 未返回有效 JSON。") from exc
        return self._result_from_response(data, protocol)


class ASRStream:
    """为单个 VAD 语音片段管理音频快照和转写版本。"""

    def __init__(self, provider: TranscriptionAPIProvider):
        """绑定可替换的 API 适配器并初始化片段状态。"""
        self.provider = provider
        self.audio = bytearray()
        self.partial_task: asyncio.Task[ASRResult] | None = None
        self.last_requested_bytes = 0
        self.last_partial = ""
        self.revision = 0

    def reset(self) -> None:
        """清空上一片段，并取消尚未完成的 partial 请求。"""
        if self.partial_task and not self.partial_task.done():
            self.partial_task.cancel()
        self.audio.clear()
        self.partial_task = None
        self.last_requested_bytes = 0
        self.last_partial = ""
        self.revision = 0

    async def feed(self, pcm: bytes, is_final: bool = False) -> tuple[str, int, float | None] | None:
        """接收音频；partial 可覆盖，final 总是使用完整片段重新识别。"""
        self.audio.extend(pcm)
        if is_final:
            if self.partial_task:
                task = self.partial_task
                self.partial_task = None
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, ASRError):
                    pass
            result = await self.provider.transcribe(bytes(self.audio))
            self.revision += 1
            return result.text, self.revision, result.confidence

        update = None
        if self.partial_task and self.partial_task.done():
            try:
                result = self.partial_task.result()
            except ASRError:
                # partial 失败只影响临时字幕，完整 final 仍独立请求。
                self.partial_task = None
                return None
            self.partial_task = None
            if result.text and result.text != self.last_partial:
                self.last_partial = result.text
                self.revision += 1
                update = result.text, self.revision, result.confidence
        if self.partial_task is None and len(self.audio) >= PARTIAL_INTERVAL_BYTES and len(self.audio) - self.last_requested_bytes >= PARTIAL_INTERVAL_BYTES:
            self.last_requested_bytes = len(self.audio)
            self.partial_task = asyncio.create_task(self.provider.transcribe(bytes(self.audio)))
        return update

    async def close(self) -> None:
        """取消并等待未结束的 partial，主动消费已完成任务的异常。"""
        task = self.partial_task
        self.partial_task = None
        if task:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.audio.clear()
