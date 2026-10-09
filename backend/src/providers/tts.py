"""作用：通过兼容 Speech API 的服务合成完整语段。

当前功能：
- 使用后端专有的 URL、Key、模型名和音色 reference_id 请求 PCM WAV。
- 对 Fish Audio 使用其原生的 model 请求头和 text/format 请求字段。
- 给每句合成文本加上独立的方括号情感标签，不改动展示文本。
- 允许将 Avatar 表情名映射为目标模型支持的标签名称。
- 对 HTTP 响应设置超时、大小上限并校验音频时长。
- 通过异步请求取消传播停止尚未完成的合成。
"""

import io
import json
import re
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args
from urllib.parse import urlsplit

import httpx

from src.core.config import Settings
from src.contracts.behavior import Expression


MAX_WAV_BYTES = 8 * 1024 * 1024
EMOTION_TAG = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
SENTENCE_PART = re.compile(r"[^。！？!?\n]+[。！？!?]*|[。！？!?]+")
AVATAR_EXPRESSIONS = frozenset(get_args(Expression))


class TTSError(Exception):
    """表示 TTS 接口请求失败或返回无效音频。"""


@dataclass(frozen=True)
class SynthesizedAudio:
    """保存单个语段的 WAV 字节、采样参数及实际时长。"""

    data: bytes
    duration_ms: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class SpeechPart:
    """保存一段原始回复文本和仅供 TTS 使用的情感标签。"""

    text: str
    emotion_tag: str


def format_emotional_input(parts: Sequence[SpeechPart]) -> str:
    """为每句话添加独立情感标签，支持一次请求中切换多种情感。"""
    lines: list[str] = []
    for part in parts:
        if not EMOTION_TAG.fullmatch(part.emotion_tag):
            raise TTSError("TTS 情感标签格式不正确。")
        for sentence in SENTENCE_PART.findall(part.text):
            clean_text = sentence.strip()
            if clean_text:
                lines.append(f"[{part.emotion_tag}] {clean_text}")
    if not lines:
        raise TTSError("没有可合成的语音文本。")
    return "\n".join(lines)


class SpeechAPIProvider:
    """适配采用 Speech API JSON 请求和 WAV 响应的 TTS 服务。"""

    def __init__(self, settings: Settings):
        """保存本轮服务进程的 TTS 配置。"""
        self.settings = settings
        self._expression_tags: dict[str, str] | None = None

    def _tag_for_expression(self, expression: str) -> str:
        """校验 Avatar 表情，并查找目标 TTS 模型使用的标签。"""
        if expression not in AVATAR_EXPRESSIONS:
            raise TTSError("Avatar 表情不在允许的情感标签范围内。")
        if self._expression_tags is None:
            try:
                configured = json.loads(self.settings.tts_emotion_tags) if self.settings.tts_emotion_tags else {}
            except ValueError as exc:
                raise TTSError("MIRA_TTS_EMOTION_TAGS 不是有效 JSON 对象。") from exc
            if not isinstance(configured, dict) or any(
                key not in AVATAR_EXPRESSIONS or not isinstance(value, str) or not EMOTION_TAG.fullmatch(value)
                for key, value in configured.items()
            ):
                raise TTSError("MIRA_TTS_EMOTION_TAGS 包含无效表情或标签。")
            self._expression_tags = configured
        return self._expression_tags.get(expression, expression)

    async def synthesize(self, text: str, expression: str = "neutral") -> SynthesizedAudio:
        """用当前 Avatar 表情标签合成一个完整语段。"""
        return await self.synthesize_parts([SpeechPart(text=text, emotion_tag=self._tag_for_expression(expression))])

    async def synthesize_parts(self, parts: Sequence[SpeechPart]) -> SynthesizedAudio:
        """在同一次请求中按句切换情感标签并合成有界 PCM WAV。"""
        if not self.settings.tts_ready:
            raise TTSError("请填写 MIRA_TTS_URL、MIRA_TTS_KEY、MIRA_TTS_MODEL_NAME 和 MIRA_TTS_VOICE。")
        speech_input = format_emotional_input(parts)
        timeout = httpx.Timeout(connect=10, read=45, write=10, pool=10)
        headers = {"Authorization": f"Bearer {self.settings.tts_key}"}
        if urlsplit(self.settings.tts_url).hostname == "api.fish.audio":
            # Fish Audio 用 model 请求头选模型；省略时会回退到付费模型。
            headers["model"] = self.settings.tts_model_name
            request = {"text": speech_input, "reference_id": self.settings.tts_voice, "format": "wav"}
        else:
            request = {
                "model": self.settings.tts_model_name,
                "reference_id": self.settings.tts_voice,
                "input": speech_input,
                "response_format": "wav",
            }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", self.settings.tts_url,
                    headers=headers, json=request,
                ) as response:
                    response.raise_for_status()
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > MAX_WAV_BYTES:
                            raise TTSError("单个语段的 TTS 音频超过 8 MB 上限。")
        except httpx.HTTPStatusError as exc:
            raise TTSError(f"TTS API 返回 HTTP {exc.response.status_code}。") from exc
        except httpx.RequestError as exc:
            raise TTSError("无法连接 TTS API，请检查 URL、网络和密钥。") from exc
        return self._validate_wav(bytes(chunks))

    @staticmethod
    def _validate_wav(data: bytes) -> SynthesizedAudio:
        """检查 PCM WAV 头和帧数，并从采样信息计算播放时长。"""
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                if audio.getcomptype() != "NONE" or audio.getnframes() == 0:
                    raise TTSError("TTS 返回了空音频或非 PCM WAV。")
                return SynthesizedAudio(
                    data=data,
                    duration_ms=round(audio.getnframes() * 1000 / audio.getframerate()),
                    sample_rate=audio.getframerate(),
                    channels=audio.getnchannels(),
                )
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise TTSError("TTS API 未返回有效的 WAV 音频。") from exc
