"""作用：持续处理本地 VAD，并异步收集远端 ASR 结果。

当前功能：
- VAD 与 ASR final 解耦，网络等待不会暂停插话检测。
- 以最终文本、最后有效语音时间和静音窗口提交轮次。
- 支持持续会话、有限待识别片段和单片段最大时长。
- 关闭时回收所有 partial/final 子任务，partial 仅用于字幕。
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.core.config import Settings
from src.speech.asr import ASRStream, TranscriptionAPIProvider
from src.speech.frontend import AudioFrame
from src.speech.vad import SileroVAD


VAD_MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "silero_vad_16k_op15.onnx"


@dataclass
class PendingUtterance:
    """保存独立语音片段，避免新起点重置旧 final 的音频。"""
    utterance_id: str
    start_ms: int
    end_ms: int
    asr: ASRStream
    final_task: asyncio.Task | None = None


class SpeechService:
    """为每条持续音频流创建独立 VAD 与片段转写状态。"""

    def __init__(self, settings: Settings):
        """保存配置并创建可复用 ASR HTTP 适配器。"""
        self.settings = settings
        self.asr_provider = TranscriptionAPIProvider(settings)

    @property
    def ready(self) -> bool:
        """检查配置、模型文件和推理依赖；设备效果由实测决定。"""
        from importlib.util import find_spec
        return self.settings.asr_ready and VAD_MODEL_PATH.is_file() and all(
            find_spec(name) is not None for name in ("numpy", "onnxruntime"))

    def create_stream(self, stream_id: str):
        """在线程中初始化本地模型，实时控制不等待模型加载。"""
        return SpeechStream(stream_id, SileroVAD(VAD_MODEL_PATH), ASRStream(self.asr_provider),
                            silence_ms=getattr(self.settings, "turn_silence_ms", 700),
                            max_utterance_ms=getattr(self.settings, "max_utterance_ms", 60000))


class SpeechStream:
    """VAD 顺序消费帧，独立 ASR 任务按片段顺序提交 final。"""

    def __init__(self, stream_id: str, vad, asr: ASRStream, silence_ms: int = 700,
                 max_utterance_ms: int = 60000):
        """建立有限片段缓存；整条持续音频流不受 60 秒片段限制。"""
        self.stream_id = stream_id
        self.vad = vad
        self.provider = asr.provider
        self.initial_asr = asr
        self.silence_ms = silence_ms
        self.max_utterance_ms = max_utterance_ms
        self.preroll: deque[bytes] = deque(maxlen=10)
        self.pending: deque[PendingUtterance] = deque()
        self.current: PendingUtterance | None = None
        self.parts: list[tuple[str, str]] = []
        self.last_seq = -1
        self.last_end_ms = -1
        self.last_valid_ms: int | None = None
        self.closed = False

    def end_utterance(self) -> None:
        """冻结当前音频并启动 final 请求，当前 VAD 任务立即继续。"""
        if self.current is None:
            return
        utterance = self.current
        utterance.end_ms = self.last_valid_ms or self.last_end_ms
        utterance.final_task = asyncio.create_task(utterance.asr.feed(b"", is_final=True))
        self.pending.append(utterance)
        self.current = None
        self.preroll.clear()

    async def feed(self, frame: AudioFrame) -> list[tuple[str, dict]]:
        """处理一帧，所有远端 final 均在后台等待而不阻塞下一帧。"""
        if self.closed or frame.stream_id != self.stream_id or frame.seq != self.last_seq + 1 or frame.capture_start_ms < self.last_end_ms:
            raise ValueError("语音帧丢失、乱序或时间倒退")
        self.last_seq = frame.seq
        self.last_end_ms = frame.capture_end_ms
        samples = frame.samples()
        clipping = float((abs(samples) >= .999).mean())
        phases = await asyncio.to_thread(self.vad.feed, samples)
        self.preroll.append(frame.pcm)
        events = []
        started = any(phase == "start" for phase, _ in phases)
        ended = any(phase == "end" for phase, _ in phases)
        if started:
            if len(self.pending) >= 4:
                raise ValueError("ASR 待处理片段已满，请稍后重试")
            asr = self.initial_asr or ASRStream(self.provider)
            self.initial_asr = None
            self.current = PendingUtterance(str(uuid4()), frame.capture_start_ms, frame.capture_end_ms, asr)
        if any(probability >= .35 and phase != "end" for phase, probability in phases):
            self.last_valid_ms = frame.capture_end_ms
        if phases:
            phase = "start" if started else "end" if ended else "active"
            probability = max(prob for _, prob in phases) if not ended else min(prob for _, prob in phases)
            events.append(("vad.event", {"phase": phase, "start_ms": frame.capture_start_ms,
                           "end_ms": frame.capture_end_ms, "p_speech": probability,
                           "clipping_ratio": clipping, "echo_likelihood": None}))
        if self.current:
            pcm = b"".join(self.preroll) if started else frame.pcm
            update = await self.current.asr.feed(pcm)
            if update:
                text, revision, confidence = update
                events.append(("asr.partial", {"utterance_id": self.current.utterance_id,
                               "revision": revision, "text": text, "is_final": False, "confidence": confidence}))
            if ended or frame.capture_end_ms - self.current.start_ms >= self.max_utterance_ms:
                if not ended:
                    events.append(("audio.segment_limit", {"utterance_id": self.current.utterance_id,
                                   "end_ms": frame.capture_end_ms, "reason": "max_utterance"}))
                    self.vad.reset()
                self.end_utterance()
        events.extend(await self.poll())
        return events

    async def poll(self, force: bool = False) -> list[tuple[str, dict]]:
        """按原片段顺序收集 final，静音从最后有效语音时刻计算。"""
        events = []
        while self.pending and self.pending[0].final_task.done():
            utterance = self.pending.popleft()
            text, revision, confidence = utterance.final_task.result()
            events.append(("asr.final", {"utterance_id": utterance.utterance_id, "revision": revision,
                           "text": text, "is_final": True, "confidence": confidence,
                           "start_ms": utterance.start_ms, "end_ms": utterance.end_ms}))
            if text.strip():
                self.parts.append((utterance.utterance_id, text.strip()))
                if sum(len(part) for _, part in self.parts) > 4000:
                    raise ValueError("本轮语音超过 4000 字，请结束后分段输入")
        # 只分析完整 final 的末尾，不使用 partial；含糊末尾最多多等待 700ms。
        text_tail = "".join(text for _, text in self.parts).rstrip("。！？!?，, ")
        incomplete = text_tail.endswith(("因为", "然后", "如果", "比如", "首先", "但是", "以及", "或者", "我想"))
        required_silence = self.silence_ms + (700 if incomplete else 0)
        if (not self.current and not self.pending and self.parts
                and (force or self.last_end_ms - (self.last_valid_ms or self.last_end_ms) >= required_silence)):
            events.append(("turn.completed", {"text": "".join(text for _, text in self.parts),
                           "utterance_ids": [identifier for identifier, _ in self.parts],
                           "reason": "capture_end" if force else "silence",
                           "end_ms": self.last_valid_ms}))
            self.parts.clear()
        return events

    async def finish(self) -> list[tuple[str, dict]]:
        """显式结束采集时冲刷最后片段；停止控制仍由独立网关处理。"""
        self.end_utterance()
        await asyncio.gather(*(item.final_task for item in self.pending))
        return await self.poll(force=True)

    async def close(self) -> None:
        """取消并等待所有片段子任务，不留下孤儿 partial 请求。"""
        self.closed = True
        utterances = list(self.pending) + ([self.current] if self.current else [])
        tasks = [item.final_task for item in utterances if item.final_task]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(item.asr.close() for item in utterances), return_exceptions=True)
        self.pending.clear()
        self.current = None
        self.parts.clear()

