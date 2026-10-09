"""作用：按稳定语段调度 TTS，保持生成与合成解耦。

当前功能：
- 限制语段前瞻与已发出音频总缓冲，顺序输出完整 WAV。
- 取消时回收 HTTP 合成任务，拒绝旧 generation 晚到结果。
- 合成或播放失败记录为降级，不计为正常交付。
"""

import asyncio
import base64
from src.contracts.realtime import TtsChunkPayload
from src.providers.tts import TTSError


class OutputCoordinator:
    """一个 generation 拥有独立合成队列及任务。"""

    def __init__(self, session, generation_id: str, provider, send,
                 queue_size: int = 3, buffer_ms: int = 15000):
        """注入服务接口并建立有界队列与实际播放信用预算。"""
        self.session = session
        self.generation_id = generation_id
        self.provider = provider
        self.send = send
        self.buffer_ms = buffer_ms
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.task = asyncio.create_task(self.run())

    async def put(self, item) -> None:
        """队列等待同时观察消费者失败，避免后台任务异常后永久挂起。"""
        pending = asyncio.create_task(self.queue.put(item))
        try:
            await asyncio.wait({pending, self.task}, return_when=asyncio.FIRST_COMPLETED)
            if self.task.done():
                await self.task
                raise RuntimeError("TTS 调度已结束")
            await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def submit(self, segment: dict, expression: str) -> None:
        """入队完整稳定语段，限制模型消费端的合成前瞻。"""
        await self.put((segment, expression))

    async def finish(self) -> None:
        """等待已入队语段完成，不阻塞实时接收任务。"""
        if self.task.done():
            await self.task
            return
        pending = asyncio.create_task(self.queue.put(None))
        try:
            await asyncio.wait({pending, self.task}, return_when=asyncio.FIRST_COMPLETED)
            if pending.done():
                await self.task
            else:
                await self.task
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def close(self) -> None:
        """取消并等待合成退出，防止孤儿 HTTP 请求。"""
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    async def wait_capacity(self) -> bool:
        """依据实际播放进度释放发送信用，而非按生成速度无限排队。"""
        while self.session.generation_id == self.generation_id:
            record = self.session.outputs[self.generation_id]
            buffered = sum(max(0, (segment.duration_ms or 0) - segment.played_ms)
                           for segment in record.segments.values()
                           if segment.playback not in {"ended", "failed", "stopped"})
            if buffered < self.buffer_ms:
                return True
            await asyncio.sleep(.05)
        return False

    async def run(self) -> None:
        """顺序合成；任何外部等待后都重新校验代次与容量。"""
        while True:
            item = await self.queue.get()
            if item is None:
                return
            segment, expression = item
            if not await self.wait_capacity():
                return
            try:
                audio = await self.provider.synthesize(segment["text"], expression)
            except TTSError as exc:
                if self.session.generation_id != self.generation_id:
                    return
                record = self.session.outputs[self.generation_id]
                record.segments[segment["segment_id"]].playback = "failed"
                await self.send("tts.error", {"segment_id": segment["segment_id"], "message": str(exc)})
                continue
            if self.session.generation_id != self.generation_id:
                return
            self.session.register_audio(segment["segment_id"], audio.duration_ms, self.generation_id)
            chunk = TtsChunkPayload(generation_id=self.generation_id, segment_id=segment["segment_id"],
                                    audio_base64=base64.b64encode(audio.data).decode("ascii"),
                                    sample_rate=audio.sample_rate, channels=audio.channels,
                                    duration_ms=audio.duration_ms)
            await self.send("tts.chunk", chunk.model_dump())
