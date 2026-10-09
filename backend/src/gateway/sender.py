"""作用：串行发送实时事件并优先处理取消与错误。

当前功能：
- 在真正发送时分配序号并重新校验连接及输出代次。
- 使用有界优先级队列，取消不等待未发送的媒体队列。
- 对网络发送设置截止时间，关闭时回收等待者。
"""

import asyncio
from itertools import count
from src.contracts.realtime import ServerEvent


class EventSender:
    """一个连接拥有唯一写任务，避免多个协程同时操作 WebSocket。"""

    def __init__(self, websocket, session, epoch: int):
        """绑定连接所有权并启动串行发送任务。"""
        self.websocket = websocket
        self.session = session
        self.epoch = epoch
        self.queue = asyncio.PriorityQueue(maxsize=128)
        self.counter = count()
        self.task = asyncio.create_task(self.run())

    async def send(self, kind: str, payload: dict, turn_id: str | None = None,
                   generation_id: str | None = None, source: str = "gateway",
                   trace_id: str | None = None, causation_id: str | None = None) -> bool:
        """控制消息优先入队；发送完成才返回，丢弃过期消息返回 False。"""
        if self.task.done():
            raise ConnectionError("实时连接发送任务已结束")
        record = self.session.outputs.get(generation_id)
        trace_id = trace_id or (record.trace_id if record else self.session.session_id)
        future = asyncio.get_running_loop().create_future()
        priority = 0 if kind in {"output.cancel", "error", "barge_in.confirmed"} else 1 if kind.startswith(("turn.", "audio.", "session.")) else 2
        item = (priority, next(self.counter), kind, payload, turn_id, generation_id, source, trace_id, causation_id, future)
        # timeout 保留取消语义，避免完成回执与取消同时抵达时吞掉取消。
        async with asyncio.timeout(2):
            await self.queue.put(item)
        async with asyncio.timeout(4):
            return await future

    async def run(self) -> None:
        """按优先级发包，任何等待后再次检查 epoch 和 generation。"""
        try:
            while True:
                _, _, kind, payload, turn_id, generation_id, source, trace_id, causation_id, future = await self.queue.get()
                output_kind = kind in {"behavior.plan", "response.segment", "response.completed", "tts.chunk", "tts.error"}
                stale = self.session.connection_epoch != self.epoch or (
                    output_kind and generation_id != self.session.generation_id)
                if stale or future.cancelled():
                    if not future.done():
                        future.set_result(False)
                    continue
                self.session.next_seq += 1
                event = ServerEvent(type=kind, payload=payload, source=source,
                                    session_id=self.session.session_id, connection_epoch=self.epoch,
                                    seq=self.session.next_seq, turn_id=turn_id, generation_id=generation_id,
                                    trace_id=trace_id, causation_id=causation_id)
                try:
                    async with asyncio.timeout(2):
                        await self.websocket.send_json(event.model_dump(mode="json"))
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                    raise
                if not future.done():
                    future.set_result(True)
                if kind != "vad.event":
                    self.session.trace.record(kind, trace_id, {**payload, "generation_id": generation_id}, causation_id)
        finally:
            while not self.queue.empty():
                future = self.queue.get_nowait()[-1]
                if not future.done():
                    future.set_result(False)

    async def close(self) -> None:
        """停止写任务并回收排队事件，不遗留 Future 异常。"""
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

