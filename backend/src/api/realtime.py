"""作用：连接实时输入、确定性轮次控制及现有 Agent。

当前功能：
- 校验所有客户端信封、epoch、顺序和二进制音频边界。
- 持续 VAD 与异步 ASR 并行；partial 只显示字幕。
- 直接执行插话取消，保留旧输出最终回执与恢复快照。
- 委托有界输出协调器调度 TTS，并提供状态与共享因果轨迹。
- 在断线、接管、超时及会话删除时回收全部连接任务。
- 在可取消的生成任务中识别最终意图，冻结结果后交给现有 Agent 路由。
- 提供真实工具目录、保存会话参数澄清及结果，过滤取消后的旧工具事件。
"""

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.agent.recovery import control_action
from src.contracts.realtime import (
    AudioEndInput, AudioFrameInput, AudioStartInput, CommandAckInput, DisplayedInput,
    PlaybackInput, StopInput, TextInput,
)
from src.contracts.tools import PendingClarification, ToolResult
from src.evaluation.trace import now_ms
from src.gateway.sender import EventSender
from src.intent.service import build_context_summary
from src.output.coordinator import OutputCoordinator
from src.providers.llm import LLMError
from src.speech.asr import ASRError
from src.speech.frontend import validate_frame


router = APIRouter()
logger = logging.getLogger(__name__)
INPUT_MODELS = {
    "input.text": TextInput, "input.audio_start": AudioStartInput,
    "input.audio_frame": AudioFrameInput, "input.audio_end": AudioEndInput,
    "control.stop": StopInput, "response.displayed": DisplayedInput,
    "command.ack": CommandAckInput,
    **{f"playback.{phase}": PlaybackInput for phase in ("started", "progress", "ended", "stopped", "failed", "paused")},
}


def decode_message(message: dict) -> tuple[dict, bytes | None]:
    """解析有限 JSON 或固定 PCM 二进制包，禁止任意媒体格式。"""
    pcm = None
    if message.get("bytes") is not None:
        packet = message["bytes"]
        if len(packet) < 3 or len(packet) > 2048:
            raise ValueError("二进制音频包大小无效")
        header_size = int.from_bytes(packet[:2], "big")
        if not 2 <= header_size <= 1300 or len(packet) != 2 + header_size + 640:
            raise ValueError("二进制音频包结构无效")
        raw = json.loads(packet[2:2 + header_size])
        pcm = packet[2 + header_size:]
    else:
        text = message.get("text")
        if not isinstance(text, str) or len(text) > 65536:
            raise ValueError("事件必须是有限 JSON 对象")
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("事件必须是 JSON 对象")
    return raw, pcm


@router.websocket("/v1/sessions/{session_id}/realtime")
async def realtime(websocket: WebSocket, session_id: str):
    """建立单播放端连接，实时控制不依赖模型及转写任务。"""
    session = websocket.app.state.sessions.get(session_id)
    if session is None:
        await websocket.close(code=1008, reason="未知会话")
        return
    await websocket.accept()
    session.cancel_generation("takeover")
    session.connection_epoch += 1
    epoch = session.connection_epoch
    session.last_client_seq = -1
    session.seen_event_ids.clear()
    settings = websocket.app.state.settings
    session.turn.barge_in_ms = getattr(settings, "barge_in_ms", 600)
    session.stop_ack_timeout_ms = getattr(settings, "stop_ack_timeout_ms", 2000)
    session.turn.stopped(session.turn.generation_id)
    session.turn.generating = False
    session.turn.configure(False, "unavailable")
    sender = EventSender(websocket, session, epoch)
    generation_tasks: set[asyncio.Task] = set()
    audio_task = None
    audio_queue = None
    audio_stream_id = None
    completed_streams = set()
    last_audio_received_ms = now_ms()
    received_audio_frame = False

    async def send(kind, payload, turn_id=None, generation_id=None, source="gateway", trace_id=None, causation_id=None):
        """通过唯一写任务发送事件并继承所属轮次的因果标识。"""
        return await sender.send(kind, payload, turn_id, generation_id, source, trace_id, causation_id)

    async def send_state(turn_id=None, generation_id=None):
        """报告实际轮次观测，而非把模型结束误报为播放结束。"""
        state = session.turn.snapshot().model_dump()
        await send("turn.state", {"state": state.pop("mode"), **state}, turn_id, generation_id)

    async def cancel(reason, speech_start_ms=None):
        """先令 generation 失效，再优先发送幂等取消；不等待停止确认。"""
        old = session.cancel_generation(reason)
        if old:
            payload = session.cancel_payload(old)
            if speech_start_ms is not None:
                payload["speech_start_ms"] = speech_start_ms
            await send("output.cancel", payload, session.outputs[old].turn_id, old)
        return old

    async def delivered():
        """仅在全部音频终态后报告真实交付或降级结果。"""
        record = session.finish_if_delivered()
        if record:
            # 新回复已交付后，下一次重述应指向最新输出，不沿用更早的中断。
            if session.interruption and session.interruption.generation_id != record.generation_id:
                session.interruption = None
            await send("turn.delivered", {"generation_id": record.generation_id, "status": record.status},
                       record.turn_id, record.generation_id)
            await send_state(record.turn_id)

    async def generate(text, turn_id, generation_id, source):
        """在可取消任务内完成分类与生成，不阻塞网关接收和 VAD。"""
        coordinator = None
        try:
            if session.interruption and not session.interruption.stop_confirmed:
                previous = session.outputs.get(session.interruption.generation_id)
                if previous and previous.cancelled_at_ms:
                    # 恢复上下文等待最终位置，但实时接收与停止链路不等待。
                    while (not previous.stop_confirmed
                           and now_ms() - previous.cancelled_at_ms < session.stop_ack_timeout_ms):
                        await asyncio.sleep(.02)
            if session.generation_id != generation_id:
                return
            history = session.context_history(generation_id)
            summary = build_context_summary(history, session.interruption.model_dump() if session.interruption else None)
            started_ms = now_ms()
            registry = getattr(websocket.app.state, "tool_registry", None)
            available_tools = registry.list_tools() if registry else []
            pending = session.pending_clarification
            classify_options = {"source": source}
            if pending:
                classify_options["pending_clarification"] = pending.model_copy(deep=True)
            intent = await websocket.app.state.intent_service.classify(text, summary, available_tools, **classify_options)
            # 即使提供者吞掉取消，旧分类结果也不能更新状态或开始输出。
            if session.generation_id != generation_id or session.connection_epoch != epoch:
                return
            record = session.outputs[generation_id]
            # 分类降级时保留原待澄清请求；独立的新任务和控制请求清除它。
            if not intent.continues_pending and not (intent.intent == "other" and intent.confidence is None):
                session.pending_clarification = None
            session.trace.record("intent.classified", record.trace_id,
                                 {"generation_id": generation_id, "intent": intent.intent,
                                  "confidence": intent.confidence, "needs_clarification": intent.needs_clarification,
                                  "elapsed_ms": now_ms() - started_ms})
            if intent.intent == "control" and not intent.needs_clarification:
                if intent.control_action == "change_topic":
                    session.interruption = None
                if intent.control_action in {"stop", "cancel", "change_topic"}:
                    session.pending_clarification = None
                elif intent.control_action in {"repeat", "resume"} and session.interruption is None:
                    previous = [item for item in session.outputs.values() if item.generation_id != generation_id]
                    if previous:
                        session.interruption = session.snapshot_interruption(previous[-1])
            runtime_context = {"social_state": session.social_snapshot(text, intent).model_dump(),
                               "interruption_state": session.interruption.model_dump() if session.interruption else None,
                               "pending_clarification": pending.model_dump() if pending and intent.continues_pending else None}
            if session.interruption:
                interrupted_record = session.outputs.get(session.interruption.generation_id)
                if interrupted_record and interrupted_record.tool_result:
                    runtime_context["interrupted_tool_result"] = interrupted_record.tool_result.model_dump(mode="json")
                    if intent.intent == "control" and intent.control_action in {"repeat", "resume"}:
                        record.tool_result = interrupted_record.tool_result.model_copy(deep=True)
            await send_state(turn_id, generation_id)
            if settings.tts_ready:
                async def send_output(kind, payload):
                    """保持 TTS 结果的轮次归属，发送前再次校验输出代次。"""
                    await send(kind, payload, turn_id, generation_id, "tts")
                coordinator = OutputCoordinator(session, generation_id, websocket.app.state.tts, send_output,
                                                buffer_ms=getattr(settings, "output_buffer_ms", 15000))
            expression = "neutral"
            record = session.outputs[generation_id]
            async for item in websocket.app.state.agent.stream(
                text, history, session.recent_avatar_cues.copy(),
                generation_id, runtime_context=runtime_context):
                if session.generation_id != generation_id or session.connection_epoch != epoch:
                    return
                kind, payload = item["type"], item["payload"]
                if kind == "tool.pending":
                    session.pending_clarification = (PendingClarification.model_validate(payload["pending"])
                                                     if payload["pending"] else None)
                    continue
                if kind == "tool.result":
                    record.tool_result = ToolResult.model_validate(payload["result"])
                    result = record.tool_result
                    session.trace.record("tool.completed", record.trace_id,
                                         {"generation_id": generation_id, "call_id": result.call_id,
                                          "tool_name": result.tool_name, "status": result.status,
                                          "verified": result.verified, "elapsed_ms": result.elapsed_ms,
                                          "error_code": result.error.code if result.error else None})
                    continue
                if kind == "tool.started":
                    session.trace.record(kind, record.trace_id, {"generation_id": generation_id, **payload})
                    continue
                if kind == "behavior.plan":
                    expression = payload["face"]["expression"]
                    # 文字降级仍即时执行；有音频时在前端实际首音时启动动作。
                    payload["timing"]["anchor"] = "speech_start" if settings.tts_ready else "plan_ready"
                    cue = {"expression": expression, "gaze": payload["gaze"]["target"],
                           "gesture": payload["gestures"][0]["name"] if payload["gestures"] else None}
                    session.recent_avatar_cues = (session.recent_avatar_cues + [cue])[-4:]
                elif kind == "response.segment":
                    session.register_segment(generation_id, payload)
                elif kind == "response.completed":
                    record.response_complete = True
                if not await send(kind, payload, turn_id, generation_id, "agent"):
                    return
                if kind == "response.segment" and coordinator:
                    await coordinator.submit(payload, expression)
            if coordinator:
                await coordinator.finish()
            if session.generation_id != generation_id or session.connection_epoch != epoch:
                return
            record.synthesis_complete = True
            session.turn.generating = False
            session.trace.record("generation.finished", record.trace_id, {"generation_id": generation_id})
            await delivered()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("回复生成失败")
            if session.generation_id == generation_id and session.connection_epoch == epoch:
                code = "llm_error" if isinstance(exc, LLMError) else "agent_error"
                await cancel(code)
                await send("error", {"code": code, "message": str(exc) if isinstance(exc, LLMError) else "生成回复失败，请重试。",
                                    "retryable": True}, turn_id, generation_id)
        finally:
            if coordinator:
                await coordinator.close()
            if session.generation_task is asyncio.current_task():
                session.generation_task = None

    async def bounded_generate(text, turn_id, generation_id, source):
        """为分类、生成与合成设置整轮预算，超时走同一取消协议。"""
        try:
            async with asyncio.timeout(getattr(settings, "generation_timeout_ms", 120000) / 1000):
                await generate(text, turn_id, generation_id, source)
        except TimeoutError:
            if session.connection_epoch == epoch and session.generation_id == generation_id:
                await cancel("generation_timeout")
                await send("error", {"code": "generation_timeout", "message": "本轮回复超时，已停止，请重试。",
                                    "retryable": True}, turn_id, generation_id)

    async def submit_turn(text, turn_id, trace_id=None, source="text"):
        """保存最终输入，明确控制短句走快捷路径，其余进入同一个 Agent。"""
        action = control_action(text)
        if action == "stop":
            session.pending_clarification = None
            await cancel("user_stop")
            await send("turn.controlled", {"action": "stop"}, turn_id)
            await send_state(turn_id)
            return
        await cancel("new_turn")
        if action == "change_topic":
            session.interruption = None
            session.pending_clarification = None
        if action == "repeat" and session.interruption is None and session.outputs:
            session.interruption = session.snapshot_interruption(next(reversed(session.outputs.values())))
        generation_id = str(uuid4())
        session.begin_generation(generation_id, turn_id, text, trace_id or turn_id, settings.tts_ready)
        await send("turn.state", {"state": "received", "user_speaking": False,
                                "robot_playing": False, "continuous": session.turn.continuous}, turn_id, generation_id)
        task = asyncio.create_task(bounded_generate(text, turn_id, generation_id, source))
        generation_tasks.add(task)
        task.add_done_callback(generation_tasks.discard)
        session.generation_task = task

    async def process_audio(queue, stream_id, continuous, trace_id):
        """持续消费 VAD 帧，定时收集 ASR 完成事件；结束时回收子任务。"""
        nonlocal audio_stream_id
        stream = None
        turn_id = str(uuid4())
        try:
            async with asyncio.timeout(15):
                stream = await asyncio.to_thread(websocket.app.state.speech_service.create_stream, stream_id)
            await send("turn.state", {"state": "listening", "audio_ready": True,
                                    "stream_id": stream_id, "continuous": continuous}, turn_id, trace_id=trace_id)
            while session.connection_epoch == epoch:
                try:
                    async with asyncio.timeout(.05):
                        frame = await queue.get()
                except asyncio.TimeoutError:
                    events = await stream.poll() if hasattr(stream, "poll") else []
                    frame = "poll"
                else:
                    events = await (stream.finish() if frame is None else stream.feed(frame))
                for kind, payload in events:
                    if kind == "vad.event":
                        phase = payload["phase"]
                        start_ms = payload.get("start_ms", payload.get("end_ms", 0))
                        end_ms = payload.get("end_ms", start_ms)
                        confirmed = session.turn.vad(phase, start_ms, end_ms, payload.get("p_speech", 0),
                                                     payload.get("echo_likelihood"), payload.get("clipping_ratio", 0))
                        session.trace.record(kind, trace_id, payload)
                        if confirmed and session.generation_id:
                            start = session.turn.candidate_start_ms
                            old = await cancel("barge_in", start)
                            await send("barge_in.confirmed", {"generation_id": old, "start_ms": start,
                                                           "end_ms": end_ms}, generation_id=old, trace_id=trace_id)
                            await send_state(turn_id)
                        elif phase == "start" and session.generation_id and not session.turn.robot_playing:
                            await cancel("speech_during_thinking")
                    if kind == "turn.completed":
                        await send(kind, payload, turn_id, source="turn", trace_id=trace_id)
                        await submit_turn(payload["text"], turn_id, trace_id, source="speech")
                        turn_id = str(uuid4())
                        if not continuous:
                            return
                    else:
                        await send(kind, payload, turn_id, source="speech", trace_id=trace_id)
                if frame is None:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("语音处理失败")
            await send("error", {"code": "asr_error" if isinstance(exc, ASRError) else "speech_error",
                                "message": str(exc) if isinstance(exc, (ASRError, ValueError)) else "语音处理失败，请重新开启麦克风。",
                                "retryable": True}, turn_id, trace_id=trace_id)
        finally:
            if stream and hasattr(stream, "close"):
                await stream.close()
            if session.connection_epoch == epoch and audio_stream_id == stream_id:
                audio_stream_id = None
                completed_streams.add(stream_id)
                if len(completed_streams) > 8:
                    completed_streams.pop()
                session.turn.continuous = False
                session.turn.user_speaking = False
                try:
                    await send("audio.closed", {"stream_id": stream_id}, trace_id=trace_id)
                except Exception:
                    pass

    async def watchdog():
        """检查连接所有权和有界播放等待，异常时清理而非永久卡住。"""
        while True:
            await asyncio.sleep(.25)
            if (session.connection_epoch != epoch or websocket.app.state.sessions.get(session_id) is not session
                    or sender.task.done()):
                await websocket.close(code=1008, reason="会话已关闭或被接管")
                return
            for old in list(session.outputs.values()):
                if (old.status == "cancelled" and not old.stop_confirmed and not old.stop_timed_out
                        and now_ms() - old.cancelled_at_ms > session.stop_ack_timeout_ms):
                    old.stop_timed_out = True
                    session.trace.record("output.stop_timeout", old.trace_id, {"generation_id": old.generation_id})
                    await send("output.stop_timeout", {"generation_id": old.generation_id,
                               "reason": "stop_ack_timeout"}, generation_id=old.generation_id)
            if audio_task and not audio_task.done() and audio_stream_id:
                timeout_ms = 5000 if received_audio_frame else 15000
                if now_ms() - last_audio_received_ms > timeout_ms:
                    audio_task.cancel()
                    await send("error", {"code": "audio_timeout", "message": "麦克风音频中断，已结束采集，请重新开启。"})
            record = session.outputs.get(session.generation_id)
            if record and any(segment.duration_ms and segment.playback not in {"ended", "failed"}
                              for segment in record.segments.values()):
                if now_ms() - session.last_playback_at_ms > getattr(settings, "playback_timeout_ms", 30000):
                    await cancel("playback_timeout")
                    await send("error", {"code": "playback_timeout", "message": "播放确认超时，已停止本轮输出。", "retryable": True})

    monitor = asyncio.create_task(watchdog())
    try:
        await send("session.status", {"status": "connected", "llm_ready": settings.llm_ready,
                                "tts_ready": settings.tts_ready, "speech_ready": websocket.app.state.speech_service.ready,
                                "full_duplex_supported": getattr(settings, "full_duplex_enabled", True)})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if session.connection_epoch != epoch:
                break
            try:
                raw, pcm = decode_message(message)
                model = INPUT_MODELS.get(raw.get("type"))
                if model is None:
                    raise ValueError("不支持的实时事件")
                incoming = model.model_validate(raw)
                if (incoming.session_id != session_id or incoming.connection_epoch != epoch
                        or not session.accept_event(incoming.event_id, incoming.seq)):
                    raise ValueError("会话、连接代次或事件顺序无效")
            except (ValidationError, ValueError, UnicodeDecodeError):
                await send("error", {"code": "invalid_event", "message": "实时事件格式、顺序或连接代次无效。"})
                continue
            kind, payload = incoming.type, incoming.payload
            if kind == "input.text":
                text = payload.text.strip()
                if not text:
                    await send("error", {"code": "invalid_text", "message": "请输入非空文本。"})
                    continue
                if audio_task and not audio_task.done():
                    audio_task.cancel()
                    await asyncio.gather(audio_task, return_exceptions=True)
                await submit_turn(text, incoming.turn_id, incoming.trace_id)
            elif kind == "input.audio_start":
                if (not websocket.app.state.speech_service.ready or (audio_task and not audio_task.done())
                        or (session.generation_id and not (getattr(settings, "full_duplex_enabled", True) and payload.continuous and payload.aec_mode == "browser"))):
                    await send("error", {"code": "audio_unavailable", "message": "语音输入尚未就绪，或当前模式不支持同时录音与播放。"})
                    continue
                session.turn.configure(payload.continuous and getattr(settings, "full_duplex_enabled", True), payload.aec_mode)
                session.trace.record("audio.configured", incoming.trace_id,
                                     {"continuous": session.turn.continuous, "aec_mode": payload.aec_mode}, incoming.event_id)
                audio_stream_id = payload.stream_id
                last_audio_received_ms = now_ms()
                received_audio_frame = False
                audio_queue = asyncio.Queue(maxsize=50)
                audio_task = asyncio.create_task(process_audio(audio_queue, audio_stream_id, session.turn.continuous, incoming.trace_id))
            elif kind == "input.audio_frame":
                if payload.stream_id in completed_streams:
                    continue
                try:
                    if payload.stream_id != audio_stream_id or audio_task is None or audio_task.done() or pcm is None:
                        raise ValueError("语音流未开启")
                    frame = validate_frame(payload.stream_id, payload.frame_seq, payload.capture_start_ms, payload.capture_end_ms, pcm)
                    audio_queue.put_nowait(frame)
                    last_audio_received_ms = now_ms()
                    received_audio_frame = True
                    if frame.seq == 0:
                        await send("audio.received", {"stream_id": frame.stream_id})
                except (ValueError, asyncio.QueueFull):
                    if audio_task:
                        audio_task.cancel()
                    await send("error", {"code": "invalid_audio", "message": "音频无效或队列积压，已结束采集，请重新开启。"})
            elif kind == "input.audio_end":
                if payload.stream_id == audio_stream_id and audio_queue:
                    if payload.submit:
                        try:
                            audio_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            audio_task.cancel()
                            await send("error", {"code": "invalid_audio", "message": "音频队列已满，请重新录音。"})
                    elif audio_task:
                        audio_task.cancel()
            elif kind == "control.stop":
                target = payload.generation_id
                if target is None or target == session.generation_id:
                    session.pending_clarification = None
                    await cancel("user_stop")
                elif target in session.outputs and session.outputs[target].status == "cancelled":
                    await send("output.cancel", session.cancel_payload(target), generation_id=target)
                await send_state()
            elif kind == "response.displayed":
                session.displayed(payload.generation_id, payload.segment_id)
            elif kind.startswith("playback."):
                if kind == "playback.stopped" and payload.generation_id == session.generation_id:
                    await cancel("client_stop")
                if session.record_playback(payload.segment_id, kind, payload.played_ms, payload.duration_ms, payload.generation_id):
                    record = session.outputs[payload.generation_id]
                    session.trace.record(kind, record.trace_id, payload.model_dump(), incoming.event_id)
                    await delivered()
                    await send_state(generation_id=session.generation_id)
            elif kind == "command.ack":
                if session.confirm_stop(payload.generation_id, payload.command_id):
                    record = session.outputs[payload.generation_id]
                    session.trace.record(kind, record.trace_id, payload.model_dump(), incoming.event_id)
                    await send("interruption.updated", session.snapshot_interruption(record).model_dump(),
                               record.turn_id, record.generation_id)
                    await send_state()
    except (WebSocketDisconnect, RuntimeError, ConnectionError, asyncio.TimeoutError):
        pass
    finally:
        monitor.cancel()
        if audio_task and not audio_task.done():
            audio_task.cancel()
        if session.connection_epoch == epoch:
            session.pending_clarification = None
            task = session.generation_task
            session.cancel_generation("disconnect")
            session.turn.disconnected = True
        else:
            task = None
        # 每个连接回收自己启动的任务；接管不能丢失已取消旧生成的引用。
        owned_tasks = list(generation_tasks)
        for owned in owned_tasks:
            if not owned.done():
                owned.cancel()
        await asyncio.gather(*(item for item in (monitor, audio_task, task, *owned_tasks) if item), return_exceptions=True)
        await sender.close()




