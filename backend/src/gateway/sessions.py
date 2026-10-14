"""作用：维护会话、输出账本及取消后的最终停止回执。

当前功能：
- 校验连接事件顺序并保存有界轮次历史。
- 分离生成计划、字幕展示、音频播放和输出失败。
- 取消任务时保存中断快照，迟到回执只更新所属旧输出。
- 提供冻结社会状态及脱敏轨迹，不让存储阻塞停止链路。
- 将当前轮次的意图显式加入快照，不保存跨轮次的隐式意图。
- 保存一个待澄清工具请求和有界结果账本，控制请求不重新执行查询。
- 保存一个活跃任务计划，并在取消或连接接管时终止剩余步骤。
- 按浏览器身份隔离长期记忆；删除后清除相关交付缓存并通知各连接。
"""

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from src.contracts.output import InterruptionState, OutputRecord, SegmentDelivery
from src.contracts.state import SocialState
from src.contracts.perception import IntentEstimate
from src.contracts.tools import PendingClarification
from src.contracts.agent import TaskPlan
from src.evaluation.trace import TraceRecorder, now_ms
from src.turn.manager import TurnManager


@dataclass
class Session:
    """保存单会话连接、输出代次与有界交付记录。"""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    owner_token_hash: str | None = None
    memory_notifications: list[dict] = field(default_factory=list)
    invalidate_checkpoint: Callable[[str, str], None] | None = None
    connection_epoch: int = 0
    next_seq: int = 0
    last_client_seq: int = -1
    seen_event_ids: set[str] = field(default_factory=set)
    recent_avatar_cues: list[dict[str, str | None]] = field(default_factory=list)
    generation_task: asyncio.Task | None = None
    generation_id: str | None = None
    outputs: OrderedDict[str, OutputRecord] = field(default_factory=OrderedDict)
    interruption: InterruptionState | None = None
    pending_clarification: PendingClarification | None = None
    active_plan: TaskPlan | None = None
    turn: TurnManager = field(default_factory=TurnManager)
    trace: TraceRecorder = field(default_factory=TraceRecorder)
    last_playback_at_ms: int = 0
    stop_ack_timeout_ms: int = 2000

    def accept_event(self, event_id: str, seq: int) -> bool:
        """拒绝重复事件和倒退序号，保持单连接的顺序边界。"""
        if event_id in self.seen_event_ids or seq <= self.last_client_seq:
            return False
        self.last_client_seq = seq
        self.seen_event_ids.add(event_id)
        if len(self.seen_event_ids) > 256:
            self.seen_event_ids = {event_id}
        return True

    def begin_generation(self, generation_id: str, turn_id: str = "", user_text: str = "",
                         trace_id: str = "", speech_enabled: bool = False) -> OutputRecord:
        """立即保存用户最终输入，新代次不会清除旧输出的停止记录。"""
        self.generation_id = generation_id
        record = OutputRecord(generation_id=generation_id, turn_id=turn_id,
                              user_text=user_text, trace_id=trace_id or generation_id,
                              connection_epoch=self.connection_epoch, speech_enabled=speech_enabled)
        self.outputs[generation_id] = record
        while len(self.outputs) > 24:
            self.outputs.popitem(last=False)
        self.turn.begin(generation_id)
        self.last_playback_at_ms = now_ms()
        self.trace.record("generation.started", record.trace_id, {"generation_id": generation_id})
        return record

    def cancel_generation(self, reason: str = "user_stop") -> str | None:
        """立即失效并取消生成任务，保留账本以接收最终停止位置。"""
        if self.active_plan and (self.active_plan.status == "running" or
                                 reason in {"user_stop", "disconnect", "takeover", "generation_timeout", "change_topic"}):
            self.cancel_plan()
        old = self.generation_id
        if old is None:
            return None
        record = self.outputs[old]
        record.status = "cancelled"
        record.cancel_reason = reason
        record.cancel_command_id = str(uuid4())
        record.cancelled_at_ms = now_ms()
        self.generation_id = None
        if self.invalidate_checkpoint:
            self.invalidate_checkpoint(self.session_id, old)
        self.turn.cancel(old)
        self.interruption = self.snapshot_interruption(record)
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task = self.generation_task
        self.generation_task = None
        if task and not task.done() and task is not current_task:
            task.cancel()
        return old

    def cancel_plan(self) -> None:
        """结束未完成计划并保留成功结果；明确停止也清除尚未播报的澄清目标。"""
        if self.active_plan:
            plan = self.active_plan.model_copy(deep=True)
            plan.status = "cancelled"
            for step in plan.steps:
                if step.status in {"pending", "running", "waiting_input"}:
                    step.status = "cancelled"
            for record in self.outputs.values():
                if record.task_plan and record.task_plan.plan_id == plan.plan_id:
                    # 历史输出保留当时的结果，不能把新轮次结果回填到旧 generation。
                    snapshot = record.task_plan.model_copy(deep=True)
                    snapshot.status = "cancelled"
                    for step in snapshot.steps:
                        if step.status in {"pending", "running", "waiting_input"}:
                            step.status = "cancelled"
                    record.task_plan = snapshot
        self.active_plan = None
        self.pending_clarification = None

    def cancel_payload(self, generation_id: str) -> dict:
        """构造幂等取消命令，同一输出重复发送使用原 command_id。"""
        record = self.outputs[generation_id]
        return {"command_id": record.cancel_command_id, "generation_id": generation_id,
                "reason": record.cancel_reason, "stop_audio": True, "clear_timeline": True}

    def register_segment(self, generation_id: str, payload: dict) -> None:
        """登记不可变稳定文本段，同一标识只能提交一次。"""
        record = self.outputs[generation_id]
        segment_id = payload["segment_id"]
        if segment_id not in record.segments:
            record.segments[segment_id] = SegmentDelivery(
                segment_id=segment_id, index=payload.get("index", len(record.segments) + 1),
                text=payload["text"])

    def register_audio(self, segment_id: str, duration_ms: int, generation_id: str | None = None) -> None:
        """仅对已登记稳定语段保存实际合成时长。"""
        record = self.outputs[generation_id or self.generation_id]
        record.segments[segment_id].duration_ms = duration_ms
        self.last_playback_at_ms = now_ms()

    def displayed(self, generation_id: str, segment_id: str) -> bool:
        """确认已显示的字幕，旧取消输出不能新增展示事实。"""
        record = self.outputs.get(generation_id)
        if record is None or record.connection_epoch != self.connection_epoch or record.status == "cancelled":
            return False
        segment = record.segments.get(segment_id)
        if segment is None:
            return False
        segment.displayed = True
        return True

    def record_playback(self, segment_id: str, event_type: str, played_ms: int, duration_ms: int,
                        generation_id: str | None = None) -> bool:
        """校验真实时长与单调位置；取消后只接受有限窗口的终态回执。"""
        target = generation_id or self.generation_id
        record = self.outputs.get(target)
        if record is None or record.connection_epoch != self.connection_epoch:
            return False
        segment = record.segments.get(segment_id)
        if (segment is None or segment.duration_ms != duration_ms or played_ms < 0
                or played_ms > duration_ms + 500):
            return False
        if record.status == "cancelled":
            if (record.stop_confirmed or event_type not in {"playback.stopped", "playback.ended", "playback.failed"}
                    or now_ms() - record.cancelled_at_ms > self.stop_ack_timeout_ms):
                return False
        elif target != self.generation_id:
            return False
        if segment.playback in {"ended", "failed"}:
            return False
        phase = event_type.split(".")[1]

        if phase == "ended" and played_ms < duration_ms - 150:
            return False
        segment.played_ms = max(segment.played_ms, min(played_ms, duration_ms))
        if phase in {"started", "ended", "stopped", "failed"}:
            segment.playback = phase
        self.turn.playback(target, segment_id, phase)
        self.last_playback_at_ms = now_ms()
        if self.interruption and self.interruption.generation_id == target:
            self.interruption = self.snapshot_interruption(record)
        return True

    def confirm_stop(self, generation_id: str, command_id: str) -> bool:
        """确认取消命令已执行；旧命令不能解除新输出的播放状态。"""
        record = self.outputs.get(generation_id)
        if (record is None or record.status != "cancelled" or record.stop_confirmed
                or record.connection_epoch != self.connection_epoch
                or record.cancel_command_id != command_id
                or now_ms() - record.cancelled_at_ms > self.stop_ack_timeout_ms):
            return False
        record.stop_confirmed = True
        self.turn.stopped(generation_id)
        if self.interruption and self.interruption.generation_id == generation_id:
            self.interruption = self.snapshot_interruption(record)
        self.trace.record("output.stop_confirmed", record.trace_id,
                          {"generation_id": generation_id, "elapsed_ms": now_ms() - record.cancelled_at_ms})
        return True

    @staticmethod
    def snapshot_interruption(record: OutputRecord) -> InterruptionState:
        """只把完整播放句子视为确定已说；部分句子保守从句首恢复。"""
        spoken, remaining, displayed = [], [], []
        last_segment = None
        played_ms = 0
        precision = "sentence"
        for segment in sorted(record.segments.values(), key=lambda item: item.index):
            if segment.displayed:
                displayed.append(segment.text)
            if segment.played_ms > 0:
                last_segment, played_ms = segment.segment_id, segment.played_ms
            if segment.playback == "ended":
                spoken.append(segment.text)
            else:
                remaining.append(segment.text)
                if segment.played_ms > 0:
                    precision = "approximate"
        return InterruptionState(
            generation_id=record.generation_id, turn_id=record.turn_id,
            original_input=record.user_text, reason=record.cancel_reason or "interrupted",
            last_played_segment_id=last_segment, played_audio_ms=played_ms,
            spoken_text="".join(spoken), displayed_text="".join(displayed),
            remaining_text="".join(remaining), spoken_text_precision=precision,
            response_complete=record.response_complete, stop_confirmed=record.stop_confirmed,
            completed_tool_call_ids=list(dict.fromkeys(result.call_id for result in
                                                       [*record.tool_results, *([record.tool_result] if record.tool_result else [])]
                                                       if result.verified)),
            resume_allowed=record.cancel_reason not in {"disconnect", "takeover"})

    def finish_if_delivered(self) -> OutputRecord | None:
        """生成及合成结束后，全部音频进入终态才释放输出；失败不会算成功。"""
        record = self.outputs.get(self.generation_id)
        if record is None or not record.response_complete or not record.synthesis_complete:
            return None
        if any(segment.duration_ms is not None and segment.playback not in {"ended", "failed"}
               for segment in record.segments.values()):
            return None
        record.status = ("degraded" if any(segment.playback == "failed" for segment in record.segments.values())
                         else "delivered")
        self.generation_id = None
        self.turn.generating = False
        self.turn.stopped(record.generation_id)
        return record

    def context_history(self, exclude_generation: str | None = None) -> list[dict[str, str]]:
        """仅使用交付事实构建近期上下文，并显式区分字幕与实际说出。"""
        messages = []
        for turn in self.context_turns():
            if turn["generation_id"] == exclude_generation:
                continue
            if turn["user"]:
                messages.append({"role": "user", "content": turn["user"]})
            if turn["assistant"]:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages[-12:]

    def context_turns(self) -> list[dict[str, str]]:
        """提供可按 generation 合并的交付事实，由 Agent 同步到 Checkpointer。"""
        turns = []
        for record in self.outputs.values():
            displayed = "".join(segment.text for segment in record.segments.values() if segment.displayed)
            spoken = "".join(segment.text for segment in record.segments.values() if segment.playback == "ended")
            content = ""
            if displayed or spoken:
                content = displayed or spoken
                if record.speech_enabled:
                    content += f"\n【交付记录：完整播完的句子={spoken or '无'}；状态={record.status}；字幕不等于已说出】"
            turns.append({"generation_id": record.generation_id, "user": record.user_text, "assistant": content,
                          "created_at": str(record.created_at)})
        return turns

    @property
    def history(self) -> list[dict[str, str]]:
        """提供由输出账本推导的兼容历史读取接口。"""
        return self.context_history()

    def social_snapshot(self, text: str | None = None, intent: IntentEstimate | None = None) -> SocialState:
        """冻结本轮文本与意图；没有新识别结果时意图为空。"""
        return SocialState(session_id=self.session_id, user_id=self.user_id, version=self.turn.version,
                           as_of_ms=now_ms(), interaction=self.turn.snapshot(), committed_text=text,
                           intent=intent.model_copy(deep=True) if intent else None)


class SessionRegistry:
    """管理服务进程内会话并在删除时停止生成。"""

    def __init__(self):
        """初始化按会话标识索引的内存表。"""
        self.sessions: dict[str, Session] = {}
        self.forget_checkpoint = None
        self.invalidate_checkpoint = None

    def create(self, user_id: str | None = None, owner_token_hash: str | None = None) -> Session:
        """创建会话；用户归属由服务端验证后的浏览器凭据提供。"""
        session = Session(user_id=user_id, owner_token_hash=owner_token_hash,
                          invalidate_checkpoint=self.invalidate_checkpoint)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """按标识查询会话，不存在时返回空值。"""
        return self.sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        """删除会话并取消其生成；连接任务由网关退出时统一回收。"""
        session = self.sessions.pop(session_id, None)
        if session:
            session.cancel_generation("disconnect")
            if self.forget_checkpoint:
                self.forget_checkpoint(session_id)

    def forget_memories(self, user_id: str, current_session: str = "", current_generation: str = ""):
        """删除用户记忆后清理全部会话派生上下文，取消其他正在使用旧事实的输出。"""
        for session in self.sessions.values():
            if session.user_id != user_id:
                continue
            current = session.session_id == current_session and session.generation_id == current_generation
            record = session.outputs.get(current_generation) if current else None
            if not current:
                old = session.cancel_generation("memory_deleted")
                if old:
                    session.memory_notifications.append({"type": "output.cancel", "payload": session.cancel_payload(old),
                                                         "generation_id": old})
            session.cancel_plan()
            session.interruption = None
            session.outputs.clear()
            session.recent_avatar_cues.clear()
            if record:
                record.user_text = ""
                record.tool_result = None
                record.tool_results = []
                record.task_plan = None
                session.outputs[current_generation] = record
            if self.forget_checkpoint:
                self.forget_checkpoint(session.session_id)
            session.memory_notifications.append({"type": "memory.cleared", "payload": {}, "generation_id": None})

