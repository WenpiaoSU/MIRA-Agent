"""作用：定义输出交付与中断恢复的设备无关契约。

当前功能：
- 区分文本展示、音频播放、取消确认及播放失败。
- 保存带来源和精确度标志的中断快照。
"""

from typing import Literal
from pydantic import BaseModel, Field


class SegmentDelivery(BaseModel):
    """保存稳定语段的文本、音频和真实交付位置。"""
    segment_id: str
    index: int
    text: str
    displayed: bool = False
    duration_ms: int | None = None
    played_ms: int = 0
    playback: Literal["pending", "started", "ended", "stopped", "failed"] = "pending"


class InterruptionState(BaseModel):
    """表示可恢复的已生成输出；未生成的内容不假造为剩余文本。"""
    generation_id: str
    turn_id: str
    original_input: str
    reason: str
    last_played_segment_id: str | None = None
    played_audio_ms: int = 0
    spoken_text: str = ""
    displayed_text: str = ""
    remaining_text: str = ""
    spoken_text_precision: Literal["sentence", "approximate"] = "sentence"
    response_complete: bool = False
    stop_confirmed: bool = False
    completed_tool_call_ids: list[str] = Field(default_factory=list)
    resume_allowed: bool = True


class OutputRecord(BaseModel):
    """保存一个 generation 的生成计划和客户端交付事实。"""
    generation_id: str
    turn_id: str
    user_text: str
    trace_id: str
    connection_epoch: int
    speech_enabled: bool
    segments: dict[str, SegmentDelivery] = Field(default_factory=dict)
    response_complete: bool = False
    synthesis_complete: bool = False
    status: Literal["active", "delivered", "degraded", "cancelled"] = "active"
    cancel_reason: str | None = None
    cancel_command_id: str | None = None
    cancelled_at_ms: int | None = None
    stop_confirmed: bool = False
    stop_timed_out: bool = False
