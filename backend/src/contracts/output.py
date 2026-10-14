"""作用：定义输出交付与中断恢复的设备无关契约。

当前功能：
- 区分文本展示、音频播放、取消确认及播放失败。
- 保存带来源和精确度标志的中断快照。
- 在输出账本保存单次工具结果，恢复时复述原事实而不重复执行。
- 保存有界的多步结果和任务快照，取消后恢复只复述原事实。
- 记录轮次创建时间，为短期记忆过期过滤提供依据。
"""

from typing import Literal
from time import time
from pydantic import BaseModel, Field
from src.contracts.tools import ToolResult
from src.contracts.agent import TaskPlan


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
    created_at: float = Field(default_factory=time)
    turn_id: str
    user_text: str
    trace_id: str
    connection_epoch: int
    speech_enabled: bool
    tool_result: ToolResult | None = None
    tool_results: list[ToolResult] = Field(default_factory=list, max_length=24)
    task_plan: TaskPlan | None = None
    segments: dict[str, SegmentDelivery] = Field(default_factory=dict)
    response_complete: bool = False
    synthesis_complete: bool = False
    status: Literal["active", "delivered", "degraded", "cancelled"] = "active"
    cancel_reason: str | None = None
    cancel_command_id: str | None = None
    cancelled_at_ms: int | None = None
    stop_confirmed: bool = False
    stop_timed_out: bool = False
