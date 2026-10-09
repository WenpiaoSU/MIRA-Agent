"""作用：定义 P3 使用的最小、不可变社会状态快照。

当前功能：
- 分离用户语音活动与机器人实际播放状态。
- 为 Agent 提供状态版本、最终输入、中断和健康摘要。
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict


class InteractionState(BaseModel):
    """保存一次冻结的轮次观测，不以生成计划冒充实际播放。"""
    model_config = ConfigDict(frozen=True)
    mode: Literal["idle", "listening", "thinking", "speaking", "recovering", "disconnected"]
    user_speaking: bool
    robot_playing: bool
    generation_id: str | None = None
    continuous: bool = False
    aec_mode: Literal["browser", "unavailable"] = "unavailable"


class SocialState(BaseModel):
    """表示提供给 Agent 的最小状态；多模态扩展沿用同一版本边界。"""
    model_config = ConfigDict(frozen=True)
    session_id: str
    user_id: None = None
    version: int
    as_of_ms: int
    interaction: InteractionState
    committed_text: str | None = None
