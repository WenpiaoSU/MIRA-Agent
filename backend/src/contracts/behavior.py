"""作用：定义 Agent 可输出的角色语义及行为计划契约。

当前功能：
- 限定表情、视线、动作和强度的合法取值。
- 约束发送给前端的 BehaviorPlan 结构。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# 以下类型限定 Agent 可使用的语义名称，不包含具体模型参数。
Expression = Literal["neutral", "gentle_smile", "concerned", "sad", "confused"]
Gaze = Literal["center", "away"]
Gesture = Literal["nod", "shake_head", "tilt", "lower_head"]


class AvatarCue(BaseModel):
    """校验语言模型首行 JSON 中的角色表达线索。"""
    model_config = ConfigDict(extra="forbid")

    expression: Expression = "neutral"
    gaze: Gaze = "center"
    gesture: Gesture | None = None
    intensity: float = Field(default=0.65, ge=0, le=1)


class BehaviorPlan(BaseModel):
    """表示后端发送给前端的单轮角色行为安排。"""
    plan_id: str
    generation_id: str
    response_id: str
    speech: None = None
    emotion: dict[str, str | float]
    face: dict[str, str | dict]
    gaze: dict[str, str]
    gestures: list[dict[str, str | float]]
    state: Literal["responding"] = "responding"
    timing: dict[str, str | int]
    constraints: dict[str, bool | str | float]
