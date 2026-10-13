"""作用：定义已提交轮次的意图识别结果。

当前功能：
- 约束七类意图、控制动作、参数值与文本证据。
- 校验互斥字段、缺失参数和重复槽位，防止非法结果进入路由。
- 显式区分补充待澄清请求与发起新的工具请求。
- 标记需要多步规划的组合任务，维持现有七类意图。
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


IntentName = Literal["chat", "question", "tool_request", "memory_query", "memory_update", "control", "other"]
ControlAction = Literal["stop", "cancel", "repeat", "resume", "change_topic"]


class IntentSlot(BaseModel):
    """保存一个参数的 JSON 值及最终输入或近期上下文中的原文证据。"""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    name: str = Field(min_length=1, max_length=80)
    value: JsonValue
    evidence: str | None = Field(default=None, min_length=1, max_length=500)


class IntentEstimate(BaseModel):
    """表示单个主要意图；置信度仅作观测，不提供执行权限。"""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    intent: IntentName
    slots: list[IntentSlot] = Field(max_length=32)
    confidence: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    tool_intent: str | None = Field(default=None, min_length=1, max_length=100)
    needs_clarification: bool
    missing_slots: list[str] = Field(max_length=32)
    control_action: ControlAction | None
    continues_pending: bool = False
    needs_planning: bool = False

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        """检查意图与附属字段的一致性，并拒绝冲突或重复参数。"""
        if self.intent != "tool_request" and self.tool_intent is not None:
            raise ValueError("非工具意图不能指定工具")
        if self.needs_planning and self.intent != "tool_request":
            raise ValueError("只有实际工具任务可以进入规划")
        if self.continues_pending and (self.intent != "tool_request" or self.tool_intent is None):
            raise ValueError("只有明确工具请求可以补充待澄清任务")
        if (self.intent == "control") != (self.control_action is not None):
            raise ValueError("控制意图必须且只能提供控制动作")
        if self.missing_slots and not self.needs_clarification:
            raise ValueError("缺少参数时必须请求澄清")
        names = [slot.name for slot in self.slots]
        if len(names) != len(set(names)) or len(self.missing_slots) != len(set(self.missing_slots)):
            raise ValueError("参数名称不能重复")
        if any(not name.strip() or len(name) > 80 for name in self.missing_slots):
            raise ValueError("缺失参数名称无效")
        supplied = {slot.name for slot in self.slots if slot.value is not None}
        if supplied.intersection(self.missing_slots):
            raise ValueError("已提供参数不能同时标记缺失")
        return self

    @classmethod
    def uncertain(cls) -> Self:
        """构造不含猜测参数的统一澄清结果。"""
        return cls(intent="other", slots=[], confidence=None, tool_intent=None,
                   needs_clarification=True, missing_slots=[], control_action=None)
