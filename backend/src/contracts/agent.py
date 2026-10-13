"""作用：定义 Agent 回复内容的数据结构。

当前功能：
- 约束流式文本片段的标识、顺序与内容。
- 保存本轮回复计划及预留的语义字段。
- 定义有界任务草案、前序结果引用和服务端执行状态。
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from src.contracts.tools import ToolResult


class TextSegment(BaseModel):
    """描述一段按顺序输出给用户的文本。"""
    segment_id: str
    index: int = Field(ge=1)
    text: str = Field(min_length=1)


class ResponsePlan(BaseModel):
    """汇总本轮回复的文本片段和语义元信息。"""
    response_id: str
    text_segments: list[TextSegment] = Field(default_factory=list)
    speech_act: Literal["answer", "empathic_ack", "clarify"] = "answer"
    style: Literal["warm", "neutral"] = "warm"
    claims: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    clarification_slots: list[str] = Field(default_factory=list)


class ResultReference(BaseModel):
    """引用前序工具步骤的 JSON Pointer 字段；step 从 1 开始计数。"""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    step: int = Field(ge=1, le=12)
    path: str = Field(min_length=1, max_length=200, pattern=r"^/(?:[^~]|~[01])*$")


class StepDraft(BaseModel):
    """模型仅描述步骤和参数，不得指定执行状态或宣称成功。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["tool", "respond"]
    tool_name: str | None = Field(default=None, max_length=100)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    argument_evidence: dict[str, str] = Field(default_factory=dict)
    bindings: dict[str, ResultReference] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        """工具名称仅属于调用步骤，字面参数与结果引用不能覆盖同一参数。"""
        if self.kind == "respond" and (self.tool_name or self.arguments or self.argument_evidence or self.bindings):
            raise ValueError("回答步骤不能携带工具参数")
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("工具步骤缺少名称")
        if set(self.arguments).intersection(self.bindings):
            raise ValueError("参数不能同时提供字面值和引用")
        return self


class PlanDraft(BaseModel):
    """约束模型生成的有限步骤列表，实际步数上限由应用再次校验。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: str = Field(min_length=1, max_length=1000)
    steps: list[StepDraft] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def validate_sequence(self) -> Self:
        """一个计划必须包含工具步骤，并以唯一的回答步骤结束。"""
        if self.steps[-1].kind != "respond" or any(step.kind != "tool" for step in self.steps[:-1]):
            raise ValueError("计划必须先执行工具，最后回答")
        return self


class PlanStep(StepDraft):
    """保存一个步骤的服务器标识、执行进度和真实工具结果。"""
    step_id: str
    status: Literal["pending", "running", "waiting_input", "succeeded", "failed", "cancelled"] = "pending"
    result: ToolResult | None = None


class TaskPlan(BaseModel):
    """保存会话内的一个任务；多轮补参继续原步骤而不重新生成计划。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    plan_id: str
    original_text: str = Field(min_length=1, max_length=4000)
    goal: str
    steps: list[PlanStep] = Field(min_length=2, max_length=12)
    status: Literal["pending", "running", "waiting_input", "completed", "failed", "cancelled"] = "pending"
    response_text: str = ""


class PlanAnswer(BaseModel):
    """只允许有界建议或综合解释，引用必须对应已验证的工具步骤。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)
    result_refs: list[int] = Field(min_length=1, max_length=11)
