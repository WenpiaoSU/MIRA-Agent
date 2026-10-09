"""作用：定义 Agent 回复内容的数据结构。

当前功能：
- 约束流式文本片段的标识、顺序与内容。
- 保存本轮回复计划及预留的语义字段。
"""

from typing import Literal

from pydantic import BaseModel, Field


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
