"""作用：定义长期记忆的候选、记录与真实提交结果。

当前功能：
- 约束单轮写入、修改、删除和澄清的结构化输出。
- 保存用户归属、原文证据、事实版本及有效性。
- 将提交回执与模型建议分离，禁止模型自行声明写入成功。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCandidate(BaseModel):
    """描述本轮一个记忆操作；身份、版本与执行结果由服务端附加。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["remember", "update", "delete", "reject", "clarify"]
    kind: Literal["semantic", "episodic"] = "semantic"
    fact_key: str = Field(default="", max_length=100)
    content: str = Field(default="", max_length=500)
    evidence: str = Field(default="", max_length=1000)
    target_ids: list[str] = Field(default_factory=list, max_length=20)
    delete_all: bool = False
    question: str = Field(default="", max_length=300)


class MemoryRecord(BaseModel):
    """保存一条可追溯的长期事实或事件，旧版本仅用于显式历史查询。"""
    memory_id: str
    user_id: str
    persona_id: str = "mira"
    kind: Literal["semantic", "episodic"]
    fact_key: str
    content: str
    evidence: str
    source_session_id: str
    source_turn_id: str
    version: int
    status: Literal["active", "inactive"]
    created_at: str
    valid_to: str | None = None
    supersedes_id: str | None = None


class MemoryResult(BaseModel):
    """提供数据库确认后的结果，删除回执只描述本系统实际清理的范围。"""
    status: Literal["saved", "unchanged", "deleted", "clarify", "rejected", "failed", "conflict"]
    message: str
    memory_ids: list[str] = Field(default_factory=list)
    deleted_count: int = 0
    cleared: bool = False
