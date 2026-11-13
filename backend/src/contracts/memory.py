"""作用：定义长期记忆的候选、记录与真实提交结果。

当前功能：
- 约束原子事实提取、语义比较、分级门控和明确删除操作。
- 保存用户归属、原文证据、事实版本及有效性。
- 将提交回执与模型建议分离，禁止模型自行声明写入成功。
- 定义待确认快照、用户决定及展示回执，绑定旧事实版本与具体候选。
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
    memory_type: Literal["fact", "preference", "goal", "decision", "event"] = "fact"
    domain_scope: str = Field(default="daily", min_length=1, max_length=120)
    expires_at: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    source_kind: Literal["user", "tool"] = "user"
    source_ref: str = "user"


class ExtractedFact(BaseModel):
    """固定语义门控输出；正文须为原子原文，不允许模型补充事实。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    should_write: bool
    memory_type: Literal["fact", "preference", "goal", "decision", "event"]
    content: str = Field(max_length=500)
    evidence: str = Field(max_length=1000)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    is_temporary: bool
    fact_key: str = Field(default="", max_length=100)
    domain_scope: str = Field(default="daily", min_length=1, max_length=120)
    expires_at: str | None = None
    source_id: str = "user"


class FactExtraction(BaseModel):
    """一轮最多提取八条独立事实，不把整段对话当成长期摘要。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    facts: list[ExtractedFact] = Field(max_length=8)


class FactRelation(BaseModel):
    """判断候选与已有事实的语义关系；相似度不参与最终动作判定。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    relation: Literal["same", "different", "conflict", "uncertain"]
    target_id: str | None = None


class MemoryDecision(BaseModel):
    """记录整合决策；PENDING 与有效事实分开存放。"""
    action: Literal["ADD", "UPDATE", "SKIP", "PENDING"]
    candidate: MemoryCandidate
    target_id: str | None = None
    reason: str = ""


class MemoryEvidence(BaseModel):
    """供服务端注入可信工具证据，不能从意图槽位或模型正文构造。"""
    source_id: str
    text: str
    verified: bool = False


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
    memory_type: Literal["fact", "preference", "goal", "decision", "event"] = "fact"
    domain_scope: str = "daily"
    expires_at: str | None = None
    confidence: float = 1.0
    source_kind: Literal["user", "tool"] = "user"
    source_ref: str = "user"


class MemoryResult(BaseModel):
    """提供数据库确认后的结果，删除回执只描述本系统实际清理的范围。"""
    status: Literal["saved", "unchanged", "deleted", "clarify", "rejected", "failed", "conflict", "queued", "pending", "not_found", "expired"]
    message: str
    memory_ids: list[str] = Field(default_factory=list)
    deleted_count: int = 0
    cleared: bool = False
    decisions: list[Literal["ADD", "UPDATE", "SKIP", "PENDING"]] = Field(default_factory=list)
    pending_ids: list[str] = Field(default_factory=list)
    changed: bool = False


class MemoryResolution(BaseModel):
    """用户对一个具体版本冲突的决定，不能在客户端指定归属或新事实正文。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["accept_new", "keep_old"]
    target_version: int | None = Field(ge=1)


class PendingMemory(BaseModel):
    """用于查看和提示的待确认快照，不等同于有效长期事实。"""
    pending_id: str
    candidate: MemoryCandidate
    target_id: str | None
    target_version: int | None
    target_content: str
    reason: str
    question: str
    created_at: str
    expires_at: float


class MemoryPendingAck(BaseModel):
    """确认本连接已展示指定候选问题，令短答具备唯一可验证的目标。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    pending_id: str = Field(min_length=1, max_length=100)
    target_version: int | None = Field(ge=1)
    delivery_token: str = Field(min_length=1, max_length=100)


class MemoryConfirmation(MemoryResolution):
    """由网关确认上下文产生的受约束决定，不由意图分类或模型生成。"""
    pending_id: str
