"""作用：定义 M08 社会状态、类型化事件和 Agent 紧凑上下文。

当前功能：
- 分离最终输入、用户情感、视觉观测与已确认的机器人表达。
- 约束事件归属、时间、来源、健康、关系及执行状态。
- 提供版本快照、增量回执和兼容原有 Agent 的状态字段。
"""

from typing import Literal, Self
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator
from src.contracts.behavior import Expression
from src.contracts.perception import IntentEstimate


class StateModel(BaseModel):
    """冻结属性并拒绝未知字段；服务边界深拷贝容器以隔离快照。"""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class InteractionState(StateModel):
    """保存一次冻结的轮次观测，不以生成计划冒充实际播放。"""
    mode: Literal["idle", "listening", "thinking", "speaking", "recovering", "disconnected"]
    user_speaking: bool
    robot_playing: bool
    generation_id: str | None = Field(default=None, max_length=160)
    continuous: bool = False
    aec_mode: Literal["browser", "unavailable"] = "unavailable"
    turn_id: str | None = Field(default=None, max_length=160)


class CommittedUtterance(StateModel):
    """只接收轮次提交后的完整文本；质量未知时保留空值。"""
    text: str = Field(min_length=1, max_length=16000)
    source: Literal["text", "speech"]
    utterance_id: str | None = None
    quality: float | None = Field(default=None, ge=0, le=1)


class ModalitySignal(StateModel):
    """携带单个情感模态的可用性、归一化质量和融合权重。"""
    available: bool
    quality: float | None = Field(default=None, ge=0, le=1)
    weight: float = Field(default=0, ge=0, le=1)


class UserAffectState(StateModel):
    """保存用户情绪线索，不解释为心理事实。"""
    emotion: Literal["unknown", "neutral", "happy", "frustrated", "sad", "angry", "anxious"] = "unknown"
    confidence: float | None = Field(default=None, ge=0, le=1)
    affect_distribution: dict[str, float] = Field(default_factory=dict, max_length=7)
    is_uncertain: bool = True
    modalities: dict[Literal["text", "audio", "visual"], ModalitySignal] = Field(default_factory=dict)
    evidence: tuple[str, ...] = Field(default=(), max_length=16)
    conflicts: tuple[str, ...] = Field(default=(), max_length=8)
    missing_modalities: tuple[Literal["text", "audio", "visual"], ...] = ()
    window_start_ms: int | None = Field(default=None, ge=0)
    window_end_ms: int | None = Field(default=None, ge=0)
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        """校验概率分布、模态权重及成对时间窗口；空分布表示没有证据。"""
        labels = {"unknown", "neutral", "happy", "frustrated", "sad", "angry", "anxious"}
        distribution = self.affect_distribution
        if distribution and (set(distribution) - labels or
                             any(not 0 <= value <= 1 for value in distribution.values()) or
                             abs(sum(distribution.values()) - 1) > 1e-6):
            raise ValueError("情感分布必须包含合法标签且归一化")
        if self.modalities:
            if any(not item.available and item.weight != 0 for item in self.modalities.values()):
                raise ValueError("缺失模态不能分配权重")
            weight = sum(item.weight for item in self.modalities.values())
            if any(item.available for item in self.modalities.values()) and abs(weight - 1) > 1e-6:
                raise ValueError("有效模态权重之和必须为 1")
        if (self.window_start_ms is None) != (self.window_end_ms is None) or (
                self.window_start_ms is not None and self.window_end_ms < self.window_start_ms):
            raise ValueError("情感窗口必须成对提供且不能倒退")
        return self


class VisualState(StateModel):
    """保存当前选定跟踪对象；track_id 不代表用户身份。"""
    presence: Literal["unknown", "present", "absent"] = "unknown"
    active_track_id: str | None = Field(default=None, max_length=160)
    face_center: tuple[float, float] | None = None
    scene_summary: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        """脸中心使用归一化坐标；未检测到人时不得携带跟踪位置。"""
        if self.face_center and any(not 0 <= value <= 1 for value in self.face_center):
            raise ValueError("脸中心坐标必须位于 [0,1]")
        if self.presence != "present" and (self.active_track_id is not None or self.face_center is not None):
            raise ValueError("没有当前目标时不能提供跟踪位置")
        return self


class DialogueState(StateModel):
    """保存已提交轮次和待澄清目标，历史正文仍由现有 Checkpointer 管理。"""
    topic: str | None = Field(default=None, max_length=500)
    pending_question: str | None = Field(default=None, max_length=1000)
    recent_turn_ids: tuple[str, ...] = Field(default=(), max_length=12)
    domain_scope: Literal["daily", "study", "project"] = "daily"


class RelationshipState(StateModel):
    """只接收同一用户的关系汇总，熟悉度不赋予操作权限。"""
    user_id: str | None = None
    version: int = Field(default=0, ge=0)
    familiarity: float | None = Field(default=None, ge=0, le=1)
    interaction_count: int = Field(default=0, ge=0)
    stage: Literal["unknown", "new", "familiar"] = "unknown"
    proactive_allowed: bool = False


class RobotAffectState(StateModel):
    """记录 M16 表达经执行回执确认的状态；未知时不借用用户情绪。"""
    expression: Expression | None = None
    intensity: float | None = Field(default=None, ge=0, le=1)
    behavior_plan_id: str | None = Field(default=None, max_length=160)


class ActionState(StateModel):
    """保存绑定代次与调用标识的执行摘要，不携带完整工具数据。"""
    action_id: str = Field(min_length=1, max_length=160)
    kind: Literal["tool", "playback"]
    status: Literal["pending", "running", "ok", "degraded", "error", "cancelled", "unknown"]
    generation_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=160)
    verified: bool = False


class HealthState(StateModel):
    """记录组件可用性；更新时间保存在对应 provenance 中。"""
    component: str = Field(min_length=1, max_length=80)
    status: Literal["ok", "degraded", "unavailable"]
    reason: str | None = Field(default=None, max_length=500)
    updated_at: AwareDatetime | None = None


class Provenance(StateModel):
    """保留某个状态域最近一次接受的证据及有效期。"""
    event_id: str
    source: str
    seq: int
    captured_at: AwareDatetime
    received_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    turn_id: str | None = None
    generation_id: str | None = None


class SocialState(StateModel):
    """提供与会话和身份绑定的版本快照；旧快照不随新事件变化。"""
    session_id: str = Field(min_length=1, max_length=160)
    user_id: str | None = Field(default=None, max_length=160)
    connection_epoch: int = 0
    version: int = Field(ge=0)
    as_of: AwareDatetime
    as_of_ms: int
    interaction: InteractionState
    committed_text: str | None = None
    intent: IntentEstimate | None = None
    utterance: CommittedUtterance | None = None
    user_affect: UserAffectState | None = None
    visual: VisualState = Field(default_factory=VisualState)
    dialogue: DialogueState = Field(default_factory=DialogueState)
    relationship: RelationshipState = Field(default_factory=RelationshipState)
    robot_affect: RobotAffectState = Field(default_factory=RobotAffectState)
    pending_actions: tuple[ActionState, ...] = ()
    last_action: ActionState | None = None
    health: dict[str, HealthState] = Field(default_factory=dict)
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    staleness: dict[str, Literal["missing", "fresh", "expired", "unavailable"]] = Field(default_factory=dict)


class SocialContext(StateModel):
    """携带紧凑观测及来源引用；完整输入另作为用户消息传递。"""
    session_id: str
    user_id: str | None
    version: int
    data: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_event_ids: tuple[str, ...] = ()
    missing_modalities: tuple[str, ...] = ()
    truncated: bool = False


EventType = Literal["turn.updated", "utterance.committed", "intent.updated", "affect.updated",
                    "visual.updated", "dialogue.updated", "relationship.updated", "robot_affect.confirmed",
                    "action.updated", "health.updated", "clock.tick", "memory.cleared"]


class SocialEvent(StateModel):
    """接收服务端校验后的事件；内部接口不直接开放给 WebSocket 客户端。"""
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(min_length=1, max_length=160)
    type: EventType
    session_id: str
    user_id: str | None = None
    connection_epoch: int = Field(ge=0)
    source: str = Field(min_length=1, max_length=80)
    seq: int = Field(ge=0)
    captured_at: AwareDatetime
    received_at: AwareDatetime
    turn_id: str | None = None
    generation_id: str | None = None
    payload: (InteractionState | CommittedUtterance | IntentEstimate | UserAffectState | VisualState |
              DialogueState | RelationshipState | RobotAffectState | ActionState | HealthState | None) = None

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        """事件类型必须匹配明确的载荷，阻止 partial 或任意字典进入状态。"""
        expected = {"turn.updated": InteractionState, "utterance.committed": CommittedUtterance,
                    "intent.updated": IntentEstimate, "affect.updated": UserAffectState,
                    "visual.updated": VisualState, "dialogue.updated": DialogueState,
                    "relationship.updated": RelationshipState, "robot_affect.confirmed": RobotAffectState,
                    "action.updated": ActionState, "health.updated": HealthState,
                    "clock.tick": type(None), "memory.cleared": type(None)}
        if not isinstance(self.payload, expected[self.type]):
            raise ValueError("事件类型与载荷不匹配")
        return self


class StateDelta(StateModel):
    """报告归约是否生效及忽略原因，正常乱序不抛出异常。"""
    session_id: str
    version: int
    accepted: bool
    changed_domains: tuple[str, ...] = ()
    reason: str | None = None
