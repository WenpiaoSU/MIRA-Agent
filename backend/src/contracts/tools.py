"""作用：定义工具目录、调用、结果与会话澄清契约。

当前功能：
- 保存双向 JSON Schema，并禁止校验时加载外部引用。
- 约束一次只读调用的截止时间、验证结果和安全错误。
- 保存每个会话最多一个待补充的工具请求。
"""

from datetime import datetime
from typing import Literal, Self

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ToolSpec(BaseModel):
    """描述服务端已经允许调用的工具；执行器和权限检查由工具模块负责。"""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] = Field(default_factory=lambda: {"type": "object"})

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        """校验对象参数 Schema，并拒绝需要远程加载的引用。"""
        for schema in (self.input_schema, self.output_schema):
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError("工具 Schema 无效") from exc
            if schema.get("type") != "object":
                raise ValueError("工具输入输出必须是对象")
            self.validate_local_refs(schema)
        return self

    @staticmethod
    def validate_local_refs(value: JsonValue) -> None:
        """只允许目录内部引用，校验过程不访问外部网络。"""
        if isinstance(value, dict):
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                reference = value.get(keyword)
                if isinstance(reference, str) and not reference.startswith("#"):
                    raise ValueError("工具 Schema 不允许外部引用")
            for child in value.values():
                ToolSpec.validate_local_refs(child)
        elif isinstance(value, list):
            for child in value:
                ToolSpec.validate_local_refs(child)


class ToolCallRequest(BaseModel):
    """绑定逻辑调用及 UTC 绝对截止时间；首版仅执行只读工具。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    deadline_at: datetime

    @model_validator(mode="after")
    def require_timezone(self) -> Self:
        """拒绝无时区截止时间，避免服务端产生隐式时区解释。"""
        if self.deadline_at.tzinfo is None or self.deadline_at.utcoffset() is None:
            raise ValueError("工具截止时间必须包含时区")
        return self


class ToolError(BaseModel):
    """提供安全错误与可选澄清字段，不暴露 HTTP 内容和密钥。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    missing_slots: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """仅成功且输出通过 Schema 校验时允许 verified=true。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    data: dict[str, JsonValue] | None = None
    verified: bool = False
    error: ToolError | None = None
    elapsed_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """禁止失败携带可展示事实，或成功缺少已验证的数据。"""
        if self.status == "ok":
            if not self.verified or self.data is None or self.error is not None:
                raise ValueError("成功结果必须含已验证数据且无错误")
        elif self.verified or self.data is not None or self.error is None:
            raise ValueError("失败结果必须只有错误且未经验证")
        return self


class PendingClarification(BaseModel):
    """保存原请求和已有参数；补充、替换任务或取消时由网关更新。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    original_text: str = Field(min_length=1, max_length=4000)
    tool_name: str
    arguments: dict[str, JsonValue]
    missing_slots: list[str]
    question: str
