"""作用：定义一次性提醒的持久化记录和展示回执。

当前功能：
- 分离提醒业务状态与客户端展示事实。
- 约束提醒内容、时间、用户归属和有界交付凭据。
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ReminderRecord(BaseModel):
    """提供可安全交给工具和界面的提醒记录，不包含用户凭据。"""
    model_config = ConfigDict(extra="forbid")
    reminder_id: str
    title: str = Field(min_length=1, max_length=300)
    due_at: str
    timezone: str
    status: Literal["pending", "delivered", "cancelled"]
    created_at: str


class ReminderAckPayload(BaseModel):
    """确认当前连接展示了指定投递，不能替其他用户或其他租约确认。"""
    model_config = ConfigDict(extra="forbid")
    reminder_id: str = Field(min_length=1, max_length=80)
    delivery_token: str = Field(min_length=1, max_length=80)
