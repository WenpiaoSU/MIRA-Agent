"""作用：提供本地只读时间查询。

当前功能：
- 使用配置默认时区或用户明确给出的 IANA 时区读取当前时间。
- 返回带 UTC 偏移的 ISO 时间；无效时区要求补充，不静默回退。
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, JsonValue
from src.contracts.tools import ToolSpec
from src.tools.executor import ToolFailure


class TimeData(BaseModel):
    """保存带时区的时间事实和 UTC 偏移。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    iso_time: str
    timezone: str
    utc_offset: str


TIME_SPEC = ToolSpec(
    name="get_current_time",
    description="查询现在的时间。timezone 可省略，使用已配置默认时区；指定时区须保留用户原文，不猜城市时区。",
    input_schema={"type": "object", "properties": {
        "timezone": {"type": "string", "minLength": 1, "maxLength": 100,
                     "description": "IANA 时区，例如 Asia/Shanghai、Europe/London 或 UTC。"}},
        "additionalProperties": False},
    output_schema=TimeData.model_json_schema(),
)


class CurrentTimeTool:
    """默认时区由应用配置确定，不从模型或上一轮请求隐式推断。"""

    def __init__(self, default_timezone: str = "Asia/Shanghai"):
        """启动时校验默认时区，配置错误尽早报告。"""
        ZoneInfo(default_timezone)
        self.default_timezone = default_timezone

    async def __call__(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """读取所选时区时间；不访问网络、不修改会话。"""
        name = arguments.get("timezone", self.default_timezone)
        try:
            zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            raise ToolFailure("invalid_timezone", "请提供有效的 IANA 时区，例如 Asia/Shanghai 或 Europe/London。",
                              ["timezone"]) from None
        current = datetime.now(zone)
        return TimeData(iso_time=current.isoformat(timespec="seconds"), timezone=name,
                        utc_offset=current.strftime("%z")).model_dump(mode="json")
