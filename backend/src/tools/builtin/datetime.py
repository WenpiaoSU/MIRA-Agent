"""作用：提供本地时间转换与日期偏移，支持前序查询结果到后续参数的转换。

当前功能：
- 将带时区的 ISO 时间转换为明确的 IANA 时区。
- 按目标时区的当地日期增加或减少整天，返回日期及 ISO 时间。
- 拒绝无时区输入和无效时区，不访问网络或修改外部状态。
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, JsonValue
from src.contracts.tools import ToolSpec
from src.tools.executor import ToolFailure


class ConvertedTime(BaseModel):
    """保存转换后的本地时间、日期和明确时区。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    iso_time: str
    date: str
    time: str
    timezone: str
    utc_offset: str


DATETIME_SPEC = ToolSpec(
    name="convert_datetime",
    description="转换带时区的 ISO 时间，按目标时区当地日历增加 days_offset 天，默认 0。"
                "通常将 get_current_time 的 iso_time 引用到此工具，输出 date 可作为 get_weather 的 date。"
                "明天可表达为 days_offset=1，并保留用户原文证据。",
    input_schema={"type": "object", "properties": {
        "iso_time": {"type": "string", "minLength": 1, "maxLength": 100},
        "timezone": {"type": "string", "minLength": 1, "maxLength": 100},
        "days_offset": {"type": "integer", "minimum": -366, "maximum": 366}},
        "required": ["iso_time", "timezone"], "additionalProperties": False},
    output_schema=ConvertedTime.model_json_schema(),
)


async def convert_datetime(arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """完成确定性的时区与当地日期转换；歧义参数返回安全澄清要求。"""
    try:
        value = datetime.fromisoformat(arguments["iso_time"])
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError
    except ValueError:
        raise ToolFailure("invalid_time", "请提供带时区的 ISO 时间，例如 2026-10-13T10:00:00+08:00。",
                          ["iso_time"]) from None
    try:
        zone = ZoneInfo(arguments["timezone"])
    except (ZoneInfoNotFoundError, ValueError):
        raise ToolFailure("invalid_timezone", "请提供有效的 IANA 时区，例如 Asia/Shanghai 或 Europe/London。",
                          ["timezone"]) from None
    converted = value.astimezone(zone) + timedelta(days=arguments.get("days_offset", 0))
    return ConvertedTime(iso_time=converted.isoformat(timespec="seconds"), date=converted.date().isoformat(),
                         time=converted.strftime("%H:%M:%S"), timezone=arguments["timezone"],
                         utc_offset=converted.strftime("%z")).model_dump(mode="json")
