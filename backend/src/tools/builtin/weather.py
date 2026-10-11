"""作用：通过固定 Open-Meteo 来源查询天气，不提供供应商切换。

当前功能：
- 先解析地点；无结果或重名时要求补充城市、省份或国家。
- 查询当前模拟天气，或当地今天至未来第六天的逐日预报。
- 校验供应商字段、单位、天气代码与时间，返回带来源的结构化事实。
- HTTP 连接使用异步上下文回收，整次调用预算由工具执行器控制。
"""

import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from src.contracts.tools import ToolSpec
from src.tools.executor import ToolFailure


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴", 45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "大毛毛雨", 56: "小冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "小冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒", 80: "小阵雨", 81: "中阵雨",
    82: "强阵雨", 85: "小阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


class GeoLocation(BaseModel):
    """校验供应商地点坐标、时区和可供用户消歧的名称。"""
    model_config = ConfigDict(strict=True)
    name: str = Field(min_length=1, max_length=200)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    timezone: str
    admin1: str = ""
    country: str = ""

    def label(self) -> str:
        """组合名称、一级行政区和国家，显示供应商解析出的实际地点。"""
        return "，".join(dict.fromkeys(part for part in (self.name, self.admin1, self.country) if part))


class CurrentWeather(BaseModel):
    """约束当前天气的 Unix 时间、摄氏温度和 WMO 天气代码。"""
    model_config = ConfigDict(strict=True)
    time: int
    temperature_2m: float = Field(ge=-100, le=70, allow_inf_nan=False)
    weather_code: int


class DailyWeather(BaseModel):
    """校验逐日预报的类型；数组等长和目标日期由处理函数检查。"""
    model_config = ConfigDict(strict=True)
    time: list[int]
    weather_code: list[int]
    temperature_2m_max: list[float] = Field(min_length=1)
    temperature_2m_min: list[float] = Field(min_length=1)


class WeatherForecast(BaseModel):
    """提供单日预报和有明确单位的高低温。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    date: str
    condition: str
    temperature_min: float = Field(ge=-100, le=70, allow_inf_nan=False)
    temperature_max: float = Field(ge=-100, le=70, allow_inf_nan=False)


class WeatherData(BaseModel):
    """区分当前天气与逐日预报，标明数据有效时间、查询时间和来源。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    location: str
    timezone: str
    kind: Literal["current", "forecast"]
    valid_at: str
    fetched_at: str
    temperature: float | None = Field(ge=-100, le=70, allow_inf_nan=False)
    condition: str | None
    forecast: WeatherForecast | None
    units: Literal["°C"] = "°C"
    source: Literal["Open-Meteo"] = "Open-Meteo"


WEATHER_SPEC = ToolSpec(
    name="get_weather",
    description="查询天气，唯一来源 Open-Meteo。location 必填，保留城市和用户提供的省份/国家；"
                "如有明确限定可整理为“杭州, 浙江”。date 省略或现在/当前表示当前天气；"
                "支持今天/明天/后天及 YYYY-MM-DD，限当地今天起七天。不根据历史猜地点或日期。",
    input_schema={"type": "object", "properties": {
        "location": {"type": "string", "minLength": 1, "maxLength": 200},
        "date": {"type": "string", "minLength": 1, "maxLength": 40}},
        "required": ["location"], "additionalProperties": False},
    output_schema=WeatherData.model_json_schema(),
)


def forecast_date(value: str | None, today: date) -> date | None:
    """按查询地点的当地日期解释有限日期词；模糊或超范围日期必须澄清。"""
    if value is None or value in {"现在", "当前"}:
        return None
    relative = {"今天": 0, "明天": 1, "后天": 2}
    if value in relative:
        target = today + timedelta(days=relative[value])
    else:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError
            target = date.fromisoformat(value)
        except ValueError:
            raise ToolFailure("invalid_date", "请说明具体日期：今天、明天、后天或 YYYY-MM-DD。", ["date"]) from None
    if not today <= target <= today + timedelta(days=6):
        raise ToolFailure("unsupported_date", "目前只能查询当地今天起七天内的天气，请选择范围内的日期。", ["date"])
    return target


def weather_condition(code: int) -> str:
    """仅接受官方公布的 WMO 代码，不将未知代码描述为晴天。"""
    if code not in WEATHER_CODES:
        raise ValueError("未知天气代码")
    return WEATHER_CODES[code]


class WeatherTool:
    """固定使用 Open-Meteo 两个公开端点，不接收用户 URL，不自动重试。"""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        """允许测试注入 HTTP 传输替身；生产使用默认网络传输。"""
        self.transport = transport

    async def resolve_location(self, client: httpx.AsyncClient, name: str) -> GeoLocation:
        """通过地理检索消歧，重名时返回候选说明而不选择第一项。"""
        query = name.strip().replace("，", ",")
        response = await client.get(GEOCODING_URL, params={"name": query, "count": 10, "language": "zh"})
        response.raise_for_status()
        payload = response.json()
        results = [GeoLocation.model_validate(item) for item in payload.get("results", [])]
        if not results:
            raise ToolFailure("location_not_found", "未找到这个地点，请提供城市，并补充省份或国家。", ["location"])
        # 排除前缀检索出现的机场等不同名称；不因人口大小猜测同名地点。
        base_name = query.split(",", 1)[0].strip()
        exact = [item for item in results if item.name.casefold() == base_name.casefold()]
        candidates = exact or results
        unique = {(item.latitude, item.longitude): item for item in candidates}
        if len(unique) != 1:
            labels = "；".join(item.label() for item in list(unique.values())[:3])
            raise ToolFailure("location_ambiguous", f"找到多个同名地点：{labels}。请补充城市所属省份或国家。", ["location"])
        return next(iter(unique.values()))

    async def __call__(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """在同一可取消 HTTP 上下文中完成地点检索、天气查询和领域校验。"""
        async with httpx.AsyncClient(timeout=5, transport=self.transport) as client:
            location = await self.resolve_location(client, arguments["location"])
            zone = ZoneInfo(location.timezone)
            fetched_at = datetime.now(timezone.utc)
            target = forecast_date(arguments.get("date"), fetched_at.astimezone(zone).date())
            params = {"latitude": location.latitude, "longitude": location.longitude,
                      "timezone": location.timezone, "timeformat": "unixtime", "temperature_unit": "celsius"}
            if target is None:
                params["current"] = "temperature_2m,weather_code"
            else:
                params.update(daily="weather_code,temperature_2m_max,temperature_2m_min", forecast_days=7)
            response = await client.get(WEATHER_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            fetched_at = datetime.now(timezone.utc)
            if payload["timezone"] != location.timezone:
                raise ValueError("供应商时区不一致")
            if target is None:
                current = CurrentWeather.model_validate(payload["current"])
                if payload["current_units"]["temperature_2m"] != "°C":
                    raise ValueError("温度单位不一致")
                valid_at = datetime.fromtimestamp(current.time, zone)
                if abs((fetched_at - valid_at).total_seconds()) > 3600:
                    raise ValueError("当前天气数据已过期")
                data = WeatherData(location=location.label(), timezone=location.timezone, kind="current",
                                   valid_at=valid_at.isoformat(), fetched_at=fetched_at.isoformat(),
                                   temperature=current.temperature_2m, condition=weather_condition(current.weather_code),
                                   forecast=None)
            else:
                daily = DailyWeather.model_validate(payload["daily"])
                if not len(daily.time) == len(daily.weather_code) == len(daily.temperature_2m_max) == len(daily.temperature_2m_min):
                    raise ValueError("预报字段长度不一致")
                if any(payload["daily_units"][key] != "°C" for key in ("temperature_2m_max", "temperature_2m_min")):
                    raise ValueError("预报温度单位不一致")
                dates = [datetime.fromtimestamp(value, zone).date() for value in daily.time]
                if target not in dates:
                    raise ValueError("供应商未提供目标日期")
                index = dates.index(target)
                forecast = WeatherForecast(date=target.isoformat(), condition=weather_condition(daily.weather_code[index]),
                                           temperature_min=daily.temperature_2m_min[index],
                                           temperature_max=daily.temperature_2m_max[index])
                if forecast.temperature_min > forecast.temperature_max:
                    raise ValueError("高低温顺序错误")
                data = WeatherData(location=location.label(), timezone=location.timezone, kind="forecast",
                                   valid_at=datetime.fromtimestamp(daily.time[index], zone).isoformat(),
                                   fetched_at=fetched_at.isoformat(), temperature=None, condition=None, forecast=forecast)
        return data.model_dump(mode="json")
