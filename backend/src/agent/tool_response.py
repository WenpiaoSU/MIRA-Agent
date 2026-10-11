"""作用：将已验证工具事实转换为确定性的回复。

当前功能：
- 格式化时间、当前天气和逐日预报，保留时区、时间与来源。
- 失败仅展示安全错误；重复或继续只复述原结果，不重新查询。
"""

from datetime import datetime

from src.contracts.tools import ToolResult
from src.tools.builtin.time import TimeData
from src.tools.builtin.weather import WeatherData


def render_tool_result(result: ToolResult) -> str:
    """展示执行器验证过的事实，不将工具结果交给模型自由补充。"""
    if result.status != "ok" or not result.verified:
        return result.error.message if result.error else "查询尚未完成。"
    if result.tool_name == "get_current_time":
        data = TimeData.model_validate(result.data)
        value = datetime.fromisoformat(data.iso_time)
        return f"{data.timezone} 当前时间是 {value:%Y年%m月%d日 %H:%M:%S}（UTC{data.utc_offset}）。"
    if result.tool_name == "get_weather":
        data = WeatherData.model_validate(result.data)
        if data.kind == "current":
            value = datetime.fromisoformat(data.valid_at)
            return (f"{data.location} 当前{data.condition}，气温 {data.temperature:g}{data.units}。"
                    f"数据时间：{value:%Y-%m-%d %H:%M}（{data.timezone}）；来源：{data.source} 模拟天气。")
        forecast = data.forecast
        return (f"{data.location} {forecast.date} 的预报为{forecast.condition}，"
                f"气温 {forecast.temperature_min:g}～{forecast.temperature_max:g}{data.units}。"
                f"来源：{data.source}；查询时间：{datetime.fromisoformat(data.fetched_at):%Y-%m-%d %H:%M} UTC。")
    return "查询已完成，但当前没有这项工具的结果展示模板。"
