"""作用：确定性解释一次性提醒时间。

当前功能：
- 以服务端收到请求的时刻计算秒、分钟、小时、天后的相对时间。
- 解释今天、明天、后天和完整日期的中文时刻及带偏移 ISO 时间。
- 对上午下午歧义、过去时间、重复规则和夏令时歧义提出澄清。
"""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from src.tools.executor import ToolFailure


def number(value: str) -> int:
    """转换常用中文整数与阿拉伯数字，限定在基础时间表达范围。"""
    if value.isdecimal():
        return int(value)
    digits = dict(zip("零一二三四五六七八九两", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2]))
    total, current = 0, 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in "十百":
            total += (current or 1) * (10 if char == "十" else 100)
            current = 0
        else:
            raise ValueError("不支持的数字")
    return total + current


def time_question(message: str) -> ToolFailure:
    """为时间消歧生成统一的可跨轮补充错误。"""
    return ToolFailure("clarification_required", message, ["due_at"])


def local_datetime(naive: datetime, zone: ZoneInfo) -> datetime:
    """检查当地时刻的存在性和唯一性，再附加明确的 IANA 时区。"""
    due = naive.replace(tzinfo=zone)
    roundtrip = due.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if roundtrip != naive or due.utcoffset() != naive.replace(tzinfo=zone, fold=1).utcoffset():
        raise time_question("该当地时刻受夏令时影响，请提供带 UTC 偏移的 ISO 时间。")
    return due


def resolve_due_at(value: str, zone_name: str, reference: datetime) -> datetime:
    """解释受支持的完整时间表达，不让模型决定当前时刻或隐式跨天。"""
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ToolFailure("invalid_timezone", "请提供有效的 IANA 时区，例如 Asia/Shanghai。", ["timezone"]) from None
    text = value.strip().replace(" ", "")
    if re.search(r"每天|每周|每月|每年|工作日|循环|重复", text):
        raise time_question("目前支持一次性提醒，请提供某一天的具体提醒时间。")
    try:
        due = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        due = None
    if due is not None:
        if due.tzinfo is None:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?", value.strip()):
                raise time_question("请说明完整日期和具体时刻，例如明天下午三点。")
            due = local_datetime(due, zone)
    else:
        quantity = r"[0-9零一二三四五六七八九十百两]+"
        relative = re.fullmatch(rf"({quantity}|半)(秒|秒钟|分钟|分|小时|天)(?:之)?后", text)
        if relative:
            amount = .5 if relative[1] == "半" else number(relative[1])
            seconds = amount * {"秒": 1, "秒钟": 1, "分": 60, "分钟": 60, "小时": 3600, "天": 86400}[relative[2]]
            if not 0 < seconds <= 366 * 86400:
                raise time_question("请提供大于零、且在一年内的提醒时间。")
            due = reference + timedelta(seconds=seconds)
        else:
            match = re.fullmatch(rf"(今天|明天|明早|后天|\d{{4}}-\d{{2}}-\d{{2}}|\d{{4}}年\d{{1,2}}月\d{{1,2}}日)(凌晨|早上|上午|中午|下午|晚上)?({quantity})(?:点|:|：)({quantity}|半)?(?:分)?", text)
            if not match:
                raise time_question("请说明具体提醒时间，例如十分钟后、明早八点、明天下午三点，或完整日期和时刻。")
            day, period, hour, minute = match.groups()
            now = reference.astimezone(zone)
            if day in {"今天", "明天", "明早", "后天"}:
                date = (now + timedelta(days={"今天": 0, "明天": 1, "明早": 1, "后天": 2}[day])).date()
            else:
                try:
                    date = datetime.strptime(day, "%Y年%m月%d日" if "年" in day else "%Y-%m-%d").date()
                except ValueError:
                    raise time_question("日期无效，请重新说明具体提醒时间。") from None
            hour = number(hour)
            minute = 30 if minute == "半" else number(minute) if minute else 0
            period = period or ("早上" if day == "明早" else None)
            clock_24h = bool(re.search(r"\d{1,2}[:：]\d{2}$", text))
            if period is None and hour <= 12 and not clock_24h:
                raise time_question("这个时刻是上午还是下午？请补充完整时间，例如明天下午三点。")
            if period and not 1 <= hour <= 12 and not (period == "凌晨" and hour == 0):
                raise time_question("时段和小时不匹配，请重新说明具体提醒时间。")
            if hour == 12 and period in {"早上", "上午", "晚上"}:
                raise time_question("十二点可能指中午或午夜，请使用 24 小时制或带偏移的完整时间。")
            if period in {"下午", "晚上"}:
                hour = hour % 12 + 12
            elif period in {"早上", "上午", "凌晨"}:
                hour %= 12
            elif period == "中午" and hour < 11:
                raise time_question("中午时间不明确，请使用 24 小时制或说明上午/下午。")
            try:
                naive = datetime(date.year, date.month, date.day, hour, minute)
            except ValueError:
                raise time_question("日期或时刻无效，请重新说明提醒时间。") from None
            due = local_datetime(naive, zone)
    if due <= reference:
        raise time_question("提醒时间必须晚于本次请求时间，请提供将来的具体时间。")
    if due - reference > timedelta(days=366):
        raise time_question("请提供一年内的一次性提醒时间。")
    return due.astimezone(timezone.utc)
