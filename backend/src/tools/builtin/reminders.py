"""作用：提供创建、查看与取消一次性提醒的内置工具。

当前功能：
- 从执行器注入的身份和轮次上下文操作 SQLite，不接受模型传入用户标识。
- 确定性计算提醒时间，歧义交给现有跨轮参数澄清。
- 限制写操作必须来自当前明确请求，返回数据库已提交事实。
"""

import re
from src.contracts.reminders import ReminderRecord
from src.contracts.tools import ToolContext, ToolSpec
from src.repositories.reminders import ReminderRepository
from src.tools.builtin.reminder_time import resolve_due_at
from src.tools.executor import ToolFailure


CREATE_REMINDER_SPEC = ToolSpec(
    name="create_reminder", side_effect="write",
    description="创建用户明确要求的一次性提醒。title 保留事项原文；due_at 保留时间原文（十分钟后、明早八点、明天下午三点、完整日期时刻或带偏移ISO），禁止模型计算日期。timezone 可省略使用配置默认时区。不支持重复提醒。",
    input_schema={"type": "object", "additionalProperties": False, "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 300},
        "due_at": {"type": "string", "minLength": 1, "maxLength": 120},
        "timezone": {"type": "string", "minLength": 1, "maxLength": 80}}, "required": ["title", "due_at"]},
    output_schema=ReminderRecord.model_json_schema())
LIST_REMINDERS_SPEC = ToolSpec(
    name="list_reminders", description="查看当前用户全部尚未展示的提醒，包括到期但离线未送达的事项。",
    input_schema={"type": "object", "additionalProperties": False, "properties": {}},
    output_schema={"type": "object", "properties": {"reminders": {"type": "array", "maxItems": 100, "items": ReminderRecord.model_json_schema()}}, "required": ["reminders"], "additionalProperties": False})
CANCEL_REMINDER_SPEC = ToolSpec(
    name="cancel_reminder", side_effect="write", description="取消明确指定的一条提醒。target 使用列表中的完整 reminder_id、唯一事项标题，或 latest（仅当前会话刚才创建的提醒）。不能取消所有提醒，不能用来停止机器人说话。",
    input_schema={"type": "object", "additionalProperties": False, "properties": {"target": {"type": "string", "minLength": 1, "maxLength": 300}}, "required": ["target"]},
    output_schema=ReminderRecord.model_json_schema())


def authorize_reminder(tool_name: str, text: str) -> None:
    """写权限只来自本次请求原文，避免历史或工具结果诱导额外业务写入。"""
    if tool_name not in {"create_reminder", "cancel_reminder"}:
        return
    if "提醒" not in text:
        raise ToolFailure("explicit_request_required", "请明确说明要创建或取消的提醒。")
    cancel = bool(re.search(r"(?:取消|删除|撤销)[^，。！？\n]{0,80}提醒|提醒(?:取消|删除|撤销)", text))
    create = bool(re.search(r"提醒我|(?:创建|设置|添加|新建)[^，。！？\n]{0,30}提醒", text))
    if re.search(r"不要(?:再)?提醒|别提醒|不(?:要|用)(?:创建|设置)|不(?:是|想).*提醒", text):
        raise ToolFailure("explicit_request_required", "提醒请求存在否定或歧义，请明确说明希望进行的操作。")
    if (tool_name == "cancel_reminder" and not cancel) or (tool_name == "create_reminder" and cancel and not create):
        raise ToolFailure("explicit_request_required", "提醒操作与本次请求不一致，请明确要创建还是取消。")
    if tool_name == "create_reminder" and re.search(r"每天|每周|每月|每年|工作日|循环|重复", text.split("提醒", 1)[0]):
        raise ToolFailure("unsupported_recurrence", "目前只支持一次性提醒，请指定某一天的具体时间。", ["due_at"])


class ReminderTools:
    """以同一存储实现三个固定工具，沿用现有 Registry 和 Executor。"""

    def __init__(self, repository: ReminderRepository, default_timezone: str):
        """注入持久化存储和明确的默认时区。"""
        self.repository = repository
        self.default_timezone = default_timezone

    async def create(self, arguments: dict, context: ToolContext) -> dict:
        """校验用户原文和时间后一次提交；取消输出不撤销已提交提醒。"""
        authorize_reminder("create_reminder", context.request_text)
        for name in ("title", "due_at", "timezone"):
            if name in arguments and name not in context.bound_arguments and arguments[name] not in context.request_text:
                raise ToolFailure("clarification_required", "请用原文明确提醒事项、具体时间和需要的时区。", [name])
        title = arguments["title"].strip()
        if not title:
            raise ToolFailure("clarification_required", "请说明要提醒的事项。", ["title"])
        zone = arguments.get("timezone", self.default_timezone)
        due = resolve_due_at(arguments["due_at"], zone, context.reference_at)
        return await self.repository.create(context, title, due, zone, arguments)

    async def list(self, arguments: dict, context: ToolContext) -> dict:
        """按可信身份读取列表，模型无权指定其他用户。"""
        return await self.repository.list_pending(context.user_id)

    async def cancel(self, arguments: dict, context: ToolContext) -> dict:
        """按标识或唯一标题取消；latest 只在用户明确指代刚才时接受。"""
        authorize_reminder("cancel_reminder", context.request_text)
        target = arguments["target"]
        if target == "latest" and not re.search(r"刚才|刚刚|上一个|上一条|最近", context.request_text):
            raise ToolFailure("clarification_required", "请提供要取消的提醒编号或唯一事项标题。", ["target"])
        if target != "latest" and "target" not in context.bound_arguments and target not in context.request_text:
            raise ToolFailure("clarification_required", "请明确提供要取消的提醒编号或唯一事项标题。", ["target"])
        return await self.repository.cancel(context, target)


def register_reminder_tools(registry, repository: ReminderRepository, default_timezone: str) -> None:
    """启动时注册真实可用的三个工具，数据库不可用时不公布能力。"""
    tools = ReminderTools(repository, default_timezone)
    for spec, handler in ((CREATE_REMINDER_SPEC, tools.create), (LIST_REMINDERS_SPEC, tools.list), (CANCEL_REMINDER_SPEC, tools.cancel)):
        registry.register(spec, handler, needs_context=True)
