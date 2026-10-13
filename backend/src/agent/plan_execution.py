"""作用：推进任务步骤、解析前序结果引用和保存计划内澄清。

当前功能：
- 顺序执行一个工具步骤，依赖值仅来自已验证成功结果。
- 保存参数缺项与领域消歧状态，补充后跳过已成功步骤。
- 在失败时停止剩余步骤，以固定模板展示已完成事实。
"""

from datetime import datetime, timedelta, timezone
from copy import deepcopy
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import JsonValue

from src.agent.planner import pointer_parts
from src.agent.tool_response import render_tool_result
from src.contracts.agent import PlanStep, TaskPlan
from src.contracts.perception import IntentEstimate
from src.contracts.tools import PendingClarification, ToolCallRequest
from src.tools.executor import ToolExecutor


def current_step(plan: TaskPlan) -> PlanStep | None:
    """找出首个尚未成功的步骤，已完成步骤不重新执行。"""
    return next((step for step in plan.steps if step.status != "succeeded"), None)


def resolve_arguments(plan: TaskPlan, step: PlanStep) -> dict[str, JsonValue]:
    """从前序成功结果解码 JSON 字段，不执行字符串表达式或代码。"""
    arguments = deepcopy(step.arguments)
    for name, reference in step.bindings.items():
        source = plan.steps[reference.step - 1]
        if source.status != "succeeded" or source.result is None or not source.result.verified:
            raise ValueError("依赖结果尚未验证")
        value = source.result.data
        for part in pointer_parts(reference.path):
            if isinstance(value, dict):
                value = value[part]
            elif isinstance(value, list) and part.isdecimal():
                value = value[int(part)]
            else:
                raise ValueError("结果引用无法解析")
        arguments[name] = deepcopy(value)
    return arguments


def facts_text(plan: TaskPlan) -> str:
    """以确定性模板保留所有已成功步骤的事实及来源。"""
    return "\n".join(f"步骤 {i}：{render_tool_result(step.result)}" for i, step in enumerate(plan.steps, 1)
                     if step.result and step.result.verified and step.status == "succeeded")


class PlanExecution:
    """只推进服务端状态，模型不参与执行顺序与取消判断。"""

    def __init__(self, executor: ToolExecutor):
        """复用已有工具白名单和执行预算。"""
        self.executor = executor

    @staticmethod
    def publish(writer, plan: TaskPlan, pending: PendingClarification | None = None) -> None:
        """发出内部原子快照，网关检查代次后同时更新计划和待补充请求。"""
        writer({"type": "plan.updated", "payload": {"plan": plan.model_dump(mode="json"),
                                                      "pending": pending.model_dump() if pending else None}})

    def supplement(self, plan: TaskPlan, pending: PendingClarification, intent: IntentEstimate) -> None:
        """只修改原等待步骤，用户明确提供的参数替换该参数的旧引用。"""
        step = next((item for item in plan.steps if item.step_id == pending.step_id), None)
        if (plan.status != "waiting_input" or plan.plan_id != pending.plan_id or step is None
                or step.status != "waiting_input" or step.tool_name != intent.tool_intent):
            raise ValueError("不存在对应待补充任务")
        registered = self.executor.registry.get(step.tool_name)
        if registered is None:
            raise ValueError("待补充工具不可用")
        properties = registered.spec.input_schema.get("properties", {})
        for slot in intent.slots:
            if slot.value is not None:
                if slot.name not in properties or not slot.evidence:
                    raise ValueError("补充参数不合法")
                step.arguments[slot.name] = slot.value
                step.argument_evidence[slot.name] = slot.evidence
                step.bindings.pop(slot.name, None)
        if any(error.validator != "required" or error.path for error in
               Draft202012Validator(registered.spec.input_schema).iter_errors(step.arguments)):
            raise ValueError("补充参数结构不合法")
        step.status = "pending"
        plan.status = "running"

    def preflight(self, writer, plan: TaskPlan) -> str | None:
        """执行前找出所有未完成工具步骤的静态缺项，已绑定的依赖参数不误报缺失。"""
        for step in plan.steps[:-1]:
            if step.status == "succeeded":
                continue
            registered = self.executor.registry.get(step.tool_name)
            if registered is None:
                return self.fail(writer, plan, step, "所需工具当前不可用。")
            supplied = set(step.arguments).union(step.bindings)
            missing = [name for name in registered.spec.input_schema.get("required", []) if name not in supplied]
            if missing:
                labels = {"location": "城市及省份或国家", "timezone": "IANA 时区", "iso_time": "带时区的 ISO 时间"}
                question = f"任务步骤需要补充：{'、'.join(labels.get(name, name) for name in missing)}。"
                self.wait(writer, plan, step, missing, question)
                return question
        return None

    def wait(self, writer, plan: TaskPlan, step: PlanStep, missing: list[str], question: str) -> None:
        """保留整个任务和成功结果，结束本轮等待用户补充。"""
        for name in missing:
            step.arguments.pop(name, None)
            step.argument_evidence.pop(name, None)
            step.bindings.pop(name, None)
        step.status = "waiting_input"
        plan.status = "waiting_input"
        pending = PendingClarification(original_text=plan.original_text, tool_name=step.tool_name,
                                       arguments=step.arguments, missing_slots=missing, question=question,
                                       plan_id=plan.plan_id, step_id=step.step_id)
        self.publish(writer, plan, pending)

    def fail(self, writer, plan: TaskPlan, step: PlanStep, message: str) -> str:
        """失败后取消剩余步骤，并明确区分完成事实与未完成目标。"""
        step.status = "failed"
        plan.status = "failed"
        for remaining in plan.steps:
            if remaining.status == "pending":
                remaining.status = "cancelled"
        reply = facts_text(plan)
        plan.response_text = (reply + "\n" if reply else "") + f"任务未全部完成：{message}"
        self.publish(writer, plan)
        return plan.response_text

    async def execute(self, writer, plan: TaskPlan, deadline_at: datetime) -> str | None:
        """执行当前一个工具步骤；缺参或失败返回收尾文本，成功推进下一节点。"""
        step = current_step(plan)
        binding = self.executor.registry.get(step.tool_name)
        if binding is None:
            return self.fail(writer, plan, step, "所需工具当前不可用。")
        try:
            arguments = resolve_arguments(plan, step)
        except (ValueError, KeyError, IndexError, TypeError):
            return self.fail(writer, plan, step, "前序结果无法提供可靠的下一步参数。")
        errors = list(Draft202012Validator(binding.spec.input_schema).iter_errors(arguments))
        if any(error.validator != "required" or error.path for error in errors):
            return self.fail(writer, plan, step, "参数或依赖字段不符合工具规范。")
        missing = [name for name in binding.spec.input_schema.get("required", []) if name not in arguments]
        if missing:
            labels = {"location": "城市及省份或国家", "timezone": "IANA 时区", "iso_time": "带时区的 ISO 时间", "date": "日期"}
            question = f"任务步骤需要补充：{'、'.join(labels.get(name, name) for name in missing)}。"
            self.wait(writer, plan, step, missing, question)
            return question
        step.status = "running"
        plan.status = "running"
        self.publish(writer, plan)
        call = ToolCallRequest(call_id=str(uuid4()), tool_name=step.tool_name, arguments=arguments,
                               deadline_at=min(deadline_at, datetime.now(timezone.utc) +
                                               timedelta(milliseconds=self.executor.timeout_ms)))
        metadata = {"task_plan_id": plan.plan_id, "step_id": step.step_id, "call_id": call.call_id, "tool_name": call.tool_name}
        writer({"type": "tool.started", "payload": metadata})
        result = await self.executor.execute(call)
        step.result = result
        writer({"type": "tool.result", "payload": {"result": result.model_dump(mode="json"), **metadata}})
        if result.error and result.error.missing_slots:
            self.wait(writer, plan, step, result.error.missing_slots, result.error.message)
            return result.error.message
        if not result.verified:
            return self.fail(writer, plan, step, result.error.message)
        step.status = "succeeded"
        self.publish(writer, plan)
        return None
