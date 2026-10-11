"""作用：执行一次有界只读工具调用并验证返回值。

当前功能：
- 按真实白名单检查工具及输入输出 JSON Schema。
- 统一处理截止时间、服务异常和安全参数澄清，不自动重试。
- 将任务取消向上传递，不把取消或非法结果当作查询成功。
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from time import monotonic

import httpx
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError
from pydantic import ValidationError
from referencing.exceptions import Unresolvable

from src.contracts.tools import ToolCallRequest, ToolError, ToolResult
from src.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


class ToolFailure(Exception):
    """由内置工具提出已脱敏的业务错误或参数补充要求。"""

    def __init__(self, code: str, message: str, missing_slots: list[str] | None = None):
        """保存可直接展示的说明，禁止传入供应商原始异常正文。"""
        super().__init__(message)
        self.error = ToolError(code=code, message=message, missing_slots=missing_slots or [])


class ToolExecutor:
    """一次请求最多调用一个已注册处理函数，没有重试和写入副作用。"""

    def __init__(self, registry: ToolRegistry, timeout_ms: int = 5000):
        """注入真实目录与执行总预算，网络步骤共用同一次截止时间。"""
        if timeout_ms <= 0:
            raise ValueError("工具超时必须为正数")
        self.registry = registry
        self.timeout_ms = timeout_ms

    async def execute(self, request: ToolCallRequest) -> ToolResult:
        """验证后执行，并只为完整、有限、符合输出 Schema 的数据标记成功。"""
        started = monotonic()
        task = asyncio.current_task()
        cancellations = task.cancelling() if task else 0
        error = None
        tool = self.registry.get(request.tool_name)
        budget = min(self.timeout_ms / 1000,
                     (request.deadline_at - datetime.now(timezone.utc)).total_seconds())
        try:
            if tool is None:
                raise ToolFailure("unavailable", "当前没有开放这项工具能力。")
            if budget <= 0:
                raise TimeoutError
            if not Draft202012Validator(tool.spec.input_schema).is_valid(request.arguments):
                raise ToolFailure("invalid_arguments", "工具参数无效，请重新说明查询条件。")
            async with asyncio.timeout(budget):
                data = await tool.handler(dict(request.arguments))
                if task and task.cancelling() > cancellations:
                    raise asyncio.CancelledError
                if monotonic() - started >= budget:
                    raise TimeoutError
                # JSON Schema 默认接受 NaN；额外确保结果能以标准 JSON 安全发送。
                json.dumps(data, allow_nan=False)
                Draft202012Validator(tool.spec.output_schema).validate(data)
                if not isinstance(data, dict):
                    raise ValueError("工具输出必须为对象")
                return ToolResult(call_id=request.call_id, tool_name=request.tool_name, status="ok",
                                  data=data, verified=True, elapsed_ms=round((monotonic() - started) * 1000))
        except asyncio.CancelledError:
            raise
        except ToolFailure as exc:
            error = exc.error
        except (TimeoutError, httpx.TimeoutException):
            error = ToolError(code="timeout", message="查询超时，尚未获取结果，请稍后重试。")
        except httpx.HTTPError:
            error = ToolError(code="service_unavailable", message="查询服务暂时不可用，尚未获取结果，请稍后重试。")
        except (ValueError, TypeError, KeyError, ValidationError, SchemaValidationError, Unresolvable):
            error = ToolError(code="invalid_result", message="查询结果不完整或格式异常，暂时无法提供可靠结果。")
        except Exception:
            # 不输出异常正文，避免供应商响应或参数进入轨迹日志。
            logger.error("工具执行失败：%s", request.tool_name)
            error = ToolError(code="execution_error", message="查询未完成，请稍后重试。")
        return ToolResult(call_id=request.call_id, tool_name=request.tool_name, status="error", error=error,
                          elapsed_ms=round((monotonic() - started) * 1000))
