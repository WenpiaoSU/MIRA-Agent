"""作用：保存后端实际开放的工具及其处理函数。

当前功能：
- 注册双向 Schema 和异步函数，拒绝名称重复。
- 返回目录副本，防止分类或请求修改真实注册信息。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import JsonValue
from src.contracts.tools import ToolSpec


ToolHandler = Callable[[dict[str, JsonValue]], Awaitable[dict[str, JsonValue]]]


@dataclass(frozen=True)
class RegisteredTool:
    """绑定一个可信工具规范与服务端处理函数。"""
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """由应用启动时装配的只读工具白名单，不接受客户端动态注册。"""

    def __init__(self) -> None:
        """创建进程内目录；不保存会话参数或调用状态。"""
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """复制工具规范并拒绝重复注册，避免目录和实际实现错配。"""
        if spec.name in self._tools:
            raise ValueError("工具名称重复")
        self._tools[spec.name] = RegisteredTool(spec.model_copy(deep=True), handler)

    def get(self, name: str) -> RegisteredTool | None:
        """按工具名读取内部绑定，仅供执行器和 Agent 使用。"""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """只公布实际注册的能力，返回独立目录快照。"""
        return [tool.spec.model_copy(deep=True) for tool in self._tools.values()]
