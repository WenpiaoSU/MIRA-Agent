"""作用：创建并导出 FastAPI 应用。

当前功能：
- 读取语言模型配置并初始化会话注册表与 Agent。
- 挂载健康检查、会话管理和实时对话接口。
- 注入通过 API Key 调用的 TTS 适配器。
- 注入本地 SileroVAD 与 ASR API 组成的语音服务。
- 装配有界意图识别、时间工具及可关闭的单一来源天气工具。
- 注入有界任务规划与时间转换工具，支持不同工具之间的字段依赖。
- 装配 Checkpointer 短期记忆与 SQLite 长期记忆，启动故障保留普通对话能力。
"""

from fastapi import FastAPI
import logging
import sqlite3

from src.agent.graph import AgentGraph
from src.agent.planner import TaskPlanner
from src.api import health, realtime, sessions
from src.core.config import Settings
from src.gateway.sessions import SessionRegistry
from src.intent.service import IntentService
from src.providers.llm import ChatCompletionsProvider
from src.providers.tts import SpeechAPIProvider
from src.speech.service import SpeechService
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.builtin.time import CurrentTimeTool, TIME_SPEC
from src.tools.builtin.weather import WeatherTool, WEATHER_SPEC
from src.tools.builtin.datetime import convert_datetime, DATETIME_SPEC
from src.memory.working import WorkingMemory
from src.memory.service import MemoryService
from src.repositories.memory import MemoryRepository
from src.api import memories


def create_app() -> FastAPI:
    """组装后端依赖和路由，返回可由 ASGI 服务器运行的应用。"""
    app = FastAPI(title="MIRA Agent", version="0.1.0")
    settings = Settings()
    app.state.settings = settings
    app.state.sessions = SessionRegistry()
    provider = ChatCompletionsProvider(settings)
    registry = ToolRegistry()
    registry.register(TIME_SPEC, CurrentTimeTool(settings.default_timezone))
    registry.register(DATETIME_SPEC, convert_datetime)
    if settings.weather_enabled:
        registry.register(WEATHER_SPEC, WeatherTool())
    app.state.tool_registry = registry
    app.state.tool_executor = ToolExecutor(registry, settings.tool_timeout_ms)
    app.state.task_planner = TaskPlanner(provider, settings.plan_max_steps, settings.plan_timeout_ms)
    working = WorkingMemory(settings.working_memory_ttl_seconds)
    app.state.memory = None
    if settings.memory_enabled:
        try:
            repository = MemoryRepository(settings.memory_db_path)
            app.state.memory = MemoryService(repository, provider, settings.memory_timeout_ms)
            app.state.memory.on_delete = app.state.sessions.forget_memories
        except (sqlite3.Error, OSError):
            logging.getLogger(__name__).warning("长期记忆初始化失败，普通对话仍可使用")
    app.state.sessions.forget_checkpoint = working.forget
    app.state.sessions.invalidate_checkpoint = working.invalidate
    app.state.agent = AgentGraph(provider, app.state.tool_executor, app.state.task_planner, app.state.memory, working)
    app.state.intent_service = IntentService(provider, settings.intent_timeout_ms)
    app.state.tts = SpeechAPIProvider(settings)
    app.state.speech_service = SpeechService(settings)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(memories.router)
    app.include_router(realtime.router)
    return app


app = create_app()
