"""作用：创建并导出 FastAPI 应用。

当前功能：
- 读取语言模型配置并初始化会话注册表与 Agent。
- 挂载健康检查、会话管理和实时对话接口。
- 注入通过 API Key 调用的 TTS 适配器。
- 注入本地 SileroVAD 与 ASR API 组成的语音服务。
- 装配有界意图识别、时间工具及可关闭的单一来源天气工具。
"""

from fastapi import FastAPI

from src.agent.graph import AgentGraph
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


def create_app() -> FastAPI:
    """组装后端依赖和路由，返回可由 ASGI 服务器运行的应用。"""
    app = FastAPI(title="MIRA Agent", version="0.1.0")
    settings = Settings()
    app.state.settings = settings
    app.state.sessions = SessionRegistry()
    provider = ChatCompletionsProvider(settings)
    registry = ToolRegistry()
    registry.register(TIME_SPEC, CurrentTimeTool(settings.default_timezone))
    if settings.weather_enabled:
        registry.register(WEATHER_SPEC, WeatherTool())
    app.state.tool_registry = registry
    app.state.tool_executor = ToolExecutor(registry, settings.tool_timeout_ms)
    app.state.agent = AgentGraph(provider, app.state.tool_executor)
    app.state.intent_service = IntentService(provider, settings.intent_timeout_ms)
    app.state.tts = SpeechAPIProvider(settings)
    app.state.speech_service = SpeechService(settings)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(realtime.router)
    return app


app = create_app()
