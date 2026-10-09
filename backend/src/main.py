"""作用：创建并导出 FastAPI 应用。

当前功能：
- 读取语言模型配置并初始化会话注册表与 Agent。
- 挂载健康检查、会话管理和实时对话接口。
- 注入通过 API Key 调用的 TTS 适配器。
- 注入本地 SileroVAD 与 ASR API 组成的语音服务。
"""

from fastapi import FastAPI

from src.agent.graph import AgentGraph
from src.api import health, realtime, sessions
from src.core.config import Settings
from src.gateway.sessions import SessionRegistry
from src.providers.llm import ChatCompletionsProvider
from src.providers.tts import SpeechAPIProvider
from src.speech.service import SpeechService


def create_app() -> FastAPI:
    """组装后端依赖和路由，返回可由 ASGI 服务器运行的应用。"""
    app = FastAPI(title="MIRA Agent", version="0.1.0")
    settings = Settings()
    app.state.settings = settings
    app.state.sessions = SessionRegistry()
    app.state.agent = AgentGraph(ChatCompletionsProvider(settings))
    app.state.tts = SpeechAPIProvider(settings)
    app.state.speech_service = SpeechService(settings)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(realtime.router)
    return app


app = create_app()
