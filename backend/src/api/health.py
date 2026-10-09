"""作用：提供后端进程和语言模型配置的状态接口。

当前功能：
- /health 报告服务存活状态。
- /ready 报告语言模型接口是否完成配置。
- /ready 额外报告 TTS 接口配置状态。
- /ready 报告本地 VAD 与 ASR API 配置是否齐备。
"""

from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/health")
async def health():
    """返回服务存活状态。"""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """报告语言模型、TTS、ASR 配置及本地 VAD 语音能力的状态。"""
    settings = request.app.state.settings
    return {"status": "ok" if settings.llm_ready else "degraded", "llm_configured": settings.llm_ready, "tts_configured": settings.tts_ready, "asr_configured": settings.asr_ready, "speech_ready": request.app.state.speech_service.ready}
