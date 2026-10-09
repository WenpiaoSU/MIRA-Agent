"""作用：提供文本与语音会话的创建和关闭接口。

当前功能：
- 创建进程内会话并返回连接所需标识与语音能力。
- 关闭指定会话，并提供仅含控制元数据的轨迹查看接口。
"""

from fastapi import APIRouter, HTTPException, Request

from src.gateway.sessions import SessionRegistry


router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("")
async def open_session(request: Request):
    """创建会话并告知前端当前支持的输入与角色能力。"""
    registry: SessionRegistry = request.app.state.sessions
    session = registry.create()
    return {"session_id": session.session_id, "user_id": None, "connection_epoch": session.connection_epoch, "feature_flags": {"text": True, "avatar": True, "audio": request.app.state.speech_service.ready, "vision": False}, "negotiated_formats": {"audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "frame_ms": 20}}}


@router.delete("/{session_id}")
async def close_session(session_id: str, request: Request):
    """校验会话标识，删除会话并取消正在生成的回复。"""
    registry: SessionRegistry = request.app.state.sessions
    if not registry.get(session_id):
        raise HTTPException(404, "会话不存在")
    registry.delete(session_id)
    return {"status": "ok"}

@router.get("/{session_id}/trace")
async def session_trace(session_id: str, request: Request):
    """导出会话的有界脱敏轨迹，缺失样本的指标保持为空。"""
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "会话不存在")
    return session.trace.report()
