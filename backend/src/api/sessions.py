"""作用：提供文本与语音会话的创建和关闭接口。

当前功能：
- 创建进程内会话并返回连接所需标识与语音能力。
- 关闭指定会话，并提供仅含控制元数据的轨迹查看接口。
- 创建或恢复浏览器身份，并验证关闭和轨迹查询的会话归属。
"""

import sqlite3
from fastapi import APIRouter, HTTPException, Request, Response

from src.gateway.sessions import SessionRegistry
from src.gateway.identity import COOKIE_NAME, owns_session
from src.repositories.memory import token_digest


router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("")
async def open_session(request: Request, response: Response):
    """创建会话并告知前端当前支持的输入与角色能力。"""
    registry: SessionRegistry = request.app.state.sessions
    memory = getattr(request.app.state, "memory", None)
    identity = None
    if memory:
        try:
            identity = await memory.repository.identity(request.cookies.get(COOKIE_NAME), create=True)
        except (sqlite3.Error, OSError):
            pass
    if identity:
        user_id, token = identity
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict", secure=request.url.scheme == "https",
                            max_age=365 * 24 * 3600)
        session = registry.create(user_id, token_digest(token))
    else:
        session = registry.create()
    return {"session_id": session.session_id, "user_id": session.user_id, "connection_epoch": session.connection_epoch, "feature_flags": {"text": True, "avatar": True, "audio": request.app.state.speech_service.ready, "vision": False, "long_term_memory": bool(identity)}, "negotiated_formats": {"audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "frame_ms": 20}}}


@router.delete("/{session_id}")
async def close_session(session_id: str, request: Request):
    """校验会话标识，删除会话并取消正在生成的回复。"""
    registry: SessionRegistry = request.app.state.sessions
    session = registry.get(session_id)
    if not session or not owns_session(request, session):
        raise HTTPException(404, "会话不存在")
    registry.delete(session_id)
    return {"status": "ok"}

@router.get("/{session_id}/trace")
async def session_trace(session_id: str, request: Request):
    """导出会话的有界脱敏轨迹，缺失样本的指标保持为空。"""
    session = request.app.state.sessions.get(session_id)
    if session is None or not owns_session(request, session):
        raise HTTPException(404, "会话不存在")
    return session.trace.report()
