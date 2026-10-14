"""作用：提供当前浏览器用户的长期记忆查看与删除接口。

当前功能：
- Cookie 验证用户归属，支持查看有效记录和历史版本。
- 按单条事实或全部记忆执行删除，并同步清除会话派生缓存。
"""

import sqlite3
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from src.contracts.memory import MemoryCandidate
from src.gateway.identity import COOKIE_NAME


router = APIRouter(prefix="/v1/memories", tags=["memories"])


async def owner(request: Request):
    """验证当前凭据，故障或匿名请求不能访问长期记录。"""
    service = getattr(request.app.state, "memory", None)
    if not service:
        raise HTTPException(503, "长期记忆不可用")
    try:
        identity = await service.repository.identity(request.cookies.get(COOKIE_NAME))
    except (sqlite3.Error, OSError):
        raise HTTPException(503, "记忆存储暂时不可用") from None
    if identity is None:
        raise HTTPException(401, "请先创建会话以建立浏览器身份")
    return service, identity[0]


@router.get("")
async def list_memories(request: Request, historical: bool = False):
    """返回当前用户记录，历史记录明确保留 inactive 状态。"""
    service, user_id = await owner(request)
    try:
        _, records = await service.repository.snapshot(user_id, historical=historical)
        return {"records": [record.model_dump() for record in records]}
    except (sqlite3.Error, OSError):
        raise HTTPException(503, "记忆存储暂时不可用") from None


async def remove(request: Request, memory_id: str | None):
    """以明确的 HTTP 删除目标提交事务，成功后清除当前用户的派生记忆。"""
    service, user_id = await owner(request)
    try:
        revision, _ = await service.repository.snapshot(user_id)
        candidate = MemoryCandidate(action="delete", target_ids=[memory_id] if memory_id else [], delete_all=memory_id is None)
        result = await service.repository.commit(user_id, "api:" + str(uuid4()), candidate, revision, "", "",
                                                  service.deletion_callback(user_id))
        if result.status == "conflict":
            raise HTTPException(409, result.message)
        if not result.cleared:
            raise HTTPException(404, result.message)
        return result.model_dump()
    except (sqlite3.Error, OSError):
        raise HTTPException(503, "删除未确认完成") from None


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    """删除指定事实及全部版本，不能跨用户选择记录。"""
    return await remove(request, memory_id)


@router.delete("")
async def delete_all_memories(request: Request):
    """明确清空当前用户的全部长期记忆，不删除其他用户的数据。"""
    return await remove(request, None)
