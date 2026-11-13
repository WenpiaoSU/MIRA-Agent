"""作用：向当前可信连接展示待确认记忆并绑定可消费的短答上下文。

当前功能：
- 从持久化候选恢复列表，仅变化时通知，后台冲突不阻塞主对话。
- 用连接专属展示凭据验证问题已呈现，五分钟内绑定唯一候选。
- 每次最终输入消费或撤销上下文；工具补参、多候选和未展示短答不授权更新。
"""

import json
import re
from time import monotonic
from uuid import uuid4

from src.contracts.memory import MemoryConfirmation


class MemoryPendingDelivery:
    """管理一个 WebSocket 的候选快照与展示上下文，不承担事实事务。"""

    def __init__(self, service, session):
        """候选从 SQLite 读取，展示令牌和确认焦点仅属于当前连接。"""
        self.service = service
        self.session = session
        self.fingerprint = None
        self.offers = {}
        self.focus = None
        self.acknowledged = set()

    async def poll(self):
        """返回变化后的完整快照；重连从持久化候选重建，不依赖内存通知。"""
        rows = await self.service.repository.pending(self.session.user_id)
        fingerprint = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        if fingerprint == self.fingerprint:
            return None
        self.fingerprint = fingerprint
        self.offers = {row["pending_id"]: {**row, "delivery_token": str(uuid4())}
                       for row in rows}
        self.focus = None
        return {"pending": list(self.offers.values())}

    def acknowledge(self, acknowledgement):
        """只有唯一候选的真实展示回执可建立一次性确认焦点，重复回执不延长授权。"""
        offer = self.offers.get(acknowledgement.pending_id)
        if (len(self.offers) != 1 or not offer or offer["delivery_token"] != acknowledgement.delivery_token
                or offer["target_version"] != acknowledgement.target_version or acknowledgement.delivery_token in self.acknowledged):
            return False
        self.acknowledged.add(acknowledgement.delivery_token)
        if len(self.acknowledged) > 256:
            self.acknowledged = {acknowledgement.delivery_token}
        self.focus = {"pending_id": acknowledgement.pending_id, "target_version": acknowledgement.target_version,
                      "shown_at": monotonic()}
        return True

    def consume(self, text):
        """精确短答仅作用于最近已展示且无其他补参任务的候选，每个输入都会撤销旧焦点。"""
        focus, self.focus = self.focus, None
        if (not focus or len(self.offers) != 1 or monotonic() - focus["shown_at"] > 300
                or self.session.pending_clarification or self.session.active_plan or self.session.turn.generating
                or self.session.turn.robot_playing):
            return None
        normalized = re.sub(r"[\s。！!，,]", "", text)
        if normalized in {"是", "是的", "对", "确认", "用新的", "使用新信息", "采用新信息", "采用新记忆"}:
            action = "accept_new"
        elif normalized in {"不", "不是", "不要", "保留原来的", "保留原记录", "保留旧记忆", "用旧的"}:
            action = "keep_old"
        else:
            return None
        return MemoryConfirmation(pending_id=focus["pending_id"], target_version=focus["target_version"], action=action)

    def hide(self, acknowledgement):
        """卡片隐藏或离开聊天页时撤销对应焦点，旧回执不能重新建立授权。"""
        offer = self.offers.get(acknowledgement.pending_id)
        if (self.focus and self.focus["pending_id"] == acknowledgement.pending_id and offer
                and offer["delivery_token"] == acknowledgement.delivery_token):
            self.focus = None

    def invalidate(self):
        """删除、更新和连接状态变化时立即撤销旧确认上下文并触发快照刷新。"""
        self.focus = None
        self.fingerprint = None
        self.offers.clear()
