"""作用：实现独立于意图识别的轻量记忆预筛选与证据规则。

当前功能：
- 区分明确记忆操作、历史查询、稳定事实与临时状态。
- 按原文所在分句检查推测、引用和凭据，防止省略限定词后入库。
- 将严格校验的原子事实转换为存储候选。
"""

import re
from datetime import datetime, timezone

from src.contracts.memory import ExtractedFact, MemoryCandidate


EXPLICIT = re.compile(r"(?:请|帮我|给我)?(?:记住|记下|保存|记录)(?:我|一下|这|以下|：|:)|以后.{0,8}(?:叫我|称呼我|回答|回复)")
DELETE = re.compile(r"(?:忘掉|忘记|删除|清除|清空|不要再记|别再记).{0,60}(?:我|记忆|偏好|信息|记录|事实)")
CORRECTION = re.compile(r"更正|纠正|改为|改成|更新.{0,8}(?:偏好|事实|记忆)|不再|以后|其实|现在.{0,8}(?:喜欢|住|叫|研究)|换成")
QUERY = re.compile(r"你.{0,4}记得|记住了什么|有哪些.{0,4}记忆|所有.{0,4}记忆|全部.{0,4}记忆|之前聊过|之前说过|以前告诉过")
STABLE = re.compile(r"我(?:的.{0,20})?(?:叫|住在|居住|来自|喜欢|不喜欢|偏好|习惯|研究|长期|目标|职业|工作是|是一名|是个|从事|在.{0,20}工作)|以后.{0,12}(?:回答|回复|叫我)|(?:项目|方案|计划).{0,30}(?:确定|决定|确认|采用)|(?:确定|决定|确认).{0,30}(?:项目|方案|长期|采用)")
TEMPORARY = re.compile(r"今天|今晚|今早|此刻|目前心情|现在心情|刚才|这会儿|情绪|心情|中间转写|ASR.?partial", re.I)
UNCERTAIN = re.compile(r"可能|也许|或许|大概|假如|假设|如果|打算考虑|还没决定|不确定|猜测|推测|似乎|没说过|从没说|不是真的")
SECRET = re.compile(r"密码|口令|私钥|API.?KEY|访问令牌|身份证号|银行卡号", re.I)


def clauses(text: str) -> list[str]:
    """保留语义限定所在的句子，避免只校验模型选择的短证据。"""
    return [part.strip() for part in re.split(r"[。！？!?\n；;]", text) if part.strip()]


def quoted_or_discussed(text: str, evidence: str) -> bool:
    """引用、示例和讨论删除方法不能被当作用户事实或操作授权。"""
    for match in re.finditer(r'[“「『"].*?[”」』"]', text):
        if evidence in match.group():
            return True
    surrounding = next((part for part in clauses(text) if evidence in part), text)
    return bool(re.search(r"他说|她说|别人说|例如|比如|举例|怎么.{0,8}(?:删除|记住)|如何.{0,8}(?:删除|保存)|不要(?:忘掉|删除|清空|记住|保存|记录)|别(?:忘掉|删除|清空|记住|保存)", surrounding))


def operation(text: str) -> str | None:
    """返回记忆模块自己的明确操作；不接受引用或否定指令。"""
    for name, pattern in (("delete", DELETE), ("write", EXPLICIT), ("query", QUERY)):
        match = pattern.search(text)
        if match and not quoted_or_discussed(text, match.group()):
            return name
    correction = CORRECTION.search(text)
    if (correction and re.search(r"偏好|称呼|名字|回答|回复|住址|城市|研究方向|记忆|事实|长期目标", text)
            and not UNCERTAIN.search(text) and not quoted_or_discussed(text, correction.group())):
        return "write"
    return None


def eligible(text: str) -> bool:
    """稳定事实按分句预筛选，临时内容不会排除其他句中的有效事实。"""
    if operation(text) == "write":
        return True
    return any(STABLE.search(local) and not TEMPORARY.search(local) and not UNCERTAIN.search(part)
               and not SECRET.search(part) and not quoted_or_discussed(text, local)
               for part in clauses(text) for local in re.split(r"[，,]", part))


def validate_fact(fact: ExtractedFact, source: str, *, explicit: bool, source_kind: str = "user") -> MemoryCandidate | None:
    """核对原文、时效和限定语；置信度不能替代来源证据。"""
    if not fact.should_write or fact.is_temporary or fact.confidence < .85:
        return None
    if not fact.content.strip() or not fact.evidence or fact.evidence not in source or fact.content not in fact.evidence:
        raise ValueError("事实缺少完整原文证据")
    enclosing = next((part for part in clauses(source) if fact.evidence in part), fact.evidence)
    if UNCERTAIN.search(enclosing) or SECRET.search(enclosing) or quoted_or_discussed(source, fact.content):
        return None
    local = next((part for part in re.split(r"[，,]", enclosing) if fact.content in part), enclosing)
    if re.search(r"吗|什么|是否|猜猜|你猜", local):
        return None
    if re.search(r"不|没|并非|从未", local) and not re.search(r"不|没|并非|从未", fact.content):
        return None
    if re.search(r"心情|情绪|今晚.{0,8}(?:想吃|要吃)|今天.{0,8}(?:好累|难过|烦躁)", local):
        return None
    if TEMPORARY.search(local) and fact.memory_type != "event" and not re.search(r"以后|长期|一直|习惯|总是", local):
        return None
    if fact.memory_type == "event" and not explicit:
        return None
    if fact.memory_type == "decision" and not re.search(r"确定|决定|确认|采用", enclosing):
        return None
    if source_kind == "user" and not (STABLE.search(enclosing) or explicit or CORRECTION.search(enclosing)):
        return None
    if fact.memory_type == "decision" and fact.domain_scope == "daily":
        return None
    if fact.domain_scope != "daily" and fact.domain_scope.removeprefix("project:") not in source:
        raise ValueError("项目范围没有原文来源")
    prefix = "event" if fact.memory_type == "event" else "user"
    if not re.fullmatch(prefix + r"\.[a-z][a-z0-9_]{0,80}", fact.fact_key):
        raise ValueError("事实键无效")
    if fact.expires_at:
        expiry = datetime.fromisoformat(fact.expires_at)
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
            return None
        if fact.expires_at[:10] not in source:
            raise ValueError("有效期没有原文来源")
    return MemoryCandidate(action="remember", kind="episodic" if prefix == "event" else "semantic",
                           fact_key=fact.fact_key, content=fact.content, evidence=fact.evidence,
                           memory_type=fact.memory_type, domain_scope=fact.domain_scope,
                           expires_at=fact.expires_at, confidence=fact.confidence, source_kind=source_kind,
                           source_ref=fact.source_id)
