"""作用：在小规模结构化记忆中进行可解释的中文关键词检索。

当前功能：
- 使用事实键、英文词、中文双字片段与本地特征向量筛选候选。
- 普通对话附带称呼与回答风格，限定返回条数和正文预算。
- 向量仅用于候选筛选，最终事实关系由规则或语义比较决定。
"""

import re
import hashlib
import math

from src.contracts.memory import MemoryRecord


ALIASES = {"preferred_name": "称呼名字姓名叫", "reply_style": "回答回复风格简短详细",
           "research_topic": "研究方向课题主题", "home_city": "居住住址城市家"}


def terms(text: str) -> set[str]:
    """抽取英文词和中文相邻双字，过滤常见提问套语。"""
    for word in ("你还记得", "你记得", "我以前", "我之前", "告诉过你", "记住", "忘掉", "请问", "我的", "什么"):
        text = text.replace(word, " ")
    result = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(part[i:i + 2] for i in range(len(part) - 1))
    return result


def retrieve(records: list[MemoryRecord], query: str, *, preferences: bool = False,
             limit: int = 5, budget: int = 3000) -> list[MemoryRecord]:
    """相关性优先、同分按输入顺序保留较新记录；没有匹配时返回空列表。"""
    query_terms = terms(query)
    scored = []
    for record in records:
        aliases = ALIASES.get(record.fact_key.removeprefix("user."), "")
        score = len(query_terms & terms(record.content + " " + record.fact_key + " " + aliases))
        if preferences and record.fact_key in {"user.preferred_name", "user.reply_style"}:
            score += 2
        if score:
            scored.append((score, record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = []
    for _, record in scored:
        size = len(record.model_dump_json())
        if size <= budget:
            selected.append(record)
            budget -= size
        if len(selected) == limit:
            break
    return selected


def feature_vector(text: str, dimensions: int = 256) -> list[float]:
    """生成可复现的本地词片段向量，不等同于语义 embedding 模型。"""
    vector = [0.0] * dimensions
    for term in terms(text):
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % dimensions] += 1 if digest[4] % 2 else -1
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def match_candidates(records: list[MemoryRecord], candidate, limit: int = 8) -> list[MemoryRecord]:
    """合并精确键、正文与向量召回，候选相似不表示事实相同。"""
    query = feature_vector(candidate.content + " " + candidate.fact_key)
    scored = []
    for record in records:
        if record.domain_scope != candidate.domain_scope:
            continue
        exact = record.fact_key == candidate.fact_key or record.content == candidate.content
        similarity = sum(left * right for left, right in zip(query, feature_vector(record.content + " " + record.fact_key)))
        if exact or similarity > .15:
            scored.append((2 if exact else similarity, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]
