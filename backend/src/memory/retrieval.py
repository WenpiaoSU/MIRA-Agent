"""作用：在小规模结构化记忆中进行可解释的中文关键词检索。

当前功能：
- 使用事实键、英文词与中文双字片段匹配，兼顾简称和问法变化。
- 普通对话附带称呼与回答风格，限定返回条数和正文预算。
"""

import re

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
