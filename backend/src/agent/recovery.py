"""作用：识别可信的明确控制短句并构建恢复约束。

当前功能：
- 对完整输入识别停止、继续、重述和换话题指令。
- 根据停止确认与句子级进度决定保守续说，禁止自动重播。
"""

import re


def control_action(text: str) -> str:
    """仅匹配完整控制短句，不将含有停止词的普通问题当作命令。"""
    normalized = re.sub(r"[\s。！？!?，,、]", "", text)
    if normalized in {"停止", "停一下", "停下", "别说了", "暂停", "取消"}:
        return "stop"
    if normalized in {"继续", "继续说", "接着说", "继续刚才的", "继续刚才的话题", "请继续"}:
        return "resume"
    if normalized in {"再说一遍", "重复一遍", "重复一下", "刚才没听清", "重说一遍", "请重复"}:
        return "repeat"
    if normalized in {"换个话题", "不聊这个了"}:
        return "change_topic"
    return "respond"


def recovery_instruction(action: str, interruption: dict | None) -> str:
    """提供公开的恢复约束，不保存或索要模型内部推理过程。"""
    if action in {"resume", "repeat"} and (not interruption or not interruption.get("resume_allowed")):
        return "当前没有可恢复的上下文，请简短询问用户希望继续或重述哪一部分，不编造历史。"
    if action == "resume":
        return ("用户明确要求续说。依据原问题和最后完整播完的句子继续；部分播放句子可从句首简短重述。"
                "未确认停止或标为 approximate 的位置不可当作精确断点。remaining_text 仅包含已生成文本，"
                "response_complete=false 时按原问题重新生成后续内容。不要重复已完成的工具操作。")
    if action == "repeat":
        return "用户明确要求重述。用简洁语言重述相关内容，不执行旧工具或自动重播旧队列。"
    if action == "change_topic":
        return "用户明确要求换话题，停止延续旧回复。"
    return "先回答当前输入；中断快照只用于理解上下文，不自动续说，必要时自然询问是否继续。"

