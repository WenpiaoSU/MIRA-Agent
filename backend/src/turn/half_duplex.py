"""作用：决定 ASR final 何时提交到现有 Agent。

当前功能：
- 合并短暂停顿间的多个 final。
- 在最后一段语音结束并经过静音窗口后提交完整轮次。
- 空白识别结果不触发 Agent。
"""


class HalfDuplexTurn:
    """为单条麦克风流保存尚未提交的识别文本。"""

    def __init__(self, silence_ms: int = 700):
        """设置最后一段语音后的轮次确认窗口。"""
        self.silence_ms = silence_ms
        self.parts: list[str] = []
        self.last_end_ms: int | None = None
        self.speaking = False

    def speech_started(self):
        """标记新语音开始，保留前一片段以支持短暂停顿。"""
        self.speaking = True
        self.last_end_ms = None

    def accept_final(self, text: str, end_ms: int):
        """只保存非空 final，并记录最后一段的结束时间。"""
        if text.strip():
            self.parts.append(text.strip())
        self.speaking = False
        self.last_end_ms = end_ms

    def complete_if_ready(self, now_ms: int, force: bool = False) -> str | None:
        """满足静音确认或显式结束时返回一次完整轮次文本。"""
        if self.speaking or (not force and (self.last_end_ms is None or now_ms - self.last_end_ms < self.silence_ms)):
            return None
        text = "".join(self.parts).strip()
        self.parts.clear()
        self.last_end_ms = None
        return text or None
