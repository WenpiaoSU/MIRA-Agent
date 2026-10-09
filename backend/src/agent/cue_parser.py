"""作用：从语言模型的流式文本中提取角色语义首部。

当前功能：
- 容忍 JSON 前的空行、说明文字和可选代码围栏。
- 校验角色线索，并把剩余文本交给回复分段逻辑。
- 首部无效时回退到中性线索，同时保留原始回复文本。
"""

import json

from pydantic import ValidationError

from src.contracts.behavior import AvatarCue


class AvatarCueParser:
    """按流式块寻找首个合法的角色 JSON，避免将其显示为回复正文。"""

    def __init__(self, max_header_lines: int = 6, max_header_chars: int = 600):
        """设置首部扫描上限，防止无结构化内容时长期阻塞文本输出。"""
        self.max_header_lines = max_header_lines
        self.max_header_chars = max_header_chars
        self.buffer = ""
        self.header_lines: list[str] = []
        self.cue: AvatarCue | None = None
        self.cue_emitted = False
        self.awaiting_fence_end = False
        self.used_fallback = False

    @staticmethod
    def _parse_line(line: str) -> AvatarCue | None:
        """尝试把一整行解析为合法角色线索，失败时返回空值。"""
        try:
            return AvatarCue.model_validate(json.loads(line.strip().lstrip("\ufeff").strip()))
        except (ValueError, ValidationError):
            return None

    def _fallback(self) -> None:
        """发出中性线索，并把尚未识别的首部还原为回复正文。"""
        self.cue = AvatarCue()
        self.used_fallback = True
        self.buffer = "\n".join([*self.header_lines, self.buffer])
        self.header_lines.clear()

    def _read_header(self, final: bool) -> None:
        """扫描前几行，找到角色 JSON 或达到回退条件。"""
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            parsed = self._parse_line(line)
            if parsed is not None:
                self.cue = parsed
                self.awaiting_fence_end = any(item.strip().startswith("```") for item in self.header_lines)
                self.header_lines.clear()
                return
            self.header_lines.append(line)
            if len(self.header_lines) >= self.max_header_lines or sum(map(len, self.header_lines)) >= self.max_header_chars:
                self._fallback()
                return

        if final:
            parsed = self._parse_line(self.buffer)
            if parsed is not None:
                self.cue = parsed
                self.buffer = ""
                self.awaiting_fence_end = any(item.strip().startswith("```") for item in self.header_lines)
                self.header_lines.clear()
            else:
                self._fallback()
        elif len(self.buffer) + sum(map(len, self.header_lines)) > self.max_header_chars:
            self._fallback()

    def _read_body(self, final: bool) -> str:
        """移除 JSON 后的可选围栏结束行，并返回可见回复文本。"""
        if self.awaiting_fence_end:
            while "\n" in self.buffer:
                line, remainder = self.buffer.split("\n", 1)
                if not line.strip():
                    self.buffer = remainder
                    continue
                if line.strip() == "```":
                    self.buffer = remainder
                self.awaiting_fence_end = False
                break
            if self.awaiting_fence_end:
                if not final:
                    return ""
                if self.buffer.strip() == "```":
                    self.buffer = ""
                self.awaiting_fence_end = False
        body, self.buffer = self.buffer, ""
        return body

    def consume(self, chunk: str, *, final: bool = False) -> tuple[AvatarCue | None, str]:
        """接收文本增量；仅首次返回角色线索，正文可分批返回。"""
        self.buffer += chunk
        if self.cue is None:
            self._read_header(final)
        if self.cue is None:
            return None, ""
        cue = None if self.cue_emitted else self.cue
        self.cue_emitted = True
        return cue, self._read_body(final)
