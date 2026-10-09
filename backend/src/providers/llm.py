"""作用：适配兼容 Chat Completions 协议的语言模型接口。

当前功能：
- 发送包含对话上下文的流式请求。
- 解析服务端事件中的文本增量并统一连接错误。
"""

import json
from collections.abc import AsyncIterator

import httpx

from src.core.config import Settings


class LLMError(Exception):
    """表示语言模型配置、连接或响应格式错误。"""
    pass


class ChatCompletionsProvider:
    """封装流式对话接口；配置中的 URL 应为完整请求地址。"""

    def __init__(self, settings: Settings):
        """保存当前请求使用的语言模型配置。"""
        self.settings = settings

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """逐段读取语言模型响应，并产出非空的文本增量。"""
        if not self.settings.llm_ready:
            raise LLMError("请先填写 backend/.env 中的 MIRA_LLM_URL、MIRA_LLM_KEY 和 MIRA_LLM_MODEL_NAME。")
        timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    self.settings.llm_url,
                    headers={"Authorization": f"Bearer {self.settings.llm_key}", "Content-Type": "application/json"},
                    json={"model": self.settings.llm_model_name, "messages": messages, "stream": True},
                ) as response:
                    response.raise_for_status()
                    # 逐行解析流式事件，只提取助手正文的文本增量。
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        if not data:
                            continue
                        try:
                            payload = json.loads(data)
                            choices = payload.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0]["delta"].get("content")
                        except (ValueError, KeyError, IndexError, TypeError) as exc:
                            raise LLMError("LLM 流式响应格式不正确。") from exc
                        if isinstance(delta, str) and delta:
                            yield delta
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM API 返回 HTTP {exc.response.status_code}。") from exc
        except httpx.RequestError as exc:
            raise LLMError("无法连接 LLM API，请检查 URL 与网络。") from exc
