"""作用：独立收集记忆模型的有界结构化响应。

当前功能：
- 复用已有流式供应商，不依赖意图识别服务。
- 限制响应大小与总超时，校验 JSON 和 Pydantic Schema。
"""

import asyncio
import json
import re


async def structured(provider, prompt, payload, schema, timeout_ms=10000):
    """收集严格 JSON；超时、取消和格式错误交给调用方决定降级。"""
    messages = [{"role": "system", "content": prompt + "\nSchema：" + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    chunks = []
    size = 0
    stream = provider.stream(messages)
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            async for chunk in stream:
                size += len(chunk.encode("utf-8"))
                if size > 24000:
                    raise ValueError("记忆模型响应超限")
                chunks.append(chunk)
    finally:
        close = getattr(stream, "aclose", None)
        if close:
            await close()
    raw = "".join(chunks).strip()
    wrapped = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    return schema.model_validate_json(wrapped.group(1) if wrapped else raw)
