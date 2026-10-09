"""作用：集中读取后端语言、识别和语音合成接口配置。

当前功能：
- 从 backend/.env 或环境变量读取接口地址、密钥和模型名称。
- 判断流式语言模型调用所需配置是否完整。
- 从环境变量读取 TTS URL、Key、模型名、音色和可选情感标签映射。
- 从环境变量读取 ASR 配置与 P3 轮次、停止确认及缓冲预算。
"""

from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Settings:
    """保存语言模型、ASR 和 TTS 接口配置的不可变快照。"""
    llm_url: str = getenv("MIRA_LLM_URL", "").strip()
    llm_key: str = getenv("MIRA_LLM_KEY", "").strip()
    llm_model_name: str = getenv("MIRA_LLM_MODEL_NAME", "").strip()
    asr_url: str = getenv("MIRA_ASR_URL", "").strip()
    asr_key: str = getenv("MIRA_ASR_KEY", "").strip()
    asr_model_name: str = getenv("MIRA_ASR_MODEL_NAME", "").strip()
    asr_language: str = getenv("MIRA_ASR_LANGUAGE", "").strip()
    asr_provider: str = getenv("MIRA_ASR_PROVIDER", "auto").strip().lower()
    tts_url: str = getenv("MIRA_TTS_URL", "").strip()
    tts_key: str = getenv("MIRA_TTS_KEY", "").strip()
    tts_model_name: str = getenv("MIRA_TTS_MODEL_NAME", "").strip()
    tts_voice: str = getenv("MIRA_TTS_VOICE", "").strip()
    tts_emotion_tags: str = getenv("MIRA_TTS_EMOTION_TAGS", "").strip()

    full_duplex_enabled: bool = getenv("MIRA_FULL_DUPLEX_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    turn_silence_ms: int = max(300, int(getenv("MIRA_TURN_SILENCE_MS", "700")))
    barge_in_ms: int = max(200, int(getenv("MIRA_BARGE_IN_MS", "600")))
    max_utterance_ms: int = max(1000, int(getenv("MIRA_MAX_UTTERANCE_MS", "60000")))
    stop_ack_timeout_ms: int = max(500, int(getenv("MIRA_STOP_ACK_TIMEOUT_MS", "2000")))
    playback_timeout_ms: int = max(5000, int(getenv("MIRA_PLAYBACK_TIMEOUT_MS", "30000")))
    generation_timeout_ms: int = max(10000, int(getenv("MIRA_GENERATION_TIMEOUT_MS", "120000")))
    output_buffer_ms: int = max(1000, int(getenv("MIRA_OUTPUT_BUFFER_MS", "15000")))

    @property
    def llm_ready(self) -> bool:
        """仅在地址、密钥和模型名称均已填写时返回真。"""
        return bool(self.llm_url and self.llm_key and self.llm_model_name)

    @property
    def tts_ready(self) -> bool:
        """仅在 TTS URL、Key、模型名和音色引用均已填写时返回真。"""
        return bool(self.tts_url and self.tts_key and self.tts_model_name and self.tts_voice)

    @property
    def asr_ready(self) -> bool:
        """仅在转写接口地址、密钥和模型名齐备时返回真。"""
        return bool(self.asr_url and self.asr_key and self.asr_model_name)

