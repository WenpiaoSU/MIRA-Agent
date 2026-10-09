"""作用：校验浏览器上传的 PCM 音频帧。

当前功能：
- 校验单声道 16 kHz s16le、帧长度与连续采集时间。
- 将 PCM 字节转换成供模型使用的归一化浮点样本。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFrame:
    """保存已通过格式校验的 20 毫秒音频帧。"""

    stream_id: str
    seq: int
    capture_start_ms: int
    capture_end_ms: int
    pcm: bytes

    def samples(self):
        """按需将 PCM 转成单声道 float32 数组。"""
        import numpy as np

        return np.frombuffer(self.pcm, dtype="<i2").astype(np.float32) / 32768.0


def validate_frame(stream_id: str, seq: int, start_ms: int, end_ms: int, pcm: bytes) -> AudioFrame:
    """拒绝错误格式和时间范围，返回标准音频帧。"""
    if not stream_id or seq < 0 or start_ms < 0 or end_ms - start_ms != 20 or len(pcm) != 640:
        raise ValueError("音频必须是 16 kHz 单声道 PCM s16le，每帧 20 ms")
    return AudioFrame(stream_id, seq, start_ms, end_ms, pcm)
