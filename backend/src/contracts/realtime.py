"""作用：定义浏览器与后端之间的实时事件契约。

当前功能：
- 校验文本、停止、字幕交付与取消执行回执。
- 校验语音流边界及 20 毫秒 PCM 帧的元数据。
- 校验浏览器实际播放的语段位置反馈。
- 定义单个完整 WAV 语段的 TTS 输出负载。
- 生成带时间戳、序号和追踪标识的服务端事件。
"""

from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TimeRef(BaseModel):
    """记录事件所使用的时钟和采集时间范围。"""
    clock_id: str
    capture_start_ms: int = Field(ge=0)
    capture_end_ms: int = Field(ge=0)


def server_time_ref() -> TimeRef:
    """生成以服务端单调时钟为基准的时间引用。"""
    now = round(monotonic() * 1000)
    return TimeRef(clock_id="server-monotonic", capture_start_ms=now, capture_end_ms=now)


class ClientEnvelope(BaseModel):
    """定义所有客户端实时事件共用的信封字段。"""
    schema_version: Literal["1.0"]
    event_id: str
    source: Literal["avatar-web"]
    session_id: str
    connection_epoch: int = Field(ge=1)
    seq: int = Field(ge=0)
    occurred_at: datetime
    time_ref: TimeRef
    trace_id: str
    causation_id: str | None = None


class TextPayload(BaseModel):
    """限制用户输入文本的长度。"""
    text: str = Field(min_length=1, max_length=4000)


class TextInput(ClientEnvelope):
    """表示带轮次标识的用户文本输入事件。"""
    type: Literal["input.text"]
    turn_id: str
    payload: TextPayload


class AudioStartPayload(BaseModel):
    """声明新语音流的固定音频格式。"""
    stream_id: str = Field(min_length=1, max_length=80)
    encoding: Literal["pcm_s16le"]
    sample_rate: Literal[16000]
    channels: Literal[1]
    continuous: bool = False
    aec_mode: Literal["browser", "unavailable"] = "unavailable"


class AudioStartInput(ClientEnvelope):
    """表示麦克风开始向当前连接送入语音帧。"""
    type: Literal["input.audio_start"]
    payload: AudioStartPayload


class AudioFramePayload(BaseModel):
    """描述二进制帧中 PCM 数据的采集位置。"""
    stream_id: str
    frame_seq: int = Field(ge=0)
    capture_start_ms: int = Field(ge=0)
    capture_end_ms: int = Field(ge=0)


class AudioFrameInput(ClientEnvelope):
    """表示上传一帧标准 PCM 音频。"""
    type: Literal["input.audio_frame"]
    payload: AudioFramePayload


class AudioEndPayload(BaseModel):
    """指定要结束的麦克风流。"""
    stream_id: str
    submit: bool = True


class AudioEndInput(ClientEnvelope):
    """表示用户主动结束麦克风采集。"""
    type: Literal["input.audio_end"]
    payload: AudioEndPayload


class StopPayload(BaseModel):
    """携带可选的目标生成标识。"""
    generation_id: str | None = None


class StopInput(ClientEnvelope):
    """表示客户端请求停止生成的事件。"""
    type: Literal["control.stop"]
    payload: StopPayload


class PlaybackPayload(BaseModel):
    """描述当前语段的真实播放位置和浏览器单调时钟。"""
    generation_id: str
    segment_id: str
    played_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=1)
    clock_ms: int = Field(ge=0)


class PlaybackInput(ClientEnvelope):
    """表示浏览器实际开始、推进、结束或停止播放的事件。"""
    type: Literal["playback.started", "playback.progress", "playback.ended", "playback.stopped", "playback.failed", "playback.paused"]
    payload: PlaybackPayload


class TtsChunkPayload(BaseModel):
    """封装一个可直接播放的完整语段 WAV 与相对音频位置。"""
    generation_id: str
    segment_id: str
    chunk_seq: Literal[0] = 0
    audio_base64: str
    encoding: Literal["wav"] = "wav"
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    audio_offset_ms: Literal[0] = 0
    duration_ms: int = Field(gt=0)
    is_last: Literal[True] = True


class ServerEvent(BaseModel):
    """统一封装发往浏览器的服务端事件。"""
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    source: str
    session_id: str
    connection_epoch: int
    user_id: None = None
    turn_id: str | None = None
    generation_id: str | None = None
    seq: int
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    time_ref: TimeRef = Field(default_factory=server_time_ref)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    causation_id: str | None = None
    payload: dict[str, Any]

class DisplayedPayload(BaseModel):
    """确认客户端已经显示的稳定文本段。"""
    generation_id: str
    segment_id: str


class DisplayedInput(ClientEnvelope):
    """表示字幕展示完成，不将展示等同于说出。"""
    type: Literal["response.displayed"]
    payload: DisplayedPayload


class CommandAckPayload(BaseModel):
    """确认整代输出已停，包含客户端本地测量的停止耗时。"""
    command_id: str
    generation_id: str
    status: Literal["completed"]
    clock_ms: int = Field(ge=0)
    stop_elapsed_ms: int = Field(ge=0, le=60000)
    speech_to_stop_ms: int | None = Field(default=None, ge=0)


class CommandAckInput(ClientEnvelope):
    """表示客户端已经实际执行取消，不以入队冒充停止。"""
    type: Literal["command.ack"]
    payload: CommandAckPayload
