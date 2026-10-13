"""作用：记录和回放 P3 的确定性事件链。

当前功能：
- 使用共享 trace_id、因果事件 ID 和单调时间记录有界轨迹。
- 汇总取消到停止确认耗时，不混用客户端与服务端时钟。
- 用录制事件回放轮次状态机，不执行模型或外部工具。
- 记录意图枚举、分类耗时与澄清状态，不保留槽位和用户正文。
- 记录工具名称、调用标识、验证状态与安全错误码，不保留参数或结果正文。
- 记录任务计划标识、步骤归属和步数，保持任务计划与行为计划分离。
"""

from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from src.turn.manager import TurnManager


def now_ms() -> int:
    """返回服务端单调毫秒，不与浏览器时钟直接相减。"""
    return round(monotonic() * 1000)


@dataclass
class TraceRecorder:
    """仅保留控制及分类元数据；不保存密钥、原始音频或完整正文。"""
    events: deque = field(default_factory=lambda: deque(maxlen=2048))

    def record(self, event_type: str, trace_id: str, payload: dict,
               causation_id: str | None = None) -> None:
        """筛选控制与分类元数据，记录事件与本机接收时间。"""
        allowed = {"generation_id", "segment_id", "state", "phase", "start_ms", "end_ms",
                   "p_speech", "echo_likelihood", "clipping_ratio", "aec_mode", "continuous",
                   "reason", "played_ms", "clock_ms", "command_id", "status", "elapsed_ms",
                   "stop_elapsed_ms", "speech_to_stop_ms", "intent", "confidence", "needs_clarification",
                   "tool_name", "call_id", "verified", "error_code", "needs_planning",
                   "task_plan_id", "step_id", "step_count"}
        self.events.append({"type": event_type, "trace_id": trace_id,
                            "causation_id": causation_id, "server_ms": now_ms(),
                            "payload": {key: value for key, value in payload.items() if key in allowed}})

    def report(self) -> dict:
        """返回真实样本数与取消确认时延分位数，缺样本时不伪造指标。"""
        samples = sorted(event["payload"]["elapsed_ms"] for event in self.events
                         if event["type"] == "output.stop_confirmed")
        client_samples = sorted(event["payload"]["stop_elapsed_ms"] for event in self.events
                                if event["type"] == "command.ack" and "stop_elapsed_ms" in event["payload"])
        speech_samples = sorted(event["payload"]["speech_to_stop_ms"] for event in self.events
                                if event["type"] == "command.ack"
                                and event["payload"].get("speech_to_stop_ms") is not None)
        return {"events": list(self.events),
                "client_stop_samples": len(client_samples),
                "client_stop_p95_ms": client_samples[min(len(client_samples) - 1, int(len(client_samples) * .95))] if client_samples else None,
                "speech_to_stop_samples": len(speech_samples),
                "speech_to_stop_p95_ms": speech_samples[min(len(speech_samples) - 1, int(len(speech_samples) * .95))] if speech_samples else None, "stop_ack_samples": len(samples),
                "stop_ack_p50_ms": samples[(len(samples) - 1) // 2] if samples else None,
                "stop_ack_p95_ms": samples[min(len(samples) - 1, int(len(samples) * .95))] if samples else None,
                "latency_scope": "stop_ack 包含往返网络；client_stop 为客户端收到取消到停音；speech_to_stop 使用同一浏览器采集时钟的估计起点"}


def replay_turn(events: list[dict], silence_ms: int = 700, barge_in_ms: int = 600) -> dict:
    """消费录制事件的采集时间，确定性回放轮次控制且不调用写工具。"""
    turn = TurnManager(silence_ms=silence_ms, barge_in_ms=barge_in_ms)
    confirmed = []
    for event in events:
        payload = event["payload"]
        kind = event["type"]
        if kind == "audio.configured":
            turn.configure(payload["continuous"], payload["aec_mode"])
        elif kind == "generation.started":
            turn.begin(payload["generation_id"])
        elif kind.startswith("playback."):
            turn.playback(payload["generation_id"], payload["segment_id"], kind.split(".")[1])
        elif kind == "vad.event" and turn.vad(payload["phase"], payload["start_ms"],
                                             payload["end_ms"], payload["p_speech"],
                                             payload.get("echo_likelihood", 0),
                                             payload.get("clipping_ratio", 0)):
            confirmed.append(turn.generation_id)
        elif kind == "generation.finished":
            if turn.generation_id == payload["generation_id"]:
                turn.generating = False
        elif kind == "turn.delivered":
            if turn.generation_id == payload["generation_id"]:
                turn.generating = False
                turn.stopped(payload["generation_id"])
        elif kind == "audio.closed":
            turn.continuous = False
            turn.user_speaking = False
        elif kind == "output.cancel":
            turn.cancel(payload["generation_id"])
        elif kind == "output.stop_confirmed":
            turn.stopped(payload["generation_id"])
    return {"confirmed_generations": confirmed, "interaction": turn.snapshot().model_dump()}
