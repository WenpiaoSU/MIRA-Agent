"""作用：以确定性状态机管理 P3 轮次与语音插话。

当前功能：
- 分离生成、实际播放、用户语音活动和持续监听。
- 仅在浏览器 AEC 可用且实际播放时确认持续有效语音插话。
- 使用采集时钟、连续概率和质量门控拒绝回声风险及短噪声。
"""

from dataclasses import dataclass
from src.contracts.state import InteractionState


@dataclass
class TurnManager:
    """维护单会话轮次状态，所有修改由事件循环顺序执行。"""
    silence_ms: int = 700
    barge_in_ms: int = 600
    continuous: bool = False
    aec_mode: str = "unavailable"
    user_speaking: bool = False
    robot_playing: bool = False
    generating: bool = False
    recovering: bool = False
    disconnected: bool = False
    generation_id: str | None = None
    playing_segment: str | None = None
    candidate_start_ms: int | None = None
    candidate_end_ms: int | None = None
    confirmed_generation: str | None = None
    version: int = 0

    def configure(self, continuous: bool, aec_mode: str) -> None:
        """只为已声明浏览器 AEC 的连接开放持续双工模式。"""
        self.continuous = continuous and aec_mode == "browser"
        self.aec_mode = aec_mode
        self.disconnected = False
        self.user_speaking = False
        self.reset_candidate()
        self.version += 1

    def reset_candidate(self) -> None:
        """清除连续语音计时，避免跨静音或输出代次累加。"""
        self.candidate_start_ms = None
        self.candidate_end_ms = None

    def begin(self, generation_id: str) -> None:
        """登记新输出；实际说话状态仍由播放反馈决定。"""
        self.generation_id = generation_id
        self.generating = True
        self.recovering = False
        self.robot_playing = False
        self.playing_segment = None
        self.confirmed_generation = None
        self.reset_candidate()
        self.version += 1

    def cancel(self, generation_id: str) -> None:
        """令输出不可继续提交，保留实际播放事实直到停止确认。"""
        if self.generation_id == generation_id:
            self.generating = False
            self.recovering = True
            self.version += 1

    def playback(self, generation_id: str, segment_id: str, phase: str) -> None:
        """只更新当前代次的实际播放事实，旧回执不能改变新一轮。"""
        if generation_id != self.generation_id:
            return
        if phase == "started":
            self.robot_playing = True
            self.playing_segment = segment_id
        elif phase in {"ended", "stopped", "failed", "paused"} and self.playing_segment == segment_id:
            self.robot_playing = False
            self.playing_segment = None
            self.reset_candidate()
        self.version += 1

    def stopped(self, generation_id: str) -> None:
        """根据整代输出停止确认解除播放与恢复状态。"""
        if generation_id == self.generation_id:
            self.robot_playing = False
            self.playing_segment = None
            self.recovering = False
            self.reset_candidate()
            self.version += 1

    def vad(self, phase: str, start_ms: int, end_ms: int, probability: float,
            echo_likelihood: float | None = None, clipping_ratio: float = 0) -> bool:
        """确认超过配置窗口的连续有效语音，完全不等待转写或模型。"""
        self.user_speaking = phase != "end"
        self.version += 1
        valid = (phase in {"start", "active"} and probability >= 0.55
                 and (echo_likelihood is None or echo_likelihood < 0.3) and clipping_ratio < 0.2)
        if (not valid or not self.robot_playing or not self.continuous
                or self.aec_mode != "browser" or self.confirmed_generation == self.generation_id):
            self.reset_candidate()
            return False
        if (self.candidate_end_ms is None or start_ms - self.candidate_end_ms > 100
                or start_ms < self.candidate_end_ms - 40):
            self.candidate_start_ms = start_ms
        self.candidate_end_ms = end_ms
        if end_ms - (self.candidate_start_ms if self.candidate_start_ms is not None else end_ms) > self.barge_in_ms:
            self.confirmed_generation = self.generation_id
            return True
        return False

    def snapshot(self) -> InteractionState:
        """冻结可序列化状态，优先表达实际播放和用户语音。"""
        mode = ("disconnected" if self.disconnected else
                "listening" if self.user_speaking else
                "recovering" if self.recovering else
                "speaking" if self.robot_playing else
                "thinking" if self.generating else
                "listening" if self.continuous else "idle")
        return InteractionState(mode=mode, user_speaking=self.user_speaking,
                                robot_playing=self.robot_playing, generation_id=self.generation_id,
                                continuous=self.continuous, aec_mode=self.aec_mode)

