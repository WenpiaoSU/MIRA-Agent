"""作用：通过本地 Silero ONNX 模型检测语音活动。

当前功能：
- 以 512 采样窗口运行 16 kHz 模型并保留跨窗口状态。
- 使用起点迟滞和结束 hangover 生成 start/end 阶段。
"""

from collections import deque
from pathlib import Path


class SileroVAD:
    """为一条音频流保存 Silero 推理状态与迟滞计数。"""

    def __init__(self, model_path: Path):
        """从指定本地路径加载 ONNX 图。"""
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        """清除模型与 VAD 状态，供下一音频流使用。"""
        self.buffer = self.np.empty(0, dtype=self.np.float32)
        self.context = self.np.zeros((1, 64), dtype=self.np.float32)
        self.state = self.np.zeros((2, 1, 128), dtype=self.np.float32)
        self.recent = deque(maxlen=3)
        self.active = False
        self.silence_windows = 0

    def feed(self, samples) -> list[tuple[str, float]]:
        """消费样本并返回每个已完成窗口的阶段与语音概率。"""
        self.buffer = self.np.concatenate((self.buffer, samples))
        events = []
        while len(self.buffer) >= 512:
            chunk, self.buffer = self.buffer[:512], self.buffer[512:]
            model_input = self.np.concatenate((self.context, chunk.reshape(1, -1)), axis=1)
            output, self.state = self.session.run(None, {"input": model_input, "state": self.state, "sr": self.np.array(16000, dtype="int64")})
            self.context = model_input[:, -64:]
            probability = float(output.reshape(-1)[0])
            self.recent.append(probability >= 0.55)
            if not self.active and sum(self.recent) >= 2:
                self.active = True
                self.silence_windows = 0
                events.append(("start", probability))
            elif self.active:
                self.silence_windows = 0 if probability >= 0.35 else self.silence_windows + 1
                if self.silence_windows >= 10:
                    self.active = False
                    self.recent.clear()
                    events.append(("end", probability))
                else:
                    events.append(("active", probability))
        return events
