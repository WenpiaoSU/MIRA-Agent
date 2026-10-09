"""作用：从 JSON 轨迹文件回放 P3 轮次控制。

当前功能：
- 读取导出的脱敏事件，用采集时钟驱动确定性状态机。
- 输出回放报告，全程不调用模型、TTS 或写工具。
"""

import argparse
import json
from pathlib import Path
from src.evaluation.trace import replay_turn


def main() -> None:
    """解析轨迹路径，验证事件数组并输出可复现控制报告。"""
    parser = argparse.ArgumentParser(description="回放 MIRA P3 的脱敏轨迹")
    parser.add_argument("trajectory", type=Path)
    arguments = parser.parse_args()
    data = json.loads(arguments.trajectory.read_text(encoding="utf-8-sig"))
    events = data.get("events") if isinstance(data, dict) else data
    if not isinstance(events, list):
        parser.error("轨迹必须是事件数组或包含 events 的对象")
    print(json.dumps(replay_turn(events), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
