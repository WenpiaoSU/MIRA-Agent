"""作用：把 Agent 的语义输出编译为前端角色行为计划。

当前功能：
- 将经过校验的表情、视线和动作线索转换为 BehaviorPlan。
- 保持语义名称与 Live2D 模型参数之间的职责分离。
"""

from uuid import uuid4

from src.contracts.agent import ResponsePlan
from src.contracts.behavior import AvatarCue, BehaviorPlan


def compile_response(response: ResponsePlan, cue: AvatarCue, generation_id: str) -> BehaviorPlan:
    """把已校验的语义线索编译为角色行为计划，不向 Agent 暴露模型参数。"""
    return BehaviorPlan(
        plan_id=str(uuid4()),
        generation_id=generation_id,
        response_id=response.response_id,
        emotion={"name": cue.expression, "intensity": cue.intensity},
        face={"expression": cue.expression, "blend": {}},
        gaze={"target": cue.gaze},
        gestures=([{"name": cue.gesture, "intensity": cue.intensity}] if cue.gesture else []),
        timing={"anchor": "plan_ready", "response_delay_ms": 0},
        constraints={"interruptible": True, "unsupported_action_policy": "drop_and_report", "max_expression_intensity": 1.0},
    )
