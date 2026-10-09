/**
 * 作用：将角色语义动作按时间采样为 Live2D 参数帧。
 * 当前功能：
 * - 按配置曲线计算渐入、保持与渐出。
 * - 将 idle 动作按完整周期往返采样，供模型持续待机。
 * - 让表情和视线持续显示，并让短动作回到基础姿态。
 */

import type { AvatarAction, ParameterDefinition, ParameterTrack } from './types';

/** 一条待播放动作及其强度和持续方式。 */
export type TimelineAction = {
  action: AvatarAction;
  intensity: number;
  persistent: boolean;
};

/** 将比例值限制在零到一之间。 */
const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
/** 在两个参数值之间做线性插值。 */
const mix = (from: number, to: number, progress: number) => from + (to - from) * progress;

/** 根据轨道配置计算当前进度对应的曲线权重。 */
function curve(track: ParameterTrack, progress: number): number {
  const p = clamp01(progress);
  switch (track.curve) {
    case 'ease-in': return p * p;
    case 'ease-out': return 1 - (1 - p) * (1 - p);
    case 'smooth': return p * p * (3 - 2 * p);
    case 'spring': return 1 - Math.cos(p * Math.PI * 3) * Math.exp(-5 * p);
    default: return p;
  }
}

/** 按 idle 动作时长循环采样参数，从轨道中点开始并平滑往返。 */
export function sampleIdleLoop(
  action: AvatarAction,
  elapsedMs: number,
  definitions: ParameterDefinition[],
): Record<string, number> {
  const duration = Math.max(1, action.durationMs);
  const phase = (elapsedMs % duration) / duration;
  const progress = (1 + Math.sin(phase * Math.PI * 2)) / 2;
  const byId = new Map(definitions.map(definition => [definition.id, definition]));
  const values: Record<string, number> = {};
  for (const track of action.tracks) {
    const definition = byId.get(track.parameterId);
    if (definitions.length && !definition) continue;
    const value = mix(track.from, track.to, curve(track, progress));
    values[track.parameterId] = definition
      ? Math.max(definition.min, Math.min(definition.max, value))
      : value;
  }
  return values;
}

/** 按时间采样角色参数：表情与视线保持，短动作执行后回落。 */
export function sampleTimeline(
  actions: TimelineAction[],
  elapsedMs: number,
  definitions: ParameterDefinition[],
  initial: Record<string, number>,
): Record<string, number> {
  const byId = new Map(definitions.map(definition => [definition.id, definition]));
  const values: Record<string, number> = {};

  for (const { action, intensity, persistent } of actions) {
    const duration = Math.max(action.fadeInMs, action.durationMs);
    const fadeIn = Math.max(1, action.fadeInMs);
    const fadeOut = Math.max(1, action.fadeOutMs);
    for (const track of action.tracks) {
      const definition = byId.get(track.parameterId);
      if (definitions.length && !definition) continue;
      const base = values[track.parameterId] ?? definition?.default ?? track.from;
      const start = initial[track.parameterId] ?? base;
      const target = mix(track.from, track.to, clamp01(intensity));
      let value: number;

      if (persistent) {
        value = mix(start, target, curve(track, elapsedMs / fadeIn));
      } else if (elapsedMs >= duration + fadeOut) {
        continue;
      } else if (elapsedMs > duration) {
        value = mix(target, base, curve(track, (elapsedMs - duration) / fadeOut));
      } else if (action.id === 'shake_head') {
        const firstSweep = Math.max(1, Math.min(fadeIn, duration / 3));
        const left = mix(0, track.from, clamp01(intensity));
        value = elapsedMs < firstSweep
          ? mix(base, left, curve(track, elapsedMs / firstSweep))
          : mix(left, target, curve(track, (elapsedMs - firstSweep) / Math.max(1, duration - firstSweep)));
      } else {
        value = mix(start, target, curve(track, elapsedMs / fadeIn));
      }

      values[track.parameterId] = definition
        ? Math.max(definition.min, Math.min(definition.max, value))
        : value;
    }
  }
  return values;
}

/** 计算这组动作需要持续采样的最长时间。 */
export function timelineDuration(actions: TimelineAction[]): number {
  return Math.max(0, ...actions.map(({ action, persistent }) =>
    persistent ? action.fadeInMs : Math.max(action.fadeInMs, action.durationMs) + action.fadeOutMs));
}
