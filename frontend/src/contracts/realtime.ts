/**
 * 作用：定义浏览器接收的实时事件与角色计划类型。
 * 当前功能：
 * - 约束后端行为计划的表情、视线和动作字段。
 * - 约束完整语段 TTS 音频和播放反馈字段。
 * - 约束 WebSocket 的连接、追踪、取消和实际交付事件。
 */

/** 后端发送给角色渲染层的单轮语义行为计划。 */
export type BehaviorPlan = {
  plan_id: string;
  generation_id: string;
  response_id: string;
  emotion: { name: string; intensity: number };
  face: { expression: string; blend: Record<string, number> };
  gaze: { target: string };
  gestures: Array<{ name: string; intensity: number }>;
  state: 'responding';
  timing: { anchor: string; response_delay_ms: number };
  speech: null;
  constraints: { interruptible: boolean; unsupported_action_policy: string; max_expression_intensity: number };
};

/** 后端按完整语段发送的 WAV 音频负载。 */
export type TtsChunkPayload = {
  generation_id: string;
  segment_id: string;
  chunk_seq: 0;
  audio_base64: string;
  encoding: 'wav';
  sample_rate: number;
  channels: number;
  audio_offset_ms: 0;
  duration_ms: number;
  is_last: true;
};

/** 浏览器向后端报告的实际播放位置。 */
export type PlaybackFeedbackPayload = {
  generation_id: string;
  segment_id: string;
  played_ms: number;
  duration_ms: number;
  clock_ms: number;
};

/** WebSocket 服务端事件的信封和负载。 */
export type ServerEvent = {
  schema_version: '1.0';
  event_id: string;
  type: string;
  source: string;
  session_id: string;
  connection_epoch: number;
  turn_id: string | null;
  generation_id: string | null;
  seq: number;
  occurred_at: string;
  trace_id: string;
  causation_id: string | null;
  payload: Record<string, unknown>;
};
