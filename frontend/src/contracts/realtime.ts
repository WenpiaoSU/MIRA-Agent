/**
 * 作用：定义浏览器接收的实时事件与角色计划类型。
 * 当前功能：
 * - 约束后端行为计划的表情、视线和动作字段。
 * - 约束完整语段 TTS 音频和播放反馈字段。
 * - 约束 WebSocket 的连接、追踪、取消和实际交付事件。
 * - 定义版本绑定的待确认记忆快照和明确处理动作。
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

/** 后台冲突通知只携带已验证候选，不能直接作为有效长期事实使用。 */
export type PendingMemoryOffer = {
  pending_id: string;
  target_id: string | null;
  target_version: number | null;
  target_content: string;
  candidate: { content: string; evidence: string; domain_scope: string; memory_type: string };
  reason: string;
  question: string;
  created_at: string;
  expires_at: number;
  delivery_token: string;
};

/** 用户决定只选择已展示快照，不能由前端改写候选正文或事实键。 */
export type MemoryResolutionAction = 'accept_new' | 'keep_old';
