/**
 * 作用：定义 Avatar 映射和模型参数的数据类型。
 * 当前功能：
 * - 描述动作曲线、参数轨道和语义动作。
 * - 约束映射文件与参数目录的结构。
 */

/** 参数变化所使用的过渡曲线。 */
export type Curve = 'smooth' | 'linear' | 'ease-in' | 'ease-out' | 'spring';
/** 预留的单个参数关键帧结构。 */
export type Keyframe = { at: number; value: number };
/** 一条模型参数从起始值到目标值的变化轨道。 */
export type ParameterTrack = { parameterId: string; label: string; from: number; to: number; curve: Curve; keyframes?: Keyframe[] };
/** 可在工作台预览或由语义计划触发的动作配置。 */
export type AvatarAction = { id: string; label: string; category: 'expression' | 'state' | 'gaze' | 'gesture' | 'viseme'; description: string; durationMs: number; fadeInMs: number; fadeOutMs: number; tracks: ParameterTrack[] };
/** 当前模型的全部语义动作映射。 */
export type AvatarMapping = { schemaVersion: '1.0'; modelId: string; updatedAt: string; actions: AvatarAction[] };
/** 模型参数的名称、默认值和合法范围。 */
export type ParameterDefinition = { id: string; name: string; min: number; max: number; default: number };
