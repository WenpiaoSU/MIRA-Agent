/**
 * 作用：提供 Avatar 语义动作的前端默认映射。
 * 当前功能：
 * - 定义表情、视线、姿态和口型动作的参数轨道。
 * - 在外部映射文件不可用时提供可加载的初始配置。
 */

import type { AvatarAction, AvatarMapping, ParameterTrack } from './types';
/** 快速创建一条模型参数轨道。 */
const t=(parameterId:string,label:string,from:number,to:number,curve:ParameterTrack['curve']='smooth'):ParameterTrack=>({parameterId,label,from,to,curve});
/** 组合动作元数据、时长和参数轨道。 */
const action=(id:string,label:string,category:AvatarAction['category'],description:string,durationMs:number,tracks:ParameterTrack[],fadeInMs=100,fadeOutMs=150):AvatarAction=>({id,label,category,description,durationMs,fadeInMs,fadeOutMs,tracks});
/** 内置动作集合，仅在外部映射缺失时作为默认数据。 */
export const initialActions:AvatarAction[]=[
 action('neutral','自然','expression','回到角色默认表情',900,[t('ParamMouthForm','嘴形',0,0),t('ParamBrowLY','左眉',0,0),t('ParamBrowRY','右眉',0,0)]),
 action('gentle_smile','微笑','expression','明显、温和的笑意',1800,[t('ParamMouthForm','嘴形',0,0.95),t('ParamBrowLY','左眉',0,0.3),t('ParamBrowRY','右眉',0,0.3)]),
 action('concerned','关切','expression','收紧眉眼并减弱笑意',1700,[t('ParamMouthForm','嘴形',0,-0.85),t('ParamBrowLY','左眉',0,-0.58),t('ParamBrowRY','右眉',0,-0.58),t('ParamEyeLOpen','左眼开合',1,0.62),t('ParamEyeROpen','右眼开合',1,0.62)]),
 action('sad','难过','expression','明显的低落表情',1900,[t('ParamMouthForm','嘴形',0,-1.1),t('ParamBrowLY','左眉',0,-0.78),t('ParamBrowRY','右眉',0,-0.78),t('ParamEyeLOpen','左眼开合',1,0.48),t('ParamEyeROpen','右眼开合',1,0.48)]),
 action('confused','困惑','expression','明显歪头和不对称眉形，缓慢倾斜并停留',1800,[t('ParamMouthForm','嘴形',0,-0.38),t('ParamBrowLY','左眉',0,0.62),t('ParamBrowRY','右眉',0,-0.42),t('ParamAngleZ','头部倾斜',0,13,'smooth')],700,500),
 action('idle','待机','state','轻呼吸、微摆并允许自然眨眼',5000,[t('ParamBreath','呼吸',0,0.8),t('ParamAngleZ','头部轻摆',-4,4),t('ParamBodyAngleZ','身体轻摆',-2,2)],140,180),
 action('listening','倾听','state','看向用户并缓慢点头',1600,[t('ParamEyeBallX','眼球水平',0,0),t('ParamEyeBallY','眼球垂直',0,0.3),t('ParamAngleY','头部俯仰',0,5)],500,350),
 action('thinking','思考','state','视线移开并缓慢侧头思考',1800,[t('ParamEyeBallX','眼球水平',0,-0.55),t('ParamEyeBallY','眼球垂直',0,0.35),t('ParamAngleZ','头部倾斜',0,-8,'smooth')],600,400),
 action('speaking','说话','state','实际播放语音时按音频能量驱动的口型',300,[t('ParamMouthOpenY','嘴部开合',0,0.48)]),
 action('gaze_center','注视中心','gaze','视线回到画面中心',900,[t('ParamEyeBallX','眼球水平',0,0),t('ParamEyeBallY','眼球垂直',0,0)]),
 action('gaze_away','移开视线','gaze','明显移开视线',1000,[t('ParamEyeBallX','眼球水平',0,0.72),t('ParamEyeBallY','眼球垂直',0,-0.28)]),
 action('nod','点头','gesture','缓慢点头并平稳回正',1900,[t('ParamAngleY','头部俯仰',0,15,'smooth'),t('ParamBodyAngleY','身体俯仰',0,4,'smooth')],700,550),
 action('shake_head','摇头','gesture','缓慢地左右摇头并平稳停止',2000,[t('ParamAngleX','头部左右',-14,14,'smooth'),t('ParamBodyAngleX','身体左右',-3,3,'smooth')],750,550),
 action('tilt','歪头','gesture','头部缓慢倾斜并平稳回正',1800,[t('ParamAngleZ','头部倾斜',0,15,'smooth'),t('ParamBodyAngleZ','身体倾斜',0,4,'smooth')],700,550),
 action('lower_head','低头','gesture','缓慢低头并带动视线下垂',1800,[t('ParamAngleY','头部俯仰',0,-15,'smooth'),t('ParamEyeBallY','眼球垂直',0,-0.32,'smooth')],700,550),
 action('mouth_open','口型：张开','viseme','调试口型幅度；正式驱动来自播放音频',900,[t('ParamMouthOpenY','嘴部开合',0,1.7)]),
 action('mouth_closed','口型：闭合','viseme','停止播放时嘴形归零',500,[t('ParamMouthOpenY','嘴部开合',0,0)])
];
/** 包含模型标识、版本和全部内置动作的默认映射。 */
export const defaultMapping:AvatarMapping={schemaVersion:'1.0',modelId:'c9db5a658c584955859280be904a7097',updatedAt:new Date().toISOString(),actions:initialActions};
