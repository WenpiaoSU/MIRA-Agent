/**
 * 作用：在浏览器中加载并渲染 Live2D 角色。
 * 当前功能：
 * - 使用页面预加载的 Cubism Core 初始化 Pixi 渲染器，并由同一 ticker 推进模型与画布。
 * - 按机器人空间或调参工作台设置角色取景。
 * - 模型就绪后持续循环 idle，并叠加倾听、思考和对话行为参数。
 * - 将行为计划或手动预览转换为逐帧模型参数。
 * - 在实际语音播放期间按音频能量驱动嘴部，动作使用实际音频时钟。
 */

import { useEffect, useRef, useState } from 'react';
import type { AvatarAction, AvatarMapping, ParameterDefinition } from './types';
import type { BehaviorPlan } from '../contracts/realtime';
import type { PlaybackClock } from '../audio/player';
import { sampleIdleLoop, sampleTimeline, timelineDuration, type TimelineAction } from './behaviorTimeline';

/** 声明由外部脚本注入的 Cubism Core 全局对象。 */
declare global { interface Window { Live2DCubismCore?: unknown } }
/** 表示模型资源与渲染器的加载状态。 */
type RendererState='loading'|'ready'|'missing'|'error';
// 机器人空间与调参工作台分别配置角色取景；scale 控制缩放，x/y 控制画布内位置。
const AVATAR_FRAMING = {
 home: { scale: 1.7, x: 0.5, y: 1.75 },
 studio: { scale: 2.5, x: 0.5, y: 1.65 },
} as const;
/** 加载 Live2D 模型，并根据语义行为或手动动作更新可见参数。 */
export default function Live2DCanvas({mapping,action,behavior,definitions,speaking=false,mouthLevel=0,audioClock,interactionState='idle',replayKey=0,className=''}:{mapping:AvatarMapping;action?:AvatarAction;behavior?:BehaviorPlan|null;definitions:ParameterDefinition[];speaking?:boolean;mouthLevel?:number;audioClock?:()=>PlaybackClock|null;interactionState?:string;replayKey?:number;className?:string}){
 const mount=useRef<HTMLDivElement>(null);const [state,setState]=useState<RendererState>('loading');const runtime=useRef<any>({});
 const mappingRef=useRef(mapping);const definitionsRef=useRef(definitions);
 mappingRef.current=mapping;definitionsRef.current=definitions;
 const idleFrame=useRef<Record<string,number>>({});
 const parameterFrame=useRef<Record<string,number>>({});
 const mouthFrame=useRef<Record<string,number>>({});
 const selected=action?mapping.actions.find(a=>a.id===action.id)??action:undefined;
 // 仅在组件挂载时创建渲染器；卸载时释放监听器、模型与画布资源。
 useEffect(()=>{let dead=false;let resize:ResizeObserver|undefined;const run=async()=>{
  // 页面入口先加载 Cubism Core；当前组件只负责加载 Pixi 和模型资源。
  if(!window.Live2DCubismCore){if(!dead)setState('missing');return;}
  try{const PIXI=await import('pixi.js');const live2d=await import('pixi-live2d-display/cubism4');const app=new PIXI.Application({backgroundAlpha:0,antialias:true,autoDensity:true,resolution:Math.min(window.devicePixelRatio||1,2)});const view=app.view as HTMLCanvasElement;view.style.width='100%';view.style.height='100%';view.style.display='block';mount.current?.appendChild(view);const model=await live2d.Live2DModel.from('/models/robot/白-免费版.model3.json',{autoInteract:false,autoUpdate:false});if(dead){app.destroy(true,{children:true});return;}
   // 兼容 Cubism Core 6 中绘制顺序字段的位置变化。
   const coreModel:any=model.internalModel?.coreModel;const rawModel=coreModel?.getModel?.();if(coreModel&&rawModel&&(rawModel.renderOrders||rawModel.getRenderOrders)){coreModel.getDrawableRenderOrders=()=>rawModel.renderOrders??rawModel.getRenderOrders();}
   // 待机与行为参数先交给物理系统，再在绘制前叠加口型和精确的表情值。
   const applyMotionParameters=()=>{for(const frame of [idleFrame.current,parameterFrame.current])for(const [id,value] of Object.entries(frame))coreModel.setParameterValueById(id,value);};
   const applyVisualParameters=()=>{applyMotionParameters();for(const [id,value] of Object.entries(mouthFrame.current))coreModel.setParameterValueById(id,value);};
   model.internalModel.on('afterMotionUpdate',applyMotionParameters);
   model.internalModel.on('beforeModelUpdate',applyVisualParameters);
   // 模型就绪后在每次重绘前采样最新 idle 配置，避免首次挂载依赖额外的就绪 effect。
   const idleStart=performance.now();
   const advanceModel=()=>{
    const idle=mappingRef.current.actions.find(a=>a.id==='idle');
    idleFrame.current=idle?sampleIdleLoop(idle,performance.now()-idleStart,definitionsRef.current):{};
    model.update(app.ticker.deltaMS);
   };
   app.ticker.add(advanceModel,undefined,PIXI.UPDATE_PRIORITY.HIGH);
   app.ticker.start();
   // 根据容器尺寸和使用场景调整模型大小及位置。
   model.anchor.set(.5,.5);app.stage.addChild(model);advanceModel();const fit=()=>{if(!mount.current)return;const w=mount.current.clientWidth,h=mount.current.clientHeight,iw=model.internalModel?.width||model.internalModel?.originalWidth,ih=model.internalModel?.height||model.internalModel?.originalHeight;if(w<1||h<1||!iw||!ih||!Number.isFinite(iw)||!Number.isFinite(ih))throw new Error(`Invalid Live2D canvas/model size: canvas=${w}x${h}, model=${iw}x${ih}`);app.renderer.resize(w,h);
    // 从全身模型中取脸部与上半身，并使头部处于可见区域。
    const framing=className.includes('studio-live2d')?AVATAR_FRAMING.studio:AVATAR_FRAMING.home;const zoom=Math.max(w/iw,h/ih)*framing.scale;model.scale.set(zoom);model.position.set(w*framing.x,h*framing.y);app.renderer.render(app.stage);};fit();resize=new ResizeObserver(()=>{try{fit();}catch(e){console.error(e);}});if(mount.current)resize.observe(mount.current);runtime.current={app,model,advanceModel,applyMotionParameters,applyVisualParameters};setState('ready');window.dispatchEvent(new Event('mira:live2d-ready'));}catch(e){console.error('Live2D renderer unavailable',e);if(!dead)setState('error');}
 };void run();return()=>{dead=true;resize?.disconnect();try{runtime.current.app?.ticker?.remove(runtime.current.advanceModel);runtime.current.model?.internalModel?.off('afterMotionUpdate',runtime.current.applyMotionParameters);runtime.current.model?.internalModel?.off('beforeModelUpdate',runtime.current.applyVisualParameters);runtime.current.model?.destroy();runtime.current.app?.destroy(true,{children:true,texture:true,baseTexture:true});}catch{};idleFrame.current={};parameterFrame.current={};mouthFrame.current={};runtime.current={};};},[]);
 // 行为计划或预览动作发生变化时，重新计算每一帧的模型参数。
 useEffect(()=>{const {app,model}=runtime.current;if(state!=='ready'||!app||!model)return;
  const cueActions:TimelineAction[]=[];
  // 把语义名称解析为映射中的动作，并保留各动作的强度与持续方式。
  const append=(id:string,intensity:number,persistent:boolean)=>{const current=mapping.actions.find(a=>a.id===id);if(current)cueActions.push({action:current,intensity,persistent});};
  if(behavior){
   const intensity=behavior.emotion.intensity;
   append(behavior.face.expression,intensity,true);
   append(behavior.gaze.target==='away'?'gaze_away':'gaze_center',intensity,true);
   for(const gesture of behavior.gestures)append(gesture.name,gesture.intensity,false);
  }else if(className.includes('home-live2d')&&['listening','thinking'].includes(interactionState))append(interactionState,1,true);
  else if(selected)append(selected.id,1,false);
  if(!cueActions.length){parameterFrame.current={};return;}
  const initial={...parameterFrame.current},start=performance.now(),total=timelineDuration(cueActions);let frame=0;
  // 语音动作跟随实际媒体时钟；文字降级和工作台预览使用浏览器时钟。
  const tick=(now:number)=>{
   const clock=audioClock?.();
   const elapsed=behavior?.timing.anchor==='speech_start'&&clock?.generationId===behavior.generation_id
    ?clock.elapsedMs:now-start;
   parameterFrame.current=sampleTimeline(cueActions,elapsed,definitions,initial);
   if(elapsed<total)frame=requestAnimationFrame(tick);
  };
  tick(start);
  return()=>cancelAnimationFrame(frame);
 },[selected,behavior,mapping,definitions,state,replayKey,audioClock,interactionState]);
 // 说话口型使用独立轨道，仅在真实播放时按照波形能量开合。
 useEffect(()=>{
  if(!className.includes('home-live2d')){mouthFrame.current={};return;}
  if(state!=='ready'){mouthFrame.current={};return;}
  const action=mapping.actions.find(a=>a.id==='speaking');
  const track=action?.tracks.find(t=>t.parameterId==='ParamMouthOpenY');
  if(!track){mouthFrame.current={};return;}
  if(!speaking){mouthFrame.current={[track.parameterId]:track.from};return;}
  const range=definitions.find(d=>d.id===track.parameterId);
  // 以工作台映射的开合幅度为上限，并限制在模型参数范围内。
  const value=track.from+(track.to-track.from)*Math.min(1,Math.max(0,mouthLevel));
  mouthFrame.current={[track.parameterId]:range?Math.min(range.max,Math.max(range.min,value)):value};
  return()=>{mouthFrame.current={[track.parameterId]:track.from};};
 },[speaking,mouthLevel,mapping,definitions,state,className]);
 return <div className={`live2d-mount ${className}`} ref={mount}><span className={`renderer-badge ${state==='ready'?'renderer-ready':'renderer-fallback'}`}>{state==='ready'?'Live2D 实时模型':state==='loading'?'正在加载模型':state==='error'?'Live2D 加载失败':'缺少 Cubism Core'}</span></div>;
}

