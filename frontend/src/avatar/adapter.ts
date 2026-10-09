/**
 * 作用：把语义动作映射为 Live2D 模型参数值。
 * 当前功能：
 * - 按强度计算并限幅动作目标值。
 * - 按曲线进度采样单个动作的参数帧。
 */

import type { AvatarAction, AvatarMapping, ParameterDefinition } from './types';
/** 参数标识到数值的映射。 */
export type ParameterFrame=Record<string,number>;
/** 根据动作映射和强度计算目标参数，并限制在模型允许范围内。 */
export function compileAvatarAction(mapping:AvatarMapping,actionId:string,definitions:ParameterDefinition[],intensity=1):ParameterFrame {
 const action=mapping.actions.find(x=>x.id===actionId); if(!action)return {};
 const byId=new Map(definitions.map(x=>[x.id,x])); const out:ParameterFrame={};
 for(const track of action.tracks){const d=byId.get(track.parameterId);if(!d)continue;const target=track.from+(track.to-track.from)*Math.max(0,Math.min(1,intensity));out[track.parameterId]=Math.max(d.min,Math.min(d.max,target));}
 return out;
}
/** 按指定进度采样单个动作的参数轨道。 */
export function sampleAction(action:AvatarAction,progress:number):ParameterFrame {
 const p=Math.max(0,Math.min(1,progress)); const ease=(x:number)=>x*x*(3-2*x); const out:ParameterFrame={};
 for(const t of action.tracks){let x=p;if(t.curve==='ease-in')x=p*p;if(t.curve==='ease-out')x=1-(1-p)*(1-p);if(t.curve==='smooth'||t.curve==='spring')x=ease(p);out[t.parameterId]=t.from+(t.to-t.from)*x;}
 return out;
}
