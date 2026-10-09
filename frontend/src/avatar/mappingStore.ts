/**
 * 作用：加载、保存和导出 Avatar 动作映射。
 * 当前功能：
 * - 优先读取项目公开映射文件，失败时回退到浏览器副本或默认值。
 * - 校验基本结构、移除旧动作并支持本地保存及 JSON 导出。
 */

import { defaultMapping } from './defaultMapping';
import type { AvatarMapping } from './types';
/** 浏览器本地存储中的映射键名。 */
const KEY='mira.avatar.mapping.v1';
/** 检查载入值是否具备当前映射版本和动作列表。 */
function validMapping(value:unknown):value is AvatarMapping { const x=value as AvatarMapping|null;return Boolean(x&&x.schemaVersion==='1.0'&&Array.isArray(x.actions)); }
/** 清理旧版本字段和已停用的动作。 */
function cleanMapping(mapping:AvatarMapping):AvatarMapping { const {mappingVersion:_legacyVersion,...rest}=mapping as AvatarMapping&{mappingVersion?:number};return {...rest,actions:mapping.actions.filter(a=>a.id!=='excited')}; }
/** 读取并校验浏览器保存的映射副本。 */
function loadStoredMapping():AvatarMapping|undefined { try { const raw=localStorage.getItem(KEY);if(!raw)return;const x=JSON.parse(raw);if(validMapping(x))return cleanMapping(x); } catch {} }
/** 优先读取项目中的映射 JSON，必要时使用浏览器副本或内置配置。 */
export async function loadMapping():Promise<AvatarMapping> {
 try { const response=await fetch('/map/avatar-mapping.json',{cache:'no-store'});if(response.ok){const value:unknown=await response.json();if(validMapping(value))return cleanMapping(value);console.error('Invalid avatar mapping at /map/avatar-mapping.json');} }
 catch(error){console.warn('Could not read /map/avatar-mapping.json; using browser copy if available.',error);}
 return loadStoredMapping()??structuredClone(defaultMapping);
}
/** 更新保存时间，并把映射写入浏览器本地存储。 */
export function saveMapping(mapping:AvatarMapping):AvatarMapping { const saved={...cleanMapping(mapping),updatedAt:new Date().toISOString()};localStorage.setItem(KEY,JSON.stringify(saved,null,2));return saved; }
/** 生成可下载的映射 JSON 文件。 */
export function exportMapping(mapping:AvatarMapping){const blob=new Blob([JSON.stringify(cleanMapping(mapping),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='avatar-mapping.json';a.click();URL.revokeObjectURL(a.href);}
