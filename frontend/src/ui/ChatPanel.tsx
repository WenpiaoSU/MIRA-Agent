/**
 * 作用：提供用户与 MIRA 的文本对话界面。
 * 当前功能：
 * - 展示连接状态、历史消息和流式回复。
 * - 接收文字和麦克风输入，展示尚未提交的 ASR 转写。
 * - 录音时仅由麦克风图标控制采集，非录音时提供停止回复按钮。
 * - 支持持续倾听和 AEC 不可用时的单轮采集。
 */

import { useEffect, useRef, useState } from 'react';
import { Headphones, MessageCircle, Mic, Sparkles } from 'lucide-react';
import type { ChatMessage, ChatStatus } from '../transport/useChatSession';

/** 将连接状态映射为用户可见的中文提示。 */
const statusText: Record<ChatStatus, string> = {
  connecting: '正在连接 Agent',
  ready: '文本对话已连接',
  offline: 'Agent 服务未连接',
  unconfigured: '请先配置 LLM API',
};

/** 渲染对话列表、输入区和当前错误提示。 */
export default function ChatPanel({ messages, status, error, ttsReady, speechReady, speechBusy, recording, transcript, audioActive, duplexReady, send, stop, toggleMicrophone }: {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string;
  ttsReady: boolean;
  speechReady: boolean;
  speechBusy: boolean;
  recording: boolean;
  transcript: string;
  audioActive: boolean;
  duplexReady: boolean;
  send: (text: string) => boolean;
  stop: () => void;
  toggleMicrophone: () => void;
}) {
  const [draft, setDraft] = useState('');
  const bottom = useRef<HTMLDivElement>(null);
  const pending = messages.some(item => item.role === 'assistant' && item.pending);
  // 消息变化后把对话区域滚动到最新内容。
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' });
  }, [messages]);
  // 成功发送当前草稿后清空输入框。
  const submit = () => { if (send(draft)) setDraft(''); };

  return <section className="conversation card">
    <div className="section-heading"><div><h2>与 MIRA 对话</h2><p>{statusText[status]}</p></div><span className="demo-badge">实时对话</span></div>
    <div className="chat-body">
      <div className="assistant-line"><div className="tiny-mark"><Sparkles size={14}/></div><div className="bubble assistant-bubble">嗨，我是 MIRA。今天想聊些什么？</div></div>
      {messages.map(item => item.role === 'user'
        ? <div className="user-line" key={item.id}><div className="bubble user-bubble">{item.text}</div></div>
        : <div className="assistant-line" key={item.id}><div className="tiny-mark"><Sparkles size={14}/></div><div className="bubble assistant-bubble">{item.text || (item.pending ? '正在思考…' : '')}</div></div>)}
      {(recording || transcript) && <div className="user-line"><div className="bubble user-bubble asr-preview">{transcript || '正在听你说话…'}<small>{recording ? (duplexReady ? '持续倾听 · 再点麦克风结束' : '语音输入 · 再点麦克风结束') : '识别中'}</small></div></div>}
      <div ref={bottom}/>
    </div>
    {error && <div className="chat-error" role="alert">{error}</div>}
    <div className="chat-input"><button className={`input-tool ${recording ? 'recording' : ''}`} aria-pressed={recording} aria-label={recording ? '停止语音输入' : '开始语音输入'} title={recording ? '停止语音输入并提交剩余语音' : speechReady ? '开始语音输入' : '请配置 ASR API 并确认本地 VAD 可用'} disabled={status !== 'ready' || (!recording && !speechReady)} onClick={toggleMicrophone}><Mic size={18}/></button><input value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') submit(); }} disabled={status !== 'ready'} placeholder={status === 'ready' ? '输入消息…' : statusText[status]}/>{!recording && (pending || audioActive) && <button className="text-button" onClick={stop}>停止回复</button>}<button className="send-button" onClick={submit} disabled={status !== 'ready' || !draft.trim()} title="发送消息"><MessageCircle size={17}/></button></div>
    <div className="chat-foot"><span><Headphones size={13}/> {recording ? (duplexReady ? '语音模式 · 可插话' : '单轮语音') : speechReady ? '文字或语音输入' : '文字输入'} · {ttsReady ? '语音回复' : '文字回复'}</span><span>当前会话仅保存在服务进程内</span></div>
  </section>;
}

