/**
 * 作用：管理文本、持续语音、可打断输出和连接恢复。
 * 当前功能：
 * - 校验连接代次与服务端序号，去重字幕并拒绝取消后的晚到输出。
 * - 在实际 AEC 可用时持续采集，无法启用时使用半双工。
 * - 同步停音、清动作并发送最终位置和取消执行回执。
 * - 以实际播放开始与媒体时钟驱动角色，断线后有界重连。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { BehaviorPlan, PlaybackFeedbackPayload, ServerEvent, TtsChunkPayload } from '../contracts/realtime';
import { AudioPlayer, type AudioSegment, type PlaybackState, type PlaybackClock } from '../audio/player';
import { MicrophoneCapture, type CapturedFrame } from '../capture/microphone';

/** 对话界面的一条最终输入或流式输出。 */
export type ChatMessage = { id: string; role: 'user' | 'assistant'; text: string; pending?: boolean };
/** 连接与模型配置状态。 */
export type ChatStatus = 'connecting' | 'ready' | 'offline' | 'unconfigured';

/** 创建会话并提供语音、文本、停止和实际音频时钟。 */
export function useChatSession(onBehavior: (plan: BehaviorPlan | null) => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('connecting');
  const [error, setError] = useState('');
  const [ttsReady, setTtsReady] = useState(false);
  const [speechReady, setSpeechReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [speechBusy, setSpeechBusy] = useState(false);
  const [duplexReady, setDuplexReady] = useState(false);
  const [interactionState, setInteractionState] = useState('idle');
  const [speaking, setSpeaking] = useState(false);
  const [mouthLevel, setMouthLevel] = useState(0);
  const [audioActive, setAudioActive] = useState(false);
  const socket = useRef<WebSocket | null>(null);
  const sessionId = useRef<string | null>(null);
  const connectionEpoch = useRef(0);
  const nextSeq = useRef(0);
  const serverSeq = useRef(-1);
  const activeTurnId = useRef<string | null>(null);
  const activeGenerationId = useRef<string | null>(null);
  const player = useRef<AudioPlayer | null>(null);
  const microphone = useRef<MicrophoneCapture | null>(null);
  const audioStreamId = useRef<string | null>(null);
  const continuous = useRef(false);
  const serverDuplex = useRef(false);
  const pendingBehavior = useRef<BehaviorPlan | null>(null);
  const currentBehavior = useRef<BehaviorPlan | null>(null);
  const cancelled = useRef(new Set<string>());
  const displayed = useRef(new Set<string>());
  const pendingDisplay = useRef(new Map<string, { messageId: string; generationId: string; segmentId: string }>());
  const traceId = useRef<string>(crypto.randomUUID());
  const behaviorCallback = useRef(onBehavior);
  behaviorCallback.current = onBehavior;

  /** 为当前连接构造事件；同轮输出继承 trace_id。 */
  const envelope = useCallback((type: string, payload: Record<string, unknown>) => {
    const now = Math.round(performance.now());
    return { schema_version: '1.0', event_id: crypto.randomUUID(), type, source: 'avatar-web',
      session_id: sessionId.current, connection_epoch: connectionEpoch.current, seq: nextSeq.current++,
      occurred_at: new Date().toISOString(), time_ref: { clock_id: 'browser-performance',
      capture_start_ms: now, capture_end_ms: now }, trace_id: traceId.current, causation_id: null, payload };
  }, []);

  /** 仅向当前已建立的连接发送控制消息。 */
  const sendEvent = useCallback((type: string, payload: Record<string, unknown>) => {
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify(envelope(type, payload)));
  }, [envelope]);

  /** React 提交字幕后再确认展示，避免把仅入队的更新记成已显示。 */
  useEffect(() => {
    for (const [key, receipt] of pendingDisplay.current) {
      if (cancelled.current.has(receipt.generationId)) {
        pendingDisplay.current.delete(key);
      } else if (messages.some(item => item.id === receipt.messageId && item.text)) {
        sendEvent('response.displayed', { generation_id: receipt.generationId, segment_id: receipt.segmentId });
        pendingDisplay.current.delete(key);
      }
    }
  }, [messages, sendEvent]);

  /** 定向停音，旧输出取消不影响新音源。 */
  const stopAudio = useCallback((generationId?: string) => {
    player.current?.stop(generationId);
    if (!generationId || generationId === activeGenerationId.current) {
      setSpeaking(false);
      setMouthLevel(0);
      setAudioActive(false);
    }
  }, []);

  /** 关闭麦克风并明确通知服务器提交或丢弃残余音频。 */
  const stopMicrophone = useCallback((submit: boolean) => {
    const streamId = audioStreamId.current;
    audioStreamId.current = null;
    continuous.current = false;
    microphone.current?.stop();
    microphone.current = null;
    setRecording(false);
    if (streamId) sendEvent('input.audio_end', { stream_id: streamId, submit });
  }, [sendEvent]);

  /** 编码固定音频包，主线程或网络积压时明确结束采集。 */
  const sendAudioFrame = useCallback((frame: CapturedFrame, streamId: string) => {
    const ws = socket.current;
    if (audioStreamId.current !== streamId || ws?.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > 256000 || performance.now() - frame.endMs > 1000) {
      setError('语音上传积压，已结束采集，请重新开启麦克风。');
      stopMicrophone(false);
      return;
    }
    const header = new TextEncoder().encode(JSON.stringify(envelope('input.audio_frame', {
      stream_id: streamId, frame_seq: frame.seq, capture_start_ms: frame.startMs, capture_end_ms: frame.endMs,
    })));
    const packet = new Uint8Array(2 + header.length + frame.pcm.length);
    new DataView(packet.buffer).setUint16(0, header.length, false);
    packet.set(header, 2);
    packet.set(frame.pcm, 2 + header.length);
    ws.send(packet);
  }, [envelope, stopMicrophone]);

  /** 停止反馈允许属于被取消旧代次；当前音源起点解锁对应动作。 */
  const reportPlayback = useCallback((state: PlaybackState, segment: AudioSegment, playedMs: number) => {
    if (socket.current?.readyState !== WebSocket.OPEN) return;
    const payload: PlaybackFeedbackPayload = { generation_id: segment.generationId, segment_id: segment.segmentId,
      played_ms: playedMs, duration_ms: segment.durationMs, clock_ms: Math.round(performance.now()) };
    sendEvent('playback.' + state, payload);
    if (state === 'started' && segment.generationId === activeGenerationId.current) {
      const plan = pendingBehavior.current;
      if (plan?.generation_id === segment.generationId) {
        pendingBehavior.current = null;
        behaviorCallback.current(plan);
      }
    }
  }, [sendEvent]);

  /** 为 Avatar 提供稳定读取函数，避免进度更新重启动作 effect。 */
  const audioClock = useCallback((): PlaybackClock | null => player.current?.getClock() ?? null, []);

  // 生命周期只创建一个播放器；关闭、接管和重连均清除旧采集与输出。
  useEffect(() => {
    let disposed = false;
    let attempts = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const controller = new AbortController();
    player.current = new AudioPlayer(reportPlayback, setAudioActive, setSpeaking, setMouthLevel, setError);

    /** 清理当前连接的所有本地执行事实，不自动恢复旧队列或麦克风权限。 */
    const resetConnection = () => {
      stopAudio();
      stopMicrophone(false);
      activeGenerationId.current = null;
      activeTurnId.current = null;
      pendingBehavior.current = null;
      currentBehavior.current = null;
      behaviorCallback.current(null);
      setSpeechBusy(false);
      pendingDisplay.current.clear();
      setInteractionState('disconnected');
      setTranscript('');
      setMessages(previous => previous.map(item => ({ ...item, pending: false })));
    };

    /** 有界重连保持会话身份，服务器丢失会话时重新创建。 */
    const reconnect = () => {
      if (disposed || attempts >= 3) {
        if (!disposed) { setStatus('offline'); setError('连接中断，重新连接失败，请刷新页面。'); }
        return;
      }
      setStatus('connecting');
      reconnectTimer = setTimeout(() => { attempts++; void connect(); }, Math.min(8000, 1000 * 2 ** attempts));
    };

    /** 建立连接并安装仅属于当前 WebSocket 的事件处理。 */
    const connect = async () => {
      try {
        if (!sessionId.current) {
          const response = await fetch('/v1/sessions', { method: 'POST', signal: controller.signal });
          if (!response.ok) throw new Error('无法创建会话');
          sessionId.current = (await response.json() as { session_id: string }).session_id;
        }
        if (disposed) return;
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(scheme + '//' + window.location.host + '/v1/sessions/' + sessionId.current + '/realtime');
        socket.current = ws;
        serverSeq.current = -1;

        /** 验证事件所属连接和单调序号，再转交具体交付处理。 */
        ws.onmessage = message => {
          if (disposed || socket.current !== ws) return;
          let event: ServerEvent;
          try { event = JSON.parse(message.data) as ServerEvent; } catch { setError('服务器返回了无效事件。'); return; }
          if (event.session_id !== sessionId.current || !Number.isInteger(event.seq) || event.seq <= serverSeq.current) return;
          if (event.type === 'session.status') {
            connectionEpoch.current = event.connection_epoch;
            serverSeq.current = event.seq;
            nextSeq.current = 0;
            attempts = 0;
            setStatus(event.payload.llm_ready ? 'ready' : 'unconfigured');
            setTtsReady(event.payload.tts_ready === true);
            setSpeechReady(event.payload.speech_ready === true);
            serverDuplex.current = event.payload.full_duplex_supported === true;
            setInteractionState('idle');
            return;
          }
          if (event.connection_epoch !== connectionEpoch.current) return;
          serverSeq.current = event.seq;
          const generationId = event.generation_id;
          if (event.trace_id) traceId.current = event.trace_id;
          if (generationId && cancelled.current.has(generationId)
              && ['behavior.plan', 'response.segment', 'response.completed', 'tts.chunk'].includes(event.type)) return;

          if (event.type === 'output.cancel' && generationId) {
            const received = performance.now();
            cancelled.current.add(generationId);
            if (cancelled.current.size > 32) cancelled.current.delete(cancelled.current.values().next().value!);
            stopAudio(generationId);
            if (generationId === activeGenerationId.current) {
              activeGenerationId.current = null;
              activeTurnId.current = null;
              pendingBehavior.current = null;
              currentBehavior.current = null;
              behaviorCallback.current(null);
              setSpeechBusy(false);
              setMessages(previous => previous.map(item => ({ ...item, pending: false })));
            }
            const stopped = Math.round(performance.now());
            sendEvent('command.ack', { command_id: event.payload.command_id, generation_id: generationId,
              status: 'completed', clock_ms: stopped, stop_elapsed_ms: Math.max(0, stopped - Math.round(received)),
              speech_to_stop_ms: typeof event.payload.speech_start_ms === 'number'
                ? Math.max(0, stopped - event.payload.speech_start_ms) : null });
            return;
          }
          if (event.type === 'turn.controlled' && event.turn_id === activeTurnId.current) {
            activeTurnId.current = null;
            setSpeechBusy(false);
            setMessages(previous => previous.filter(item => item.id !== 'assistant:' + event.turn_id || item.text)
              .map(item => ({ ...item, pending: false })));
          }
          if (event.type === 'turn.state') {
            const state = String(event.payload.state ?? 'idle');
            if (state !== 'received') setInteractionState(state);
            if (state === 'received' && event.turn_id === activeTurnId.current) {
              activeGenerationId.current = generationId;
              setSpeechBusy(true);
            }
            if (event.payload.audio_ready === true && event.payload.stream_id === audioStreamId.current && microphone.current) {
              const id = audioStreamId.current!;
              continuous.current = event.payload.continuous === true;
              microphone.current.start(frame => sendAudioFrame(frame, id));
              setTranscript('');
              player.current?.unlock();
            }
          }
          if (event.type === 'audio.closed' && event.payload.stream_id === audioStreamId.current) {
            stopMicrophone(false);
            setTranscript('');
          }
          if (event.type === 'asr.partial' || event.type === 'asr.final') setTranscript(String(event.payload.text ?? ''));
          if (event.type === 'vad.event' && event.payload.phase === 'start') setTranscript('正在听你说话…');
          if (event.type === 'turn.completed' && event.turn_id) {
            const text = String(event.payload.text ?? '').trim();
            if (!text) return;
            // 最终新输入先切断旧输出归属，随后迟到的旧取消不能清除新 turn_id。
            const old = activeGenerationId.current;
            if (old) cancelled.current.add(old);
            stopAudio(old ?? undefined);
            activeGenerationId.current = null;
            if (!continuous.current) stopMicrophone(false);
            activeTurnId.current = event.turn_id;
            setTranscript('');
            setSpeechBusy(true);
            pendingBehavior.current = null;
            currentBehavior.current = null;
            behaviorCallback.current(null);
            setMessages(previous => [...previous.map(item => ({ ...item, pending: false })),
              { id: 'user:' + event.turn_id, role: 'user', text },
              { id: 'assistant:' + event.turn_id, role: 'assistant', text: '', pending: true }]);
          }
          if (event.type === 'behavior.plan' && generationId === activeGenerationId.current) {
            const plan = event.payload as unknown as BehaviorPlan;
            if (plan.generation_id !== generationId) return;
            currentBehavior.current = plan;
            if (plan.timing.anchor === 'speech_start') pendingBehavior.current = plan;
            else behaviorCallback.current(plan);
          }
          if (event.type === 'response.segment' && generationId === activeGenerationId.current && event.turn_id) {
            const segmentId = String(event.payload.segment_id);
            const key = generationId + ':' + segmentId;
            if (displayed.current.has(key)) return;
            displayed.current.add(key);
            if (displayed.current.size > 512) displayed.current.delete(displayed.current.values().next().value!);
            const text = String(event.payload.text ?? '');
            setMessages(previous => previous.map(item => item.id === 'assistant:' + event.turn_id
              ? { ...item, text: item.text + text, pending: true } : item));
            pendingDisplay.current.set(key, { messageId: 'assistant:' + event.turn_id, generationId: generationId!, segmentId });
          }
          if (event.type === 'tts.chunk' && generationId === activeGenerationId.current) {
            const chunk = event.payload as unknown as TtsChunkPayload;
            if (chunk.generation_id === generationId && typeof chunk.segment_id === 'string'
                && typeof chunk.audio_base64 === 'string' && chunk.encoding === 'wav'
                && chunk.chunk_seq === 0 && chunk.is_last && Number.isFinite(chunk.duration_ms) && chunk.duration_ms > 0) {
              player.current?.enqueue({ generationId: generationId!, segmentId: chunk.segment_id,
                base64: chunk.audio_base64, durationMs: chunk.duration_ms });
            }
          }
          if (event.type === 'response.completed' && generationId === activeGenerationId.current) {
            setMessages(previous => previous.map(item => item.id === 'assistant:' + event.turn_id ? { ...item, pending: false } : item));
          }
          if (event.type === 'turn.delivered' && generationId === activeGenerationId.current) {
            activeGenerationId.current = null;
            activeTurnId.current = null;
            setSpeechBusy(false);
            // 输出进入终态后保留表情和视线，收起未走完的短句手势。
            const completed = currentBehavior.current;
            if (completed) behaviorCallback.current({ ...completed, timing: { ...completed.timing, anchor: 'plan_ready' }, gestures: [] });
            pendingBehavior.current = null;
          }
          if (event.type === 'tts.error' && generationId === activeGenerationId.current) setError(String(event.payload.message));
          if (event.type === 'error') {
            const code = String(event.payload.code ?? '');
            if (['audio_unavailable', 'invalid_audio', 'speech_error', 'asr_error', 'audio_timeout'].includes(code)) stopMicrophone(false);
            setError(String(event.payload.message ?? '请求失败'));
            if (generationId === activeGenerationId.current) {
              stopAudio(generationId!);
              setSpeechBusy(false);
              setMessages(previous => previous.map(item => ({ ...item, pending: false })));
            }
          }
        };
        ws.onclose = event => {
          if (disposed || socket.current !== ws) return;
          resetConnection();
          if (event.code === 1008) sessionId.current = null;
          reconnect();
        };
        ws.onerror = () => {
          if (!disposed && socket.current === ws) setError('Agent 连接异常，正在尝试恢复。');
        };
      } catch (cause) {
        if (!disposed && !controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : '无法连接 Agent');
          reconnect();
        }
      }
    };
    void connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      controller.abort();
      stopMicrophone(false);
      player.current?.dispose();
      player.current = null;
      socket.current?.close();
      if (sessionId.current) void fetch('/v1/sessions/' + sessionId.current, { method: 'DELETE', keepalive: true }).catch(() => {});
    };
  }, [reportPlayback, sendAudioFrame, sendEvent, stopAudio, stopMicrophone]);

  /** 文本提交先在本地停音，再由服务端确认并建立新代次。 */
  const send = useCallback((text: string) => {
    const value = text.trim();
    if (!value || socket.current?.readyState !== WebSocket.OPEN || status !== 'ready') return false;
    const old = activeGenerationId.current;
    if (old) cancelled.current.add(old);
    stopAudio(old ?? undefined);
    stopMicrophone(false);
    activeGenerationId.current = null;
    if (ttsReady) player.current?.unlock();
    const turnId = crypto.randomUUID();
    traceId.current = turnId;
    activeTurnId.current = turnId;
    pendingBehavior.current = null;
    setSpeechBusy(true);
    setError('');
    currentBehavior.current = null;
    behaviorCallback.current(null);
    setMessages(previous => [...previous.map(item => ({ ...item, pending: false })),
      { id: 'user:' + turnId, role: 'user', text: value },
      { id: 'assistant:' + turnId, role: 'assistant', text: '', pending: true }]);
    socket.current.send(JSON.stringify({ ...envelope('input.text', { text: value }), turn_id: turnId }));
    return true;
  }, [status, ttsReady, envelope, stopAudio, stopMicrophone]);

  /** 一次点击进入语音模式；AEC 不可用时只开放单轮半双工采集。 */
  const toggleMicrophone = useCallback(() => {
    if (microphone.current) { stopMicrophone(true); return; }
    if (status !== 'ready' || !speechReady || socket.current?.readyState !== WebSocket.OPEN) return;
    const capture = new MicrophoneCapture(message => { setError(message); stopMicrophone(false); });
    microphone.current = capture;
    setRecording(true);
    setTranscript('正在准备麦克风…');
    setError('');
    player.current?.unlock();
    void capture.prepare().then(() => {
      if (microphone.current !== capture || socket.current?.readyState !== WebSocket.OPEN) { capture.stop(); return; }
      const capabilities = { ...capture.capabilities, continuous: capture.capabilities.continuous && serverDuplex.current };
      setDuplexReady(capabilities.continuous);
      if (!capabilities.continuous && activeGenerationId.current) {
        setError('当前使用单轮语音模式，请等待回复结束后使用语音。');
        stopMicrophone(false);
        return;
      }
      const id = crypto.randomUUID();
      audioStreamId.current = id;
      continuous.current = capabilities.continuous;
      sendEvent('input.audio_start', { stream_id: id, encoding: 'pcm_s16le', sample_rate: 16000,
        channels: 1, ...capabilities });
      if (!capabilities.continuous) setError('当前使用单轮语音模式，回复时暂停采集。');
    }).catch(cause => {
      if (microphone.current !== capture) return;
      stopMicrophone(false);
      setTranscript('');
      setError(cause instanceof Error ? cause.message : '麦克风不可用');
    });
  }, [status, speechReady, sendEvent, stopMicrophone]);

  /** 停止当前回复；持续语音模式保持倾听，麦克风按钮负责结束采集。 */
  const stop = useCallback(() => {
    const generationId = activeGenerationId.current;
    if (generationId) cancelled.current.add(generationId);
    stopAudio(generationId ?? undefined);
    sendEvent('control.stop', { generation_id: generationId });
    activeTurnId.current = null;
    activeGenerationId.current = null;
    pendingBehavior.current = null;
    setSpeechBusy(false);
    currentBehavior.current = null;
    behaviorCallback.current(null);
    setMessages(previous => previous.map(item => ({ ...item, pending: false })));
  }, [sendEvent, stopAudio]);

  return { messages, status, error, ttsReady, speechReady, speechBusy, recording, transcript, duplexReady,
    interactionState, speaking, mouthLevel, audioActive, audioClock, send, stop, toggleMicrophone };
}



