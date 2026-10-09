/**
 * 作用：按实际媒体时钟播放、确认并取消完整 WAV 语段。
 * 当前功能：
 * - 去重语段，限制缓冲并区分正常结束、暂停、停止与失败。
 * - 按实际时钟反馈进度，能量口型与动作共用播放观测。
 * - 定向取消 generation，确认当前和所有排队语段，拒绝晚到音频。
 */

export type AudioSegment = { generationId: string; segmentId: string; base64: string; durationMs: number };
/** 实际播放器反馈，不以 failed 或 stopped 冒充 ended。 */
export type PlaybackState = 'started' | 'progress' | 'ended' | 'stopped' | 'failed' | 'paused';
/** 供动作时间线采样的整代音频时间，间隙及欠载时暂停推进。 */
export type PlaybackClock = { generationId: string; elapsedMs: number };

/** 管理唯一音源及有界待播语段。 */
export class AudioPlayer {
  private queue: AudioSegment[] = [];
  private current: AudioSegment | null = null;
  private audio: HTMLAudioElement | null = null;
  private context: AudioContext | null = null;
  private source: MediaElementAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private frame = 0;
  private progressTimer: ReturnType<typeof setInterval> | null = null;
  private url: string | null = null;
  private serial = 0;
  private lastLevel = 0;
  private seen = new Set<string>();
  private cancelled = new Set<string>();
  private clock: PlaybackClock | null = null;

  /** 注入交付、界面和故障回调。 */
  constructor(
    private feedback: (state: PlaybackState, segment: AudioSegment, playedMs: number) => void,
    private onActivity: (active: boolean) => void,
    private onSounding: (speaking: boolean) => void,
    private onLevel: (level: number) => void,
    private onError: (message: string) => void,
  ) {}

  /** 在用户手势内解锁音频播放上下文。 */
  unlock(): void {
    try {
      this.context ??= new AudioContext();
      void this.context.resume().catch(() => this.onError('浏览器未允许音频播放。'));
    } catch {
      this.onError('当前浏览器不支持语音播放。');
    }
  }

  /** 读取真实播放位置，供 Avatar 动作逐帧采样。 */
  getClock(): PlaybackClock | null {
    if (!this.clock) return null;
    return { ...this.clock, elapsedMs: this.clock.elapsedMs + (this.audio?.currentTime ?? 0) * 1000 };
  }

  /** 去重并追加语段，缓冲满时明确报告失败。 */
  enqueue(segment: AudioSegment): void {
    const key = segment.generationId + ':' + segment.segmentId;
    if (this.cancelled.has(segment.generationId) || this.seen.has(key)) return;
    this.seen.add(key);
    if (this.seen.size > 512) this.seen.delete(this.seen.values().next().value!);
    const buffered = this.queue.reduce((sum, item) => sum + item.durationMs, 0)
      + Math.max(0, (this.current?.durationMs ?? 0) - (this.audio?.currentTime ?? 0) * 1000);
    if (this.queue.length >= 12 || buffered + segment.durationMs > 60000) {
      this.onError('语音播放缓冲已满，本语段保留字幕。');
      this.feedback('failed', segment, 0);
      return;
    }
    this.queue.push(segment);
    this.onActivity(true);
    if (!this.current) this.playNext();
  }

  /** 同步清除指定 generation；旧取消命令不能停止新一轮音源。 */
  stop(generationId?: string): void {
    if (generationId) {
      this.cancelled.add(generationId);
      if (this.cancelled.size > 32) this.cancelled.delete(this.cancelled.values().next().value!);
    }
    const discarded = this.queue.filter(item => !generationId || item.generationId === generationId);
    this.queue = this.queue.filter(item => generationId && item.generationId !== generationId);
    if (this.current && (!generationId || this.current.generationId === generationId)) {
      const current = this.current;
      const playedMs = Math.round((this.audio?.currentTime ?? 0) * 1000);
      this.serial++;
      this.cleanup();
      this.current = null;
      this.clock = null;
      this.onSounding(false);
      this.onLevel(0);
      this.feedback('stopped', current, playedMs);
    }
    discarded.forEach(segment => this.feedback('stopped', segment, 0));
    this.onActivity(Boolean(this.current || this.queue.length));
    if (!this.current && this.queue.length) this.playNext();
  }

  /** 卸载时回收所有播放器资源与定时器。 */
  dispose(): void {
    this.stop();
    void this.context?.close();
    this.context = null;
  }

  /** 对单段失败回报真实位置，并继续下一段字幕对应的音频。 */
  private fail(serial: number, message: string): void {
    if (serial !== this.serial || !this.current) return;
    const current = this.current;
    const elapsed = (this.audio?.currentTime ?? 0) * 1000;
    this.feedback('failed', current, Math.round(elapsed));
    if (this.clock) this.clock.elapsedMs += elapsed;
    this.onError(message);
    this.serial++;
    this.cleanup();
    this.current = null;
    this.onSounding(false);
    this.onLevel(0);
    this.playNext();
  }

  /** 将下一语段接入 Web Audio，所有回调都检查音源代次。 */
  private playNext(): void {
    const segment = this.queue.shift();
    if (!segment) { this.onActivity(false); return; }
    this.current = segment;
    if (this.clock?.generationId !== segment.generationId) this.clock = { generationId: segment.generationId, elapsedMs: 0 };
    const serial = ++this.serial;
    try {
      const bytes = Uint8Array.from(atob(segment.base64), character => character.charCodeAt(0));
      this.url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
      const audio = new Audio(this.url);
      this.audio = audio;
      this.unlock();
      if (!this.context) throw new Error('音频上下文不可用');
      this.source = this.context.createMediaElementSource(audio);
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 1024;
      this.source.connect(this.analyser);
      this.analyser.connect(this.context.destination);
      audio.onplaying = () => {
        if (serial !== this.serial) return;
        this.feedback('started', segment, Math.round(audio.currentTime * 1000));
        this.onSounding(true);
        cancelAnimationFrame(this.frame);
        this.tick(serial);
        if (!this.progressTimer) this.progressTimer = setInterval(() => {
          if (serial === this.serial && !audio.paused) this.feedback('progress', segment, Math.round(audio.currentTime * 1000));
        }, 200);
      };
      audio.onwaiting = () => {
        if (serial !== this.serial) return;
        this.feedback('paused', segment, Math.round(audio.currentTime * 1000));
        cancelAnimationFrame(this.frame);
        this.onSounding(false);
        this.onLevel(0);
        this.lastLevel = 0;
      };
      audio.onended = () => {
        if (serial !== this.serial) return;
        this.feedback('ended', segment, Math.round(audio.currentTime * 1000));
        if (this.clock) this.clock.elapsedMs += audio.currentTime * 1000;
        this.cleanup();
        this.current = null;
        this.onSounding(false);
        this.onLevel(0);
        this.playNext();
      };
      audio.onerror = () => this.fail(serial, '语音音频无法解码，本语段保留字幕。');
      void audio.play().catch(() => this.fail(serial, '浏览器阻止了音频播放，请再次点击发送。'));
    } catch {
      this.fail(serial, '收到的音频格式不正确，本语段保留字幕。');
    }
  }

  /** 只在实际发声时采样 RMS 波形，嘴型不由生成文本驱动。 */
  private tick(serial: number): void {
    if (serial !== this.serial || !this.audio || this.audio.paused) return;
    if (this.analyser) {
      const samples = new Uint8Array(this.analyser.fftSize);
      this.analyser.getByteTimeDomainData(samples);
      let power = 0;
      for (const sample of samples) power += ((sample - 128) / 128) ** 2;
      const level = Math.min(1, Math.sqrt(power / samples.length) * 6);
      if (Math.abs(level - this.lastLevel) > .025) { this.lastLevel = level; this.onLevel(level); }
    }
    this.frame = requestAnimationFrame(() => this.tick(serial));
  }

  /** 清理当前音源、计时器、音频连接与对象 URL。 */
  private cleanup(): void {
    cancelAnimationFrame(this.frame);
    if (this.progressTimer) clearInterval(this.progressTimer);
    this.progressTimer = null;
    if (this.audio) {
      this.audio.onplaying = null;
      this.audio.onwaiting = null;
      this.audio.onended = null;
      this.audio.onerror = null;
      this.audio.pause();
      this.audio.removeAttribute('src');
      this.audio.load();
    }
    this.source?.disconnect();
    this.analyser?.disconnect();
    if (this.url) URL.revokeObjectURL(this.url);
    this.audio = null;
    this.source = null;
    this.analyser = null;
    this.url = null;
    this.lastLevel = 0;
  }
}

