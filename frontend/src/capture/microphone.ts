/**
 * 作用：准备浏览器麦克风并上传持续、标准化的音频帧。
 * 当前功能：
 * - 请求浏览器 AEC，并读取实际音轨设置声明双工能力。
 * - 使用 AudioWorklet 重采样和分帧，避免主线程音频处理。
 * - 在服务端就绪后开始采集，处理设备断流与资源释放。
 */

export type CapturedFrame = { seq: number; startMs: number; endMs: number; pcm: Uint8Array };
/** 仅声明实际启用的 AEC；该设置不代替设备回声效果验收。 */
export type CaptureCapabilities = { aec_mode: 'browser' | 'unavailable'; continuous: boolean };

/** 为一条音频流管理设备、音频线程和连续帧。 */
export class MicrophoneCapture {
  private media: MediaStream | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: AudioWorkletNode | null = null;
  private sink: GainNode | null = null;
  private capturing = false;
  private onFrame: ((frame: CapturedFrame) => void) | null = null;
  capabilities: CaptureCapabilities = { aec_mode: 'unavailable', continuous: false };

  /** 注入设备断流回调，避免采集停止后界面继续显示录音。 */
  constructor(private onError: (message: string) => void = () => {}) {}

  /** 在用户手势内解锁音频上下文，随后读取实际设备设置。 */
  async prepare(): Promise<void> {
    const context = new AudioContext();
    this.context = context;
    const resumed = context.resume();
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
        video: false,
      });
      if (this.context !== context) {
        media.getTracks().forEach(track => track.stop());
        throw new Error('录音已取消');
      }
      this.media = media;
      const track = media.getAudioTracks()[0];
      const aec = track.getSettings().echoCancellation === true;
      this.capabilities = { aec_mode: aec ? 'browser' : 'unavailable', continuous: aec };
      track.onended = () => this.onError('麦克风连接已中断，请重新开启。');
      if (!context.audioWorklet) throw new Error('当前浏览器不支持持续音频采集，请使用支持 AudioWorklet 的浏览器。');
      await context.audioWorklet.addModule(new URL('./pcmWorklet.js', import.meta.url));
      if (this.context !== context) throw new Error('录音已取消');
      this.source = context.createMediaStreamSource(media);
      this.processor = new AudioWorkletNode(context, 'mira-pcm');
      this.processor.port.onmessage = event => {
        if (this.capturing && this.onFrame) this.onFrame(event.data as CapturedFrame);
      };
      this.processor.onprocessorerror = () => this.onError('音频采集线程异常，请重新开启麦克风。');
      this.sink = context.createGain();
      this.sink.gain.value = 0;
      this.source.connect(this.processor);
      this.processor.connect(this.sink);
      this.sink.connect(context.destination);
      await resumed;
    } catch (error) {
      this.stop();
      throw error;
    }
  }

  /** 仅启动一次采集，后续倾听状态事件不能重置帧序号。 */
  start(onFrame: (frame: CapturedFrame) => void): void {
    if (this.capturing) return;
    if (!this.context || !this.processor || this.context.state !== 'running') throw new Error('麦克风尚未就绪');
    this.onFrame = onFrame;
    this.capturing = true;
    this.processor.port.postMessage({ type: 'start', startMs: performance.now(), audioTime: this.context.currentTime });
  }

  /** 清除回调、停止线程并释放设备，可重复调用。 */
  stop(): void {
    this.capturing = false;
    this.onFrame = null;
    if (this.processor) {
      this.processor.port.postMessage({ type: 'stop' });
      this.processor.port.onmessage = null;
      this.processor.onprocessorerror = null;
      this.processor.disconnect();
    }
    this.source?.disconnect();
    this.sink?.disconnect();
    this.media?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    void this.context?.close();
    this.media = null;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.sink = null;
  }
}
