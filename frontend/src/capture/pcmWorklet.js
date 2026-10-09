/**
 * 作用：在音频线程中将麦克风采样转换为连续 PCM 帧。
 * 当前功能：
 * - 将设备采样率重采样至 16 kHz，并每 320 样本发送一帧。
 * - 用音频采样时钟映射浏览器采集时间，持续运行时保持序号连续。
 * - 支持显式启动和停止，传输 PCM 时转移缓冲区所有权。
 */

/** 管理一条音频线程中的重采样与分帧状态。 */
class MiraPcmProcessor extends AudioWorkletProcessor {
  /** 初始化端口控制与跨渲染块的分帧缓存。 */
  constructor() {
    super();
    this.active = false;
    this.pending = [];
    this.cursor = 0;
    this.samples = [];
    this.seq = 0;
    this.baseMs = null;
    this.port.onmessage = event => {
      if (event.data.type === 'start') {
        this.active = true;
        this.pending = [];
        this.samples = [];
        this.cursor = 0;
        this.seq = 0;
        this.baseMs = null;
        this.startMs = event.data.startMs;
        this.startAudioTime = event.data.audioTime;
      } else if (event.data.type === 'stop') {
        this.active = false;
      }
    };
  }

  /** 消费实际采样块，保留插值余量并产生标准二进制帧。 */
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!this.active || !input) return true;
    if (this.baseMs === null) {
      this.baseMs = Math.max(0, Math.round(this.startMs + (currentFrame / sampleRate - this.startAudioTime) * 1000));
    }
    for (const value of input) this.pending.push(value);
    const step = sampleRate / 16000;
    while (this.cursor + 1 < this.pending.length) {
      const index = Math.floor(this.cursor);
      const fraction = this.cursor - index;
      const value = this.pending[index] * (1 - fraction) + this.pending[index + 1] * fraction;
      this.cursor += step;
      this.samples.push(Math.max(-32768, Math.min(32767, Math.round(value * (value < 0 ? 32768 : 32767)))));
      if (this.samples.length === 320) {
        const pcm = new Uint8Array(640);
        const view = new DataView(pcm.buffer);
        this.samples.forEach((sample, offset) => view.setInt16(offset * 2, sample, true));
        const seq = this.seq++;
        this.port.postMessage({ seq, startMs: this.baseMs + seq * 20,
          endMs: this.baseMs + (seq + 1) * 20, pcm }, [pcm.buffer]);
        this.samples = [];
      }
    }
    const consumed = Math.min(Math.floor(this.cursor), this.pending.length - 1);
    if (consumed > 0) {
      this.pending.splice(0, consumed);
      this.cursor -= consumed;
    }
    return true;
  }
}

registerProcessor('mira-pcm', MiraPcmProcessor);
