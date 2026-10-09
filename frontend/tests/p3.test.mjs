/**
 * 作用：在无真实音频设备时验证播放器与音频线程的确定性边界。
 * 当前功能：
 * - 检查定向取消、排队回执、去重、失败降级和媒体时钟。
 * - 验证 AudioWorklet 的采样转换、连续帧与停止行为。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

/** 为生产播放器建立媒体、时钟和计时器替身。 */
function playerFixture() {
  const media = [], reports = [], timers = new Set();
  /** 模拟真实媒体元素的手动播放事件。 */
  class FakeAudio {
    /** 初始化音源并记录实例。 */
    constructor() { this.currentTime = 0; this.paused = true; media.push(this); }
    /** 模拟已接受播放请求，事件由测试显式触发。 */
    play() { this.paused = false; return Promise.resolve(); }
    /** 模拟同步停音。 */
    pause() { this.paused = true; }
    /** 提供释放接口。 */
    removeAttribute() {}
    /** 提供释放接口。 */
    load() {}
  }
  /** 提供无实际声卡的 Web Audio 节点。 */
  class FakeContext {
    /** 初始化目标端。 */
    constructor() { this.destination = {}; }
    /** 模拟用户手势解锁。 */
    resume() { return Promise.resolve(); }
    /** 模拟上下文关闭。 */
    close() { return Promise.resolve(); }
    /** 提供音源连接。 */
    createMediaElementSource() { return { connect() {}, disconnect() {} }; }
    /** 提供静音 RMS 采样。 */
    createAnalyser() { return { connect() {}, disconnect() {}, getByteTimeDomainData(samples) { samples.fill(128); } }; }
  }
  const source = fs.readFileSync(new URL('../src/audio/player.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {
    exports, Audio: FakeAudio, AudioContext: FakeContext, Blob, Uint8Array,
    URL: { createObjectURL() { return 'blob:test'; }, revokeObjectURL() {} },
    atob, performance, requestAnimationFrame() { return 1; }, cancelAnimationFrame() {},
    setInterval(callback) { timers.add(callback); return callback; }, clearInterval(callback) { timers.delete(callback); },
  });
  const player = new exports.AudioPlayer((state, segment, position) => reports.push({ state, ...segment, position }),
    () => {}, () => {}, () => {}, () => {});
  return { player, media, reports, timers };
}

/** 构造固定语段元数据。 */
function segment(generationId, segmentId) {
  return { generationId, segmentId, base64: 'UklGRg==', durationMs: 1000 };
}

test('取消确认包含当前进度和全部待播语段，晚到音频不能复活', () => {
  const { player, media, reports, timers } = playerFixture();
  player.enqueue(segment('old', 's1'));
  player.enqueue(segment('old', 's2'));
  media[0].onplaying();
  media[0].currentTime = .4;
  player.stop('old');
  assert.deepEqual(reports.filter(item => item.state === 'stopped').map(item => [item.segmentId, item.position]),
    [['s1', 400], ['s2', 0]]);
  assert.equal(media[0].paused, true);
  assert.equal(timers.size, 0);
  player.enqueue(segment('old', 'late'));
  assert.equal(media.length, 1);
  player.dispose();
});

test('旧 generation 的取消命令不能停止新音源', () => {
  const { player, media } = playerFixture();
  player.enqueue(segment('old', 's1'));
  player.stop('old');
  player.enqueue(segment('new', 's1'));
  media[1].onplaying();
  player.stop('old');
  assert.equal(media[1].paused, false);
  assert.equal(player.getClock().generationId, 'new');
  player.dispose();
});

test('重复语段只播放一次，解码失败不会报告正常结束', () => {
  const { player, media, reports } = playerFixture();
  player.enqueue(segment('g', 's1'));
  player.enqueue(segment('g', 's1'));
  player.enqueue(segment('g', 's2'));
  assert.equal(media.length, 1);
  media[0].onerror();
  assert.equal(reports.some(item => item.state === 'failed' && item.segmentId === 's1'), true);
  assert.equal(reports.some(item => item.state === 'ended'), false);
  assert.equal(media.length, 2);
  player.dispose();
});

test('动作时钟累计实际音频而非网络到达时间', () => {
  const { player, media } = playerFixture();
  player.enqueue(segment('g', 's1'));
  player.enqueue(segment('g', 's2'));
  media[0].currentTime = 1;
  media[0].onended();
  media[1].currentTime = .25;
  assert.equal(player.getClock().elapsedMs, 1250);
  player.dispose();
});

test('AudioWorklet 输出连续 20ms 小端 PCM，停止后不再发帧', () => {
  let Processor;
  const frames = [];
  /** 模拟音频线程端口。 */
  class WorkletBase {
    /** 保存发送帧，忽略跨线程转移参数。 */
    constructor() { this.port = { postMessage(frame) { frames.push(frame); } }; }
  }
  const sandbox = { AudioWorkletProcessor: WorkletBase, sampleRate: 48000, currentFrame: 0,
    Uint8Array, DataView, Math, registerProcessor(name, processor) { Processor = processor; } };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(new URL('../src/capture/pcmWorklet.js', import.meta.url), 'utf8'), sandbox);
  const processor = new Processor();
  processor.port.onmessage({ data: { type: 'start', startMs: 1000, audioTime: 0 } });
  for (let index = 0; index < 376; index++) {
    sandbox.currentFrame = index * 128;
    processor.process([[new Float32Array(128).fill(.5)]]);
  }
  assert.equal(frames.length, 50);
  frames.forEach((frame, index) => {
    assert.equal(frame.seq, index);
    assert.equal(frame.startMs, 1000 + index * 20);
    assert.equal(frame.endMs - frame.startMs, 20);
    assert.equal(frame.pcm.byteLength, 640);
    assert.equal(new DataView(frame.pcm.buffer).getInt16(0, true), 16384);
  });
  processor.port.onmessage({ data: { type: 'stop' } });
  processor.process([[new Float32Array(128)]]);
  assert.equal(frames.length, 50);
});
