/**
 * 作用：在无浏览器设备时验证生产会话 Hook 的提醒交付边界。
 * 当前功能：
 * - 检查提醒重发按编号去重，React 提交后才确认展示。
 * - 检查提醒语音复用播放器、可停止，且不能接管本地新输入。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { webcrypto } from 'node:crypto';
import ts from 'typescript';

/** 用真实 Hook 和最小生命周期替身运行网络消息与 React 提交。 */
async function fixture() {
  const slots = [], effects = [], sockets = [], sent = [], audio = [], stops = [];
  let cursor = 0, sequence = 0, result;
  /** 比较依赖，避免测试替身人为重建连接。 */
  function changed(previous, next) {
    return !previous || previous.length !== next.length || previous.some((value, index) => !Object.is(value, next[index]));
  }
  const react = {
    /** 保存跨渲染引用。 */
    useRef(value) { const index = cursor++; return slots[index] ??= { current: value }; },
    /** 模拟 React 状态入队，只有显式 render 才提交 effects。 */
    useState(value) {
      const index = cursor++;
      slots[index] ??= { value };
      return [slots[index].value, update => { slots[index].value = typeof update === 'function' ? update(slots[index].value) : update; }];
    },
    /** 保留依赖稳定的回调引用。 */
    useCallback(callback, dependencies) {
      const index = cursor++;
      if (!slots[index] || changed(slots[index].dependencies, dependencies)) slots[index] = { callback, dependencies };
      return slots[index].callback;
    },
    /** 将生命周期副作用延迟到当前渲染完成。 */
    useEffect(callback, dependencies) {
      const index = cursor++;
      if (!slots[index] || changed(slots[index].dependencies, dependencies)) {
        const prior = slots[index];
        slots[index] = { dependencies };
        effects.push(() => { prior?.cleanup?.(); slots[index].cleanup = callback(); });
      }
    },
  };
  /** 记录经过生产 Hook 校验后的播放器操作。 */
  class FakePlayer {
    /** 记录待播放语段。 */
    enqueue(segment) { audio.push(segment); }
    /** 记录定向停止。 */
    stop(generation) { stops.push(generation); }
    /** 提供用户手势解锁入口。 */
    unlock() {}
    /** 提供生命周期释放入口。 */
    dispose() {}
  }
  /** 模拟网络连接，保留由生产 Hook 设置的所有回调。 */
  class FakeSocket {
    static OPEN = 1;
    /** 自动完成连接，事件由测试显式送入。 */
    constructor() { this.readyState = 1; sockets.push(this); }
    /** 记录客户端真正提交到网络的信封。 */
    send(value) { sent.push(JSON.parse(value)); }
    /** 清理测试连接。 */
    close() { this.readyState = 3; }
  }
  const source = fs.readFileSync(new URL('../src/transport/useChatSession.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {
    exports,
    /** 仅替换设备和 React 运行时，保留生产网络处理逻辑。 */
    require(name) {
      if (name === 'react') return react;
      if (name.includes('audio/player')) return { AudioPlayer: FakePlayer };
      if (name.includes('capture/microphone')) return { MicrophoneCapture: class {} };
      throw new Error('意外依赖：' + name);
    },
    crypto: webcrypto, performance, WebSocket: FakeSocket, AbortController,
    window: { location: { protocol: 'http:', host: 'localhost' } },
    /** 返回已有浏览器身份下的新会话。 */
    async fetch() { return { ok: true, async json() { return { session_id: 'session' }; } }; },
    setTimeout, clearTimeout,
  });
  /** 提交一次 React 渲染，再执行展示回执等副作用。 */
  function render() {
    cursor = 0;
    result = exports.useChatSession(() => {});
    while (effects.length) effects.shift()();
    return result;
  }
  /** 送入带单调序号的服务器信封。 */
  function receive(type, payload, generationId = null, turnId = null) {
    sockets.at(-1).onmessage({ data: JSON.stringify({ type, payload, session_id: 'session', connection_epoch: 1,
      seq: sequence++, generation_id: generationId, turn_id: turnId }) });
  }
  render();
  await new Promise(resolve => setImmediate(resolve));
  receive('session.status', { llm_ready: true, tts_ready: true });
  render();
  return {
    render, receive, sent, audio, stops,
    /** 卸载 Hook，回收连接与播放器。 */
    close() { for (const slot of slots) slot?.cleanup?.(); },
  };
}

test('到期通知在提交界面后确认，同编号重发只保留一行并确认新凭据', async () => {
  const app = await fixture();
  try {
    app.receive('reminder.due', { reminder_id: 'r1', delivery_token: 'first', text: '提醒：休息。' });
    assert.equal(app.sent.filter(event => event.type === 'reminder.displayed').length, 0);
    const initial = app.render();
    assert.equal(initial.messages.length, 1);
    assert.equal(initial.messages[0].text, '提醒：休息。');
    assert.equal(app.sent.at(-1).payload.delivery_token, 'first');
    app.receive('reminder.due', { reminder_id: 'r1', delivery_token: 'retry', text: '提醒：休息。' });
    assert.equal(app.sent.at(-1).payload.delivery_token, 'first');
    assert.equal(app.render().messages.length, 1);
    assert.equal(app.sent.at(-1).payload.delivery_token, 'retry');
  } finally { app.close(); }
});

test('提醒语音能停止，本地用户提交后晚到的提醒语音不能接管输出', async () => {
  const app = await fixture();
  /** 构造与现有播放器协议一致的完整语段。 */
  function chunk(generation) {
    return { generation_id: generation, segment_id: 'reminder_001', audio_base64: 'UklGRg==', encoding: 'wav',
      chunk_seq: 0, is_last: true, duration_ms: 1000 };
  }
  try {
    app.receive('reminder.speech', { reminder_id: 'r1' }, 'voice', 'reminder:r1');
    app.receive('tts.chunk', chunk('voice'), 'voice', 'reminder:r1');
    assert.equal(app.audio.length, 1);
    app.render().stop();
    assert.equal(app.sent.at(-1).type, 'control.stop');
    assert.equal(app.sent.at(-1).payload.generation_id, 'voice');
    assert.ok(app.stops.includes('voice'));
    assert.equal(app.render().send('现在几点'), true);
    app.receive('reminder.speech', { reminder_id: 'r2' }, 'late', 'reminder:r2');
    app.receive('tts.chunk', chunk('late'), 'late', 'reminder:r2');
    assert.equal(app.audio.length, 1);
  } finally { app.close(); }
});
