/**
 * 作用：在无浏览器设备时验证生产会话 Hook 的提醒交付边界。
 * 当前功能：
 * - 检查提醒重发按编号去重，React 提交后才确认展示。
 * - 检查提醒语音复用播放器、可停止，且不能接管本地新输入。
 * - 检查记忆冲突通知、版本绑定选择、失败重试和旧版本刷新。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { webcrypto } from 'node:crypto';
import ts from 'typescript';

/** 用真实 Hook 和最小生命周期替身运行网络消息与 React 提交。 */
async function fixture(options = {}) {
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
    async fetch(path, input) {
      if (options.fetch && path.startsWith('/v1/memories/')) return options.fetch(path, input);
      return { ok: true, async json() { return { session_id: 'session' }; } };
    },
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

/** 构造服务器下发的真实冲突快照字段。 */
function offer(id = 'pending-1') {
  return { pending_id: id, target_id: 'old', target_version: 1, target_content: '我喜欢简短回答',
    candidate: { content: '我喜欢详细回答', evidence: '我喜欢详细回答', domain_scope: 'daily', memory_type: 'preference' },
    question: '是否使用新信息替换原记录？', reason: '存在冲突', created_at: '2026-10-25T00:00:00+00:00',
    expires_at: 2000000000, delivery_token: 'token-1' };
}

test('记忆通知到达不自动授权，展示回执绑定 ID、版本和连接令牌', async () => {
  const app = await fixture();
  try {
    app.receive('memory.pending', { pending: [offer()] });
    assert.equal(app.sent.filter(item => item.type === 'memory.pending.displayed').length, 0);
    const state = app.render();
    assert.equal(state.memoryPending[0].target_content, '我喜欢简短回答');
    state.acknowledgeMemory(state.memoryPending[0]);
    assert.equal(app.sent.at(-1).type, 'memory.pending.displayed');
    assert.deepEqual(JSON.parse(JSON.stringify(app.sent.at(-1).payload)), {
      pending_id: 'pending-1', target_version: 1, delivery_token: 'token-1',
    });
    app.receive('memory.cleared', {});
    assert.equal(app.render().memoryPending.length, 0);
  } finally { app.close(); }
});

test('采纳只提交所选版本，成功回执关闭候选且不清空聊天', async () => {
  const requests = [];
  const app = await fixture({ async fetch(path, input) {
    requests.push({ path, input });
    return { ok: true, async json() { return { status: 'saved', message: '已采用新信息。' }; } };
  } });
  try {
    app.receive('reminder.due', { reminder_id: 'r', delivery_token: 't', text: '仍然保留的聊天记录' });
    app.receive('memory.pending', { pending: [offer(), offer('pending-2')] });
    const state = app.render();
    await state.resolveMemory(state.memoryPending[0], 'accept_new');
    assert.equal(requests[0].path, '/v1/memories/pending/pending-1/resolve');
    assert.deepEqual(JSON.parse(requests[0].input.body), { action: 'accept_new', target_version: 1 });
    assert.equal(app.render().memoryPending[0].pending_id, 'pending-2');
    assert.equal(app.render().memoryNotice, '已采用新信息。');
    app.receive('memory.updated', {});
    assert.equal(app.render().messages[0].text, '仍然保留的聊天记录');
  } finally { app.close(); }
});

test('网络结果不明保留卡片，同一决定可重试，快速重复点击只发一个请求', async () => {
  let calls = 0, complete;
  const app = await fixture({ async fetch() {
    calls += 1;
    if (calls === 1) throw new Error('connection closed');
    await new Promise(resolve => { complete = resolve; });
    return { ok: true, async json() { return { message: '已保留当前记忆。' }; } };
  } });
  try {
    app.receive('memory.pending', { pending: [offer()] });
    let state = app.render();
    await state.resolveMemory(state.memoryPending[0], 'keep_old');
    state = app.render();
    assert.equal(state.memoryPending.length, 1);
    assert.match(state.memoryError, /未确认/);
    const first = state.resolveMemory(state.memoryPending[0], 'keep_old');
    const second = state.resolveMemory(state.memoryPending[0], 'keep_old');
    assert.equal(calls, 2);
    complete();
    await Promise.all([first, second]);
    assert.equal(app.render().memoryPending.length, 0);
    assert.equal(app.render().memoryBusy, false);
  } finally { app.close(); }
});

test('目标版本变化后展示明确错误并刷新列表，不伪报更新成功', async () => {
  const app = await fixture({ async fetch(path, input) {
    if (input?.method === 'POST') return { ok: false, status: 409, async json() { return { detail: '原记忆已变化，请刷新。' }; } };
    return { ok: true, async json() { return { pending: [] }; } };
  } });
  try {
    app.receive('memory.pending', { pending: [offer()] });
    const state = app.render();
    await state.resolveMemory(state.memoryPending[0], 'accept_new');
    const refreshed = app.render();
    assert.equal(refreshed.memoryPending.length, 0);
    assert.match(refreshed.memoryError, /已变化/);
    assert.equal(refreshed.memoryNotice, '');
  } finally { app.close(); }
});
