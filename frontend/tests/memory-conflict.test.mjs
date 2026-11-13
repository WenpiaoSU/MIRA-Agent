/**
 * 作用：验证真实冲突卡的可见内容、明确选择和展示授权边界。
 * 当前功能：
 * - 检查空闲时才呈现冲突，只有唯一问题才回报展示。
 * - 检查原文按文本转义、按钮动作与处理中禁用状态。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { renderToStaticMarkup } from 'react-dom/server';
import ts from 'typescript';

const load = createRequire(import.meta.url);
const source = fs.readFileSync(new URL('../src/ui/MemoryConflict.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX,
} }).outputText;

/** 仅替换 effect 调度，使用生产组件与 React 静态渲染检查真实输出。 */
function render(props, strictMode = false) {
  const effects = [], acknowledgements = [], decisions = [], cleanups = [], microtasks = [], exports = {};
  vm.runInNewContext(compiled, { exports, require(name) {
    if (name === 'react') return { useEffect(callback) { effects.push(callback); } };
    return load(name);
  }, queueMicrotask(callback) { microtasks.push(callback); } });
  const item = { pending_id: 'p', target_content: '<script>原事实</script>', candidate: { content: '新事实' }, delivery_token: 't' };
  const element = exports.default({ offers: [item], visible: true, busy: false, notice: '', error: '',
    acknowledge(value, shown = true) { acknowledgements.push({ value, shown }); }, async resolve(value, action) { decisions.push({ value, action }); }, ...props });
  const markup = renderToStaticMarkup(element);
  for (const effect of effects) {
    let cleanup = effect();
    if (strictMode) { cleanup?.(); cleanup = effect(); }
    if (cleanup) cleanups.push(cleanup);
  }
  while (microtasks.length) microtasks.shift()();
  return { element, markup, acknowledgements, decisions, cleanups };
}

test('可见且唯一的冲突回报展示，用户原文被转义并保留两个明确操作', () => {
  const result = render({});
  assert.equal(result.acknowledgements.length, 1);
  assert.match(result.markup, /&lt;script&gt;原事实&lt;\/script&gt;/);
  assert.ok(!result.markup.includes('<script>'));
  assert.match(result.markup, /使用新信息/);
  assert.match(result.markup, /保留原记录/);
  result.cleanups[0]();
  assert.equal(result.acknowledgements.at(-1).shown, false);
});

test('后台忙碌时不展示问题也不授权短答，多候选只允许明确按钮选择', () => {
  const hidden = render({ visible: false });
  assert.equal(hidden.acknowledgements.length, 0);
  assert.ok(!hidden.markup.includes('是否使用新信息'));
  const multiple = render({ offers: [
    { pending_id: 'p1', target_content: '旧事实', candidate: { content: '新事实' }, delivery_token: 't1' },
    { pending_id: 'p2', target_content: '另一个旧事实', candidate: { content: '另一个新事实' }, delivery_token: 't2' },
  ] });
  assert.equal(multiple.acknowledgements.length, 0);
  assert.match(multiple.markup, /共 2 条，逐条处理/);
});

test('提交过程中两个决定按钮禁用，失败提示保持可读', () => {
  const result = render({ busy: true, error: '目标版本已变化' });
  assert.equal((result.markup.match(/disabled=""/g) || []).length, 2);
  assert.match(result.markup, /处理中/);
  assert.match(result.markup, /role="alert"/);
  assert.match(result.markup, /目标版本已变化/);
});

test('StrictMode 模拟卸载不会撤销真实挂载的唯一展示授权', () => {
  const result = render({}, true);
  assert.equal(result.acknowledgements.length, 1);
  assert.equal(result.acknowledgements[0].shown, true);
  result.cleanups[0]();
  assert.equal(result.acknowledgements.length, 2);
  assert.equal(result.acknowledgements[1].shown, false);
});
