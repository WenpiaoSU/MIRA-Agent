/**
 * 作用：在交互空闲时展示具体记忆冲突并收集明确选择。
 * 当前功能：
 * - 并列展示原事实、新信息、适用范围及原始依据，逐条处理多条候选。
 * - React 提交后才发送展示回执，不把列表入队当作用户已看到问题。
 * - 提供采纳、保留及提交中/失败/成功状态，正文始终按普通文本渲染。
 */

import { useEffect } from 'react';
import type { PendingMemoryOffer, MemoryResolutionAction } from '../contracts/realtime';

/** 展示一个候选，其余候选待当前项处理完后再出现。 */
export default function MemoryConflict({ offers, visible, busy, notice, error, acknowledge, resolve }: {
  offers: PendingMemoryOffer[]; visible: boolean; busy: boolean; notice: string; error: string;
  acknowledge: (offer: PendingMemoryOffer, shown?: boolean) => void;
  resolve: (offer: PendingMemoryOffer, action: MemoryResolutionAction) => Promise<void>;
}) {
  const offer = visible ? offers[0] : undefined;
  const scope = offer?.candidate.domain_scope || 'daily';
  useEffect(() => {
    if (offer?.delivery_token && offers.length === 1) {
      let active = true;
      let shown = false;
      // 等待本次 effect 批次稳定，StrictMode 的模拟卸载不能先授权再立即撤销。
      queueMicrotask(() => {
        if (active) {
          acknowledge(offer);
          shown = true;
        }
      });
      return () => {
        active = false;
        if (shown) acknowledge(offer, false);
      };
    }
  }, [offer?.pending_id, offer?.delivery_token, offers.length, visible, acknowledge]);
  return <div className="memory-confirmation-area">
    {offer && <section className="memory-conflict" aria-label="待确认记忆">
      <div className="memory-conflict-heading"><strong>有一条记忆需要你确认</strong><span>{offers.length > 1 ? `共 ${offers.length} 条，逐条处理` : '当前记录仍然有效'}</span></div>
      <dl><div><dt>原记录</dt><dd>{offer.target_content || '未能核实原记录'}</dd></div>
        <div><dt>新信息</dt><dd>{offer.candidate.content}</dd></div></dl>
      <p>适用范围：{scope === 'daily' ? '日常' : scope.replace(/^project:/, '')}</p>
      <details><summary>查看原始依据</summary><p>{offer.candidate.evidence}</p></details>
      <p>是否使用新信息替换原记录？</p>
      <div className="memory-conflict-actions">
        <button className="primary-button" disabled={busy} onClick={() => void resolve(offer, 'accept_new')}>{busy ? '处理中…' : '使用新信息'}</button>
        <button className="soft-button" disabled={busy} onClick={() => void resolve(offer, 'keep_old')}>保留原记录</button>
      </div>
    </section>}
    {error && <div className="memory-feedback memory-feedback-error" role="alert">{error}</div>}
    {notice && <div className="memory-feedback" role="status">{notice}</div>}
  </div>;
}
