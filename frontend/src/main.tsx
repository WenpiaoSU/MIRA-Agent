/**
 * 作用：作为 React 前端的浏览器入口。
 * 当前功能：
 * - 加载全局样式。
 * - 将应用根组件挂载到页面根节点。
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/app.css';

// 开发阶段启用严格模式，以便尽早发现组件生命周期问题。
createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
