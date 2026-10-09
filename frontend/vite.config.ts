/**
 * 作用：配置前端开发服务器和构建工具。
 * 当前功能：
 * - 启用 React 插件，并把 AudioWorklet 输出为独立资源。
 * - 将会话、健康检查与就绪接口代理到本地后端。
 */

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], build: { assetsInlineLimit: 0 }, server: { host: '0.0.0.0', proxy: { '/v1': { target: 'http://127.0.0.1:8000', ws: true }, '/health': 'http://127.0.0.1:8000', '/ready': 'http://127.0.0.1:8000' } } });

