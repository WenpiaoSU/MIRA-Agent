# MIRA-Agent

MIRA-Agent 是一个支持文字与语音对话的 Live2D 交互应用。浏览器负责麦克风采集、语音播放和角色渲染，后端通过 FastAPI 与 LangGraph 协调语言模型、语音识别、语音合成和实时轮次控制。

语言模型、ASR、TTS 均通过可配置的 HTTP API 接入；VAD 使用本地 Silero ONNX 模型。可以先配置 LLM 使用文字对话，再按需要启用语音和角色资源。

## 功能

- 文字输入、流式回复和逐句字幕。
- 浏览器麦克风输入、本地 VAD、ASR 转写与自动轮次提交；临时转写只用于显示。
- 浏览器实际启用回声消除时持续倾听，支持播放期间插话；不可用时退回单轮半双工。
- 逐句 TTS、按真实音频能量驱动口型、按媒体播放时钟执行角色动作。
- 取消旧输出、确认停止位置，支持“继续”“重复一遍”和“换个话题”。
- Live2D 表情、视线和手势，以及参数工作台、映射预览、浏览器保存和 JSON 导出。
- 会话管理、健康检查、控制轨迹查看与离线回放。
- 已提交轮次的七类意图识别、槽位校验与 Agent 分支路由；失败时澄清，未开放能力给出明确说明。
- 基础只读工具：当前时间、Open-Meteo 当前天气与七天内逐日预报；支持跨轮次补充参数及结果验证。
- 轻量任务规划：默认最多 6 步，支持不同工具配合、前序结果传参及跨轮补参；时间转换与日期计算在本地工具中完成。
- 基础记忆：LangGraph Checkpointer 短期上下文、SQLite 长期事实/偏好，支持跨会话查询、更正和删除。

当前会话、短期 Checkpointer、待澄清请求和工具结果保存在后端进程内，服务重启后清空。长期记忆保存在 SQLite，通过服务端签发的浏览器 Cookie 关联同一用户，新会话和后端重启后可继续查询；清除 Cookie 或换浏览器会建立新身份。视觉输入、提醒、MCP 和账户登录尚未接入。语音恢复采用句子级精度；真实设备的 AEC 效果与停止延迟需要实测。天气固定使用 Open-Meteo，无需 API Key；地点重名时先询问省份或国家。

## 系统结构

```text
文字输入 ───────────────────────────────┐
麦克风 → AudioWorklet → PCM → VAD/ASR → 轮次管理
                                       ↓
                              FastAPI / WebSocket
                                       ↓
                              LangGraph / LLM API
                                       ↓
                           稳定语段 + 角色行为计划
                                       ↓
                              TTS API / 输出协调
                                       ↓
                           字幕 + 音频播放 + Live2D
                                       ↑
                              实际播放反馈与取消确认
```

```text
MIRA-Agent/
├── backend/
│   ├── src/
│   │   ├── api/、gateway/       # HTTP、WebSocket、会话与事件发送
│   │   ├── agent/              # 对话图、上下文与中断恢复
│   │   ├── contracts/          # 实时事件、状态与输出契约
│   │   ├── speech/、turn/      # VAD、ASR 与轮次控制
│   │   ├── providers/          # LLM、TTS API 适配
│   │   ├── intent/、tools/     # 意图识别、执行器与 builtin 时间/天气工具
│   │   ├── memory/、repositories/ # 检查点记忆、候选门控与 SQLite 事务
│   │   ├── expression/、output/ # 行为计划与语音输出协调
│   │   └── evaluation/         # 控制轨迹与离线回放
│   ├── tests/
│   ├── .env.example
│   ├── environment.yml
│   └── pyproject.toml
├── frontend/
│   ├── src/                    # React、音频采集/播放、Live2D 与界面
│   ├── public/map/             # 默认 Avatar 语义映射
│   ├── tests/
│   ├── package.json
│   └── package-lock.json
├── shared/schemas/             # Avatar 配置 JSON Schema
└── docs/                       # 输出契约与实时控制说明
```

模型权重、角色文件和 Cubism Core 不包含在 Git 上传范围内，首次运行需按下文准备。依赖安装与构建不会自动补齐这些资源。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | 3.11 或以上；Conda 模板使用 3.11 |
| Node.js | 当前工具链已在 Node.js 22 下验证，使用配套 npm |
| Python 环境 | Conda 或 venv |
| 浏览器 | 支持 Web Audio、AudioWorklet、WebSocket；允许麦克风访问 |
| 外部接口 | 文字对话需要 LLM；语音输入另需 ASR；语音回复另需 TTS |
| 本地资源 | 语音输入需 VAD 权重；Live2D 需 Core、角色文件与参数目录 |

本机访问使用 `localhost`。远程语音访问使用 HTTPS，WebSocket 使用 WSS。麦克风与 AudioWorklet 依赖安全上下文，参见 [麦克风访问说明](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) 和 [AudioWorklet 文档](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)。

## 安装

克隆仓库后，以下路径均相对于仓库根目录，不依赖特定盘符。命令示例使用 PowerShell；macOS/Linux 将 `Copy-Item` 换为 `cp`。

### 后端

```powershell
cd backend
conda env create -f environment.yml
conda activate mira-agent
python -m pip install -e .
Copy-Item -LiteralPath .env.example -Destination .env
```

只在首次配置时复制模板，已有 `.env` 时直接编辑，避免覆盖自己的配置。`environment.yml` 创建 Python 环境，Python 依赖以 `pyproject.toml` 为准。没有 Conda 时，可使用 Python 3.11+ 创建 venv 后执行 `python -m pip install -e .`。

### 前端

在另一个终端从仓库根目录执行：

```powershell
cd frontend
npm ci
```

仓库保留 `package-lock.json`，`npm ci` 按锁文件安装依赖。

## API 配置

编辑 `backend/.env`。也可通过进程环境变量注入配置；已存在的环境变量优先于 dotenv 文件。修改后重启后端。所有 API 密钥仅由后端读取，前端没有需要填写的密钥，也不要通过 `VITE_*` 暴露密钥。

### LLM：文字对话必需

| 变量 | 含义 |
| --- | --- |
| `MIRA_LLM_URL` | 完整的 Chat Completions POST 地址，包括接口路径 |
| `MIRA_LLM_KEY` | 接口密钥，以 Bearer 认证发送 |
| `MIRA_LLM_MODEL_NAME` | 服务端接受的模型名称 |
| `MIRA_INTENT_TIMEOUT_MS` | 意图分类总超时，默认 30000 毫秒；复杂任务实测曾约 20 秒，超时进入澄清 |

服务需接收 `model`、`messages`、`stream=true`，通过 SSE 返回 `choices[0].delta.content`，可用 `[DONE]` 结束流。配置完整表示具备调用条件，健康接口不会自动验证密钥或服务端模型是否有效。

意图分类复用同一 LLM 接口，独立收集短 JSON，不将分类内容输出给用户。普通交流或问答通常先分类，再发起流式回复，因此新增一次模型请求及相应费用、延迟。完整控制短句不调用分类模型；分类失败、缺少必填参数或能力未开放时可直接给出确定性说明。分类只接收最终文本、来源及近期上下文，不接收 ASR 置信度。实现、边界和测试见 [意图识别说明](docs/intent-implementation.md)。

### 基础工具

| 变量 | 含义 |
| --- | --- |
| `MIRA_DEFAULT_TIMEZONE` | IANA 默认时区，默认 `Asia/Shanghai` |
| `MIRA_TOOL_TIMEOUT_MS` | 每次查询总预算，默认 5000 毫秒，包含天气地点检索 |
| `MIRA_PLAN_MAX_STEPS` | 计划总步骤，默认 6，范围 2–12，包含最后回答步骤 |
| `MIRA_PLAN_TIMEOUT_MS` | 规划和综合回复各自的模型总超时，默认 15000 毫秒；同时受整轮剩余预算限制 |
| `MIRA_WEATHER_ENABLED` | 默认 `true`；关闭时天气不进入工具目录 |

天气固定使用 Open-Meteo，无需密钥。可以尝试“现在几点？”或“查杭州, 浙江明天的天气”；“查明天天气”会先问地点，再接收下一轮补充。工具成功后直接输出已验证事实；“继续／重复”不会重新查询。见 [基础工具实现](docs/tools-implementation.md)。

组合任务可以尝试：“查询 UTC 当前时间，转换为 Asia/Shanghai 的明天日期，查杭州, 浙江该日期天气，再查天气地点现在几点，最后给穿衣建议。”计划通过前序字段引用串联 `get_current_time → convert_datetime → get_weather → get_current_time → respond`，不限定两次查询。缺参时保存整个任务，补充后跳过成功步骤；失败停止后续调用，取消后不会自动继续工具操作。见 [任务规划实现](docs/planning-implementation.md)。

### 记忆

默认启用基础记忆，无需额外数据库服务。长期数据位于 `backend/data/memory.sqlite3`，短期上下文使用 LangGraph 内存型 Checkpointer。

| 变量 | 含义 |
| --- | --- |
| `MIRA_MEMORY_ENABLED` | 默认 `true`，关闭后新会话仅使用短期记忆 |
| `MIRA_MEMORY_DB_PATH` | SQLite 文件路径；不设置时使用后端 `data/memory.sqlite3` |
| `MIRA_MEMORY_TIMEOUT_MS` | 每次记忆候选提取的模型预算，默认 10000 毫秒 |
| `MIRA_WORKING_MEMORY_TTL_SECONDS` | 短期上下文保留时间，默认 3600 秒 |

可以依次尝试“我喜欢简短回答”“你记得我的回答风格吗”“现在把回答偏好改为详细解释”“忘掉我的回答偏好”。同一浏览器重新打开页面后可查询已保存事实。删除一个事实同时删除该事实的历史版本，并清空该用户各会话的上下文及显示，防止已忘内容被旧对话带回。查看与删除接口、边界见 [记忆实现说明](docs/memory-implementation.md)。

### ASR：语音输入必需


| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `MIRA_ASR_URL` | 空 | 完整的转写 POST 地址 |
| `MIRA_ASR_KEY` | 空 | 转写接口密钥 |
| `MIRA_ASR_MODEL_NAME` | 空 | 转写模型名称 |
| `MIRA_ASR_PROVIDER` | `auto` | `auto`、`transcriptions` 或 `deepgram` |
| `MIRA_ASR_LANGUAGE` | 空 | 可选语言代码，例如 `zh`；服务需支持该值 |

浏览器发送 16 kHz、单声道、PCM s16le，每帧 20ms；后端将语音片段包装为 WAV。当前适配协议：

| 协议 | 认证与请求 | 必需响应字段 |
| --- | --- | --- |
| `transcriptions` | Bearer；multipart `file`、`model`、`response_format=json`，可选 `language` | JSON 字符串 `text` |
| `deepgram` | Token；原始 `audio/wav` 请求体，查询参数 `model`、可选 `language` | `results.channels[0].alternatives[0].transcript` |

`auto` 根据 Deepgram 官方域名选择协议，其他域名使用 `transcriptions`。自建代理应显式设置协议。约每累计 1.5s 音频转写当前快照显示 partial，片段结束后再次转写得到 final；这会产生多次 API 请求及费用，属于 HTTP 快照转写，并非供应商原生流式 ASR。只有 final 才能进入轮次提交与 Agent。

### TTS：语音回复必需

| 变量 | 含义 |
| --- | --- |
| `MIRA_TTS_URL` | 完整的语音合成 POST 地址 |
| `MIRA_TTS_KEY` | 以 Bearer 认证发送的密钥 |
| `MIRA_TTS_MODEL_NAME` | 合成模型名称 |
| `MIRA_TTS_VOICE` | 音色的 `reference_id` |
| `MIRA_TTS_EMOTION_TAGS` | 可选 JSON：Avatar 表情到模型情感标签的映射 |

当 URL 主机名为 `api.fish.audio` 时，适配器使用 `model` 请求头，JSON 包含 `text`、`reference_id`、`format=wav`。其他主机使用 JSON `model`、`reference_id`、`input`、`response_format=wav`。自建 Fish 代理不会自动按 Fish 协议识别，需匹配当前通用请求格式或修改适配器。

这里的音色字段是 `reference_id`，并未发送标准 `voice` 字段；不能假定所有名称相似的 Speech API 都可直接使用。响应必须是非空、未压缩 PCM WAV，单语段最大 8MiB。

合成文本每句前自动添加 `[情感标签]`，如 `[sad] 听到这个消息我很难过。`；字幕与历史保持原始文本。未配置映射的表情使用自身名称。供应商模型需要支持这种标签；可按模型词表配置，例如：

```dotenv
MIRA_TTS_EMOTION_TAGS={"gentle_smile":"hopeful","concerned":"neutral"}
```

允许的 Avatar 表情为 `neutral`、`gentle_smile`、`concerned`、`sad`、`confused`；标签须以英文字母开头，仅包含字母、数字、下划线或连字符。



## 本地资源准备

### Silero VAD

从 [Silero VAD 官方模型文件](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad_16k_op15.onnx) 下载原始文件，放到：

```text
backend/model/silero_vad_16k_op15.onnx
```

### Cubism Core

从 [Live2D Cubism SDK for Web 官方下载页](https://www.live2d.com/en/sdk/download/web/) 获取适用版本，阅读其许可条款，取出 `live2dcubismcore.min.js`，放到：

```text
frontend/public/vendor/live2dcubismcore.min.js
```


### Live2D 角色与参数映射

自行提供已获使用授权的 Cubism 模型，并完整保留模型清单引用的纹理、物理、显示信息等文件及相对目录。当前加载入口是：

```text
frontend/public/models/robot/白-免费版.model3.json
frontend/public/models/robot/parameter-catalog.json
```

角色文件不随仓库分发。使用不同模型时，修改 `frontend/src/avatar/Live2DCanvas.tsx` 中 `Live2DModel.from(...)` 的入口 URL；同时提供 `/models/robot/parameter-catalog.json`，或修改 `frontend/src/App.tsx` 中的目录 URL。不能只重命名入口而忽略 `.model3.json` 内部的 `FileReferences`。

参数目录是数组，每项提供真实参数 ID、中文名称、范围和默认值；下面仅为格式示例，数值应取自自己的模型：

```json
[
  {"id":"ParamAngleX","name":"头部左右转动","min":-30,"max":30,"default":0}
]
```

默认语义映射在 `frontend/public/map/avatar-mapping.json`，映射 Schema 在 `shared/schemas/avatar-mapping.schema.json`。更换角色后需调整参数 ID、范围、强度与取景，不能直接假设预设映射适用。参数工作台的“保存”写入浏览器本地副本；启动时优先读取项目公开映射文件，要同步默认映射，应导出 JSON 并更新公开文件。

## 启动与使用

后端终端：

```powershell
cd backend
conda activate mira-agent
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

前端终端：

```powershell
cd frontend
npm run dev
```

在本机打开 Vite 输出的 Local 地址，通常为 `http://localhost:5173`。开发服务器将 `/v1`（包括 WebSocket）、`/health`、`/ready` 代理到 `http://127.0.0.1:8000`；使用其他后端端口时更新 `frontend/vite.config.ts`。后端接口文档位于 `http://127.0.0.1:8000/docs`。

- 文字：输入消息后按 Enter 或点击发送，新输入取消旧输出。
- 语音：点击左侧麦克风开始；持续模式下说完自动提交，播放期间仍可插话。再次点击麦克风结束并提交残余语音。
- 录音时只由麦克风图标控制采集，右侧不显示停止按钮；非录音且机器人正在回复时显示“停止回复”。
- “停止”“停一下”等完整控制短句直接取消回复；“继续”依据中断快照生成后续内容，“重复一遍”重新表达，“换个话题”放弃旧恢复任务。
- 恢复时只将完整播完的句子视为确定已说出，部分句子可从句首重述。不会自动重播旧队列。
- 断线停止本地音频、动作和录音，前端有限重连；恢复连接后需主动重新打开麦克风。

浏览器首次播放可能要求用户手势；点击发送或麦克风会尝试解锁音频上下文。



## 常见问题

| 问题 | 排查 |
| --- | --- |
| 页面显示 Agent 未连接 | 确认后端启动、8000 端口和 Vite 代理；检查 `/health` |
| 配置后仍显示待配置 | 确认文件位于 `backend/.env`，重启后端，并检查是否被进程环境变量覆盖 |
| 麦克风灰色或无法打开 | 检查 `speech_ready`、ASR 三项配置、ONNX 文件、onnxruntime、浏览器权限及安全上下文 |
| 开始录音后模型加载失败 | 配置存在不代表 ONNX 可加载；检查模型版本、下载文件是否为真实二进制 |
| ASR 返回 401/403 或格式错误 | 检查密钥、完整 URL、模型名与 `MIRA_ASR_PROVIDER`；代理需匹配请求/响应协议 |
| TTS 保留字幕但无声 | 检查四项 TTS 配置、`reference_id` 协议、情感标签、PCM WAV 响应与浏览器播放权限 |
| 机器人播放误触发插话 | 检查设备 AEC 效果；设 `MIRA_FULL_DUPLEX_ENABLED=false` 后重启，使用单轮半双工 |
| Live2D 不显示或动作异常 | 检查 Core、入口及引用文件是否 404，模型兼容性、参数目录、映射和取景 |
| 工作台保存后新页面仍是旧映射 | 加载优先读取公开映射；导出并更新 `public/map/avatar-mapping.json` |
| Python 环境安装失败 | 确认 Python ≥3.11，使用独立环境并按 `pyproject.toml` 安装 |

音频片段限制、播放背压、网络超时及中断协议详见 [实时控制说明](docs/p3-implementation.md)；Agent 行为字段见 [Avatar 输出契约](docs/avatar-output-contract.md)。


