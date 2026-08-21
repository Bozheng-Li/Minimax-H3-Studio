# MiniMax-H3 Studio

MiniMax-H3 Studio 是一个面向 MiniMax-H3 视频模型的本地/API 双模式视频工作台。它把提示词、参考素材、模型分区、生成队列和导演剪辑组织在一个可审计的工作流中，适合从单镜头试拍逐步扩展到多镜头连续短剧。

## 能做什么

- 普通模式：文生视频（T2VA）和首帧/尾帧/首尾帧图生视频（FL2VA）。
- Ref2VA：用人物图、场景图和参考视频保持角色与动作连续性。
- 导演模式：故事设定、角色/场景卡、镜头调度、导演剪辑四层页面；支持 pipeline（默认自动连续生成）与 review-gate（逐镜审核后继续）。
- 连续镜头：显式绑定场景图时切换场景；未绑定时继承上一镜最后画面和尾部参考视频。
- 队列与作品：实时去噪步数、引擎加载阶段、ETA、失败重试、补解码、历史作品和一次性清理失败任务。
- 导演剪辑：调整镜头顺序、裁剪入点/出点、音量、拼接并导出成片。
- 素材库：图片、视频、音频可跨普通模式、Ref2VA 和导演项目复用；删除项目素材会自动解除引用。

## 架构

```text
Browser (single-page UI)
        |
        v
FastAPI workbench :7860  ---- YAML config/provider ---->  local vllm-omni :8000
        |                                                or remote H3 API
        +--> jobs / library / director state (data/)
        +--> ffmpeg assembly and NVENC export (outputs/)
```

工作台只依赖 H3 兼容的 `/v1/videos` 接口。`provider.mode=local` 时由 `run_h3.sh` 管理本地推理引擎；`provider.mode=api` 时不加载本地权重，直接请求远程 API。FL2VA 与 Ref2VA 是互斥的本地 DiT 分区，可在界面顶部切换；导演模式会跟随当前可用分区自动适配。

## 快速开始

```bash
git clone https://github.com/Bozheng-Li/Minimax-H3-Studio.git
cd Minimax-H3-Studio
cp config.example.yaml config.yaml
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

系统侧需要 CUDA 驱动、`ffmpeg`；本地模型切换还需要 `tmux`。准备完成后编辑 `config.yaml`。API Key 推荐使用环境变量：

```bash
export MINIMAX_H3_API_KEY='your-key'
```

也支持 `provider.api_key` 写文件路径或明文；明文只适合本地临时测试，生产环境应使用环境变量或权限为 `0600` 的密钥文件。真实 `config.yaml`、模型权重和用户素材默认不会被 Git 跟踪。

### API 模式

```yaml
provider:
  mode: api
  base_url: https://your-h3-endpoint.example.com
  api_key_env: MINIMAX_H3_API_KEY
  model: MiniMax-H3
```

API 模式不需要 `run_h3.sh`。启动工作台：

```bash
./run_frontend.sh
```

浏览器打开 <http://127.0.0.1:7860>。

### 本地模式

把 `provider.mode` 设为 `local`，并将 `local.model_root` 指向包含 `FL2VA/` 与 `Ref2VA/` 的 MiniMax-H3 权重目录。安装与 CUDA 匹配的 `vllm-omni` 后：

```bash
./run_h3.sh                 # 默认分区来自 config.yaml
./run_frontend.sh           # 另一个终端
```

也可以用工作台的模型切换接口：

```bash
curl -X POST http://127.0.0.1:7860/api/model/switch \
  -H 'Content-Type: application/json' -d '{"partition":"ref2va"}'
```

## 配置要点

所有运行参数集中在一个 YAML 文件中：服务地址、Key 来源、模型分区、GPU/TP、NVENC、存储目录、心跳策略都在这里。配置解析支持相对路径，便于迁移到另一台机器。

本地双卡默认只使用两张同架构 GPU：TP 分片的瓶颈是显存较小的那张卡，较大显存主要为 VAE 解码和编码留余量。不要同时加载 FL2VA 与 Ref2VA；内存不足时会被系统 OOM killer 终止。

## 导演模式工作流

1. 故事设定：填写梗概、视觉圣经、时长、比例、声音和生成策略。
2. 角色与场景卡：创建、编辑、删除角色/场景，绑定素材库参考图并锁定外观特征。
3. 镜头调度：每镜填写一个主要动作、开始/结束状态、镜头语言和声音；短镜头应保持单一空间与单一动作。
4. 导演剪辑：预览、排序、裁剪、调音、拼接和导出。

`pipeline` 模式默认无需审核即可继续下一镜；审核只影响已完成版本，退回后可独立重生成该镜。`review_gate` 模式才会在每镜完成后等待审核。导演项目不把剧本写死在代码里，所有内容都来自前端保存的项目数据。

## 进度、队列与恢复

引擎加载显示“停止旧分区 → 加载权重 → 初始化 → 就绪”等阶段。扩散阶段从本地日志读取真实 `step/total`；ETA 是基于历史任务和当前工作量的动态估算，不是模型提供的精确剩余时间。视频生成、VAE 解码和 NVENC 输出是不同阶段，解码超时可使用“补解码”恢复已有 latent checkpoint。

```bash
curl http://127.0.0.1:7860/api/health
curl http://127.0.0.1:7860/api/jobs
nvidia-smi
```

暂停导演项目会取消未完成请求；本地 CUDA 内核无法即时中断时，工作台会按 `local.pause_restart_on_stuck` 重启同一分区以释放 GPU。失败任务位于历史记录，可批量清理；运行中的项目、角色和场景不会被误删。

## 心跳监控

心跳只监控，不会隐式启动项目。默认首次检查等待一小时，之后按 YAML 间隔检查；完成、失败、后端不可用和审核等待都会写入 `data/logs/director_heartbeat.log`。

```bash
python3 scripts/director_heartbeat.py \
  --project-id <project-id> --first-check 3600 --interval 120
```

无人值守时显式加 `--auto-retry` 或 `--auto-approve`，不要把审核模式和自动批准同时用于需要人工把关的项目。

## 开发与测试

```bash
python -m py_compile app.py h3studio/*.py scripts/director_heartbeat.py
node --check /tmp/h3-index-check.js   # 若已生成前端语法检查副本
python test_frontend_ui.py             # 需要运行中的工作台和 Chromium
```

不需要 GPU 即可运行配置模块、API schema 和大部分前端回归测试；真实视频生成仍需要 H3 推理服务或远程 H3 API。

## 安全与仓库边界

本仓库不包含 MiniMax-H3 权重、API Key、用户人物图/场景图、生成视频、队列状态、日志、latent checkpoint 或本地 NVIDIA 运行库。部署时请自行准备这些文件，并检查 `git status` 后再提交。公开仓库中不要写入真实密钥，即使之后删除，Git 历史也可能保留。

## 常见问题

- **模型一直加载**：查看 `/api/model` 和 `data/logs/h3_backend.log`；确认权重分区完整、宿主内存和显存足够。
- **导演镜头停在第一镜**：确认模型状态为 `ready`、项目为 `running`，并检查队列中第一镜是否为 `queued/in_progress`；后续 `pending` 是串联等待，不是错误。
- **视频完成但无画面**：检查 NVENC/ffmpeg 和 VAE 解码日志；有 checkpoint 时使用补解码。
- **角色或场景漂移**：为角色卡绑定同一身份参考图，后续镜头使用上一镜尾部参考，避免在 4 秒镜头塞入多个场景或多个主要动作。

## 模型与许可

MiniMax-H3 模型权重及其许可不随本仓库分发。工作台代码与模型是独立组件；部署和发布生成内容时，请分别遵循代码仓库声明、模型提供方条款以及素材版权要求。
