#!/usr/bin/env bash
# MiniMax-H3 combined FL2VA + Ref2VA — shared encoders/VAE, dual DiT partitions
# 参考: vllm-omni/recipes/MiniMaxAI/MiniMax-H3-4090.md
set -euo pipefail

# GPU0 = RTX 5880 Ada 48GB, GPU1 = RTX 4090 D 24GB —— 两张同为 sm_89。
# 不引入 GPU3/4(3090, sm_86)：TP 分片等大，混架构未经验证，
# 且 attention backend 只按 GPU0 的 capability 选，混用会静默取错后端。
# TP 对称，显存上限由 GPU1 的 24GB 决定，故沿用官方 24GB 档配方；
# GPU0 多出的 24GB 给 rank0 的 VAE 解码和 pipeline 对象留余量。
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 15 秒长任务的去噪可能超过 1 小时，放宽同步等待上限
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
# 必须保留：绕开 vLLM #38967（SM89 上 TP>1 时 cuMemCreate 段错误）
export NCCL_CUMEM_HOST_ENABLE=0

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bootstrap_python="${H3_STUDIO_PYTHON:-python3}"
config_get=(/usr/bin/env PYTHONPATH="$project_dir" "$bootstrap_python" -m h3studio.config)
h3_python="$("${config_get[@]}" local python)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$("${config_get[@]}" local cuda_visible_devices)}"
h3_api_key="$(PYTHONPATH="$project_dir" "$h3_python" -c 'from h3studio.provider import load_provider; print(load_provider().api_key())')"
if [[ -z "$h3_api_key" ]]; then
  echo "H3 API key 为空；请设置 provider.api_key_env、H3_API_KEY 或 provider.api_key" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# 分区选择：默认 Ref2VA 单分区。
#
# 为什么不用 --task-type combined：combined 会同时加载 FL2VA 与 Ref2VA 两个
# 62GB DiT（日志 "Applying distributed layer-wise offloading on
# ['transformer', 'transformers_ref']"，92 blocks / 4 groups）。DLO 的常驻块
# 保留 pinned CPU 主副本，不减宿主内存，实测峰值 263.8GB
# （/sys/fs/cgroup/memory.peak）> 本机 251GB 且无 swap，rank1 被 OOM killer
# SIGKILL（Exit code: -9 / cgroup oom_kill=1），服务起不来。
#
# Ref2VA 单分区约 200GB，可安全常驻，且 ref2va 任务本身支持图片+视频+音频参考，
# 首帧身份约束与镜头间连续性都能覆盖，故导演模式全程走 ref2va。
#
# 需要 FL2VA（t2va / fl2va 首尾帧）时显式切换，不能与 Ref2VA 同时跑：
#   H3_PARTITION=fl2va ./run_h3.sh
h3_partition="${H3_PARTITION:-$("${config_get[@]}" local partition)}"
h3_model_root="$("${config_get[@]}" local model_root)"
[[ "$h3_model_root" = /* ]] || h3_model_root="$project_dir/$h3_model_root"
h3_vllm_bin="$("${config_get[@]}" local vllm_omni)"
h3_gpu_count="$("${config_get[@]}" local gpu_count)"
h3_tp_size="$("${config_get[@]}" local tensor_parallel_size)"
h3_host="$("${config_get[@]}" local host)"
h3_port="$("${config_get[@]}" local port)"
h3_usp="$("${config_get[@]}" local usp)"
h3_ring="$("${config_get[@]}" local ring)"
h3_dlo_resident_layers="$("${config_get[@]}" local dlo_resident_layers)"
h3_attention_backend="$("${config_get[@]}" local attention_backend)"
h3_vae_patch_parallel_size="$("${config_get[@]}" local vae_patch_parallel_size)"
h3_init_timeout="$("${config_get[@]}" local init_timeout_seconds)"
h3_stage_init_timeout="$("${config_get[@]}" local stage_init_timeout_seconds)"
if [[ "$h3_vllm_bin" != */* ]]; then
  h3_vllm_bin="$(command -v "$h3_vllm_bin" || true)"
fi
test -x "$h3_vllm_bin" || { echo "找不到 vllm-omni：$h3_vllm_bin" >&2; exit 2; }

# Keep model-side post-denoise checkpoints in the configured state directory
# so the workbench can resume a decode after an output-stage timeout.
latent_checkpoint_dir="$("${config_get[@]}" storage latent_checkpoints)"
[[ "$latent_checkpoint_dir" = /* ]] || latent_checkpoint_dir="$project_dir/$latent_checkpoint_dir"
mkdir -p "$latent_checkpoint_dir"
export MINIMAX_H3_LATENT_CHECKPOINT_DIR="$latent_checkpoint_dir"
export MINIMAX_H3_ASYNC_OUTPUT_TIMEOUT="$("${config_get[@]}" local async_output_timeout_seconds)"
gpu_quantization="$("${config_get[@]}" output gpu_quantization)"
case "${gpu_quantization,,}" in
  1|true|yes|on) export MINIMAX_H3_GPU_OUTPUT_UINT8=1 ;;
  *) export MINIMAX_H3_GPU_OUTPUT_UINT8=0 ;;
esac
export MINIMAX_H3_VIDEO_CODEC="$("${config_get[@]}" output video_codec)"
export MINIMAX_H3_VIDEO_CODEC_OPTIONS="$({
  NVENC_GPU="$("${config_get[@]}" output nvenc_gpu)" \
  NVENC_PRESET="$("${config_get[@]}" output nvenc_preset)" \
  NVENC_TUNE="$("${config_get[@]}" output nvenc_tune)" \
  NVENC_RC="$("${config_get[@]}" output nvenc_rate_control)" \
  NVENC_QP="$("${config_get[@]}" output nvenc_qp)" \
  "$h3_python" -c '
import json, os
print(json.dumps({"gpu": os.environ["NVENC_GPU"], "preset": os.environ["NVENC_PRESET"],
                  "tune": os.environ["NVENC_TUNE"], "rc": os.environ["NVENC_RC"],
                  "qp": os.environ["NVENC_QP"]}))
'
} 2>/dev/null)"
nvidia_library_dir="$("${config_get[@]}" output nvidia_library_dir)"
[[ "$nvidia_library_dir" = /* ]] || nvidia_library_dir="$project_dir/$nvidia_library_dir"
if [ -d "$nvidia_library_dir" ]; then
  export LD_LIBRARY_PATH="$nvidia_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
sync_output="$("${config_get[@]}" local sync_output)"
case "${sync_output,,}" in
  1|true|yes|on) export MINIMAX_H3_SYNC_OUTPUT=1 ;;
  *) export MINIMAX_H3_SYNC_OUTPUT=0 ;;
esac

state_file="$("${config_get[@]}" storage model_state_file)"
[[ "$state_file" = /* ]] || state_file="$project_dir/$state_file"
mkdir -p "$(dirname "$state_file")"
write_model_state() {
  STATE_FILE="$state_file" STATE_PARTITION="$h3_partition" STATE_STATUS="$1" \
  STATE_STAGE="$2" STATE_PROGRESS="$3" STATE_MESSAGE="$4" \
  PYTHONPATH="$project_dir" "$h3_python" -c '
import json, os, time
from pathlib import Path
path = Path(os.environ["STATE_FILE"])
payload = {"partition": os.environ["STATE_PARTITION"], "target": os.environ["STATE_PARTITION"],
           "status": os.environ["STATE_STATUS"], "stage": os.environ["STATE_STAGE"],
           "progress": int(os.environ["STATE_PROGRESS"]), "message": os.environ["STATE_MESSAGE"],
           "error": None, "started_at": int(time.time()), "updated_at": int(time.time())}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
'
}
write_model_state loading loading_weights 8 "正在加载 ${h3_partition^^} 权重…"
case "$h3_partition" in
  ref2va)
    h3_model_dir="$h3_model_root/Ref2VA"
    h3_task_type=ref2va
    ;;
  fl2va)
    h3_model_dir="$h3_model_root/FL2VA"
    h3_task_type=fl2va
    ;;
  *)
    echo "H3_PARTITION 只能是 ref2va 或 fl2va，收到：$h3_partition" >&2
    exit 2
    ;;
esac
test -s "$h3_model_dir/model_index.json" || {
  echo "分区权重缺失：$h3_model_dir/model_index.json" >&2
  exit 2
}

# Ref2VA 分区只含 transformer 分片，text_encoder 权重 / tokenizer / video_vae
# 与 FL2VA 共用（两边 config 逐字节相同，顶层 combined 的 model_index.json 也只
# 声明一份）。缺了会在 rank0 直接 FileNotFoundError:
#   Ref2VA/text_encoder/model.safetensors.index.json
# 这里用符号链接补齐：不占额外磁盘，且权重重新下载后每次启动都能自愈。
if [ "$h3_partition" = ref2va ]; then
  fl2va_dir="$h3_model_root/FL2VA"
  if [ -d "$fl2va_dir" ]; then
    for shared in tokenizer video_vae; do
      [ -e "$h3_model_dir/$shared" ] || ln -sfn "$fl2va_dir/$shared" "$h3_model_dir/$shared"
    done
    for component in text_encoder processor audio_vae; do
      [ -d "$fl2va_dir/$component" ] || continue
      mkdir -p "$h3_model_dir/$component"
      for source in "$fl2va_dir/$component"/*; do
        [ -e "$source" ] || continue
        target="$h3_model_dir/$component/$(basename "$source")"
        [ -e "$target" ] || ln -sfn "$source" "$target"
      done
    done
  fi
  test -s "$h3_model_dir/text_encoder/model.safetensors.index.json" || {
    echo "Ref2VA 的 text_encoder 权重缺失，且无法从 FL2VA 补齐" >&2
    exit 2
  }
fi

# 日志落盘：H3 的去噪循环用 tqdm 把真实步进写到 stderr
# (diffusion/models/minimax_h3/pipeline_minimax_h3.py:1541，total = 去噪步数)，
# 但那个进度永远不进 API —— 后端的 progress 字段只有 0 和 100 两个值。
# 落到文件后前端就能解析出真实的 "12/50"，不必再靠历史耗时估算。
#
# 每次启动截断并留一份 .prev：任务活不过后端重启，所以日志按进程周期切分，
# 前端解析时不会读到上一个进程的残留步数。
h3_log="$("${config_get[@]}" local log_file)"
[[ "$h3_log" = /* ]] || h3_log="$project_dir/$h3_log"
mkdir -p "$(dirname "$h3_log")"
[ -f "$h3_log" ] && mv -f "$h3_log" "$h3_log.prev"

# tqdm 自己会 flush，但 vLLM 的常规日志走 stdout 缓冲，不设这个会延迟几分钟才落盘
export PYTHONUNBUFFERED=1

# 用进程替换而不是 `exec ... | tee`：管道里的 exec 会让 tee 的退出码顶替
# 真实进程的退出码，set -e 就再也发现不了后端崩溃。这样写 exec 语义完整保留，
# 子进程直接继承已重定向的 fd。
exec > >(tee -a "$h3_log") 2>&1

# Native H3 VAE patch parallelism can deadlock after the final tile on
# mixed-memory GPUs. Sequential tiled decode is slower but reliable.
exec "$h3_vllm_bin" serve "$h3_model_dir" \
  --omni \
  --task-type "$h3_task_type" \
  --host "$h3_host" \
  --port "$h3_port" \
  --api-key "$h3_api_key" \
  --trust-remote-code \
  --num-gpus "$h3_gpu_count" \
  --tensor-parallel-size "$h3_tp_size" \
  --text-encoder-tp-size "$h3_tp_size" \
  --usp "$h3_usp" \
  --ring "$h3_ring" \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers "$h3_dlo_resident_layers" \
  --enforce-eager \
  --vae-use-tiling \
  --vae-patch-parallel-size "$h3_vae_patch_parallel_size" \
  --vae-parallel-mode tile \
  --diffusion-attention-backend "$h3_attention_backend" \
  --init-timeout "$h3_init_timeout" \
  --stage-init-timeout "$h3_stage_init_timeout"
