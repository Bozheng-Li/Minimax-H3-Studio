#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_python="${H3_STUDIO_PYTHON:-python3}"
config_get=(/usr/bin/env PYTHONPATH="$project_dir" "$bootstrap_python" -m h3studio.config)
h3_python="$("${config_get[@]}" local python)"
backend_url="$("${config_get[@]}" provider base_url)"
state_file="$("${config_get[@]}" storage model_state_file)"
[[ "$state_file" = /* ]] || state_file="$project_dir/$state_file"
if [[ -f "$project_dir/.h3_model_state.json" && ! -f "$state_file" ]]; then
  state_file="$project_dir/.h3_model_state.json"
fi
target="${1:-}"
if [[ "$target" != "ref2va" && "$target" != "fl2va" ]]; then
  exit 2
fi

write_state() {
  local status="$1" stage="$2" progress="$3" message="$4" error="${5:-null}"
  STATE_FILE="$state_file" STATE_PARTITION="$target" STATE_STATUS="$status" \
  STATE_STAGE="$stage" STATE_PROGRESS="$progress" STATE_MESSAGE="$message" STATE_ERROR="$error" \
  "$h3_python" - <<'PY'
import json, os, time, uuid
from pathlib import Path
path = Path(os.environ["STATE_FILE"])
payload = {
    "partition": os.environ["STATE_PARTITION"],
    "target": os.environ["STATE_PARTITION"],
    "status": os.environ["STATE_STATUS"],
    "stage": os.environ["STATE_STAGE"],
    "progress": int(os.environ["STATE_PROGRESS"]),
    "message": os.environ["STATE_MESSAGE"],
    "error": None if os.environ["STATE_ERROR"] == "null" else os.environ["STATE_ERROR"],
    "updated_at": int(time.time()),
}
tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

write_state loading stopping 8 "正在停止当前 H3 模型…"
tmux kill-session -t h3-api 2>/dev/null || true

for _ in $(seq 1 60); do
  if ! curl -fsS "$backend_url/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

write_state loading loading_weights 15 "正在加载 ${target^^} 权重（约需 10–15 分钟）…"
tmux new-session -d -s h3-api "cd $project_dir && H3_PARTITION=$target exec ./run_h3.sh"

for i in $(seq 1 240); do
  if curl -fsS "$backend_url/health" >/dev/null 2>&1; then
    write_state loading initializing 92 "模型权重已加载，正在初始化推理引擎…"
    for _ in $(seq 1 60); do
      if curl -fsS "$backend_url/load" >/dev/null 2>&1; then
        write_state ready ready 100 "${target^^} 模型已就绪"
        exit 0
      fi
      sleep 2
    done
  fi
  if (( i < 180 )); then
    progress=$((15 + i * 70 / 180))
    write_state loading loading_weights "$progress" "正在加载 ${target^^} 权重…（第 ${i} 个检查）"
  fi
  sleep 5
done

write_state error error 0 "${target^^} 模型启动超时，请查看 .h3_backend.log" "启动超时"
exit 1
