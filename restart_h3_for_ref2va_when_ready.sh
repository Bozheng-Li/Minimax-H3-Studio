#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_python="${H3_STUDIO_PYTHON:-python3}"
config_get=(/usr/bin/env PYTHONPATH="$project_dir" "$bootstrap_python" -m h3studio.config)
model_dir="$("${config_get[@]}" local model_root)"
[[ "$model_dir" = /* ]] || model_dir="$project_dir/$model_dir"
python_bin="$("${config_get[@]}" local python)"
frontend_url="$("${config_get[@]}" app internal_url)"
backend_url="$("${config_get[@]}" provider base_url)"

# Wait for the Ref2VA download session to finish successfully and for the
# complete 13-shard transformer partition to exist.
while tmux has-session -t h3-ref2va-download 2>/dev/null; do
  sleep "${H3_DOWNLOAD_POLL_SECONDS:-15}"
done

for shard in $(seq -w 1 13); do
  test -s "$model_dir/Ref2VA/transformer/model-000${shard}-of-00013.safetensors"
done
test -s "$model_dir/Ref2VA/model_index.json"

# Never interrupt an active generation. Require three consecutive idle checks.
stable_idle_checks=0
while (( stable_idle_checks < 3 )); do
  active_count="$({ curl -fsS "$frontend_url/api/jobs" || printf '[]'; } | "$python_bin" -c '
import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    print(999)
else:
    terminal = {"completed", "failed", "cancelled"}
    print(sum(str(job.get("status")) not in terminal for job in jobs))
')"
  if [[ "$active_count" == "0" ]]; then
    stable_idle_checks=$((stable_idle_checks + 1))
  else
    stable_idle_checks=0
  fi
  sleep "${H3_IDLE_POLL_SECONDS:-10}"
done

# `|| true`：会话不存在时 kill-session 返回非 0，set -e 会让脚本在这里静默退出，
# 后端就永远等不到重启（2026-08-20 卡死那次正是如此：h3-api 会话已不在，
# 脚本在这一行终止，8000 端口无人拉起，只剩 curl 连接失败刷屏）。
tmux kill-session -t h3-api 2>/dev/null || true
tmux new-session -d -s h3-api "cd $project_dir && exec ./run_h3.sh"

for _ in $(seq 1 240); do
  if curl -fsS "$backend_url/health" >/dev/null; then
    exit 0
  fi
  sleep "${H3_RESTART_POLL_SECONDS:-10}"
done
exit 1
