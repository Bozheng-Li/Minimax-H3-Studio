#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${H3_STUDIO_PYTHON:-python3}"
config_get=(/usr/bin/env PYTHONPATH="$project_dir" "$python_bin" -m h3studio.config)
python_bin="$("${config_get[@]}" local python)"
frontend_url="$("${config_get[@]}" app internal_url)"
backend_url="$("${config_get[@]}" provider base_url)"
stable_idle_checks=0

# 重启后是否自动重投 .h3_oom_recovery_payload.json。
# 默认关闭：那个 payload 是 15s + lossless + generate_sound 的任务，正是把后端
# 撑爆的那一个，无条件重投等于重启后立刻再爆一次。要用就显式开：
#   H3_REPLAY_OOM_PAYLOAD=1 ./restart_h3_when_idle.sh
replay_oom_payload="${H3_REPLAY_OOM_PAYLOAD:-0}"

while true; do
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

  if (( stable_idle_checks >= 3 )); then
    break
  fi
  sleep "${H3_IDLE_POLL_SECONDS:-10}"
done

# `|| true`：会话不存在时 kill-session 返回非 0，set -e 会让脚本在这里静默退出，
# 重启就永远不会发生（见 restart_h3_for_ref2va_when_ready.sh 同一处注释）。
tmux kill-session -t h3-api 2>/dev/null || true
tmux new-session -d -s h3-api "cd $project_dir && exec ./scripts/run_h3.sh"

for _ in $(seq 1 180); do
  if curl -fsS "$backend_url/health" >/dev/null; then
    if [[ "$replay_oom_payload" != "1" ]]; then
      echo "后端已就绪。未重投 OOM payload（设 H3_REPLAY_OOM_PAYLOAD=1 可开启）。"
      exit 0
    fi
    curl --fail-with-body -sS \
      -X POST "$frontend_url/api/generate" \
      -F "payload=<$project_dir/.h3_oom_recovery_payload.json;type=application/json" \
      -F "images=@$project_dir/.h3_queue_assets/video_gen_c14a065913104c1b9dd6186006d2425b/00.png;type=image/png"
    exit 0
  fi
  sleep "${H3_RESTART_POLL_SECONDS:-10}"
done

exit 1
