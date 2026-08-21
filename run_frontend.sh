#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_python="${H3_STUDIO_PYTHON:-python3}"
config_get=(/usr/bin/env PYTHONPATH="$project_dir" "$bootstrap_python" -m h3studio.config)
host="$("${config_get[@]}" app host)"
port="$("${config_get[@]}" app port)"
frontend_python="$("${config_get[@]}" local python)"
exec "$frontend_python" -m uvicorn app:app \
  --host "$host" \
  --port "$port"
