#!/usr/bin/env python3
"""Monitor one H3 Director project without starting it implicitly.

Default mode is review-safe: it observes the project and reports blockers.
Use --auto-approve only when unattended generation of every shot is intended.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from h3studio.config import section as config_section
except Exception:  # Keep the monitor usable during partial installs.
    config_section = lambda _name: {}
DEFAULT_PROJECT = ""
DEFAULT_BASE_URL = str(config_section("app").get("internal_url", "http://127.0.0.1:7860")).rstrip("/")
LOG_FILE = ROOT / "data/logs/director_heartbeat.log"


def request_json(base_url: str, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def write_log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def summarize(project: dict[str, Any]) -> tuple[str, int, int, list[str]]:
    shots = project.get("shots") or []
    done = sum(shot.get("status") in {"approved", "completed"} for shot in shots)
    active = sum(shot.get("status") in {"queued", "in_progress"} for shot in shots)
    waiting = [str(shot.get("title") or shot.get("id")) for shot in shots if shot.get("status") == "awaiting_review"]
    return str(project.get("status") or "unknown"), done, active, waiting


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 Director project heartbeat")
    heartbeat_config = config_section("heartbeat")
    parser.add_argument("--project-id", default=os.environ.get("H3_DIRECTOR_PROJECT", heartbeat_config.get("project_id", DEFAULT_PROJECT)))
    parser.add_argument("--base-url", default=os.environ.get("H3_STUDIO_URL", DEFAULT_BASE_URL))
    parser.add_argument("--interval", type=int, default=int(heartbeat_config.get("interval_seconds", 300)), help="poll interval in seconds")
    parser.add_argument("--first-check", type=int, default=int(heartbeat_config.get("first_check_seconds", 3600)), help="seconds until the first hourly checkpoint")
    parser.add_argument("--auto-approve", action="store_true", default=bool(heartbeat_config.get("auto_approve", False)), help="approve awaiting_review shots automatically")
    parser.add_argument("--auto-retry", action="store_true", default=bool(heartbeat_config.get("auto_retry", False)), help="retry each failed shot at most once after the project was started")
    args = parser.parse_args()

    if not args.project_id:
        parser.error("--project-id is required unless heartbeat.project_id is set in config.yaml")
    if args.interval < 30:
        parser.error("--interval must be at least 30 seconds")
    if args.first_check < 0:
        parser.error("--first-check must be non-negative")

    write_log(
        f"started project={args.project_id} interval={args.interval}s first_check={args.first_check}s "
        f"auto_approve={args.auto_approve}"
    )
    started_at = time.monotonic()
    hourly_reported = False
    last_signature: tuple[Any, ...] | None = None
    observed_running = False
    retried_shots: set[str] = set()

    while True:
        try:
            project = request_json(args.base_url, f"/api/director/projects/{args.project_id}")
            health = request_json(args.base_url, "/api/health")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            write_log(f"service_check_failed error={exc}")
            time.sleep(args.interval)
            continue

        status, done, active, waiting = summarize(project)
        shots = project.get("shots") or []
        failed = [str(shot.get("title") or shot.get("id")) for shot in shots if shot.get("status") == "failed"]
        signature = (status, done, active, tuple(waiting), tuple(failed), health.get("backend"))
        signature_changed = signature != last_signature
        if signature_changed:
            last_signature = signature
            write_log(
                f"status={status} shots={done}/{len(shots)} active={active} "
                f"awaiting_review={waiting or '-'} failed={failed or '-'} backend={health.get('backend')}"
            )

        elapsed = time.monotonic() - started_at
        if not hourly_reported and elapsed >= args.first_check:
            hourly_reported = True
            write_log(
                f"hour_checkpoint elapsed={int(elapsed)}s status={status} "
                f"shots={done}/{len(shots)} active={active} backend={health.get('backend')}"
            )

        if status == "completed" and shots and done == len(shots):
            write_log("completed all shots; heartbeat exiting")
            return 0

        if status == "running":
            observed_running = True

        if failed and args.auto_retry and observed_running and health.get("backend") == "ok":
            for shot in shots:
                shot_id = str(shot.get("id"))
                if shot.get("status") != "failed" or shot_id in retried_shots:
                    continue
                try:
                    request_json(
                        args.base_url,
                        f"/api/director/projects/{args.project_id}/shots/{shot_id}/retry",
                        method="POST",
                    )
                    retried_shots.add(shot_id)
                    write_log(f"auto_retry shot={shot.get('title') or shot_id} attempt=1")
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                    write_log(f"auto_retry_failed shot={shot_id} error={exc}")

        if failed:
            write_log("blocked failed_shots=" + ",".join(failed) + " action=review_or_retry_in_director_ui")

        if waiting:
            if args.auto_approve:
                for shot in shots:
                    if shot.get("status") != "awaiting_review":
                        continue
                    shot_id = str(shot.get("id"))
                    try:
                        request_json(
                            args.base_url,
                            f"/api/director/projects/{args.project_id}/shots/{shot_id}/review",
                            method="POST",
                            payload={"approved": True, "note": "Heartbeat unattended approval"},
                        )
                        write_log(f"auto_approved shot={shot.get('title') or shot_id}")
                    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                        write_log(f"auto_approve_failed shot={shot_id} error={exc}")
            else:
                write_log("waiting for manual review; no generation action taken")

        if status == "paused" and signature_changed:
            write_log("project paused; heartbeat will wait and will not start generation")

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
