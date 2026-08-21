from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .config import ROOT, section


@dataclass(frozen=True)
class H3Provider:
    mode: str
    base_url: str
    api_key_file: Path | None
    api_key_value: str | None
    api_key_env: str | None
    model: str
    timeout: float
    tasks: frozenset[str]

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    def api_key(self) -> str:
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "").strip()
            if value:
                return value
        value = os.environ.get("H3_API_KEY", "").strip()
        if value:
            return value
        if self.api_key_file and self.api_key_file.is_file():
            return self.api_key_file.read_text(encoding="utf-8").strip()
        return (self.api_key_value or "").strip()

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key()}"}

    def supports(self, task: str) -> bool:
        return task in self.tasks


def load_provider() -> H3Provider:
    config = section("provider")
    mode = str(config.get("mode", "local")).lower()
    if mode not in {"local", "api"}:
        raise RuntimeError("provider.mode 必须是 local 或 api")
    raw_tasks: Any = config.get("tasks", ["t2va", "fl2va", "ref2va"])
    tasks = frozenset(str(item) for item in raw_tasks) if isinstance(raw_tasks, list) else frozenset()
    raw_key = str(config.get("api_key", "")).strip()
    key_file = Path(raw_key).expanduser() if raw_key else None
    if key_file and not key_file.is_absolute():
        key_file = ROOT / key_file
    # A non-existent api_key path is treated as an intentional plaintext value.
    key_value = raw_key if key_file and not key_file.is_file() else None
    return H3Provider(
        mode=mode,
        base_url=str(config.get("base_url", "http://127.0.0.1:8000")).rstrip("/"),
        api_key_file=key_file,
        api_key_value=key_value,
        api_key_env=str(config.get("api_key_env", "")).strip() or None,
        model=str(config.get("model", "MiniMax-H3")),
        timeout=float(config.get("request_timeout_seconds", 90)),
        tasks=tasks,
    )
