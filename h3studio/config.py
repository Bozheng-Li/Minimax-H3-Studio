from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.is_file():
        raise RuntimeError(f"配置文件不存在：{CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError("config.yaml 顶层必须是对象")
    return data


def section(name: str) -> dict[str, Any]:
    value = load_config().get(name, {})
    return value if isinstance(value, dict) else {}


def path(section_name: str, key: str) -> Path:
    value = section(section_name).get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"配置缺少 {section_name}.{key}")
    return _resolve(value)


def value(section_name: str, key: str, default: str = "") -> str:
    raw = section(section_name).get(key, default)
    return str(raw)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m h3studio.config SECTION KEY")
    print(value(sys.argv[1], sys.argv[2]))
