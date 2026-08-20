"""Shared paths, config loading and logging."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        die(f"config not found at {CONFIG_PATH}")
    try:
        with CONFIG_PATH.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        die(f"config.json is not valid JSON: {exc}")


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def log(tag: str, msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)
