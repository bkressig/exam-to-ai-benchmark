"""Utility helpers for reading and writing JSON files."""

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sanitize_name(value: str) -> str:
    safe = value.replace("/", "__").replace(":", "_").replace(" ", "_")
    return safe
