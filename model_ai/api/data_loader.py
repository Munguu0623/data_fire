"""
dashboard_data.json-ийг санах ойд кэш хийж, /admin/reload дуудлагаар
шинэчилдэг нэг л газрын өгөгдлийн давхарга.
"""
import json
from pathlib import Path
from threading import Lock

_BASE = Path(__file__).parent.parent
_PATH = _BASE / "data" / "dashboard_data.json"
_lock = Lock()

_cache: dict = {}


def _load() -> dict:
    with open(_PATH, encoding="utf-8") as f:
        return json.load(f)


def startup() -> None:
    global _cache
    _cache = _load()


def reload() -> None:
    global _cache
    fresh = _load()
    with _lock:
        _cache = fresh


def get_records() -> list[dict]:
    return _cache.get("records", [])


def get_districts() -> list[str]:
    return _cache.get("districts", [])


def get_metrics() -> dict:
    return {
        "metrics":    _cache.get("metrics", []),
        "importance": _cache.get("importance", {}),
        "meta":       _cache.get("meta", {}),
    }
