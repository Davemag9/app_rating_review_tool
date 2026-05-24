import threading
from typing import Any


class AppStore:
    """
    Thread-safe in-memory store keyed by app_id.

    Each app entry holds:
      reviews   – raw review list (from scraper)
      processed – normalised review list
      metrics   – rating metrics dict
      insights  – AI analysis dict
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save(self, app_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(app_id, {})[key] = value

    def get(self, app_id: str, key: str) -> Any | None:
        with self._lock:
            return self._data.get(app_id, {}).get(key)

    def has(self, app_id: str, key: str) -> bool:
        with self._lock:
            return key in self._data.get(app_id, {})

    def apps(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


store = AppStore()
