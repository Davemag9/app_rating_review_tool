import threading
from collections import OrderedDict
from typing import Any

_MAX_APPS = 1


class AppStore:
    """
    Thread-safe in-memory store keyed by app_id.

    Each app entry holds:
      reviews   – raw review list (from scraper)
      processed – normalised review list
      metrics   – rating metrics dict
      insights  – AI analysis dict

    At most _MAX_APPS entries are kept. The least-recently-written app is
    evicted when the limit is exceeded to bound memory usage.
    """

    def __init__(self, max_apps: int = _MAX_APPS) -> None:
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_apps = max_apps

    def save(self, app_id: str, key: str, value: Any) -> None:
        with self._lock:
            if app_id in self._data:
                self._data.move_to_end(app_id)
            else:
                while len(self._data) >= self._max_apps:
                    self._data.popitem(last=False)
                self._data[app_id] = {}
            self._data[app_id][key] = value

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
