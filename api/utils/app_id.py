"""Parse and validate Google Play app identifiers."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

PACKAGE_RE = re.compile(r"^[a-zA-Z][\w]*(?:\.[\w]+)+$")


def parse_app_id(raw: str) -> str:
    """
    Accept an Android package name or Google Play Store app URL.
    Returns the normalised package name (e.g. genesis.nebula).
    Raises ValueError with a human-readable message on invalid input.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(
            "App package ID is required. "
            "Enter a package name (e.g. genesis.nebula) or paste a Play Store URL."
        )

    if raw.startswith("http://") or raw.startswith("https://"):
        if "play.google.com/store/apps/details" not in raw.lower():
            raise ValueError(
                "That link is not a Google Play app page. "
                "Paste a URL like: "
                "https://play.google.com/store/apps/details?id=com.example.app"
            )
        package = parse_qs(urlparse(raw).query).get("id", [None])[0]
        if not package:
            raise ValueError(
                "Play Store URL is missing the app ID (?id=…). "
                "Example: https://play.google.com/store/apps/details?id=genesis.nebula"
            )
        raw = package

    if not PACKAGE_RE.match(raw):
        raise ValueError(
            f"'{raw}' is not a valid Android package name. "
            "Use the format com.company.app (letters, numbers, dots). "
            "Examples: genesis.nebula, com.spotify.music"
        )

    return raw
