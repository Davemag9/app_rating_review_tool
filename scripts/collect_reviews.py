import json
import random
import sys
from datetime import datetime
from pathlib import Path

from google_play_scraper import Sort, reviews
from google_play_scraper.exceptions import NotFoundError

from api.utils.app_id import parse_app_id

SAMPLE, POOL = 100, 500


def fetch_pool(app: str) -> list:
    pool, token = [], None

    while len(pool) < POOL:
        try:
            batch, token = reviews(
                app, count=min(200, POOL - len(pool)), sort=Sort.NEWEST, continuation_token=token
            )

        except NotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"fetch failed: {e}") from e

        if not batch:
            break

        pool.extend(batch)

        if not token or not token.token:
            break

    return pool[:POOL]


def main() -> int:
    try:
        app = parse_app_id(sys.argv[1] if len(sys.argv) > 1 else "genesis.nebula")
    except ValueError as e:
        print(f"Invalid input: {e}", file=sys.stderr)
        return 2

    try:
        pool = fetch_pool(app)

    except NotFoundError:
        print(f"App not found: {app}", file=sys.stderr)
        return 3
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not pool:
        print("Error: no reviews returned for this app", file=sys.stderr)
        return 1

    n = min(SAMPLE, len(pool))
    if n < SAMPLE:
        print(f"Warning: only {n} reviews available (requested {SAMPLE})", file=sys.stderr)

    sample = random.sample(pool, n)
    out = Path("./data/reviews.json")

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(sample, indent=2, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o)),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"Error: could not write {out}: {e}", file=sys.stderr)
        return 1

    print(f"Saved {n} reviews to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
