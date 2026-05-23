import json
import sys
from pathlib import Path

DEFAULT_IN = Path("./data/reviews.json")
DEFAULT_OUT = Path("./data/metrics.json")


def extract_rating(review: dict):
    for key in ("rating", "score"):
        value = review.get(key)
        if value is None:
            continue
        try:
            rating = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= rating <= 5:
            return rating
    return None


def calculate_metrics(reviews: list) -> dict:
    ratings = [r for r in (extract_rating(r) for r in reviews) if r is not None]
    total = len(ratings)

    distribution = {}
    for star in range(1, 6):
        count = sum(1 for r in ratings if r == star) if total else 0
        distribution[str(star)] = {
            "count": count,
            "percentage": round(100.0 * count / total, 2) if total else 0.0,
        }

    return {
        "average_rating": round(sum(ratings) / total, 2) if total else None,
        "total_reviews": total,
        "distribution": distribution,
    }


def load_reviews(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise RuntimeError(f"could not read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise RuntimeError(f"expected a JSON array in {path}")
    return data


def save_metrics(metrics: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"could not write {path}: {e}") from e


def print_metrics(metrics: dict) -> None:
    avg = metrics["average_rating"]
    avg_label = f"{avg:.2f}" if avg is not None else "n/a"
    print(f"Average rating: {avg_label} ({metrics['total_reviews']} reviews)")
    print("Rating distribution:")
    for star in range(5, 0, -1):
        entry = metrics["distribution"][str(star)]
        print(f"  {star}-star: {entry['percentage']:.2f}% ({entry['count']} reviews)")


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    try:
        reviews = load_reviews(in_path)
        metrics = calculate_metrics(reviews)
        save_metrics(metrics, out_path)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_metrics(metrics)
    print(f"Saved metrics -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
