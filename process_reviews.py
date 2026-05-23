import json
import sys
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

DEFAULT_IN = Path("./data/reviews.json")
DEFAULT_OUT = Path("./data/reviews_processed.json")


def text_tokenization(text):
    return word_tokenize(text.lower())


def text_cleaner(text):
    tokens = text_tokenization(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in tokens if word not in stop_words]
    return filtered_words


def text_simplifier(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    text = " ".join(stemmed_tokens)
    return text


def text_preprocessing(text):
    filtered_words = text_cleaner(text)
    text = text_simplifier(filtered_words)
    return text


def extract_fields(raw: dict) -> dict:
    review_text = (raw.get("content") or "").strip()
    return {
        "review_id": raw.get("reviewId"),
        "rating": raw.get("score"),
        "review_text": review_text,
        "text_processed": text_preprocessing(review_text),
        "created_at": raw.get("at"),
    }


def process_reviews(raw_reviews: list) -> list:
    return [extract_fields(r) for r in raw_reviews]


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


def save_processed(records: list, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"could not write {path}: {e}") from e


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    try:
        raw = load_reviews(in_path)
        processed = process_reviews(raw)
        save_processed(processed, out_path)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Processed {len(processed)} reviews -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
