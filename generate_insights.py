import json
import os
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from string import punctuation

from dotenv import load_dotenv
from google import genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash"
MAX_SENTIMENT_BATCH = 50
MAX_NEGATIVE_CHARS = 8_000
_MAX_RETRIES = 4
_RETRY_BASE_SLEEP = 6

DEFAULT_IN = Path("./data/reviews_processed.json")
DEFAULT_OUT = Path("./data/insights.json")

# Prompts.
SENTIMENT_PROMPT = """\
You are a sentiment classifier for app reviews.

For EACH numbered review below, reply with ONLY a JSON array where every \
element has exactly two keys:
  "index"     – the review number (integer, 1-based)
  "sentiment" – one of: "positive", "neutral", "negative"

Do not include any explanation or extra text outside the JSON array.

Reviews:
{reviews}
"""


KEYWORD_PROMPT = """\
You are an NLP specialist analysing negative app reviews.

Below are negative reviews collected from an app store. Identify the 10 most \
common keywords or short phrases that capture recurring complaints or pain \
points. Return ONLY a JSON array of strings (the keywords/phrases), ordered \
from most to least frequent. No extra text.

Negative reviews:
{text}
"""


INSIGHTS_PROMPT = """\
You are a product manager analysing user feedback for a mobile app.

Sentiment summary:
  Total reviews : {total}
  Positive      : {positive} ({pos_pct:.1f}%)
  Neutral       : {neutral}  ({neu_pct:.1f}%)
  Negative      : {negative} ({neg_pct:.1f}%)

Top keywords / phrases in negative reviews:
{keywords}

Based on this data, provide 3 to 5 specific, actionable improvement \
suggestions the development team should prioritise. Format your response \
as a JSON array of objects, each with:
  "area"       – short label for the area of improvement (e.g. "Performance")
  "suggestion" – one concise sentence describing what to do

Return ONLY valid JSON, no markdown, no extra text.
"""


def make_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)


def _call(client: genai.Client, prompt: str) -> str:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" not in msg or attempt == _MAX_RETRIES:
                raise

            match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
            wait = float(match.group(1)) + 1 if match else _RETRY_BASE_SLEEP * (attempt + 1)
            print(
                f"  Rate limit hit – waiting {wait:.1f}s before retry "
                f"({attempt + 1}/{_MAX_RETRIES})…"
            )
            time.sleep(wait)
    raise RuntimeError("Exceeded maximum retries due to rate limiting")


def classify_sentiment_batch(
    client: genai.Client, batch: list[dict]
) -> list[dict]:
    numbered = "\n".join(
        f"{i+1}. {r.get('review_text') or r.get('text_processed') or ''}"
        for i, r in enumerate(batch)
    )
    raw = _call(client, SENTIMENT_PROMPT.format(reviews=numbered))

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    return parsed   # list of {"index": int, "sentiment": str}


def classify_all_sentiments(
    client: genai.Client, reviews: list[dict]
) -> list[str]:
    labels = ["neutral"] * len(reviews)

    for start in range(0, len(reviews), MAX_SENTIMENT_BATCH):
        batch = reviews[start : start + MAX_SENTIMENT_BATCH]
        results = classify_sentiment_batch(client, batch)
        for item in results:
            idx = item["index"] - 1 + start
            if 0 <= idx < len(labels):
                labels[idx] = item.get("sentiment", "neutral")

    return labels


# --------------------------------------------------------------------------- #
# NLP-based keyword extraction (no LLM calls)
# --------------------------------------------------------------------------- #
_STOP_WORDS = set(stopwords.words("english")) | set(punctuation) | {"'s", "n't", "'re", "'ve", "'ll", "...", "app"}
_TOP_N = 10


def _tokenize(text: str) -> list[str]:
    return [
        t.lower() for t in word_tokenize(text)
        if t.isalpha() and t.lower() not in _STOP_WORDS and len(t) > 2
    ]


def _top_tfidf(negative_texts: list[str], all_texts: list[str], top_n: int) -> list[dict]:
    """Words most characteristic of negative reviews vs the full corpus."""
    if len(negative_texts) < 2:
        return []

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
        max_features=500,
    )
    vec.fit(all_texts)
    neg_matrix = vec.transform(negative_texts)
    mean_scores = neg_matrix.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, mean_scores), key=lambda x: x[1], reverse=True)
    return [{"keyword": term, "score": round(float(score), 4)} for term, score in ranked[:top_n]]


def _top_ngrams(negative_texts: list[str], n: int, top_n: int) -> list[dict]:
    """Most frequent n-grams across all negative reviews."""
    all_ngrams: list[tuple] = []
    for text in negative_texts:
        tokens = _tokenize(text)
        all_ngrams.extend(ngrams(tokens, n))
    counts = Counter(all_ngrams)
    return [
        {"keyword": " ".join(gram), "count": cnt}
        for gram, cnt in counts.most_common(top_n)
        if cnt > 1  # only phrases that appear more than once
    ]


def extract_nlp_keywords(
    negative_texts: list[str],
    all_texts: list[str],
    top_n: int = _TOP_N,
) -> dict:
    """
    Returns a dict with three NLP-derived keyword lists:
      tfidf_unigrams – words distinctive to negative reviews (TF-IDF)
      bigrams        – most common 2-word phrases
      trigrams       – most common 3-word phrases
    """
    return {
        "tfidf_unigrams": _top_tfidf(negative_texts, all_texts, top_n),
        "bigrams": _top_ngrams(negative_texts, 2, top_n),
        "trigrams": _top_ngrams(negative_texts, 3, top_n),
    }


def extract_negative_keywords(
    client: genai.Client, negative_texts: list[str]
) -> list[str]:
    if not negative_texts:
        return []

    combined = "\n---\n".join(negative_texts)
    if len(combined) > MAX_NEGATIVE_CHARS:
        combined = combined[:MAX_NEGATIVE_CHARS]

    raw = _call(client, KEYWORD_PROMPT.format(text=combined))
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def generate_actionable_insights(
    client: genai.Client,
    counts: dict,
    keywords: list[str],
) -> list[dict]:
    total = counts["total"]
    pos = counts["positive"]
    neu = counts["neutral"]
    neg = counts["negative"]

    def pct(n):
        return 100.0 * n / total if total else 0.0

    kw_text = (
        ", ".join(keywords) if keywords else "(no negative reviews found)"
    )

    prompt = INSIGHTS_PROMPT.format(
        total=total,
        positive=pos,
        neutral=neu,
        negative=neg,
        pos_pct=pct(pos),
        neu_pct=pct(neu),
        neg_pct=pct(neg),
        keywords=kw_text,
    )

    raw = _call(client, prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


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


def save_insights(data: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"could not write {path}: {e}") from e


def print_insights(result: dict) -> None:
    sc = result["sentiment_counts"]
    total = sc["total"]

    def pct(n):
        return 100.0 * n / total if total else 0.0

    print("\n=== Sentiment Analysis ===")
    print(f"  Total     : {total}")
    print(f"  Positive  : {sc['positive']}  ({pct(sc['positive']):.1f}%)")
    print(f"  Neutral   : {sc['neutral']}   ({pct(sc['neutral']):.1f}%)")
    print(f"  Negative  : {sc['negative']}  ({pct(sc['negative']):.1f}%)")

    print("\n=== Keywords in Negative Reviews (LLM) ===")
    llm_kws = result.get("negative_keywords_llm", [])
    if llm_kws:
        for i, kw in enumerate(llm_kws, 1):
            print(f"  {i:2}. {kw}")
    else:
        print("  (none)")

    print("\n=== Keywords in Negative Reviews (NLP) ===")
    nlp = result.get("negative_keywords_nlp", {})

    print("  -- TF-IDF distinctive words --")
    for e in nlp.get("tfidf_unigrams", []):
        print(f"    {e['keyword']:<25}  score={e['score']:.4f}")

    print("  -- Top bigrams (2-word phrases) --")
    for e in nlp.get("bigrams", []):
        print(f"    {e['keyword']:<25}  count={e['count']}")

    print("  -- Top trigrams (3-word phrases) --")
    for e in nlp.get("trigrams", []):
        print(f"    {e['keyword']:<25}  count={e['count']}")

    print("\n=== Actionable Insights ===")
    for item in result["actionable_insights"]:
        area = item.get("area", "")
        suggestion = item.get("suggestion", "")
        print(f"  [{area}]")
        for line in textwrap.wrap(suggestion, width=70):
            print(f"    {line}")


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    try:
        reviews = load_reviews(in_path)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not reviews:
        print("Error: no reviews to analyse", file=sys.stderr)
        return 1

    print(f"Loaded {len(reviews)} reviews from {in_path}")
    print("Classifying sentiment (this may take a moment)…")

    client = make_client()

    try:
        sentiments = classify_all_sentiments(client, reviews)

        counts = {
            "positive": sentiments.count("positive"),
            "neutral": sentiments.count("neutral"),
            "negative": sentiments.count("negative"),
            "total": len(sentiments),
        }

        all_texts = [
            (r.get("review_text") or r.get("text_processed") or "").strip()
            for r in reviews
        ]
        negative_texts = [
            text for text, s in zip(all_texts, sentiments)
            if s == "negative" and text
        ]

        print("Extracting keywords via NLP (TF-IDF + n-grams)…")
        nlp_keywords = extract_nlp_keywords(negative_texts, [t for t in all_texts if t])

        print("Extracting keywords via Gemini LLM…")
        llm_keywords = extract_negative_keywords(client, negative_texts)

        # Combine both sources for the insights prompt
        combined_kw = llm_keywords + [e["keyword"] for e in nlp_keywords["tfidf_unigrams"]]

        print("Generating actionable insights…")
        insights = generate_actionable_insights(client, counts, combined_kw)

    except Exception as e:
        print(f"Gemini API error: {e}", file=sys.stderr)
        return 1

    labelled_reviews = [
        {**r, "sentiment": s} for r, s in zip(reviews, sentiments)
    ]

    result = {
        "sentiment_counts": counts,
        "negative_keywords_llm": llm_keywords,
        "negative_keywords_nlp": nlp_keywords,
        "actionable_insights": insights,
        "reviews": labelled_reviews,
    }

    try:
        save_insights(result, out_path)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_insights(result)
    print(f"\nSaved insights -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
