import random
from datetime import datetime
from typing import Any

from google_play_scraper import Sort
from google_play_scraper import reviews as gps_reviews
from google_play_scraper.exceptions import NotFoundError

from api.services.store import store
from scripts.calculate_metrics import calculate_metrics
from scripts.generate_insights import (
    classify_all_sentiments,
    extract_negative_keywords,
    extract_nlp_keywords,
    generate_actionable_insights,
    make_client,
)
from scripts.process_reviews import process_reviews


def _sanitize(obj: Any) -> Any:
    """Recursively convert datetime objects so the data is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _fetch_pool(app_id: str, pool_size: int) -> list[dict]:
    pool: list[dict] = []
    token = None
    while len(pool) < pool_size:
        batch, token = gps_reviews(
            app_id,
            count=min(200, pool_size - len(pool)),
            sort=Sort.NEWEST,
            continuation_token=token,
        )
        if not batch:
            break
        pool.extend(batch)
        if not token or not token.token:
            break
    return pool[:pool_size]


def collect_reviews(app_id: str, sample: int, pool_size: int) -> dict:
    """
    Fetch → sanitize → process → compute metrics.
    Stores results in the in-memory store and returns the metrics dict.
    Raises NotFoundError if the app doesn't exist on Google Play.
    Raises RuntimeError on any other failure.
    """
    pool = _fetch_pool(app_id, pool_size)
    if not pool:
        raise RuntimeError("No reviews returned for this app.")

    n = min(sample, len(pool))
    sampled = _sanitize(random.sample(pool, n))

    processed = _sanitize(process_reviews(sampled))
    metrics = calculate_metrics(sampled)

    store.save(app_id, "reviews", sampled)
    store.save(app_id, "processed", processed)
    store.save(app_id, "metrics", metrics)

    return {"reviews_collected": n, **metrics}


def analyze_reviews(app_id: str) -> dict:
    """
    Run Gemini sentiment analysis + NLP keywords + actionable insights.
    Reads processed reviews from the store; writes insights back to the store.
    Raises RuntimeError if reviews haven't been collected yet.
    """
    processed = store.get(app_id, "processed")
    if not processed:
        raise RuntimeError(
            "No reviews found in memory. Collect reviews first via POST /collect."
        )

    client = make_client()

    sentiments = classify_all_sentiments(client, processed)
    counts = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative"),
        "total": len(sentiments),
    }

    all_texts = [
        (r.get("review_text") or r.get("text_processed") or "").strip()
        for r in processed
    ]
    negative_texts = [
        text for text, s in zip(all_texts, sentiments) if s == "negative" and text
    ]

    nlp_keywords = extract_nlp_keywords(negative_texts, [t for t in all_texts if t])
    llm_keywords = extract_negative_keywords(client, negative_texts)
    combined_kw = llm_keywords + [e["keyword"] for e in nlp_keywords["tfidf_unigrams"]]
    insights = generate_actionable_insights(client, counts, combined_kw)

    labelled = [{**r, "sentiment": s} for r, s in zip(processed, sentiments)]

    store.save(app_id, "processed", labelled)

    result = {
        "sentiment_counts": counts,
        "negative_keywords_llm": llm_keywords,
        "negative_keywords_nlp": nlp_keywords,
        "actionable_insights": insights,
        "reviews": labelled,
    }

    store.save(app_id, "insights", result)
    return result
