import re

from fastapi import APIRouter, HTTPException, status

from api.config import settings
from api.schemas.insights import (
    ActionableInsight,
    InsightsResponse,
    NgramEntry,
    NlpKeywords,
    SentimentCounts,
    TfidfEntry,
)
from api.services.pipeline import analyze_reviews
from api.services.store import store

router = APIRouter()

_APP_RE = re.compile(r"^[a-zA-Z][\w]*(?:\.[\w]+)+$")


def _validate_app_id(app_id: str) -> None:
    if not _APP_RE.match(app_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid Android package name: {app_id!r}.",
        )


def _build_insights_response(app_id: str, data: dict) -> InsightsResponse:
    sc = data["sentiment_counts"]
    nlp = data.get("negative_keywords_nlp", {})
    return InsightsResponse(
        app_id=app_id,
        sentiment_counts=SentimentCounts(**sc),
        negative_keywords_llm=data.get("negative_keywords_llm", []),
        negative_keywords_nlp=NlpKeywords(
            tfidf_unigrams=[TfidfEntry(**e) for e in nlp.get("tfidf_unigrams", [])],
            bigrams=[NgramEntry(**e) for e in nlp.get("bigrams", [])],
            trigrams=[NgramEntry(**e) for e in nlp.get("trigrams", [])],
        ),
        actionable_insights=[
            ActionableInsight(**i) for i in data.get("actionable_insights", [])
        ],
    )


@router.post(
    "/apps/{app_id}/analyze",
    response_model=InsightsResponse,
    summary="Run AI sentiment analysis and insights",
    description=(
        "Runs Gemini sentiment classification, NLP keyword extraction, and "
        "actionable suggestions on the collected reviews. Requires `GEMINI_API_KEY` "
        "and reviews to have been collected first via POST /collect."
    ),
)
def analyze(app_id: str) -> InsightsResponse:
    _validate_app_id(app_id)
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured. Add it to your .env file.",
        )
    if not store.has(app_id, "processed"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reviews in memory for '{app_id}'. Run POST /collect first.",
        )

    try:
        result = analyze_reviews(app_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {exc}",
        )

    return _build_insights_response(app_id, result)


@router.get(
    "/apps/{app_id}/insights",
    response_model=InsightsResponse,
    summary="Get AI-generated insights",
    description=(
        "Returns sentiment analysis, NLP/LLM keyword extraction, and actionable "
        "improvement suggestions. Requires POST /analyze to have been run first."
    ),
)
def get_insights(app_id: str) -> InsightsResponse:
    _validate_app_id(app_id)
    data = store.get(app_id, "insights")
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No insights found for '{app_id}'. Run POST /apps/{app_id}/analyze first.",
        )

    return _build_insights_response(app_id, data)
