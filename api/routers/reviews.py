import csv
import io
import json
from typing import Generator

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from google_play_scraper.exceptions import NotFoundError

from api.config import settings
from api.schemas.metrics import MetricsResponse, StarBucket
from api.schemas.reviews import (
    CollectAppRequest,
    CollectRequest,
    ReviewItem,
    ReviewsListResponse,
)
from api.services.pipeline import collect_reviews
from api.services.store import store
from api.utils.app_id import PACKAGE_RE, parse_app_id

router = APIRouter()

# Constants.
_CSV_FIELDS_PROCESSED = [
    "review_id",
    "rating",
    "review_text",
    "text_processed",
    "created_at",
    "sentiment",
]
_RAW_CSV_FIELDS = [
    "reviewId",
    "userName",
    "content",
    "score",
    "thumbsUpCount",
    "at",
    "appVersion",
    "replyContent",
    "repliedAt",
]


def _csv_fieldnames(rows: list[dict], data: str) -> list[str]:
    preferred = _RAW_CSV_FIELDS if data == "raw" else _CSV_FIELDS_PROCESSED
    seen: set[str] = set()
    fields: list[str] = []
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields or preferred


def _get_download_rows(app_id: str, data: str) -> list[dict]:
    if data == "raw":
        rows = store.get(app_id, "reviews")
    else:
        rows = store.get(app_id, "processed")
    return rows


def _validate_app_id(app_id: str) -> str:
    if not PACKAGE_RE.match(app_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid Android package name: {app_id!r}. "
                "Use the format com.company.app (e.g. genesis.nebula). "
                "For Play Store URLs, use POST /collect with the URL in the request body."
            ),
        )
    return app_id


@router.post(
    "/collect",
    response_model=MetricsResponse,
    summary="Collect reviews for an app",
    description=(
        "Fetches reviews from Google Play using a package name or Play Store URL, "
        "normalises them, and computes rating metrics."
    ),
)
def collect_app(body: CollectAppRequest) -> MetricsResponse:
    try:
        app_id = parse_app_id(body.app)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        result = collect_reviews(app_id, body.sample, body.pool)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No app found on Google Play for '{app_id}'. "
                "Check the package name or URL — the app may be unavailable in your region "
                "or removed from the store."
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not fetch reviews: {exc}",
        )

    return MetricsResponse(
        app_id=app_id,
        reviews_collected=result["reviews_collected"],
        average_rating=result.get("average_rating"),
        total_reviews=result["total_reviews"],
        distribution={star: StarBucket(**b) for star, b in result["distribution"].items()},
    )


@router.post(
    "/apps/{app_id}/collect",
    response_model=MetricsResponse,
    summary="Collect reviews for an app",
    description=(
        "Fetches reviews from Google Play, normalises them, and computes rating "
        "metrics. All data is held in memory and returned immediately."
    ),
)
def collect(app_id: str, body: CollectRequest = CollectRequest()) -> MetricsResponse:
    app_id = _validate_app_id(app_id)
    try:
        result = collect_reviews(app_id, body.sample, body.pool)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No app found on Google Play for '{app_id}'. "
                "Check the package name — the app may be unavailable or removed."
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not fetch reviews: {exc}",
        )

    return MetricsResponse(
        app_id=app_id,
        reviews_collected=result["reviews_collected"],
        average_rating=result.get("average_rating"),
        total_reviews=result["total_reviews"],
        distribution={star: StarBucket(**b) for star, b in result["distribution"].items()},
    )


@router.get(
    "/apps/{app_id}/reviews/download",
    response_model=None,
    summary="Download review data",
    description=(
        "Download collected reviews as JSON or CSV. "
        "Use `data=raw` for original Google Play fields, or `data=processed` for "
        "normalised text, ratings, and sentiment (after analysis)."
    ),
    responses={
        200: {"description": "Streamed file (JSON or CSV)"},
        404: {"description": "No reviews collected yet for this app"},
    },
)
def download_reviews(
    app_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$", description="`json` or `csv`"),
    data: str = Query(
        default="raw",
        pattern="^(raw|processed)$",
        description="`raw` = scraper output; `processed` = cleaned fields (+ sentiment if analysed)",
    ),
) -> StreamingResponse:
    app_id = _validate_app_id(app_id)
    reviews = _get_download_rows(app_id, data)
    if not reviews:
        label = "raw" if data == "raw" else "processed"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No {label} reviews in memory for '{app_id}'. "
                "Collect reviews first via POST /api/v1/collect."
            ),
        )

    safe_name = app_id.replace(".", "_")
    suffix = "raw" if data == "raw" else "processed"

    if format == "json":
        def _stream_json() -> Generator[str, None, None]:
            yield json.dumps(reviews, indent=2, ensure_ascii=False)

        return StreamingResponse(
            content=_stream_json(),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}_{suffix}_reviews.json"'
            },
        )

    fieldnames = _csv_fieldnames(reviews, data)

    def _stream_csv() -> Generator[str, None, None]:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        yield buf.getvalue()
        for row in reviews:
            buf.seek(0)
            buf.truncate()
            writer.writerow(row)
            yield buf.getvalue()

    return StreamingResponse(
        content=_stream_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_{suffix}_reviews.csv"'
        },
    )


@router.get(
    "/apps/{app_id}/reviews",
    response_model=ReviewsListResponse,
    summary="List collected reviews",
    description="Returns paginated reviews from memory. Use `limit` and `offset` for paging.",
)
def list_reviews(
    app_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ReviewsListResponse:
    app_id = _validate_app_id(app_id)
    rows = store.get(app_id, "processed") or store.get(app_id, "reviews")
    if rows is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reviews in memory for '{app_id}'. Run POST /collect first.",
        )

    page = rows[offset : offset + limit]
    return ReviewsListResponse(
        app_id=app_id,
        total=len(rows),
        reviews=[ReviewItem(**r) for r in page],
    )
