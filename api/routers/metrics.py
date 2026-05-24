import re

from fastapi import APIRouter, HTTPException, status

from api.schemas.metrics import MetricsResponse, StarBucket
from api.services.store import store

router = APIRouter()

_APP_RE = re.compile(r"^[a-zA-Z][\w]*(?:\.[\w]+)+$")


def _validate_app_id(app_id: str) -> None:
    if not _APP_RE.match(app_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid Android package name: {app_id!r}.",
        )


@router.get(
    "/apps/{app_id}/metrics",
    response_model=MetricsResponse,
    summary="Get rating metrics",
    description=(
        "Returns average rating, total review count, and per-star distribution. "
        "Requires reviews to have been collected first via POST /collect."
    ),
)
def get_metrics(app_id: str) -> MetricsResponse:
    _validate_app_id(app_id)
    data = store.get(app_id, "metrics")
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for '{app_id}'. Run POST /apps/{app_id}/collect first.",
        )

    return MetricsResponse(
        app_id=app_id,
        average_rating=data.get("average_rating"),
        total_reviews=data["total_reviews"],
        distribution={star: StarBucket(**b) for star, b in data["distribution"].items()},
    )
