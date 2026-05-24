from pydantic import BaseModel


class StarBucket(BaseModel):
    count: int
    percentage: float


class MetricsResponse(BaseModel):
    app_id: str
    reviews_collected: int | None = None
    average_rating: float | None
    total_reviews: int
    distribution: dict[str, StarBucket]
