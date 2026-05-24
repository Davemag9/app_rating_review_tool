from typing import Any

from pydantic import BaseModel, Field, model_validator


class CollectRequest(BaseModel):
    sample: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of reviews to randomly sample from the pool.",
    )
    pool: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Total reviews to fetch from Google Play before sampling.",
    )

    @model_validator(mode="after")
    def pool_must_cover_sample(self) -> "CollectRequest":
        if self.pool < self.sample:
            raise ValueError("`pool` must be greater than or equal to `sample`.")
        return self


class CollectAppRequest(CollectRequest):
    app: str = Field(
        description="Android package name (e.g. genesis.nebula) or Google Play Store URL.",
        examples=["genesis.nebula", "https://play.google.com/store/apps/details?id=genesis.nebula"],
    )


class ReviewItem(BaseModel):
    review_id: str | None = None
    rating: int | None = None
    review_text: str = ""
    text_processed: str = ""
    created_at: Any = None
    sentiment: str | None = None


class ReviewsListResponse(BaseModel):
    app_id: str
    total: int
    reviews: list[ReviewItem]
