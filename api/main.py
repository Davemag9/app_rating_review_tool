from contextlib import asynccontextmanager
from typing import AsyncGenerator

from api.nltk_setup import ensure_nltk_data

ensure_nltk_data()

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.config import settings
from api.routers import insights, metrics, reviews
from frontend.app import build_ui

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="App Rating & Review API",
    description=(
        "REST API for collecting Google Play reviews, computing rating metrics, "
        "running AI-powered sentiment analysis, and downloading raw review data. "
        "All data is held in memory — no files or database required."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(reviews.router, prefix=API_PREFIX, tags=["Reviews"])
app.include_router(metrics.router, prefix=API_PREFIX, tags=["Metrics"])
app.include_router(insights.router, prefix=API_PREFIX, tags=["Insights"])


@app.get("/health", tags=["Health"], summary="Health check")
def health_check() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/frontend")


app = gr.mount_gradio_app(app, build_ui(), path="/frontend")
