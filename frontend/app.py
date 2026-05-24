"""Gradio UI for the App Rating & Review API."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.utils.app_id import parse_app_id

import gradio as gr
import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))

CSS = """
.gradio-container { max-width: min(1400px, 100%) !important; margin: auto; padding: 0 1rem; }
.stat-row { gap: 12px; align-items: stretch; }
.stat-row > .html-container { min-width: 0; flex: 1 1 0; }
.stat-box {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    background: #f9fafb;
    text-align: center;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}
.stat-box.wide { text-align: left; padding: 12px 16px; }
.stat-box .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.2;
    word-break: break-word;
    overflow-wrap: anywhere;
}
.stat-box .value.app-id {
    font-size: 1.05rem;
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: normal;
}
.stat-box .label { font-size: 0.85rem; color: #6b7280; margin-top: 4px; }
.section-title { font-weight: 600; margin: 12px 0 6px; color: #374151; }
.reviews-table .table-wrap { overflow-x: auto; width: 100%; }
.reviews-table table { width: 100%; min-width: 960px; table-layout: auto; }
.reviews-table th { white-space: nowrap !important; }
.reviews-table td:nth-child(1),
.reviews-table th:nth-child(1) { width: 3rem; min-width: 3rem; white-space: nowrap !important; }
.reviews-table td:nth-child(2),
.reviews-table th:nth-child(2) { width: 4rem; min-width: 4rem; white-space: nowrap !important; }
.reviews-table td:nth-child(3),
.reviews-table th:nth-child(3) { width: 6rem; min-width: 6rem; white-space: nowrap !important; }
.reviews-table td:nth-child(4),
.reviews-table th:nth-child(4) { width: 9rem; min-width: 9rem; white-space: nowrap !important; }
.reviews-table td:nth-child(5) { white-space: normal !important; word-break: break-word; min-width: 20rem; }
.input-panel .wrap { gap: 0.75rem; }
.input-panel .slider-row { gap: 1.5rem; align-items: flex-start; }
.input-panel .slider-row > div { flex: 1 1 280px; min-width: 0; }
.input-panel .app-input textarea,
.input-panel .app-input input {
    min-width: 0 !important;
    width: 100% !important;
}
"""

EMPTY_DIST = pd.DataFrame({"Stars": [], "Count": [], "Share (%)": []})
EMPTY_SENTIMENT = pd.DataFrame({"Sentiment": [], "Count": [], "Share (%)": []})
EMPTY_INSIGHTS = pd.DataFrame({"Area": [], "Suggestion": []})
EMPTY_KEYWORDS = pd.DataFrame({"Source": [], "Keyword": [], "Score": []})
EMPTY_REVIEWS = pd.DataFrame(
    columns=["#", "Rating", "Sentiment", "Date", "Review"]
)


def _format_api_error(status: int, detail: Any, *, context: str = "") -> str:
    if isinstance(detail, list):
        detail = detail[0].get("msg", str(detail[0])) if detail else "Validation failed."
    detail_text = str(detail).strip()

    if status == 404:
        if detail_text == "Not Found":
            return (
                "**Could not reach that API endpoint.** "
                "Make sure the API server is running (`uvicorn api.main:app --reload`) "
                "and you are using a package ID or Play Store URL — not a full URL in the path."
            )
        return f"**App not found on Google Play.** {detail_text}"

    if status == 422:
        return f"**Invalid input.** {detail_text}"

    if status == 502:
        return f"**Could not fetch reviews from Google Play.** {detail_text}"

    if status == 503:
        return f"**Service unavailable.** {detail_text}"

    prefix = f"**{context}** " if context else ""
    return f"{prefix}Request failed (HTTP {status}): {detail_text}"


def _request(method: str, path: str, **kwargs: Any) -> tuple[Any | None, str | None]:
    url = f"{API_BASE}{path}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return None, (
            f"**API unreachable** — could not connect to `{API_BASE}`. "
            "Start the server with: `uvicorn api.main:app --reload`"
        )
    except httpx.TimeoutException:
        return None, (
            "**Request timed out.** "
            "Collecting or analysing 100 reviews can take up to a minute — please try again."
        )

    if response.is_success:
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json(), None
        return response.text, None

    detail: Any = response.text
    try:
        detail = response.json().get("detail", detail)
    except Exception:
        pass
    return None, _format_api_error(response.status_code, detail)


def _resolve_app_id(state: str, textbox: str) -> tuple[str | None, str | None]:
    raw = (state or textbox or "").strip()
    if not raw:
        return None, None
    try:
        return parse_app_id(raw), None
    except ValueError as exc:
        return None, f"**Invalid input.** {exc}"


def _format_date(raw: Any) -> str:
    if not raw:
        return ""
    text = str(raw)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16]


def _sentiment_label(raw: str | None, rating: int | None) -> str:
    if raw:
        return raw.capitalize()
    return "Pending"


def _distribution_df(data: dict) -> pd.DataFrame:
    distribution = data.get("distribution") or {}
    rows = []
    for star in range(5, 0, -1):
        bucket = distribution.get(str(star), {})
        rows.append({
            "Stars": f"{star} ★",
            "Count": int(bucket.get("count", 0)),
            "Share (%)": float(bucket.get("percentage", 0.0)),
        })
    return pd.DataFrame(rows)


def _bar_chart_image(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    colors: list[str] | None = None,
) -> Image.Image | None:
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = df[x_col].astype(str).tolist()
    values = df[y_col].astype(float).tolist()
    bar_colors = colors or ["#6366f1"] * len(values)

    ax.bar(labels, values, color=bar_colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="600", color="#111827", pad=10)
    ax.set_ylabel(y_col, fontsize=11, color="#374151")
    ax.set_xlabel(x_col, fontsize=11, color="#374151")
    ax.tick_params(colors="#374151", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ymax = max(values) if values else 1
    ax.set_ylim(0, ymax * 1.12 if ymax else 1)
    for idx, value in enumerate(values):
        if value:
            ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=9, color="#111827")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


STAR_COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]
SENTIMENT_COLORS = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}


def _distribution_chart(data: dict) -> Image.Image | None:
    distribution = data.get("distribution") or {}
    rows = []
    colors = []
    for star in range(1, 6):
        bucket = distribution.get(str(star), {})
        rows.append({"Stars": f"{star} ★", "Count": int(bucket.get("count", 0))})
        colors.append(STAR_COLORS[star - 1])
    df = pd.DataFrame(rows)
    return _bar_chart_image(df, "Stars", "Count", "Reviews by star rating", colors)


def _sentiment_chart(sc: dict) -> Image.Image | None:
    df = _sentiment_df(sc)
    colors = [SENTIMENT_COLORS.get(label, "#6366f1") for label in df["Sentiment"]]
    return _bar_chart_image(df, "Sentiment", "Count", "Sentiment counts", colors)


def _stat_html(label: str, value: str, *, wide: bool = False, app_id: bool = False) -> str:
    box_class = "stat-box wide" if wide else "stat-box"
    value_class = "value app-id" if app_id else "value"
    return (
        f'<motionless class="{box_class}">'
        f'<div class="{value_class}">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    ).replace("motionless", "div")


def collect_reviews(app_input: str, sample: int, pool: int):
    raw = (app_input or "").strip()
    if not raw:
        empty = (
            "**Enter an app** — package name (e.g. `genesis.nebula`) or a Google Play Store URL.",
            _stat_html("Average rating", "—"),
            _stat_html("Reviews", "—"),
            _stat_html("App", "—", wide=True, app_id=True),
            EMPTY_DIST,
            None,
            "",
            EMPTY_REVIEWS,
        )
        return empty

    try:
        package = parse_app_id(raw)
    except ValueError as exc:
        return (
            f"**Invalid input.** {exc}",
            _stat_html("Average rating", "—"),
            _stat_html("Reviews", "—"),
            _stat_html("App", "—", wide=True, app_id=True),
            EMPTY_DIST,
            None,
            "",
            EMPTY_REVIEWS,
        )

    data, err = _request(
        "POST",
        "/collect",
        json={"app": raw, "sample": int(sample), "pool": int(pool)},
    )
    if err:
        return (
            err,
            _stat_html("Average rating", "—"),
            _stat_html("Reviews", "—"),
            _stat_html("App", package, wide=True, app_id=True),
            EMPTY_DIST,
            None,
            "",
            EMPTY_REVIEWS,
        )

    avg = data.get("average_rating")
    count = data.get("reviews_collected") or data.get("total_reviews", 0)
    resolved = data.get("app_id") or package
    status = f"✓ Collected **{count}** reviews for `{resolved}`. Run **Analyze** for sentiment labels."
    dist_df = _distribution_df(data)
    chart = _distribution_chart(data)

    return (
        status,
        _stat_html("Average rating", f"{avg:.2f} / 5" if avg is not None else "n/a"),
        _stat_html("Reviews", str(count)),
        _stat_html("App", resolved, wide=True, app_id=True),
        dist_df,
        chart,
        resolved,
        _build_reviews_df(resolved),
    )


def _sentiment_df(sc: dict) -> pd.DataFrame:
    total = sc.get("total") or 0
    rows = []
    for label, key in [("Positive", "positive"), ("Neutral", "neutral"), ("Negative", "negative")]:
        count = sc.get(key, 0)
        pct = round(100 * count / total, 1) if total else 0.0
        rows.append({"Sentiment": label, "Count": count, "Share (%)": pct})
    return pd.DataFrame(rows)


def _keywords_df(data: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for kw in data.get("negative_keywords_llm") or []:
        rows.append({"Source": "LLM", "Keyword": kw, "Score": "—"})

    nlp = data.get("negative_keywords_nlp") or {}
    for entry in nlp.get("tfidf_unigrams") or []:
        rows.append({"Source": "TF-IDF", "Keyword": entry["keyword"], "Score": str(entry["score"])})
    for entry in nlp.get("bigrams") or []:
        rows.append({"Source": "Bigram", "Keyword": entry["keyword"], "Score": str(entry["count"])})
    for entry in nlp.get("trigrams") or []:
        rows.append({"Source": "Trigram", "Keyword": entry["keyword"], "Score": str(entry["count"])})

    return pd.DataFrame(rows) if rows else EMPTY_KEYWORDS


def _insights_df(data: dict) -> pd.DataFrame:
    rows = [
        {"Area": item.get("area", ""), "Suggestion": item.get("suggestion", "")}
        for item in data.get("actionable_insights") or []
    ]
    return pd.DataFrame(rows) if rows else EMPTY_INSIGHTS


def run_analysis(app_state: str, app_input: str):
    app_id, parse_err = _resolve_app_id(app_state, app_input)
    if parse_err:
        return parse_err, EMPTY_SENTIMENT, None, EMPTY_KEYWORDS, EMPTY_INSIGHTS, EMPTY_REVIEWS
    if not app_id:
        return (
            "**Collect reviews first** — enter an app package ID or Play Store URL, then click **Collect reviews**.",
            EMPTY_SENTIMENT,
            None,
            EMPTY_KEYWORDS,
            EMPTY_INSIGHTS,
            EMPTY_REVIEWS,
        )

    data, err = _request("POST", f"/apps/{app_id}/analyze")
    if err:
        return err, EMPTY_SENTIMENT, None, EMPTY_KEYWORDS, EMPTY_INSIGHTS, EMPTY_REVIEWS

    sc = data["sentiment_counts"]
    status = (
        f"✓ Analysis complete — "
        f"**{sc['positive']}** positive, **{sc['neutral']}** neutral, **{sc['negative']}** negative."
    )
    sentiment_df = _sentiment_df(sc)
    return (
        status,
        sentiment_df,
        _sentiment_chart(sc),
        _keywords_df(data),
        _insights_df(data),
        _build_reviews_df(app_id),
    )


def _build_reviews_df(app_id: str) -> pd.DataFrame:
    data, err = _request("GET", f"/apps/{app_id}/reviews", params={"limit": 500, "offset": 0})
    if err or not data:
        return EMPTY_REVIEWS

    rows = []
    for i, r in enumerate(data.get("reviews") or [], start=1):
        rating = r.get("rating")
        rows.append({
            "#": i,
            "Rating": f"{rating}★" if rating is not None else "",
            "Sentiment": _sentiment_label(r.get("sentiment"), rating),
            "Date": _format_date(r.get("created_at")),
            "Review": (r.get("review_text") or "").strip(),
        })
    return pd.DataFrame(rows) if rows else EMPTY_REVIEWS


def load_reviews_table(app_state: str, app_input: str) -> pd.DataFrame:
    app_id, err = _resolve_app_id(app_state, app_input)
    if err or not app_id:
        return EMPTY_REVIEWS
    return _build_reviews_df(app_id)


def download_reviews(
    app_state: str, app_input: str, fmt: str, data: str
) -> tuple[str | None, str]:
    app_id, err = _resolve_app_id(app_state, app_input)
    if err:
        return None, err
    if not app_id:
        return None, "**Collect reviews first** — then download raw or processed data."

    url = f"{API_BASE}/apps/{app_id}/reviews/download"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(url, params={"format": fmt, "data": data})
    except httpx.ConnectError:
        return None, "**API unreachable** — start the server with `uvicorn api.main:app --reload`."

    if not response.is_success:
        detail: Any = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        return None, _format_api_error(response.status_code, detail)

    ext = "json" if fmt == "json" else "csv"
    label = "raw" if data == "raw" else "processed"
    fd, path = tempfile.mkstemp(
        suffix=f".{ext}", prefix=f"{app_id.replace('.', '_')}_{label}_"
    )
    os.close(fd)
    with open(path, "wb") as f:
        f.write(response.content)
    return path, f"Downloaded **{label}** {fmt.upper()} ({len(response.content):,} bytes)."


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="gray")

    with gr.Blocks(title="App Review Insights", theme=theme, css=CSS) as demo:
        gr.Markdown(
            "## App Rating & Review Tool\n"
            "Collect Google Play reviews, view rating metrics, and run AI sentiment analysis."
        )

        with gr.Group(elem_classes=["input-panel"]):
            app_id = gr.Textbox(
                label="App ID or Play Store URL",
                placeholder="genesis.nebula",
                elem_classes=["app-input"],
            )
            with gr.Row(elem_classes=["slider-row"]):
                sample = gr.Slider(
                    10, 500, value=100, step=10, label="Sample size", scale=1,
                )
                pool = gr.Slider(
                    100, 2000, value=500, step=50, label="Fetch pool", scale=1,
                )

        with gr.Row():
            collect_btn = gr.Button("1. Collect reviews", variant="primary")
            analyze_btn = gr.Button("2. Run analysis", variant="secondary")
            refresh_btn = gr.Button("Refresh table")

        collect_status = gr.Markdown()
        active_app = gr.State("")

        with gr.Tabs():
            with gr.Tab("Metrics"):
                with gr.Row(elem_classes=["stat-row"]):
                    stat_avg = gr.HTML(_stat_html("Average rating", "—"))
                    stat_count = gr.HTML(_stat_html("Reviews", "—"))
                stat_app = gr.HTML(_stat_html("App", "—", wide=True, app_id=True))
                gr.Markdown('<p class="section-title">Rating distribution</p>')
                dist_table = gr.Dataframe(
                    headers=["Stars", "Count", "Share (%)"],
                    interactive=False,
                    wrap=True,
                )
                dist_chart = gr.Image(label="Reviews by star rating", type="pil", height=320)

            with gr.Tab("Insights"):
                analysis_status = gr.Markdown()
                gr.Markdown('<p class="section-title">Sentiment breakdown</p>')
                with gr.Row():
                    sentiment_table = gr.Dataframe(
                        headers=["Sentiment", "Count", "Share (%)"],
                        interactive=False,
                    )
                    sentiment_chart = gr.Image(label="Sentiment counts", type="pil", height=320)
                gr.Markdown('<p class="section-title">Keywords in negative reviews</p>')
                keywords_table = gr.Dataframe(
                    headers=["Source", "Keyword", "Score"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown('<p class="section-title">Actionable improvements</p>')
                insights_table = gr.Dataframe(
                    headers=["Area", "Suggestion"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("Reviews"):
                gr.Markdown("*Run **Analyze** to add sentiment labels to processed downloads.*")
                reviews_table = gr.Dataframe(
                    headers=["#", "Rating", "Sentiment", "Date", "Review"],
                    interactive=False,
                    wrap=False,
                    elem_classes=["reviews-table"],
                )
                gr.Markdown('<p class="section-title">Download review data</p>')
                gr.Markdown(
                    "**Raw** — original Google Play fields (review text, score, user, date). "
                    "**Processed** — cleaned text, rating, and sentiment (if analysed)."
                )
                with gr.Row():
                    raw_json_dl = gr.DownloadButton("Raw JSON", variant="secondary")
                    raw_csv_dl = gr.DownloadButton("Raw CSV", variant="secondary")
                    proc_json_dl = gr.DownloadButton("Processed JSON", variant="secondary")
                    proc_csv_dl = gr.DownloadButton("Processed CSV", variant="secondary")
                dl_status = gr.Markdown()

        collect_btn.click(
            collect_reviews,
            inputs=[app_id, sample, pool],
            outputs=[
                collect_status,
                stat_avg,
                stat_count,
                stat_app,
                dist_table,
                dist_chart,
                active_app,
                reviews_table,
            ],
        )

        analyze_btn.click(
            run_analysis,
            inputs=[active_app, app_id],
            outputs=[
                analysis_status,
                sentiment_table,
                sentiment_chart,
                keywords_table,
                insights_table,
                reviews_table,
            ],
        )

        refresh_btn.click(load_reviews_table, inputs=[active_app, app_id], outputs=reviews_table)

        raw_json_dl.click(
            lambda s, i: download_reviews(s, i, "json", "raw"),
            inputs=[active_app, app_id],
            outputs=[raw_json_dl, dl_status],
        )
        raw_csv_dl.click(
            lambda s, i: download_reviews(s, i, "csv", "raw"),
            inputs=[active_app, app_id],
            outputs=[raw_csv_dl, dl_status],
        )
        proc_json_dl.click(
            lambda s, i: download_reviews(s, i, "json", "processed"),
            inputs=[active_app, app_id],
            outputs=[proc_json_dl, dl_status],
        )
        proc_csv_dl.click(
            lambda s, i: download_reviews(s, i, "csv", "processed"),
            inputs=[active_app, app_id],
            outputs=[proc_csv_dl, dl_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
    )
