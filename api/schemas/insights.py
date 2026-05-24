from pydantic import BaseModel


class SentimentCounts(BaseModel):
    positive: int
    neutral: int
    negative: int
    total: int


class TfidfEntry(BaseModel):
    keyword: str
    score: float


class NgramEntry(BaseModel):
    keyword: str
    count: int


class NlpKeywords(BaseModel):
    tfidf_unigrams: list[TfidfEntry]
    bigrams: list[NgramEntry]
    trigrams: list[NgramEntry]


class ActionableInsight(BaseModel):
    area: str
    suggestion: str


class InsightsResponse(BaseModel):
    app_id: str
    sentiment_counts: SentimentCounts
    negative_keywords_llm: list[str]
    negative_keywords_nlp: NlpKeywords
    actionable_insights: list[ActionableInsight]
