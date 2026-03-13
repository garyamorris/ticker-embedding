from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class AnalysisRequest:
    tickers: list[str]
    benchmark: str
    lookback_days: int = 180
    interval: str = "1d"
    embedding_window: int = 30
    news_lookback_days: int = 7
    historical_lookback_days: int = 365
    price_weight: float = 0.65
    news_weight: float = 0.35
    peer_groups: dict[str, list[str]] = field(default_factory=dict)
    historical_start: date | None = None
    historical_end: date | None = None
    enable_news: bool = True

    @property
    def normalized_tickers(self) -> list[str]:
        return [ticker.strip().upper() for ticker in self.tickers if ticker.strip()]

    @property
    def all_tickers(self) -> list[str]:
        tickers = self.normalized_tickers.copy()
        if self.benchmark and self.benchmark.upper() not in tickers:
            tickers.append(self.benchmark.upper())
        return tickers

    @property
    def normalized_benchmark(self) -> str:
        return self.benchmark.strip().upper()

    @property
    def fetch_start(self) -> date:
        days = max(
            self.lookback_days + self.embedding_window + 45,
            self.historical_lookback_days + self.embedding_window + 45,
        )
        if self.historical_start:
            historical_days = (date.today() - self.historical_start).days + self.embedding_window + 10
            days = max(days, historical_days)
        return date.today() - timedelta(days=days)

    @property
    def fetch_end(self) -> date:
        return date.today() + timedelta(days=1)

    def to_cache_key(self) -> dict[str, Any]:
        return {
            "tickers": tuple(self.normalized_tickers),
            "benchmark": self.normalized_benchmark,
            "lookback_days": self.lookback_days,
            "interval": self.interval,
            "embedding_window": self.embedding_window,
            "news_lookback_days": self.news_lookback_days,
            "historical_lookback_days": self.historical_lookback_days,
            "price_weight": round(self.price_weight, 4),
            "news_weight": round(self.news_weight, 4),
            "peer_groups": tuple((name, tuple(values)) for name, values in sorted(self.peer_groups.items())),
            "historical_start": self.historical_start.isoformat() if self.historical_start else None,
            "historical_end": self.historical_end.isoformat() if self.historical_end else None,
            "enable_news": self.enable_news,
        }


@dataclass(slots=True)
class NewsItem:
    id: str
    ticker: str
    related_tickers: list[str]
    published_at: datetime
    title: str
    summary: str
    source: str
    url: str
    theme: str
    sentiment: float


@dataclass(slots=True)
class AgentTrace:
    agent_name: str
    objective: str
    actions: list[str]
    metrics: dict[str, Any]
    status: str = "completed"


@dataclass(slots=True)
class MarketReport:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    realized_vol: pd.DataFrame
    drawdown: pd.DataFrame
    rolling_beta: pd.DataFrame
    rolling_correlation: pd.DataFrame
    relative_strength: pd.DataFrame
    volume_shock: pd.DataFrame
    latest_metrics: pd.DataFrame
    corr_matrix: pd.DataFrame
    lead_lag: pd.DataFrame
    similarity_edges: pd.DataFrame
    basket_dispersion: pd.Series
    regime_summary: dict[str, Any]
    anomalies: pd.DataFrame


@dataclass(slots=True)
class NewsReport:
    items: list[NewsItem]
    news_frame: pd.DataFrame
    theme_summary: pd.DataFrame
    narrative_clusters: pd.DataFrame
    ticker_map: dict[str, list[NewsItem]]


@dataclass(slots=True)
class EmbeddingReport:
    price_window_frame: pd.DataFrame
    price_embedding_frame: pd.DataFrame
    news_embedding_frame: pd.DataFrame
    fused_embedding_frame: pd.DataFrame
    similarity_matrix: pd.DataFrame
    neighbor_frame: pd.DataFrame
    analogue_frame: pd.DataFrame
    outlier_scores: pd.DataFrame
    basket_embedding_frame: pd.DataFrame
    vector_index_backend: str


@dataclass(slots=True)
class Insight:
    category: str
    title: str
    description: str
    evidence: list[str]
    severity: int = 50


@dataclass(slots=True)
class InsightReport:
    synchronized_moves: list[Insight]
    divergences: list[Insight]
    analogues: list[Insight]
    narratives: list[Insight]
    regime: Insight
    basket_value: Insight
    ranked: list[Insight]


@dataclass(slots=True)
class RiskAlert:
    alert_type: str
    ticker: str
    severity: int
    headline: str
    reason: str
    evidence: list[str]


@dataclass(slots=True)
class RiskReport:
    alerts: list[RiskAlert]
    alert_frame: pd.DataFrame


@dataclass(slots=True)
class SynthesisReport:
    executive_summary: str
    deep_dive: str
    why_multi_ticker_matters: str


@dataclass(slots=True)
class QueryResponse:
    query_type: str
    answer: str
    evidence: list[str]


@dataclass(slots=True)
class OrchestrationReport:
    request: AnalysisRequest
    market: MarketReport
    news: NewsReport
    embeddings: EmbeddingReport
    insights: InsightReport
    synthesis: SynthesisReport
    risk: RiskReport
    traces: list[AgentTrace]


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def zero_matrix(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=float)
