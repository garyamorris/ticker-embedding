from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .analytics import build_embedding_report, build_market_report, build_news_report, classify_query
from .config import AppConfig
from .models import (
    AgentTrace,
    AnalysisRequest,
    EmbeddingReport,
    Insight,
    InsightReport,
    MarketReport,
    NewsReport,
    OrchestrationReport,
    QueryResponse,
    RiskAlert,
    RiskReport,
    SynthesisReport,
)
from .providers import (
    EmbeddingProvider,
    LocalTfidfEmbeddingProvider,
    ResilientMarketDataProvider,
    ResilientNewsProvider,
    build_embedding_provider,
    build_openai_client,
)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


TICKER_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "apple inc"),
    "MSFT": ("microsoft",),
    "GOOGL": ("google", "alphabet"),
    "META": ("facebook", "meta platforms"),
    "ORCL": ("oracle",),
    "CRM": ("salesforce",),
    "NVDA": ("nvidia",),
    "AMD": ("advanced micro devices",),
    "AVGO": ("broadcom",),
    "TSM": ("tsmc", "taiwan semiconductor"),
    "ASML": ("asml holding",),
    "ARM": ("arm holdings",),
    "MU": ("micron", "micron technology"),
    "QCOM": ("qualcomm",),
    "MRVL": ("marvell",),
    "AMAT": ("applied materials",),
    "LRCX": ("lam research",),
    "KLAC": ("kla",),
    "INTC": ("intel",),
    "SMH": ("semiconductor etf", "vaneck semiconductor"),
    "QQQ": ("nasdaq 100 etf", "invesco qqq"),
    "XOM": ("exxon", "exxon mobil"),
    "CVX": ("chevron",),
    "COP": ("conocophillips",),
    "EOG": ("eog resources",),
    "OXY": ("occidental", "occidental petroleum"),
    "SLB": ("schlumberger",),
    "HAL": ("halliburton",),
    "BKR": ("baker hughes",),
    "VLO": ("valero",),
    "MPC": ("marathon petroleum",),
    "PSX": ("phillips 66",),
    "LNG": ("cheniere", "cheniere energy"),
    "EQT": ("eqt corporation",),
    "KMI": ("kinder morgan",),
    "WMB": ("williams companies",),
    "CF": ("cf industries",),
    "MOS": ("mosaic",),
    "NTR": ("nutrien",),
    "ICL": ("icl group",),
    "SHEL": ("shell plc",),
    "BP": ("bp plc", "british petroleum"),
    "TTE": ("totalenergies",),
    "XLE": ("energy etf", "energy select sector spdr"),
    "CL=F": ("wti crude", "crude futures", "oil futures"),
    "NG=F": ("natural gas futures", "natgas futures"),
}


class ReasoningLLMResponse(BaseModel):
    answer: str = Field(default="")
    reasoning: list[str] = Field(default_factory=list)
    counterpoints: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class BaseAgent:
    name = "Base Agent"
    objective = ""

    def trace(self, actions: list[str], metrics: dict[str, object], status: str = "completed") -> AgentTrace:
        return AgentTrace(agent_name=self.name, objective=self.objective, actions=actions, metrics=metrics, status=status)


class MarketDataAgent(BaseAgent):
    name = "Market Data Agent"
    objective = "Fetch and align basket timeseries, then compute cross-sectional market structure."

    def __init__(self, provider: ResilientMarketDataProvider) -> None:
        self.provider = provider

    def run(self, request: AnalysisRequest) -> tuple[MarketReport, AgentTrace]:
        frames = self.provider.fetch(request.all_tickers, request.fetch_start, request.fetch_end, request.interval)
        report = build_market_report(frames, request.normalized_tickers, request.normalized_benchmark, request.lookback_days)
        corr_values = report.corr_matrix.values
        avg_pair_corr = float(corr_values[np.triu_indices(len(corr_values), 1)].mean()) if len(corr_values) > 1 else 1.0
        actions = [
            f"Fetched OHLCV history for {len(frames)} symbols via {self.provider.last_provider_name}.",
            "Aligned timeseries on a shared trading calendar.",
            "Computed returns, volatility, drawdown, beta, benchmark correlation, dispersion, and lead/lag signals.",
        ]
        metrics = {
            "provider": self.provider.last_provider_name,
            "rows": len(report.prices),
            "avg_pair_corr": round(avg_pair_corr, 3),
            "regime": report.regime_summary["label"],
            "top_divergence": report.anomalies.index[0] if not report.anomalies.empty else "n/a",
        }
        return report, self.trace(actions, metrics)


class NewsAgent(BaseAgent):
    name = "News Agent"
    objective = "Fetch recent headlines, map them to tickers, deduplicate stories, and surface cross-basket narratives."

    def __init__(self, provider: ResilientNewsProvider) -> None:
        self.provider = provider

    def run(self, request: AnalysisRequest, market_report: MarketReport) -> tuple[NewsReport, AgentTrace]:
        if not request.enable_news:
            report = build_news_report([], request.normalized_tickers)
            return report, self.trace(["News analysis disabled by user input."], {"provider": "disabled", "headlines": 0}, status="skipped")

        start = datetime.today().date() - timedelta(days=request.news_lookback_days)
        items = self.provider.fetch(
            request.normalized_tickers,
            start=start,
            end=datetime.today().date(),
            limit_per_ticker=5,
            market_snapshot=market_report.latest_metrics,
        )
        report = build_news_report(items, request.normalized_tickers)
        actions = [
            f"Fetched headline evidence via {self.provider.last_provider_name}.",
            "Deduplicated repeated coverage and mapped related tickers per story.",
            "Grouped headlines into reusable cross-ticker narrative themes.",
        ]
        metrics = {
            "provider": self.provider.last_provider_name,
            "headlines": len(report.news_frame),
            "themes": int(report.theme_summary["theme"].nunique()) if not report.theme_summary.empty else 0,
        }
        return report, self.trace(actions, metrics)


class EmbeddingAgent(BaseAgent):
    name = "Embedding Agent"
    objective = "Build price, news, and fused embeddings; then run clustering, similarity search, and analogue retrieval."

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def run(self, request: AnalysisRequest, market_report: MarketReport, news_report: NewsReport):
        historical_start = pd.to_datetime(request.historical_start) if request.historical_start else None
        historical_end = pd.to_datetime(request.historical_end) if request.historical_end else None
        news_item_vectors = None
        if not news_report.news_frame.empty:
            texts = news_report.news_frame["embedding_text"].tolist()
            try:
                news_item_vectors = self.embedding_provider.fit_transform(texts)
            except Exception:
                self.embedding_provider = LocalTfidfEmbeddingProvider()
                news_item_vectors = self.embedding_provider.fit_transform(texts)
        (
            price_window_frame,
            current_price_embeddings,
            news_embedding_frame,
            fused_embedding_frame,
            similarity_matrix,
            neighbor_frame,
            outlier_scores,
            analogue_frame,
            basket_embedding_frame,
            backend,
        ) = build_embedding_report(
            market_report=market_report,
            news_report=news_report,
            benchmark=request.normalized_benchmark,
            embedding_window=request.embedding_window,
            price_weight=request.price_weight,
            news_weight=request.news_weight,
            news_item_vectors=news_item_vectors,
            historical_start=historical_start,
            historical_end=historical_end,
        )
        report = EmbeddingReport(
            price_window_frame=price_window_frame,
            price_embedding_frame=current_price_embeddings,
            news_embedding_frame=news_embedding_frame,
            fused_embedding_frame=fused_embedding_frame,
            similarity_matrix=similarity_matrix,
            neighbor_frame=neighbor_frame,
            analogue_frame=analogue_frame,
            outlier_scores=outlier_scores,
            basket_embedding_frame=basket_embedding_frame,
            vector_index_backend=backend,
        )
        actions = [
            "Constructed rolling-window price embeddings from normalized returns, volatility, volume, drawdown, beta, correlation, and momentum state.",
            "Built ticker-level narrative vectors from the basket's recent news themes.",
            f"Fused price and news representations, clustered current behavior, and searched historical analogue windows with {report.vector_index_backend}.",
        ]
        metrics = {
            "price_windows": len(report.price_window_frame),
            "current_clusters": int(report.fused_embedding_frame["cluster"].nunique()) if not report.fused_embedding_frame.empty else 0,
            "analogues": len(report.analogue_frame),
            "vector_search": report.vector_index_backend,
            "news_embedding_provider": self.embedding_provider.name,
        }
        return report, self.trace(actions, metrics)


class CrossTickerInsightAgent(BaseAgent):
    name = "Cross-Ticker Insight Agent"
    objective = "Reason across the basket to identify synchronization, divergence, analogue matches, and multi-name structure."

    def run(
        self,
        request: AnalysisRequest,
        market_report: MarketReport,
        news_report: NewsReport,
        embedding_report,
    ) -> tuple[InsightReport, AgentTrace]:
        synchronized = self._synchronized_moves(market_report, news_report, embedding_report)
        divergences = self._divergence_insights(request, market_report, news_report, embedding_report)
        analogues = self._analogue_insights(embedding_report)
        narratives = self._narrative_insights(news_report, market_report)
        regime = self._regime_insight(market_report)
        basket_value = self._basket_value_insight(market_report, news_report, embedding_report)
        ranked = sorted(
            synchronized + divergences + analogues + narratives + [regime, basket_value],
            key=lambda insight: insight.severity,
            reverse=True,
        )
        report = InsightReport(
            synchronized_moves=synchronized,
            divergences=divergences,
            analogues=analogues,
            narratives=narratives,
            regime=regime,
            basket_value=basket_value,
            ranked=ranked,
        )
        actions = [
            "Compared fused similarity clusters against realized correlation structure.",
            "Ranked divergence using outlier scores and peer-relative performance.",
            "Connected shared narratives to synchronized moves and analogue windows.",
        ]
        metrics = {
            "synchronized_groups": len(synchronized),
            "divergence_flags": len(divergences),
            "narrative_clusters": len(narratives),
            "top_regime": regime.title,
        }
        return report, self.trace(actions, metrics)

    def _synchronized_moves(self, market_report: MarketReport, news_report: NewsReport, embedding_report) -> list[Insight]:
        insights: list[Insight] = []
        frame = embedding_report.fused_embedding_frame
        if frame.empty:
            return insights
        for cluster_id, cluster in frame.groupby("cluster"):
            tickers = cluster["ticker"].tolist()
            if len(tickers) < 2:
                continue
            cluster_corr = market_report.corr_matrix.loc[tickers, tickers].values
            avg_corr = float(cluster_corr[np.triu_indices(len(tickers), 1)].mean()) if len(tickers) > 1 else 1.0
            theme_hits = []
            if not news_report.theme_summary.empty:
                related = news_report.news_frame.explode("related_tickers")
                theme_hits = related.loc[related["related_tickers"].isin(tickers), "theme"].value_counts().head(2).index.tolist()
            insights.append(
                Insight(
                    category="synchronized_moves",
                    title=f"{', '.join(tickers)} are moving together",
                    description=f"Cluster {cluster_id} has an average correlation of {avg_corr:.2f}. The group shares the same current embedding neighborhood, which means the basket is seeing common state rather than isolated moves.",
                    evidence=[
                        f"Tickers in cluster: {', '.join(tickers)}.",
                        f"Average pairwise correlation: {avg_corr:.2f}.",
                        f"Shared narrative themes: {', '.join(theme_hits) if theme_hits else 'No dominant shared theme detected.'}",
                    ],
                    severity=min(95, int(60 + avg_corr * 35)),
                )
            )
        return insights

    def _divergence_insights(self, request: AnalysisRequest, market_report: MarketReport, news_report: NewsReport, embedding_report) -> list[Insight]:
        insights: list[Insight] = []
        if embedding_report.outlier_scores.empty:
            return insights
        for _, row in embedding_report.outlier_scores.head(2).iterrows():
            ticker = row["ticker"]
            score = float(row["outlier_score"])
            peer_group = self._find_peer_group(request, ticker) or self._cluster_peers(embedding_report, ticker)
            peer_return = market_report.latest_metrics.loc[peer_group, "return_20d"].mean() if peer_group else 0.0
            ticker_return = float(market_report.latest_metrics.loc[ticker, "return_20d"])
            ticker_headlines = len(news_report.ticker_map.get(ticker, []))
            support = "supported by ticker-specific news" if ticker_headlines >= 2 else "lightly supported by recent headlines"
            insights.append(
                Insight(
                    category="divergence",
                    title=f"{ticker} is diverging from its usual peer set",
                    description=f"{ticker} ranks as a basket outlier with embedding distance {score:.2f}. Its 20-day return is {ticker_return:.1%} versus {peer_return:.1%} for peers, which points to a move that is no longer explained by the broad cluster alone.",
                    evidence=[
                        f"Peer reference set: {', '.join(peer_group) if peer_group else 'Cluster-based peers unavailable.'}",
                        f"Outlier score: {score:.2f}.",
                        f"News support: {support}.",
                    ],
                    severity=min(92, int(58 + score * 40)),
                )
            )
        return insights

    def _analogue_insights(self, embedding_report) -> list[Insight]:
        insights: list[Insight] = []
        if embedding_report.analogue_frame.empty:
            return insights
        top = embedding_report.analogue_frame.head(3)
        mean_forward = top["forward_5d_basket_return"].mean()
        best_date = top.iloc[0]["analogue_end"].date()
        insights.append(
            Insight(
                category="historical_analogue",
                title=f"The current basket most resembles {best_date.isoformat()}",
                description=f"The top analogue set implies an average forward 5-day basket move of {mean_forward:.1%}. This is descriptive, not predictive, but it frames what historically followed similar cross-ticker geometry.",
                evidence=[
                    f"Top analogue date: {best_date.isoformat()}.",
                    f"Average 5-day forward basket return across top 3 matches: {mean_forward:.1%}.",
                    f"Average 20-day forward basket return across top 3 matches: {top['forward_20d_basket_return'].mean():.1%}.",
                ],
                severity=67,
            )
        )
        return insights

    def _narrative_insights(self, news_report: NewsReport, market_report: MarketReport) -> list[Insight]:
        insights: list[Insight] = []
        if news_report.theme_summary.empty:
            return insights
        for _, row in news_report.theme_summary.head(2).iterrows():
            theme = row["theme"]
            if int(row["ticker_count"]) < 2:
                continue
            exposed = news_report.news_frame.explode("related_tickers")
            exposed_tickers = sorted(set(exposed.loc[exposed["theme"] == theme, "related_tickers"].dropna().tolist()))
            basket_move = market_report.latest_metrics.loc[exposed_tickers, "return_5d"].mean() if exposed_tickers else 0.0
            insights.append(
                Insight(
                    category="narrative",
                    title=f"{theme} is propagating across the basket",
                    description=f"{theme} appears across {int(row['ticker_count'])} tickers and {int(row['headline_count'])} deduplicated headlines. The exposed names are up {basket_move:.1%} on average over the last 5 sessions, suggesting narrative propagation rather than isolated company news.",
                    evidence=[
                        f"Exposed tickers: {', '.join(exposed_tickers)}.",
                        f"Average sentiment: {float(row['average_sentiment']):.2f}.",
                        f"Sample headline: {row['sample_headline']}",
                    ],
                    severity=min(90, int(50 + row["ticker_count"] * 12 + row["headline_count"] * 3)),
                )
            )
        return insights

    def _regime_insight(self, market_report: MarketReport) -> Insight:
        regime = market_report.regime_summary
        return Insight(
            category="regime",
            title=f"{regime['label']} regime",
            description=f"Basket return over the last 20 sessions is {regime['basket_return_20d']:.1%} with mean pairwise correlation at {regime['mean_pairwise_corr']:.2f}. The dispersion profile is what separates this from a simple benchmark move.",
            evidence=[
                f"Benchmark return over 20 sessions: {regime['benchmark_return_20d']:.1%}.",
                f"Recent basket dispersion: {regime['recent_dispersion']:.4f}.",
                f"Historical average dispersion: {regime['historical_dispersion']:.4f}.",
            ],
            severity=64,
        )

    def _basket_value_insight(self, market_report: MarketReport, news_report: NewsReport, embedding_report) -> Insight:
        top_corr_pair = None
        if not market_report.similarity_edges.empty:
            edge = market_report.similarity_edges.sort_values("weight", ascending=False).iloc[0]
            top_corr_pair = f"{edge['source']} / {edge['target']} ({edge['weight']:.2f})"
        top_outlier = embedding_report.outlier_scores.iloc[0]["ticker"] if not embedding_report.outlier_scores.empty else "n/a"
        shared_theme = news_report.theme_summary.iloc[0]["theme"] if not news_report.theme_summary.empty else "no dominant shared theme"
        return Insight(
            category="basket_value",
            title="Why basket analysis matters",
            description="Joint analysis reveals structure that single-name workflows miss: synchronized leaders, peer breakouts, and narratives that propagate across names before every ticker reacts the same way.",
            evidence=[
                f"Strongest pair relationship: {top_corr_pair or 'not enough history for pair ranking.'}",
                f"Most divergent name: {top_outlier}.",
                f"Dominant shared theme: {shared_theme}.",
            ],
            severity=70,
        )

    @staticmethod
    def _cluster_peers(embedding_report, ticker: str) -> list[str]:
        frame = embedding_report.fused_embedding_frame
        cluster_id = int(frame.loc[frame["ticker"] == ticker, "cluster"].iloc[0])
        return frame.loc[(frame["cluster"] == cluster_id) & (frame["ticker"] != ticker), "ticker"].tolist()

    @staticmethod
    def _find_peer_group(request: AnalysisRequest, ticker: str) -> list[str]:
        for members in request.peer_groups.values():
            members_upper = [member.upper() for member in members]
            if ticker in members_upper:
                return [member for member in members_upper if member != ticker]
        return []


class NarrativeSynthesisAgent(BaseAgent):
    name = "Narrative Synthesis Agent"
    objective = "Convert quantitative findings plus news evidence into plain-English, evidence-backed market commentary."

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = build_openai_client(config.openai_api_key) if config.enable_llm_synthesis and config.openai_api_key and OpenAI is not None else None
        self._last_llm_error: str | None = None

    def run(self, insights: InsightReport) -> tuple[SynthesisReport, AgentTrace]:
        executive = " | ".join(insight.title for insight in insights.ranked[:3])
        deep_dive = "\n".join(f"- {insight.title}: {insight.description}" for insight in insights.ranked[:5])
        synthesis = SynthesisReport(
            executive_summary=executive,
            deep_dive=deep_dive,
            why_multi_ticker_matters=insights.basket_value.description,
        )
        actions = [
            "Compressed the ranked insight set into a short executive summary.",
            "Separated descriptive facts from inference in the long-form narrative.",
            "Prepared a reusable response layer for natural-language queries.",
        ]
        metrics = {"llm_enabled": bool(self.client), "summary_items": len(insights.ranked[:5])}
        return synthesis, self.trace(actions, metrics)

    def answer_query(self, query: str, orchestration_report: OrchestrationReport) -> QueryResponse:
        query_type = classify_query(query)
        answer, evidence = self._rule_based_answer(query_type, orchestration_report)
        model_name = "deterministic"
        warning = None
        if self.client is not None:
            try:
                response = self.client.responses.create(
                    model=self.config.openai_response_model,
                    input=self._llm_prompt(query, orchestration_report, answer, evidence),
                )
                if getattr(response, "output_text", None):
                    answer = response.output_text
                    model_name = self.config.openai_response_model
            except Exception as exc:
                warning = f"OpenAI synthesis unavailable: {self._format_exception(exc)}"
        return QueryResponse(query_type=query_type, answer=answer, evidence=evidence, mode="overview", model_name=model_name, warning=warning)

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        cause = getattr(exc, "__cause__", None)
        detail = str(cause or exc).strip()
        return detail if len(detail) <= 180 else f"{detail[:177]}..."

    def _rule_based_answer(self, query_type: str, report: OrchestrationReport) -> tuple[str, list[str]]:
        insights = report.insights
        if query_type == "synchronized_moves" and insights.synchronized_moves:
            top = insights.synchronized_moves[0]
            return top.description, top.evidence
        if query_type == "divergence" and insights.divergences:
            top = insights.divergences[0]
            return top.description, top.evidence
        if query_type == "historical_analogue" and insights.analogues:
            top = insights.analogues[0]
            return top.description, top.evidence
        if query_type == "narrative" and insights.narratives:
            top = insights.narratives[0]
            return top.description, top.evidence
        if query_type == "regime":
            return insights.regime.description, insights.regime.evidence
        top_ranked = insights.ranked[:3]
        answer = " ".join(insight.description for insight in top_ranked)
        evidence = [item for insight in top_ranked for item in insight.evidence[:1]]
        return answer, evidence

    @staticmethod
    def _llm_prompt(query: str, report: OrchestrationReport, fallback: str, evidence: list[str]) -> str:
        ranked = "\n".join(f"- {insight.title}: {insight.description}" for insight in report.insights.ranked[:6])
        evidence_block = "\n".join(f"- {item}" for item in evidence)
        return (
            "You are the Narrative Synthesis Agent for a cross-ticker intelligence demo.\n"
            "Answer the user question using only the supplied facts. Separate fact from inference, stay concise, and do not overclaim.\n"
            f"User question: {query}\n"
            f"Fallback answer: {fallback}\n"
            f"Ranked insights:\n{ranked}\n"
            f"Evidence:\n{evidence_block}\n"
        )


class ReasoningQueryAgent(BaseAgent):
    name = "Reasoning Query Agent"
    objective = "Answer questions with multi-step reasoning across basket regime, peer structure, narratives, analogues, and risk."

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = build_openai_client(config.openai_api_key) if config.enable_llm_synthesis and config.openai_api_key and OpenAI is not None else None

    def answer_query(self, query: str, report: OrchestrationReport, model_override: str | None = None) -> QueryResponse:
        query_type = classify_query(query)
        mentioned_tickers = self._extract_query_mentions(query, report)
        in_basket_tickers = [ticker for ticker in mentioned_tickers if ticker in report.request.normalized_tickers]
        off_basket_tickers = [ticker for ticker in mentioned_tickers if ticker not in report.request.normalized_tickers]
        model_name = "deterministic"
        warning = None

        if in_basket_tickers:
            answer, reasoning, counterpoints, evidence = self._focused_answer(query, report, in_basket_tickers[:2])
            if off_basket_tickers:
                ignored_names = ", ".join(off_basket_tickers[:2])
                counterpoints.insert(
                    0,
                    f"{ignored_names} {'is' if len(off_basket_tickers[:2]) == 1 else 'are'} not in the current basket, so the reasoning read only used current constituents.",
                )
        else:
            if off_basket_tickers:
                answer, reasoning, counterpoints, evidence = self._out_of_basket_answer(off_basket_tickers, report)
                model_name = "scope guard"
            else:
                answer, reasoning, counterpoints, evidence = self._basket_answer(query_type, report)

        if self.client is not None and model_name != "scope guard":
            selected_model = model_override or self.config.openai_reasoning_model
            llm_response = self._llm_answer(
                query=query,
                query_type=query_type,
                report=report,
                focus_tickers=in_basket_tickers[:2],
                selected_model=selected_model,
                fallback_answer=answer,
                fallback_reasoning=reasoning,
                fallback_counterpoints=counterpoints,
                fallback_evidence=evidence,
            )
            if llm_response is not None:
                answer, reasoning, counterpoints, evidence = llm_response
                model_name = selected_model
            else:
                model_name = "deterministic fallback"
                if self._last_llm_error is not None:
                    warning = f"OpenAI reasoning unavailable: {self._last_llm_error}"

        return QueryResponse(
            query_type=query_type,
            answer=answer,
            evidence=evidence,
            mode="reasoning",
            reasoning=reasoning,
            counterpoints=counterpoints,
            model_name=model_name,
            warning=warning,
        )

    def _basket_answer(self, query_type: str, report: OrchestrationReport) -> tuple[str, list[str], list[str], list[str]]:
        regime = report.market.regime_summary
        top_sync = report.insights.synchronized_moves[0] if report.insights.synchronized_moves else None
        top_divergences = report.insights.divergences[:2]
        top_narrative = report.insights.narratives[0] if report.insights.narratives else None
        top_analogue = report.insights.analogues[0] if report.insights.analogues else None
        top_risk = report.risk.alerts[0] if report.risk.alerts else None

        if query_type == "historical_analogue" and top_analogue is not None:
            answer = f"{top_analogue.description} The current regime is {regime['label'].lower()}, so the analogue matters mainly as context for the whole basket rather than for one isolated name."
        elif query_type == "narrative" and top_narrative is not None:
            answer = f"{top_narrative.description} The reason it matters is that the basket is still correlated enough for shared narrative propagation to move several names together."
        elif query_type == "regime":
            answer = f"The basket reads as {regime['label'].lower()}. Mean pairwise correlation is {regime['mean_pairwise_corr']:.2f}, 20-day basket return is {regime['basket_return_20d']:.1%}, and the current setup looks driven by common state before idiosyncratic stories fully separate."
        else:
            sync_summary = top_sync.description if top_sync is not None else "The basket still has common structure, but no dominant synchronized cluster stands out."
            narrative_summary = top_narrative.description if top_narrative is not None else "There is no single dominant mapped narrative."
            risk_summary = top_risk.reason if top_risk is not None else "No single alert dominates the current setup."
            answer = f"The basket is in a {regime['label'].lower()} regime. {sync_summary} {narrative_summary} The main thing that could break the current read is {risk_summary.lower()}"

        reasoning: list[str] = [
            f"Regime: 20-day basket return is {regime['basket_return_20d']:.1%}, mean pairwise correlation is {regime['mean_pairwise_corr']:.2f}, and recent dispersion is {regime['recent_dispersion']:.4f} versus {regime['historical_dispersion']:.4f} historically.",
        ]
        if top_sync is not None:
            reasoning.append(f"Structure: {top_sync.description}")
        if top_divergences:
            divergence_names = ", ".join(insight.title.split(' is ')[0] for insight in top_divergences)
            reasoning.append(f"Divergence: the main names breaking from the pack are {divergence_names}, which means the basket is not moving as a single undifferentiated beta trade.")
        if top_narrative is not None:
            reasoning.append(f"Narratives: {top_narrative.description}")
        if top_analogue is not None:
            reasoning.append(f"Historical context: {top_analogue.description}")
        if top_risk is not None:
            reasoning.append(f"Risk check: {top_risk.reason}")

        counterpoints = [
            "If pairwise correlation falls and the current divergence names normalize, the basket-level story becomes less coherent and more stock-specific.",
        ]
        if top_risk is not None and top_risk.ticker == "BASKET":
            counterpoints.append("Cross-sectional dispersion is already elevated, so single-name views can override the basket narrative faster than usual.")
        elif regime["mean_pairwise_corr"] >= 0.6:
            counterpoints.append("Correlation is still high enough that broad beta can overwhelm a seemingly clean narrative read.")
        if top_analogue is None:
            counterpoints.append("There is no strong historical analogue anchor, so this answer leans more on current structure than on prior episodes.")

        evidence = self._dedupe_items(
            [
                *(top_sync.evidence[:2] if top_sync is not None else []),
                *(top_narrative.evidence[:2] if top_narrative is not None else []),
                *(top_analogue.evidence[:1] if top_analogue is not None else []),
                *(top_risk.evidence[:2] if top_risk is not None else []),
            ]
        )
        return answer, reasoning, counterpoints, evidence

    def _focused_answer(self, query: str, report: OrchestrationReport, focus_tickers: list[str]) -> tuple[str, list[str], list[str], list[str]]:
        advice_terms = ("buy", "sell", "long", "short", "bullish", "bearish", "add", "trim")
        advice_preface = "This is a basket-relative read, not a standalone buy or sell recommendation. " if any(term in query.lower() for term in advice_terms) else ""
        regime = report.market.regime_summary
        analogue_frame = report.embeddings.analogue_frame
        analogue_hint = ""
        if not analogue_frame.empty:
            analogue_mean = analogue_frame.head(3)["forward_5d_basket_return"].mean()
            analogue_hint = f" The nearest analogue set implies an average forward 5-day basket move of {analogue_mean:.1%}, which is context for the backdrop rather than a prediction."

        answer_parts: list[str] = []
        reasoning: list[str] = []
        counterpoints: list[str] = []
        evidence: list[str] = []

        for ticker in focus_tickers:
            snapshot = self._ticker_snapshot(report, ticker)
            if snapshot is None:
                continue

            if snapshot["is_divergent"]:
                if snapshot["return_20d"] > snapshot["peer_return_20d"] and snapshot["headline_count"] >= 2:
                    stance = "is breaking out versus its peers with narrative confirmation"
                elif snapshot["return_20d"] > snapshot["peer_return_20d"]:
                    stance = "is outperforming peers, but the move is light on mapped narrative confirmation"
                elif snapshot["return_20d"] < snapshot["peer_return_20d"]:
                    stance = "is lagging its peers and currently reads as a relative weak spot"
                else:
                    stance = "is diverging from its peers, so the name is trading more idiosyncratically than the rest of the basket"
            elif snapshot["avg_pair_corr"] >= regime["mean_pairwise_corr"]:
                stance = "is still trading mostly with the basket rather than on a standalone catalyst"
            else:
                stance = "is not the main driver of the basket right now"

            answer_parts.append(
                f"{ticker} {stance}. Its 20-day return is {snapshot['return_20d']:.1%} versus {snapshot['peer_return_20d']:.1%} for {snapshot['peer_label']}, and its dominant mapped theme is {snapshot['dominant_theme']}."
            )
            reasoning.append(
                f"{ticker} peer context: {snapshot['peer_label']} are at {snapshot['peer_return_20d']:.1%} over 20 days versus {snapshot['return_20d']:.1%} for {ticker}; relative strength is {snapshot['relative_strength']:.1%} and average pairwise correlation is {snapshot['avg_pair_corr']:.2f}."
            )
            reasoning.append(
                f"{ticker} narrative context: mapped headline count is {snapshot['headline_count']}, dominant theme is {snapshot['dominant_theme']}, and nearest embedding neighbors are {snapshot['neighbors']}."
            )
            if snapshot["risk_reason"]:
                reasoning.append(f"{ticker} risk check: {snapshot['risk_reason']}")

            counterpoints.append(f"{ticker}: a new company-specific catalyst could override the current peer-relative read quickly.")
            if regime["mean_pairwise_corr"] >= 0.6:
                counterpoints.append(f"{ticker}: the basket is still correlated enough that market beta can dominate the single-name setup.")
            if snapshot["headline_count"] == 0:
                counterpoints.append(f"{ticker}: there is little mapped narrative support, so the read is coming more from structure than from confirmed catalyst flow.")

            evidence.extend(
                [
                    f"{ticker} 20-day return: {snapshot['return_20d']:.1%}.",
                    f"{ticker} peer reference ({snapshot['peer_label']}): {snapshot['peer_members'] or 'none available'}.",
                    f"{ticker} dominant theme: {snapshot['dominant_theme']}.",
                ]
            )
            if snapshot["risk_headline"]:
                evidence.append(snapshot["risk_headline"])

        if not answer_parts:
            return self._basket_answer(classify_query(query), report)

        answer = advice_preface + " ".join(answer_parts) + analogue_hint
        return answer, self._dedupe_items(reasoning), self._dedupe_items(counterpoints), self._dedupe_items(evidence)

    def _out_of_basket_answer(self, off_basket_tickers: list[str], report: OrchestrationReport) -> tuple[str, list[str], list[str], list[str]]:
        requested = ", ".join(off_basket_tickers[:2])
        basket_size = len(report.request.normalized_tickers)
        answer = (
            f"{requested} {'is' if len(off_basket_tickers[:2]) == 1 else 'are'} not in the current basket, so I cannot make a basket-relative inference for "
            f"{'that name' if len(off_basket_tickers[:2]) == 1 else 'those names'} from this report. "
            "Switch to a basket that contains it, or ask about one of the current constituents."
        )
        reasoning = [
            "The reasoning agent only uses the already-computed basket report, so peer structure, narrative propagation, analogues, and risk flags are all scoped to the current constituents.",
            f"The current basket contains {basket_size} symbols and excludes {requested}.",
        ]
        counterpoints = [
            "If you want a single-name inference for an outside ticker, add it to a basket so the answer can stay comparative instead of falling back to generic market knowledge.",
        ]
        evidence = [f"Current basket tickers: {', '.join(report.request.normalized_tickers)}."]
        return answer, reasoning, counterpoints, evidence

    def _llm_answer(
        self,
        query: str,
        query_type: str,
        report: OrchestrationReport,
        focus_tickers: list[str],
        selected_model: str,
        fallback_answer: str,
        fallback_reasoning: list[str],
        fallback_counterpoints: list[str],
        fallback_evidence: list[str],
    ) -> tuple[str, list[str], list[str], list[str]] | None:
        self._last_llm_error = None
        try:
            request_kwargs: dict[str, object] = {
                "model": selected_model,
                "instructions": (
                    "You are the Reasoning Query Agent for a cross-ticker intelligence application. "
                    "Use only the supplied cross-agent context. Do not invent facts. "
                    "Return valid JSON only with keys answer, reasoning, counterpoints, and evidence. "
                    "The reasoning list must contain concise public-facing rationale bullets, not hidden chain-of-thought."
                ),
                "input": self._llm_prompt(
                    query=query,
                    query_type=query_type,
                    report=report,
                    focus_tickers=focus_tickers,
                    fallback_answer=fallback_answer,
                    fallback_reasoning=fallback_reasoning,
                    fallback_counterpoints=fallback_counterpoints,
                    fallback_evidence=fallback_evidence,
                ),
                "max_output_tokens": 2200,
                "text_format": ReasoningLLMResponse,
            }
            if selected_model.startswith("gpt-5"):
                request_kwargs["reasoning"] = {"effort": self.config.openai_reasoning_effort}
            response = self.client.responses.parse(**request_kwargs)
        except Exception as exc:
            self._last_llm_error = NarrativeSynthesisAgent._format_exception(exc)
            return None

        payload = getattr(response, "output_parsed", None)
        if payload is None:
            self._last_llm_error = "Model response did not match the expected schema."
            return None

        answer = payload.answer.strip() or fallback_answer
        reasoning = self._coerce_items(payload.reasoning, fallback_reasoning, limit=6)
        counterpoints = self._coerce_items(payload.counterpoints, fallback_counterpoints, limit=4)
        evidence = self._coerce_items(payload.evidence, fallback_evidence, limit=8)
        return answer, reasoning, counterpoints, evidence

    def _llm_prompt(
        self,
        query: str,
        query_type: str,
        report: OrchestrationReport,
        focus_tickers: list[str],
        fallback_answer: str,
        fallback_reasoning: list[str],
        fallback_counterpoints: list[str],
        fallback_evidence: list[str],
    ) -> str:
        focus_block = "none"
        if focus_tickers:
            snapshots = []
            for ticker in focus_tickers:
                snapshot = self._ticker_snapshot(report, ticker)
                if snapshot is None:
                    continue
                snapshots.append(f"{ticker}: {json.dumps(snapshot, default=str)}")
            if snapshots:
                focus_block = "\n".join(snapshots)

        latest_metrics = self._select_frame_columns(
            report.market.latest_metrics,
            ["return_1d", "return_5d", "return_20d", "realized_vol", "drawdown", "relative_strength", "avg_pair_corr", "volume_shock"],
        )
        embedding_summary = self._select_frame_columns(
            report.embeddings.fused_embedding_frame,
            ["ticker", "x", "y", "cluster", "outlier_score", "headline_count", "news_sentiment"],
        )
        headline_summary = self._select_frame_columns(
            report.news.news_frame,
            ["published_at", "ticker", "related_tickers", "theme", "sentiment", "title", "source"],
        )
        anomaly_summary = self._select_frame_columns(
            report.market.anomalies,
            ["return_20d", "relative_strength", "avg_pair_corr", "volume_shock", "divergence_score"],
        )

        ranked_insights = "\n".join(
            f"- [{insight.category}] {insight.title}: {insight.description} | evidence={'; '.join(insight.evidence[:3])}"
            for insight in report.insights.ranked
        ) or "none"
        risk_summary = "\n".join(
            f"- {alert.ticker} | {alert.alert_type} | {alert.reason} | evidence={'; '.join(alert.evidence[:2])}"
            for alert in report.risk.alerts
        ) or "none"
        trace_summary = self._trace_block(report.traces)
        market_edge_summary = self._frame_to_csv(report.market.similarity_edges, max_rows=20)

        return (
            "Return JSON only in this shape:\n"
            '{"answer":"...", "reasoning":["..."], "counterpoints":["..."], "evidence":["..."]}\n'
            "Constraints:\n"
            "- Keep the answer concise and directly responsive.\n"
            "- Provide 3 to 6 reasoning bullets, 1 to 4 counterpoints, and 3 to 8 evidence bullets.\n"
            "- Use exact figures and ticker names when available.\n"
            "- If the question is basket-level, reason across the whole basket.\n"
            "- If the question is about a ticker in the basket, make the answer basket-relative rather than generic investment advice.\n"
            "- Do not reveal hidden chain-of-thought; provide concise external reasoning bullets only.\n\n"
            f"User question: {query}\n"
            f"Detected query type: {query_type}\n"
            f"Focus tickers in basket: {', '.join(focus_tickers) if focus_tickers else 'none'}\n\n"
            "Deterministic fallback scaffold:\n"
            f"Answer: {fallback_answer}\n"
            f"Reasoning: {json.dumps(fallback_reasoning, ensure_ascii=True)}\n"
            f"Counterpoints: {json.dumps(fallback_counterpoints, ensure_ascii=True)}\n"
            f"Evidence: {json.dumps(fallback_evidence, ensure_ascii=True)}\n\n"
            "Cross-agent context pack:\n"
            f"Request: {json.dumps({'tickers': report.request.normalized_tickers, 'benchmark': report.request.normalized_benchmark, 'lookback_days': report.request.lookback_days, 'embedding_window': report.request.embedding_window, 'news_enabled': report.request.enable_news, 'peer_groups': report.request.peer_groups}, default=str)}\n"
            f"Synthesis executive summary: {report.synthesis.executive_summary}\n"
            f"Synthesis why basket matters: {report.synthesis.why_multi_ticker_matters}\n"
            f"Synthesis deep dive: {report.synthesis.deep_dive}\n"
            f"Market regime summary: {json.dumps(report.market.regime_summary, default=str)}\n"
            f"Latest market metrics:\n{self._frame_to_csv(latest_metrics)}\n"
            f"Top correlation edges:\n{market_edge_summary}\n"
            f"Lead-lag table:\n{self._frame_to_csv(report.market.lead_lag, max_rows=20)}\n"
            f"Market anomalies:\n{self._frame_to_csv(anomaly_summary, max_rows=12)}\n"
            f"News theme summary:\n{self._frame_to_csv(report.news.theme_summary, max_rows=12)}\n"
            f"Narrative clusters:\n{self._frame_to_csv(report.news.narrative_clusters, max_rows=15)}\n"
            f"Recent headline evidence:\n{self._frame_to_csv(headline_summary, max_rows=18)}\n"
            f"Embedding summary:\n{self._frame_to_csv(embedding_summary)}\n"
            f"Nearest neighbors:\n{self._frame_to_csv(report.embeddings.neighbor_frame, max_rows=24)}\n"
            f"Outlier scores:\n{self._frame_to_csv(report.embeddings.outlier_scores, max_rows=12)}\n"
            f"Historical analogues:\n{self._frame_to_csv(report.embeddings.analogue_frame, max_rows=5)}\n"
            f"Vector search backend: {report.embeddings.vector_index_backend}\n"
            f"Focus ticker snapshots:\n{focus_block}\n"
            f"Ranked insights:\n{ranked_insights}\n"
            f"Risk alerts:\n{risk_summary}\n"
            f"Agent traces:\n{trace_summary}\n"
        )

    def _ticker_snapshot(self, report: OrchestrationReport, ticker: str) -> dict[str, object] | None:
        if ticker not in report.market.latest_metrics.index:
            return None

        row = report.market.latest_metrics.loc[ticker]
        peer_group = CrossTickerInsightAgent._find_peer_group(report.request, ticker)
        peer_label = "saved peer group"
        if not peer_group:
            peer_group = CrossTickerInsightAgent._cluster_peers(report.embeddings, ticker)
            peer_label = "cluster peers"
        peer_return_20d = float(report.market.latest_metrics.loc[peer_group, "return_20d"].mean()) if peer_group else float(report.market.regime_summary["basket_return_20d"])
        headline_items = report.news.ticker_map.get(ticker, [])
        dominant_theme = "no mapped theme"
        if headline_items:
            dominant_theme = pd.Series([item.theme for item in headline_items]).value_counts().index[0]
        neighbors = report.embeddings.neighbor_frame.loc[report.embeddings.neighbor_frame["ticker"] == ticker, "neighbor"].head(3).tolist()
        outlier_lookup = report.embeddings.outlier_scores.set_index("ticker")["outlier_score"] if not report.embeddings.outlier_scores.empty else pd.Series(dtype=float)
        outlier_threshold = float(outlier_lookup.quantile(0.75)) if not outlier_lookup.empty else 0.0
        outlier_score = float(outlier_lookup.get(ticker, 0.0))
        matching_alerts = [alert for alert in report.risk.alerts if alert.ticker == ticker]

        return {
            "return_20d": float(row["return_20d"]),
            "relative_strength": float(row["relative_strength"]),
            "avg_pair_corr": float(row["avg_pair_corr"]),
            "headline_count": len(headline_items),
            "dominant_theme": dominant_theme,
            "neighbors": ", ".join(neighbors) if neighbors else "no close neighbors",
            "peer_label": peer_label,
            "peer_members": ", ".join(peer_group),
            "peer_return_20d": peer_return_20d,
            "is_divergent": outlier_score >= outlier_threshold and outlier_threshold > 0,
            "risk_reason": matching_alerts[0].reason if matching_alerts else "",
            "risk_headline": matching_alerts[0].headline if matching_alerts else "",
        }

    @staticmethod
    def _extract_query_mentions(query: str, report: OrchestrationReport) -> list[str]:
        lowered = query.lower()
        matches: list[tuple[int, str]] = []
        candidates = sorted(set(TICKER_QUERY_ALIASES) | set(report.request.normalized_tickers))
        for ticker in candidates:
            tokens = (ticker.lower(),) + TICKER_QUERY_ALIASES.get(ticker, ())
            positions: list[int] = []
            for token in tokens:
                match = re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", lowered)
                if match is not None:
                    positions.append(match.start())
            if positions:
                matches.append((min(positions), ticker))
        ordered = [ticker for _, ticker in sorted(matches, key=lambda item: item[0])]
        return ReasoningQueryAgent._dedupe_items(ordered)

    @staticmethod
    def _dedupe_items(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _frame_to_csv(frame: pd.DataFrame, max_rows: int | None = None, round_digits: int = 4) -> str:
        if frame.empty:
            return "none"
        working = frame.copy()
        if max_rows is not None:
            working = working.head(max_rows)
        numeric_columns = working.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            working.loc[:, numeric_columns] = working.loc[:, numeric_columns].round(round_digits)
        if "published_at" in working.columns:
            working.loc[:, "published_at"] = pd.to_datetime(working["published_at"]).dt.strftime("%Y-%m-%d %H:%M")
        return working.to_csv(index=True)

    @staticmethod
    def _trace_block(traces: list[AgentTrace]) -> str:
        if not traces:
            return "none"
        blocks = []
        for trace in traces:
            blocks.append(
                (
                    f"{trace.agent_name} [{trace.status}]\n"
                    f"Objective: {trace.objective}\n"
                    f"Actions: {json.dumps(trace.actions[:3], ensure_ascii=True)}\n"
                    f"Metrics: {json.dumps(trace.metrics, default=str)}"
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _select_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        existing = [column for column in columns if column in frame.columns]
        if not existing:
            return pd.DataFrame()
        return frame.loc[:, existing].copy()

    @staticmethod
    def _coerce_items(value: object, fallback: list[str], limit: int) -> list[str]:
        if not isinstance(value, list):
            return fallback
        items = [str(item).strip() for item in value if str(item).strip()]
        return ReasoningQueryAgent._dedupe_items(items)[:limit] or fallback


class RiskAgent(BaseAgent):
    name = "Alert / Risk Agent"
    objective = "Flag unusual divergence, unsupported moves, regime shifts, and narrative-heavy setups."

    def run(self, report: OrchestrationReport) -> tuple[RiskReport, AgentTrace]:
        alerts: list[RiskAlert] = []
        if report.news.news_frame.empty:
            news_counts = pd.Series(dtype=int)
        else:
            news_counts = report.news.news_frame.explode("related_tickers")["related_tickers"].value_counts()

        outlier_lookup = report.embeddings.outlier_scores.set_index("ticker")["outlier_score"] if not report.embeddings.outlier_scores.empty else pd.Series(dtype=float)
        divergence_threshold = outlier_lookup.quantile(0.75) if not outlier_lookup.empty else None

        for ticker, row in report.market.latest_metrics.iterrows():
            move_5d = float(row["return_5d"])
            headlines = int(news_counts.get(ticker, 0))
            outlier = float(outlier_lookup.get(ticker, 0.0))
            if abs(move_5d) >= 0.08 and headlines == 0:
                alerts.append(
                    RiskAlert(
                        alert_type="unsupported_move",
                        ticker=ticker,
                        severity=84,
                        headline=f"{ticker} moved hard without matching news support",
                        reason=f"5-day move is {move_5d:.1%} with no mapped headlines in the selected news window.",
                        evidence=[f"5-day return: {move_5d:.1%}.", "Headline count: 0."],
                    )
                )
            if headlines >= 3 and abs(move_5d) <= 0.02:
                alerts.append(
                    RiskAlert(
                        alert_type="narrative_heavy",
                        ticker=ticker,
                        severity=68,
                        headline=f"{ticker} has narrative pressure without price follow-through",
                        reason=f"{headlines} mapped headlines landed while price stayed within {move_5d:.1%} over 5 days.",
                        evidence=[f"Headline count: {headlines}.", f"5-day return: {move_5d:.1%}."],
                    )
                )
            if divergence_threshold is not None and outlier >= divergence_threshold:
                alerts.append(
                    RiskAlert(
                        alert_type="divergence",
                        ticker=ticker,
                        severity=min(90, int(55 + outlier * 35)),
                        headline=f"{ticker} is the current divergence risk",
                        reason=f"Embedding outlier score is {outlier:.2f}, marking a break from the rest of the basket.",
                        evidence=[f"Outlier score: {outlier:.2f}.", f"Relative strength vs benchmark: {row['relative_strength']:.1%}."],
                    )
                )

        regime = report.market.regime_summary
        dispersion_ratio = regime["recent_dispersion"] / max(regime["historical_dispersion"], 1e-6)
        if dispersion_ratio > 1.3:
            alerts.append(
                RiskAlert(
                    alert_type="regime_shift",
                    ticker="BASKET",
                    severity=76,
                    headline="Cross-sectional dispersion has stepped higher",
                    reason=f"Recent dispersion is {dispersion_ratio:.2f}x the historical baseline, so ticker-specific risk is dominating beta.",
                    evidence=[
                        f"Recent dispersion: {regime['recent_dispersion']:.4f}.",
                        f"Historical dispersion: {regime['historical_dispersion']:.4f}.",
                    ],
                )
            )

        alert_frame = pd.DataFrame(
            [
                {
                    "alert_type": alert.alert_type,
                    "ticker": alert.ticker,
                    "severity": alert.severity,
                    "headline": alert.headline,
                    "reason": alert.reason,
                }
                for alert in sorted(alerts, key=lambda value: value.severity, reverse=True)
            ]
        )
        risk_report = RiskReport(alerts=sorted(alerts, key=lambda value: value.severity, reverse=True), alert_frame=alert_frame)
        actions = [
            "Scanned for unsupported moves, narrative-heavy setups, and cluster breaks.",
            "Compared recent basket dispersion to its own history to flag regime shifts.",
            "Ranked alerts by severity for the dashboard and query layer.",
        ]
        metrics = {"alerts": len(risk_report.alerts), "highest_severity": risk_report.alerts[0].severity if risk_report.alerts else 0}
        return risk_report, self.trace(actions, metrics)


class OrchestratorAgent(BaseAgent):
    name = "Orchestrator Agent"
    objective = "Receive the request, invoke the specialist agents in order, and combine their outputs into a coherent answer."

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.market_agent = MarketDataAgent(ResilientMarketDataProvider())
        self.news_agent = NewsAgent(ResilientNewsProvider())
        self.embedding_agent = EmbeddingAgent(build_embedding_provider(config.openai_api_key, config.openai_embedding_model))
        self.insight_agent = CrossTickerInsightAgent()
        self.synthesis_agent = NarrativeSynthesisAgent(config)
        self.reasoning_query_agent = ReasoningQueryAgent(config)
        self.risk_agent = RiskAgent()

    def run(self, request: AnalysisRequest) -> OrchestrationReport:
        traces: list[AgentTrace] = []
        market_report, market_trace = self.market_agent.run(request)
        traces.append(market_trace)

        news_report, news_trace = self.news_agent.run(request, market_report)
        traces.append(news_trace)

        embedding_report, embedding_trace = self.embedding_agent.run(request, market_report, news_report)
        traces.append(embedding_trace)

        insight_report, insight_trace = self.insight_agent.run(request, market_report, news_report, embedding_report)
        traces.append(insight_trace)

        synthesis_report, synthesis_trace = self.synthesis_agent.run(insight_report)
        traces.append(synthesis_trace)

        scaffold = OrchestrationReport(
            request=request,
            market=market_report,
            news=news_report,
            embeddings=embedding_report,
            insights=insight_report,
            synthesis=synthesis_report,
            risk=RiskReport(alerts=[], alert_frame=pd.DataFrame()),
            traces=[],
        )
        risk_report, risk_trace = self.risk_agent.run(scaffold)
        traces.append(risk_trace)

        scaffold.risk = risk_report
        scaffold.traces = traces + [
            self.trace(
                actions=[
                    "Sequenced market, news, embeddings, insights, synthesis, and risk analysis.",
                    "Preserved each agent trace for UI transparency.",
                    "Prepared the shared report object for natural-language queries.",
                ],
                metrics={
                    "tickers": len(request.normalized_tickers),
                    "news_enabled": request.enable_news,
                    "ranked_insights": len(scaffold.insights.ranked),
                },
            )
        ]
        return scaffold

    def answer_query(self, query: str, report: OrchestrationReport, mode: str = "overview", model_override: str | None = None) -> QueryResponse:
        if mode == "reasoning":
            return self.reasoning_query_agent.answer_query(query, report, model_override=model_override)
        return self.synthesis_agent.answer_query(query, report)
