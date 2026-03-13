from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


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
        self.client = OpenAI(api_key=config.openai_api_key) if config.enable_llm_synthesis and config.openai_api_key and OpenAI is not None else None

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
        if self.client is not None:
            try:
                response = self.client.responses.create(
                    model=self.config.openai_response_model,
                    input=self._llm_prompt(query, orchestration_report, answer, evidence),
                )
                if getattr(response, "output_text", None):
                    answer = response.output_text
            except Exception:
                pass
        return QueryResponse(query_type=query_type, answer=answer, evidence=evidence)

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

    def answer_query(self, query: str, report: OrchestrationReport) -> QueryResponse:
        return self.synthesis_agent.answer_query(query, report)
