from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cross_ticker_lab import AnalysisRequest, OrchestratorAgent, PRESETS, load_config


st.set_page_config(page_title="Cross-Ticker Intelligence Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.3rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #10233b 0%, #15354f 45%, #0f5257 100%);
        color: #f4fbff;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 38px rgba(7, 19, 32, 0.22);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -0.04em; }
    .hero p { margin: 0.5rem 0 0 0; color: rgba(244,251,255,0.86); }
    .section-note {
        padding: 0.8rem 1rem;
        border-radius: 16px;
        background: #f5f8fb;
        border: 1px solid #dbe5ee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(load_config())


@st.cache_data(show_spinner=False)
def run_analysis_cached(request_payload: dict) -> object:
    request = AnalysisRequest(
        tickers=list(request_payload["tickers"]),
        benchmark=request_payload["benchmark"],
        lookback_days=request_payload["lookback_days"],
        interval=request_payload["interval"],
        embedding_window=request_payload["embedding_window"],
        news_lookback_days=request_payload["news_lookback_days"],
        historical_lookback_days=request_payload["historical_lookback_days"],
        price_weight=request_payload["price_weight"],
        news_weight=request_payload["news_weight"],
        peer_groups=dict(request_payload["peer_groups"]),
        historical_start=date.fromisoformat(request_payload["historical_start"]) if request_payload["historical_start"] else None,
        historical_end=date.fromisoformat(request_payload["historical_end"]) if request_payload["historical_end"] else None,
        enable_news=request_payload["enable_news"],
    )
    return get_orchestrator().run(request)


def parse_tickers(raw_value: str) -> list[str]:
    return [token.strip().upper() for token in raw_value.replace("\n", ",").split(",") if token.strip()]


def format_tickers_for_input(tickers: list[str], per_line: int = 6) -> str:
    return "\n".join(", ".join(tickers[index : index + per_line]) for index in range(0, len(tickers), per_line))


def parse_peer_groups(raw_value: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in raw_value.splitlines():
        if ":" not in line:
            continue
        name, members = line.split(":", 1)
        parsed_members = parse_tickers(members)
        if parsed_members:
            groups[name.strip()] = parsed_members
    return groups


def make_heatmap(frame: pd.DataFrame, title: str, zmid: float | None = None) -> go.Figure:
    if frame.empty:
        return go.Figure()
    figure = px.imshow(
        frame.round(2),
        text_auto=".2f",
        color_continuous_scale="Tealgrn",
        aspect="auto",
        zmin=-1 if zmid == 0 else None,
        zmax=1 if zmid == 0 else None,
    )
    figure.update_layout(title=title, margin=dict(l=10, r=10, t=45, b=10), coloraxis_colorbar_title="")
    return figure


def make_cumulative_return_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    plot_frame = frame.reset_index().rename(columns={"index": "date"})
    melted = plot_frame.melt(id_vars=plot_frame.columns[0], var_name="ticker", value_name="cumulative_return")
    figure = px.line(melted, x=plot_frame.columns[0], y="cumulative_return", color="ticker", title="Cumulative Returns")
    figure.update_layout(margin=dict(l=10, r=10, t=45, b=10), yaxis_tickformat=".0%")
    return figure


def make_embedding_scatter(frame: pd.DataFrame, title: str) -> go.Figure:
    if frame.empty:
        return go.Figure()
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        size="outlier_score",
        hover_name="ticker",
        hover_data={"headline_count": True, "news_sentiment": ":.2f", "cluster": True, "outlier_score": ":.2f"},
        title=title,
        color_continuous_scale="Teal",
    )
    figure.update_traces(marker=dict(line=dict(width=1, color="#0d2137")))
    figure.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    return figure


def make_network_chart(embedding_frame: pd.DataFrame, edges: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if embedding_frame.empty:
        return figure
    coordinates = embedding_frame.set_index("ticker")[["x", "y"]]
    for _, edge in edges.iterrows():
        if edge["source"] not in coordinates.index or edge["target"] not in coordinates.index:
            continue
        figure.add_trace(
            go.Scatter(
                x=[coordinates.loc[edge["source"], "x"], coordinates.loc[edge["target"], "x"]],
                y=[coordinates.loc[edge["source"], "y"], coordinates.loc[edge["target"], "y"]],
                mode="lines",
                line=dict(width=max(1, edge["weight"] * 4), color="rgba(21, 82, 87, 0.35)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=embedding_frame["x"],
            y=embedding_frame["y"],
            mode="markers+text",
            text=embedding_frame["ticker"],
            textposition="top center",
            marker=dict(
                size=18,
                color=embedding_frame["cluster"],
                colorscale="Tealgrn",
                line=dict(width=1.2, color="#10233b"),
            ),
            hovertemplate="Ticker=%{text}<br>Cluster=%{marker.color}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(title="Similarity Network", margin=dict(l=10, r=10, t=45, b=10), xaxis_visible=False, yaxis_visible=False)
    return figure


def render_news_evidence(news_frame: pd.DataFrame) -> None:
    if news_frame.empty:
        st.info("No headline evidence is available for this run.")
        return
    for ticker, ticker_rows in news_frame.explode("related_tickers").groupby("related_tickers"):
        with st.expander(f"{ticker} headlines", expanded=False):
            for _, row in ticker_rows.sort_values("published_at", ascending=False).head(5).iterrows():
                published = pd.to_datetime(row["published_at"]).strftime("%Y-%m-%d %H:%M")
                st.markdown(
                    f"- [{row['title']}]({row['url']})  \n"
                    f"  `{row['theme']}` | sentiment `{row['sentiment']:.2f}` | {row['source']} | {published}"
                )


def main() -> None:
    config = load_config()
    preset_names = list(PRESETS.keys())
    preset = PRESETS[preset_names[0]]
    default_start = date.today() - timedelta(days=365)
    default_end = date.today()

    st.markdown(
        """
        <div class="hero">
            <h1>Cross-Ticker Intelligence Lab</h1>
            <p>Joint market, narrative, and embedding analysis across a basket. The demo is built to show what becomes visible only when several tickers are analyzed together.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The app can run on live Yahoo Finance data, or fall back to deterministic synthetic market and narrative inputs when live calls are unavailable.")

    with st.sidebar.form("controls"):
        selected_preset = st.selectbox("Demo basket", preset_names, index=0)
        preset = PRESETS[selected_preset]
        tickers_raw = st.text_area("Tickers", value=format_tickers_for_input(preset.tickers), height=110)
        benchmark = st.text_input("Benchmark ticker", value=preset.benchmark).upper()
        lookback_days = st.select_slider("Lookback period", options=[90, 120, 180, 252, 365], value=180)
        interval = st.selectbox("Interval", options=["1d"], index=0)
        embedding_window = st.slider("Rolling embedding window", min_value=15, max_value=60, value=30, step=5)
        news_lookback_days = st.slider("News lookback window", min_value=3, max_value=30, value=7)
        historical_lookback_days = st.slider("Historical analogue lookback", min_value=180, max_value=900, value=365, step=30)
        enable_news = st.toggle("Include news analysis", value=True)
        price_weight = st.slider("Price embedding weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
        news_weight = 1.0 - price_weight
        st.caption(f"Fused embedding weights: price `{price_weight:.2f}` | news `{news_weight:.2f}`")
        peer_groups_raw = st.text_area(
            "Optional peer groups",
            value="\n".join(f"{name}: {', '.join(members)}" for name, members in preset.peer_groups.items()),
            height=110,
        )
        historical_range = st.date_input("Analogue search date range", value=(default_start, default_end))
        submitted = st.form_submit_button("Run analysis", use_container_width=True)

    if isinstance(historical_range, tuple) and len(historical_range) == 2:
        historical_start, historical_end = historical_range
    else:
        historical_start, historical_end = None, None

    tickers = parse_tickers(tickers_raw)
    peer_groups = parse_peer_groups(peer_groups_raw)
    request = AnalysisRequest(
        tickers=tickers,
        benchmark=benchmark,
        lookback_days=lookback_days,
        interval=interval,
        embedding_window=embedding_window,
        news_lookback_days=news_lookback_days,
        historical_lookback_days=historical_lookback_days,
        price_weight=price_weight,
        news_weight=news_weight,
        peer_groups=peer_groups,
        historical_start=historical_start,
        historical_end=historical_end,
        enable_news=enable_news,
    )

    if submitted or "report" not in st.session_state or st.session_state.get("request_key") != request.to_cache_key():
        with st.spinner("Running cross-ticker market, narrative, and embedding analysis..."):
            st.session_state["report"] = run_analysis_cached(request.to_cache_key())
            st.session_state["request_key"] = request.to_cache_key()

    report = st.session_state.get("report")
    if report is None:
        st.stop()

    top_divergence = report.market.anomalies.index[0] if not report.market.anomalies.empty else "n/a"
    avg_corr = report.market.regime_summary["mean_pairwise_corr"]
    basket_return = report.market.regime_summary["basket_return_20d"]
    shared_theme = report.news.theme_summary.iloc[0]["theme"] if not report.news.theme_summary.empty else "No dominant shared theme"

    metric_columns = st.columns(4)
    metric_columns[0].metric("20D Basket Return", f"{basket_return:.1%}")
    metric_columns[1].metric("Mean Pairwise Correlation", f"{avg_corr:.2f}")
    metric_columns[2].metric("Top Divergence", top_divergence)
    metric_columns[3].metric("Dominant Shared Theme", shared_theme)

    st.markdown(
        f"""
        <div class="section-note">
            <strong>Executive read:</strong> {report.synthesis.executive_summary}<br/>
            <strong>Why basket analysis matters:</strong> {report.synthesis.why_multi_ticker_matters}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Natural-Language Query")
        query = st.text_input(
            "Ask a basket-level question",
            value="What is moving together and why?",
            help="The answer is generated from the already-computed market, embedding, and news analysis.",
        )
        if st.button("Answer question", type="primary"):
            response = get_orchestrator().answer_query(query, report)
            st.markdown(f"**Answer ({response.query_type.replace('_', ' ')}):** {response.answer}")
            if response.evidence:
                st.markdown("**Supporting evidence**")
                for item in response.evidence:
                    st.markdown(f"- {item}")

    tabs = st.tabs(
        [
            "Basket Overview",
            "Cross-Ticker Relationships",
            "Embedding View",
            "News & Narratives",
            "Alerts & Divergences",
            "Agent Trace",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.plotly_chart(make_cumulative_return_chart(report.market.cumulative_returns), use_container_width=True)
        with right:
            leader_table = report.market.latest_metrics.sort_values("return_20d", ascending=False).copy()
            leader_table = leader_table[["return_1d", "return_5d", "return_20d", "realized_vol", "drawdown", "relative_strength"]]
            st.dataframe(leader_table.style.format("{:.2%}"), use_container_width=True)
        st.markdown("**Ranked cross-ticker insights**")
        for insight in report.insights.ranked[:5]:
            with st.container(border=True):
                st.markdown(f"**{insight.title}**")
                st.write(insight.description)
                for item in insight.evidence:
                    st.markdown(f"- {item}")

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_heatmap(report.market.corr_matrix, "Pairwise Correlation Heatmap", zmid=0), use_container_width=True)
            if not report.market.lead_lag.empty:
                st.dataframe(report.market.lead_lag, use_container_width=True)
        with right:
            st.plotly_chart(make_heatmap(report.embeddings.similarity_matrix, "Fused Similarity Matrix"), use_container_width=True)
            st.plotly_chart(make_network_chart(report.embeddings.fused_embedding_frame, report.market.similarity_edges), use_container_width=True)

    with tabs[2]:
        left, right = st.columns([1.2, 0.8])
        with left:
            st.plotly_chart(make_embedding_scatter(report.embeddings.fused_embedding_frame, "Current Fused Embedding Projection"), use_container_width=True)
        with right:
            st.markdown("**Nearest neighbors**")
            st.dataframe(report.embeddings.neighbor_frame.style.format({"similarity": "{:.2f}"}), use_container_width=True)
            st.markdown("**Historical analogue panel**")
            if report.embeddings.analogue_frame.empty:
                st.info("Not enough historical windows to compute analogues for this run.")
            else:
                analogue_table = report.embeddings.analogue_frame.copy()
                analogue_table["analogue_end"] = pd.to_datetime(analogue_table["analogue_end"]).dt.date
                st.dataframe(
                    analogue_table.style.format(
                        {
                            "similarity": "{:.2f}",
                            "forward_5d_basket_return": "{:.2%}",
                            "forward_20d_basket_return": "{:.2%}",
                            "forward_5d_benchmark_return": "{:.2%}",
                        }
                    ),
                    use_container_width=True,
                )
                st.caption(f"Vector search backend: {report.embeddings.vector_index_backend}")

    with tabs[3]:
        left, right = st.columns([0.9, 1.1])
        with left:
            if report.news.theme_summary.empty:
                st.info("News analysis is disabled or the selected provider returned no headlines.")
            else:
                st.dataframe(
                    report.news.theme_summary.style.format({"average_sentiment": "{:.2f}"}),
                    use_container_width=True,
                )
                st.dataframe(
                    report.news.narrative_clusters.style.format({"average_sentiment": "{:.2f}"}),
                    use_container_width=True,
                )
        with right:
            render_news_evidence(report.news.news_frame)

    with tabs[4]:
        left, right = st.columns([0.9, 1.1])
        with left:
            st.dataframe(
                report.market.anomalies[["return_20d", "relative_strength", "avg_pair_corr", "volume_shock", "divergence_score"]].style.format(
                    {
                        "return_20d": "{:.2%}",
                        "relative_strength": "{:.2%}",
                        "avg_pair_corr": "{:.2f}",
                        "volume_shock": "{:.2f}",
                        "divergence_score": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )
            st.dataframe(report.embeddings.outlier_scores.style.format({"outlier_score": "{:.2f}"}), use_container_width=True)
        with right:
            if report.risk.alert_frame.empty:
                st.success("No high-severity alerts were triggered for this run.")
            else:
                st.dataframe(report.risk.alert_frame, use_container_width=True)
                for alert in report.risk.alerts[:5]:
                    with st.container(border=True):
                        st.markdown(f"**{alert.headline}**")
                        st.write(alert.reason)
                        for item in alert.evidence:
                            st.markdown(f"- {item}")

    with tabs[5]:
        for trace in report.traces:
            with st.expander(trace.agent_name, expanded=False):
                st.markdown(f"**Objective:** {trace.objective}")
                st.markdown(f"**Status:** `{trace.status}`")
                st.markdown("**Actions**")
                for action in trace.actions:
                    st.markdown(f"- {action}")
                st.markdown("**Metrics**")
                st.json(trace.metrics)

    if not config.openai_api_key:
        st.info("`OPENAI_API_KEY` is not set. The app still works with local deterministic synthesis and local news fallbacks.")


if __name__ == "__main__":
    main()
