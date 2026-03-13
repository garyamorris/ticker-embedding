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
    :root {
        --surface-strong: linear-gradient(135deg, #131c29 0%, #172537 58%, #13353f 100%);
        --surface-elevated: linear-gradient(135deg, rgba(14, 20, 31, 0.98) 0%, rgba(17, 27, 40, 0.98) 58%, rgba(15, 43, 53, 0.98) 100%);
        --surface-control: rgba(10, 16, 24, 0.96);
        --surface-border: rgba(89, 212, 196, 0.22);
        --text-strong: #f4fbff;
        --text-soft: rgba(235, 248, 255, 0.86);
        --text-muted: rgba(188, 211, 224, 0.78);
        --accent: #59d4c4;
    }
    .stApp {
        --st-primary-color: #59d4c4;
        --st-background-color: #040b14;
        --st-secondary-background-color: rgba(17, 25, 39, 0.96);
        --st-text-color: #f4fbff;
    }
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #040b14 0%, #07101a 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121925 0%, #171d2a 100%);
        border-right: 1px solid rgba(89, 212, 196, 0.08);
    }
    .hero {
        padding: 1.3rem 1.6rem;
        border-radius: 20px;
        background: var(--surface-strong);
        color: var(--text-strong) !important;
        border: 1px solid rgba(89, 212, 196, 0.14);
        box-shadow: 0 18px 38px rgba(3, 9, 18, 0.28);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.04em;
        color: var(--text-strong) !important;
        text-shadow: 0 1px 0 rgba(8, 18, 32, 0.35);
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        color: var(--text-soft) !important;
    }
    .hero * {
        color: inherit !important;
    }
    .app-caption {
        margin: 0.45rem 0 1rem 0;
        color: var(--text-muted);
        font-size: 0.93rem;
    }
    .section-note {
        padding: 0.95rem 1.05rem;
        border-radius: 16px;
        background: var(--surface-elevated);
        border: 1px solid var(--surface-border);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 14px 28px rgba(3, 9, 18, 0.24);
        color: var(--text-strong) !important;
        line-height: 1.55;
    }
    .section-note * {
        color: inherit !important;
    }
    .section-note-row + .section-note-row {
        margin-top: 0.45rem;
    }
    .section-note-label {
        display: inline-block;
        margin-right: 0.35rem;
        color: var(--accent) !important;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .section-note-value {
        color: var(--text-strong) !important;
    }
    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #12323d 0%, #1a5159 100%) !important;
        color: #f4fbff !important;
        border: 1px solid rgba(89, 212, 196, 0.36) !important;
        box-shadow: 0 10px 22px rgba(7, 22, 30, 0.28);
        font-weight: 700 !important;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: rgba(159, 239, 229, 0.68) !important;
        box-shadow: 0 14px 26px rgba(7, 22, 30, 0.34);
        filter: brightness(1.06);
    }
    .stButton > button:focus-visible,
    .stFormSubmitButton > button:focus-visible {
        outline: 2px solid rgba(89, 212, 196, 0.45) !important;
        outline-offset: 2px !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div,
    [data-testid="stDateInputField"] {
        background: var(--surface-control) !important;
        border: 1px solid rgba(89, 212, 196, 0.18) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="base-input"]:focus-within > div,
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="textarea"]:focus-within > div,
    [data-testid="stDateInputField"]:focus-within {
        border-color: rgba(89, 212, 196, 0.42) !important;
        box-shadow: 0 0 0 1px rgba(89, 212, 196, 0.34), 0 0 0 4px rgba(89, 212, 196, 0.10) !important;
    }
    input,
    textarea {
        color: var(--text-strong) !important;
        caret-color: var(--accent) !important;
    }
    [data-baseweb="select"] * {
        color: var(--text-strong) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #59d4c4 !important;
        border: 2px solid rgba(6, 17, 25, 0.9) !important;
        box-shadow: 0 0 0 4px rgba(89, 212, 196, 0.14) !important;
    }
    button[data-baseweb="tab"] {
        color: var(--text-soft) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
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


def apply_plotly_theme(figure: go.Figure) -> go.Figure:
    axis_style = dict(
        gridcolor="rgba(120, 160, 182, 0.10)",
        zerolinecolor="rgba(120, 160, 182, 0.14)",
        linecolor="rgba(120, 160, 182, 0.18)",
    )
    figure.update_layout(
        paper_bgcolor="#040b14",
        plot_bgcolor="#040b14",
        font=dict(color="#f4fbff"),
        title=dict(font=dict(color="#f4fbff", size=18)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color="#f4fbff")),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#f4fbff"),
            title=dict(font=dict(color="#f4fbff")),
        ),
    )
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    return figure


def make_heatmap(frame: pd.DataFrame, title: str, zmid: float | None = None) -> go.Figure:
    if frame.empty:
        return go.Figure()
    figure = px.imshow(
        frame.round(2),
        text_auto=".2f",
        color_continuous_scale=["#baf3d2", "#7fe0c6", "#45c0c0", "#2b91b0"],
        aspect="auto",
        zmin=-1 if zmid == 0 else None,
        zmax=1 if zmid == 0 else None,
    )
    figure.update_layout(title=title, margin=dict(l=10, r=10, t=45, b=10), coloraxis_colorbar_title="")
    return apply_plotly_theme(figure)


def make_cumulative_return_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()
    plot_frame = frame.reset_index().rename(columns={"index": "date"})
    melted = plot_frame.melt(id_vars=plot_frame.columns[0], var_name="ticker", value_name="cumulative_return")
    figure = px.line(melted, x=plot_frame.columns[0], y="cumulative_return", color="ticker", title="Cumulative Returns")
    figure.update_layout(margin=dict(l=10, r=10, t=45, b=10), yaxis_tickformat=".0%")
    return apply_plotly_theme(figure)


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
        color_continuous_scale=["#7fe0c6", "#45c0c0", "#2b91b0"],
    )
    figure.update_traces(marker=dict(line=dict(width=1, color="#0d2137")))
    figure.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    return apply_plotly_theme(figure)


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
                line=dict(width=max(1, edge["weight"] * 4), color="rgba(35, 157, 170, 0.28)"),
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
                colorscale=[[0.0, "#4bc2be"], [0.5, "#2f93b2"], [1.0, "#baf3d2"]],
                line=dict(width=1.2, color="#0d1b2a"),
            ),
            hovertemplate="Ticker=%{text}<br>Cluster=%{marker.color}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(title="Similarity Network", margin=dict(l=10, r=10, t=45, b=10), xaxis_visible=False, yaxis_visible=False)
    return apply_plotly_theme(figure)


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
    st.markdown(
        '<div class="app-caption">The app can run on live Yahoo Finance data, or fall back to deterministic synthetic market and narrative inputs when live calls are unavailable.</div>',
        unsafe_allow_html=True,
    )

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
            <div class="section-note-row">
                <span class="section-note-label">Executive read:</span>
                <span class="section-note-value">{report.synthesis.executive_summary}</span>
            </div>
            <div class="section-note-row">
                <span class="section-note-label">Why basket analysis matters:</span>
                <span class="section-note-value">{report.synthesis.why_multi_ticker_matters}</span>
            </div>
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
