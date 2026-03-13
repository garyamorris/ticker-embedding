from __future__ import annotations

from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

from .config import THEME_KEYWORDS
from .models import MarketReport, NewsItem, NewsReport, empty_frame
from .providers import VectorSearchIndex


ROLLING_WINDOW = 20


def align_market_data(raw_frames: dict[str, pd.DataFrame], tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_frames = []
    volume_frames = []
    for ticker in tickers:
        frame = raw_frames.get(ticker)
        if frame is None or frame.empty:
            continue
        close_frames.append(frame["close"].rename(ticker))
        if "volume" in frame:
            volume_frames.append(frame["volume"].rename(ticker))

    prices = pd.concat(close_frames, axis=1).sort_index().ffill().dropna(how="all")
    volumes = pd.concat(volume_frames, axis=1).sort_index().reindex(prices.index).ffill() if volume_frames else pd.DataFrame(index=prices.index)
    prices = prices.dropna(axis=0, how="any")
    volumes = volumes.ffill().fillna(0.0)
    return prices, volumes


def build_market_report(
    raw_frames: dict[str, pd.DataFrame],
    basket_tickers: list[str],
    benchmark: str,
    lookback_days: int,
) -> MarketReport:
    prices_all, volumes_all = align_market_data(raw_frames, list(raw_frames.keys()))
    analysis_columns = basket_tickers.copy()
    if benchmark not in prices_all.columns:
        raise ValueError(f"Benchmark {benchmark} is missing from the aligned price frame")
    analysis_columns_with_benchmark = analysis_columns if benchmark in analysis_columns else analysis_columns + [benchmark]

    prices = prices_all[analysis_columns_with_benchmark].copy()
    volumes = volumes_all.reindex(columns=analysis_columns_with_benchmark, fill_value=0.0)
    prices = prices.tail(lookback_days + ROLLING_WINDOW + 5)
    volumes = volumes.reindex(prices.index).fillna(0.0)

    returns = prices.pct_change().dropna()
    cumulative_returns = prices.div(prices.iloc[0]).sub(1.0)
    realized_vol = returns.rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    drawdown = prices.div(prices.cummax()).sub(1.0)
    rolling_correlation = returns.rolling(ROLLING_WINDOW).corr(returns[benchmark])

    benchmark_var = returns[benchmark].rolling(ROLLING_WINDOW).var().replace(0.0, np.nan)
    rolling_beta = pd.DataFrame(index=returns.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        covariance = returns[ticker].rolling(ROLLING_WINDOW).cov(returns[benchmark])
        rolling_beta[ticker] = covariance.div(benchmark_var)

    rel_strength = cumulative_returns.sub(cumulative_returns[benchmark], axis=0)
    if not volumes.empty:
        volume_mean = volumes.rolling(ROLLING_WINDOW).mean()
        volume_std = volumes.rolling(ROLLING_WINDOW).std().replace(0.0, np.nan)
        volume_shock = volumes.sub(volume_mean).div(volume_std)
    else:
        volume_shock = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    corr_matrix = returns[basket_tickers].corr()
    basket_dispersion = returns[basket_tickers].std(axis=1)
    avg_pair_corr = corr_matrix.mean(axis=1)

    latest_metrics = pd.DataFrame(index=basket_tickers)
    latest_metrics["return_1d"] = returns[basket_tickers].iloc[-1]
    latest_metrics["return_5d"] = prices[basket_tickers].pct_change(5).iloc[-1]
    latest_metrics["return_20d"] = prices[basket_tickers].pct_change(20).iloc[-1]
    latest_metrics["cumulative_return"] = cumulative_returns[basket_tickers].iloc[-1]
    latest_metrics["realized_vol"] = realized_vol[basket_tickers].iloc[-1]
    latest_metrics["drawdown"] = drawdown[basket_tickers].iloc[-1]
    latest_metrics["beta_to_benchmark"] = rolling_beta[basket_tickers].iloc[-1]
    latest_metrics["corr_to_benchmark"] = rolling_correlation[basket_tickers].iloc[-1]
    latest_metrics["relative_strength"] = rel_strength[basket_tickers].iloc[-1]
    latest_metrics["volume_shock"] = volume_shock[basket_tickers].iloc[-1]
    latest_metrics["avg_pair_corr"] = avg_pair_corr.reindex(basket_tickers)
    latest_metrics["dispersion_contrib"] = returns[basket_tickers].tail(20).std()
    latest_metrics = latest_metrics.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    divergence_score = (
        _zscore_series(latest_metrics["relative_strength"].abs()) * 0.4
        + _zscore_series(1 - latest_metrics["avg_pair_corr"]) * 0.4
        + _zscore_series(latest_metrics["volume_shock"].abs()) * 0.2
    )
    anomalies = latest_metrics.assign(divergence_score=divergence_score).sort_values("divergence_score", ascending=False)

    lead_lag = compute_lead_lag(returns[basket_tickers])
    similarity_edges = build_similarity_edges(corr_matrix)
    regime_summary = infer_regime(returns, basket_tickers, benchmark, basket_dispersion)

    return MarketReport(
        prices=prices[basket_tickers],
        volumes=volumes[basket_tickers],
        returns=returns[basket_tickers],
        cumulative_returns=cumulative_returns[basket_tickers],
        realized_vol=realized_vol[basket_tickers],
        drawdown=drawdown[basket_tickers],
        rolling_beta=rolling_beta[basket_tickers],
        rolling_correlation=rolling_correlation[basket_tickers],
        relative_strength=rel_strength[basket_tickers],
        volume_shock=volume_shock[basket_tickers],
        latest_metrics=latest_metrics.sort_values("return_20d", ascending=False),
        corr_matrix=corr_matrix,
        lead_lag=lead_lag,
        similarity_edges=similarity_edges,
        basket_dispersion=basket_dispersion,
        regime_summary=regime_summary,
        anomalies=anomalies,
    )


def build_news_report(items: list[NewsItem], tickers: list[str]) -> NewsReport:
    if not items:
        return NewsReport(items=[], news_frame=empty_frame(["ticker", "title"]), theme_summary=pd.DataFrame(), narrative_clusters=pd.DataFrame(), ticker_map={ticker: [] for ticker in tickers})

    deduped: list[NewsItem] = []
    seen_keys: set[str] = set()
    for item in sorted(items, key=lambda value: value.published_at, reverse=True):
        key = _normalize_text(item.title)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        theme = item.theme or infer_theme(f"{item.title}. {item.summary}")
        related = sorted({ticker.upper() for ticker in item.related_tickers if ticker.upper() in tickers} | ({item.ticker.upper()} if item.ticker.upper() in tickers else set()))
        deduped.append(
            NewsItem(
                id=item.id,
                ticker=item.ticker.upper(),
                related_tickers=related or [item.ticker.upper()],
                published_at=item.published_at,
                title=item.title.strip(),
                summary=item.summary.strip(),
                source=item.source.strip(),
                url=item.url.strip(),
                theme=theme,
                sentiment=item.sentiment,
            )
        )

    frame = pd.DataFrame(
        [
            {
                "id": item.id,
                "ticker": item.ticker,
                "related_tickers": item.related_tickers,
                "published_at": pd.to_datetime(item.published_at),
                "title": item.title,
                "summary": item.summary,
                "source": item.source,
                "url": item.url,
                "theme": item.theme,
                "sentiment": item.sentiment,
                "embedding_text": f"{item.title}. {item.summary}",
            }
            for item in deduped
        ]
    ).sort_values("published_at", ascending=False)

    exploded = frame.explode("related_tickers")
    theme_summary = (
        exploded.groupby("theme")
        .agg(
            headline_count=("id", "count"),
            ticker_count=("related_tickers", lambda values: values.nunique()),
            average_sentiment=("sentiment", "mean"),
            latest_time=("published_at", "max"),
            sample_headline=("title", "first"),
        )
        .sort_values(["ticker_count", "headline_count"], ascending=False)
        .reset_index()
    )

    narrative_clusters = (
        exploded.groupby(["theme", "related_tickers"])
        .agg(headline_count=("id", "count"), average_sentiment=("sentiment", "mean"), latest_time=("published_at", "max"))
        .reset_index()
        .sort_values(["headline_count", "latest_time"], ascending=False)
        .rename(columns={"related_tickers": "ticker"})
    )

    ticker_map = {ticker: [] for ticker in tickers}
    for item in deduped:
        for ticker in item.related_tickers:
            if ticker in ticker_map:
                ticker_map[ticker].append(item)

    return NewsReport(
        items=deduped,
        news_frame=frame,
        theme_summary=theme_summary,
        narrative_clusters=narrative_clusters,
        ticker_map=ticker_map,
    )


def build_embedding_report(
    market_report: MarketReport,
    news_report: NewsReport,
    benchmark: str,
    embedding_window: int,
    price_weight: float,
    news_weight: float,
    news_item_vectors: np.ndarray | None = None,
    historical_start: datetime | None = None,
    historical_end: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    price_window_frame, current_price_embeddings = build_price_embeddings(market_report, benchmark, embedding_window)
    news_embedding_frame = build_news_embeddings(news_report, list(market_report.prices.columns), item_vectors=news_item_vectors)
    fused_embedding_frame = fuse_embeddings(current_price_embeddings, news_embedding_frame, price_weight, news_weight)
    if fused_embedding_frame.empty:
        similarity_matrix = pd.DataFrame()
    else:
        similarity_matrix = pd.DataFrame(
            cosine_similarity(fused_embedding_frame.filter(like="dim_")),
            index=fused_embedding_frame["ticker"],
            columns=fused_embedding_frame["ticker"],
        )
    neighbor_frame = build_neighbor_frame(similarity_matrix)
    analogue_frame, basket_embedding_frame, backend = find_historical_analogues(
        price_window_frame,
        market_report.returns,
        benchmark,
        embedding_window,
        historical_start=historical_start,
        historical_end=historical_end,
    )
    outlier_scores = fused_embedding_frame[["ticker", "cluster", "outlier_score"]].sort_values("outlier_score", ascending=False).reset_index(drop=True)
    return (
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
    )


def build_price_embeddings(
    market_report: MarketReport,
    benchmark: str,
    embedding_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = market_report.prices
    returns = prices.pct_change().fillna(0.0)
    if benchmark in prices.columns:
        relative_path = prices.div(prices.iloc[0]).sub(prices[benchmark].div(prices[benchmark].iloc[0]), axis=0)
    else:
        relative_path = prices.div(prices.iloc[0]).sub(1.0)
    volume_change = np.log1p(market_report.volumes).diff().fillna(0.0)
    momentum = prices.pct_change(10).fillna(0.0)
    reversal = -prices.pct_change(5).fillna(0.0)

    records: list[dict[str, object]] = []
    matrix: list[np.ndarray] = []
    for ticker in prices.columns:
        feature_frame = pd.DataFrame(
            {
                "returns": returns[ticker],
                "vol": market_report.realized_vol[ticker].fillna(0.0),
                "volume_change": volume_change[ticker],
                "drawdown": market_report.drawdown[ticker].fillna(0.0),
                "beta": market_report.rolling_beta[ticker].fillna(0.0),
                "corr": market_report.rolling_correlation[ticker].fillna(0.0),
                "rel_strength": market_report.relative_strength[ticker].fillna(0.0),
                "momentum": momentum[ticker],
                "reversal": reversal[ticker],
                "distance_to_benchmark": relative_path[ticker].fillna(0.0),
            }
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        for end_idx in range(embedding_window, len(feature_frame)):
            window = feature_frame.iloc[end_idx - embedding_window : end_idx]
            if len(window) < embedding_window:
                continue
            vector_parts = []
            for column in window.columns:
                values = window[column].to_numpy(dtype=float)
                std = values.std()
                values = (values - values.mean()) / std if std > 1e-9 else values - values.mean()
                vector_parts.append(values)
            summary = np.array(
                [
                    window["returns"].mean(),
                    window["returns"].std(),
                    window["vol"].mean(),
                    window["drawdown"].min(),
                    window["beta"].iloc[-1],
                    window["corr"].iloc[-1],
                    window["rel_strength"].iloc[-1],
                    window["momentum"].iloc[-1],
                    window["reversal"].iloc[-1],
                ],
                dtype=float,
            )
            matrix.append(np.concatenate(vector_parts + [summary]))
            records.append({"ticker": ticker, "end_date": feature_frame.index[end_idx - 1]})

    matrix_np = np.vstack(matrix) if matrix else np.zeros((0, 1), dtype=float)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_np) if len(matrix_np) else matrix_np
    n_components = min(8, matrix_scaled.shape[0], matrix_scaled.shape[1]) if len(matrix_scaled) else 1
    reducer = PCA(n_components=max(1, n_components), random_state=42)
    matrix_embedded = reducer.fit_transform(matrix_scaled) if len(matrix_scaled) else matrix_scaled
    embedded_columns = [f"dim_{index}" for index in range(matrix_embedded.shape[1])]

    price_window_frame = pd.DataFrame(records)
    if not price_window_frame.empty:
        for index, column in enumerate(embedded_columns):
            price_window_frame[column] = matrix_embedded[:, index]
    else:
        current_frame = pd.DataFrame(columns=["ticker", "end_date", "x", "y", "cluster", "outlier_score"])
        for column in embedded_columns:
            current_frame[column] = pd.Series(dtype=float)
        return price_window_frame, current_frame

    current_rows = []
    for ticker in prices.columns:
        ticker_rows = price_window_frame[price_window_frame["ticker"] == ticker]
        if not ticker_rows.empty:
            current_rows.append(ticker_rows.iloc[-1].to_dict())

    current_frame = pd.DataFrame(current_rows)
    embedding_values = current_frame[embedded_columns].to_numpy(dtype=float) if not current_frame.empty else np.zeros((0, 1))
    if len(current_frame) >= 2:
        projection = PCA(n_components=2, random_state=42).fit_transform(embedding_values)
        current_frame["x"] = projection[:, 0]
        current_frame["y"] = projection[:, 1]
        current_frame["cluster"] = _cluster_labels(embedding_values)
        current_frame["outlier_score"] = (1 - cosine_similarity(embedding_values)).mean(axis=1)
    else:
        current_frame["x"] = 0.0
        current_frame["y"] = 0.0
        current_frame["cluster"] = 0
        current_frame["outlier_score"] = 0.0
    return price_window_frame, current_frame.sort_values("ticker").reset_index(drop=True)


def build_news_embeddings(
    news_report: NewsReport,
    tickers: list[str],
    item_vectors: np.ndarray | None = None,
) -> pd.DataFrame:
    dim_columns = [f"dim_{index}" for index in range(8)]
    if news_report.news_frame.empty:
        frame = pd.DataFrame({"ticker": tickers})
        for column in dim_columns:
            frame[column] = 0.0
        frame["news_sentiment"] = 0.0
        frame["headline_count"] = 0
        return frame

    exploded = news_report.news_frame.explode("related_tickers")
    reduced_vectors = None
    if item_vectors is not None and len(item_vectors) == len(news_report.news_frame):
        reduced_vectors = _reduce_news_vectors(item_vectors, dim=len(dim_columns))
        exploded = exploded.merge(
            pd.DataFrame(
                {
                    "id": news_report.news_frame["id"].tolist(),
                    **{f"vec_{index}": reduced_vectors[:, index] for index in range(reduced_vectors.shape[1])},
                }
            ),
            on="id",
            how="left",
        )
    rows = []
    for ticker in tickers:
        ticker_rows = exploded[exploded["related_tickers"] == ticker]
        if reduced_vectors is not None and not ticker_rows.empty:
            vec_columns = [column for column in ticker_rows.columns if column.startswith("vec_")]
            values = normalize(ticker_rows[vec_columns].to_numpy(dtype=float).mean(axis=0).reshape(1, -1))[0]
        else:
            text_blob = " ".join(ticker_rows["embedding_text"].tolist()).strip()
            values = _thematic_vector(text_blob, dim=len(dim_columns))
        row = {"ticker": ticker, "headline_count": int(len(ticker_rows)), "news_sentiment": float(ticker_rows["sentiment"].mean()) if not ticker_rows.empty else 0.0}
        for index, column in enumerate(dim_columns):
            row[column] = values[index]
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def fuse_embeddings(
    price_embedding_frame: pd.DataFrame,
    news_embedding_frame: pd.DataFrame,
    price_weight: float,
    news_weight: float,
) -> pd.DataFrame:
    price_dim_cols = [column for column in price_embedding_frame.columns if column.startswith("dim_")]
    news_dim_cols = [column for column in news_embedding_frame.columns if column.startswith("dim_")]
    merged = price_embedding_frame.merge(news_embedding_frame, on="ticker", how="left", suffixes=("_price", "_news"))

    price_matrix = normalize(merged[[f"{column}_price" for column in price_dim_cols]].to_numpy(dtype=float)) if price_dim_cols else np.zeros((len(merged), 1))
    news_matrix = normalize(merged[[f"{column}_news" for column in news_dim_cols]].to_numpy(dtype=float)) if news_dim_cols else np.zeros((len(merged), 1))
    weight_total = price_weight + news_weight or 1.0
    price_matrix *= price_weight / weight_total
    news_matrix *= news_weight / weight_total
    fused_matrix = normalize(np.concatenate([price_matrix, news_matrix], axis=1))

    fused = merged[["ticker", "x", "y", "cluster", "outlier_score", "headline_count", "news_sentiment"]].copy()
    for index in range(fused_matrix.shape[1]):
        fused[f"dim_{index}"] = fused_matrix[:, index]

    if len(fused) >= 2:
        projection = PCA(n_components=2, random_state=42).fit_transform(fused.filter(like="dim_"))
        fused["x"] = projection[:, 0]
        fused["y"] = projection[:, 1]
        fused["cluster"] = _cluster_labels(fused.filter(like="dim_").to_numpy(dtype=float))
        fused["outlier_score"] = (1 - cosine_similarity(fused.filter(like="dim_"))).mean(axis=1)
    return fused.sort_values("ticker").reset_index(drop=True)


def build_neighbor_frame(similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in similarity_matrix.index:
        peers = similarity_matrix.loc[ticker].drop(index=ticker).sort_values(ascending=False).head(3)
        for peer, similarity in peers.items():
            rows.append({"ticker": ticker, "neighbor": peer, "similarity": float(similarity)})
    return pd.DataFrame(rows)


def find_historical_analogues(
    price_window_frame: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: str,
    embedding_window: int,
    historical_start: datetime | None = None,
    historical_end: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    embed_cols = [column for column in price_window_frame.columns if column.startswith("dim_")]
    tickers = sorted(price_window_frame["ticker"].unique())
    if not embed_cols or not tickers:
        return pd.DataFrame(), pd.DataFrame(), "numpy"

    grouped = price_window_frame.groupby("end_date")
    eligible_dates = grouped["ticker"].nunique()
    common_dates = eligible_dates[eligible_dates == len(tickers)].index
    rows = []
    vectors = []
    for end_date in common_dates:
        if historical_start and end_date < pd.to_datetime(historical_start):
            continue
        if historical_end and end_date > pd.to_datetime(historical_end):
            continue
        subset = grouped.get_group(end_date).sort_values("ticker")
        window_returns = returns.loc[:end_date].tail(embedding_window)
        if len(window_returns) < embedding_window:
            continue
        corr = window_returns.corr().to_numpy()
        upper = corr[np.triu_indices_from(corr, k=1)]
        benchmark_series = window_returns[benchmark] if benchmark in window_returns.columns else window_returns.mean(axis=1)
        basket_vector = np.concatenate(
            [
                subset[embed_cols].to_numpy(dtype=float).ravel(),
                upper,
                np.array(
                    [
                        window_returns.mean().mean(),
                        window_returns.std().mean(),
                        window_returns.std(axis=1).mean(),
                        benchmark_series.mean(),
                    ],
                    dtype=float,
                ),
            ]
        )
        vectors.append(basket_vector)
        rows.append({"end_date": pd.to_datetime(end_date)})

    basket_embedding_frame = pd.DataFrame(rows)
    if basket_embedding_frame.empty:
        return pd.DataFrame(), basket_embedding_frame, "numpy"
    basket_matrix = normalize(np.vstack(vectors))
    for index in range(basket_matrix.shape[1]):
        basket_embedding_frame[f"dim_{index}"] = basket_matrix[:, index]

    latest_vector = basket_matrix[-1:]
    history_vectors = basket_matrix[:-embedding_window] if len(basket_matrix) > embedding_window else basket_matrix[:-1]
    history_meta = basket_embedding_frame.iloc[:-embedding_window] if len(basket_embedding_frame) > embedding_window else basket_embedding_frame.iloc[:-1]
    if len(history_meta) == 0:
        return pd.DataFrame(), basket_embedding_frame, "numpy"

    index = VectorSearchIndex(history_vectors)
    k = min(5, len(history_meta))
    scores, positions = index.query(latest_vector, k=k)
    basket_returns = returns.mean(axis=1)
    benchmark_returns = returns[benchmark] if benchmark in returns.columns else basket_returns
    analogue_rows = []
    for score, pos in zip(scores[0], positions[0], strict=True):
        analogue_date = pd.to_datetime(history_meta.iloc[int(pos)]["end_date"])
        analogue_rows.append(
            {
                "analogue_end": analogue_date,
                "similarity": float(score),
                "forward_5d_basket_return": _forward_return(basket_returns, analogue_date, 5),
                "forward_20d_basket_return": _forward_return(basket_returns, analogue_date, 20),
                "forward_5d_benchmark_return": _forward_return(benchmark_returns, analogue_date, 5),
            }
        )
    analogue_frame = pd.DataFrame(analogue_rows).sort_values("similarity", ascending=False)
    return analogue_frame, basket_embedding_frame, index.backend


def compute_lead_lag(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for leader, follower in combinations(returns.columns, 2):
        best = None
        for lag in range(-3, 4):
            shifted = returns[leader].shift(lag)
            corr = shifted.corr(returns[follower])
            if pd.isna(corr):
                continue
            if best is None or abs(corr) > abs(best["correlation"]):
                best = {"leader": leader if lag >= 0 else follower, "follower": follower if lag >= 0 else leader, "lag_days": abs(lag), "correlation": float(corr)}
        if best:
            rows.append(best)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("correlation", ascending=False).head(10).reset_index(drop=True)


def build_similarity_edges(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, target in combinations(corr_matrix.columns, 2):
        weight = float(corr_matrix.loc[source, target])
        if weight >= 0.35:
            rows.append({"source": source, "target": target, "weight": weight})
    return pd.DataFrame(rows)


def infer_regime(
    returns: pd.DataFrame,
    basket_tickers: list[str],
    benchmark: str,
    basket_dispersion: pd.Series,
) -> dict[str, object]:
    basket_returns = returns[basket_tickers].mean(axis=1)
    recent_return = float((1 + basket_returns.tail(20)).prod() - 1)
    if len(basket_tickers) > 1:
        corr_values = returns[basket_tickers].corr().values
        mean_corr = float(corr_values[np.triu_indices(len(basket_tickers), 1)].mean())
    else:
        mean_corr = 1.0
    recent_dispersion = float(basket_dispersion.tail(20).mean())
    historical_dispersion = float(basket_dispersion.mean())
    benchmark_return = float((1 + returns[benchmark].tail(20)).prod() - 1) if benchmark in returns.columns else recent_return
    if recent_return > 0.08 and mean_corr > 0.55:
        label = "Momentum-led"
    elif recent_return < -0.05 and mean_corr > 0.6:
        label = "Risk-off"
    elif recent_dispersion > historical_dispersion * 1.25:
        label = "Dispersion spike"
    elif abs(recent_return - benchmark_return) < 0.01 and mean_corr > 0.65:
        label = "Benchmark-following"
    else:
        label = "Mixed / ticker-specific"
    return {
        "label": label,
        "basket_return_20d": recent_return,
        "benchmark_return_20d": benchmark_return,
        "mean_pairwise_corr": mean_corr,
        "recent_dispersion": recent_dispersion,
        "historical_dispersion": historical_dispersion,
    }


def infer_theme(text: str) -> str:
    lowered = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "Cross-ticker narrative"


def classify_query(query: str) -> str:
    lowered = query.lower()
    if any(term in lowered for term in ("move together", "moving together", "synchronized", "sync", "peer group")):
        return "synchronized_moves"
    if any(term in lowered for term in ("diverge", "unlike", "outlier", "peer")):
        return "divergence"
    if any(term in lowered for term in ("historical", "past period", "analogue", "similar to the current")):
        return "historical_analogue"
    if any(term in lowered for term in ("shared news", "theme", "narrative", "driving this basket", "headline")):
        return "narrative"
    if any(term in lowered for term in ("regime", "risk-off", "momentum")):
        return "regime"
    return "overview"


def _cluster_labels(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) <= 1:
        return np.zeros(len(matrix), dtype=int)
    clusters = min(3, len(matrix))
    model = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
    return model.fit_predict(matrix)


def _normalize_text(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or char.isspace()).strip()


def _thematic_vector(text: str, dim: int) -> np.ndarray:
    lowered = text.lower()
    values = np.zeros(dim, dtype=float)
    themes = list(THEME_KEYWORDS.items())
    for index, (_, keywords) in enumerate(themes[:dim]):
        values[index] = sum(keyword in lowered for keyword in keywords)
    if values.sum() == 0:
        values[0] = 1.0
    return normalize(values.reshape(1, -1))[0]


def _reduce_news_vectors(vectors: np.ndarray, dim: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] > dim and matrix.shape[0] >= 2:
        reducer = PCA(n_components=min(dim, matrix.shape[0], matrix.shape[1]), random_state=42)
        matrix = reducer.fit_transform(matrix)
    if matrix.shape[1] < dim:
        matrix = np.pad(matrix, ((0, 0), (0, dim - matrix.shape[1])), mode="constant")
    return normalize(matrix[:, :dim])


def _forward_return(series: pd.Series, anchor_date: pd.Timestamp, periods: int) -> float:
    if anchor_date not in series.index:
        return float("nan")
    position = int(series.index.get_loc(anchor_date))
    future = series.iloc[position + 1 : position + 1 + periods]
    if len(future) < periods:
        return float("nan")
    return float((1 + future).prod() - 1)


def _zscore_series(series: pd.Series) -> pd.Series:
    std = series.std()
    if std <= 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
