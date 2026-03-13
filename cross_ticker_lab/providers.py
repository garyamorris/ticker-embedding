from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import THEME_KEYWORDS, TICKER_SECTOR_MAP
from .models import NewsItem

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - faiss is optional
    faiss = None

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - provider gracefully falls back
    yf = None

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - provider gracefully falls back
    OpenAI = None


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _non_empty_texts(texts: list[str]) -> list[str]:
    return [text if text.strip() else "no_content" for text in texts]


def _lexical_sentiment(text: str) -> float:
    positive_words = {
        "beat",
        "strong",
        "accelerate",
        "improve",
        "resilient",
        "upgrade",
        "momentum",
        "demand",
        "tight",
        "leadership",
    }
    negative_words = {
        "miss",
        "weak",
        "slow",
        "downside",
        "cut",
        "pressure",
        "risk",
        "slip",
        "downgrade",
        "uncertain",
    }
    lowered = text.lower()
    pos = sum(word in lowered for word in positive_words)
    neg = sum(word in lowered for word in negative_words)
    if pos == 0 and neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 3)


def infer_sector(ticker: str) -> str:
    return TICKER_SECTOR_MAP.get(ticker.upper(), "general-equity")


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, tickers: list[str], start: date, end: date, interval: str) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


class NewsProvider(ABC):
    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
        limit_per_ticker: int,
        market_snapshot: pd.DataFrame | None = None,
    ) -> list[NewsItem]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class YahooFinanceMarketDataProvider(MarketDataProvider):
    def fetch(self, tickers: list[str], start: date, end: date, interval: str) -> dict[str, pd.DataFrame]:
        if yf is None:
            raise RuntimeError("yfinance is not installed")

        data = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data.empty:
            raise RuntimeError("Yahoo Finance returned no data")

        result: dict[str, pd.DataFrame] = {}
        multi_columns = isinstance(data.columns, pd.MultiIndex)
        for ticker in tickers:
            frame = data[ticker].copy() if multi_columns else data.copy()
            if frame.empty:
                continue

            frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
            if "adj_close" in frame.columns and "close" not in frame.columns:
                frame["close"] = frame["adj_close"]
            keep_columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]
            frame = frame[keep_columns].dropna(subset=["close"])
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            if not frame.empty:
                result[ticker] = frame

        if len(result) < max(2, math.ceil(len(tickers) * 0.6)):
            raise RuntimeError("Yahoo Finance returned too few populated tickers")
        return result


class SyntheticMarketDataProvider(MarketDataProvider):
    def fetch(self, tickers: list[str], start: date, end: date, interval: str) -> dict[str, pd.DataFrame]:
        if interval != "1d":
            raise RuntimeError("Synthetic provider currently supports 1d interval only")

        index = pd.bdate_range(start=start, end=end)
        if len(index) < 80:
            index = pd.bdate_range(end=date.today(), periods=120)

        rng = np.random.default_rng(_stable_seed(*tickers, start.isoformat(), end.isoformat()))
        market_factor = rng.normal(0.0005, 0.012, size=len(index))
        tech_factor = rng.normal(0.0008, 0.016, size=len(index))
        energy_factor = rng.normal(0.0002, 0.018, size=len(index))
        macro_shock = np.zeros(len(index))
        event_positions = np.linspace(12, len(index) - 12, 4, dtype=int)
        macro_shock[event_positions] = rng.normal(0.0, 0.035, size=len(event_positions))

        frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            ticker_rng = np.random.default_rng(_stable_seed(ticker, start.isoformat()))
            sector = infer_sector(ticker)
            tech_beta = 0.85 if sector in {"semiconductors", "software-platform", "consumer-tech", "growth-index"} else 0.15
            energy_beta = 0.9 if sector in {"integrated-energy", "energy-etf", "oil", "natural-gas"} else 0.1
            market_beta = 0.95 if ticker not in {"CL=F", "NG=F"} else 0.45
            idiosyncratic = ticker_rng.normal(0.0, 0.014, size=len(index))
            drift = 0.0009 if sector in {"semiconductors", "software-platform"} else 0.0003
            if ticker == "QQQ":
                drift = 0.0006
                tech_beta = 0.6
            if ticker == "SMH":
                tech_beta = 1.05
            if ticker == "XLE":
                energy_beta = 0.8
            if ticker == "CL=F":
                energy_beta = 1.2
                market_beta = 0.2
            if ticker == "NG=F":
                energy_beta = 0.6
                market_beta = 0.1

            returns = (
                drift
                + market_beta * market_factor
                + tech_beta * tech_factor
                + energy_beta * energy_factor
                + macro_shock
                + idiosyncratic
            )
            if ticker in {"NVDA", "AMD", "SMH"}:
                returns[event_positions[1]] += 0.05
            if ticker in {"XOM", "CVX", "SHEL", "XLE"}:
                returns[event_positions[2]] += 0.04
            if ticker in {"AAPL", "MSFT"}:
                returns[event_positions[3]] -= 0.03

            price = 80 + (_stable_seed(ticker) % 120)
            close = price * np.cumprod(1 + returns)
            open_ = close * (1 + ticker_rng.normal(0.0, 0.004, size=len(index)))
            high = np.maximum(open_, close) * (1 + np.abs(ticker_rng.normal(0.006, 0.003, size=len(index))))
            low = np.minimum(open_, close) * (1 - np.abs(ticker_rng.normal(0.006, 0.003, size=len(index))))
            baseline_volume = 1.2e7 if ticker not in {"CL=F", "NG=F"} else 3.5e5
            volume = baseline_volume * np.exp(ticker_rng.normal(0.0, 0.35, size=len(index)))
            volume[event_positions] *= ticker_rng.uniform(1.8, 3.1, size=len(event_positions))

            frames[ticker] = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume.astype(float),
                },
                index=index,
            )
        return frames


class ResilientMarketDataProvider(MarketDataProvider):
    def __init__(self) -> None:
        self.primary = YahooFinanceMarketDataProvider()
        self.fallback = SyntheticMarketDataProvider()
        self.last_provider_name = "synthetic"

    def fetch(self, tickers: list[str], start: date, end: date, interval: str) -> dict[str, pd.DataFrame]:
        try:
            result = self.primary.fetch(tickers, start, end, interval)
            self.last_provider_name = "yfinance"
            return result
        except Exception:
            self.last_provider_name = "synthetic"
            return self.fallback.fetch(tickers, start, end, interval)


class YahooFinanceNewsProvider(NewsProvider):
    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
        limit_per_ticker: int,
        market_snapshot: pd.DataFrame | None = None,
    ) -> list[NewsItem]:
        if yf is None:
            raise RuntimeError("yfinance is not installed")

        items: list[NewsItem] = []
        start_dt = datetime.combine(start, datetime.min.time())
        for ticker in tickers:
            ticker_obj = yf.Ticker(ticker)
            raw_news = []
            if hasattr(ticker_obj, "get_news"):
                try:
                    raw_news = ticker_obj.get_news(count=limit_per_ticker) or []
                except TypeError:
                    raw_news = ticker_obj.get_news() or []
            if not raw_news:
                raw_news = getattr(ticker_obj, "news", []) or []

            for raw in raw_news[:limit_per_ticker]:
                title = raw.get("title") or raw.get("content", {}).get("title") or raw.get("headline") or ""
                summary = raw.get("summary") or raw.get("content", {}).get("summary") or raw.get("description") or ""
                publisher = (
                    raw.get("publisher")
                    or raw.get("provider")
                    or raw.get("content", {}).get("provider", {}).get("displayName")
                    or "Yahoo Finance"
                )
                timestamp = raw.get("providerPublishTime") or raw.get("published_at") or raw.get("content", {}).get("pubDate")
                if isinstance(timestamp, str):
                    published_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
                elif timestamp:
                    published_at = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).replace(tzinfo=None)
                else:
                    published_at = datetime.utcnow()
                if published_at < start_dt:
                    continue

                url = (
                    raw.get("link")
                    or raw.get("canonicalUrl", {}).get("url")
                    or raw.get("content", {}).get("clickThroughUrl", {}).get("url")
                    or f"https://finance.yahoo.com/quote/{ticker}"
                )
                text = f"{title}. {summary}".lower()
                theme = next(
                    (
                        theme_name
                        for theme_name, keywords in THEME_KEYWORDS.items()
                        if any(keyword in text for keyword in keywords)
                    ),
                    "Cross-ticker narrative",
                )
                items.append(
                    NewsItem(
                        id=f"{ticker}-{abs(hash((ticker, title, published_at.isoformat())))}",
                        ticker=ticker,
                        related_tickers=[ticker],
                        published_at=published_at,
                        title=title,
                        summary=summary,
                        source=publisher,
                        url=url,
                        theme=theme,
                        sentiment=_lexical_sentiment(text),
                    )
                )

        if not items:
            raise RuntimeError("Yahoo Finance returned no news")
        return items


class SyntheticNewsProvider(NewsProvider):
    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
        limit_per_ticker: int,
        market_snapshot: pd.DataFrame | None = None,
    ) -> list[NewsItem]:
        reference = datetime.combine(end, datetime.min.time())
        items: list[NewsItem] = []
        tickers = [ticker.upper() for ticker in tickers]
        sector_groups: dict[str, list[str]] = {}
        for ticker in tickers:
            sector_groups.setdefault(infer_sector(ticker), []).append(ticker)

        shared_templates = {
            "semiconductors": [
                ("AI infrastructure demand", "Cloud buyers lean into accelerator orders as data-center build-outs stay aggressive."),
                ("Semiconductor supply chain", "Chip supply conversations shift from shortage to mix, packaging, and export-screening risk."),
            ],
            "software-platform": [
                ("Cloud and platform spend", "Enterprise platform budgets look steadier than expected even as CIO scrutiny remains high."),
            ],
            "consumer-tech": [
                ("Earnings and guidance", "Device and services positioning stays in focus heading into the next product cycle."),
            ],
            "integrated-energy": [
                ("Energy supply and OPEC", "Crude-linked equities respond to tighter supply rhetoric and refining margin chatter."),
                ("Rates and macro", "Energy leadership broadens as macro desks debate growth resilience against inventories."),
            ],
            "oil": [
                ("Energy supply and OPEC", "Oil futures react to inventory prints and producer discipline headlines."),
            ],
            "natural-gas": [
                ("Gas and weather", "Natural-gas traders reset positioning around weather, storage, and LNG export utilization."),
            ],
        }

        for sector, sector_tickers in sector_groups.items():
            templates = shared_templates.get(sector, [("Rates and macro", "Macro cross-currents keep investors focused on relative strength and flows.")])
            for offset, (theme, summary) in enumerate(templates):
                title = self._make_title(theme, sector_tickers)
                published_at = reference - timedelta(hours=10 + offset * 7)
                for ticker in sector_tickers[:limit_per_ticker]:
                    items.append(
                        NewsItem(
                            id=f"{ticker}-{abs(hash((title, published_at.isoformat())))}",
                            ticker=ticker,
                            related_tickers=sector_tickers,
                            published_at=published_at,
                            title=title,
                            summary=summary,
                            source="Demo Wire",
                            url=f"https://example.com/demo-news/{ticker.lower()}-{offset}",
                            theme=theme,
                            sentiment=_lexical_sentiment(f"{title}. {summary}"),
                        )
                    )

        if market_snapshot is not None and not market_snapshot.empty:
            market_snapshot = market_snapshot.copy()
            for ticker, row in market_snapshot.iterrows():
                ret_5d = float(row.get("return_5d", 0.0))
                volume_shock = float(row.get("volume_shock", 0.0))
                if ret_5d > 0.06:
                    title = f"{ticker} extends momentum as desks chase relative strength"
                    summary = "Short-term performance is pulling peer comparisons tighter as flows concentrate in leaders."
                    theme = "Earnings and guidance"
                elif ret_5d < -0.05:
                    title = f"{ticker} slips versus peers as positioning resets"
                    summary = "The move stands out because peer performance is firmer, raising the odds of ticker-specific follow-through."
                    theme = "Cross-ticker narrative"
                elif volume_shock > 1.2:
                    title = f"{ticker} sees unusual volume as market searches for confirmation"
                    summary = "Participation jumped faster than price, often a sign that narrative adoption is still catching up."
                    theme = "Rates and macro"
                else:
                    continue

                published_at = reference - timedelta(hours=3 + abs(hash(ticker)) % 6)
                items.append(
                    NewsItem(
                        id=f"{ticker}-{abs(hash((title, published_at.isoformat())))}",
                        ticker=ticker,
                        related_tickers=[ticker],
                        published_at=published_at,
                        title=title,
                        summary=summary,
                        source="Synthetic Desk Feed",
                        url=f"https://example.com/desk-feed/{ticker.lower()}",
                        theme=theme,
                        sentiment=_lexical_sentiment(f"{title}. {summary}"),
                    )
                )
        return items

    @staticmethod
    def _make_title(theme: str, sector_tickers: list[str]) -> str:
        joined = ", ".join(sector_tickers[:3])
        if len(sector_tickers) > 3:
            joined = f"{joined}, and peers"
        return f"{joined} respond to {theme.lower()}"


class ResilientNewsProvider(NewsProvider):
    def __init__(self) -> None:
        self.primary = YahooFinanceNewsProvider()
        self.fallback = SyntheticNewsProvider()
        self.last_provider_name = "synthetic"

    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
        limit_per_ticker: int,
        market_snapshot: pd.DataFrame | None = None,
    ) -> list[NewsItem]:
        try:
            result = self.primary.fetch(tickers, start, end, limit_per_ticker, market_snapshot)
            self.last_provider_name = "yfinance"
            return result
        except Exception:
            self.last_provider_name = "synthetic"
            return self.fallback.fetch(tickers, start, end, limit_per_ticker, market_snapshot)


class LocalTfidfEmbeddingProvider(EmbeddingProvider):
    name = "local-tfidf"

    def __init__(self, output_dim: int = 32) -> None:
        self.output_dim = output_dim
        self.vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2), stop_words="english")
        self.reducer: TruncatedSVD | None = None
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(_non_empty_texts(texts))
        self._fitted = True
        if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
            dim = min(self.output_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if dim >= 2:
                self.reducer = TruncatedSVD(n_components=dim, random_state=42)
                dense = self.reducer.fit_transform(matrix)
                return normalize(dense)
        self.reducer = None
        return normalize(matrix.toarray())

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            return self.fit_transform(texts)
        matrix = self.vectorizer.transform(_non_empty_texts(texts))
        dense = self.reducer.transform(matrix) if self.reducer is not None else matrix.toarray()
        return normalize(dense)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=_non_empty_texts(texts))
        vectors = np.array([item.embedding for item in response.data], dtype=float)
        return normalize(vectors)


class VectorSearchIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self.backend = "numpy"
        self._index = None
        if faiss is not None and len(self.vectors) > 0:
            self.backend = "faiss"
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    def query(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = normalize(np.asarray(queries, dtype=np.float32))
        if self._index is not None:
            scores, indices = self._index.search(queries, k)
            return scores, indices
        scores = queries @ self.vectors.T
        top_indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores, top_indices


def build_embedding_provider(api_key: str | None, model: str) -> EmbeddingProvider:
    if api_key:
        try:
            return OpenAIEmbeddingProvider(api_key=api_key, model=model)
        except Exception:
            return LocalTfidfEmbeddingProvider()
    return LocalTfidfEmbeddingProvider()
