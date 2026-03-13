# AGENTS.md

## Purpose

This repository is a Streamlit proof of concept for multi-ticker basket analysis. It combines market structure, news themes, embeddings, historical analogue search, and a lightweight multi-agent orchestration layer.

## Stack

- Python
- Streamlit
- pandas / numpy / scikit-learn / plotly
- yfinance with deterministic synthetic fallbacks
- optional OpenAI synthesis and embeddings

## Run

From the repo root, prefer one of these:

```powershell
.\start.ps1
```

or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Windows, avoid depending on `.\.venv\Scripts\streamlit.exe` after moving the repo. The generated launcher can keep the old path.

## Key Files

- `app.py`: Streamlit UI and cached execution path.
- `cross_ticker_lab/config.py`: demo presets, ticker sector map, and environment loading.
- `cross_ticker_lab/agents.py`: orchestration and agent-level reasoning.
- `cross_ticker_lab/analytics.py`: market, news, embedding, and analogue calculations.
- `cross_ticker_lab/providers.py`: Yahoo Finance providers, synthetic fallbacks, and embedding providers.

## Working Rules

- Keep the synthetic fallback path working. If you add tickers to presets, update `TICKER_SECTOR_MAP` in `cross_ticker_lab/config.py`.
- Prefer verifying analysis changes without launching Streamlit repeatedly.
- The ticker input now supports multiline values. `parse_tickers` accepts commas and newlines.
- Environment variables may be loaded from `.env.local` or `.env` in the repo root or its parent directory.

## Verification

There is no formal test suite in the repo right now. For non-UI verification, run the orchestration flow against the synthetic providers:

```powershell
@'
from cross_ticker_lab import AnalysisRequest, OrchestratorAgent, PRESETS, load_config

agent = OrchestratorAgent(load_config())
agent.market_agent.provider.primary = agent.market_agent.provider.fallback
agent.news_agent.provider.primary = agent.news_agent.provider.fallback

for name, preset in PRESETS.items():
    request = AnalysisRequest(
        tickers=preset.tickers,
        benchmark=preset.benchmark,
        lookback_days=120,
        historical_lookback_days=240,
        peer_groups=preset.peer_groups,
        enable_news=True,
    )
    report = agent.run(request)
    print(name, report.market.prices.shape, report.embeddings.fused_embedding_frame.shape)
'@ | .\.venv\Scripts\python.exe -
```
