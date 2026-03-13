# Cross-Ticker Intelligence Lab

Cross-Ticker Intelligence Lab is a Streamlit proof of concept for joint basket analysis. It combines market structure, news narratives, rolling-window embeddings, and a lightweight multi-agent layer to surface insights that do not show up when each ticker is analyzed in isolation.

## What it demonstrates

- synchronized moves across a basket
- diverging names versus usual peer groups
- shared news themes behind coordinated moves
- historical analogue windows across the whole basket
- regime-driven versus ticker-specific behavior
- transparent agent traces instead of a black-box answer

## Stack

- Python 3.11 target
- Streamlit
- pandas / numpy / scikit-learn / plotly
- Yahoo Finance adapters with deterministic synthetic fallbacks
- Optional OpenAI synthesis
- Optional FAISS similarity search if installed

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment file if you want OpenAI-powered synthesis:

```bash
copy .env.example .env.local
```

4. Start the app:

```bash
python -m streamlit run app.py
```

On Windows, if you move the repository after creating `.venv`, the generated `streamlit.exe` launcher can stop working because it keeps the old path. The repo-local launchers avoid that:

```powershell
.\start.ps1
```

or:

```cmd
start.cmd
```

## Default demo baskets

- `NVDA, AMD, MSFT, AAPL, QQQ, SMH`
- `XOM, CVX, SHEL, CL=F, NG=F, XLE`

## Notes

- If live Yahoo Finance calls fail, the app falls back to deterministic synthetic market and news data so the demo still runs.
- If `OPENAI_API_KEY` is missing, the natural-language layer uses deterministic synthesis instead of a hosted model.
