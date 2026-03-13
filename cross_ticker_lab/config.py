from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency during static inspection
    load_dotenv = None


@dataclass(frozen=True, slots=True)
class DemoPreset:
    name: str
    tickers: list[str]
    benchmark: str
    peer_groups: dict[str, list[str]]
    description: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    openai_api_key: str | None
    openai_embedding_model: str
    openai_response_model: str
    enable_llm_synthesis: bool


TECH_PRESET = DemoPreset(
    name="AI Infrastructure Basket",
    tickers=["NVDA", "AMD", "MSFT", "AAPL", "QQQ", "SMH"],
    benchmark="QQQ",
    peer_groups={
        "AI semis": ["NVDA", "AMD", "SMH"],
        "platform": ["MSFT", "AAPL", "QQQ"],
    },
    description="Semiconductors, platform mega-cap, and the sector ETF in one basket.",
)

ENERGY_PRESET = DemoPreset(
    name="Energy Chain Basket",
    tickers=["XOM", "CVX", "SHEL", "CL=F", "NG=F", "XLE"],
    benchmark="XLE",
    peer_groups={
        "integrated oils": ["XOM", "CVX", "SHEL", "XLE"],
        "commodities": ["CL=F", "NG=F", "XLE"],
    },
    description="Integrated majors, sector ETF, and key commodity contracts.",
)

PRESETS: dict[str, DemoPreset] = {
    TECH_PRESET.name: TECH_PRESET,
    ENERGY_PRESET.name: ENERGY_PRESET,
}

TICKER_SECTOR_MAP: dict[str, str] = {
    "NVDA": "semiconductors",
    "AMD": "semiconductors",
    "SMH": "semiconductors",
    "MSFT": "software-platform",
    "AAPL": "consumer-tech",
    "QQQ": "growth-index",
    "XOM": "integrated-energy",
    "CVX": "integrated-energy",
    "SHEL": "integrated-energy",
    "XLE": "energy-etf",
    "CL=F": "oil",
    "NG=F": "natural-gas",
}

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI infrastructure demand": ("ai", "accelerator", "gpu", "data center", "inference", "h100", "training"),
    "Cloud and platform spend": ("cloud", "capex", "enterprise", "platform", "software", "m365", "iphone"),
    "Semiconductor supply chain": ("semiconductor", "chip", "wafer", "capacity", "export control", "foundry"),
    "Energy supply and OPEC": ("opec", "supply", "barrel", "crude", "inventory", "refinery", "production"),
    "Gas and weather": ("lng", "gas", "weather", "storage", "pipeline", "heating", "natgas"),
    "Rates and macro": ("rates", "treasury", "macro", "inflation", "fed", "yield"),
    "Earnings and guidance": ("earnings", "guidance", "outlook", "revenue", "margin", "bookings"),
    "Regulation and policy": ("antitrust", "regulation", "policy", "tariff", "sanction", "export"),
}


def _load_env_files() -> None:
    if load_dotenv is None:
        return

    cwd = Path.cwd()
    candidates = [
        cwd / ".env.local",
        cwd / ".env",
        cwd.parent / ".env.local",
        cwd.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


def load_config() -> AppConfig:
    _load_env_files()
    api_key = os.getenv("OPENAI_API_KEY") or None
    return AppConfig(
        openai_api_key=api_key,
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_response_model=os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4.1-mini"),
        enable_llm_synthesis=os.getenv("CROSS_TICKER_ENABLE_LLM_SYNTHESIS", "0") == "1",
    )
