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
    tickers=[
        "NVDA",
        "AMD",
        "AVGO",
        "TSM",
        "ASML",
        "ARM",
        "MU",
        "QCOM",
        "MRVL",
        "AMAT",
        "LRCX",
        "KLAC",
        "INTC",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "ORCL",
        "CRM",
        "QQQ",
        "SMH",
    ],
    benchmark="QQQ",
    peer_groups={
        "AI semis": ["NVDA", "AMD", "AVGO", "ARM", "MRVL", "SMH"],
        "chip stack": ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU", "QCOM", "INTC"],
        "platform": ["MSFT", "GOOGL", "META", "ORCL", "CRM", "AAPL", "QQQ"],
    },
    description="Expanded semiconductor, chip-equipment, platform, and sector ETF basket for broader AI infrastructure analysis.",
)

ENERGY_PRESET = DemoPreset(
    name="Energy Chain Basket",
    tickers=[
        "XOM",
        "CVX",
        "SHEL",
        "BP",
        "TTE",
        "COP",
        "EOG",
        "OXY",
        "SLB",
        "HAL",
        "BKR",
        "MPC",
        "VLO",
        "PSX",
        "KMI",
        "WMB",
        "LNG",
        "EQT",
        "CL=F",
        "NG=F",
        "XLE",
    ],
    benchmark="XLE",
    peer_groups={
        "integrated oils": ["XOM", "CVX", "SHEL", "BP", "TTE", "COP", "XLE"],
        "exploration and gas": ["EOG", "OXY", "LNG", "EQT", "CL=F", "NG=F"],
        "services and downstream": ["SLB", "HAL", "BKR", "MPC", "VLO", "PSX", "KMI", "WMB"],
    },
    description="Expanded energy chain basket spanning majors, upstream, services, midstream, refiners, and benchmark contracts.",
)

IRAN_SUPPLY_SHOCK_PRESET = DemoPreset(
    name="Iran Supply Shock Basket",
    tickers=[
        "XOM",
        "CVX",
        "COP",
        "OXY",
        "EOG",
        "SLB",
        "HAL",
        "BKR",
        "VLO",
        "MPC",
        "LNG",
        "EQT",
        "KMI",
        "WMB",
        "CF",
        "MOS",
        "NTR",
        "ICL",
        "XLE",
        "CL=F",
        "NG=F",
    ],
    benchmark="XLE",
    peer_groups={
        "oil producers": ["XOM", "CVX", "COP", "OXY", "EOG", "XLE"],
        "services and downstream": ["SLB", "HAL", "BKR", "VLO", "MPC"],
        "gas and transport": ["LNG", "EQT", "KMI", "WMB", "NG=F"],
        "fertilizers": ["CF", "MOS", "NTR", "ICL"],
        "commodities": ["CL=F", "NG=F", "XLE"],
    },
    description="Energy, gas, and fertilizer names with sensitivity to Middle East supply disruption, shipping pressure, and higher input costs.",
)

PRESETS: dict[str, DemoPreset] = {
    TECH_PRESET.name: TECH_PRESET,
    ENERGY_PRESET.name: ENERGY_PRESET,
    IRAN_SUPPLY_SHOCK_PRESET.name: IRAN_SUPPLY_SHOCK_PRESET,
}

TICKER_SECTOR_MAP: dict[str, str] = {
    "NVDA": "semiconductors",
    "AMD": "semiconductors",
    "AVGO": "semiconductors",
    "TSM": "semiconductors",
    "ASML": "semiconductors",
    "ARM": "semiconductors",
    "MU": "semiconductors",
    "QCOM": "semiconductors",
    "MRVL": "semiconductors",
    "AMAT": "semiconductors",
    "LRCX": "semiconductors",
    "KLAC": "semiconductors",
    "INTC": "semiconductors",
    "SMH": "semiconductors",
    "MSFT": "software-platform",
    "AAPL": "consumer-tech",
    "GOOGL": "software-platform",
    "META": "software-platform",
    "ORCL": "software-platform",
    "CRM": "software-platform",
    "QQQ": "growth-index",
    "XOM": "integrated-energy",
    "CVX": "integrated-energy",
    "SHEL": "integrated-energy",
    "BP": "integrated-energy",
    "TTE": "integrated-energy",
    "COP": "integrated-energy",
    "EOG": "integrated-energy",
    "OXY": "integrated-energy",
    "SLB": "integrated-energy",
    "HAL": "integrated-energy",
    "BKR": "integrated-energy",
    "MPC": "integrated-energy",
    "VLO": "integrated-energy",
    "PSX": "integrated-energy",
    "KMI": "natural-gas",
    "WMB": "natural-gas",
    "LNG": "natural-gas",
    "EQT": "natural-gas",
    "CF": "fertilizers",
    "MOS": "fertilizers",
    "NTR": "fertilizers",
    "ICL": "fertilizers",
    "XLE": "energy-etf",
    "CL=F": "oil",
    "NG=F": "natural-gas",
}

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI infrastructure demand": ("ai", "accelerator", "gpu", "data center", "inference", "h100", "training"),
    "Cloud and platform spend": ("cloud", "capex", "enterprise", "platform", "software", "m365", "iphone"),
    "Semiconductor supply chain": ("semiconductor", "chip", "wafer", "capacity", "export control", "foundry"),
    "Energy supply and OPEC": ("opec", "supply", "barrel", "crude", "inventory", "refinery", "production", "hormuz", "tanker"),
    "Gas and weather": ("lng", "gas", "weather", "storage", "pipeline", "heating", "natgas"),
    "Fertilizer and crop inputs": ("fertilizer", "urea", "ammonia", "potash", "crop input", "phosphate", "nitrogen"),
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
