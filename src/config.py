"""
config.py
=========
Central configuration for the option-pricing research project.

Everything a user is likely to want to tweak (ticker universe, bucket
definitions, liquidity filters, model settings) lives here so the rest of
the codebase never hard-codes these choices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Universe of underlyings (override with --tickers on the CLI)
# --------------------------------------------------------------------------
DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "SPY", "QQQ",
]

# --------------------------------------------------------------------------
# Underlying price-history settings
# --------------------------------------------------------------------------
PRICE_HISTORY_PERIOD = "2y"          # yfinance history window for realized vol
VOL_WINDOWS = (20, 60, 252)          # rolling realized-vol windows (trading days)
TRADING_DAYS_PER_YEAR = 252

# --------------------------------------------------------------------------
# Volatility regimes (percentile cut points on rolling 60D realized vol)
# --------------------------------------------------------------------------
VOL_REGIME_LOW_PCTL = 33
VOL_REGIME_HIGH_PCTL = 67
VOL_REGIME_LABELS = ("Low", "Medium", "High")

# --------------------------------------------------------------------------
# Liquidity / contract filters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LiquidityFilter:
    min_bid: float = 0.01
    min_ask: float = 0.01
    require_ask_gte_bid: bool = True
    min_open_interest: int = 10
    min_volume: int = 0          # OI is usually a better liquidity signal than daily volume
    max_relative_spread: float = 1.50  # (ask-bid)/mid must be below this (150%) to keep
    min_days_to_expiry: int = 1
    max_days_to_expiry: int = 730


DEFAULT_LIQUIDITY_FILTER = LiquidityFilter()

# --------------------------------------------------------------------------
# Moneyness buckets (based on S/K)
# --------------------------------------------------------------------------
# Defined on "moneyness ratio" m = S/K for calls in the usual sense; we bucket
# using the same ratio for puts too, but label meaning is model-based (see
# preprocessing.py for the exact call/put-aware logic).
MONEYNESS_BUCKETS: List[Tuple[str, float, float]] = [
    ("Deep ITM", 1.10, float("inf")),
    ("ITM", 1.02, 1.10),
    ("ATM", 0.98, 1.02),
    ("OTM", 0.90, 0.98),
    ("Deep OTM", 0.0, 0.90),
]

# --------------------------------------------------------------------------
# Maturity buckets (in days)
# --------------------------------------------------------------------------
MATURITY_BUCKETS: List[Tuple[str, int, int]] = [
    ("0-30D", 0, 30),
    ("30-90D", 30, 90),
    ("90-180D", 90, 180),
    ("180-365D", 180, 365),
    (">365D", 365, 10_000),
]

# --------------------------------------------------------------------------
# Risk-free rate proxies
# --------------------------------------------------------------------------
RISK_FREE_SHORT_TICKER = "^IRX"   # 13-week T-bill discount rate (annualized, %)
RISK_FREE_LONG_TICKER = "^TNX"    # 10-year T-note yield (annualized, %)
# Maturity (in years) below which we use the short-end proxy exclusively,
# above which we use the long-end proxy exclusively; linear interpolation
# is used in between. This is a simplification -- see README for caveats.
RISK_FREE_SHORT_CUTOFF_YEARS = 0.25
RISK_FREE_LONG_CUTOFF_YEARS = 2.0

# --------------------------------------------------------------------------
# CRR binomial tree settings
# --------------------------------------------------------------------------
CRR_CONVERGENCE_STEPS: List[int] = [25, 50, 100, 200, 500]
CRR_DEFAULT_STEPS = 200

# --------------------------------------------------------------------------
# Monte Carlo settings
# --------------------------------------------------------------------------
MC_CONVERGENCE_PATHS: List[int] = [10_000, 50_000, 100_000, 500_000]
MC_DEFAULT_PATHS = 100_000
MC_RANDOM_SEED = 42
MC_DELTA_BUMP = 0.01   # relative bump for finite-difference delta (1%)

# --------------------------------------------------------------------------
# Implied volatility solver
# --------------------------------------------------------------------------
IV_SOLVER_LOW = 1e-4
IV_SOLVER_HIGH = 5.0
IV_SOLVER_TOL = 1e-6
IV_SOLVER_MAXITER = 100

# --------------------------------------------------------------------------
# Delta hedging backtest
# --------------------------------------------------------------------------
REBALANCE_FREQUENCIES = ("daily", "weekly")

@dataclass(frozen=True)
class TransactionCostModel:
    per_share_cost: float = 0.005      # $ per share, e.g. commission + slippage
    spread_cost_fraction: float = 0.5  # fraction of quoted spread paid on each trade


DEFAULT_TXN_COST = TransactionCostModel()

# --------------------------------------------------------------------------
# Stress testing scenarios
# --------------------------------------------------------------------------
VOL_SHOCKS = (-0.40, -0.20, -0.10, 0.10, 0.20, 0.40)
RATE_SHOCKS_BPS = (-100, -50, 50, 100)
SPOT_SHOCKS_PCT = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)

# --------------------------------------------------------------------------
# Evaluation thresholds
# --------------------------------------------------------------------------
ACCURACY_THRESHOLDS = (0.05, 0.10, 0.20)  # 5%, 10%, 20% "within" bands
MIN_MARKET_PRICE_FOR_MAPE = 0.05           # avoid dividing by near-zero prices
