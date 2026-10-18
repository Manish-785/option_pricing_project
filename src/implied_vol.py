"""
implied_vol.py
===============
Computes market-implied volatility per contract (inverting Black-Scholes
against the mid-price) and builds the implied-volatility surface
(log-moneyness x time-to-maturity x IV) used for smile/skew/term-structure
analysis.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from . import black_scholes as bs


def compute_implied_vols(df: pd.DataFrame, price_col: str = "mid_price") -> pd.DataFrame:
    """
    Add a `market_iv` column computed by inverting Black-Scholes against
    `price_col` for every row. Rows where inversion fails get NaN (this is
    expected for prices outside no-arbitrage bounds or extremely illiquid
    quotes) and are reported, not silently dropped.
    """
    out = df.copy()
    ivs = np.full(len(out), np.nan)

    for i, row in enumerate(out.itertuples(index=False)):
        price = getattr(row, price_col)
        iv = bs.implied_volatility(
            market_price=price,
            S=row.spot_price,
            K=row.strike,
            T=row.T_years,
            r=row.risk_free_rate,
            option_type=row.option_type,
        )
        ivs[i] = iv

    out["market_iv"] = ivs
    n_failed = np.isnan(ivs).sum()
    if n_failed:
        print(f"[implied_vol] IV inversion failed for {n_failed}/{len(out)} "
              f"({n_failed / len(out):.1%}) contracts (likely arbitrage-bound "
              f"violations or extreme illiquidity). These rows are kept with "
              f"market_iv = NaN and excluded from IV-surface / IV-pricing analysis.")
    return out


def build_iv_surface_grid(
    df: pd.DataFrame,
    moneyness_bins: Optional[np.ndarray] = None,
    maturity_bins: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Bin market_iv observations onto a (log-moneyness x maturity) grid and
    return the mean IV per grid cell as a tidy DataFrame (for heatmap /
    3D-surface plotting).
    """
    valid = df.dropna(subset=["market_iv", "log_moneyness", "T_years"])
    valid = valid[(valid["market_iv"] > 0) & (valid["market_iv"] < 3.0)]  # drop garbage inversions
    if valid.empty:
        return pd.DataFrame(columns=["log_moneyness_bin", "T_years_bin", "mean_iv", "n_obs"])

    if moneyness_bins is None:
        moneyness_bins = np.linspace(valid["log_moneyness"].quantile(0.01),
                                      valid["log_moneyness"].quantile(0.99), 15)
    if maturity_bins is None:
        maturity_bins = np.linspace(valid["T_years"].quantile(0.01),
                                     valid["T_years"].quantile(0.99), 10)

    valid = valid.copy()
    valid["log_moneyness_bin"] = pd.cut(valid["log_moneyness"], bins=moneyness_bins)
    valid["T_years_bin"] = pd.cut(valid["T_years"], bins=maturity_bins)

    grid = (
        valid.groupby(["log_moneyness_bin", "T_years_bin"], observed=True)["market_iv"]
        .agg(mean_iv="mean", n_obs="count")
        .reset_index()
    )
    return grid


def smile_by_maturity_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean market_iv by (maturity_bucket, moneyness_bucket) -- a compact
    tabular view of the volatility smile/skew across maturities.
    """
    valid = df.dropna(subset=["market_iv"])
    valid = valid[(valid["market_iv"] > 0) & (valid["market_iv"] < 3.0)]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["maturity_bucket", "moneyness_bucket"], observed=True)["market_iv"]
        .agg(mean_iv="mean", median_iv="median", n_obs="count")
        .reset_index()
    )


def term_structure(df: pd.DataFrame, moneyness_window: float = 0.05) -> pd.DataFrame:
    """
    Approximate ATM term structure: mean market_iv for near-ATM contracts
    (|log_moneyness| < moneyness_window) grouped by maturity bucket.
    """
    valid = df.dropna(subset=["market_iv"])
    valid = valid[
        (valid["market_iv"] > 0) & (valid["market_iv"] < 3.0)
        & (valid["log_moneyness"].abs() < moneyness_window)
    ]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby("maturity_bucket", observed=True)["market_iv"]
        .agg(mean_atm_iv="mean", n_obs="count")
        .reset_index()
    )
