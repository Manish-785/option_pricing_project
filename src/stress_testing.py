"""
stress_testing.py
==================
Applies controlled shocks (volatility, interest rate, spot) to a sample of
contracts and reports how each model's valuation responds.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from . import black_scholes as bs
from . import binomial as bn


def stress_test_contract(S, K, T, r, sigma, option_type: str) -> pd.DataFrame:
    """Run vol/rate/spot shocks on a single contract and return a tidy DataFrame."""
    rows = []
    base_bs = bs.bs_price(S, K, T, r, sigma, option_type)
    base_crr = bn.crr_price(S, K, T, r, sigma, N=config.CRR_DEFAULT_STEPS, option_type=option_type)["price"]
    rows.append({"scenario": "base", "shock_type": "none", "shock_value": 0.0,
                 "bs_price": base_bs, "crr_price": base_crr,
                 "bs_pct_change": 0.0, "crr_pct_change": 0.0})

    for shock in config.VOL_SHOCKS:
        sigma_s = max(sigma * (1 + shock), 1e-6)
        p_bs = bs.bs_price(S, K, T, r, sigma_s, option_type)
        p_crr = bn.crr_price(S, K, T, r, sigma_s, N=config.CRR_DEFAULT_STEPS, option_type=option_type)["price"]
        rows.append({"scenario": f"vol {shock:+.0%}", "shock_type": "volatility", "shock_value": shock,
                     "bs_price": p_bs, "crr_price": p_crr,
                     "bs_pct_change": (p_bs / base_bs - 1) * 100 if base_bs else np.nan,
                     "crr_pct_change": (p_crr / base_crr - 1) * 100 if base_crr else np.nan})

    for bps in config.RATE_SHOCKS_BPS:
        r_s = r + bps / 10_000
        p_bs = bs.bs_price(S, K, T, r_s, sigma, option_type)
        p_crr = bn.crr_price(S, K, T, r_s, sigma, N=config.CRR_DEFAULT_STEPS, option_type=option_type)["price"]
        rows.append({"scenario": f"rate {bps:+d}bps", "shock_type": "rate", "shock_value": bps,
                     "bs_price": p_bs, "crr_price": p_crr,
                     "bs_pct_change": (p_bs / base_bs - 1) * 100 if base_bs else np.nan,
                     "crr_pct_change": (p_crr / base_crr - 1) * 100 if base_crr else np.nan})

    for pct in config.SPOT_SHOCKS_PCT:
        S_s = S * (1 + pct)
        p_bs = bs.bs_price(S_s, K, T, r, sigma, option_type)
        p_crr = bn.crr_price(S_s, K, T, r, sigma, N=config.CRR_DEFAULT_STEPS, option_type=option_type)["price"]
        rows.append({"scenario": f"spot {pct:+.0%}", "shock_type": "spot", "shock_value": pct,
                     "bs_price": p_bs, "crr_price": p_crr,
                     "bs_pct_change": (p_bs / base_bs - 1) * 100 if base_bs else np.nan,
                     "crr_pct_change": (p_crr / base_crr - 1) * 100 if base_crr else np.nan})

    return pd.DataFrame(rows)


def stress_test_sample(df: pd.DataFrame, n_sample: int = 20, seed: int = 42) -> pd.DataFrame:
    """Run stress tests across a random sample of contracts from the dataset."""
    if df.empty:
        return pd.DataFrame()
    sample = df.dropna(subset=["spot_price", "strike", "T_years", "risk_free_rate", "realized_vol_60d"])
    sample = sample.sample(n=min(n_sample, len(sample)), random_state=seed)

    frames = []
    for _, row in sample.iterrows():
        sigma = row["realized_vol_60d"] if pd.notna(row["realized_vol_60d"]) and row["realized_vol_60d"] > 0 else 0.25
        result = stress_test_contract(
            row["spot_price"], row["strike"], row["T_years"], row["risk_free_rate"],
            sigma, row["option_type"],
        )
        result["ticker"] = row["ticker"]
        result["contract_symbol"] = row.get("contract_symbol", np.nan)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)
