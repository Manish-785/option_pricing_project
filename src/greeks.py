"""
greeks.py
=========
Vectorised application of the full Black-Scholes Greek suite (Delta,
Gamma, Vega, Theta, Rho) across an options DataFrame, under both the
historical-volatility and market-implied-volatility assumptions.

CRR and Monte Carlo Greeks are limited to Delta (per the project scope --
full higher-order Greeks for those models are considerably more expensive
to compute at scale) and are produced separately in evaluation.py, since
they require row-by-row tree/simulation calls rather than a closed-form
vectorised expression.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import black_scholes as bs


def add_bs_greeks(df: pd.DataFrame, vol_col: str, prefix: str) -> pd.DataFrame:
    """
    Compute the full BS Greek suite for every row using the volatility in
    `vol_col`, and add columns named f"{prefix}_delta", f"{prefix}_gamma", etc.
    """
    out = df.copy()
    S = out["spot_price"].to_numpy()
    K = out["strike"].to_numpy()
    T = out["T_years"].to_numpy()
    r = out["risk_free_rate"].to_numpy()
    sigma = out[vol_col].to_numpy()

    is_call = (out["option_type"] == "call").to_numpy()
    is_put = ~is_call

    delta = np.full(len(out), np.nan)
    gamma = bs.bs_gamma(S, K, T, r, sigma)
    vega = bs.bs_vega(S, K, T, r, sigma)
    theta = np.full(len(out), np.nan)
    rho = np.full(len(out), np.nan)

    if is_call.any():
        delta[is_call] = bs.bs_delta(S[is_call], K[is_call], T[is_call], r[is_call], sigma[is_call], "call")
        theta[is_call] = bs.bs_theta(S[is_call], K[is_call], T[is_call], r[is_call], sigma[is_call], "call")
        rho[is_call] = bs.bs_rho(S[is_call], K[is_call], T[is_call], r[is_call], sigma[is_call], "call")
    if is_put.any():
        delta[is_put] = bs.bs_delta(S[is_put], K[is_put], T[is_put], r[is_put], sigma[is_put], "put")
        theta[is_put] = bs.bs_theta(S[is_put], K[is_put], T[is_put], r[is_put], sigma[is_put], "put")
        rho[is_put] = bs.bs_rho(S[is_put], K[is_put], T[is_put], r[is_put], sigma[is_put], "put")

    out[f"{prefix}_delta"] = delta
    out[f"{prefix}_gamma"] = gamma
    out[f"{prefix}_vega"] = vega
    out[f"{prefix}_theta"] = theta
    out[f"{prefix}_rho"] = rho
    return out
