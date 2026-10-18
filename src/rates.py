"""
rates.py
========
Risk-free rate retrieval and maturity-based interpolation.

APPROXIMATION NOTE (documented per project requirements):
Yahoo Finance exposes ^IRX (13-week T-bill discount-basis annualized yield,
quoted in percent) and ^TNX (10-year T-note yield, quoted in percent, x10).
We treat ^IRX as a proxy for the short end of the risk-free curve and ^TNX
as a proxy for the long end, and linearly interpolate the continuously
compounded rate for maturities in between. This is a simplification of the
true term structure (which would require a full Treasury curve, e.g. from
FRED) and does not adjust ^IRX's discount-yield quoting convention to a
true bond-equivalent / continuously-compounded basis -- the error this
introduces is on the order of a few basis points and is immaterial next to
typical bid-ask-spread-driven pricing noise, but it is NOT exact.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from . import config


def fetch_treasury_yields(yf_module) -> pd.DataFrame:
    """
    Download recent ^IRX and ^TNX history and return a DataFrame indexed by
    date with columns 'short_rate' and 'long_rate' expressed as decimals
    (e.g. 0.045 for 4.5%).
    """
    irx = yf_module.Ticker(config.RISK_FREE_SHORT_TICKER).history(period="1y")["Close"] / 100.0
    tnx = yf_module.Ticker(config.RISK_FREE_LONG_TICKER).history(period="1y")["Close"] / 100.0
    df = pd.DataFrame({"short_rate": irx, "long_rate": tnx})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.ffill().dropna()
    return df


def risk_free_rate_for_maturity(
    T_years: float,
    short_rate: float,
    long_rate: float,
) -> float:
    """
    Interpolate a continuously-compounded risk-free rate for a given time to
    maturity (in years) between the short-end and long-end proxies.

    T <= RISK_FREE_SHORT_CUTOFF_YEARS -> short_rate
    T >= RISK_FREE_LONG_CUTOFF_YEARS  -> long_rate
    otherwise -> linear interpolation in T
    """
    lo, hi = config.RISK_FREE_SHORT_CUTOFF_YEARS, config.RISK_FREE_LONG_CUTOFF_YEARS
    if T_years <= lo:
        return float(short_rate)
    if T_years >= hi:
        return float(long_rate)
    w = (T_years - lo) / (hi - lo)
    return float(short_rate * (1 - w) + long_rate * w)


def latest_rate_curve(yields_df: pd.DataFrame, as_of: Optional[pd.Timestamp] = None):
    """Return the (short_rate, long_rate) tuple as-of a given date (or latest available)."""
    if yields_df.empty:
        raise ValueError("Treasury yield DataFrame is empty; cannot determine risk-free rate.")
    if as_of is None:
        row = yields_df.iloc[-1]
    else:
        eligible = yields_df[yields_df.index <= as_of]
        row = eligible.iloc[-1] if not eligible.empty else yields_df.iloc[0]
    return float(row["short_rate"]), float(row["long_rate"])
