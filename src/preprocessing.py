"""
preprocessing.py
=================
Cleans raw option-chain data, applies liquidity filters, classifies
contracts by moneyness / maturity / volatility regime, computes the
market mid-price and bid-ask-spread-as-%-of-mid, and merges in the
appropriate risk-free rate and realized volatility for each observation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

from . import config
from . import rates


# --------------------------------------------------------------------------
# Liquidity filtering
# --------------------------------------------------------------------------
def clean_option_data(
    df: pd.DataFrame,
    filt: config.LiquidityFilter = None,
) -> pd.DataFrame:
    """Apply liquidity / sanity filters to raw option-chain data."""
    filt = config.DEFAULT_LIQUIDITY_FILTER if filt is None else filt
    if df.empty:
        return df

    out = df.copy()
    n_start = len(out)

    out["bid"] = pd.to_numeric(out.get("bid"), errors="coerce")
    out["ask"] = pd.to_numeric(out.get("ask"), errors="coerce")
    out["last_price"] = pd.to_numeric(out.get("last_price"), errors="coerce")
    out["open_interest"] = pd.to_numeric(out.get("open_interest"), errors="coerce").fillna(0)
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0)
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    out["spot_price"] = pd.to_numeric(out.get("spot_price"), errors="coerce")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.tz_localize(None)
    out["days_to_expiry"] = (out["expiration"] - now).dt.days
    out["T_years"] = out["days_to_expiry"].clip(lower=0) / 365.0

    mask = (
        out["bid"].notna() & out["ask"].notna() & out["strike"].notna() & out["spot_price"].notna()
        & (out["bid"] >= filt.min_bid)
        & (out["ask"] >= filt.min_ask)
        & (out["strike"] > 0)
        & (out["spot_price"] > 0)
        & (out["days_to_expiry"] >= filt.min_days_to_expiry)
        & (out["days_to_expiry"] <= filt.max_days_to_expiry)
        & ((out["open_interest"] >= filt.min_open_interest) | (out["volume"] >= filt.min_volume))
    )
    if filt.require_ask_gte_bid:
        mask &= out["ask"] >= out["bid"]

    out = out[mask].copy()

    # Mid price and relative spread
    out["mid_price"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    out["relative_spread"] = out["spread"] / out["mid_price"].replace(0, np.nan)

    out = out[out["relative_spread"] <= filt.max_relative_spread]
    out = out[out["mid_price"] > 0]

    n_end = len(out)
    print(f"[preprocessing] Liquidity filter kept {n_end}/{n_start} "
          f"({n_end / max(n_start, 1):.1%}) contracts.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Moneyness classification
# --------------------------------------------------------------------------
def classify_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `moneyness_ratio` (S/K), `log_moneyness` (log(S/K)), and a
    call/put-aware `moneyness_bucket` label.

    For calls: higher S/K => more ITM.
    For puts:  higher S/K => more OTM (we invert the ratio for bucketing
               so that "ITM"/"OTM" labels remain economically meaningful
               for both option types).
    """
    out = df.copy()
    out["moneyness_ratio"] = out["spot_price"] / out["strike"]
    out["log_moneyness"] = np.log(out["moneyness_ratio"])

    def bucket_for_row(row) -> str:
        m = row["moneyness_ratio"]
        # For puts, ITM means K > S, i.e. moneyness_ratio < 1; invert so that
        # the same bucket table (defined on "in-the-moneyness ratio") applies.
        effective = m if row["option_type"] == "call" else 1.0 / m
        for label, low, high in config.MONEYNESS_BUCKETS:
            if low <= effective < high:
                return label
        return "Unclassified"

    out["moneyness_bucket"] = out.apply(bucket_for_row, axis=1)
    return out


def classify_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `maturity_bucket` label based on days_to_expiry."""
    out = df.copy()

    def bucket_for_days(d) -> str:
        for label, low, high in config.MATURITY_BUCKETS:
            if low <= d < high:
                return label
        return "Unclassified"

    out["maturity_bucket"] = out["days_to_expiry"].apply(bucket_for_days)
    return out


# --------------------------------------------------------------------------
# Volatility regime classification (based on realized vol at snapshot time)
# --------------------------------------------------------------------------
def classify_volatility_regime(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    vol_col: str = "realized_vol_60d",
) -> pd.DataFrame:
    """
    Tag each option observation with a volatility regime (Low/Medium/High)
    based on the ticker's rolling realized volatility on the most recent
    available underlying trading date at or before the option snapshot.
    """
    out = options_df.copy()
    if underlying_df.empty or vol_col not in underlying_df.columns:
        out["realized_vol"] = np.nan
        out["vol_regime"] = "Unknown"
        return out

    out["realized_vol"] = np.nan
    for ticker, sub in out.groupby("ticker"):
        u = underlying_df[underlying_df["ticker"] == ticker].sort_index()
        if u.empty:
            continue
        latest_vol = u[vol_col].dropna()
        if latest_vol.empty:
            continue
        out.loc[sub.index, "realized_vol"] = latest_vol.iloc[-1]

    valid = out["realized_vol"].dropna()
    if valid.empty:
        out["vol_regime"] = "Unknown"
        return out

    low_cut = np.percentile(valid, config.VOL_REGIME_LOW_PCTL)
    high_cut = np.percentile(valid, config.VOL_REGIME_HIGH_PCTL)

    def regime(v):
        if pd.isna(v):
            return "Unknown"
        if v <= low_cut:
            return "Low"
        elif v >= high_cut:
            return "High"
        return "Medium"

    out["vol_regime"] = out["realized_vol"].apply(regime)
    return out


# --------------------------------------------------------------------------
# Historical volatility merge (attaches the actual rolling-vol columns, not
# just the regime label, so downstream pricing can use e.g. realized_vol_60d)
# --------------------------------------------------------------------------
def attach_historical_vols(df: pd.DataFrame, underlying_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest available rolling realized-vol columns (20D/60D/252D) per ticker."""
    out = df.copy()
    vol_cols = [f"realized_vol_{w}d" for w in config.VOL_WINDOWS]
    for col in vol_cols:
        out[col] = np.nan

    if underlying_df.empty:
        return out

    for ticker, sub in out.groupby("ticker"):
        u = underlying_df[underlying_df["ticker"] == ticker].sort_index()
        if u.empty:
            continue
        for col in vol_cols:
            if col not in u.columns:
                continue
            latest = u[col].dropna()
            if not latest.empty:
                out.loc[sub.index, col] = latest.iloc[-1]
    return out


# --------------------------------------------------------------------------
# Risk-free rate merge
# --------------------------------------------------------------------------
def attach_risk_free_rate(df: pd.DataFrame, yields_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a maturity-appropriate risk-free rate to every option row."""
    out = df.copy()
    if yields_df.empty:
        out["risk_free_rate"] = 0.04  # documented fallback assumption
        print("[preprocessing] WARNING: no Treasury yield data available; "
              "falling back to a flat 4% risk-free rate assumption.")
        return out

    short_rate, long_rate = rates.latest_rate_curve(yields_df)
    out["risk_free_rate"] = out["T_years"].apply(
        lambda t: rates.risk_free_rate_for_maturity(t, short_rate, long_rate)
    )
    return out


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def build_analysis_dataset(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    yields_df: pd.DataFrame,
    filt: config.LiquidityFilter = None,
) -> pd.DataFrame:
    """Run the full cleaning + classification pipeline end to end."""
    df = clean_option_data(options_df, filt)
    if df.empty:
        return df
    df = classify_moneyness(df)
    df = classify_maturity(df)
    df = classify_volatility_regime(df, underlying_df)
    df = attach_historical_vols(df, underlying_df)
    df = attach_risk_free_rate(df, yields_df)

    n_nan_before = len(df)
    critical_cols = ["spot_price", "strike", "T_years", "risk_free_rate", "mid_price"]
    df = df.dropna(subset=critical_cols)
    if len(df) < n_nan_before:
        print(f"[preprocessing] Dropped {n_nan_before - len(df)} rows with missing "
              f"critical fields ({critical_cols}).")

    return df.reset_index(drop=True)
