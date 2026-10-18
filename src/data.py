"""
data.py
=======
Reusable data pipeline built on `yfinance`.

Responsibilities:
    * Download underlying price history and compute realized volatility.
    * Download the full option chain (all expirations) for a ticker.
    * Cache everything to disk (Parquet, falling back to CSV) so repeated
      runs of the analysis do not repeatedly hit the Yahoo Finance API.

NOTE: This module requires outbound network access to Yahoo Finance
(query1/query2.finance.yahoo.com). If your environment blocks this, run
the pipeline somewhere with internet access and point downstream analysis
at the resulting `data/cache` directory.
"""

from __future__ import annotations
import os
import time
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from e


# --------------------------------------------------------------------------
# Caching helpers
# --------------------------------------------------------------------------
def _cache_path(name: str) -> str:
    return os.path.join(config.CACHE_DIR, name)


def _save(df: pd.DataFrame, name: str) -> None:
    path = _cache_path(name)
    try:
        df.to_parquet(path.replace(".csv", ".parquet"))
    except Exception:
        df.to_csv(path if path.endswith(".csv") else path.replace(".parquet", ".csv"))


def _load(name: str) -> Optional[pd.DataFrame]:
    parquet_path = _cache_path(name.replace(".csv", ".parquet"))
    csv_path = _cache_path(name.replace(".parquet", ".csv"))
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return None


def _is_fresh(name: str, max_age_hours: float = 6.0) -> bool:
    for ext in (".parquet", ".csv"):
        path = _cache_path(name.replace(".parquet", ext).replace(".csv", ext))
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
            return age_hours < max_age_hours
    return False


# --------------------------------------------------------------------------
# Underlying data
# --------------------------------------------------------------------------
def get_underlying_history(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download adjusted close, volume, daily returns, and rolling realized
    volatility for a single ticker.
    """
    cache_name = f"underlying_{ticker}.parquet"
    if use_cache and _is_fresh(cache_name, max_age_hours=6.0):
        cached = _load(cache_name)
        if cached is not None:
            return cached

    tk = yf.Ticker(ticker)
    hist = tk.history(period=config.PRICE_HISTORY_PERIOD, auto_adjust=True)
    if hist.empty:
        warnings.warn(f"No price history returned for {ticker}.")
        return pd.DataFrame()

    df = pd.DataFrame(index=hist.index)
    df["adj_close"] = hist["Close"]
    df["volume"] = hist["Volume"]
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

    for w in config.VOL_WINDOWS:
        df[f"realized_vol_{w}d"] = (
            df["log_return"].rolling(w).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        )

    df["ticker"] = ticker
    df.index.name = "date"

    _save(df, cache_name)
    return df


def get_underlying_universe(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """Download underlying history for a list of tickers and concatenate."""
    frames = []
    for t in tickers:
        try:
            df = get_underlying_history(t, use_cache=use_cache)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            warnings.warn(f"Failed to download underlying data for {t}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


# --------------------------------------------------------------------------
# Option chain data
# --------------------------------------------------------------------------
def get_option_chain(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download the full option chain (all available expirations, calls + puts)
    for a single ticker, tagged with the current spot price and download
    timestamp.
    """
    cache_name = f"options_{ticker}.parquet"
    if use_cache and _is_fresh(cache_name, max_age_hours=6.0):
        cached = _load(cache_name)
        if cached is not None:
            return cached

    tk = yf.Ticker(ticker)
    try:
        expirations = tk.options
    except Exception as e:
        warnings.warn(f"Could not retrieve expirations for {ticker}: {e}")
        return pd.DataFrame()

    if not expirations:
        warnings.warn(f"No option expirations available for {ticker}.")
        return pd.DataFrame()

    # Spot price snapshot (fast_info is cheaper than full .info)
    try:
        spot = tk.fast_info["last_price"]
    except Exception:
        hist = tk.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan

    snapshot_time = datetime.utcnow()
    rows = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception as e:
            warnings.warn(f"Failed to fetch chain for {ticker} {exp}: {e}")
            continue

        for opt_type, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            f = frame.copy()
            f["option_type"] = opt_type
            f["expiration"] = pd.to_datetime(exp)
            rows.append(f)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["ticker"] = ticker
    df["spot_price"] = spot
    df["snapshot_time"] = snapshot_time

    keep_cols = [
        "ticker", "contractSymbol", "option_type", "expiration", "strike",
        "bid", "ask", "lastPrice", "volume", "openInterest",
        "impliedVolatility", "spot_price", "snapshot_time", "inTheMoney",
        "lastTradeDate",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].rename(columns={
        "contractSymbol": "contract_symbol",
        "lastPrice": "last_price",
        "openInterest": "open_interest",
        "impliedVolatility": "yf_implied_vol",
        "inTheMoney": "in_the_money",
        "lastTradeDate": "last_trade_date",
    })

    _save(df, cache_name)
    return df


def get_option_universe(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """Download option chains for a list of tickers and concatenate."""
    frames = []
    for t in tickers:
        try:
            df = get_option_chain(t, use_cache=use_cache)
            if not df.empty:
                frames.append(df)
            else:
                warnings.warn(f"WARNING: no option data retrieved for {t}; skipping.")
        except Exception as e:
            warnings.warn(f"Failed to download option chain for {t}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Risk-free rate data
# --------------------------------------------------------------------------
def get_treasury_yields(use_cache: bool = True) -> pd.DataFrame:
    from . import rates
    cache_name = "treasury_yields.parquet"
    if use_cache and _is_fresh(cache_name, max_age_hours=12.0):
        cached = _load(cache_name)
        if cached is not None:
            return cached
    df = rates.fetch_treasury_yields(yf)
    _save(df, cache_name)
    return df
