"""
black_scholes.py
=================
Black-Scholes-Merton pricing, Greeks, and implied-volatility inversion.

All functions are vectorised (accept scalars or numpy arrays / pandas
Series) and numerically guarded against the standard degenerate cases:
T -> 0, sigma -> 0, S <= 0, K <= 0.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from . import config


def _sanitize(S, K, T, sigma):
    """Clip inputs into numerically safe ranges; returns arrays."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    S = np.where(S <= 0, np.nan, S)
    K = np.where(K <= 0, np.nan, K)
    T = np.clip(T, 1e-8, None)      # avoid division by zero as T -> 0
    sigma = np.clip(sigma, 1e-6, None)  # avoid division by zero as sigma -> 0
    return S, K, T, sigma


def d1_d2(S, K, T, r, sigma, q: float = 0.0):
    """Compute d1 and d2 for the BSM formula (q = continuous dividend yield)."""
    S, K, T, sigma = _sanitize(S, K, T, sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0):
    """
    Black-Scholes-Merton European option price.

    Parameters
    ----------
    S, K, T, r, sigma : scalars or arrays
        Spot, strike, time-to-expiry (years), risk-free rate, volatility.
    option_type : 'call' or 'put'
    q : continuous dividend yield (annualised)

    Edge cases (T -> 0) are handled by falling back to intrinsic value.
    """
    S_raw = np.asarray(S, dtype=float)
    K_raw = np.asarray(K, dtype=float)
    T_raw = np.asarray(T, dtype=float)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_s, K_s, T_s, sigma_s = _sanitize(S, K, T, sigma)

    if option_type == "call":
        price = S_s * np.exp(-q * T_s) * norm.cdf(d1) - K_s * np.exp(-r * T_s) * norm.cdf(d2)
        intrinsic = np.maximum(S_raw - K_raw, 0.0)
    elif option_type == "put":
        price = K_s * np.exp(-r * T_s) * norm.cdf(-d2) - S_s * np.exp(-q * T_s) * norm.cdf(-d1)
        intrinsic = np.maximum(K_raw - S_raw, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # For (near-)expired options, use intrinsic value instead of the formula.
    near_expiry = T_raw <= 1e-6
    price = np.where(near_expiry, intrinsic, price)
    price = np.where(np.isnan(price), intrinsic, price)  # NaN safety net
    return price if price.shape else float(price)


def bs_delta(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        delta = np.exp(-q * np.clip(np.asarray(T, dtype=float), 1e-8, None)) * norm.cdf(d1)
    elif option_type == "put":
        delta = -np.exp(-q * np.clip(np.asarray(T, dtype=float), 1e-8, None)) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return delta


def bs_gamma(S, K, T, r, sigma, q: float = 0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_s, _, T_s, sigma_s = _sanitize(S, K, T, sigma)
    gamma = np.exp(-q * T_s) * norm.pdf(d1) / (S_s * sigma_s * np.sqrt(T_s))
    return gamma


def bs_vega(S, K, T, r, sigma, q: float = 0.0):
    """Vega per 1.00 (100%) change in vol; divide by 100 for per-vol-point."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_s, _, T_s, _ = _sanitize(S, K, T, sigma)
    vega = S_s * np.exp(-q * T_s) * norm.pdf(d1) * np.sqrt(T_s)
    return vega


def bs_theta(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0):
    """Theta per year; divide by 365 for per-calendar-day."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_s, K_s, T_s, sigma_s = _sanitize(S, K, T, sigma)
    term1 = -(S_s * np.exp(-q * T_s) * norm.pdf(d1) * sigma_s) / (2 * np.sqrt(T_s))
    if option_type == "call":
        theta = term1 - r * K_s * np.exp(-r * T_s) * norm.cdf(d2) + q * S_s * np.exp(-q * T_s) * norm.cdf(d1)
    elif option_type == "put":
        theta = term1 + r * K_s * np.exp(-r * T_s) * norm.cdf(-d2) - q * S_s * np.exp(-q * T_s) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return theta


def bs_rho(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0):
    """Rho per 1.00 (100%) change in r; divide by 100 for per-percentage-point."""
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    _, K_s, T_s, _ = _sanitize(S, K, T, sigma)
    if option_type == "call":
        rho = K_s * T_s * np.exp(-r * T_s) * norm.cdf(d2)
    elif option_type == "put":
        rho = -K_s * T_s * np.exp(-r * T_s) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return rho


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    lo: float = None,
    hi: float = None,
    tol: float = None,
    maxiter: int = None,
):
    """
    Invert the Black-Scholes formula for implied volatility using Brent's
    method. Returns np.nan if inversion fails (e.g. price outside
    no-arbitrage bounds, or a root can't be bracketed).
    """
    lo = config.IV_SOLVER_LOW if lo is None else lo
    hi = config.IV_SOLVER_HIGH if hi is None else hi
    tol = config.IV_SOLVER_TOL if tol is None else tol
    maxiter = config.IV_SOLVER_MAXITER if maxiter is None else maxiter

    if market_price is None or not np.isfinite(market_price) or market_price <= 0:
        return np.nan
    if S <= 0 or K <= 0 or T <= 0:
        return np.nan

    # No-arbitrage sanity bounds: price must exceed intrinsic (approx, no div)
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return np.nan  # price violates arbitrage bound -> cannot invert sensibly

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
            return np.nan
        iv = brentq(objective, lo, hi, xtol=tol, maxiter=maxiter)
        return iv
    except Exception:
        return np.nan
