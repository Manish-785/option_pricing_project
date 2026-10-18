"""
monte_carlo.py
===============
Risk-neutral Monte Carlo option pricing under GBM.

Features:
    * Standard and antithetic-variate simulation
    * Standard error and 95% confidence intervals
    * Delta via finite differences (bump-and-reprice, common random numbers)
    * Convergence testing across path counts
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List

from . import config


def _terminal_prices(S, T, r, sigma, n_paths, q, seed, antithetic, z=None):
    """Simulate terminal GBM prices; optionally reuse a supplied z (for CRN delta)."""
    rng = np.random.default_rng(seed)
    if z is None:
        if antithetic:
            half = n_paths // 2
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
        else:
            z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    ST = S * np.exp(drift + diffusion)
    return ST, z


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = None,
    antithetic: bool = True,
    seed: int = None,
    compute_delta: bool = True,
    delta_bump: float = None,
) -> Dict[str, float]:
    """
    Price a European option via risk-neutral Monte Carlo simulation.

    Returns price, standard error, 95% CI, and (optionally) a finite-difference
    delta estimated with common random numbers to reduce variance.
    """
    n_paths = config.MC_DEFAULT_PATHS if n_paths is None else n_paths
    seed = config.MC_RANDOM_SEED if seed is None else seed
    delta_bump = config.MC_DELTA_BUMP if delta_bump is None else delta_bump

    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return {"price": intrinsic, "std_error": 0.0, "ci_low": intrinsic,
                "ci_high": intrinsic, "delta": np.nan, "n_paths": n_paths}

    sigma = max(sigma, 1e-6)
    ST, z = _terminal_prices(S, T, r, sigma, n_paths, q, seed, antithetic)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    disc = np.exp(-r * T)
    discounted = disc * payoff

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci_low, ci_high = price - 1.96 * std_error, price + 1.96 * std_error

    result = {
        "price": price,
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_paths": n_paths,
    }

    if compute_delta:
        # Finite-difference delta using the SAME random draws (common random
        # numbers) at bumped spot levels to cut variance in the delta estimate.
        bump = S * delta_bump
        ST_up, _ = _terminal_prices(S + bump, T, r, sigma, n_paths, q, seed, antithetic, z=z)
        ST_dn, _ = _terminal_prices(S - bump, T, r, sigma, n_paths, q, seed, antithetic, z=z)
        if option_type == "call":
            pay_up = np.maximum(ST_up - K, 0.0)
            pay_dn = np.maximum(ST_dn - K, 0.0)
        else:
            pay_up = np.maximum(K - ST_up, 0.0)
            pay_dn = np.maximum(K - ST_dn, 0.0)
        price_up = disc * np.mean(pay_up)
        price_dn = disc * np.mean(pay_dn)
        result["delta"] = float((price_up - price_dn) / (2 * bump))
        result["delta_method"] = "finite_difference_crn"
    else:
        result["delta"] = np.nan

    return result


def mc_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    antithetic: bool = True,
    seed: int = None,
    paths: List[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Price the same option across a range of path counts to show convergence."""
    paths = config.MC_CONVERGENCE_PATHS if paths is None else paths
    return {
        n: mc_price(S, K, T, r, sigma, option_type=option_type, q=q,
                    n_paths=n, antithetic=antithetic, seed=seed, compute_delta=False)
        for n in paths
    }
