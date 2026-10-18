"""
binomial.py
===========
Cox-Ross-Rubinstein (CRR) binomial tree option pricer.

Supports European and American exercise, returns price and delta, and
provides a convenience function for convergence testing across tree sizes.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List

from . import config


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "call",
    exercise: str = "european",
    q: float = 0.0,
) -> Dict[str, float]:
    """
    Price a European or American option with a CRR binomial tree.

    Returns a dict with 'price' and 'delta' (delta estimated from the first
    step of the tree, the standard binomial-tree delta estimator).
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return {"price": intrinsic, "delta": 1.0 if (option_type == "call" and S > K) else
                (-1.0 if (option_type == "put" and S < K) else 0.0)}

    sigma = max(sigma, 1e-6)
    N = max(int(N), 2)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # numerical safety; should hold for reasonable sigma*sqrt(dt)

    # Terminal asset prices: S * u^j * d^(N-j) for j = 0..N
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction
    # Keep asset-price tree at each step only when needed for American exercise
    for step in range(N - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if exercise == "american":
            j_step = np.arange(step + 1)
            ST_step = S * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                exercise_value = np.maximum(ST_step - K, 0.0)
            else:
                exercise_value = np.maximum(K - ST_step, 0.0)
            values = np.maximum(values, exercise_value)
        if step == 1:
            # capture the two nodes at t=dt to compute delta
            up_val, down_val = values[1], values[0]
            S_up, S_down = S * u, S * d

    price = float(values[0])
    delta = float((up_val - down_val) / (S_up - S_down))
    return {"price": price, "delta": delta}


def crr_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    exercise: str = "european",
    q: float = 0.0,
    steps: List[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Price the same option across a range of tree sizes N to show convergence."""
    steps = config.CRR_CONVERGENCE_STEPS if steps is None else steps
    return {
        N: crr_price(S, K, T, r, sigma, N=N, option_type=option_type, exercise=exercise, q=q)
        for N in steps
    }
