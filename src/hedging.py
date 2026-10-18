"""
hedging.py
==========
Delta-hedging backtest engine.

IMPORTANT DATA CAVEAT (documented per project requirements):
`yfinance` only exposes the *current* option chain -- it does not provide
historical bid/ask/IV time series for a given contract across its life.
That means a truly "replay the actual historical option market" backtest
is not possible from this data source for options that have not yet
expired (i.e. every contract in a live chain). To still produce a
rigorous, non-fabricated backtest, this module uses a **historical
block-bootstrap** of the underlying's own realized daily log-returns:
a contiguous block of actual historical daily returns (length = the
option's time-to-expiry in trading days) is resampled from the ticker's
real return history and replayed forward from today's spot price. This
means the *path shape* is drawn from real historical return dynamics
(fat tails, volatility clustering to the extent it appears within the
sampled block, etc.), rather than an idealized constant-volatility GBM,
while still being explicit that it is a simulated forward path, not the
option's actual realized future.

Everything downstream (Greeks, discounting, transaction costs) is
computed exactly as it would be in a live hedging desk.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import config
from . import black_scholes as bs
from . import binomial as bn


# --------------------------------------------------------------------------
# Path generation
# --------------------------------------------------------------------------
def bootstrap_price_path(
    log_returns: np.ndarray,
    S0: float,
    n_days: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw a contiguous block of `n_days` real historical daily log-returns
    from `log_returns` (sampled uniformly over all valid start points) and
    replay them forward from S0. Returns an array of length n_days+1
    (S0, S1, ..., S_{n_days}).
    """
    log_returns = log_returns[np.isfinite(log_returns)]
    if len(log_returns) <= n_days:
        # Not enough history for a contiguous block: sample with replacement
        rng = np.random.default_rng(seed)
        block = rng.choice(log_returns, size=n_days, replace=True)
    else:
        rng = np.random.default_rng(seed)
        start = rng.integers(0, len(log_returns) - n_days)
        block = log_returns[start:start + n_days]

    path = np.empty(n_days + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(block))
    return path


# --------------------------------------------------------------------------
# Single-contract hedge simulation
# --------------------------------------------------------------------------
@dataclass
class HedgeResult:
    final_pnl: float
    pnl_before_costs: float
    total_txn_costs: float
    n_rebalances: int
    max_drawdown: float
    pnl_path: np.ndarray
    delta_path: np.ndarray


def _delta_for_model(model: str, S, K, T, r, sigma, option_type) -> float:
    if T <= 1e-8:
        intrinsic_itm = (S > K) if option_type == "call" else (S < K)
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    if model == "bs":
        return float(bs.bs_delta(S, K, T, r, sigma, option_type))
    elif model == "crr":
        return bn.crr_price(S, K, T, r, sigma, N=config.CRR_DEFAULT_STEPS,
                             option_type=option_type, exercise="european")["delta"]
    else:
        raise ValueError("model must be 'bs' or 'crr'")


def run_delta_hedge(
    S0: float,
    K: float,
    T_years: float,
    r: float,
    sigma: float,
    option_type: str,
    price_path: np.ndarray,
    model: str = "bs",
    rebalance_freq: str = "daily",
    txn_cost: Optional[config.TransactionCostModel] = None,
    quoted_spread_frac: float = 0.02,
) -> HedgeResult:
    """
    Simulate selling one option and dynamically delta-hedging it along
    `price_path` (length n_days+1, evenly spaced trading days spanning
    [0, T_years]).

    Cash-flow convention:
        t=0: receive option premium P0 (model price). Buy delta_0 shares.
        each rebalance: trade to the new target delta; cash absorbs the
                         cost/proceeds of the trade plus any transaction cost.
        between rebalances: cash accrues at the continuously-compounded
                         risk-free rate.
        at expiry: pay the option's intrinsic payoff to the counterparty;
                         liquidate the remaining share position.

    `final_pnl` = terminal portfolio value (cash + stock - payoff), i.e. the
    hedging error. A perfect, continuous hedge in a world matching the
    model's assumptions would have final_pnl == 0 in expectation.
    """
    n_days = len(price_path) - 1
    dt = T_years / n_days if n_days > 0 else T_years
    step = 1 if rebalance_freq == "daily" else 5  # ~weekly if daily obs

    apply_costs = txn_cost is not None

    P0 = bs.bs_price(S0, K, T_years, r, sigma, option_type) if model == "bs" else \
        bn.crr_price(S0, K, T_years, r, sigma, N=config.CRR_DEFAULT_STEPS, option_type=option_type, exercise="european")["price"]

    cash = P0
    delta_prev = 0.0
    pnl_path = np.zeros(n_days + 1)
    delta_path = np.zeros(n_days + 1)
    total_costs = 0.0
    n_rebalances = 0

    for t in range(0, n_days + 1):
        S_t = price_path[t]
        T_remaining = max(T_years - t * dt, 0.0)

        rebalance_today = (t == 0) or (t % step == 0) or (t == n_days)
        if rebalance_today:
            delta_t = _delta_for_model(model, S_t, K, T_remaining, r, sigma, option_type)
            trade_size = delta_t - delta_prev
            trade_notional = trade_size * S_t
            cash -= trade_notional  # buying shares (positive trade) costs cash

            if apply_costs and trade_size != 0:
                per_share = txn_cost.per_share_cost * abs(trade_size)
                spread_cost = txn_cost.spread_cost_fraction * quoted_spread_frac * S_t * abs(trade_size)
                cost = per_share + spread_cost
                cash -= cost
                total_costs += cost

            delta_prev = delta_t
            n_rebalances += 1
        else:
            delta_t = delta_prev

        delta_path[t] = delta_t

        # Accrue interest on cash for one day (approx dt) before the next step
        if t < n_days:
            cash *= np.exp(r * dt)

        stock_value = delta_prev * S_t
        pnl_path[t] = cash + stock_value - P0  # mark-to-market P&L vs premium received

    S_T = price_path[-1]
    payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
    final_stock_value = delta_prev * S_T
    final_portfolio = cash + final_stock_value - payoff
    final_pnl = final_portfolio - 0.0  # hedging error relative to a theoretically-perfect (zero) outcome

    running_max = np.maximum.accumulate(pnl_path)
    drawdown = pnl_path - running_max
    max_drawdown = float(drawdown.min())

    return HedgeResult(
        final_pnl=float(final_pnl),
        pnl_before_costs=float(final_pnl + total_costs),
        total_txn_costs=float(total_costs),
        n_rebalances=n_rebalances,
        max_drawdown=max_drawdown,
        pnl_path=pnl_path,
        delta_path=delta_path,
    )


# --------------------------------------------------------------------------
# Batch backtest across many contracts
# --------------------------------------------------------------------------
def batch_hedge_backtest(
    contracts: pd.DataFrame,
    log_returns_by_ticker: Dict[str, np.ndarray],
    models: List[str] = ("bs", "crr"),
    rebalance_freqs: List[str] = ("daily", "weekly"),
    apply_txn_costs: List[bool] = (False, True),
    n_paths_per_contract: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the delta-hedging backtest across a DataFrame of contracts (each row
    must have: ticker, spot_price, strike, T_years, risk_free_rate, a
    volatility column 'hedge_vol', and option_type). For each contract,
    `n_paths_per_contract` independent bootstrap paths are simulated to
    average out idiosyncratic path noise.
    """
    rows = []
    rng = np.random.default_rng(seed)

    for _, row in contracts.iterrows():
        ticker = row["ticker"]
        lr = log_returns_by_ticker.get(ticker)
        if lr is None or len(lr) < 5:
            continue

        n_days = max(int(round(row["T_years"] * config.TRADING_DAYS_PER_YEAR)), 1)
        n_days = min(n_days, 500)  # cap for computational tractability

        for path_idx in range(n_paths_per_contract):
            path_seed = int(rng.integers(0, 1_000_000))
            path = bootstrap_price_path(lr, row["spot_price"], n_days, seed=path_seed)

            for model in models:
                for freq in rebalance_freqs:
                    for use_costs in apply_txn_costs:
                        txn = config.DEFAULT_TXN_COST if use_costs else None
                        result = run_delta_hedge(
                            S0=row["spot_price"], K=row["strike"], T_years=row["T_years"],
                            r=row["risk_free_rate"], sigma=row["hedge_vol"],
                            option_type=row["option_type"], price_path=path,
                            model=model, rebalance_freq=freq, txn_cost=txn,
                        )
                        rows.append({
                            "ticker": ticker,
                            "contract_symbol": row.get("contract_symbol", np.nan),
                            "moneyness_bucket": row.get("moneyness_bucket", np.nan),
                            "maturity_bucket": row.get("maturity_bucket", np.nan),
                            "model": model,
                            "rebalance_freq": freq,
                            "txn_costs_applied": use_costs,
                            "path_idx": path_idx,
                            "final_pnl": result.final_pnl,
                            "abs_pnl": abs(result.final_pnl),
                            "total_txn_costs": result.total_txn_costs,
                            "max_drawdown": result.max_drawdown,
                            "n_rebalances": result.n_rebalances,
                        })

    return pd.DataFrame(rows)


def hedge_performance_summary(hedge_results: pd.DataFrame, groupby: List[str] = None) -> pd.DataFrame:
    """Aggregate hedge P&L statistics, optionally broken out by grouping columns."""
    default_groups = ["model", "rebalance_freq", "txn_costs_applied"]
    groupby = default_groups if groupby is None else groupby

    def agg(g):
        return pd.Series({
            "n_simulations": len(g),
            "mean_pnl": g["final_pnl"].mean(),
            "std_pnl": g["final_pnl"].std(),
            "rmse_pnl": np.sqrt((g["final_pnl"] ** 2).mean()),
            "mean_abs_pnl": g["abs_pnl"].mean(),
            "max_loss": g["final_pnl"].min(),
            "mean_txn_costs": g["total_txn_costs"].mean(),
        })

    return hedge_results.groupby(groupby, observed=True).apply(agg).reset_index()
