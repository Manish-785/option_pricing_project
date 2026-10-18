"""
visualization.py
=================
Publication-quality matplotlib figures for the option-pricing research
project. Every function saves a PNG to config.FIGURES_DIR and returns the
path, so run_analysis.py can catalogue what was produced.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from . import config

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, name: str) -> str:
    path = os.path.join(config.FIGURES_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_model_vs_market_scatter(df: pd.DataFrame, model_col: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    sub = df.dropna(subset=[model_col, "mid_price"])
    ax.scatter(sub["mid_price"], sub[model_col], s=6, alpha=0.35, edgecolor="none")
    lims = [0, max(sub["mid_price"].max(), sub[model_col].max()) * 1.05] if not sub.empty else [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect agreement (y=x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Market mid price ($)")
    ax.set_ylabel(f"Model price ($) [{model_col}]")
    ax.set_title(f"Model vs Market Price: {model_col}")
    ax.legend()
    return _save(fig, name)


def fig_error_distribution(df: pd.DataFrame, err_cols: list, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in err_cols:
        data = df[col].dropna()
        data = data[np.abs(data) < data.quantile(0.99) + 1e-9] if len(data) else data
        if len(data):
            ax.hist(data, bins=60, alpha=0.5, label=col.replace("_pct_err_mid", ""), density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Relative pricing error ((model - mid) / mid)")
    ax.set_ylabel("Density")
    ax.set_title("Pricing Error Distribution by Model")
    ax.legend(fontsize=8)
    return _save(fig, name)


def fig_error_by_category(df: pd.DataFrame, category_col: str, err_col: str, name: str, order: list = None) -> str:
    sub = df.dropna(subset=[category_col, err_col]).copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return _save(fig, name)
    if order:
        cats = [c for c in order if c in sub[category_col].unique()]
    else:
        cats = sorted(sub[category_col].unique(), key=str)
    data = [sub.loc[sub[category_col] == c, err_col].dropna().values for c in cats]
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.boxplot(data, tick_labels=cats, showfliers=False)
    except TypeError:
        # Matplotlib < 3.9 used `labels`; newer versions renamed it to `tick_labels`.
        ax.boxplot(data, labels=cats, showfliers=False)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_ylabel(err_col)
    ax.set_xlabel(category_col)
    ax.set_title(f"Pricing Error by {category_col}")
    plt.xticks(rotation=30, ha="right")
    return _save(fig, name)


def fig_crr_convergence(convergence: dict, bs_reference: float, name: str) -> str:
    Ns = sorted(convergence.keys())
    prices = [convergence[n]["price"] for n in Ns]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ns, prices, "o-", label="CRR price")
    ax.axhline(bs_reference, color="red", linestyle="--", label="Black-Scholes price")
    ax.set_xlabel("Number of tree steps (N)")
    ax.set_ylabel("Option price ($)")
    ax.set_title("CRR Binomial Convergence to Black-Scholes")
    ax.legend()
    return _save(fig, name)


def fig_mc_convergence(convergence: dict, bs_reference: float, name: str) -> str:
    Ns = sorted(convergence.keys())
    prices = [convergence[n]["price"] for n in Ns]
    ses = [convergence[n]["std_error"] for n in Ns]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(Ns, prices, yerr=[1.96 * s for s in ses], fmt="o-", capsize=4, label="MC price ± 95% CI")
    ax.axhline(bs_reference, color="red", linestyle="--", label="Black-Scholes price")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Option price ($)")
    ax.set_title("Monte Carlo Convergence to Black-Scholes")
    ax.legend()
    return _save(fig, name)


def fig_iv_smile(df: pd.DataFrame, maturity_bucket: str, name: str) -> str:
    sub = df[(df["maturity_bucket"] == maturity_bucket)].dropna(subset=["market_iv", "log_moneyness"])
    sub = sub[(sub["market_iv"] > 0) & (sub["market_iv"] < 3.0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not sub.empty:
        for ot, marker in (("call", "o"), ("put", "^")):
            s2 = sub[sub["option_type"] == ot]
            ax.scatter(s2["log_moneyness"], s2["market_iv"], s=10, alpha=0.5, label=ot, marker=marker)
    ax.set_xlabel("Log-moneyness log(S/K)")
    ax.set_ylabel("Implied volatility")
    ax.set_title(f"Implied Volatility Smile ({maturity_bucket})")
    ax.legend()
    return _save(fig, name)


def fig_iv_surface(grid: pd.DataFrame, name: str) -> str:
    fig = plt.figure(figsize=(8, 6))
    if grid.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Insufficient data for IV surface", ha="center", va="center")
        return _save(fig, name)

    def interval_mid(value):
        if pd.isna(value):
            return np.nan
        if hasattr(value, "mid"):
            return value.mid
        if isinstance(value, str):
            try:
                low, high = value.strip("()[]").split(",")
                return (float(low) + float(high)) / 2.0
            except ValueError:
                return np.nan
        return np.nan

    grid = grid.copy()
    grid["log_moneyness_mid"] = grid["log_moneyness_bin"].apply(interval_mid)
    grid["T_mid"] = grid["T_years_bin"].apply(interval_mid)
    grid = grid.dropna(subset=["log_moneyness_mid", "T_mid", "mean_iv"])

    ax = fig.add_subplot(111, projection="3d")
    if not grid.empty:
        surf = ax.plot_trisurf(grid["log_moneyness_mid"], grid["T_mid"], grid["mean_iv"],
                                cmap="viridis", edgecolor="none", alpha=0.9)
        fig.colorbar(surf, shrink=0.6, label="Implied Vol")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Time to maturity (yrs)")
    ax.set_zlabel("Implied vol")
    ax.set_title("Implied Volatility Surface")
    return _save(fig, name)


def fig_delta_comparison(df: pd.DataFrame, delta_cols: list, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    sub = df.dropna(subset=delta_cols)
    if not sub.empty:
        ax.scatter(sub[delta_cols[0]], sub[delta_cols[1]], s=6, alpha=0.35)
        lims = [-1.05, 1.05]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(delta_cols[0])
    ax.set_ylabel(delta_cols[1])
    ax.set_title("Model Delta Comparison")
    return _save(fig, name)


def fig_hedge_pnl_distribution(hedge_results: pd.DataFrame, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in hedge_results["model"].unique():
        data = hedge_results.loc[hedge_results["model"] == model, "final_pnl"].dropna()
        if len(data):
            ax.hist(data, bins=40, alpha=0.5, label=model, density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Hedging P&L ($)")
    ax.set_ylabel("Density")
    ax.set_title("Delta-Hedging P&L Distribution by Model")
    ax.legend()
    return _save(fig, name)


def fig_cumulative_hedge_pnl(pnl_paths: dict, name: str) -> str:
    """pnl_paths: dict[label] -> 1D array of cumulative P&L over the hedge horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in pnl_paths.items():
        ax.plot(path, label=label, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Trading day into the hedge")
    ax.set_ylabel("Cumulative hedging P&L ($)")
    ax.set_title("Cumulative Delta-Hedging P&L")
    ax.legend(fontsize=8)
    return _save(fig, name)
