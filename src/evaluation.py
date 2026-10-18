"""
evaluation.py
=============
Applies each pricing model (Black-Scholes, CRR Binomial, Monte Carlo) to
every contract in the cleaned dataset -- separately under historical
volatility and market-implied volatility -- computes pricing errors against
both last-price and mid-price market references, and produces the
error-metric summary tables and paired statistical tests required by the
research design.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

from . import config
from . import black_scholes as bs
from . import binomial as bn
from . import monte_carlo as mc


# --------------------------------------------------------------------------
# Model price generation
# --------------------------------------------------------------------------
def add_model_prices(
    df: pd.DataFrame,
    hist_vol_col: str = "realized_vol_60d",
    iv_col: str = "market_iv",
    crr_steps: int = None,
    mc_paths: int = None,
    include_american_crr: bool = True,
) -> pd.DataFrame:
    """
    Compute BS / CRR-European / CRR-American / MC prices (and available
    Deltas) for every row, under both historical and implied volatility.

    This is a row-by-row loop for CRR and MC (they are not embarrassingly
    vectorisable across strikes/maturities the way BS is) -- the tree size
    and path count therefore default to the project's "reasonable" settings
    rather than the full convergence-test grid, which is run separately on
    a small sample of contracts (see run_analysis.py).
    """
    crr_steps = config.CRR_DEFAULT_STEPS if crr_steps is None else crr_steps
    mc_paths = config.MC_DEFAULT_PATHS if mc_paths is None else mc_paths

    out = df.copy()

    for vol_source, vol_col in (("hist", hist_vol_col), ("iv", iv_col)):
        S = out["spot_price"].to_numpy()
        K = out["strike"].to_numpy()
        T = out["T_years"].to_numpy()
        r = out["risk_free_rate"].to_numpy()
        sigma = out[vol_col].to_numpy()
        opt_type = out["option_type"].to_numpy()

        # --- Black-Scholes: fully vectorised ---
        bs_price = np.full(len(out), np.nan)
        bs_delta = np.full(len(out), np.nan)
        for ot in ("call", "put"):
            m = opt_type == ot
            if m.any():
                bs_price[m] = bs.bs_price(S[m], K[m], T[m], r[m], sigma[m], ot)
                bs_delta[m] = bs.bs_delta(S[m], K[m], T[m], r[m], sigma[m], ot)
        out[f"bs_price_{vol_source}"] = bs_price
        out[f"bs_delta_{vol_source}"] = bs_delta

        # --- CRR / MC: row-by-row ---
        crr_eu_price = np.full(len(out), np.nan)
        crr_eu_delta = np.full(len(out), np.nan)
        crr_am_price = np.full(len(out), np.nan)
        crr_am_delta = np.full(len(out), np.nan)
        mc_price_arr = np.full(len(out), np.nan)
        mc_delta_arr = np.full(len(out), np.nan)
        mc_se_arr = np.full(len(out), np.nan)

        valid_rows = np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(sigma) & (T > 0) & (sigma > 0)

        for i in np.where(valid_rows)[0]:
            ot = opt_type[i]
            res_eu = bn.crr_price(S[i], K[i], T[i], r[i], sigma[i], N=crr_steps, option_type=ot, exercise="european")
            crr_eu_price[i], crr_eu_delta[i] = res_eu["price"], res_eu["delta"]

            if include_american_crr:
                res_am = bn.crr_price(S[i], K[i], T[i], r[i], sigma[i], N=crr_steps, option_type=ot, exercise="american")
                crr_am_price[i], crr_am_delta[i] = res_am["price"], res_am["delta"]

            res_mc = mc.mc_price(S[i], K[i], T[i], r[i], sigma[i], option_type=ot,
                                  n_paths=mc_paths, seed=config.MC_RANDOM_SEED, compute_delta=True)
            mc_price_arr[i] = res_mc["price"]
            mc_delta_arr[i] = res_mc["delta"]
            mc_se_arr[i] = res_mc["std_error"]

        out[f"crr_eu_price_{vol_source}"] = crr_eu_price
        out[f"crr_eu_delta_{vol_source}"] = crr_eu_delta
        if include_american_crr:
            out[f"crr_am_price_{vol_source}"] = crr_am_price
            out[f"crr_am_delta_{vol_source}"] = crr_am_delta
        out[f"mc_price_{vol_source}"] = mc_price_arr
        out[f"mc_delta_{vol_source}"] = mc_delta_arr
        out[f"mc_std_error_{vol_source}"] = mc_se_arr

    return out


# --------------------------------------------------------------------------
# Error columns
# --------------------------------------------------------------------------
MODEL_PRICE_COLUMNS = {
    "BlackScholes_Hist": "bs_price_hist",
    "BlackScholes_IV": "bs_price_iv",
    "CRR_European_Hist": "crr_eu_price_hist",
    "CRR_European_IV": "crr_eu_price_iv",
    "CRR_American_Hist": "crr_am_price_hist",
    "CRR_American_IV": "crr_am_price_iv",
    "MonteCarlo_Hist": "mc_price_hist",
    "MonteCarlo_IV": "mc_price_iv",
}


def add_pricing_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every model-price column present, compute:
        {model}_err_last  = model_price - last_price
        {model}_err_mid   = model_price - mid_price
        {model}_abs_err_mid
        {model}_pct_err_mid  (relative to mid_price, only where mid_price
                               exceeds config.MIN_MARKET_PRICE_FOR_MAPE)
    """
    out = df.copy()
    for model_name, col in MODEL_PRICE_COLUMNS.items():
        if col not in out.columns:
            continue
        out[f"{model_name}_err_last"] = out[col] - out["last_price"]
        out[f"{model_name}_err_mid"] = out[col] - out["mid_price"]
        out[f"{model_name}_abs_err_mid"] = out[f"{model_name}_err_mid"].abs()

        safe_mid = out["mid_price"].where(out["mid_price"] >= config.MIN_MARKET_PRICE_FOR_MAPE)
        out[f"{model_name}_pct_err_mid"] = out[f"{model_name}_err_mid"] / safe_mid
    return out


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------
def _metrics_for_group(g: pd.DataFrame, model_name: str) -> Dict[str, float]:
    abs_err = g[f"{model_name}_abs_err_mid"].dropna()
    pct_err = g[f"{model_name}_pct_err_mid"].dropna()
    signed_err = g[f"{model_name}_err_mid"].dropna()

    if abs_err.empty:
        return {
            "n_obs": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
            "MedianAPE": np.nan, "MeanError": np.nan,
            "Pct_within_5": np.nan, "Pct_within_10": np.nan, "Pct_within_20": np.nan,
        }

    thresholds = config.ACCURACY_THRESHOLDS
    pct_abs = pct_err.abs()

    return {
        "n_obs": int(len(abs_err)),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((signed_err ** 2).mean())),
        "MAPE": float(pct_abs.mean() * 100) if not pct_abs.empty else np.nan,
        "MedianAPE": float(pct_abs.median() * 100) if not pct_abs.empty else np.nan,
        "MeanError": float(signed_err.mean()),
        "Pct_within_5": float((pct_abs <= thresholds[0]).mean() * 100) if not pct_abs.empty else np.nan,
        "Pct_within_10": float((pct_abs <= thresholds[1]).mean() * 100) if not pct_abs.empty else np.nan,
        "Pct_within_20": float((pct_abs <= thresholds[2]).mean() * 100) if not pct_abs.empty else np.nan,
    }


def model_performance_table(df: pd.DataFrame, groupby: List[str] = None) -> pd.DataFrame:
    """
    Build the model-comparison table, optionally broken out by additional
    grouping columns (e.g. ['ticker'], ['moneyness_bucket'], ['maturity_bucket']).
    """
    models = [m for m in MODEL_PRICE_COLUMNS if f"{m}_abs_err_mid" in df.columns]
    rows = []

    if groupby:
        for keys, g in df.groupby(groupby, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            for model in models:
                metrics = _metrics_for_group(g, model)
                row = dict(zip(groupby, keys))
                row["Model"] = model
                row.update(metrics)
                rows.append(row)
    else:
        for model in models:
            metrics = _metrics_for_group(df, model)
            row = {"Model": model}
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def paired_error_test(df: pd.DataFrame, model_a: str, model_b: str) -> Dict[str, float]:
    """
    Paired comparison of absolute pricing error between two models on the
    same contracts. Uses a paired t-test (mean-difference significance) and
    the non-parametric Wilcoxon signed-rank test (robust to the heavy right
    skew typical of option pricing errors) and reports both.
    """
    col_a = f"{model_a}_abs_err_mid"
    col_b = f"{model_b}_abs_err_mid"
    if col_a not in df.columns or col_b not in df.columns:
        return {}

    paired = df[[col_a, col_b]].dropna()
    if len(paired) < 2:
        return {"n_pairs": len(paired)}

    diff = paired[col_a] - paired[col_b]
    t_stat, t_p = stats.ttest_rel(paired[col_a], paired[col_b])
    try:
        w_stat, w_p = stats.wilcoxon(paired[col_a], paired[col_b])
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    return {
        "n_pairs": int(len(paired)),
        "mean_abs_error_A": float(paired[col_a].mean()),
        "mean_abs_error_B": float(paired[col_b].mean()),
        "mean_difference_A_minus_B": float(diff.mean()),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_pvalue": float(w_p) if np.isfinite(w_p) else np.nan,
        "interpretation": (
            f"{model_a} has {'lower' if diff.mean() < 0 else 'higher'} mean absolute "
            f"pricing error than {model_b} by {abs(diff.mean()):.4f} on average "
            f"across {len(paired)} paired contracts. "
            f"{'This difference is statistically significant at the 5% level (paired t-test p='+format(t_p,'.4g')+').' if t_p < 0.05 else 'This difference is NOT statistically significant at the 5% level (paired t-test p='+format(t_p,'.4g')+'), so it could plausibly be due to sampling noise rather than a genuine accuracy gap.'}"
        ),
    }


def run_all_pairwise_tests(df: pd.DataFrame, model_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for a, b in model_pairs:
        result = paired_error_test(df, a, b)
        if result:
            result["Model_A"] = a
            result["Model_B"] = b
            rows.append(result)
    return pd.DataFrame(rows)
