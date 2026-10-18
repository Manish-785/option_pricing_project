#!/usr/bin/env python3
"""
run_analysis.py
================
End-to-end research pipeline:

    1. Download & cache underlying + option-chain + Treasury-yield data.
    2. Clean and classify the option dataset (liquidity, moneyness, maturity,
       vol regime, risk-free rate).
    3. Compute market-implied volatility per contract; build the IV surface.
    4. Price every contract with Black-Scholes, CRR (European & American),
       and Monte Carlo, under both historical and implied volatility.
    5. Compute pricing-error metrics, breakdowns, and paired significance tests.
    6. Run CRR / Monte Carlo convergence studies on a sample of contracts.
    7. Run the delta-hedging backtest (BS vs CRR deltas; daily vs weekly;
       with/without transaction costs).
    8. Run controlled stress tests (vol / rate / spot shocks).
    9. Generate all required figures and CSV tables.
    10. Write research_summary.md with real, computed findings only.

Usage:
    python run_analysis.py
    python run_analysis.py --tickers AAPL MSFT NVDA
    python run_analysis.py --no-cache
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import config
from src import data
from src import preprocessing
from src import black_scholes as bs
from src import binomial as bn
from src import monte_carlo as mc
from src import implied_vol
from src import greeks
from src import evaluation
from src import hedging
from src import visualization as viz
from src import stress_testing


def parse_args():
    p = argparse.ArgumentParser(description="Option pricing model comparison research pipeline")
    p.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS,
                    help="List of tickers to analyze")
    p.add_argument("--no-cache", action="store_true", help="Force fresh downloads, ignore cache")
    p.add_argument("--sample-contracts", type=int, default=15,
                    help="Number of contracts sampled for CRR/MC convergence plots")
    p.add_argument("--hedge-sample", type=int, default=40,
                    help="Number of contracts sampled for the delta-hedging backtest")
    p.add_argument("--figures-only", action="store_true",
                    help="Resume from existing data/results files and run sections 10-12 only")
    return p.parse_args()


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_existing_file(path: str) -> str:
    if not os.path.exists(path):
        print(f"FATAL: required resume artifact is missing: {path}")
        print("Run the full pipeline first, or restore the missing file before using --figures-only.")
        sys.exit(1)
    return path


def load_resume_artifacts():
    """Load the persisted artifacts needed to resume at section 10."""
    df = pd.read_parquet(require_existing_file(f"{config.DATA_DIR}/priced_options.parquet"))
    iv_surface_grid = pd.read_csv(require_existing_file(f"{config.RESULTS_DIR}/iv_surface_grid.csv"))
    overall_table = pd.read_csv(require_existing_file(f"{config.RESULTS_DIR}/table2_model_performance.csv"))
    by_moneyness = pd.read_csv(require_existing_file(f"{config.RESULTS_DIR}/table3_performance_by_moneyness.csv"))
    by_maturity = pd.read_csv(require_existing_file(f"{config.RESULTS_DIR}/table4_performance_by_maturity.csv"))
    stat_tests = pd.read_csv(require_existing_file(f"{config.RESULTS_DIR}/pairwise_statistical_tests.csv"))

    hedge_path = f"{config.RESULTS_DIR}/hedge_backtest_raw.csv"
    hedge_results = pd.read_csv(hedge_path) if os.path.exists(hedge_path) else pd.DataFrame()
    return df, iv_surface_grid, overall_table, by_moneyness, by_maturity, stat_tests, hedge_results


def main():
    args = parse_args()
    use_cache = not args.no_cache
    tickers = args.tickers

    if args.figures_only:
        print("Resuming from existing artifacts; skipping data collection, pricing, and backtests.")
        df, iv_surface_grid, overall_table, by_moneyness, by_maturity, stat_tests, hedge_results = load_resume_artifacts()
        run_reporting_sections(df, iv_surface_grid, hedge_results,
                               overall_table, by_moneyness, by_maturity, stat_tests)
        return

    # ----------------------------------------------------------------
    section("1. DATA COLLECTION")
    # ----------------------------------------------------------------
    print(f"Universe: {tickers}")
    underlying_df = data.get_underlying_universe(tickers, use_cache=use_cache)
    options_df = data.get_option_universe(tickers, use_cache=use_cache)
    yields_df = data.get_treasury_yields(use_cache=use_cache)

    if options_df.empty:
        print("\nFATAL: no option data was retrieved for any ticker. "
              "This usually means outbound network access to Yahoo Finance "
              "(query1/query2.finance.yahoo.com) is blocked in this environment. "
              "Run this script somewhere with internet access.")
        sys.exit(1)

    underlying_df.to_csv(f"{config.DATA_DIR}/underlying_raw.csv")
    options_df.to_csv(f"{config.DATA_DIR}/options_raw.csv", index=False)

    # ----------------------------------------------------------------
    section("2. PREPROCESSING & CLASSIFICATION")
    # ----------------------------------------------------------------
    df = preprocessing.build_analysis_dataset(options_df, underlying_df, yields_df)
    if df.empty:
        print("FATAL: no contracts survived liquidity filtering. Try a different "
              "ticker universe or relax config.DEFAULT_LIQUIDITY_FILTER.")
        sys.exit(1)
    df.to_parquet(f"{config.DATA_DIR}/cleaned_options.parquet")
    print(f"Final analysis dataset: {len(df):,} contracts across {df['ticker'].nunique()} tickers.")

    # ----------------------------------------------------------------
    section("3. IMPLIED VOLATILITY")
    # ----------------------------------------------------------------
    df = implied_vol.compute_implied_vols(df, price_col="mid_price")

    iv_surface_grid = implied_vol.build_iv_surface_grid(df)
    iv_surface_grid.to_csv(f"{config.RESULTS_DIR}/iv_surface_grid.csv", index=False)

    smile_table = implied_vol.smile_by_maturity_bucket(df)
    smile_table.to_csv(f"{config.RESULTS_DIR}/iv_smile_by_maturity.csv", index=False)

    term_struct = implied_vol.term_structure(df)
    term_struct.to_csv(f"{config.RESULTS_DIR}/iv_term_structure.csv", index=False)

    # ----------------------------------------------------------------
    section("4. MODEL PRICING (BS / CRR / Monte Carlo)")
    # ----------------------------------------------------------------
    print("Pricing every contract under historical and implied volatility "
          "(this is the most computationally expensive step -- CRR/MC are "
          "priced row-by-row)...")
    df = evaluation.add_model_prices(df, hist_vol_col="realized_vol_60d", iv_col="market_iv")
    df = greeks.add_bs_greeks(df, vol_col="realized_vol_60d", prefix="bs_hist_greeks")
    df = greeks.add_bs_greeks(df, vol_col="market_iv", prefix="bs_iv_greeks")
    df = evaluation.add_pricing_errors(df)
    df.to_parquet(f"{config.DATA_DIR}/priced_options.parquet")

    # ----------------------------------------------------------------
    section("5. MODEL PERFORMANCE TABLES")
    # ----------------------------------------------------------------
    overall_table = evaluation.model_performance_table(df)
    overall_table.to_csv(f"{config.RESULTS_DIR}/table2_model_performance.csv", index=False)
    print(overall_table.to_string(index=False))

    by_ticker = evaluation.model_performance_table(df, groupby=["ticker"])
    by_ticker.to_csv(f"{config.RESULTS_DIR}/table1_performance_by_ticker.csv", index=False)

    by_moneyness = evaluation.model_performance_table(df, groupby=["moneyness_bucket"])
    by_moneyness.to_csv(f"{config.RESULTS_DIR}/table3_performance_by_moneyness.csv", index=False)

    by_maturity = evaluation.model_performance_table(df, groupby=["maturity_bucket"])
    by_maturity.to_csv(f"{config.RESULTS_DIR}/table4_performance_by_maturity.csv", index=False)

    by_vol_regime = evaluation.model_performance_table(df, groupby=["vol_regime"])
    by_vol_regime.to_csv(f"{config.RESULTS_DIR}/performance_by_vol_regime.csv", index=False)

    by_option_type = evaluation.model_performance_table(df, groupby=["option_type"])
    by_option_type.to_csv(f"{config.RESULTS_DIR}/performance_by_option_type.csv", index=False)

    contracts_by_ticker = df.groupby("ticker").size().reset_index(name="n_contracts")
    contracts_by_ticker.to_csv(f"{config.RESULTS_DIR}/table1_contracts_by_ticker.csv", index=False)

    # ----------------------------------------------------------------
    section("6. STATISTICAL SIGNIFICANCE TESTS")
    # ----------------------------------------------------------------
    pairs = [
        ("BlackScholes_IV", "CRR_European_IV"),
        ("BlackScholes_IV", "MonteCarlo_IV"),
        ("CRR_European_IV", "MonteCarlo_IV"),
        ("BlackScholes_Hist", "BlackScholes_IV"),
    ]
    stat_tests = evaluation.run_all_pairwise_tests(df, pairs)
    stat_tests.to_csv(f"{config.RESULTS_DIR}/pairwise_statistical_tests.csv", index=False)
    for _, row in stat_tests.iterrows():
        print(f"\n{row['Model_A']} vs {row['Model_B']}: {row.get('interpretation', 'N/A')}")

    # ----------------------------------------------------------------
    section("7. CRR / MONTE CARLO CONVERGENCE STUDY")
    # ----------------------------------------------------------------
    convergence_pool = df.dropna(
        subset=["spot_price", "strike", "T_years", "risk_free_rate", "market_iv"]
    )
    sample_for_convergence = convergence_pool.sample(
        n=min(args.sample_contracts, len(convergence_pool)), random_state=42
    )
    print(f"Convergence sample: {len(sample_for_convergence):,} contracts "
          f"from {len(convergence_pool):,} contracts with valid market IV.")

    crr_conv_rows, mc_conv_rows = [], []
    for _, row in sample_for_convergence.iterrows():
        sigma = row["market_iv"] if pd.notna(row["market_iv"]) else row["realized_vol_60d"]
        if pd.isna(sigma) or sigma <= 0:
            continue
        crr_conv = bn.crr_convergence(row["spot_price"], row["strike"], row["T_years"],
                                       row["risk_free_rate"], sigma, option_type=row["option_type"])
        mc_conv = mc.mc_convergence(row["spot_price"], row["strike"], row["T_years"],
                                     row["risk_free_rate"], sigma, option_type=row["option_type"])
        bs_ref = bs.bs_price(row["spot_price"], row["strike"], row["T_years"],
                              row["risk_free_rate"], sigma, row["option_type"])
        for N, res in crr_conv.items():
            crr_conv_rows.append({"contract_symbol": row.get("contract_symbol"), "N": N,
                                   "price": res["price"], "bs_reference": bs_ref})
        for n, res in mc_conv.items():
            mc_conv_rows.append({"contract_symbol": row.get("contract_symbol"), "n_paths": n,
                                  "price": res["price"], "std_error": res["std_error"],
                                  "bs_reference": bs_ref})

    crr_conv_df = pd.DataFrame(crr_conv_rows)
    mc_conv_df = pd.DataFrame(mc_conv_rows)
    crr_conv_df.to_csv(f"{config.RESULTS_DIR}/crr_convergence.csv", index=False)
    mc_conv_df.to_csv(f"{config.RESULTS_DIR}/mc_convergence.csv", index=False)
    print(f"CRR convergence rows written: {len(crr_conv_df):,} "
          f"({config.RESULTS_DIR}/crr_convergence.csv)")
    print(f"Monte Carlo convergence rows written: {len(mc_conv_df):,} "
          f"({config.RESULTS_DIR}/mc_convergence.csv)")

    if not crr_conv_df.empty:
        crr_summary = crr_conv_df.assign(
            abs_error=lambda x: (x["price"] - x["bs_reference"]).abs()
        ).groupby("N")["abs_error"].mean().reset_index(name="mean_abs_error_vs_bs")
        print("\nCRR mean absolute convergence error vs Black-Scholes:")
        print(crr_summary.to_string(index=False))

    if not mc_conv_df.empty:
        mc_summary = mc_conv_df.assign(
            abs_error=lambda x: (x["price"] - x["bs_reference"]).abs()
        ).groupby("n_paths").agg(
            mean_abs_error_vs_bs=("abs_error", "mean"),
            mean_std_error=("std_error", "mean"),
        ).reset_index()
        print("\nMonte Carlo mean convergence error / standard error vs Black-Scholes:")
        print(mc_summary.to_string(index=False))

    if not sample_for_convergence.empty:
        example = sample_for_convergence.iloc[0]
        example_sigma = example["market_iv"] if pd.notna(example["market_iv"]) else example["realized_vol_60d"]
        crr_example = bn.crr_convergence(example["spot_price"], example["strike"], example["T_years"],
                                          example["risk_free_rate"], example_sigma, option_type=example["option_type"])
        mc_example = mc.mc_convergence(example["spot_price"], example["strike"], example["T_years"],
                                        example["risk_free_rate"], example_sigma, option_type=example["option_type"])
        bs_ref_example = bs.bs_price(example["spot_price"], example["strike"], example["T_years"],
                                      example["risk_free_rate"], example_sigma, example["option_type"])
        crr_fig = viz.fig_crr_convergence(crr_example, bs_ref_example, "fig6_crr_convergence")
        mc_fig = viz.fig_mc_convergence(mc_example, bs_ref_example, "fig7_mc_convergence")
        print(f"Example convergence figures written: {crr_fig}, {mc_fig}")

    # ----------------------------------------------------------------
    section("8. DELTA-HEDGING BACKTEST")
    # ----------------------------------------------------------------
    log_returns_by_ticker = {
        t: underlying_df.loc[underlying_df["ticker"] == t, "log_return"].dropna().to_numpy()
        for t in tickers
    }

    hedge_sample = df.dropna(
        subset=["spot_price", "strike", "T_years", "risk_free_rate", "market_iv"]
    ).copy()
    hedge_sample = hedge_sample[hedge_sample["T_years"] <= 1.0]  # keep tractable path lengths
    hedge_sample["hedge_vol"] = hedge_sample["market_iv"].fillna(hedge_sample["realized_vol_60d"])
    hedge_sample = hedge_sample.sample(n=min(args.hedge_sample, len(hedge_sample)), random_state=42)

    hedge_results = hedging.batch_hedge_backtest(
        hedge_sample, log_returns_by_ticker,
        models=["bs", "crr"], rebalance_freqs=["daily", "weekly"],
        apply_txn_costs=[False, True], n_paths_per_contract=5, seed=42,
    )
    hedge_results.to_csv(f"{config.RESULTS_DIR}/hedge_backtest_raw.csv", index=False)

    if not hedge_results.empty:
        hedge_summary = hedging.hedge_performance_summary(hedge_results)
        hedge_summary.to_csv(f"{config.RESULTS_DIR}/table5_hedging_performance.csv", index=False)
        print(hedge_summary.to_string(index=False))

        txn_cost_compare = hedging.hedge_performance_summary(
            hedge_results, groupby=["model", "txn_costs_applied"]
        )
        txn_cost_compare.to_csv(f"{config.RESULTS_DIR}/table6_txn_cost_comparison.csv", index=False)

        by_bucket = hedging.hedge_performance_summary(
            hedge_results, groupby=["model", "moneyness_bucket", "maturity_bucket"]
        )
        by_bucket.to_csv(f"{config.RESULTS_DIR}/hedging_by_moneyness_maturity.csv", index=False)
    else:
        print("WARNING: hedge backtest produced no results (insufficient return history?).")

    # ----------------------------------------------------------------
    section("9. STRESS TESTING")
    # ----------------------------------------------------------------
    stress_df = stress_testing.stress_test_sample(df, n_sample=20, seed=42)
    stress_df.to_csv(f"{config.RESULTS_DIR}/stress_test_results.csv", index=False)

    run_reporting_sections(df, iv_surface_grid, hedge_results,
                           overall_table, by_moneyness, by_maturity, stat_tests)


def run_reporting_sections(df, iv_surface_grid, hedge_results,
                           overall_table, by_moneyness, by_maturity, stat_tests):
    # ----------------------------------------------------------------
    section("10. FIGURES")
    # ----------------------------------------------------------------
    figs = {}
    figs["model_vs_market_bs"] = viz.fig_model_vs_market_scatter(df, "bs_price_iv", "fig1_bs_model_vs_market")
    figs["error_dist"] = viz.fig_error_distribution(
        df, ["BlackScholes_IV_pct_err_mid", "CRR_European_IV_pct_err_mid", "MonteCarlo_IV_pct_err_mid"],
        "fig2_error_distribution")
    figs["error_by_moneyness"] = viz.fig_error_by_category(
        df, "moneyness_bucket", "BlackScholes_IV_pct_err_mid", "fig3_error_by_moneyness",
        order=[b[0] for b in config.MONEYNESS_BUCKETS])
    figs["error_by_maturity"] = viz.fig_error_by_category(
        df, "maturity_bucket", "BlackScholes_IV_pct_err_mid", "fig4_error_by_maturity",
        order=[b[0] for b in config.MATURITY_BUCKETS])
    figs["error_by_vol_regime"] = viz.fig_error_by_category(
        df, "vol_regime", "BlackScholes_IV_pct_err_mid", "fig5_error_by_vol_regime",
        order=list(config.VOL_REGIME_LABELS))

    if not df["market_iv"].dropna().empty:
        most_common_bucket = df["maturity_bucket"].value_counts().idxmax()
        figs["iv_smile"] = viz.fig_iv_smile(df, most_common_bucket, "fig8_iv_smile")
        figs["iv_surface"] = viz.fig_iv_surface(iv_surface_grid, "fig9_iv_surface")

    figs["delta_comparison"] = viz.fig_delta_comparison(
        df, ["bs_delta_iv", "crr_eu_delta_iv"], "fig10_delta_comparison")

    if not hedge_results.empty:
        figs["hedge_pnl_dist"] = viz.fig_hedge_pnl_distribution(hedge_results, "fig11_hedge_pnl_distribution")

    print("Figures written to:", config.FIGURES_DIR)
    for k, v in figs.items():
        print(f"  {k}: {v}")

    # ----------------------------------------------------------------
    section("11. VALIDATION CHECKS")
    # ----------------------------------------------------------------
    warnings_list = []

    # IV round-trip sanity check
    if not df.empty:
        check_row = df.dropna(subset=["market_iv", "mid_price"]).head(1)
        if not check_row.empty:
            r0 = check_row.iloc[0]
            reconstructed = bs.bs_price(r0["spot_price"], r0["strike"], r0["T_years"],
                                         r0["risk_free_rate"], r0["market_iv"], r0["option_type"])
            diff = abs(reconstructed - r0["mid_price"])
            print(f"IV round-trip check: reconstructed price {reconstructed:.4f} vs "
                  f"market mid {r0['mid_price']:.4f} (diff={diff:.6f})")
            if diff > 0.01:
                warnings_list.append("IV round-trip reconstruction error exceeds $0.01 tolerance.")

    n_nan_price_cols = df[[c for c in evaluation.MODEL_PRICE_COLUMNS.values() if c in df.columns]].isna().sum().sum()
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaNs across model price columns: {n_nan_price_cols}")
    print(f"Inf values across numeric columns: {n_inf}")
    if n_inf > 0:
        warnings_list.append(f"{n_inf} infinite values detected in numeric columns.")

    if df["ticker"].nunique() < 3:
        warnings_list.append("Fewer than 3 tickers had usable data -- results may not generalize.")

    for w in warnings_list:
        print(f"WARNING: {w}")

    # ----------------------------------------------------------------
    section("12. RESEARCH SUMMARY")
    # ----------------------------------------------------------------
    write_research_summary(df, overall_table, by_moneyness, by_maturity,
                            hedge_results if not hedge_results.empty else pd.DataFrame(),
                            stat_tests, warnings_list)
    print(f"\nDone. See {config.RESULTS_DIR}/research_summary.md for the full write-up.")


def write_research_summary(df, overall_table, by_moneyness, by_maturity,
                            hedge_results, stat_tests, warnings_list):
    n_tickers = df["ticker"].nunique()
    n_contracts = len(df)
    date_min = pd.to_datetime(df["snapshot_time"]).min() if "snapshot_time" in df.columns else "N/A"
    date_max = pd.to_datetime(df["snapshot_time"]).max() if "snapshot_time" in df.columns else "N/A"

    def best_model(table, metric="MAE"):
        t = table.dropna(subset=[metric])
        if t.empty:
            return "N/A", np.nan
        row = t.loc[t[metric].idxmin()]
        return row["Model"], row[metric]

    best_mae_model, best_mae_val = best_model(overall_table, "MAE")
    best_rmse_model, best_rmse_val = best_model(overall_table, "RMSE")

    best_moneyness_model, best_moneyness_val = ("N/A", np.nan)
    if not by_moneyness.empty:
        bm = by_moneyness.dropna(subset=["MAE"])
        if not bm.empty:
            r = bm.loc[bm["MAE"].idxmin()]
            best_moneyness_model = f"{r['Model']} ({r['moneyness_bucket']})"
            best_moneyness_val = r["MAE"]

    best_maturity_model, best_maturity_val = ("N/A", np.nan)
    if not by_maturity.empty:
        bt = by_maturity.dropna(subset=["MAE"])
        if not bt.empty:
            r = bt.loc[bt["MAE"].idxmin()]
            best_maturity_model = f"{r['Model']} ({r['maturity_bucket']})"
            best_maturity_val = r["MAE"]

    hedge_section = "Hedging backtest did not produce results (insufficient data)."
    if not hedge_results.empty:
        hs = hedging.hedge_performance_summary(hedge_results)
        hedge_section = hs.to_markdown(index=False) if hasattr(hs, "to_markdown") else hs.to_string(index=False)

    stats_section = stat_tests[["Model_A", "Model_B", "interpretation"]].to_string(index=False) \
        if not stat_tests.empty else "No statistical tests available."

    lines = [
        "# Option Pricing Model Comparison -- Research Summary",
        "",
        f"- **Tickers analyzed:** {n_tickers}",
        f"- **Option contracts analyzed:** {n_contracts:,}",
        f"- **Data snapshot window:** {date_min} to {date_max}",
        "",
        "## Headline Findings",
        f"- Best model by MAE (mid-price, market-implied vol basis): **{best_mae_model}** (MAE = {best_mae_val:.4f})" if pd.notna(best_mae_val) else "- MAE comparison unavailable.",
        f"- Best model by RMSE: **{best_rmse_model}** (RMSE = {best_rmse_val:.4f})" if pd.notna(best_rmse_val) else "- RMSE comparison unavailable.",
        f"- Best model/moneyness-bucket combination: **{best_moneyness_model}** (MAE = {best_moneyness_val:.4f})" if pd.notna(best_moneyness_val) else "- Moneyness breakdown unavailable.",
        f"- Best model/maturity-bucket combination: **{best_maturity_model}** (MAE = {best_maturity_val:.4f})" if pd.notna(best_maturity_val) else "- Maturity breakdown unavailable.",
        "",
        "## Model Performance (Overall)",
        overall_table.to_string(index=False),
        "",
        "## Statistical Significance of Model Differences",
        stats_section,
        "",
        "## Delta-Hedging Backtest",
        "Note: hedging paths are generated via historical block-bootstrap of each "
        "ticker's own realized daily log-returns (see hedging.py docstring) since "
        "yfinance does not expose historical option-chain data for currently-listed, "
        "not-yet-expired contracts.",
        "",
        hedge_section,
        "",
        "## Data & Methodology Caveats",
        "- Market reference price is the bid-ask **mid-price**; last-trade price is "
        "reported separately since it can be stale relative to the current quote.",
        "- Implied volatility is inverted from Black-Scholes against the mid-price via "
        "Brent's method; inversion fails (and is excluded, not imputed) for quotes "
        "violating no-arbitrage bounds or where a root cannot be bracketed.",
        "- CRR and Monte Carlo are priced at "
        f"N={config.CRR_DEFAULT_STEPS} steps / {config.MC_DEFAULT_PATHS:,} paths for the "
        "main comparison; full convergence studies (N/paths grids) are run separately "
        "on a contract sample -- see crr_convergence.csv / mc_convergence.csv.",
        "- The risk-free rate is interpolated between ^IRX (short end) and ^TNX (long "
        "end); this is an approximation of the true Treasury curve, not an exact fit.",
        "- Dividends are not modeled unless a continuous yield is supplied (q=0 by "
        "default in this run) -- this can bias American-call early-exercise premia "
        "and put-call parity checks for dividend-paying names.",
        "- Liquidity filtering removes contracts with zero/crossed quotes, insufficient "
        "open interest, or extreme relative bid-ask spreads; see config.LiquidityFilter.",
        "- The delta-hedging backtest replays *bootstrapped historical* return paths, "
        "not the option's actual realized future -- true historical option-chain "
        "backtesting is not possible from yfinance's current-snapshot-only data.",
        "- A lower raw pricing error does not by itself prove a model is economically "
        "better once bid-ask spread and statistical significance are considered -- see "
        "the paired significance tests above and the bid-ask-spread-relative error "
        "columns in the cleaned dataset.",
        "",
        "## Data Quality Warnings",
    ]
    if warnings_list:
        lines += [f"- {w}" for w in warnings_list]
    else:
        lines.append("- None detected.")

    lines += [
        "",
        "## Reproducibility",
        "All numbers in this document are computed directly from the cleaned dataset "
        "in `data/cleaned_options.parquet` / `data/priced_options.parquet` and the "
        "CSV tables in `results/`. No figures in this summary are hand-entered.",
    ]

    out_path = f"{config.RESULTS_DIR}/research_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(str(l) for l in lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
