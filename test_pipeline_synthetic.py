"""
test_pipeline_synthetic.py
===========================
Dev/validation harness -- NOT the research pipeline itself (run_analysis.py
is). This exercises preprocessing -> IV inversion -> model pricing ->
evaluation -> hedging -> stress testing -> visualization end-to-end using
synthetic option data generated FROM Black-Scholes (so ground truth is
known), without requiring any network access.

Run this after `pip install -r requirements.txt` to confirm your
environment is set up correctly before doing a full live run against
Yahoo Finance:

    python test_pipeline_synthetic.py

It should finish with "ALL SYNTHETIC PIPELINE CHECKS PASSED" and leave a
handful of test_fig*.png files in figures/ that you can spot-check, then
delete.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src import config, preprocessing, black_scholes as bs, binomial as bn, monte_carlo as mc
from src import implied_vol, greeks, evaluation, hedging, visualization as viz, stress_testing

rng = np.random.default_rng(7)

# --- synthetic underlying price history ---
tickers = ["SYN1", "SYN2", "SYN3"]
dates = pd.bdate_range("2024-08-01", "2026-08-14")
frames = []
for tkr, s0, vol in zip(tickers, [150.0, 320.0, 95.0], [0.22, 0.35, 0.18]):
    rets = rng.normal(-0.5 * vol**2 / 252, vol / np.sqrt(252), len(dates))
    prices = s0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(index=dates)
    df["adj_close"] = prices
    df["volume"] = rng.integers(1_000_000, 50_000_000, len(dates))
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    for w in config.VOL_WINDOWS:
        df[f"realized_vol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    df["ticker"] = tkr
    df.index.name = "date"
    frames.append(df)
underlying_df = pd.concat(frames)

# --- synthetic option chains (generated FROM Black-Scholes so we know ground truth) ---
opt_rows = []
now = pd.Timestamp.now(tz="UTC").tz_localize(None)
for tkr, s0, vol in zip(tickers, [150.0, 320.0, 95.0], [0.22, 0.35, 0.18]):
    for days in [10, 25, 45, 80, 150, 270, 400]:
        exp = now + pd.Timedelta(days=days)
        T = days / 365.0
        for m in [0.85, 0.92, 0.97, 1.0, 1.03, 1.08, 1.15]:
            K = round(s0 / m, 1)
            true_iv = vol * (1 + 0.15 * (m - 1) ** 2) + rng.normal(0, 0.005)  # fake smile
            true_iv = max(true_iv, 0.05)
            for ot in ["call", "put"]:
                fair = bs.bs_price(s0, K, T, 0.045, true_iv, ot)
                spread = max(fair * 0.04, 0.02)
                bid = max(fair - spread / 2, 0.01)
                ask = fair + spread / 2
                opt_rows.append({
                    "ticker": tkr, "contract_symbol": f"{tkr}{exp.date()}{ot[0].upper()}{K}",
                    "option_type": ot, "expiration": exp, "strike": K,
                    "bid": round(bid, 2), "ask": round(ask, 2), "last_price": round(fair, 2),
                    "volume": int(rng.integers(0, 500)), "open_interest": int(rng.integers(0, 5000)),
                    "yf_implied_vol": true_iv, "spot_price": s0, "snapshot_time": now,
                    "in_the_money": (s0 > K if ot == "call" else s0 < K),
                })
options_df = pd.DataFrame(opt_rows)

yields_df = pd.DataFrame({"short_rate": [0.043], "long_rate": [0.041]}, index=[now])

print(f"Synthetic underlying rows: {len(underlying_df)}, option rows: {len(options_df)}")

# --- run the real pipeline modules ---
df = preprocessing.build_analysis_dataset(options_df, underlying_df, yields_df)
print(f"After cleaning: {len(df)} contracts")
assert len(df) > 100, "sanity: expect a reasonable number of contracts to survive filtering"

df = implied_vol.compute_implied_vols(df, price_col="mid_price")
iv_fail_rate = df["market_iv"].isna().mean()
print(f"IV inversion failure rate: {iv_fail_rate:.2%}")
assert iv_fail_rate < 0.15, "too many IV inversions failing on clean synthetic data"

df = evaluation.add_model_prices(df, hist_vol_col="realized_vol_60d", iv_col="market_iv",
                                  crr_steps=150, mc_paths=20000)
df = greeks.add_bs_greeks(df, vol_col="realized_vol_60d", prefix="bs_hist_greeks")
df = greeks.add_bs_greeks(df, vol_col="market_iv", prefix="bs_iv_greeks")
df = evaluation.add_pricing_errors(df)

overall = evaluation.model_performance_table(df)
print(overall.to_string(index=False))

# Since options were generated FROM Black-Scholes with the recovered IV, BS-IV
# pricing error should be small (bid/ask rounding to the cent is the main
# remaining noise source).
bs_iv_mae = overall.loc[overall["Model"] == "BlackScholes_IV", "MAE"].iloc[0]
print(f"BlackScholes_IV MAE on synthetic (should be small): {bs_iv_mae:.5f}")
assert bs_iv_mae < 0.10, "BS-IV should nearly perfectly reprice contracts generated from BS+IV"

by_moneyness = evaluation.model_performance_table(df, groupby=["moneyness_bucket"])
by_maturity = evaluation.model_performance_table(df, groupby=["maturity_bucket"])

pairs = [("BlackScholes_IV", "CRR_European_IV"), ("BlackScholes_IV", "MonteCarlo_IV")]
stat_tests = evaluation.run_all_pairwise_tests(df, pairs)
print(stat_tests[["Model_A", "Model_B", "interpretation"]].to_string(index=False))

# --- hedging backtest ---
log_returns_by_ticker = {t: underlying_df.loc[underlying_df["ticker"] == t, "log_return"].dropna().to_numpy()
                          for t in tickers}
hedge_sample = df.dropna(subset=["spot_price", "strike", "T_years", "risk_free_rate", "market_iv"]).copy()
hedge_sample = hedge_sample[hedge_sample["T_years"] <= 1.0]
hedge_sample["hedge_vol"] = hedge_sample["market_iv"]
hedge_sample = hedge_sample.sample(n=min(10, len(hedge_sample)), random_state=1)

hedge_results = hedging.batch_hedge_backtest(
    hedge_sample, log_returns_by_ticker, models=["bs", "crr"],
    rebalance_freqs=["daily", "weekly"], apply_txn_costs=[False, True],
    n_paths_per_contract=3, seed=1,
)
print(f"Hedge backtest rows: {len(hedge_results)}")
assert len(hedge_results) > 0
hedge_summary = hedging.hedge_performance_summary(hedge_results)
print(hedge_summary.to_string(index=False))

daily_std = hedge_summary.loc[(hedge_summary.model == "bs") & (hedge_summary.rebalance_freq == "daily") & (hedge_summary.txn_costs_applied == False), "std_pnl"]
weekly_std = hedge_summary.loc[(hedge_summary.model == "bs") & (hedge_summary.rebalance_freq == "weekly") & (hedge_summary.txn_costs_applied == False), "std_pnl"]
print(f"Daily std_pnl: {daily_std.values}, Weekly std_pnl: {weekly_std.values}")

# --- stress testing ---
stress_df = stress_testing.stress_test_sample(df, n_sample=5, seed=1)
print(f"Stress test rows: {len(stress_df)}")
assert not stress_df.empty

# --- visualization smoke test ---
p1 = viz.fig_model_vs_market_scatter(df, "bs_price_iv", "test_fig1")
p2 = viz.fig_error_distribution(df, ["BlackScholes_IV_pct_err_mid", "CRR_European_IV_pct_err_mid"], "test_fig2")
p3 = viz.fig_error_by_category(df, "moneyness_bucket", "BlackScholes_IV_pct_err_mid", "test_fig3")
p4 = viz.fig_hedge_pnl_distribution(hedge_results, "test_fig4")
iv_grid = implied_vol.build_iv_surface_grid(df)
p5 = viz.fig_iv_surface(iv_grid, "test_fig5")
p6 = viz.fig_iv_smile(df, df["maturity_bucket"].value_counts().idxmax(), "test_fig6")
print("Figures written:", p1, p2, p3, p4, p5, p6)

print("\nALL SYNTHETIC PIPELINE CHECKS PASSED")
