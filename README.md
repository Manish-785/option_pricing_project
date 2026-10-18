# Option Pricing Model Comparison — Research Project

A research-quality empirical comparison of Black-Scholes-Merton, CRR Binomial,
and Monte Carlo option pricing models against real market data pulled live
from Yahoo Finance, including implied-volatility surface construction and a
historical delta-hedging backtest.

## ⚠️ Before you run this

This pipeline needs **outbound internet access to Yahoo Finance**
(`query1.finance.yahoo.com`, `query2.finance.yahoo.com`). If you're running
in a sandboxed environment (locked-down CI, certain notebook services,
firewalled corporate networks) that blocks these hosts, `run_analysis.py`
will exit with a clear error at the data-collection step. Run it instead on
a machine / environment with normal internet access.

This codebase was built and validated (module correctness, convergence
behavior, pipeline wiring) against **synthetic data** in a network-restricted
environment — see `test_pipeline_synthetic.py` for that validation harness
(not part of the research pipeline itself, and it does not talk to Yahoo
Finance). You still need to run `run_analysis.py` yourself, with internet
access, to get real findings.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py
```

Optional flags:
```bash
python run_analysis.py --tickers AAPL MSFT NVDA SPY
python run_analysis.py --no-cache                 # force fresh Yahoo downloads
python run_analysis.py --sample-contracts 25       # more CRR/MC convergence samples
python run_analysis.py --hedge-sample 80           # more hedging-backtest contracts
```

Runtime: the option chain download and per-contract CRR/Monte Carlo pricing
loop are the slow parts. Expect several minutes for ~10 tickers with full
option chains; reduce the ticker list or `--hedge-sample` for a faster pass.

## Project structure

```
option_pricing_project/
  data/                  raw + cleaned datasets, Yahoo cache
  results/                CSV tables + research_summary.md
  figures/                PNG figures
  src/
    config.py             all tunable settings (universe, buckets, filters, model sizes)
    data.py                yfinance data pipeline + disk caching
    preprocessing.py       liquidity filtering, moneyness/maturity/vol-regime classification
    black_scholes.py       BSM pricing, all Greeks, Brent's-method IV solver
    binomial.py             CRR tree (European + American), convergence testing
    monte_carlo.py          risk-neutral MC w/ antithetic variates, CI, CRN finite-diff delta
    greeks.py               vectorised BS Greek application across the dataset
    implied_vol.py          per-contract IV, IV surface grid, smile/term-structure tables
    evaluation.py            model pricing application, error metrics, paired significance tests
    hedging.py               historical block-bootstrap delta-hedging backtest + txn costs
    stress_testing.py        vol / rate / spot shock scenarios
    visualization.py         all required figures
  run_analysis.py           orchestrates the full pipeline end to end
  test_pipeline_synthetic.py  dev-only correctness harness (synthetic data, no network)
  requirements.txt
```

## Methodology notes & caveats (read before quoting any numbers)

- **Market price reference**: mid-price `(bid+ask)/2` is the primary
  reference; last-trade price is tracked separately since it can be stale.
- **Implied volatility**: inverted from Black-Scholes against the mid-price
  via Brent's method. Inversions that violate no-arbitrage bounds or can't be
  bracketed are left as `NaN` and excluded — never imputed or guessed.
- **Historical vs. implied vol pricing are reported separately** everywhere;
  don't conflate them when citing "model accuracy."
- **CRR/MC pricing scale**: the main comparison prices every contract at
  `config.CRR_DEFAULT_STEPS` / `config.MC_DEFAULT_PATHS` for tractability;
  full convergence grids (`CRR_CONVERGENCE_STEPS`, `MC_CONVERGENCE_PATHS`)
  are run separately on a contract sample (`crr_convergence.csv`,
  `mc_convergence.csv`), not embedded in every row.
- **Risk-free rate** is linearly interpolated between `^IRX` (short end) and
  `^TNX` (long end) as a function of time-to-maturity — a simplification of
  the true Treasury curve, not an exact fit (see `rates.py` docstring).
- **Dividends**: not modeled unless you pass a continuous yield `q`; this can
  bias American early-exercise premia and put-call parity checks for
  dividend payers.
- **Liquidity filtering** removes zero/crossed quotes, thin open interest,
  and extreme relative spreads (`config.LiquidityFilter`) — tune it for your
  universe.
- **Delta-hedging backtest data limitation**: `yfinance` only exposes the
  *current* option chain, not historical option time series, so no data
  source can give you the true realized path of a live, not-yet-expired
  listed contract. The backtest instead replays a **historical block-bootstrap**
  of the underlying's own real daily log-returns (a contiguous block sampled
  from actual price history, replayed forward) — real historical return
  dynamics, but a simulated forward path, not the option's literal future.
  This is stated explicitly in `hedging.py` and in every generated summary.
- **Statistical significance**: model comparisons include paired t-tests and
  Wilcoxon signed-rank tests, not just point-estimate MAE/RMSE — a lower raw
  error is not proof of superiority without checking significance and
  relative bid-ask spread.
- **Timestamp mismatch**: options, underlying price, and Treasury yields are
  fetched close together but not atomically; do not treat this as a
  simultaneous cross-sectional snapshot to the second.
- **American vs. European exercise**: CRR is run in both modes; the American
  premium over European is the early-exercise value, which Black-Scholes
  (European-only) cannot capture.

## Resume-claim guidance

`run_analysis.py` writes every number it reports to `results/*.csv` and
`results/research_summary.md` directly from the computed dataset — nothing
is hand-entered. Pull your resume bullets from those files after a real run,
e.g.:

> "Compared pricing accuracy of Black-Scholes, CRR Binomial, and Monte Carlo
> models across N option contracts spanning M equities/ETFs using live
> Yahoo Finance data, finding [best model] achieved X% lower MAE than
> [baseline] (statistically significant, paired t-test p < 0.05)."

> "Constructed implied volatility surfaces across N liquid contracts,
> quantifying volatility skew of X vol points between [bucket] and [bucket]."

> "Backtested BS- and CRR-delta hedging strategies across N contracts under
> daily vs. weekly rebalancing, finding daily rebalancing reduced hedging
> P&L standard deviation by X% before transaction costs and Y% after."

Only use figures that actually appear in your own `results/` output after a
live run — the numbers above are illustrative placeholders, not real results.
