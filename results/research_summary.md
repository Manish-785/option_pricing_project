# Option Pricing Model Comparison -- Research Summary

- **Tickers analyzed:** 10
- **Option contracts analyzed:** 36,925
- **Data snapshot window:** 2026-08-15 04:41:43.296222 to 2026-08-15 04:42:27.455194

## Headline Findings
- Best model by MAE (mid-price, market-implied vol basis): **CRR_European_IV** (MAE = 0.0162)
- Best model by RMSE: **CRR_European_IV** (RMSE = 0.0278)
- Best model/moneyness-bucket combination: **BlackScholes_IV (OTM)** (MAE = 0.0000)
- Best model/maturity-bucket combination: **CRR_European_IV (0-30D)** (MAE = 0.0031)

## Model Performance (Overall)
            Model  n_obs      MAE      RMSE      MAPE  MedianAPE  MeanError  Pct_within_5  Pct_within_10  Pct_within_20
BlackScholes_Hist  36925 9.036396 50.389137 47.060331  16.577465   1.752651     33.770639      42.331735      52.670087
  BlackScholes_IV  36925 5.314055 50.456840  2.255602   0.000004   1.151209     97.011746      97.673310      98.536708
CRR_European_Hist  36925 9.038429 50.389725 47.024313  16.595894   1.754151     33.722984      42.390604      52.678496
  CRR_European_IV  32657 0.016212  0.027823  0.351307   0.082396   0.000635     99.920395     100.000000     100.000000
CRR_American_Hist  36925 8.960817 51.094056 47.067100  16.739967   2.726681     33.720181      42.079444      52.628038
  CRR_American_IV  32657 0.627632  2.444955  1.124169   0.383991   0.618597     95.570769      99.621079     100.000000
  MonteCarlo_Hist  36925 9.068712 50.393852 47.496738  16.824601   1.819285     33.585625      42.152328      52.524318
    MonteCarlo_IV  32657 0.323620 10.517551  0.758831   0.348701   0.323294     99.232606      99.681579      99.914026

## Statistical Significance of Model Differences
          Model_A         Model_B                                                                                                                                                                                                         interpretation
  BlackScholes_IV CRR_European_IV       BlackScholes_IV has lower mean absolute pricing error than CRR_European_IV by 0.0162 on average across 32657 paired contracts. This difference is statistically significant at the 5% level (paired t-test p=0).
  BlackScholes_IV   MonteCarlo_IV BlackScholes_IV has lower mean absolute pricing error than MonteCarlo_IV by 0.3236 on average across 32657 paired contracts. This difference is statistically significant at the 5% level (paired t-test p=2.673e-08).
  CRR_European_IV   MonteCarlo_IV CRR_European_IV has lower mean absolute pricing error than MonteCarlo_IV by 0.3074 on average across 32657 paired contracts. This difference is statistically significant at the 5% level (paired t-test p=1.268e-07).
BlackScholes_Hist BlackScholes_IV    BlackScholes_Hist has higher mean absolute pricing error than BlackScholes_IV by 3.7223 on average across 36925 paired contracts. This difference is statistically significant at the 5% level (paired t-test p=0).

## Delta-Hedging Backtest
Note: hedging paths are generated via historical block-bootstrap of each ticker's own realized daily log-returns (see hedging.py docstring) since yfinance does not expose historical option-chain data for currently-listed, not-yet-expired contracts.

| model   | rebalance_freq   | txn_costs_applied   |   n_simulations |   mean_pnl |   std_pnl |   rmse_pnl |   mean_abs_pnl |   max_loss |   mean_txn_costs |
|:--------|:-----------------|:--------------------|----------------:|-----------:|----------:|-----------:|---------------:|-----------:|-----------------:|
| bs      | daily            | False               |             200 |  -0.585682 |   6.52851 |    6.53846 |        3.31116 |   -39.7108 |          0       |
| bs      | daily            | True                |             200 |  -8.06763  |  11.6722  |   14.165   |        8.1324  |   -70.0337 |          7.41693 |
| bs      | weekly           | False               |             200 |   0.134945 |   7.33121 |    7.31411 |        3.75352 |   -35.834  |          0       |
| bs      | weekly           | True                |             200 |  -4.53798  |   8.49078 |    9.60865 |        5.23539 |   -46.6748 |          4.63443 |
| crr     | daily            | False               |             200 |  -0.588289 |   6.52294 |    6.53315 |        3.30594 |   -39.7406 |          0       |
| crr     | daily            | True                |             200 |  -8.07027  |  11.6708  |   14.1653  |        8.1345  |   -70.0786 |          7.41696 |
| crr     | weekly           | False               |             200 |   0.132012 |   7.32491 |    7.30777 |        3.74798 |   -35.8611 |          0       |
| crr     | weekly           | True                |             200 |  -4.54028  |   8.48716 |    9.60656 |        5.23324 |   -46.6734 |          4.6338  |

## Data & Methodology Caveats
- Market reference price is the bid-ask **mid-price**; last-trade price is reported separately since it can be stale relative to the current quote.
- Implied volatility is inverted from Black-Scholes against the mid-price via Brent's method; inversion fails (and is excluded, not imputed) for quotes violating no-arbitrage bounds or where a root cannot be bracketed.
- CRR and Monte Carlo are priced at N=200 steps / 100,000 paths for the main comparison; full convergence studies (N/paths grids) are run separately on a contract sample -- see crr_convergence.csv / mc_convergence.csv.
- The risk-free rate is interpolated between ^IRX (short end) and ^TNX (long end); this is an approximation of the true Treasury curve, not an exact fit.
- Dividends are not modeled unless a continuous yield is supplied (q=0 by default in this run) -- this can bias American-call early-exercise premia and put-call parity checks for dividend-paying names.
- Liquidity filtering removes contracts with zero/crossed quotes, insufficient open interest, or extreme relative bid-ask spreads; see config.LiquidityFilter.
- The delta-hedging backtest replays *bootstrapped historical* return paths, not the option's actual realized future -- true historical option-chain backtesting is not possible from yfinance's current-snapshot-only data.
- A lower raw pricing error does not by itself prove a model is economically better once bid-ask spread and statistical significance are considered -- see the paired significance tests above and the bid-ask-spread-relative error columns in the cleaned dataset.

## Data Quality Warnings
- None detected.

## Reproducibility
All numbers in this document are computed directly from the cleaned dataset in `data/cleaned_options.parquet` / `data/priced_options.parquet` and the CSV tables in `results/`. No figures in this summary are hand-entered.