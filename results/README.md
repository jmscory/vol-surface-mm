# Results Reviewer Summary

Artifacts in this directory let a reviewer inspect the live 252-step backtest and the refreshed 96-row live parameter sweep without running the project. The sweep now scores parameter tuples across `gbm`, `bates`, `rough`, and `real_aapl` instead of picking a single best row on `var_capture`. Run metadata for the backtest artifact: seed=42, steps=252, backend=sabr.

The headline backtest now uses the new `seeding_mode="flat"` default plus the gamma-protection wing-buying trigger.

## Per-Feed Survival

| Feed | Survived | Total PnL | Var Capture | Max Drawdown |
|------|----------|----------:|------------:|-------------:|
| GBM | No | -187.89 | -0.0406 | 251.06 |
| Bates | Yes | +53.73 | -1.0874 | 166.86 |
| Rough | Yes | +41.45 | -0.6135 | 54.51 |

Two of the three feeds (Bates and Rough) now satisfy the live-path survival rule.  GBM still finishes negative because the realized variance under pure GBM is too low to pay for the protection wings.  For comparison, the legacy seeded-short-straddle baseline run was `gbm=-159.66, bates=-195.40, rough=+28.71` with only Rough surviving.

## What `var_capture` Means

`var_capture` compares the realized gamma/theta extraction of the book with the variance opportunity implied by the path, so it measures whether realized moves actually paid for hedging and inventory carry. Values above 0 mean the strategy monetized realized variance net of its hedging burden, while values below 0 mean the path failed to cover those costs.

## Sweep Selection

The live sweep separates tuning feasibility from the stricter artifact survival flag. `tuning_survived=True` means the run stayed within the drawdown budget and kept a positive bankroll; `survived=True` still requires positive terminal P&L as well. Under that split, all 96 raw rows are tuning-feasible and 32 of 96 satisfy the stricter survival rule. The raw per-feed rows are written to [param_sweep.csv](param_sweep.csv), and the grouped cross-feed tuple summaries are written to [param_sweep_grouped.csv](param_sweep_grouped.csv) and [param_sweep_stdout.txt](param_sweep_stdout.txt).

The canonical ranking now groups rows by `(kelly_fraction, gamma_target, tail_hedge_trigger, stress_guard_multiple)` and screens for `tuning_survived=True` on every feed. It then ranks surviving tuples by worst `total_pnl`, `profitable_count`, median `total_pnl`, worst `max_drawdown`, and median `var_capture`. This is intentionally deployability-first: the selected tuple should be the least fragile bundle across the feed set, not the prettiest single-row `var_capture` outlier.

Best cross-feed tuple from [param_sweep_grouped.csv](param_sweep_grouped.csv): `kelly_fraction=0.15`, `gamma_target=0.010`, `tail_hedge_trigger=0.20`, `stress_guard_multiple=2.00`, `worst_total_pnl=-154.73`, `profitable_count=1/4`, `median_total_pnl=-73.19`, `worst_max_drawdown=288.04`, `median_var_capture=-0.8264`. Feed-level PnL for that tuple is `gbm=-154.73`, `bates=-125.99`, `rough=43.19`, `real_aapl=-20.38`. The same outcome ties across `tail_hedge_trigger=0.20/0.30` and `stress_guard_multiple=2.0/2.5` at `gamma_target=0.010`.

That result is better on worst-case PnL than the old single-row winner, but it is not a clean deployability result: no tuple achieved profitability on every feed. Because of that, the codebase keeps the current conservative defaults instead of automatically promoting the least-negative grouped winner into [vol_surface_mm/config.py](../src/vol_surface_mm/config.py).

For auditability, the legacy single-row ranking is still emitted in [param_sweep_stdout.txt](param_sweep_stdout.txt). Its best row remains the GBM row at `kelly_fraction=0.05`, `gamma_target=0.005`, `tail_hedge_trigger=0.20`, `stress_guard_multiple=2.00`, but that row is no longer the selection rule used for live-path tuning.

The focused grid still includes `feed=real_aapl`, a free public AAPL OHLC research example sourced from Plotly's `finance-charts-apple.csv`. That real-history feed remains tuning-feasible across the 252-step sweep, but it does not produce an all-feed-profitable tuple in this run.

## Surface Fitting (SABR Backend)

The default `sabr` backend fits the Hagan–Lesniewski–Lewis (2002) SABR model to each snapshot's implied vol surface. The model specifies a stochastic-alpha, stochastic-beta local volatility smile under the Heston-like approximation:

$$A(T) = \left(\frac{(1-\beta)^2}{24}\frac{\alpha^2}{(FK)^{1-\beta}} + \frac{1}{4}\frac{\rho\beta\nu\alpha}{(FK)^{(1-\beta)/2}} + \frac{2-3\rho^2}{24}\nu^2\right)T$$

Where $F$ is the forward, $K$ is the strike, $\alpha$ is the ATM volatility, $\beta$ is the CEV exponent, $\rho$ is the spot–vol correlation, and $\nu$ is the vol-of-vol. ATM handling uses the standard $\lim_{x \to 0} \frac{\log(F/K)}{x}$ approximation to avoid singularities in code.

## Limitations

- The data are fully synthetic, so there is no real microstructure, queue position, venue behavior, or regime labeling in these results.
- SABR is structurally mis-specified in rough-vol regimes, so a smooth parametric fit can hide local surface pathologies.
- The fill simulator is synthetic and is not calibrated to real order flow or toxicity metrics.
- Transaction costs are parametric and are not fitted to an empirical venue-specific cost curve.

How to reproduce: `vol-surface-mm artifacts` for the final artifact set, and either `vol-surface-mm sweep` (single-run sweep) or the shard workflow below for the grouped sweep.

Shard workflow (recommended for reliable long runs):

    vol-surface-mm sweep --kelly-shard 0.05
    vol-surface-mm sweep --kelly-shard 0.10
    vol-surface-mm sweep --kelly-shard 0.15
    vol-surface-mm sweep --merge-only

To regenerate the per-feed P&L heatmap (`docs/assets/sweep_summary.png`):

    pip install -e ".[plot]"
    vol-surface-mm plot-sweep
