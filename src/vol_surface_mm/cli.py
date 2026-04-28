"""CLI entry point for the vol-surface-mm simulator.

Commands
--------
run       live simulation with real-time Rich dashboard
backtest  silent full run; prints final AFL-style report; exit 0 if VRP > 0
stress    stress-test a single snapshot, print and exit
surface   fit and display vol surface for a single snapshot, exit
config    print effective simulation config as JSON
artifacts generate deterministic results artifacts
sweep     run grouped parameter sweep or merge sweep shards
plot-sweep render grouped sweep heatmap image

All commands share common options; see ``--help`` for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface_mm.config import DEFAULT_CONFIG, SimConfig
from vol_surface_mm.core.hedging import PortfolioManager
from vol_surface_mm.core.pricing import price_chain
from vol_surface_mm.core.quoting import QuotingEngine
from vol_surface_mm.core.stress import run_all_scenarios
from vol_surface_mm.core.surface import build_surface
from vol_surface_mm.data.feed import SyntheticFeed
from vol_surface_mm.data.options_chain import build_chain
from vol_surface_mm.diagnostics import display
from vol_surface_mm.diagnostics.metrics import (
    ResidualAttributionTracker,
    SurfaceMonitor,
    VarianceTracker,
    build_diagnostics_report,
    log_runtime_sanity_checks,
)
from vol_surface_mm.scripts import generate_artifacts, param_sweep, plot_sweep

# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_cfg(args: argparse.Namespace) -> SimConfig:
    """Merge CLI overrides into DEFAULT_CONFIG, returning a new SimConfig."""
    cfg = DEFAULT_CONFIG
    market = replace(
        cfg.market,
        s0=args.spot,
        sigma=args.vol,
        r=args.rate,
        q=args.div,
        dt=args.dt,
    )
    quoting = replace(
        cfg.quoting,
        target_net_gamma=args.gamma_target,
        kelly_fraction=args.kelly_fraction,
    )
    hedging = replace(
        cfg.hedging,
        dt=args.dt,
        discrete_hedge_interval=args.hedge_freq,
        tail_hedge_delta_trigger_ratio=args.tail_hedge_trigger,
    )
    return replace(
        cfg,
        steps=args.steps,
        seed=args.seed,
        seeding_mode=args.seeding_mode,
        market=market,
        quoting=quoting,
        hedging=hedging,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio seeding
# ─────────────────────────────────────────────────────────────────────────────


def _seed_portfolio(pm: PortfolioManager, surface, spot: float, cfg: SimConfig) -> None:
    """Seed the initial book per ``cfg.seeding_mode``.

    - ``"flat"`` (default): no seeded positions; quoting engine builds inventory.
    - ``"short_straddle"``: legacy seed — 5x short ATM straddle + 3x OTM wings.
    """
    if cfg.seeding_mode == "flat":
        return
    if cfg.seeding_mode != "short_straddle":
        raise ValueError(f"unknown seeding_mode: {cfg.seeding_mode!r}")

    mid_idx = len(surface.expiries) // 2
    expiry_t = float(surface.expiries[mid_idx])
    expiry_d = int(surface.expiries_days[mid_idx])

    strikes = np.array([spot * 0.90, spot, spot, spot * 1.10])
    Ty = np.full(4, expiry_t)
    Td = np.array([expiry_d, expiry_d, expiry_d, expiry_d])
    ot = np.array(["put", "put", "call", "call"])
    vols = np.array([surface.implied_vol(k, expiry_d) for k in strikes])
    quotes = price_chain(spot, strikes, Ty, Td, ot, vols, r=cfg.market.r, q=cfg.market.q)

    pm.add_position(quotes[0], 3, "buy")  # 90-put  long
    pm.add_position(quotes[1], 5, "sell")  # ATM put short
    pm.add_position(quotes[2], 5, "sell")  # ATM call short
    pm.add_position(quotes[3], 3, "buy")  # 110-call long


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_stress_df(reports: list) -> pd.DataFrame:
    """Convert ``list[StressReport]`` to a DataFrame for ``display.stress_panel``."""
    if not reports:
        return pd.DataFrame(columns=["scenario", "spot_mult", "vol_shift", "skew_twist", "term_slope", "pnl"])
    return pd.DataFrame(
        [
            {
                "scenario": r.scenario_name,
                "spot_mult": 1.0 + r.spot_shock_pct / 100.0,
                "vol_shift": r.vol_shock_pts / 100.0,
                "skew_twist": float(r.skew_twist),
                "term_slope": float(r.term_slope),
                "pnl": r.pnl_impact,
            }
            for r in reports
        ]
    )


def _quote_sample(engine: QuotingEngine, spot: float, surface) -> list:
    """Return up to 8 representative quotes for the dashboard.

    Failures propagate: quote generation is on the critical trading path and
    must not be masked by a broad exception handler.
    """
    near = float(surface.expiries[0])
    far = float(surface.expiries[min(2, len(surface.expiries) - 1)])
    out: list = []
    for expiry in (near, far):
        for strike, otype in (
            (spot * 0.95, "put"),
            (spot, "put"),
            (spot, "call"),
            (spot * 1.05, "call"),
        ):
            out.append(engine.quote_market(strike, expiry, otype))
    return out[:8]


def _attempt_fills(engine: QuotingEngine, spot: float, surface) -> None:
    """Simulate inbound order flow: attempt fills on a small ATM/wing grid.

    A fill that would breach risk limits raises ``ValueError`` from
    ``update_portfolio``; that is a legitimate, expected outcome and is
    tolerated.  All other exceptions propagate so that real defects in the
    quoting/fill path (e.g. interface mismatches, pricing failures) are not
    hidden by a silent swallow.
    """
    near = float(surface.expiries[0])
    mid = float(surface.expiries[min(1, len(surface.expiries) - 1)])
    for expiry in (near, mid):
        for strike in (spot * 0.95, spot, spot * 1.05):
            for otype in ("call", "put"):
                fill = engine.simulate_fill(strike, expiry, otype)
                if fill is None:
                    continue
                try:
                    engine.update_portfolio(fill)
                except ValueError:
                    # Fill would breach risk limits — expected backpressure.
                    continue


def _iv_grid(surface, spot: float) -> pd.DataFrame:
    strikes = [spot * m for m in (0.90, 0.95, 1.00, 1.05, 1.10)]
    expiries = list(surface.expiries[:4])
    return surface.iv_grid(strikes, expiries)


def _greeks_dict(pm: PortfolioManager) -> tuple[dict, object]:
    """Return ``(greeks_dict, BookGreeks)`` suitable for ``book_panel``."""
    g = pm.compute_book_greeks()
    d = {
        "delta": g.delta,
        "gamma": g.gamma,
        "vega": g.vega,
        "theta": g.theta,
        "value": g.total_value,
    }
    return d, g


def _strategy_dict(cfg: SimConfig, portfolio: object) -> dict[str, float]:
    tail_hedge_qty = sum(
        float(pos.quantity)
        for pos in getattr(portfolio, "positions", [])
        if not getattr(pos, "inventory", True)
        and getattr(pos, "label", "") == "tail_hedge"
        and getattr(pos, "option_type", "") == "put"
    )
    daily_theta = abs(float(getattr(portfolio, "last_daily_theta", 0.0)))
    stress_guard = -float(cfg.quoting.stress_guard_multiple) * max(daily_theta, 1e-12)
    return {
        "target_gamma": float(cfg.quoting.target_net_gamma),
        "kelly_fraction": float(cfg.quoting.kelly_fraction),
        "tail_trigger": float(cfg.hedging.tail_hedge_delta_trigger_ratio),
        "tail_hedge_qty": float(tail_hedge_qty),
        "worst_stress": float(getattr(portfolio, "last_worst_stress_pnl", 0.0)),
        "stress_guard": float(stress_guard),
    }


class _PortfolioView:
    """Thin read-only adapter exposing ``Portfolio`` as a ``BookState``-like
    object for ``display.book_panel``, which reads ``spot_position``."""

    __slots__ = ("_p",)

    def __init__(self, portfolio) -> None:
        self._p = portfolio

    @property
    def positions(self):
        return self._p.positions

    @property
    def spot_position(self) -> float:  # BookState attribute name
        return self._p.hedge_shares

    @property
    def cash(self) -> float:
        return self._p.cash

    @property
    def transaction_costs(self) -> float:
        return self._p.transaction_costs


# ─────────────────────────────────────────────────────────────────────────────
# Simulation state
# ─────────────────────────────────────────────────────────────────────────────


class _SimState:
    """Mutable state bag for the main simulation loop."""

    def __init__(self, cfg: SimConfig, backend: str) -> None:
        self.cfg = cfg
        self.backend = backend
        self.pm = PortfolioManager(cfg.hedging)
        self.rv_tracker = VarianceTracker(window=20)
        self.residual_attr = ResidualAttributionTracker()
        self.surf_mon = SurfaceMonitor(stability_window=30)
        self.prev_spot: float | None = None
        self.pnl_report = None
        self.diag_report = None
        self.surface = None
        self.spot: float = cfg.market.s0
        self.inst_vol: float = cfg.market.sigma
        self.spot_history: list[float] = []
        self._stress_df = _to_stress_df([])  # refreshed every 20 steps

    def step(self, tick, step_idx: int) -> None:
        cfg = self.cfg
        spot = tick.spot
        self.spot = spot
        self.inst_vol = tick.inst_vol
        self.spot_history.append(spot)
        if len(self.spot_history) > 60:
            self.spot_history = self.spot_history[-60:]

        chain = build_chain(tick.spot, tick.inst_vol, cfg.market, cfg.chain)
        surface = build_surface(
            chain,
            spot,
            cfg.market.r,
            cfg.market.q,
            cfg.surface,
            backend=self.backend,
        )
        self.surface = surface
        self.surf_mon.update(surface, spot)

        # Seed the book on the first step
        if step_idx == 0:
            _seed_portfolio(self.pm, surface, spot, cfg)

        # Quoting engine always references the freshly fitted surface
        engine = QuotingEngine(surface, self.pm.portfolio, cfg.quoting)

        # Simulate inbound fills
        _attempt_fills(engine, spot, surface)

        # Mark-to-market must be called before delta_hedge
        self.pnl_report = self.pm.mark_to_market(surface, spot)

        # Delta-hedge at the user-specified frequency
        if step_idx % cfg.hedging.discrete_hedge_interval == 0:
            self.pm.delta_hedge(spot)

        # Update diagnostics trackers
        if self.prev_spot is not None:
            ret = np.log(spot / self.prev_spot)
            mid_idx = len(surface.expiries) // 2
            iv_atm = float(surface.implied_vol(spot, float(surface.expiries[mid_idx])))
            g = self.pm.compute_book_greeks()
            self.rv_tracker.update(float(ret), iv_atm=iv_atm, vega=float(g.vega), dt=cfg.hedging.dt)
            # Residual attribution = portion of realised step P&L not
            # explained by delta/gamma/theta/vega.  Source of truth is
            # ``PnLReport.residual_pnl`` from ``PortfolioManager``.
            self.residual_attr.update(float(self.pnl_report.residual_pnl))

        self.diag_report = build_diagnostics_report(
            self.rv_tracker, self.residual_attr, self.surf_mon, dt=cfg.hedging.dt
        )
        log_runtime_sanity_checks(step_idx + 1, self.pm.portfolio, self.pnl_report)

        # Refresh stress table periodically and on the final step so the final
        # report can reuse a current snapshot instead of recomputing another
        # full scenario grid.
        if step_idx % 20 == 0 or step_idx == cfg.steps - 1:
            reports = run_all_scenarios(
                self.pm.portfolio,
                surface,
                spot,
                cfg.market.r,
                cfg.market.q,
            )
            self._stress_df = _to_stress_df(reports)

        self.prev_spot = spot


def _build_live_panels(state: _SimState, step_idx: int, tick) -> dict:
    """Assemble the 6 ``LiveDashboard`` panels from current simulation state."""
    surface = state.surface
    spot = state.spot
    gd, go = _greeks_dict(state.pm)

    return dict(
        feed_panel=display.feed_panel(
            step_idx,
            tick.t,
            spot,
            tick.inst_vol,
            spot_history=list(state.spot_history),
        ),
        surface_panel=display.surface_panel(_iv_grid(surface, spot), surface.arbitrage_report()),
        book_panel=display.book_panel(
            _PortfolioView(state.pm.portfolio),
            gd,
            go.total_value,
            vanna=go.vanna,
            volga=go.volga,
            break_even_move=go.break_even_move,
            strategy=_strategy_dict(state.cfg, state.pm.portfolio),
        ),
        stress_panel=display.stress_panel(state._stress_df),
        pnl_panel=display.pnl_panel(state.pnl_report),
        diagnostics_panel=display.diagnostics_panel(state.diag_report),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Final summary (shared between run and backtest)
# ─────────────────────────────────────────────────────────────────────────────


def _print_final_summary(state: _SimState) -> None:
    """Print an AFL-style final report to the console."""
    if state.surface is None:
        return

    surface = state.surface
    spot = state.spot
    cfg = state.cfg

    display.console.rule("[bold white]── FINAL REPORT ──[/bold white]")

    rv = state.rv_tracker.realized_vol_annual(cfg.hedging.dt)
    display.console.print(
        display.feed_panel(
            cfg.steps - 1,
            (cfg.steps - 1) * cfg.market.dt,
            spot,
            state.inst_vol,
            spot_history=list(state.spot_history),
        )
    )

    grid = _iv_grid(surface, spot)
    arb = surface.arbitrage_report()
    display.console.print(display.surface_panel(grid, arb))

    gd, go = _greeks_dict(state.pm)
    display.console.print(
        display.book_panel(
            _PortfolioView(state.pm.portfolio),
            gd,
            go.total_value,
            vanna=go.vanna,
            volga=go.volga,
            break_even_move=go.break_even_move,
            strategy=_strategy_dict(cfg, state.pm.portfolio),
        )
    )

    engine = QuotingEngine(surface, state.pm.portfolio, cfg.quoting)
    display.console.print(display.quotes_panel(_quote_sample(engine, spot, surface)))

    mid_idx = len(surface.expiries) // 2
    iv_atm = float(surface.implied_vol(spot, float(surface.expiries[mid_idx])))
    display.console.print(display.risk_panel(rv, iv_atm, state.residual_attr.residual_attribution))

    display.console.print(display.stress_panel(state._stress_df))

    display.console.print(display.pnl_panel(state.pnl_report))
    display.console.print(display.diagnostics_panel(state.diag_report))


# ─────────────────────────────────────────────────────────────────────────────
# Command implementations
# ─────────────────────────────────────────────────────────────────────────────


def cmd_run(cfg: SimConfig, backend: str, verbose: bool) -> None:
    """Live simulation with a real-time Rich dashboard."""
    display.banner("vol-surface-mm  [run]")

    state = _SimState(cfg, backend)
    feed = SyntheticFeed(cfg.market, steps=cfg.steps, seed=cfg.seed)
    shutdown = False

    def _handle_sigint(sig: int, frame: object) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    with display.LiveDashboard(refresh_rate=4.0, screen=True) as dash:
        for step_idx, tick in enumerate(feed):
            if shutdown:
                break

            state.step(tick, step_idx)

            if verbose:
                display.console.log(f"step={step_idx:>4}  spot={tick.spot:.2f}  iv={tick.inst_vol:.3f}")

            dash.update(**_build_live_panels(state, step_idx, tick))

    _print_final_summary(state)


def cmd_backtest(cfg: SimConfig, backend: str, verbose: bool) -> int:
    """Run the full simulation silently; print final report; return exit code.

    Returns 0 if the variance risk premium is positive (we captured VRP),
    or 1 otherwise — making this suitable for use in CI pipelines.
    """
    state = _SimState(cfg, backend)
    feed = SyntheticFeed(cfg.market, steps=cfg.steps, seed=cfg.seed)

    for step_idx, tick in enumerate(feed):
        state.step(tick, step_idx)
        if verbose:
            pnl = state.pnl_report.total_pnl if state.pnl_report else 0.0
            print(
                f"step={step_idx:>4}  spot={tick.spot:.2f}  pnl={pnl:+.4f}",
                file=sys.stderr,
            )

    _print_final_summary(state)

    vrp = 0.0
    if state.diag_report is not None:
        vrp = float(state.diag_report.variance_risk_premium)
    return 0 if vrp > 0 else 1


def cmd_stress(cfg: SimConfig, backend: str) -> None:
    """Fit one surface snapshot, run all stress scenarios, print and exit."""
    display.banner("vol-surface-mm  [stress]")

    feed = SyntheticFeed(cfg.market, steps=1, seed=cfg.seed)
    tick = next(iter(feed))
    chain = build_chain(tick.spot, tick.inst_vol, cfg.market, cfg.chain)
    surface = build_surface(
        chain,
        tick.spot,
        cfg.market.r,
        cfg.market.q,
        cfg.surface,
        backend=backend,
    )

    pm = PortfolioManager(cfg.hedging)
    _seed_portfolio(pm, surface, tick.spot, cfg)

    reports = run_all_scenarios(pm.portfolio, surface, tick.spot, cfg.market.r, cfg.market.q)
    display.console.print(display.stress_panel(_to_stress_df(reports)))


def cmd_surface(cfg: SimConfig, backend: str) -> None:
    """Fit vol surface for one snapshot, display grid and arb report, exit."""
    display.banner("vol-surface-mm  [surface]")

    feed = SyntheticFeed(cfg.market, steps=1, seed=cfg.seed)
    tick = next(iter(feed))
    chain = build_chain(tick.spot, tick.inst_vol, cfg.market, cfg.chain)
    surface = build_surface(
        chain,
        tick.spot,
        cfg.market.r,
        cfg.market.q,
        cfg.surface,
        backend=backend,
    )

    grid = _iv_grid(surface, tick.spot)
    arb = surface.arbitrage_report()
    display.console.print(display.surface_panel(grid, arb))


def cmd_config(cfg: SimConfig) -> None:
    """Print effective simulator configuration as JSON."""
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True))


def cmd_artifacts(args: argparse.Namespace) -> None:
    """Generate deterministic results artifacts."""
    argv = [
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--backend",
        str(args.backend),
        "--results-dir",
        str(args.results_dir),
    ]
    generate_artifacts.main(argv)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run or merge the grouped parameter sweep workflow."""
    argv: list[str] = [
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--backend",
        str(args.backend),
        "--results-dir",
        str(args.results_dir),
    ]
    if args.merge_only:
        argv.append("--merge-only")
    if args.kelly_shard is not None:
        argv.extend(["--kelly-shard", str(args.kelly_shard)])
    param_sweep.main(argv)


def cmd_plot_sweep(args: argparse.Namespace) -> None:
    """Render grouped sweep heatmap to PNG."""
    argv = ["--input", str(args.input), "--output", str(args.output)]
    plot_sweep.main(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Attach options shared across all sub-commands."""
    p.add_argument(
        "--backend",
        choices=["sabr", "spline"],
        default="sabr",
        help="vol surface backend (default: sabr)",
    )
    p.add_argument(
        "--spot",
        type=float,
        default=DEFAULT_CONFIG.market.s0,
        metavar="FLOAT",
        help="starting spot price",
    )
    p.add_argument(
        "--vol",
        type=float,
        default=DEFAULT_CONFIG.market.sigma,
        metavar="FLOAT",
        help="starting vol (annual, decimal)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_CONFIG.market.r,
        metavar="FLOAT",
        help="risk-free rate (continuous)",
    )
    p.add_argument(
        "--div",
        type=float,
        default=DEFAULT_CONFIG.market.q,
        metavar="FLOAT",
        help="dividend yield (continuous)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        metavar="INT",
        help="random seed",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_CONFIG.steps,
        metavar="INT",
        help="simulation steps",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_CONFIG.market.dt,
        metavar="FLOAT",
        help="time step in years (default: 1/252)",
    )
    p.add_argument(
        "--hedge-freq",
        type=int,
        dest="hedge_freq",
        default=DEFAULT_CONFIG.hedging.discrete_hedge_interval,
        metavar="INT",
        help="rehedge every N steps",
    )
    p.add_argument(
        "--gamma-target",
        type=float,
        default=DEFAULT_CONFIG.quoting.target_net_gamma,
        metavar="FLOAT",
        help="target net gamma for quote skewing",
    )
    p.add_argument(
        "--kelly-fraction",
        type=float,
        default=DEFAULT_CONFIG.quoting.kelly_fraction,
        metavar="FLOAT",
        help="Kelly cap applied to option risk budget",
    )
    p.add_argument(
        "--tail-hedge-trigger",
        type=float,
        default=DEFAULT_CONFIG.hedging.tail_hedge_delta_trigger_ratio,
        metavar="FLOAT",
        help="delta-limit ratio that triggers protective put hedges",
    )
    p.add_argument(
        "--seeding-mode",
        choices=["flat", "short_straddle"],
        default=DEFAULT_CONFIG.seeding_mode,
        help="initial portfolio seed mode",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="extra per-step logging",
    )


def _add_artifacts_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=42, metavar="INT", help="random seed")
    p.add_argument("--steps", type=int, default=252, metavar="INT", help="simulation steps")
    p.add_argument(
        "--backend",
        choices=["sabr", "spline"],
        default="sabr",
        help="vol surface backend",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        metavar="PATH",
        help="output directory for artifact files",
    )


def _add_sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="merge shard CSVs into canonical sweep outputs",
    )
    p.add_argument(
        "--kelly-shard",
        type=float,
        metavar="FLOAT",
        help="run only one Kelly fraction shard",
    )
    p.add_argument("--seed", type=int, default=42, metavar="INT", help="random seed")
    p.add_argument("--steps", type=int, default=252, metavar="INT", help="simulation steps")
    p.add_argument(
        "--backend",
        choices=["sabr", "spline"],
        default="sabr",
        help="vol surface backend",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        metavar="PATH",
        help="directory for sweep CSV and stdout outputs",
    )


def _add_plot_sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input",
        type=Path,
        default=Path("results/param_sweep_grouped.csv"),
        metavar="PATH",
        help="grouped sweep CSV input path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/sweep_summary.png"),
        metavar="PATH",
        help="plot image output path",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vol-surface-mm",
        description="Options market-making simulator with vol surface analytics.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("run", "start live simulation with real-time dashboard"),
        ("backtest", "run full simulation silently, output final report"),
        ("stress", "run stress scenarios for a single snapshot and exit"),
        ("surface", "fit and display vol surface for a single snapshot and exit"),
        ("config", "print effective simulation config and exit"),
    ):
        _add_common_args(sub.add_parser(name, help=help_text))

    _add_artifacts_args(
        sub.add_parser("artifacts", help="generate final report and snapshot artifacts")
    )
    _add_sweep_args(sub.add_parser("sweep", help="run grouped parameter sweep or merge shards"))
    _add_plot_sweep_args(sub.add_parser("plot-sweep", help="render grouped sweep summary image"))

    args = parser.parse_args()
    verbose = bool(getattr(args, "verbose", False))

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.cmd in {"run", "backtest", "stress", "surface", "config"}:
        cfg = _build_cfg(args)
        if args.cmd == "run":
            cmd_run(cfg, backend=args.backend, verbose=args.verbose)
        elif args.cmd == "backtest":
            sys.exit(cmd_backtest(cfg, backend=args.backend, verbose=args.verbose))
        elif args.cmd == "stress":
            cmd_stress(cfg, backend=args.backend)
        elif args.cmd == "surface":
            cmd_surface(cfg, backend=args.backend)
        elif args.cmd == "config":
            cmd_config(cfg)
        return

    if args.cmd == "artifacts":
        cmd_artifacts(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "plot-sweep":
        cmd_plot_sweep(args)


if __name__ == "__main__":
    main()
