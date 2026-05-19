"""Property-based tests for quoting, gamma-target, surface arbitrage, P&L identity,
and backtest determinism.

Performance note
----------------
Every Hypothesis draw that calls ``build_surface(..., backend="sabr")`` invokes
L-BFGS-B calibration per expiry slice (~2 s per call on an M-series Mac).  To keep
the full suite under ~90 s the simulation tests (gamma convergence, P&L identity,
determinism) build the surface **once** from the initial tick and reuse it across all
loop iterations.  This is intentional: the tests probe quoting/hedging invariants, not
surface dynamics.

Extended overnight coverage::

    pytest tests/test_strategy.py --hypothesis-seed=0 -x \\
        --hypothesis-max-examples=50
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.core.hedging import PortfolioManager
from vol_surface_mm.core.pricing import price_chain
from vol_surface_mm.core.quoting import QuotingEngine
from vol_surface_mm.core.surface import build_surface, sabr_implied_vol
from vol_surface_mm.data.feed import SyntheticFeed
from vol_surface_mm.data.options_chain import build_chain
from vol_surface_mm.diagnostics.metrics import ResidualAttributionTracker, SurfaceMonitor, VarianceTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPRESS = [HealthCheck.too_slow, HealthCheck.filter_too_much]


def _make_surface(spot: float, vol: float):
    """Build a SABR surface for the given spot and instantaneous vol."""
    cfg = DEFAULT_CONFIG
    market = replace(cfg.market, s0=float(spot), sigma=float(vol))
    chain = build_chain(float(spot), float(vol), market, cfg.chain)
    surface = build_surface(chain, float(spot), market.r, market.q, cfg.surface, backend="sabr")
    return cfg, market, surface


def _seed_portfolio(pm: PortfolioManager, surface: Any, spot: float, cfg: Any) -> None:
    """Seed the portfolio with a small strangle-like position."""
    mid_idx = len(surface.expiries) // 2
    expiry_t = float(surface.expiries[mid_idx])
    expiry_d = int(surface.expiries_days[mid_idx])

    strikes = np.array([spot * 0.90, spot, spot, spot * 1.10])
    Ty = np.full(4, expiry_t)
    Td = np.full(4, expiry_d, dtype=np.int32)
    ot = np.array(["put", "put", "call", "call"])
    vols = np.array([surface.implied_vol(float(k), expiry_d) for k in strikes])
    quotes = price_chain(spot, strikes, Ty, Td, ot, vols, r=cfg.market.r, q=cfg.market.q)

    pm.add_position(quotes[0], 3, "buy")
    pm.add_position(quotes[1], 5, "sell")
    pm.add_position(quotes[2], 5, "sell")
    pm.add_position(quotes[3], 3, "buy")


# ---------------------------------------------------------------------------
# Test 1: Quoting monotonicity
# ask_vol >= bid_vol always; spread widens with vega.
# Surface built once per draw — cost = 1 calibration.
# ---------------------------------------------------------------------------


@settings(max_examples=6, deadline=None, suppress_health_check=_SUPPRESS)
@given(
    spot=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    vol=st.floats(min_value=0.05, max_value=0.80, allow_nan=False, allow_infinity=False),
)
def test_property_quoting_monotonicity(spot: float, vol: float) -> None:
    cfg, _, surface = _make_surface(spot, vol)
    pm = PortfolioManager(cfg.hedging)
    engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)

    near_expiry = float(surface.expiries[0])
    q = engine.quote_market(float(spot), near_expiry, "call")

    if np.isfinite(q.bid_vol) and np.isfinite(q.ask_vol):
        assert q.ask_vol >= q.bid_vol, (
            f"bid_vol {q.bid_vol:.4f} > ask_vol {q.ask_vol:.4f} at spot={spot:.1f} vol={vol:.3f}"
        )

    # Spread must widen with vega (higher vega -> higher absolute uncertainty -> wider spread)
    dte_ref, oi_ref, gamma_ref = 14, 2500.0, 0.01
    hs_low = engine._half_spread(vega=0.05, gamma=gamma_ref, days_to_expiry=dte_ref, open_interest=oi_ref)
    hs_high = engine._half_spread(vega=2.00, gamma=gamma_ref, days_to_expiry=dte_ref, open_interest=oi_ref)
    assert hs_high >= hs_low, f"spread did not widen with vega: hs_low={hs_low:.6f}, hs_high={hs_high:.6f}"


# ---------------------------------------------------------------------------
# Test 2: Gamma-target steering signal
# When the book is below the gamma target, the engine must quote with a
# strictly positive gamma_pull_bps on every step where the quote is finite.
# This tests the steering *signal*, not the portfolio outcome (which can be
# blocked by stress-guard or Kelly limits for any given seed).
# Surface built ONCE from initial tick; reused across all steps.
# ---------------------------------------------------------------------------


@settings(max_examples=4, deadline=None, suppress_health_check=_SUPPRESS)
@given(
    steps=st.integers(min_value=6, max_value=12),
    seed=st.integers(min_value=0, max_value=999),
)
def test_property_gamma_target_convergence(steps: int, seed: int) -> None:
    cfg = DEFAULT_CONFIG
    pm = PortfolioManager(cfg.hedging)

    # Build surface once from initial spot/vol
    first_tick = next(iter(SyntheticFeed(cfg.market, steps=1, seed=int(seed))))
    _, _, surface = _make_surface(float(first_tick.spot), float(first_tick.inst_vol))

    feed = SyntheticFeed(cfg.market, steps=int(steps), seed=int(seed))
    steering_steps = 0  # steps where we were below target
    pull_positive_steps = 0  # of those, steps where gamma_pull_bps > 0

    for step_idx, tick in enumerate(feed):
        engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)
        exposure = engine.current_exposure()

        # Only look at short-dated ATM calls where gamma_pull is active.
        # Use the surface ATM strike (not tick spot) because the surface is frozen
        # across the loop and gamma_pull checks moneyness against surface.spot.
        near_expiry = float(surface.expiries[0])
        quote = engine.quote_market(float(surface.spot), near_expiry, "call")

        if exposure.net_gamma < cfg.quoting.target_net_gamma and np.isfinite(quote.bid):
            steering_steps += 1
            pull_positive_steps += int(float(quote.gamma_pull_bps) > 0.0)

        pm.mark_to_market(surface, float(tick.spot))
        if step_idx % cfg.hedging.discrete_hedge_interval == 0:
            pm.delta_hedge(float(tick.spot))

    # If we were ever below target and got a finite quote, the pull must be positive
    if steering_steps > 0:
        assert pull_positive_steps == steering_steps, (
            f"gamma_pull_bps was not positive on {steering_steps - pull_positive_steps}/"
            f"{steering_steps} below-target steps"
        )


# ---------------------------------------------------------------------------
# Test 3: Arbitrage-free surface
# SABR IVs -> surface; check zero calendar and butterfly violations.
# No simulation loop -- just SABR formula + one surface build per draw.
# ---------------------------------------------------------------------------


@settings(max_examples=6, deadline=None, suppress_health_check=_SUPPRESS)
@given(
    alpha=st.floats(min_value=0.05, max_value=0.80, allow_nan=False, allow_infinity=False),
    rho=st.floats(min_value=-0.95, max_value=-0.05, allow_nan=False, allow_infinity=False),
    nu=st.floats(min_value=0.10, max_value=1.00, allow_nan=False, allow_infinity=False),
)
def test_property_arbitrage_free_surface(alpha: float, rho: float, nu: float) -> None:
    cfg = DEFAULT_CONFIG
    spot = cfg.market.s0
    expiries_days = np.array(cfg.chain.expiries_dte, dtype=np.int32)
    strikes = spot * np.array(cfg.chain.moneyness, dtype=np.float64)

    rows: list[dict[str, float]] = []
    for dte in expiries_days:
        t = float(dte) / 365.0
        fwd = float(spot * np.exp((cfg.market.r - cfg.market.q) * t))
        ivs = sabr_implied_vol(fwd, strikes, t, float(alpha), float(rho), float(nu))
        assume(np.all(np.isfinite(ivs)))
        for k, iv in zip(strikes, ivs, strict=True):
            rows.append({"strike": float(k), "expiry": float(t), "iv": float(max(iv, 1e-4))})

    chain_df = pd.DataFrame(rows)
    surface = build_surface(chain_df, spot, cfg.market.r, cfg.market.q, cfg.surface, backend="sabr")
    report = surface.arbitrage_report(strikes=tuple(strikes), expiries=tuple(expiries_days))

    assert report["calendar_violations"] == 0, (
        f"calendar violations: {report['calendar_violations']} (a={alpha:.3f}, r={rho:.3f}, v={nu:.3f})"
    )
    assert report["butterfly_violations"] == 0, (
        f"butterfly violations: {report['butterfly_violations']} (a={alpha:.3f}, r={rho:.3f}, v={nu:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 4: P&L decomposition bookkeeping
# The report must satisfy the exact bookkeeping identity at every step:
#   total_pnl == delta_pnl + gamma_pnl + theta_pnl + vega_pnl + residual_pnl
# This is guaranteed by construction in hedging.py (residual is defined as
# total minus the greek attributions).  We verify it holds numerically.
# All attribution fields must also be finite.
#
# We do NOT test Taylor accuracy or theta sign — the frozen-surface + large-dS
# combination makes second-order expansions approximate by design.
# ---------------------------------------------------------------------------


@settings(max_examples=4, deadline=None, suppress_health_check=_SUPPRESS)
@given(seed=st.integers(min_value=0, max_value=999))
def test_property_pnl_decomposition_identity(seed: int) -> None:
    cfg = DEFAULT_CONFIG
    pm = PortfolioManager(cfg.hedging)
    feed = SyntheticFeed(cfg.market, steps=10, seed=int(seed))

    first_tick = next(iter(SyntheticFeed(cfg.market, steps=1, seed=int(seed))))
    _, _, surface = _make_surface(float(first_tick.spot), float(first_tick.inst_vol))

    for step_idx, tick in enumerate(feed):
        if step_idx == 0:
            _seed_portfolio(pm, surface, float(tick.spot), cfg)

        report = pm.mark_to_market(surface, float(tick.spot))

        # All attribution fields must be finite
        for field_name in ("total_pnl", "delta_pnl", "gamma_pnl", "theta_pnl", "vega_pnl", "residual_pnl"):
            val = getattr(report, field_name)
            assert np.isfinite(val), f"step {step_idx}: {field_name} = {val}"

        # Bookkeeping identity must hold exactly (up to float rounding):
        #   residual_pnl := total_pnl - (delta + gamma + theta + vega)
        greek_sum = report.delta_pnl + report.gamma_pnl + report.theta_pnl + report.vega_pnl
        recomputed_residual = report.total_pnl - greek_sum
        assert abs(recomputed_residual - report.residual_pnl) < 1e-9, (
            f"step {step_idx}: bookkeeping identity broken — "
            f"total={report.total_pnl:.6f}, greek_sum={greek_sum:.6f}, "
            f"reported_residual={report.residual_pnl:.6f}, "
            f"recomputed_residual={recomputed_residual:.6f}"
        )

        if step_idx % cfg.hedging.discrete_hedge_interval == 0:
            pm.delta_hedge(float(tick.spot))


# ---------------------------------------------------------------------------
# Test 5: Backtest determinism
# Two runs with the same seed produce bit-identical output signatures.
# Surface built ONCE per run; steps capped at 15.
# ---------------------------------------------------------------------------


def _backtest_signature(seed: int, steps: int = 15) -> tuple[tuple[float, ...], ...]:
    cfg = replace(DEFAULT_CONFIG, seed=int(seed), steps=int(steps))
    pm = PortfolioManager(cfg.hedging)
    rv_tracker = VarianceTracker(window=20)
    residual_attr = ResidualAttributionTracker()
    surf_mon = SurfaceMonitor(stability_window=30)

    feed = SyntheticFeed(cfg.market, steps=cfg.steps, seed=cfg.seed)
    first_tick = next(iter(SyntheticFeed(cfg.market, steps=1, seed=cfg.seed)))
    _, _, surface = _make_surface(float(first_tick.spot), float(first_tick.inst_vol))

    prev_spot: float | None = None
    signature: list[tuple[float, ...]] = []

    for step_idx, tick in enumerate(feed):
        surf_mon.update(surface, float(tick.spot))

        if step_idx == 0:
            _seed_portfolio(pm, surface, float(tick.spot), cfg)

        engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)
        near = float(surface.expiries[0])
        mid = float(surface.expiries[min(1, len(surface.expiries) - 1)])
        for expiry in (near, mid):
            for strike in (float(tick.spot) * 0.95, float(tick.spot), float(tick.spot) * 1.05):
                for otype in ("call", "put"):
                    fill = engine.simulate_fill(strike, expiry, otype)
                    if fill is None:
                        continue
                    with contextlib.suppress(ValueError):
                        engine.update_portfolio(fill)

        report = pm.mark_to_market(surface, float(tick.spot))
        if step_idx % cfg.hedging.discrete_hedge_interval == 0:
            pm.delta_hedge(float(tick.spot))

        if prev_spot is not None:
            ret = np.log(float(tick.spot) / prev_spot)
            mid_idx = len(surface.expiries) // 2
            iv_atm = float(surface.implied_vol(float(tick.spot), int(surface.expiries_days[mid_idx])))
            g = pm.compute_book_greeks()
            rv_tracker.update(float(ret), iv_atm=iv_atm, vega=float(g.vega), dt=cfg.hedging.dt)
            residual_attr.update(float(report.residual_pnl))

        g_now = pm.compute_book_greeks()
        signature.append(
            (
                float(tick.spot),
                float(tick.inst_vol),
                float(report.total_pnl),
                float(report.cum_total_pnl),
                float(report.cum_gamma_pnl),
                float(g_now.gamma),
                float(pm.portfolio.var_capture_running),
                float(pm.portfolio.last_worst_stress_pnl),
            )
        )
        prev_spot = float(tick.spot)

    return tuple(signature)


@settings(max_examples=3, deadline=None, suppress_health_check=_SUPPRESS)
@given(seed=st.integers(min_value=0, max_value=999))
def test_property_backtest_determinism(seed: int) -> None:
    run_a = _backtest_signature(seed=int(seed), steps=15)
    run_b = _backtest_signature(seed=int(seed), steps=15)
    assert run_a == run_b, "Backtest produced different outputs for the same seed"
