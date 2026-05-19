"""End-to-end runtime test for the quoting/fill/inventory path.

Covers the plumbing recently repaired in ``core/quoting.py`` and ``legacy CLI``:

1. ``QuotingEngine`` accepts the new ``PortfolioManager.portfolio`` shape
   (``hedge_shares`` / no ``add`` method) without raising.
2. ``quote_market`` produces finite bid/ask and non-zero fill probabilities.
3. Under a fixed seed and deterministic quoting config, ``simulate_fill``
   produces at least one fill over a small loop.
4. Booking the fill via ``update_portfolio`` mutates the book: position
   count grows and the cash balance moves.

If any of these regress, the live dashboard's empty Quotes panel and
unchanged seeded book reappear silently, so this test is the canary.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.core.hedging import OptionPosition, PortfolioManager
from vol_surface_mm.core.quoting import Fill, QuotingEngine
from vol_surface_mm.core.surface import build_surface
from vol_surface_mm.data.feed import SyntheticFeed
from vol_surface_mm.data.options_chain import build_chain


def _fixed_surface_and_spot():
    cfg = DEFAULT_CONFIG
    feed = SyntheticFeed(cfg.market, steps=1, seed=cfg.seed)
    tick = next(iter(feed))
    chain = build_chain(tick.spot, tick.inst_vol, cfg.market, cfg.chain)
    surface = build_surface(
        chain,
        tick.spot,
        cfg.market.r,
        cfg.market.q,
        cfg.surface,
        backend="sabr",
    )
    return cfg, surface, float(tick.spot)


def _seed_short_vega_book(pm: PortfolioManager, spot: float, surface) -> None:
    expiry = float(surface.expiries[len(surface.expiries) // 2])
    pm.portfolio.positions.extend(
        [
            OptionPosition(spot, expiry, "call", -5.0),
            OptionPosition(spot, expiry, "put", -5.0),
        ]
    )


def test_quoting_engine_accepts_portfolio_manager_portfolio():
    """QuotingEngine must not raise when handed ``PortfolioManager.portfolio``."""
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)
    engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)

    # Reads hedge position through the new compatibility helper.
    exposure = engine.current_exposure()
    assert np.isfinite(exposure.net_delta)
    assert np.isfinite(exposure.net_gamma)
    assert np.isfinite(exposure.net_vega)

    # quote_market must return a structurally valid Quote with finite sides
    # and a positive bid fill probability (some fill edge must exist).
    q = engine.quote_market(spot, float(surface.expiries[0]), "call")
    assert np.isfinite(q.bid)
    assert np.isfinite(q.ask)
    assert q.ask > q.bid
    assert 0.0 < q.fill_probability_bid <= 0.95
    assert 0.0 < q.fill_probability_ask <= 0.95


def test_fills_occur_and_mutate_inventory_under_fixed_seed():
    """Under tuned deterministic config, at least one fill must occur and
    the portfolio state must reflect it."""
    cfg, surface, spot = _fixed_surface_and_spot()

    # Tune fill parameters so a fill is overwhelmingly likely while keeping
    # quoting/risk limits unchanged.  ``fill_rng_seed`` pins the RNG.
    quoting = replace(
        cfg.quoting,
        fill_base_probability=0.95,
        fill_distance_sensitivity=0.0,
        fill_oi_exponent=0.0,
        fill_rng_seed=7,
    )

    pm = PortfolioManager(cfg.hedging)
    engine = QuotingEngine(surface, pm.portfolio, quoting)

    seeded_positions = len(pm.portfolio.positions)
    seeded_cash = float(pm.portfolio.cash)

    fills = 0
    expiry = float(surface.expiries[0])
    # A small 6-contract grid is enough given the deterministic config.
    for strike in (spot * 0.975, spot, spot * 1.025):
        for otype in ("call", "put"):
            fill = engine.simulate_fill(strike, expiry, otype)
            if fill is None:
                continue
            engine.update_portfolio(fill)
            fills += 1

    assert fills >= 1, "no fills produced under deterministic quoting config"
    assert len(pm.portfolio.positions) > seeded_positions, "portfolio positions unchanged after a fill"
    assert pm.portfolio.cash != pytest.approx(seeded_cash), "portfolio cash unchanged after a fill"


def test_short_dated_atm_quotes_pull_toward_long_gamma_target():
    """Short-dated ATM quotes should raise both sides when the book gamma is
    below target, making bids more competitive while keeping offers harder to
    lift."""
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)

    control_engine = QuotingEngine(
        surface,
        pm.portfolio,
        replace(cfg.quoting, gamma_sensitivity=0.0),
    )
    biased_engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)

    expiry = float(surface.expiries[0])
    control_quote = control_engine.quote_market(spot, expiry, "call")
    biased_quote = biased_engine.quote_market(spot, expiry, "call")

    assert biased_quote.gamma_pull_bps > 0.0
    assert biased_quote.bid_vol > control_quote.bid_vol
    assert biased_quote.ask_vol > control_quote.ask_vol


def test_kelly_cap_reduces_quote_size_for_high_gamma_contracts():
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)
    pm.portfolio.bankroll = 250.0
    engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)

    quote = engine.quote_market(spot, float(surface.expiries[0]), "call")

    assert 0.0 < quote.size < cfg.quoting.quote_size


def test_negative_var_capture_streak_activates_gamma_bid_boost():
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)
    _seed_short_vega_book(pm, spot, surface)

    for _ in range(cfg.hedging.cheapness_negative_streak + 1):
        pm.mark_to_market(surface, spot)

    assert pm.portfolio.var_capture_running < 0.0
    assert pm.portfolio.gamma_bid_aggressiveness_multiplier == pytest.approx(
        cfg.hedging.cheapness_boost_multiplier
    )
    assert pm.portfolio.vol_cheapness_boost_periods_remaining == cfg.hedging.cheapness_boost_periods


def test_positive_delta_triggers_non_inventory_tail_hedge():
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)
    pm.portfolio.hedge_shares = cfg.hedging.delta_limit * cfg.hedging.tail_hedge_delta_trigger_ratio + 15.0

    pm.mark_to_market(surface, spot)
    pm.delta_hedge(spot)

    tail_hedges = [
        pos
        for pos in pm.portfolio.positions
        if not pos.inventory and pos.label == "tail_hedge" and pos.option_type == "put"
    ]
    assert tail_hedges


def test_stress_guard_rejects_fill_that_worsens_tail_loss():
    cfg, surface, spot = _fixed_surface_and_spot()
    pm = PortfolioManager(cfg.hedging)
    pm.portfolio.last_daily_theta = 1.0
    pm.portfolio.last_worst_stress_pnl = -10.0
    # Seed the cached worst shock so the trial revaluation is non-trivial. For
    # an ask-side call fill (short call), spot-up + vol-up makes the trade
    # worsen the existing tail loss and the guard must reject.
    pm.portfolio.last_worst_spot_mult = 1.10
    pm.portfolio.last_worst_vol_shift = 0.05
    engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)

    quote = engine.quote_market(spot, float(surface.expiries[0]), "call")
    fill = Fill(
        strike=quote.strike,
        expiry=quote.expiry,
        option_type=quote.option_type,
        side="ask",
        quantity=max(quote.size, 1.0),
        price=quote.mid_price,
        vol=quote.model_iv,
        probability=max(quote.fill_probability_ask, 0.01),
    )

    with pytest.raises(ValueError):
        engine.update_portfolio(fill)
