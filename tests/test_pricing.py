"""Tests for Black-Scholes pricing, implied vols, and P&L decomposition."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.core.hedging import PortfolioManager
from vol_surface_mm.core.pricing import bs_greeks, bs_price, implied_vol, price_chain
from vol_surface_mm.core.surface import build_surface
from vol_surface_mm.data.feed import SyntheticFeed
from vol_surface_mm.data.options_chain import build_chain


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260424)


@pytest.fixture
def parity_cases(rng: np.random.Generator) -> list[tuple[float, float, float, float, float, float]]:
    cases: list[tuple[float, float, float, float, float, float]] = []
    for _ in range(100):
        spot = float(rng.uniform(50.0, 150.0))
        strike = float(spot * rng.uniform(0.70, 1.30))
        expiry = float(rng.uniform(0.05, 2.00))
        sigma = float(rng.uniform(0.05, 0.80))
        rate = float(rng.uniform(0.00, 0.08))
        div = float(rng.uniform(0.00, 0.05))
        cases.append((spot, strike, expiry, sigma, rate, div))
    return cases


@pytest.fixture
def roundtrip_cases(rng: np.random.Generator) -> list[tuple[float, float, float, float, float, float]]:
    cases: list[tuple[float, float, float, float, float, float]] = []
    for _ in range(50):
        spot = float(rng.uniform(60.0, 140.0))
        strike = float(spot * rng.uniform(0.80, 1.20))
        expiry = float(rng.uniform(0.10, 2.00))
        sigma = float(rng.uniform(0.05, 0.70))
        rate = float(rng.uniform(0.00, 0.07))
        div = float(rng.uniform(0.00, 0.04))
        cases.append((spot, strike, expiry, sigma, rate, div))
    return cases


def _fd_delta(spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float) -> float:
    eps = max(spot * 1e-5, 1e-4)
    up = float(bs_price(spot + eps, strike, expiry, rate, div, sigma, "call"))
    down = float(bs_price(spot - eps, strike, expiry, rate, div, sigma, "call"))
    return (up - down) / (2.0 * eps)


def _fd_gamma(spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float) -> float:
    eps = max(spot * 1e-4, 1e-3)
    up = float(bs_price(spot + eps, strike, expiry, rate, div, sigma, "call"))
    mid = float(bs_price(spot, strike, expiry, rate, div, sigma, "call"))
    down = float(bs_price(spot - eps, strike, expiry, rate, div, sigma, "call"))
    return (up - 2.0 * mid + down) / (eps * eps)


def _fd_vega(spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float) -> float:
    eps = 1e-5
    up = float(bs_price(spot, strike, expiry, rate, div, sigma + eps, "call"))
    down = float(bs_price(spot, strike, expiry, rate, div, sigma - eps, "call"))
    return (up - down) / (2.0 * eps)


def _fd_theta(spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float) -> float:
    eps = min(1e-5, expiry / 4.0)
    earlier = float(bs_price(spot, strike, expiry - eps, rate, div, sigma, "call"))
    later = float(bs_price(spot, strike, expiry + eps, rate, div, sigma, "call"))
    return (earlier - later) / (2.0 * eps * 365.0)


def _seed_portfolio(pm: PortfolioManager, surface, spot: float) -> None:
    expiry_idx = len(surface.expiries) // 2
    expiry_t = float(surface.expiries[expiry_idx])
    expiry_d = int(surface.expiries_days[expiry_idx])
    strikes = np.array([spot * 0.95, spot, spot * 1.05])
    expiry_years = np.full(3, expiry_t)
    expiry_days = np.full(3, expiry_d)
    option_types = np.array(["put", "call", "call"])
    vols = np.array([surface.implied_vol(float(strike), expiry_d) for strike in strikes])
    quotes = price_chain(
        spot,
        strikes,
        expiry_years,
        expiry_days,
        option_types,
        vols,
        r=surface.r,
        q=surface.q,
    )

    pm.add_position(quotes[0], 5, "buy")
    pm.add_position(quotes[1], 5, "sell")
    pm.add_position(quotes[2], 3, "buy")


def test_black_scholes_put_call_parity(
    parity_cases: list[tuple[float, float, float, float, float, float]],
) -> None:
    for spot, strike, expiry, sigma, rate, div in parity_cases:
        call = float(bs_price(spot, strike, expiry, rate, div, sigma, "call"))
        put = float(bs_price(spot, strike, expiry, rate, div, sigma, "put"))
        lhs = call - put
        rhs = spot * np.exp(-div * expiry) - strike * np.exp(-rate * expiry)
        assert abs(lhs - rhs) < 1e-10, (
            "put-call parity failed for "
            f"S={spot:.6f}, K={strike:.6f}, T={expiry:.6f}, sigma={sigma:.6f}, "
            f"r={rate:.6f}, q={div:.6f}: lhs={lhs:.12f}, rhs={rhs:.12f}"
        )


@pytest.mark.parametrize(
    ("greek_name", "fd_fn", "tolerance"),
    [
        ("delta", _fd_delta, 1e-4),
        ("gamma", _fd_gamma, 1e-4),
        ("vega", _fd_vega, 1e-4),
        ("theta", _fd_theta, 1e-4),
    ],
)
def test_bs_greeks_match_finite_differences(greek_name: str, fd_fn, tolerance: float) -> None:
    spot = 100.0
    strike = 97.0
    expiry = 0.60
    rate = 0.02
    div = 0.01
    sigma = 0.24

    greeks = bs_greeks(spot, strike, expiry, rate, div, sigma, "call")
    fd_value = fd_fn(spot, strike, expiry, rate, div, sigma)
    analytic_value = getattr(greeks, greek_name)

    assert abs(analytic_value - fd_value) < tolerance, (
        f"expected {greek_name} within {tolerance:.1e}, analytic={analytic_value:.8f}, fd={fd_value:.8f}"
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_round_trip(
    option_type: str,
    roundtrip_cases: list[tuple[float, float, float, float, float, float]],
) -> None:
    for spot, strike, expiry, sigma, rate, div in roundtrip_cases:
        price = float(bs_price(spot, strike, expiry, rate, div, sigma, option_type))
        sigma_implied = implied_vol(price, spot, strike, expiry, rate, div, option_type)
        assert abs(sigma - sigma_implied) < 1e-8, (
            f"expected IV round-trip error < 1e-8 for {option_type}, "
            f"got |{sigma:.12f} - {sigma_implied:.12f}|"
        )


def test_portfolio_pnl_decomposition_balances_to_total_pnl() -> None:
    cfg = DEFAULT_CONFIG
    pm = PortfolioManager(cfg.hedging)
    feed = SyntheticFeed(cfg.market, steps=10, seed=42)

    reports = []
    for step_idx, tick in enumerate(feed):
        chain = build_chain(tick.spot, tick.inst_vol, cfg.market, cfg.chain)
        surface = build_surface(chain, tick.spot, cfg.market.r, cfg.market.q, cfg.surface)

        if step_idx == 0:
            _seed_portfolio(pm, surface, tick.spot)

        report = pm.mark_to_market(surface, tick.spot)
        reports.append(report)
        pm.delta_hedge(tick.spot)

    assert len(reports) == 10
    assert len(reports[1:]) == 9

    for report in reports[1:]:
        summed = (
            report.delta_pnl + report.gamma_pnl + report.theta_pnl + report.vega_pnl + report.residual_pnl
        )
        assert abs(summed - report.total_pnl) < 1e-3, (
            f"expected P&L components to sum to total within 1e-3 at step {report.step}, "
            f"components={summed:.8f}, total={report.total_pnl:.8f}"
        )
