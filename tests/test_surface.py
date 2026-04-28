"""Tests for volatility-surface fitting and arbitrage checks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.core.pricing import implied_vol
from vol_surface_mm.core.surface import (
    VolSurface,
    build_surface,
    check_butterfly,
    check_calendar_spread,
    sabr_implied_vol,
)
from vol_surface_mm.data.options_chain import build_chain


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def sabr_smile_data(rng: np.random.Generator) -> dict[str, object]:
    spot = 100.0
    rate = 0.03
    div = 0.01
    expiry_years = 0.75
    forward = spot * np.exp((rate - div) * expiry_years)
    strikes = forward * np.exp(np.linspace(-0.20, 0.20, 11))

    alpha = 0.30
    rho = -0.30
    nu = 0.40
    true_vols = sabr_implied_vol(forward, strikes, expiry_years, alpha, rho, nu)
    noisy_vols = np.clip(true_vols + rng.normal(0.0, 0.002, size=true_vols.shape), 0.01, None)

    chain = pd.DataFrame(
        {
            "strike": strikes,
            "expiry": expiry_years,
            "iv": noisy_vols,
        }
    )
    return {
        "spot": spot,
        "rate": rate,
        "div": div,
        "expiry_years": expiry_years,
        "strikes": strikes,
        "true_vols": np.asarray(true_vols, dtype=np.float64),
        "chain": chain,
        "alpha": alpha,
        "rho": rho,
        "nu": nu,
    }


@dataclass
class CalendarSurfaceStub:
    strikes: np.ndarray
    expiries_days: np.ndarray
    total_variances: dict[tuple[float, int], float]

    def total_variance(self, strike: float, expiry_days: int) -> float:
        return self.total_variances[(float(strike), int(expiry_days))]


class ButterflySurfaceStub:
    def __init__(
        self, spot: float, rate: float, div: float, expiry_days: int, vols: dict[float, float]
    ) -> None:
        self.spot = float(spot)
        self.r = float(rate)
        self.q = float(div)
        self.strikes = np.array(sorted(vols), dtype=np.float64)
        self.expiries_days = np.array([int(expiry_days)], dtype=np.int64)
        self._vols = {float(strike): float(vol) for strike, vol in vols.items()}

    def vol(self, strike: float, expiry_days: int) -> float:
        assert int(expiry_days) == int(self.expiries_days[0])
        return self._vols[float(strike)]


def test_sabr_calibration_recovers_known_parameters(sabr_smile_data: dict[str, object]) -> None:
    surface = VolSurface(
        sabr_smile_data["chain"],
        sabr_smile_data["spot"],
        sabr_smile_data["rate"],
        sabr_smile_data["div"],
        backend="sabr",
    )
    surface.fit()

    params = next(iter(surface.sabr_params.values()))
    true_alpha = float(sabr_smile_data["alpha"])
    true_rho = float(sabr_smile_data["rho"])
    true_nu = float(sabr_smile_data["nu"])

    assert abs(params.alpha - true_alpha) < 0.02, (
        f"expected alpha within 0.02 of {true_alpha:.3f}, got {params.alpha:.6f}"
    )
    assert abs(params.rho - true_rho) < 0.05, (
        f"expected rho within 0.05 of {true_rho:.3f}, got {params.rho:.6f}"
    )
    assert abs(params.nu - true_nu) < 0.05, f"expected nu within 0.05 of {true_nu:.3f}, got {params.nu:.6f}"

    strikes = np.asarray(sabr_smile_data["strikes"], dtype=np.float64)
    expiry_years = float(sabr_smile_data["expiry_years"])
    fitted_vols = np.array([surface.implied_vol(float(strike), expiry_years) for strike in strikes])
    true_vols = np.asarray(sabr_smile_data["true_vols"], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((fitted_vols - true_vols) ** 2)))

    assert rmse < 0.005, f"expected fitted smile RMSE < 0.005, got {rmse:.6f}"


def test_check_calendar_spread_detects_violation() -> None:
    surface = CalendarSurfaceStub(
        strikes=np.array([100.0]),
        expiries_days=np.array([30, 60]),
        total_variances={
            (100.0, 30): 0.040,
            (100.0, 60): 0.030,
        },
    )

    violations = check_calendar_spread(surface, strikes=[100.0], expiries_days=[30, 60])

    assert violations, "expected a calendar-spread arbitrage violation"
    assert violations[0].strike == pytest.approx(100.0)
    assert violations[0].expiry_days_short == 30
    assert violations[0].expiry_days_long == 60
    assert violations[0].violation > 0.0


def test_check_butterfly_detects_violation() -> None:
    spot = 100.0
    rate = 0.01
    div = 0.0
    expiry_days = 180
    expiry_years = expiry_days / 365.0
    strikes = [90.0, 100.0, 110.0]
    call_prices = {
        90.0: 15.0,
        100.0: 14.0,
        110.0: 12.0,
    }
    vols = {
        strike: implied_vol(call_prices[strike], spot, strike, expiry_years, rate, div, "call")
        for strike in strikes
    }

    assert all(np.isfinite(vol) for vol in vols.values()), "expected invertible implied vols"

    surface = ButterflySurfaceStub(spot, rate, div, expiry_days, vols)
    violations = check_butterfly(surface, strikes=strikes, expiries_days=[expiry_days])

    assert violations, "expected a butterfly arbitrage violation"
    assert violations[0].expiry_days == expiry_days
    assert violations[0].strike == pytest.approx(100.0)
    assert violations[0].second_derivative < 0.0


def test_surface_build_and_query() -> None:
    cfg = DEFAULT_CONFIG
    chain = build_chain(100.0, 0.20, cfg.market, cfg.chain)
    surf = build_surface(chain, 100.0, cfg.market.r, cfg.market.q, cfg.surface)

    for t in cfg.chain.expiries_years:
        for m in cfg.chain.moneyness:
            iv = surf.implied_vol(100.0 * m, t)
            assert 0.0 < iv < 2.0


def test_surface_reprices_chain_in_reasonable_band() -> None:
    cfg = DEFAULT_CONFIG
    chain = build_chain(100.0, 0.20, cfg.market, cfg.chain)
    surf = build_surface(chain, 100.0, cfg.market.r, cfg.market.q, cfg.surface)

    errs = []
    for _, row in chain.iterrows():
        fit_iv = surf.implied_vol(float(row["strike"]), float(row["expiry"]))
        errs.append(abs(fit_iv - float(row["iv"])))

    assert np.mean(errs) < 0.02


def test_arbitrage_report_clean_on_fit_surface() -> None:
    cfg = DEFAULT_CONFIG
    chain = build_chain(100.0, 0.20, cfg.market, cfg.chain)
    surf = build_surface(chain, 100.0, cfg.market.r, cfg.market.q, cfg.surface)
    report = surf.arbitrage_report(
        strikes=[80, 90, 100, 110, 120],
        expiries=[dte / 365.0 for dte in cfg.chain.expiries_dte],
    )

    assert report.calendar_count == 0
    assert report.butterfly_count == 0


def test_surface_spline_backend_is_arbitrage_clean_on_chain_grid() -> None:
    cfg = DEFAULT_CONFIG
    chain = build_chain(100.0, 0.20, cfg.market, cfg.chain)
    surf = VolSurface(chain, 100.0, cfg.market.r, cfg.market.q, backend="spline")
    surf.fit()

    report = surf.arbitrage_report()
    grid = surf.to_grid()

    assert report.calendar_count == 0
    assert report.butterfly_count == 0
    assert grid.shape == (len(surf.strikes), len(surf.expiries_days))
    assert float(grid.iloc[len(surf.strikes) // 2, 0]) > 0.0


def test_surface_can_infer_iv_from_mid_prices() -> None:
    cfg = DEFAULT_CONFIG
    chain = build_chain(100.0, 0.20, cfg.market, cfg.chain).drop(columns=["iv"])
    surf = VolSurface(chain, 100.0, cfg.market.r, cfg.market.q, backend="sabr")
    surf.fit()

    atm_30d_vol = surf.vol(100.0, 30)
    report = surf.arbitrage_report()

    assert 0.05 < atm_30d_vol < 1.0
    assert report.calendar_count == 0
