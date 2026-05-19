"""Scenario and shock analysis — jump risk, vol shocks, correlation shocks,
combined worst case.

Extends the basic ``Shock``/``run_shocks`` primitives (preserved for backward
compatibility) with:

* Jump scenarios — instantaneous spot shocks with frozen hedge (gap risk).
* Vol shocks — parallel shift, skew slope (OTM puts / ATM unchanged), and
  vol-of-vol butterfly (wing inflation).
* Correlation shocks — SABR-rho-aware spot-vol correlation shift;
  spline-backend fallback uses a linear skew approximation.
* Combined worst-case grid — brute-force (spot × vol) scenario matrix;
  worst ``top_n`` results returned.

Public API
----------
Backward-compatible (used by legacy CLI / display layer)::

    Shock, DEFAULT_SHOCKS, run_shocks(book, surface, spot, shocks) -> DataFrame

New::

    StressReport
    JUMP_SHOCKS, VOL_SHOCKS, CORRELATION_SHOCKS
    run_all_scenarios(portfolio, surface, spot, rate, div_yield) -> list[StressReport]
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import pandas as pd

from vol_surface_mm.core.hedging import BookState, Portfolio, mark_book
from vol_surface_mm.core.pricing import bs_greeks
from vol_surface_mm.core.surface import VolSurface, sabr_implied_vol

# ──────────────────────────────────────────────────────────────────────────────
# Grid constants for combined worst-case analysis
# ──────────────────────────────────────────────────────────────────────────────
_COMBINED_SPOT_MULTS: list[float] = [0.70, 0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.20]
_COMBINED_VOL_SHIFTS: list[float] = [-0.05, -0.03, 0.0, +0.03, +0.05]

# ──────────────────────────────────────────────────────────────────────────────
# Shock primitive
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Shock:
    """Describes a joint scenario applied to spot and the vol surface.

    Parameters
    ----------
    name:       Human-readable label for the scenario.
    spot_mult:  Spot multiplier, e.g. 0.90 = −10 % crash.
    vol_shift:  Additive parallel shift to implied vol (absolute).
    skew_twist: ``dIV = skew_twist × log(K/F)`` — steepens/flattens skew.
                Negative steepens (OTM puts up, ATM unchanged).
    term_slope: Multiplies total variance by ``(1 + slope × T)``.
    vov_shift:  ``dIV = vov_shift × log(K/F)²`` — inflates/deflates wings
                (butterfly / vol-of-vol shock).
    rho_shift:  Shifts the spot-vol correlation parameter ρ by this amount.
                For the SABR backend the actual calibrated ρ per slice is
                used; for the spline backend a linear skew proxy is applied.
    """

    name: str
    spot_mult: float = 1.0
    vol_shift: float = 0.0  # additive parallel shift (absolute)
    skew_twist: float = 0.0  # dIV = skew_twist × log(K/F)
    term_slope: float = 0.0  # variance × (1 + slope × T)
    vov_shift: float = 0.0  # dIV = vov_shift × log(K/F)²  (butterfly)
    rho_shift: float = 0.0  # spot-vol correlation shift (SABR ρ ± rho_shift)


# ──────────────────────────────────────────────────────────────────────────────
# StressReport output type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StressReport:
    """P&L impact and Greek sensitivities under a single stress scenario.

    Attributes
    ----------
    scenario_name:  Label of the applied :class:`Shock`.
    spot_shock_pct: Spot move as a percentage, e.g. −30 for a 30 % crash.
    vol_shock_pts:  Parallel vol shift in vol points, e.g. +5.
    pnl_impact:     Total P&L (options + frozen hedge) under the scenario.
    delta_after:    Net delta of the book after repricing.
    gamma_after:    Net gamma of the book after repricing.
    vega_after:     Net vega of the book after repricing.
    worst_case_pnl: Minimum P&L across all combined spot × vol scenarios
                    (populated by :func:`run_all_scenarios`).
    gap_risk_pnl:   P&L with the hedge frozen at its pre-shock level
                    (equals ``pnl_impact`` since the hedge is always frozen).
    delta_attr:     First-order spot contribution: ``Δ × dS``.
    gamma_attr:     Second-order spot contribution: ``½Γ × dS²``.
    vega_attr:      First-order vol contribution: ``vega × dσ_atm``.
    vanna_attr:     Cross contribution: ``vanna × dS × dσ_atm``.
    volga_attr:     Second-order vol contribution: ``½volga × dσ_atm²``.
    dominant_risk:  Name of the Greek with the largest absolute attribution.
    """

    scenario_name: str
    spot_shock_pct: float
    vol_shock_pts: float
    skew_twist: float = 0.0
    term_slope: float = 0.0
    vov_shift: float = 0.0
    rho_shift: float = 0.0
    pnl_impact: float = 0.0
    delta_after: float = 0.0
    gamma_after: float = 0.0
    vega_after: float = 0.0
    worst_case_pnl: float = 0.0
    gap_risk_pnl: float = 0.0
    delta_attr: float = 0.0
    gamma_attr: float = 0.0
    vega_attr: float = 0.0
    vanna_attr: float = 0.0
    volga_attr: float = 0.0
    dominant_risk: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Predefined scenario families
# ──────────────────────────────────────────────────────────────────────────────

JUMP_SHOCKS: list[Shock] = [
    Shock("jump -30%", spot_mult=0.70),
    Shock("jump -20%", spot_mult=0.80),
    Shock("jump -15%", spot_mult=0.85),
    Shock("jump -10%", spot_mult=0.90),
    Shock("jump  -5%", spot_mult=0.95),
    Shock("jump  +5%", spot_mult=1.05),
    Shock("jump +10%", spot_mult=1.10),
    Shock("jump +20%", spot_mult=1.20),
]

VOL_SHOCKS: list[Shock] = [
    # Parallel shift ± 5 vol points
    Shock("vol +5pt", vol_shift=+0.05),
    Shock("vol -5pt", vol_shift=-0.05),
    # Skew slope ± 3 vol points (log-moneyness twist; OTM puts up when negative)
    Shock("skew steepen", skew_twist=-0.03),
    Shock("skew flatten", skew_twist=+0.03),
    # Vol-of-vol / butterfly: inflate/compress wings by ± 5 vol points
    Shock("vov +5pt", vov_shift=+0.05),
    Shock("vov -5pt", vov_shift=-0.05),
]

CORRELATION_SHOCKS: list[Shock] = [
    Shock("rho +0.20", rho_shift=+0.20),
    Shock("rho -0.20", rho_shift=-0.20),
]

# Backward-compatible default scenario set (used by legacy CLI / display layer)
DEFAULT_SHOCKS: list[Shock] = [
    Shock("vol +2%", vol_shift=+0.02),
    Shock("vol -2%", vol_shift=-0.02),
    Shock("spot +5%", spot_mult=1.05),
    Shock("spot -5%", spot_mult=0.95),
    Shock("crash combo", spot_mult=0.90, vol_shift=+0.05, skew_twist=-0.10),
    Shock("skew steepen", skew_twist=-0.15),
    Shock("term steepen", term_slope=+0.50),
]


# ──────────────────────────────────────────────────────────────────────────────
# Shocked surface reconstruction
# ──────────────────────────────────────────────────────────────────────────────


def _days_from_expiry(expiry: float | int) -> int:
    value = float(expiry)
    if value <= 3.0:
        return max(int(round(value * 365.0)), 1)
    return max(int(round(value)), 1)


def _year_fraction_from_expiry(expiry: float | int) -> float:
    value = float(expiry)
    if value <= 3.0:
        return max(value, 1e-12)
    return max(value / 365.0, 1e-12)


@dataclass
class _ShockedSurfaceView:
    """Apply a deterministic stress transform on top of a fitted surface.

    The stress engine only reprices a fixed book under shocked implied vols.
    Reusing the calibrated surface and transforming vols on demand avoids the
    repeated ``build_surface`` cost that dominates long backtests.
    """

    base_surface: VolSurface
    shock: Shock
    _iv_cache: dict[tuple[float, int], float] = field(default_factory=dict)

    @cached_property
    def spot(self) -> float:
        return float(self.base_surface.spot) * float(self.shock.spot_mult)

    @property
    def r(self) -> float:
        return float(self.base_surface.r)

    @property
    def q(self) -> float:
        return float(self.base_surface.q)

    @property
    def backend(self) -> str:
        return str(self.base_surface.backend)

    @property
    def strikes(self):
        return self.base_surface.strikes

    @property
    def expiries(self):
        return self.base_surface.expiries

    @property
    def expiries_days(self):
        return self.base_surface.expiries_days

    def vol(self, strike: float, expiry_days: int) -> float:
        key = (float(strike), int(expiry_days))
        cached = self._iv_cache.get(key)
        if cached is not None:
            return cached

        t = _year_fraction_from_expiry(expiry_days)
        shocked_forward = float(self.spot) * np.exp((self.r - self.q) * t)
        base_iv = float(self.base_surface.implied_vol(float(strike), int(expiry_days)))
        x = float(np.log(max(float(strike), 1e-12) / max(shocked_forward, 1e-12)))

        rho_corr = 0.0
        if float(self.shock.rho_shift) != 0.0:
            if self.backend == "sabr":
                sabr_params = self.base_surface._sabr_params.get(int(expiry_days))
                if sabr_params is not None:
                    rho_new = float(np.clip(sabr_params.rho + float(self.shock.rho_shift), -0.999, 0.999))
                    iv_rho_new = float(
                        sabr_implied_vol(
                            shocked_forward,
                            float(strike),
                            t,
                            sabr_params.alpha,
                            rho_new,
                            sabr_params.nu,
                            sabr_params.beta,
                        )
                    )
                    iv_rho_old = float(
                        sabr_implied_vol(
                            shocked_forward,
                            float(strike),
                            t,
                            sabr_params.alpha,
                            sabr_params.rho,
                            sabr_params.nu,
                            sabr_params.beta,
                        )
                    )
                    rho_corr = iv_rho_new - iv_rho_old
                else:
                    rho_corr = 0.30 * float(self.shock.rho_shift) * x
            else:
                rho_corr = 0.30 * float(self.shock.rho_shift) * x

        shocked_iv = max(
            base_iv
            + float(self.shock.vol_shift)
            + float(self.shock.skew_twist) * x
            + float(self.shock.vov_shift) * x * x
            + rho_corr,
            1e-4,
        )
        slope_mult = max(1.0 + float(self.shock.term_slope) * t, 0.1)
        shocked_w = max(shocked_iv * shocked_iv * t * slope_mult, 1e-10)
        shocked_sigma = float(np.sqrt(shocked_w / t))
        self._iv_cache[key] = shocked_sigma
        return shocked_sigma

    def implied_vol(self, strike: float, expiry: float | int) -> float:
        return self.vol(float(strike), _days_from_expiry(expiry))


def _shocked_surface(surface: VolSurface, shock: Shock) -> _ShockedSurfaceView:
    """Return a lightweight shocked view instead of refitting a new surface."""
    return _ShockedSurfaceView(base_surface=surface, shock=shock)


# ──────────────────────────────────────────────────────────────────────────────
# Book valuation — works with BookState or Portfolio
# ──────────────────────────────────────────────────────────────────────────────


def _compute_full_greeks(
    portfolio: BookState | Portfolio,
    surface: VolSurface,
    spot: float,
) -> tuple[float, dict[str, float]]:
    """Return ``(mtm_value, greeks)`` with delta/gamma/vega/theta/vanna/volga.

    Mirrors :func:`~core.hedging.aggregate_greeks` but also includes vanna and
    volga, and accepts both :class:`~core.hedging.BookState` and
    :class:`~core.hedging.Portfolio`.  The spot hedge (``spot_position`` or
    ``hedge_shares``) is included in ``delta`` and ``value``.
    """
    hedge = portfolio.spot_position if isinstance(portfolio, BookState) else portfolio.hedge_shares
    delta = float(hedge)
    gamma = vega = theta = vanna = volga = value = 0.0
    r, q = surface.r, surface.q

    for pos in portfolio.positions:
        iv = surface.implied_vol(pos.strike, pos.expiry)
        g = bs_greeks(spot, pos.strike, pos.expiry, r, q, iv, pos.option_type)
        delta += pos.quantity * g.delta
        gamma += pos.quantity * g.gamma
        vega += pos.quantity * g.vega
        theta += pos.quantity * g.theta
        vanna += pos.quantity * g.vanna
        volga += pos.quantity * g.volga
        value += pos.quantity * g.price

    value += hedge * spot
    mtm = value + portfolio.cash
    return mtm, {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "vanna": vanna,
        "volga": volga,
        "value": value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-scenario computation
# ──────────────────────────────────────────────────────────────────────────────


def _run_scenario(
    shock: Shock,
    portfolio: BookState | Portfolio,
    surface: VolSurface,
    spot: float,
    base_mtm: float,
    base_greeks: dict[str, float],
    base_iv_atm: float,
    ref_expiry: float,
) -> StressReport:
    """Compute one :class:`StressReport` for a single :class:`Shock`."""
    shocked_surf = _shocked_surface(surface, shock)
    shocked_spot = spot * shock.spot_mult
    shocked_mtm, shocked_greeks = _compute_full_greeks(portfolio, shocked_surf, shocked_spot)

    pnl = shocked_mtm - base_mtm
    dS = shocked_spot - spot

    # ATM IV change: evaluate at the new spot level (new ATM proxy)
    shocked_iv_atm = float(shocked_surf.implied_vol(shocked_spot, ref_expiry))
    d_iv = shocked_iv_atm - base_iv_atm

    # Second-order Taylor attribution using pre-shock Greeks
    delta_attr = base_greeks["delta"] * dS
    gamma_attr = 0.5 * base_greeks["gamma"] * dS**2
    vega_attr = base_greeks["vega"] * d_iv
    vanna_attr = base_greeks["vanna"] * dS * d_iv
    volga_attr = 0.5 * base_greeks["volga"] * d_iv**2

    attr_magnitudes = {
        "delta": abs(delta_attr),
        "gamma": abs(gamma_attr),
        "vega": abs(vega_attr),
        "vanna": abs(vanna_attr),
        "volga": abs(volga_attr),
    }
    dominant = max(attr_magnitudes, key=attr_magnitudes.get) if any(attr_magnitudes.values()) else "residual"

    return StressReport(
        scenario_name=shock.name,
        spot_shock_pct=(shock.spot_mult - 1.0) * 100.0,
        vol_shock_pts=shock.vol_shift * 100.0,
        skew_twist=float(shock.skew_twist),
        term_slope=float(shock.term_slope),
        vov_shift=float(shock.vov_shift),
        rho_shift=float(shock.rho_shift),
        pnl_impact=float(pnl),
        delta_after=float(shocked_greeks["delta"]),
        gamma_after=float(shocked_greeks["gamma"]),
        vega_after=float(shocked_greeks["vega"]),
        worst_case_pnl=float(pnl),  # overwritten by run_all_scenarios
        gap_risk_pnl=float(pnl),  # hedge is always frozen; equals pnl_impact
        delta_attr=float(delta_attr),
        gamma_attr=float(gamma_attr),
        vega_attr=float(vega_attr),
        vanna_attr=float(vanna_attr),
        volga_attr=float(volga_attr),
        dominant_risk=dominant,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Combined worst-case grid
# ──────────────────────────────────────────────────────────────────────────────


def _combined_worst_case(
    portfolio: BookState | Portfolio,
    surface: VolSurface,
    spot: float,
    base_mtm: float,
    base_greeks: dict[str, float],
    base_iv_atm: float,
    ref_expiry: float,
    top_n: int = 3,
) -> list[StressReport]:
    """Brute-force grid of spot × vol shocks; return the worst ``top_n``."""
    results: list[StressReport] = []
    for spot_mult in _COMBINED_SPOT_MULTS:
        for vol_shift in _COMBINED_VOL_SHIFTS:
            shock = Shock(
                name=(f"worst: spot{(spot_mult - 1.0) * 100:+.0f}% / vol{vol_shift * 100:+.0f}pt"),
                spot_mult=spot_mult,
                vol_shift=vol_shift,
            )
            rpt = _run_scenario(
                shock,
                portfolio,
                surface,
                spot,
                base_mtm,
                base_greeks,
                base_iv_atm,
                ref_expiry,
            )
            results.append(rpt)

    results.sort(key=lambda r: r.pnl_impact)
    return results[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# Public API — new
# ──────────────────────────────────────────────────────────────────────────────


def run_all_scenarios(
    portfolio: BookState | Portfolio,
    surface: VolSurface,
    spot: float,
    rate: float,
    div_yield: float,
) -> list[StressReport]:
    """Compute stress reports for all scenario families.

    Runs in sequence:

    1. :data:`JUMP_SHOCKS`        — 8 instantaneous spot shocks.
    2. :data:`VOL_SHOCKS`         — parallel ±5 pt, skew ±3 pt, butterfly ±5 pt.
    3. :data:`CORRELATION_SHOCKS` — SABR ρ ± 0.20.
    4. Combined worst-case grid   — 8 × 5 = 40 (spot × vol) combinations;
       the worst 3 are appended to the returned list.

    ``worst_case_pnl`` in every individual report is stamped with the global
    minimum P&L from the combined grid.

    Parameters
    ----------
    portfolio:  :class:`~core.hedging.BookState` or
                :class:`~core.hedging.Portfolio` (either accepted).
    surface:    Fitted :class:`~core.surface.VolSurface`.
    spot:       Current underlying price.
    rate:       Continuously compounded risk-free rate (informational only;
                ``surface.r`` is used for all repricing).
    div_yield:  Continuous dividend yield (informational only;
                ``surface.q`` is used for all repricing).

    Returns
    -------
    ``list[StressReport]`` — 16 individual scenario reports followed by the
    3 worst combined scenario reports (``scenario_name`` starts with
    ``"worst:"``).
    """
    base_mtm, base_greeks = _compute_full_greeks(portfolio, surface, spot)

    # Reference expiry for ATM IV tracking: median of available expiries (years)
    mid_idx = max(len(surface.expiries) // 2, 0)
    ref_expiry = float(surface.expiries[mid_idx])
    base_iv_atm = float(surface.implied_vol(spot, ref_expiry))

    # 1–3: individual scenario families
    individual_reports = [
        _run_scenario(
            shock,
            portfolio,
            surface,
            spot,
            base_mtm,
            base_greeks,
            base_iv_atm,
            ref_expiry,
        )
        for shock in JUMP_SHOCKS + VOL_SHOCKS + CORRELATION_SHOCKS
    ]

    # 4: combined worst-case grid
    combined = _combined_worst_case(
        portfolio,
        surface,
        spot,
        base_mtm,
        base_greeks,
        base_iv_atm,
        ref_expiry,
    )
    global_worst_pnl = combined[0].pnl_impact if combined else 0.0

    # Stamp worst_case_pnl into every individual report
    stamped = [replace(r, worst_case_pnl=global_worst_pnl) for r in individual_reports]
    return stamped + combined


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible legacy API (used by legacy CLI and display layer)
# ──────────────────────────────────────────────────────────────────────────────


def run_shocks(
    book: BookState,
    surface: VolSurface,
    spot: float,
    shocks: list[Shock],
) -> pd.DataFrame:
    """Revalue the book under each shock and return a DataFrame.

    This is the legacy API consumed by the display layer.  Prefer
    :func:`run_all_scenarios` for programmatic access to structured results.
    """
    base_value = mark_book(book, surface, spot)
    rows = []
    for sh in shocks:
        s2 = _shocked_surface(surface, sh)
        spot2 = spot * sh.spot_mult
        v2 = mark_book(book, s2, spot2)
        rows.append(
            {
                "scenario": sh.name,
                "spot_mult": sh.spot_mult,
                "vol_shift": sh.vol_shift,
                "skew_twist": sh.skew_twist,
                "term_slope": sh.term_slope,
                "pnl": v2 - base_value,
            }
        )
    return pd.DataFrame(rows)
