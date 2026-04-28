"""Implied-volatility surface construction with SABR and spline backends.

Two interchangeable slice models are implemented:

``sabr``
    Calibrate Hagan et al. (2002) lognormal SABR implied-volatility
    approximation with fixed ``beta = 0.5``. Each expiry slice fits
    ``(alpha, rho, nu)`` by least squares using ``scipy.optimize.minimize``.

``spline``
    Fit total variance ``w(x, T) = sigma(x, T)^2 T`` as a cubic spline in
    log-moneyness ``x = log(K / F)`` per expiry. The fitted dense grid is
    projected to be monotone in ``T`` and locally repaired when the
    Gatheral-style butterfly proxy becomes negative.

The public ``VolSurface`` class exposes the requested API while preserving the
legacy helpers used elsewhere in the repository: ``build_surface(...)``,
``implied_vol(...)``, ``iv_grid(...)``, and ``local_skew(...)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from vol_surface_mm.core.pricing import bs_price, bs_vega
from vol_surface_mm.core.pricing import implied_vol as invert_implied_vol

BackendName = Literal["sabr", "spline"]
_BETA = 0.5
_EPS = 1e-12


@dataclass(frozen=True)
class SABRParams:
    """Per-slice SABR parameters with calibration metadata."""

    alpha: float
    rho: float
    nu: float
    beta: float = _BETA
    objective: float = float("nan")
    success: bool = True
    message: str = ""


@dataclass(frozen=True)
class CalendarViolation:
    """Calendar-spread arbitrage violation at a fixed strike."""

    strike: float
    expiry_days_short: int
    expiry_days_long: int
    total_variance_short: float
    total_variance_long: float
    violation: float


@dataclass(frozen=True)
class ButterflyViolation:
    """Butterfly arbitrage violation measured by call-price convexity."""

    strike: float
    expiry_days: int
    second_derivative: float
    violation: float


@dataclass(frozen=True)
class ArbitrageReport:
    """Structured arbitrage checks over a surface evaluation grid.

    ``calendar`` and ``butterfly`` store per-cell violations. For backward
    compatibility with older call sites, ``report[\"calendar_violations\"]`` and
    ``report[\"butterfly_violations\"]`` return violation counts.
    """

    calendar: tuple[CalendarViolation, ...] = ()
    butterfly: tuple[ButterflyViolation, ...] = ()
    strikes: tuple[float, ...] = ()
    expiries_days: tuple[int, ...] = ()

    @property
    def calendar_count(self) -> int:
        return len(self.calendar)

    @property
    def butterfly_count(self) -> int:
        return len(self.butterfly)

    def __getitem__(self, key: str) -> int:
        if key == "calendar_violations":
            return self.calendar_count
        if key == "butterfly_violations":
            return self.butterfly_count
        raise KeyError(key)


@dataclass(frozen=True)
class _SliceData:
    """Normalised market data for one expiry slice."""

    expiry_days: int
    expiry_years: float
    forward: float
    strikes: np.ndarray
    x: np.ndarray
    iv: np.ndarray
    total_variance: np.ndarray


@dataclass
class _SplineSlice:
    """Spline backend representation on a dense log-moneyness grid."""

    expiry_days: int
    expiry_years: float
    forward: float
    x_grid: np.ndarray
    total_variance_grid: np.ndarray
    spline: CubicSpline


def _year_fraction_from_expiry(expiry: float | int) -> float:
    """Interpret small values as years and larger ones as DTE days."""
    value = float(expiry)
    if value <= 3.0:
        return max(value, _EPS)
    return max(value / 365.0, _EPS)


def _days_from_expiry(expiry: float | int) -> int:
    return max(int(round(_year_fraction_from_expiry(expiry) * 365.0)), 1)


def _forward(spot: float, rate: float, dividend_yield: float, t: float) -> float:
    return float(spot * np.exp((rate - dividend_yield) * t))


def _unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and average duplicate x points."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size == x_sorted.size:
        return x_sorted, y_sorted
    accum = np.zeros_like(unique_x)
    counts = np.zeros_like(unique_x)
    np.add.at(accum, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    return unique_x, accum / np.maximum(counts, 1.0)


def sabr_implied_vol(
    forward: float,
    strike: float | np.ndarray,
    expiry_years: float,
    alpha: float,
    rho: float,
    nu: float,
    beta: float = _BETA,
) -> np.ndarray:
    """Hagan et al. (2002) lognormal SABR implied-volatility approximation."""
    strike_array = np.asarray(strike, dtype=np.float64)
    f = max(float(forward), _EPS)
    k = np.maximum(strike_array, _EPS)
    t = max(float(expiry_years), _EPS)
    a = max(float(alpha), _EPS)
    r = float(np.clip(rho, -0.999, 0.999))
    n = max(float(nu), _EPS)

    fk = f * k
    one_minus_beta = 1.0 - beta
    log_fk = np.log(f / k)
    fk_beta = np.power(fk, 0.5 * one_minus_beta)
    z = (n / a) * fk_beta * log_fk

    sqrt_term = np.sqrt(np.maximum(1.0 - 2.0 * r * z + z * z, _EPS))
    numerator = sqrt_term + z - r
    denominator = 1.0 - r
    x_z = np.log(np.maximum(numerator / np.maximum(denominator, _EPS), _EPS))

    log_fk_sq = log_fk * log_fk
    log_fk_four = log_fk_sq * log_fk_sq
    denom = fk_beta * (
        1.0
        + (one_minus_beta * one_minus_beta / 24.0) * log_fk_sq
        + (one_minus_beta**4 / 1920.0) * log_fk_four
    )
    denom = np.maximum(denom, _EPS)

    correction = (
        ((one_minus_beta**2) / 24.0) * (a * a / np.maximum(fk**one_minus_beta, _EPS))
        + 0.25 * r * beta * n * a / np.maximum(fk_beta, _EPS)
        + ((2.0 - 3.0 * r * r) / 24.0) * n * n
    ) * t

    use_atm = np.abs(log_fk) < 1e-8
    ratio = np.divide(z, x_z, out=np.ones_like(z), where=np.abs(x_z) > 1e-10)
    sigma = (a / denom) * ratio * (1.0 + correction)

    if np.any(use_atm):
        f_beta = max(f**one_minus_beta, _EPS)
        atm_sigma = (a / f_beta) * (
            1.0
            + (
                ((one_minus_beta**2) / 24.0) * (a * a / max(f ** (2.0 * one_minus_beta), _EPS))
                + 0.25 * r * beta * n * a / max(f_beta, _EPS)
                + ((2.0 - 3.0 * r * r) / 24.0) * n * n
            )
            * t
        )
        sigma = np.where(use_atm, atm_sigma, sigma)
    return np.maximum(sigma, 1e-8)


def _gatheral_butterfly_proxy(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the spline butterfly proxy on a dense log-moneyness grid."""
    safe_w = np.maximum(w, 1e-8)
    dw = np.gradient(safe_w, x, edge_order=2)
    d2w = np.gradient(dw, x, edge_order=2)
    term = 0.25 * (0.25 - (1.0 / safe_w) + (x / safe_w)) * (dw * dw)
    return d2w - term


def _second_derivative_nonuniform(
    x0: float,
    x1: float,
    x2: float,
    y0: float,
    y1: float,
    y2: float,
) -> float:
    """Quadratic-interpolation second derivative on a nonuniform grid."""
    return 2.0 * (y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2)) + y2 / ((x2 - x0) * (x2 - x1)))


def _build_dense_x_grid(slices: Sequence[_SliceData], n_points: int = 121) -> np.ndarray:
    mins = [float(np.min(s.x)) for s in slices]
    maxs = [float(np.max(s.x)) for s in slices]
    return np.linspace(min(mins), max(maxs), n_points, dtype=np.float64)


def _project_calendar_monotone(w_matrix: np.ndarray) -> np.ndarray:
    """Project total variance to be non-decreasing in expiry at fixed x."""
    return np.maximum.accumulate(w_matrix, axis=0)


def _repair_butterfly_slice(x_grid: np.ndarray, w_grid: np.ndarray) -> np.ndarray:
    """Smooth local total variance where the Gatheral-style proxy is negative."""
    repaired = np.maximum(w_grid.copy(), 1e-8)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for _ in range(12):
        proxy = _gatheral_butterfly_proxy(x_grid, repaired)
        mask = proxy < -1e-6
        if not np.any(mask):
            break
        smoothed = np.convolve(repaired, kernel, mode="same")
        repaired[mask] = 0.5 * repaired[mask] + 0.5 * smoothed[mask]
        repaired = np.maximum(repaired, 1e-8)
    return repaired


def _repair_call_convexity_slice(
    strikes: np.ndarray,
    w_grid: np.ndarray,
    expiry_years: float,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> tuple[np.ndarray, bool]:
    """Lower local total variance until call-price convexity is non-negative."""
    repaired = np.maximum(w_grid.copy(), 1e-8)
    changed = False
    for _ in range(12):
        sigma = np.sqrt(np.maximum(repaired / max(expiry_years, _EPS), 1e-10))
        calls = np.asarray(
            bs_price(spot, strikes, expiry_years, rate, dividend_yield, sigma, "call"),
            dtype=np.float64,
        )
        second = np.zeros_like(calls)
        for idx in range(1, strikes.size - 1):
            second[idx] = _second_derivative_nonuniform(
                float(strikes[idx - 1]),
                float(strikes[idx]),
                float(strikes[idx + 1]),
                float(calls[idx - 1]),
                float(calls[idx]),
                float(calls[idx + 1]),
            )
        bad = second < -1e-8
        bad[0] = False
        bad[-1] = False
        if not np.any(bad):
            break

        updated = repaired.copy()
        for idx in np.where(bad)[0]:
            left = float(strikes[idx - 1])
            mid = float(strikes[idx])
            right = float(strikes[idx + 1])
            weight = (mid - left) / max(right - left, 1e-12)
            convex_cap = float(calls[idx - 1] + (calls[idx + 1] - calls[idx - 1]) * weight)
            target_call = min(float(calls[idx]), convex_cap - 1e-8)
            intrinsic = max(
                spot * np.exp(-dividend_yield * expiry_years) - mid * np.exp(-rate * expiry_years),
                0.0,
            )
            target_call = max(target_call, intrinsic + 1e-8)
            target_sigma = invert_implied_vol(
                target_call,
                spot,
                mid,
                expiry_years,
                rate,
                dividend_yield,
                "call",
            )
            if not np.isfinite(target_sigma):
                vega = bs_vega(
                    spot,
                    mid,
                    expiry_years,
                    rate,
                    dividend_yield,
                    float(sigma[idx]),
                )
                target_sigma = max(
                    float(sigma[idx]) - max((float(calls[idx]) - target_call) / max(vega, 1e-8), 0.0), 1e-4
                )
            updated[idx] = min(updated[idx], max(target_sigma * target_sigma * expiry_years, 1e-8))

        repaired = np.maximum(updated, 1e-8)
        changed = True
    return repaired, changed


class VolSurface:
    """Full implied-volatility surface with interchangeable SABR/spline slices."""

    def __init__(
        self,
        chain_df: pd.DataFrame,
        spot: float,
        rate: float,
        dividend_yield: float,
        backend: BackendName = "sabr",
    ) -> None:
        self.chain_df = chain_df.copy()
        self.spot = float(spot)
        self.r = float(rate)
        self.q = float(dividend_yield)
        self.backend: BackendName = backend.lower()  # type: ignore[assignment]
        if self.backend not in {"sabr", "spline"}:
            raise ValueError("backend must be 'sabr' or 'spline'")

        self.expiries_days: np.ndarray = np.array([], dtype=np.int64)
        self.expiries: np.ndarray = np.array([], dtype=np.float64)
        self.strikes: np.ndarray = np.array([], dtype=np.float64)
        self.forward: float = self.spot

        self._slice_data: list[_SliceData] = []
        self._sabr_params: dict[int, SABRParams] = {}
        self._spline_slices: dict[int, _SplineSlice] = {}
        self._last_report: ArbitrageReport | None = None

    @property
    def sabr_params(self) -> dict[int, SABRParams]:
        return dict(self._sabr_params)

    def fit(self) -> None:
        """Normalise the input chain and fit the selected backend."""
        self._slice_data = self._prepare_chain(self.chain_df)
        if not self._slice_data:
            raise ValueError("no valid option data available for surface fit")

        self.expiries_days = np.array([s.expiry_days for s in self._slice_data], dtype=np.int64)
        self.expiries = np.array([s.expiry_years for s in self._slice_data], dtype=np.float64)
        self.strikes = np.array(sorted(self.chain_df["strike"].astype(float).unique()), dtype=np.float64)
        self.forward = _forward(self.spot, self.r, self.q, float(self.expiries[0]))

        self._sabr_params.clear()
        self._spline_slices.clear()

        if self.backend == "sabr":
            self._fit_sabr_backend()
        else:
            self._fit_spline_backend()

        self._last_report = self.arbitrage_report()

    def vol(self, strike: float, expiry_days: int) -> float:
        """Return the implied volatility at ``(strike, expiry_days)``."""
        t = _year_fraction_from_expiry(expiry_days)
        k = max(float(strike), _EPS)
        if self.backend == "sabr":
            return self._vol_sabr(k, t)
        return self._vol_spline(k, t)

    def implied_vol(self, strike: float, expiry: float | int) -> float:
        """Compatibility alias that accepts either years or DTE days."""
        return self.vol(strike, _days_from_expiry(expiry))

    def total_variance(self, strike: float, expiry_days: int) -> float:
        """Return total variance ``sigma(K, T)^2 T``."""
        t = _year_fraction_from_expiry(expiry_days)
        sigma = self.vol(strike, expiry_days)
        return float(max(sigma * sigma * t, 1e-10))

    def arbitrage_report(
        self,
        strikes: Sequence[float] | None = None,
        expiries: Sequence[float | int] | None = None,
    ) -> ArbitrageReport:
        """Run calendar-spread and butterfly checks on a grid."""
        use_strikes = tuple(float(k) for k in (strikes if strikes is not None else self.strikes))
        use_expiries = tuple(
            _days_from_expiry(t) for t in (expiries if expiries is not None else self.expiries_days)
        )
        calendar = check_calendar_spread(self, use_strikes, use_expiries)
        butterfly = check_butterfly(self, use_strikes, use_expiries)
        report = ArbitrageReport(
            calendar=tuple(calendar),
            butterfly=tuple(butterfly),
            strikes=use_strikes,
            expiries_days=use_expiries,
        )
        self._last_report = report
        return report

    def to_grid(
        self,
        strikes: Sequence[float] | None = None,
        expiries: Sequence[float | int] | None = None,
    ) -> pd.DataFrame:
        """Return a strikes x expiries heatmap of implied volatilities."""
        use_strikes = list(strikes if strikes is not None else self.strikes)
        use_expiries = [
            _days_from_expiry(t) for t in (expiries if expiries is not None else self.expiries_days)
        ]
        data = np.zeros((len(use_strikes), len(use_expiries)), dtype=np.float64)
        for i, strike in enumerate(use_strikes):
            for j, expiry in enumerate(use_expiries):
                data[i, j] = self.vol(float(strike), int(expiry))
        return pd.DataFrame(data, index=use_strikes, columns=use_expiries)

    def iv_grid(
        self,
        strikes: Sequence[float],
        expiries: Sequence[float | int],
    ) -> pd.DataFrame:
        """Legacy helper returning an expiry x strike heatmap."""
        use_expiries = [_days_from_expiry(t) for t in expiries]
        data = np.zeros((len(use_expiries), len(strikes)), dtype=np.float64)
        for i, expiry in enumerate(use_expiries):
            for j, strike in enumerate(strikes):
                data[i, j] = self.vol(float(strike), int(expiry))
        return pd.DataFrame(data, index=[e / 365.0 for e in use_expiries], columns=list(strikes))

    def _prepare_chain(self, chain_df: pd.DataFrame) -> list[_SliceData]:
        required = {"strike", "expiry"}
        if not required.issubset(chain_df.columns):
            missing = required - set(chain_df.columns)
            raise ValueError(f"chain is missing columns: {missing}")

        data = chain_df.copy()
        data["strike"] = data["strike"].astype(float)
        data["expiry_years"] = data["expiry"].astype(float).map(_year_fraction_from_expiry)
        data["expiry_days"] = data["expiry_years"].map(lambda x: max(int(round(x * 365.0)), 1))

        if "iv" not in data.columns:
            if "mid_price" not in data.columns or "type" not in data.columns:
                raise ValueError("chain must include either 'iv' or both 'mid_price' and 'type'")
            data["iv"] = data.apply(
                lambda row: invert_implied_vol(
                    float(row["mid_price"]),
                    self.spot,
                    float(row["strike"]),
                    float(row["expiry_years"]),
                    self.r,
                    self.q,
                    str(row["type"]),
                ),
                axis=1,
            )

        data["iv"] = data["iv"].astype(float)
        data = data[np.isfinite(data["iv"]) & (data["iv"] > 0.0)]
        if data.empty:
            return []

        grouped = (
            data.groupby(["expiry_days", "expiry_years", "strike"], as_index=False)["iv"]
            .mean()
            .sort_values(["expiry_years", "strike"])
        )

        slices: list[_SliceData] = []
        for (expiry_days, expiry_years), slice_df in grouped.groupby(["expiry_days", "expiry_years"]):
            strikes = slice_df["strike"].to_numpy(dtype=np.float64)
            forward = _forward(self.spot, self.r, self.q, float(expiry_years))
            x = np.log(np.maximum(strikes, _EPS) / max(forward, _EPS))
            iv = slice_df["iv"].to_numpy(dtype=np.float64)
            total_variance = np.maximum(iv * iv * float(expiry_years), 1e-10)
            x, total_variance = _unique_sorted_xy(x, total_variance)
            iv = np.sqrt(np.maximum(total_variance / max(float(expiry_years), _EPS), 1e-10))
            strikes = forward * np.exp(x)
            slices.append(
                _SliceData(
                    expiry_days=int(expiry_days),
                    expiry_years=float(expiry_years),
                    forward=forward,
                    strikes=strikes,
                    x=x,
                    iv=iv,
                    total_variance=total_variance,
                )
            )
        slices.sort(key=lambda s: s.expiry_years)
        return slices

    def _fit_sabr_backend(self) -> None:
        for slice_data in self._slice_data:
            self._sabr_params[slice_data.expiry_days] = self._calibrate_sabr_slice(slice_data)

    def _fit_spline_backend(self) -> None:
        strike_grid = self.strikes.astype(np.float64)
        adjusted = np.zeros((len(self._slice_data), strike_grid.size), dtype=np.float64)

        for idx, slice_data in enumerate(self._slice_data):
            adjusted[idx] = np.maximum(
                np.interp(strike_grid, slice_data.strikes, slice_data.total_variance),
                1e-8,
            )

        adjusted = _project_calendar_monotone(adjusted)
        for _ in range(4):
            for idx in range(adjusted.shape[0]):
                x_nodes = np.log(np.maximum(strike_grid, _EPS) / max(self._slice_data[idx].forward, _EPS))
                repaired = _repair_butterfly_slice(x_nodes, adjusted[idx])
                repaired, _ = _repair_call_convexity_slice(
                    strike_grid,
                    repaired,
                    self._slice_data[idx].expiry_years,
                    self.spot,
                    self.r,
                    self.q,
                )
                adjusted[idx] = repaired
            adjusted = _project_calendar_monotone(adjusted)

        for idx, slice_data in enumerate(self._slice_data):
            x_nodes = np.log(np.maximum(strike_grid, _EPS) / max(slice_data.forward, _EPS))
            spline = CubicSpline(x_nodes, adjusted[idx], bc_type="natural", extrapolate=True)
            self._spline_slices[slice_data.expiry_days] = _SplineSlice(
                expiry_days=slice_data.expiry_days,
                expiry_years=slice_data.expiry_years,
                forward=slice_data.forward,
                x_grid=x_nodes.copy(),
                total_variance_grid=adjusted[idx].copy(),
                spline=spline,
            )

    def _calibrate_sabr_slice(self, slice_data: _SliceData) -> SABRParams:
        if slice_data.iv.size < 3:
            atm_idx = int(np.argmin(np.abs(slice_data.x)))
            alpha = max(slice_data.iv[atm_idx] * slice_data.forward ** (1.0 - _BETA), 1e-4)
            warnings.warn(
                f"SABR slice {slice_data.expiry_days}D has too few strikes; using ATM fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return SABRParams(
                alpha=alpha, rho=0.0, nu=0.5, success=False, message="ATM fallback", objective=float("nan")
            )

        atm_idx = int(np.argmin(np.abs(slice_data.x)))
        atm_iv = float(slice_data.iv[atm_idx])
        alpha0 = max(atm_iv * slice_data.forward ** (1.0 - _BETA), 1e-4)
        x0 = np.array([alpha0, -0.1, 0.5], dtype=np.float64)
        bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]

        def objective(theta: np.ndarray) -> float:
            alpha, rho, nu = theta
            model = sabr_implied_vol(
                slice_data.forward,
                slice_data.strikes,
                slice_data.expiry_years,
                alpha,
                rho,
                nu,
            )
            diff = model - slice_data.iv
            return float(np.mean(diff * diff))

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        alpha, rho, nu = result.x
        if not result.success:
            warnings.warn(
                (f"SABR calibration for {slice_data.expiry_days}D did not converge: {result.message}"),
                RuntimeWarning,
                stacklevel=2,
            )
        return SABRParams(
            alpha=float(alpha),
            rho=float(rho),
            nu=float(nu),
            objective=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
        )

    def _vol_sabr(self, strike: float, expiry_years: float) -> float:
        t = float(np.clip(expiry_years, self.expiries[0], self.expiries[-1]))
        params = self._interpolated_sabr_params(t)
        forward = _forward(self.spot, self.r, self.q, t)
        return float(sabr_implied_vol(forward, strike, t, params.alpha, params.rho, params.nu)[()])

    def _interpolated_sabr_params(self, expiry_years: float) -> SABRParams:
        if len(self.expiries) == 1:
            return next(iter(self._sabr_params.values()))
        alpha = np.array([p.alpha for p in self._sabr_params.values()], dtype=np.float64)
        rho = np.array([p.rho for p in self._sabr_params.values()], dtype=np.float64)
        nu = np.array([p.nu for p in self._sabr_params.values()], dtype=np.float64)
        return SABRParams(
            alpha=float(np.interp(expiry_years, self.expiries, alpha)),
            rho=float(np.interp(expiry_years, self.expiries, rho)),
            nu=float(np.interp(expiry_years, self.expiries, nu)),
        )

    def _vol_spline(self, strike: float, expiry_years: float) -> float:
        t = float(np.clip(expiry_years, self.expiries[0], self.expiries[-1]))
        if len(self.expiries) == 1:
            slice_model = next(iter(self._spline_slices.values()))
            x = np.log(max(strike, _EPS) / max(_forward(self.spot, self.r, self.q, t), _EPS))
            w = float(max(slice_model.spline(x), 1e-10))
            return float(np.sqrt(w / max(t, _EPS)))

        x = np.log(max(strike, _EPS) / max(_forward(self.spot, self.r, self.q, t), _EPS))
        w_per_slice = np.array(
            [float(max(slice_model.spline(x), 1e-10)) for slice_model in self._spline_slices.values()],
            dtype=np.float64,
        )
        w = float(np.interp(t, self.expiries, w_per_slice))
        return float(np.sqrt(max(w, 1e-10) / max(t, _EPS)))


def check_calendar_spread(
    surface: VolSurface,
    strikes: Sequence[float] | None = None,
    expiries_days: Sequence[int] | None = None,
    tolerance: float = 1e-8,
) -> list[CalendarViolation]:
    """Check monotonicity of total variance in expiry at fixed strike."""
    use_strikes = list(strikes if strikes is not None else surface.strikes)
    use_expiries = sorted(
        int(e) for e in (expiries_days if expiries_days is not None else surface.expiries_days)
    )
    violations: list[CalendarViolation] = []
    for strike in use_strikes:
        total_vars = [surface.total_variance(float(strike), expiry) for expiry in use_expiries]
        for idx in range(len(use_expiries) - 1):
            near_expiry = use_expiries[idx]
            far_expiry = use_expiries[idx + 1]
            near_w = total_vars[idx]
            far_w = total_vars[idx + 1]
            if near_w > far_w + tolerance:
                violations.append(
                    CalendarViolation(
                        strike=float(strike),
                        expiry_days_short=near_expiry,
                        expiry_days_long=far_expiry,
                        total_variance_short=float(near_w),
                        total_variance_long=float(far_w),
                        violation=float(near_w - far_w),
                    )
                )
    return violations


def check_butterfly(
    surface: VolSurface,
    strikes: Sequence[float] | None = None,
    expiries_days: Sequence[int] | None = None,
    tolerance: float = 1e-8,
) -> list[ButterflyViolation]:
    """Check convexity of call price with respect to strike."""
    use_strikes = np.array(list(strikes if strikes is not None else surface.strikes), dtype=np.float64)
    use_strikes = np.sort(use_strikes)
    use_expiries = [int(e) for e in (expiries_days if expiries_days is not None else surface.expiries_days)]
    violations: list[ButterflyViolation] = []
    if use_strikes.size < 3:
        return violations

    for expiry_days in use_expiries:
        t = _year_fraction_from_expiry(expiry_days)
        calls = np.array(
            [
                float(
                    bs_price(
                        surface.spot,
                        strike,
                        t,
                        surface.r,
                        surface.q,
                        surface.vol(float(strike), expiry_days),
                        "call",
                    )
                )
                for strike in use_strikes
            ],
            dtype=np.float64,
        )
        for idx in range(1, use_strikes.size - 1):
            second = _second_derivative_nonuniform(
                float(use_strikes[idx - 1]),
                float(use_strikes[idx]),
                float(use_strikes[idx + 1]),
                float(calls[idx - 1]),
                float(calls[idx]),
                float(calls[idx + 1]),
            )
            if second < -tolerance:
                violations.append(
                    ButterflyViolation(
                        strike=float(use_strikes[idx]),
                        expiry_days=int(expiry_days),
                        second_derivative=float(second),
                        violation=float(-second),
                    )
                )
    return violations


def build_surface(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    cfg: Any | None = None,
    backend: BackendName | None = None,
) -> VolSurface:
    """Repository compatibility wrapper for constructing and fitting a surface."""
    chosen_backend = backend or getattr(cfg, "backend", "sabr")
    surface = VolSurface(chain, spot, r, q, backend=chosen_backend)
    surface.fit()
    return surface


def local_skew(surface: VolSurface, strike: float, expiry: float | int, bump: float = 1e-3) -> float:
    """Return ``d sigma / d log(K)`` via a central difference bump."""
    k_up = float(strike) * np.exp(bump)
    k_dn = float(strike) * np.exp(-bump)
    return (surface.implied_vol(k_up, expiry) - surface.implied_vol(k_dn, expiry)) / (2.0 * bump)
