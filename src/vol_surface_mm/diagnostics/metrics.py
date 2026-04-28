"""Risk and performance metrics — realized/implied variance, residual
attribution diagnostics, and vol-surface quality monitoring.

Public symbols (used by legacy CLI)
--------------------------------
``VarianceTracker``, ``ResidualAttributionTracker``, ``rv_iv_dislocation``

Structured types
----------------
``VRPStats``, ``ResidualAttributionStats``, ``SurfaceStabilityStats``,
``DiagnosticsReport``, ``SurfaceMonitor``, ``build_diagnostics_report``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from scipy.stats import kstest as _kstest
from scipy.stats import kurtosis as _sp_kurtosis
from scipy.stats import skew as _sp_skew

from vol_surface_mm.core.surface import VolSurface

logger = logging.getLogger(__name__)

# ─── Reference constants ─────────────────────────────────────────────────────

# Fixed log-moneyness grid used for L2 surface-change diagnostics.
_REF_X: np.ndarray = np.linspace(-0.20, 0.20, 21, dtype=np.float64)

# N⁻¹(0.75) — used to approximate 25-delta strikes from ATM vol.
_DELTA25_Z: float = 0.6745


# ─── Structured output types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class VRPStats:
    """Rolling variance-risk-premium statistics.

    Attributes
    ----------
    mean:      Rolling-window mean VRP (IV² − RV, annualised).
    std:       Rolling-window standard deviation.
    lower_95:  Mean − 1.96 × standard error.
    upper_95:  Mean + 1.96 × standard error.
    n_obs:     Number of observations in the window.
    """

    mean: float
    std: float
    lower_95: float
    upper_95: float
    n_obs: int


@dataclass(frozen=True)
class ResidualAttributionStats:
    """Distributional statistics for the per-step residual-attribution series.

    The residual is the portion of realised step P&L *not* explained by the
    chosen attribution model (delta/gamma/theta/vega).  It is the output of
    :attr:`~core.hedging.PnLReport.residual_pnl` for each step and captures
    higher-order and model-misspecification effects.  It is **not** a pure
    hedging-quality metric.

    Attributes
    ----------
    mean:            Mean residual P&L.
    std:             Standard deviation.
    skewness:        Third-moment skewness.
    excess_kurtosis: Fisher's excess kurtosis (normal = 0).
    ks_pvalue:       KS-test p-value against N(0, 1) on standardised residuals.
    n_obs:           Number of observed steps.
    """

    mean: float
    std: float
    skewness: float
    excess_kurtosis: float
    ks_pvalue: float
    n_obs: int


@dataclass(frozen=True)
class SurfaceStabilityStats:
    """Vol-surface quality and stability statistics.

    Attributes
    ----------
    atm_vol_stability:      Rolling std of ATM implied vol.
    skew_stability:         Rolling std of 25-delta risk-reversal.
    fly_stability:          Rolling std of 25-delta butterfly.
    surface_rmse:           RMSE between fitted surface and market IVs.
    surface_daily_l2_change: L2 norm of Δσ(K, T) on the reference grid.
    """

    atm_vol_stability: float
    skew_stability: float
    fly_stability: float
    surface_rmse: float
    surface_daily_l2_change: float


@dataclass(frozen=True)
class DiagnosticsReport:
    """Point-in-time diagnostics snapshot.

    Produced by :func:`build_diagnostics_report`.
    """

    timestamp: datetime
    variance_risk_premium: float
    vrp_30d_mean: float
    residual_attribution_mean: float
    residual_attribution_std: float
    residual_attribution_kurtosis: float
    surface_rmse: float
    atm_vol_stability: float
    skew_stability: float


# ─── VarianceTracker ─────────────────────────────────────────────────────────


@dataclass
class VarianceTracker:
    """Rolling realized variance estimator with VRP tracking.

    Backward-compatible usage::

        tracker = VarianceTracker(window=20)
        tracker.update(log_return)
        rv = tracker.realized_vol_annual(dt)

    Extended usage — pass ``iv_atm`` and optionally ``vega`` to populate VRP
    history::

        tracker.update(log_return, iv_atm=surface.implied_vol(spot, t))
        stats = tracker.vrp_rolling_stats(window=30, dt=dt)
    """

    window: int = 20
    _returns: list[float] = field(default_factory=list)
    _iv_atm_history: list[float] = field(default_factory=list)
    _vrp_history: list[float] = field(default_factory=list)
    _vega_history: list[float] = field(default_factory=list)
    _dt_history: list[float] = field(default_factory=list)

    def update(
        self,
        ret: float,
        iv_atm: float = 0.0,
        vega: float = 0.0,
        dt: float = 1.0 / 252.0,
    ) -> None:
        """Update with a new log-return and optionally the ATM implied vol.

        Parameters
        ----------
        ret:    log(S_t / S_{t-1}).
        iv_atm: ATM implied volatility (annualised, decimal).  When > 0 the
                VRP history is updated.
        vega:   Net portfolio vega used for variance-capture P&L accumulation.
        dt:     Step length in years.
        """
        self._returns.append(float(ret))
        if len(self._returns) > self.window:
            self._returns.pop(0)

        if iv_atm > 0.0:
            rv = self.realized_variance_annual(dt)
            self._iv_atm_history.append(float(iv_atm))
            self._vrp_history.append(float(iv_atm**2 - rv))
            self._vega_history.append(float(vega))
            self._dt_history.append(float(dt))

    # ── Existing methods (unchanged) ──────────────────────────────────────

    def realized_variance_annual(self, dt: float) -> float:
        """Rolling sample variance of log-returns, annualised by ``1/dt``."""
        if len(self._returns) < 2:
            return 0.0
        return float(np.var(self._returns, ddof=1) / max(dt, 1e-12))

    def realized_vol_annual(self, dt: float) -> float:
        """Square root of :meth:`realized_variance_annual`."""
        return float(np.sqrt(self.realized_variance_annual(dt)))

    # ── New methods ───────────────────────────────────────────────────────

    def variance_risk_premium(self, dt: float) -> float:
        """Current VRP = IV_atm² − RV (annualised, variance space).

        Positive means implied variance trades above realised — the classic
        short-vol risk premium.
        """
        if not self._iv_atm_history:
            return 0.0
        return float(self._iv_atm_history[-1] ** 2 - self.realized_variance_annual(dt))

    def vrp_rolling_stats(self, window: int = 30, dt: float = 1.0 / 252.0) -> VRPStats:
        """Rolling ``window``-point VRP mean with 95 % confidence band.

        If fewer than ``window`` observations exist, all available data are
        used.
        """
        h = np.array(self._vrp_history, dtype=np.float64)
        recent = h[-window:] if len(h) >= window else h
        n = int(recent.size)
        if n == 0:
            return VRPStats(mean=0.0, std=0.0, lower_95=0.0, upper_95=0.0, n_obs=0)
        mean = float(np.mean(recent))
        std = float(np.std(recent, ddof=1)) if n > 1 else 0.0
        stderr = std / np.sqrt(n)
        return VRPStats(
            mean=mean,
            std=std,
            lower_95=mean - 1.96 * stderr,
            upper_95=mean + 1.96 * stderr,
            n_obs=n,
        )

    def variance_capture_pnl(self, dt: float = 1.0 / 252.0) -> float:
        """Cumulative variance-capture P&L: Σ (IV²_t − RV_t) · vega_t · Δt."""
        if not self._vrp_history:
            return 0.0
        vrp = np.array(self._vrp_history, dtype=np.float64)
        vega = np.array(self._vega_history, dtype=np.float64)
        dts = np.array(self._dt_history, dtype=np.float64)
        return float(np.sum(vrp * vega * dts))


# ─── ResidualAttributionTracker ──────────────────────────────────────────────


@dataclass
class ResidualAttributionTracker:
    """Cumulative residual-attribution P&L with distributional diagnostics.

    This measures the portion of realised step P&L that the chosen attribution
    model (delta/gamma/theta/vega) does *not* explain — i.e. higher-order and
    model-misspecification residual.  It is **not** a pure hedging-quality
    metric; a well-hedged book can still produce sizeable residuals if, for
    example, vega/vanna/volga effects or discretisation error are material.

    Feed it ``PnLReport.residual_pnl`` each step::

        tracker = ResidualAttributionTracker()
        tracker.update(pnl_report.residual_pnl)
        print(tracker.residual_attribution)
        stats = tracker.residual_stats()
    """

    cumulative: float = 0.0
    _residuals: list[float] = field(default_factory=list)

    def update(self, residual_step: float) -> None:
        """Record one step's residual P&L from the attribution model."""
        r = float(residual_step)
        self.cumulative += r
        self._residuals.append(r)

    @property
    def residual_attribution(self) -> float:
        """Cumulative residual P&L across all recorded steps."""
        return float(self.cumulative)

    def residual_stats(self) -> ResidualAttributionStats:
        """Distributional statistics for the per-step residual series."""
        errs = np.array(self._residuals, dtype=np.float64)
        n = int(errs.size)
        if n == 0:
            return ResidualAttributionStats(
                mean=0.0,
                std=0.0,
                skewness=0.0,
                excess_kurtosis=0.0,
                ks_pvalue=float("nan"),
                n_obs=0,
            )

        mean = float(np.mean(errs))
        std = float(np.std(errs, ddof=1)) if n > 1 else 0.0
        skewness = float(_sp_skew(errs)) if n >= 3 else 0.0
        excess_kurt = float(_sp_kurtosis(errs, fisher=True)) if n >= 4 else 0.0

        if std > 0.0 and n >= 4:
            z = (errs - mean) / std
            _, pvalue = _kstest(z, "norm")
        else:
            pvalue = float("nan")

        return ResidualAttributionStats(
            mean=mean,
            std=std,
            skewness=skewness,
            excess_kurtosis=excess_kurt,
            ks_pvalue=float(pvalue),
            n_obs=n,
        )

    def histogram_data(self, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Residual histogram.

        Returns
        -------
        bin_centers : ndarray of shape ``(bins,)``
        counts      : ndarray of shape ``(bins,)`` with float counts
        """
        errs = np.array(self._residuals, dtype=np.float64)
        if errs.size == 0:
            return np.zeros(bins, dtype=np.float64), np.zeros(bins, dtype=np.float64)
        counts, edges = np.histogram(errs, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts.astype(np.float64)


# ─── rv_iv_dislocation (unchanged) ───────────────────────────────────────────


def rv_iv_dislocation(rv_annual: float, iv_annual: float) -> float:
    """Variance-space difference: RV² − IV² (annualised)."""
    return float(rv_annual**2 - iv_annual**2)


# ─── Surface quality helpers ─────────────────────────────────────────────────


def _calibration_rmse(surface: VolSurface) -> float:
    """RMSE between fitted surface IVs and market IVs at calibration nodes."""
    sq_sum = 0.0
    n = 0
    for slc in surface._slice_data:
        for k, mkt_iv in zip(slc.strikes, slc.iv, strict=True):
            fit_iv = surface.vol(float(k), slc.expiry_days)
            sq_sum += (fit_iv - mkt_iv) ** 2
            n += 1
    return float(np.sqrt(sq_sum / n)) if n > 0 else 0.0


def _pick_ref_expiry(surface: VolSurface) -> float:
    """Nearest available expiry to 45 calendar days — a stable reference."""
    target = 45.0 / 365.0
    idx = int(np.argmin(np.abs(surface.expiries - target)))
    return float(surface.expiries[idx])


def _rr_and_fly(surface: VolSurface, spot: float, expiry: float) -> tuple[float, float]:
    """Approximate 25-delta risk-reversal and butterfly.

    Uses the log-moneyness proxy: K_{25c/p} ≈ F · exp(± z_{0.25} · σ · √T)
    where z_{0.25} = N⁻¹(0.75) ≈ 0.6745.
    """
    sigma_atm = surface.implied_vol(spot, expiry)
    sqrt_t = max(float(np.sqrt(expiry)), 1e-6)
    bump = _DELTA25_Z * sigma_atm * sqrt_t
    iv_25c = surface.implied_vol(spot * np.exp(+bump), expiry)
    iv_25p = surface.implied_vol(spot * np.exp(-bump), expiry)
    rr = float(iv_25c - iv_25p)
    fly = float(0.5 * (iv_25c + iv_25p) - sigma_atm)
    return rr, fly


# ─── SurfaceMonitor ──────────────────────────────────────────────────────────


class SurfaceMonitor:
    """Tracks vol-surface quality and stability step-by-step.

    Call :meth:`update` once per simulation step with the current surface and
    spot price.  Call :meth:`stability_stats` at any time to get a
    :class:`SurfaceStabilityStats` snapshot.

    Parameters
    ----------
    stability_window: Default rolling window for stability calculations.
    """

    def __init__(self, stability_window: int = 30) -> None:
        self._stability_window = stability_window
        self._atm_vol_history: list[float] = []
        self._rr_history: list[float] = []
        self._fly_history: list[float] = []
        self._prev_iv_vec: np.ndarray | None = None
        self._l2_change: float = 0.0
        self._last_rmse: float = 0.0

    def update(self, surface: VolSurface, spot: float) -> None:
        """Record current surface state.

        This must be called *after* the surface is fitted for the current step.
        """
        expiry = _pick_ref_expiry(surface)

        # ATM vol, 25-delta RR and fly
        atm_iv = float(surface.implied_vol(spot, expiry))
        rr, fly = _rr_and_fly(surface, spot, expiry)
        self._atm_vol_history.append(atm_iv)
        self._rr_history.append(rr)
        self._fly_history.append(fly)

        # Calibration RMSE
        self._last_rmse = _calibration_rmse(surface)

        # L2 surface change — evaluated on a fixed log-moneyness grid
        iv_vec = np.array(
            [surface.implied_vol(spot * np.exp(float(x)), expiry) for x in _REF_X],
            dtype=np.float64,
        )
        if self._prev_iv_vec is not None and self._prev_iv_vec.shape == iv_vec.shape:
            self._l2_change = float(np.sqrt(np.sum((iv_vec - self._prev_iv_vec) ** 2)))
        else:
            self._l2_change = 0.0
        self._prev_iv_vec = iv_vec

    def stability_stats(self, window: int | None = None) -> SurfaceStabilityStats:
        """Return a :class:`SurfaceStabilityStats` snapshot.

        Parameters
        ----------
        window: Rolling window size.  Defaults to :attr:`_stability_window`.
        """
        w = self._stability_window if window is None else window

        def _rstd(series: list[float]) -> float:
            h = np.array(series, dtype=np.float64)
            recent = h[-w:] if len(h) >= w else h
            return float(np.std(recent, ddof=1)) if recent.size > 1 else 0.0

        return SurfaceStabilityStats(
            atm_vol_stability=_rstd(self._atm_vol_history),
            skew_stability=_rstd(self._rr_history),
            fly_stability=_rstd(self._fly_history),
            surface_rmse=self._last_rmse,
            surface_daily_l2_change=self._l2_change,
        )


# ─── DiagnosticsReport factory ───────────────────────────────────────────────


def build_diagnostics_report(
    variance_tracker: VarianceTracker,
    residual_tracker: ResidualAttributionTracker,
    surface_monitor: SurfaceMonitor,
    dt: float = 1.0 / 252.0,
    timestamp: datetime | None = None,
    vrp_window: int = 30,
    stability_window: int = 30,
) -> DiagnosticsReport:
    """Assemble a :class:`DiagnosticsReport` from live tracker state.

    Parameters
    ----------
    variance_tracker: Updated :class:`VarianceTracker`.
    residual_tracker: Updated :class:`ResidualAttributionTracker`.
    surface_monitor:  Updated :class:`SurfaceMonitor`.
    dt:               Step length in years (used for VRP).
    timestamp:        Snapshot time; defaults to UTC now.
    vrp_window:       Rolling window for VRP mean / confidence band.
    stability_window: Rolling window for surface stability.
    """
    ts = timestamp if timestamp is not None else datetime.now(tz=UTC)
    vrp_now = variance_tracker.variance_risk_premium(dt)
    vrp_stats = variance_tracker.vrp_rolling_stats(window=vrp_window, dt=dt)
    res_stats = residual_tracker.residual_stats()
    surf_stats = surface_monitor.stability_stats(window=stability_window)
    return DiagnosticsReport(
        timestamp=ts,
        variance_risk_premium=vrp_now,
        vrp_30d_mean=vrp_stats.mean,
        residual_attribution_mean=res_stats.mean,
        residual_attribution_std=res_stats.std,
        residual_attribution_kurtosis=res_stats.excess_kurtosis,
        surface_rmse=surf_stats.surface_rmse,
        atm_vol_stability=surf_stats.atm_vol_stability,
        skew_stability=surf_stats.skew_stability,
    )


def log_runtime_sanity_checks(completed_steps: int, portfolio: object, pnl_report: object | None) -> None:
    """Emit runtime warning checks once the run has completed 252 steps.

    These are runtime guardrails rather than tests: they do not interrupt the
    simulator, but they do surface when the strategy is still violating the
    intended carry / gamma / stress envelope after a full trading year.
    """
    if completed_steps < 252 or getattr(portfolio, "sanity_checks_logged", False):
        return

    portfolio.sanity_checks_logged = True

    var_capture_running = float(getattr(portfolio, "var_capture_running", 0.0))
    if var_capture_running <= 0.0:
        logger.warning(
            "sanity check failed: var_capture_running=%.6f <= 0 after 252 steps", var_capture_running
        )
    else:
        logger.info("sanity check passed: var_capture_running=%.6f", var_capture_running)

    gamma_pnl = float(getattr(portfolio, "cum_gamma_pnl", getattr(pnl_report, "cum_gamma_pnl", 0.0)))
    total_pnl = float(getattr(portfolio, "cum_total_pnl", getattr(pnl_report, "cum_total_pnl", 0.0)))
    gamma_ratio = gamma_pnl / total_pnl if abs(total_pnl) > 1e-12 else float("inf")
    if gamma_ratio <= -0.5:
        logger.warning(
            "sanity check failed: gamma_pnl/total_pnl=%.6f <= -0.5 (gamma_pnl=%.6f total_pnl=%.6f)",
            gamma_ratio,
            gamma_pnl,
            total_pnl,
        )
    else:
        logger.info("sanity check passed: gamma_pnl/total_pnl=%.6f", gamma_ratio)

    worst_stress_pnl = float(getattr(portfolio, "last_worst_stress_pnl", 0.0))
    daily_theta = abs(float(getattr(portfolio, "last_daily_theta", 0.0)))
    stress_bound = -3.0 * max(daily_theta, 1e-12)
    if worst_stress_pnl <= stress_bound:
        logger.warning(
            "sanity check failed: worst_stress_pnl=%.6f <= %.6f (-3x daily theta)",
            worst_stress_pnl,
            stress_bound,
        )
    else:
        logger.info(
            "sanity check passed: worst_stress_pnl=%.6f > %.6f (-3x daily theta)",
            worst_stress_pnl,
            stress_bound,
        )
