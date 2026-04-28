"""Synthetic underlying price feed.

Simulates a spot price under Merton jump-diffusion:

    dS / S = mu * dt + sigma * dW + (J - 1) * dN

with ``N ~ Poisson(lambda * dt)`` and ``J = exp(Y)``,
``Y ~ Normal(mu_jump, sigma_jump^2)``. A fixed quarterly cash dividend is
subtracted on ex-div days. Each trading day is synthesised from
``intraday_steps`` GBM sub-steps plus daily jumps, producing an (O, H, L, C)
bar consumed by a rolling 30-day Yang-Zhang realized-variance estimator.

The public interface is the :class:`SyntheticFeed` iterator, which yields
:class:`Tick` records holding ``(timestamp, spot, realized_vol_30d, ...)``
suitable for downstream consumers (option-chain builder, surface fitter,
hedger).

All randomness is drawn from :class:`numpy.random.Generator` constructed
with an explicit seed so runs are bit-for-bit reproducible.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from vol_surface_mm.config import MarketConfig

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """Daily OHLC bar used by the Yang-Zhang estimator."""

    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Tick:
    """One simulation step emitted by :class:`SyntheticFeed`.

    Attributes
    ----------
    step:
        Zero-based step index (0 == first day).
    timestamp:
        UTC datetime for the close of the bar. Spacing is one calendar
        day per step for display purposes.
    t:
        Model time in years since simulation start.
    spot:
        Closing spot price for the bar (post-dividend if ex-div).
    inst_vol:
        Instantaneous diffusion vol used to simulate this step. Equal to
        ``MarketConfig.sigma`` in this model; reserved for stochastic-vol
        extensions.
    realized_vol_30d:
        Annualised Yang-Zhang realized volatility over the trailing
        window (``MarketConfig.rv_window_days`` days). ``0.0`` until at
        least two bars are available.
    bar:
        The (O, H, L, C) bar for this step.
    dividend_paid:
        Cash dividend paid at the close of this step (0 if not ex-div).
    jumps:
        Number of Merton jumps that fired during this step.
    """

    step: int
    timestamp: datetime
    t: float
    spot: float
    inst_vol: float
    realized_vol_30d: float
    bar: Bar
    dividend_paid: float
    jumps: int


# ---------------------------------------------------------------------------
# Yang-Zhang realized variance
# ---------------------------------------------------------------------------


class YangZhangEstimator:
    """Rolling Yang-Zhang realized-variance estimator.

    Yang-Zhang (2000) combines overnight, open-to-close, and
    Rogers-Satchell components. For a window of ``N`` bars:

        sigma_YZ^2 = sigma_O^2 + k * sigma_C^2 + (1 - k) * sigma_RS^2

    where

        sigma_O^2  = Var[ ln(O_t / C_{t-1}) ]              (overnight)
        sigma_C^2  = Var[ ln(C_t / O_t) ]                  (open-to-close)
        sigma_RS^2 = mean[ ln(H/C) ln(H/O) + ln(L/C) ln(L/O) ]
        k          = 0.34 / (1.34 + (N + 1) / (N - 1))

    The result is annualised by dividing by ``dt`` (bar length in years).
    Returns 0.0 until at least two bars are available.
    """

    def __init__(self, window: int, dt: float) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window: int = int(window)
        self._dt: float = float(dt)
        self._bars: deque[Bar] = deque(maxlen=window)

    def update(self, bar: Bar) -> None:
        """Append a new bar; oldest is evicted once window is full."""
        self._bars.append(bar)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        o = np.array([b.open for b in self._bars], dtype=np.float64)
        h = np.array([b.high for b in self._bars], dtype=np.float64)
        lo = np.array([b.low for b in self._bars], dtype=np.float64)
        c = np.array([b.close for b in self._bars], dtype=np.float64)
        return o, h, lo, c

    def variance(self) -> float:
        """Return the annualised Yang-Zhang variance estimate."""
        n = len(self._bars)
        if n < 2:
            return 0.0
        o, h, lo, c = self._arrays()

        overnight = np.log(o[1:] / c[:-1])  # n - 1 obs
        otc = np.log(c / o)  # n obs
        rs = np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)

        var_o = float(np.var(overnight, ddof=1)) if overnight.size >= 2 else 0.0
        var_c = float(np.var(otc, ddof=1)) if otc.size >= 2 else 0.0
        var_rs = float(np.mean(rs))

        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
        yz_daily_var = var_o + k * var_c + (1.0 - k) * var_rs
        yz_daily_var = max(yz_daily_var, 0.0)
        return float(yz_daily_var / max(self._dt, 1e-12))

    def volatility(self) -> float:
        """Return annualised Yang-Zhang volatility."""
        return float(np.sqrt(self.variance()))


# ---------------------------------------------------------------------------
# Dividend schedule
# ---------------------------------------------------------------------------


class QuarterlyDividendSchedule:
    """Fixed quarterly cash dividend schedule in trading-day offsets."""

    def __init__(
        self,
        cash: float,
        period_days: int,
        first_day: int,
        total_days: int,
    ) -> None:
        self._cash: float = float(cash)
        if period_days <= 0:
            self._ex_days: set[int] = set()
            return
        self._ex_days = {int(d) for d in range(int(first_day), int(total_days), int(period_days))}

    def amount_on(self, day: int) -> float:
        """Return cash paid on ``day`` (0 if not ex-div)."""
        return self._cash if day in self._ex_days else 0.0

    @property
    def ex_days(self) -> set[int]:
        return set(self._ex_days)


# ---------------------------------------------------------------------------
# The feed
# ---------------------------------------------------------------------------


class SyntheticFeed:
    """Iterator producing :class:`Tick`s from a Merton jump-diffusion process.

    Parameters
    ----------
    cfg:
        :class:`MarketConfig` supplying ``s0, mu, sigma, lambda_jump,
        mu_jump, sigma_jump, dt, intraday_steps`` and dividend schedule.
    steps:
        Number of daily bars to simulate.
    seed:
        Seed for :func:`numpy.random.default_rng`. The same seed
        reproduces identical (spot, rv, jumps, dividends) sequences.
    start:
        Optional datetime for the first close. Defaults to a fixed epoch
        so timestamps are reproducible across runs.

    Yields
    ------
    Tick
        ``(step, timestamp, t, spot, inst_vol, realized_vol_30d, bar,
        dividend_paid, jumps)``.
    """

    def __init__(
        self,
        cfg: MarketConfig,
        steps: int,
        seed: int = 0,
        start: datetime | None = None,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if cfg.intraday_steps <= 0:
            raise ValueError("intraday_steps must be positive")

        self._cfg: MarketConfig = cfg
        self._steps: int = int(steps)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._spot: float = float(cfg.s0)
        self._i: int = 0
        self._start: datetime = start or datetime(
            2025,
            1,
            2,
            tzinfo=UTC,
        )
        self._rv: YangZhangEstimator = YangZhangEstimator(
            window=cfg.rv_window_days,
            dt=cfg.dt,
        )
        self._dividends: QuarterlyDividendSchedule = QuarterlyDividendSchedule(
            cash=cfg.dividend_cash,
            period_days=cfg.dividend_period_days,
            first_day=cfg.dividend_first_day,
            total_days=steps,
        )

    def __iter__(self) -> Iterator[Tick]:
        return self

    def _simulate_bar(self) -> tuple[Bar, int]:
        """Generate one (O, H, L, C) bar with Merton jumps applied at close.

        The intraday path is GBM; the Poisson jump count for the day is
        drawn once and applied multiplicatively at the close. This keeps
        intraday extrema well-defined (jumps do not artificially move
        H/L) and is updated afterwards so the bar remains consistent
        (low <= close <= high).
        """
        cfg = self._cfg
        n_sub = int(cfg.intraday_steps)
        dt_sub = cfg.dt / n_sub

        open_px = self._spot
        px = open_px
        high = open_px
        low = open_px

        z = self._rng.standard_normal(n_sub)
        drift = (cfg.mu - 0.5 * cfg.sigma * cfg.sigma) * dt_sub
        diffusion = cfg.sigma * np.sqrt(dt_sub) * z
        for inc in drift + diffusion:
            px = float(px * np.exp(inc))
            if px > high:
                high = px
            if px < low:
                low = px

        n_jumps = int(self._rng.poisson(cfg.lambda_jump * cfg.dt))
        if n_jumps > 0:
            y = self._rng.normal(cfg.mu_jump, cfg.sigma_jump, size=n_jumps)
            log_jump = float(np.sum(y))
            px = float(px * np.exp(log_jump))
            if px < low:
                low = px
            if px > high:
                high = px

        return Bar(open=open_px, high=high, low=low, close=px), n_jumps

    def __next__(self) -> Tick:
        if self._i >= self._steps:
            raise StopIteration

        bar, n_jumps = self._simulate_bar()
        close = bar.close

        # Ex-div: reduce close by cash dividend and keep the bar consistent.
        div = self._dividends.amount_on(self._i)
        if div > 0.0:
            close = max(close - div, 1e-8)
            bar = Bar(open=bar.open, high=bar.high, low=min(bar.low, close), close=close)

        self._spot = close
        self._rv.update(bar)

        t = self._i * self._cfg.dt
        ts = self._start + timedelta(days=self._i)
        tick = Tick(
            step=self._i,
            timestamp=ts,
            t=t,
            spot=self._spot,
            inst_vol=self._cfg.sigma,
            realized_vol_30d=self._rv.volatility(),
            bar=bar,
            dividend_paid=div,
            jumps=n_jumps,
        )
        self._i += 1
        return tick
