from __future__ import annotations

import csv
import ssl
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

from vol_surface_mm import REPO_ROOT
from vol_surface_mm.config import DEFAULT_CONFIG, MarketConfig, SimConfig
from vol_surface_mm.core.hedging import PnLReport, PortfolioManager
from vol_surface_mm.core.quoting import QuotingEngine
from vol_surface_mm.core.stress import StressReport, run_all_scenarios
from vol_surface_mm.core.surface import VolSurface, build_surface
from vol_surface_mm.data.feed import Bar, SyntheticFeed, YangZhangEstimator
from vol_surface_mm.data.options_chain import build_chain
from vol_surface_mm.diagnostics.metrics import (
    ResidualAttributionTracker,
    SurfaceMonitor,
    VarianceTracker,
    build_diagnostics_report,
)

REAL_AAPL_DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"

FeedName = Literal["gbm", "bates", "rough", "real_aapl"]


@dataclass(frozen=True)
class FeedPoint:
    step: int
    t: float
    spot: float
    inst_vol: float


@dataclass
class RunSummary:
    feed: FeedName
    total_pnl: float
    var_capture: float
    max_drawdown: float
    gamma_pnl: float
    theta_pnl: float
    hedge_error_kurtosis: float
    surface_rmse: float
    atm_vol_stability: float
    skew_stability: float
    survived: bool
    tuning_survived: bool
    profitable: bool
    avg_daily_theta: float
    dd_limit: float
    final_bankroll: float
    final_spot: float
    final_surface: VolSurface
    stress_reports: list[StressReport]
    timestamp: str


def seed_portfolio(pm: PortfolioManager, surface: VolSurface, spot: float, cfg: SimConfig) -> None:
    """Seed the initial book per ``cfg.seeding_mode``.

    See :func:`vol_surface_mm.cli._seed_portfolio` for the canonical
    description; this helper is the artifact-pipeline copy.
    """
    if cfg.seeding_mode == "flat":
        return
    if cfg.seeding_mode != "short_straddle":
        raise ValueError(f"unknown seeding_mode: {cfg.seeding_mode!r}")

    mid_idx = len(surface.expiries) // 2
    expiry_t = float(surface.expiries[mid_idx])
    expiry_d = int(surface.expiries_days[mid_idx])

    strikes = np.array([spot * 0.90, spot, spot, spot * 1.10], dtype=np.float64)
    ty = np.full(4, expiry_t, dtype=np.float64)
    td = np.array([expiry_d, expiry_d, expiry_d, expiry_d], dtype=np.int32)
    option_types = np.array(["put", "put", "call", "call"])
    vols = np.array([surface.implied_vol(float(k), expiry_d) for k in strikes], dtype=np.float64)

    from vol_surface_mm.core.pricing import price_chain

    quotes = price_chain(spot, strikes, ty, td, option_types, vols, r=cfg.market.r, q=cfg.market.q)

    pm.add_position(quotes[0], 3, "buy")
    pm.add_position(quotes[1], 5, "sell")
    pm.add_position(quotes[2], 5, "sell")
    pm.add_position(quotes[3], 3, "buy")


def attempt_fills(engine: QuotingEngine, spot: float, surface: VolSurface) -> None:
    near = float(surface.expiries[0])
    mid = float(surface.expiries[min(1, len(surface.expiries) - 1)])
    for expiry in (near, mid):
        for strike in (spot * 0.95, spot, spot * 1.05):
            for option_type in ("call", "put"):
                fill = engine.simulate_fill(float(strike), float(expiry), option_type)
                if fill is None:
                    continue
                try:
                    engine.update_portfolio(fill)
                except ValueError:
                    continue


def _iter_gbm_points(market: MarketConfig, steps: int, seed: int) -> Iterator[FeedPoint]:
    gbm_market = replace(market, lambda_jump=0.0, mu_jump=0.0, sigma_jump=0.0)
    feed = SyntheticFeed(gbm_market, steps=steps, seed=seed)
    for tick in feed:
        yield FeedPoint(
            step=int(tick.step), t=float(tick.t), spot=float(tick.spot), inst_vol=float(tick.inst_vol)
        )


def _iter_bates_points(market: MarketConfig, steps: int, seed: int) -> Iterator[FeedPoint]:
    rng = np.random.default_rng(seed)
    dt = float(market.dt)

    # Adversarial defaults from the request.
    kappa = 3.0
    theta = 0.06
    xi = 0.5
    rho = -0.7
    lambda_jump = 0.3
    mu_jump = -0.05
    sigma_jump = 0.08

    s = float(market.s0)
    v = max(float(market.sigma) ** 2, 1e-6)

    for step in range(steps):
        z1 = float(rng.standard_normal())
        z2_raw = float(rng.standard_normal())
        z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 1e-12)) * z2_raw

        dv = kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0.0)) * np.sqrt(dt) * z2
        v = max(v + dv, 1e-8)

        n_jumps = int(rng.poisson(lambda_jump * dt))
        jump_log = 0.0
        if n_jumps > 0:
            jump_log = float(np.sum(rng.normal(mu_jump, sigma_jump, size=n_jumps)))

        drift = (
            market.mu - lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump * sigma_jump) - 1.0) - 0.5 * v
        ) * dt
        diff = np.sqrt(v * dt) * z1
        s = float(max(s * np.exp(drift + diff + jump_log), 1e-8))

        yield FeedPoint(step=step, t=step * dt, spot=s, inst_vol=float(np.sqrt(v)))


def _iter_rough_points(market: MarketConfig, steps: int, seed: int) -> Iterator[FeedPoint]:
    rng = np.random.default_rng(seed)
    dt = float(market.dt)

    hurst = 0.10
    eta = 1.35
    mean_vol = max(float(market.sigma), 1e-4)

    s = float(market.s0)
    log_var = np.log(mean_vol * mean_vol)
    ema_noise = 0.0

    for step in range(steps):
        z = float(rng.standard_normal())
        # Fractional-noise proxy with memory; exponent < 0.5 creates rough paths.
        ema_noise = 0.92 * ema_noise + 0.08 * z
        rough_increment = (dt**hurst) * (0.75 * z + 0.25 * ema_noise)

        log_var = 0.985 * log_var + 0.015 * np.log(mean_vol * mean_vol) + eta * rough_increment
        inst_vol = float(np.sqrt(max(np.exp(log_var), 1e-10)))

        s = float(
            max(s * np.exp((market.mu - 0.5 * inst_vol * inst_vol) * dt + inst_vol * np.sqrt(dt) * z), 1e-8)
        )
        yield FeedPoint(step=step, t=step * dt, spot=s, inst_vol=inst_vol)


def _real_dataset_cache_path(feed: FeedName) -> Path:
    return REPO_ROOT / "data" / "cache" / f"{feed}.csv"


def _ensure_real_dataset_cached(feed: FeedName) -> Path:
    if feed != "real_aapl":
        raise ValueError(f"unsupported real dataset feed: {feed}")

    cache_path = _real_dataset_cache_path(feed)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(REAL_AAPL_DATASET_URL, headers={"User-Agent": "vol-surface-mm/1.0"})

    contexts: list[ssl.SSLContext | None] = []
    if certifi is not None:
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    contexts.append(ssl.create_default_context())
    contexts.append(ssl._create_unverified_context())

    last_error: Exception | None = None
    for context in contexts:
        try:
            with urlopen(request, timeout=30, context=context) as response, cache_path.open("wb") as handle:
                handle.write(response.read())
            return cache_path
        except URLError as exc:
            last_error = exc

    if cache_path.exists():
        cache_path.unlink()
    raise RuntimeError(f"failed to cache dataset for {feed}") from last_error
    return cache_path


def _iter_real_aapl_points(market: MarketConfig, steps: int) -> Iterator[FeedPoint]:
    cache_path = _ensure_real_dataset_cached("real_aapl")
    with cache_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if len(rows) < steps:
        raise ValueError(f"real_aapl dataset has {len(rows)} rows, fewer than requested steps={steps}")

    estimator = YangZhangEstimator(window=market.rv_window_days, dt=market.dt)
    selected_rows = rows[-steps:]
    for step, row in enumerate(selected_rows):
        close = float(row["AAPL.Close"])
        adjusted = float(row["AAPL.Adjusted"])
        scale = adjusted / max(close, 1e-12)
        bar = Bar(
            open=float(row["AAPL.Open"]) * scale,
            high=float(row["AAPL.High"]) * scale,
            low=float(row["AAPL.Low"]) * scale,
            close=adjusted,
        )
        estimator.update(bar)
        inst_vol = estimator.volatility()
        if inst_vol <= 0.0:
            inst_vol = float(market.sigma)
        yield FeedPoint(step=step, t=step * float(market.dt), spot=float(bar.close), inst_vol=float(inst_vol))


def iter_feed_points(feed: FeedName, market: MarketConfig, steps: int, seed: int) -> Iterable[FeedPoint]:
    if feed == "gbm":
        return _iter_gbm_points(market=market, steps=steps, seed=seed)
    if feed == "bates":
        return _iter_bates_points(market=market, steps=steps, seed=seed)
    if feed == "rough":
        return _iter_rough_points(market=market, steps=steps, seed=seed)
    if feed == "real_aapl":
        return _iter_real_aapl_points(market=market, steps=steps)
    raise ValueError(f"unsupported feed: {feed}")


def _max_drawdown_from_equity(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, float(value))
        max_dd = max(max_dd, peak - float(value))
    return float(max_dd)


def _drawdown_limit(avg_daily_theta: float, stress_multiple: float) -> float:
    return float(stress_multiple * max(avg_daily_theta, 1e-12) * 252.0)


def _strict_survived(total_pnl: float, max_drawdown: float, dd_limit: float) -> bool:
    return bool(total_pnl > 0.0 and max_drawdown < dd_limit)


def _tuning_survived(final_bankroll: float, max_drawdown: float, dd_limit: float) -> bool:
    return bool(final_bankroll > 0.0 and max_drawdown < dd_limit)


def run_feed_backtest(
    feed: FeedName,
    cfg: SimConfig = DEFAULT_CONFIG,
    *,
    steps: int = 252,
    seed: int = 42,
    backend: str = "sabr",
    survival_multiple: float = 3.0,
    frozen_surface: bool = False,
    stress_refresh_interval: int = 20,
    fill_interval: int = 5,
    max_inventory_positions: int = 300,
    simulate_fills: bool = True,
) -> RunSummary:
    market = replace(cfg.market, dt=cfg.market.dt)

    pm = PortfolioManager(cfg.hedging)
    rv_tracker = VarianceTracker(window=20)
    residual_tracker = ResidualAttributionTracker()
    surf_monitor = SurfaceMonitor(stability_window=30)

    final_surface: VolSurface | None = None
    final_spot = float(market.s0)
    prev_spot: float | None = None

    equity_curve: list[float] = [0.0]
    theta_steps: list[float] = []
    theta_step_pnl: list[float] = []
    report: PnLReport | None = None
    cached_surface: VolSurface | None = None

    for point in iter_feed_points(feed=feed, market=market, steps=steps, seed=seed):
        if frozen_surface:
            if cached_surface is None:
                chain = build_chain(point.spot, point.inst_vol, market, cfg.chain)
                cached_surface = build_surface(
                    chain, point.spot, market.r, market.q, cfg.surface, backend=backend
                )
            surface = cached_surface
        else:
            chain = build_chain(point.spot, point.inst_vol, market, cfg.chain)
            surface = build_surface(chain, point.spot, market.r, market.q, cfg.surface, backend=backend)

        if point.step == 0:
            seed_portfolio(pm, surface, float(point.spot), cfg)

        engine = QuotingEngine(surface, pm.portfolio, cfg.quoting)
        if (
            simulate_fills
            and point.step % max(fill_interval, 1) == 0
            and len(pm.portfolio.positions) < max_inventory_positions
        ):
            attempt_fills(engine, float(point.spot), surface)

        # Full stress recomputation is expensive and dominates sweep runtime.
        # Keep it periodic while preserving a representative tail snapshot.
        pm.portfolio.stress_state_dirty = bool(
            point.step == 0 or point.step % max(stress_refresh_interval, 1) == 0
        )

        report = pm.mark_to_market(surface, float(point.spot))
        if point.step % cfg.hedging.discrete_hedge_interval == 0:
            pm.delta_hedge(float(point.spot))

        if prev_spot is not None:
            ret = float(np.log(max(float(point.spot), 1e-12) / max(prev_spot, 1e-12)))
            mid_idx = len(surface.expiries) // 2
            iv_atm = float(surface.implied_vol(float(point.spot), float(surface.expiries[mid_idx])))
            book_greeks = pm.compute_book_greeks()
            rv_tracker.update(ret, iv_atm=iv_atm, vega=float(book_greeks.vega), dt=cfg.hedging.dt)
            residual_tracker.update(float(report.residual_pnl))

        surf_monitor.update(surface, float(point.spot))

        final_surface = surface
        final_spot = float(point.spot)
        prev_spot = float(point.spot)
        equity_curve.append(float(report.cum_total_pnl))
        theta_steps.append(float(pm.portfolio.last_daily_theta))
        theta_step_pnl.append(float(report.theta_pnl))

    if report is None or final_surface is None:
        raise RuntimeError("empty backtest run")

    diag = build_diagnostics_report(rv_tracker, residual_tracker, surf_monitor, dt=cfg.hedging.dt)
    stress_reports = run_all_scenarios(pm.portfolio, final_surface, final_spot, market.r, market.q)

    max_drawdown = _max_drawdown_from_equity(equity_curve)
    avg_daily_theta = float(np.mean(theta_steps)) if theta_steps else 0.0
    total_pnl = float(report.cum_total_pnl)
    dd_limit = _drawdown_limit(avg_daily_theta, survival_multiple)
    final_bankroll = float(cfg.hedging.initial_bankroll + total_pnl)

    residual_stats = residual_tracker.residual_stats()

    return RunSummary(
        feed=feed,
        total_pnl=total_pnl,
        var_capture=float(pm.portfolio.var_capture_running),
        max_drawdown=max_drawdown,
        gamma_pnl=float(report.cum_gamma_pnl),
        theta_pnl=float(np.sum(theta_step_pnl)),
        hedge_error_kurtosis=float(residual_stats.excess_kurtosis),
        surface_rmse=float(diag.surface_rmse),
        atm_vol_stability=float(diag.atm_vol_stability),
        skew_stability=float(diag.skew_stability),
        survived=_strict_survived(total_pnl=total_pnl, max_drawdown=max_drawdown, dd_limit=dd_limit),
        tuning_survived=_tuning_survived(
            final_bankroll=final_bankroll, max_drawdown=max_drawdown, dd_limit=dd_limit
        ),
        profitable=bool(total_pnl > 0.0),
        avg_daily_theta=avg_daily_theta,
        dd_limit=dd_limit,
        final_bankroll=final_bankroll,
        final_spot=final_spot,
        final_surface=final_surface,
        stress_reports=list(stress_reports),
        timestamp=datetime.now(tz=UTC).isoformat(),
    )
