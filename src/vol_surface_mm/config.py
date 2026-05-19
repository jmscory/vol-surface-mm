"""Global parameters and constants for the vol-surface-mm simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MarketConfig:
    """Static market environment and Merton jump-diffusion parameters."""

    # Rates and spot
    r: float = 0.04  # risk-free rate, continuous
    q: float = 0.0  # continuous dividend yield (proxy)
    s0: float = 100.0  # initial spot

    # GBM
    mu: float = 0.07  # drift
    sigma: float = 0.20  # diffusion vol

    # Merton jump component: J ~ LogNormal(mu_j, sigma_j), N ~ Poisson(lambda)
    lambda_jump: float = 1.0  # expected jumps per year
    mu_jump: float = -0.02  # mean log-jump
    sigma_jump: float = 0.08  # stdev of log-jump

    # Discretisation
    dt: float = 1.0 / 252.0  # one trading day
    intraday_steps: int = 26  # sub-steps per day for OHLC generation

    # Dividend schedule: fixed cash paid quarterly
    dividend_cash: float = 0.25  # per-share cash amount
    dividend_period_days: int = 63  # ~quarterly in trading days
    dividend_first_day: int = 21  # first ex-div day offset

    # Realized vol estimator window (trading days)
    rv_window_days: int = 30


@dataclass(frozen=True)
class ChainConfig:
    """Option chain grid."""

    # Strikes: +/- 30% from ATM in 2.5% increments
    strike_range_pct: float = 0.30
    strike_step_pct: float = 0.025

    # Expiries in calendar days-to-expiry
    expiries_dte: tuple[int, ...] = (7, 14, 30, 60, 90, 180)

    # Quoting widths and OI synthesis controls
    spread_bps_atm: float = 40.0  # half-spread floor at ATM (bps of mid)
    spread_bps_wing: float = 250.0  # widens linearly with |log-moneyness|
    oi_base: float = 5000.0
    oi_decay: float = 3.0  # exponential decay in |log-moneyness|

    # Legacy coarse grid retained for surface fitting / smile seeding
    expiries_years: tuple[float, ...] = (
        7 / 365,
        30 / 365,
        60 / 365,
        90 / 365,
        180 / 365,
        365 / 365,
    )
    moneyness: tuple[float, ...] = (
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    )


@dataclass(frozen=True)
class SurfaceConfig:
    """SVI / spline fit controls."""

    svi_a_bounds: tuple[float, float] = (-1.0, 1.0)
    svi_b_bounds: tuple[float, float] = (1e-4, 5.0)
    svi_rho_bounds: tuple[float, float] = (-0.999, 0.999)
    svi_m_bounds: tuple[float, float] = (-1.0, 1.0)
    svi_sigma_bounds: tuple[float, float] = (1e-4, 2.0)
    min_total_variance: float = 1e-8


@dataclass(frozen=True)
class QuotingConfig:
    """Market maker quoting controls."""

    base_half_spread_vol: float = 0.005
    min_half_spread_vol: float = 0.005  # 0.5 vol points
    short_dte_weight: float = 0.010
    low_oi_weight: float = 0.15
    oi_reference: float = 5000.0
    base_vega_weight: float = 0.00005
    base_gamma_weight: float = 0.0002
    k_vega: float = 0.0002
    k_gamma: float = 0.0005
    skew_factor: float = 0.25
    target_net_gamma: float = 0.005
    gamma_sensitivity: float = 25.0
    gamma_target_max_dte_days: int = 30
    gamma_target_moneyness_band: float = 0.03
    kelly_fraction: float = 0.05
    stress_guard_multiple: float = 2.5
    vega_capacity: float = 2500.0
    lambda_delta: float = 0.20
    lambda_gamma: float = 0.10
    delta_limit: float = 300.0
    gamma_limit: float = 8.0
    vega_limit: float = 4000.0
    max_position: float = 500.0
    quote_size: float = 10.0
    inventory_penalty: float = 0.0002  # legacy additive widening on net-vega util
    skew_tilt: float = 0.5  # legacy compatibility knob
    fill_base_probability: float = 0.25
    fill_distance_sensitivity: float = 1.25
    fill_oi_exponent: float = 0.50
    fill_rng_seed: int | None = 7
    oi_base: float = 5000.0
    oi_decay: float = 3.0
    oi_put_wing_boost: float = 0.5
    oi_call_wing_boost: float = 0.2


@dataclass(frozen=True)
class HedgingConfig:
    """Delta hedger controls."""

    rebalance_threshold: float = 0.25
    transaction_cost_bps: float = 1.0
    dt: float = 1.0 / 252.0  # length of one simulation step in years
    discrete_hedge_interval: int = 5  # steps between discrete rehedges
    initial_bankroll: float = 100000.0
    delta_limit: float = 300.0
    var_capture_window_days: int = 30
    cheapness_negative_streak: int = 5
    cheapness_boost_periods: int = 10
    cheapness_boost_multiplier: float = 1.20
    tail_hedge_delta_trigger_ratio: float = 0.20
    tail_hedge_put_delta: float = 0.10
    tail_hedge_dte_days: int = 30
    # Gamma protection: buy a long OTM strangle when the book's net_gamma
    # falls below ``-gamma_protection_trigger`` (an absolute, negative
    # gamma threshold expressed as a positive number).  Set to 0 to disable.
    gamma_protection_trigger: float = 4.0
    gamma_protection_strangle_delta: float = 0.20
    gamma_protection_dte_days: int = 30
    gamma_protection_max_lots: float = 3.0


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulation controls."""

    steps: int = 250
    seed: int = 42
    # Initial book composition.  ``"flat"`` (default) starts with no
    # positions and lets the quoting engine build inventory organically.
    # ``"short_straddle"`` reproduces the legacy short-vol seed (5x ATM
    # short straddle + 3x OTM long wings).
    seeding_mode: Literal["flat", "short_straddle"] = "flat"
    market: MarketConfig = field(default_factory=MarketConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    quoting: QuotingConfig = field(default_factory=QuotingConfig)
    hedging: HedgingConfig = field(default_factory=HedgingConfig)


DEFAULT_CONFIG: SimConfig = SimConfig()
