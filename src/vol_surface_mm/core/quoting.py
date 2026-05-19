"""Skew-aware options quoting engine with inventory management.

Quotes are formed in volatility space around the surface mid and then mapped
to price space with Black-Scholes-Merton. Spread logic widens with vega,
gamma, short datedness, and low open interest. Inventory penalties shift the
quoted mid against positions that worsen current delta/gamma risk, while skew
exposure is managed via a vanna-aware adjustment on put markets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from vol_surface_mm.config import QuotingConfig
from vol_surface_mm.core.hedging import BookState, OptionPosition
from vol_surface_mm.core.hedging import Portfolio as _HedgingPortfolio
from vol_surface_mm.core.pricing import bs_greeks, bs_price
from vol_surface_mm.core.surface import VolSurface

logger = logging.getLogger(__name__)


# ``QuotingEngine`` historically assumed the legacy ``BookState`` interface
# (``spot_position``/``add``).  The live runtime now hands it the newer
# ``hedging.Portfolio`` (``hedge_shares`` and no ``add`` method).  We accept
# both shapes explicitly via small adapter helpers defined below rather than
# aliasing one onto the other.
Portfolio = BookState | _HedgingPortfolio
FillSide = Literal["bid", "ask"]


def _hedge_position(portfolio: Portfolio) -> float:
    """Return the signed underlying hedge share count for either shape."""
    if isinstance(portfolio, BookState):
        return float(portfolio.spot_position)
    return float(portfolio.hedge_shares)


def _book_option_fill(portfolio: Portfolio, pos: OptionPosition, premium: float) -> None:
    """Append ``pos`` and debit/credit cash on either portfolio shape."""
    if isinstance(portfolio, BookState):
        portfolio.add(pos, premium)
        return
    portfolio.positions.append(pos)
    portfolio.cash -= pos.quantity * premium
    portfolio.stress_state_dirty = True


def _portfolio_attr(
    portfolio: Portfolio, attr: str, default: float | int | list[str]
) -> float | int | list[str]:
    return getattr(portfolio, attr, default)


def _append_portfolio_log(portfolio: Portfolio, field_name: str, message: str) -> None:
    log = getattr(portfolio, field_name, None)
    if not isinstance(log, list):
        return
    log.append(message)
    if len(log) > 512:
        del log[:-512]


def _copy_position(pos: OptionPosition) -> OptionPosition:
    return OptionPosition(
        strike=float(pos.strike),
        expiry=float(pos.expiry),
        option_type=str(pos.option_type),
        quantity=float(pos.quantity),
        inventory=bool(pos.inventory),
        label=str(pos.label),
    )


def _clone_portfolio(portfolio: Portfolio) -> Portfolio:
    if isinstance(portfolio, BookState):
        return BookState(
            positions=[_copy_position(pos) for pos in portfolio.positions],
            spot_position=float(portfolio.spot_position),
            cash=float(portfolio.cash),
            transaction_costs=float(portfolio.transaction_costs),
        )
    return replace(
        portfolio,
        positions=[_copy_position(pos) for pos in portfolio.positions],
        audit_trail=list(getattr(portfolio, "audit_trail", [])),
        rejection_log=list(getattr(portfolio, "rejection_log", [])),
        alert_log=list(getattr(portfolio, "alert_log", [])),
    )


def _portfolio_bankroll(portfolio: Portfolio, spot: float) -> float:
    bankroll = getattr(portfolio, "bankroll", None)
    if bankroll is not None:
        return float(max(float(bankroll), 1.0))

    gross_spot = abs(_hedge_position(portfolio)) * max(float(spot), 1e-12)
    gross_options = sum(abs(float(pos.quantity)) * max(float(spot), 1e-12) for pos in portfolio.positions)
    gross_cash = abs(float(portfolio.cash))
    return float(max(gross_spot + gross_options + gross_cash, max(float(spot), 1.0)))


@dataclass(frozen=True)
class ExposureReport:
    net_delta: float
    net_gamma: float
    net_vega: float
    net_vanna: float
    gross_vega: float
    largest_position: float
    delta_utilization: float
    gamma_utilization: float
    vega_utilization: float
    within_limits: bool


@dataclass(frozen=True)
class Fill:
    strike: float
    expiry: float
    option_type: str
    side: FillSide
    quantity: float
    price: float
    vol: float
    probability: float


@dataclass(frozen=True)
class Quote:
    strike: float
    expiry: float
    option_type: str
    mid_price: float
    bid: float
    ask: float
    model_iv: float
    half_spread_vol: float
    size: float
    bid_vol: float = float("nan")
    ask_vol: float = float("nan")
    open_interest: float = float("nan")
    fill_probability_bid: float = 0.0
    fill_probability_ask: float = 0.0
    gamma_pull_bps: float = 0.0


@dataclass(frozen=True)
class _QuoteState:
    expiry_years: float
    model_iv: float
    bid_vol: float
    ask_vol: float
    open_interest: float
    half_spread: float
    gamma_pull_bps: float
    size: float


def _expiry_years(expiry: float | int) -> float:
    value = float(expiry)
    if value <= 3.0:
        return max(value, 1e-12)
    return max(value / 365.0, 1e-12)


class QuotingEngine:
    def __init__(self, surface: VolSurface, portfolio: Portfolio, config: QuotingConfig) -> None:
        self.surface = surface
        self.portfolio = portfolio
        self.config = config
        self.rng = np.random.default_rng(config.fill_rng_seed)
        self._cached_worst_shock_key: tuple[float, float, float, float, float, float] | None = None
        self._cached_worst_shocked_surface: VolSurface | None = None

    def quote(self, strike: float, expiry: float, option_type: str) -> tuple[float, float]:
        state = self._build_quote_state(strike, expiry, option_type)
        return float(state.bid_vol), float(state.ask_vol)

    def _build_quote_state(self, strike: float, expiry: float, option_type: str) -> _QuoteState:
        expiry_years = _expiry_years(expiry)
        model_iv = float(self.surface.implied_vol(strike, expiry_years))
        greeks = bs_greeks(
            self.surface.spot,
            strike,
            expiry_years,
            self.surface.r,
            self.surface.q,
            model_iv,
            option_type,
        )
        exposure = self.current_exposure()
        days_to_expiry = max(int(round(expiry_years * 365.0)), 1)
        open_interest = self._open_interest(strike, expiry_years, option_type)
        quote_size = self._quote_size(strike, expiry_years, option_type, greeks.gamma)

        half_spread = self._half_spread(greeks.vega, greeks.gamma, days_to_expiry, open_interest)
        inventory_shift = self._inventory_mid_shift(greeks, exposure)
        mid_vol = model_iv + inventory_shift

        skew_adjust = 0.0
        if option_type == "put" and self.config.vega_capacity > 0.0:
            # Invert the legacy skew tilt so the engine is no longer nudged
            # toward systematically selling downside convexity.
            raw_skew = -self.config.skew_factor * exposure.net_vanna / self.config.vega_capacity
            skew_adjust = float(np.clip(raw_skew, -0.8 * half_spread, 0.8 * half_spread))

        bid_vol = max(mid_vol - half_spread - skew_adjust, 1e-4)
        ask_vol = max(mid_vol + half_spread - skew_adjust, bid_vol + 1e-6)

        gamma_pull_bps = self._gamma_pull_bps(strike, expiry_years, exposure.net_gamma)
        gamma_pull_vol = gamma_pull_bps * 1e-4
        if gamma_pull_vol != 0.0:
            # Positive pull = buy gamma: richer bids and less attractive offers.
            # Negative pull = sell gamma: cheaper bids and tighter offers.
            bid_vol = max(bid_vol + gamma_pull_vol, 1e-4)
            ask_vol = max(ask_vol + gamma_pull_vol, bid_vol + 1e-6)

        if quote_size <= 1e-8:
            bid_vol = float("nan")
            ask_vol = float("nan")
        elif self._would_breach_limits(strike, expiry_years, option_type, +quote_size):
            bid_vol = float("nan")
        if quote_size <= 1e-8 or self._would_breach_limits(strike, expiry_years, option_type, -quote_size):
            ask_vol = float("nan")
        logger.info(
            "quote_cycle strike=%.4f expiry=%.6f option_type=%s book_gamma=%.6f target_gamma=%.6f gamma_pull_bps=%.4f size=%.4f",
            float(strike),
            float(expiry_years),
            option_type,
            float(exposure.net_gamma),
            float(self.config.target_net_gamma),
            float(gamma_pull_bps),
            float(quote_size),
        )
        _append_portfolio_log(
            self.portfolio,
            "audit_trail",
            (
                f"quote strike={float(strike):.4f} expiry={float(expiry_years):.6f} type={option_type} "
                f"gamma_pull_bps={float(gamma_pull_bps):.4f} size={float(quote_size):.4f}"
            ),
        )
        return _QuoteState(
            expiry_years=float(expiry_years),
            model_iv=float(model_iv),
            bid_vol=float(bid_vol),
            ask_vol=float(ask_vol),
            open_interest=float(open_interest),
            half_spread=float(half_spread),
            gamma_pull_bps=float(gamma_pull_bps),
            size=float(quote_size),
        )

    def quote_market(self, strike: float, expiry: float, option_type: str) -> Quote:
        state = self._build_quote_state(strike, expiry, option_type)
        expiry_years = state.expiry_years
        model_iv = state.model_iv
        bid_vol = state.bid_vol
        ask_vol = state.ask_vol
        open_interest = state.open_interest

        mid_price = float(
            bs_price(
                self.surface.spot, strike, expiry_years, self.surface.r, self.surface.q, model_iv, option_type
            )
        )
        bid = (
            float("nan")
            if not np.isfinite(bid_vol)
            else float(
                bs_price(
                    self.surface.spot,
                    strike,
                    expiry_years,
                    self.surface.r,
                    self.surface.q,
                    bid_vol,
                    option_type,
                )
            )
        )
        ask = (
            float("nan")
            if not np.isfinite(ask_vol)
            else float(
                bs_price(
                    self.surface.spot,
                    strike,
                    expiry_years,
                    self.surface.r,
                    self.surface.q,
                    ask_vol,
                    option_type,
                )
            )
        )

        half_spread = 0.0
        if np.isfinite(bid_vol) and np.isfinite(ask_vol):
            half_spread = 0.5 * (ask_vol - bid_vol)
        elif np.isfinite(bid_vol):
            half_spread = abs(model_iv - bid_vol)
        elif np.isfinite(ask_vol):
            half_spread = abs(ask_vol - model_iv)

        fill_prob_bid, fill_prob_ask = self._fill_probabilities(model_iv, bid_vol, ask_vol, open_interest)
        return Quote(
            strike=float(strike),
            expiry=float(expiry_years),
            option_type=option_type,
            mid_price=float(mid_price),
            bid=float(max(bid, 0.0)) if np.isfinite(bid) else float("nan"),
            ask=float(ask),
            model_iv=float(model_iv),
            half_spread_vol=float(half_spread),
            size=float(state.size),
            bid_vol=float(bid_vol),
            ask_vol=float(ask_vol),
            open_interest=float(open_interest),
            fill_probability_bid=float(fill_prob_bid),
            fill_probability_ask=float(fill_prob_ask),
            gamma_pull_bps=float(state.gamma_pull_bps),
        )

    def update_portfolio(self, fill: Fill) -> None:
        expiry_years = _expiry_years(fill.expiry)
        signed_quantity = float(fill.quantity if fill.side == "bid" else -fill.quantity)
        if self._would_breach_limits(
            fill.strike,
            expiry_years,
            fill.option_type,
            signed_quantity,
            premium=float(fill.price),
        ):
            raise ValueError("fill would breach quoting risk limits")
        _book_option_fill(
            self.portfolio,
            OptionPosition(fill.strike, expiry_years, fill.option_type, signed_quantity),
            premium=fill.price,
        )

    def current_exposure(self) -> ExposureReport:
        spot = self.surface.spot
        r, q = self.surface.r, self.surface.q
        net_delta = _hedge_position(self.portfolio)
        net_gamma = 0.0
        net_vega = 0.0
        net_vanna = 0.0
        gross_vega = 0.0
        positions_by_contract: dict[tuple[float, float, str], float] = defaultdict(float)

        for pos in self.portfolio.positions:
            iv = float(self.surface.implied_vol(pos.strike, pos.expiry))
            greeks = bs_greeks(spot, pos.strike, pos.expiry, r, q, iv, pos.option_type)
            net_delta += pos.quantity * greeks.delta
            net_gamma += pos.quantity * greeks.gamma
            net_vega += pos.quantity * greeks.vega
            net_vanna += pos.quantity * greeks.vanna
            gross_vega += abs(pos.quantity * greeks.vega)
            positions_by_contract[(pos.strike, pos.expiry, pos.option_type)] += pos.quantity

        largest_position = max((abs(qty) for qty in positions_by_contract.values()), default=0.0)
        delta_util = abs(net_delta) / max(self.config.delta_limit, 1e-12)
        gamma_util = abs(net_gamma) / max(self.config.gamma_limit, 1e-12)
        vega_util = abs(net_vega) / max(self.config.vega_limit, 1e-12)
        within_limits = (
            largest_position <= self.config.max_position
            and delta_util <= 1.0
            and gamma_util <= 1.0
            and vega_util <= 1.0
        )
        return ExposureReport(
            net_delta=float(net_delta),
            net_gamma=float(net_gamma),
            net_vega=float(net_vega),
            net_vanna=float(net_vanna),
            gross_vega=float(gross_vega),
            largest_position=float(largest_position),
            delta_utilization=float(delta_util),
            gamma_utilization=float(gamma_util),
            vega_utilization=float(vega_util),
            within_limits=bool(within_limits),
        )

    def simulate_fill(self, strike: float, expiry: float, option_type: str) -> Fill | None:
        quote = self.quote_market(strike, expiry, option_type)
        side_probs: list[tuple[FillSide, float]] = []
        if np.isfinite(quote.bid_vol) and quote.fill_probability_bid > 0.0:
            side_probs.append(("bid", quote.fill_probability_bid))
        if np.isfinite(quote.ask_vol) and quote.fill_probability_ask > 0.0:
            side_probs.append(("ask", quote.fill_probability_ask))
        if not side_probs:
            return None

        total_prob = min(sum(prob for _, prob in side_probs), 0.95)
        if float(self.rng.random()) >= total_prob:
            return None

        draw = float(self.rng.random()) * sum(prob for _, prob in side_probs)
        cumulative = 0.0
        for side, prob in side_probs:
            cumulative += prob
            if draw <= cumulative:
                if side == "bid":
                    return Fill(
                        strike=float(strike),
                        expiry=float(_expiry_years(expiry)),
                        option_type=option_type,
                        side="bid",
                        quantity=float(quote.size),
                        price=float(quote.bid),
                        vol=float(quote.bid_vol),
                        probability=float(prob),
                    )
                return Fill(
                    strike=float(strike),
                    expiry=float(_expiry_years(expiry)),
                    option_type=option_type,
                    side="ask",
                    quantity=float(quote.size),
                    price=float(quote.ask),
                    vol=float(quote.ask_vol),
                    probability=float(prob),
                )
        return None

    def _half_spread(self, vega: float, gamma: float, days_to_expiry: int, open_interest: float) -> float:
        dte_term = self.config.short_dte_weight / np.sqrt(max(days_to_expiry, 1))
        oi_term = (
            self.config.low_oi_weight
            * self.config.min_half_spread_vol
            * (self.config.oi_reference / max(open_interest, 1.0))
        )
        base_spread = (
            self.config.base_half_spread_vol
            + self.config.base_vega_weight * abs(vega)
            + self.config.base_gamma_weight * abs(gamma) * self.surface.spot
            + dte_term
            + oi_term
        )
        risk_spread = self.config.k_vega * abs(vega) + self.config.k_gamma * abs(gamma) * self.surface.spot
        vega_util = abs(self.current_exposure().net_vega) / max(self.config.vega_limit, 1e-12)
        net_vega_penalty = self.config.inventory_penalty * vega_util
        return float(max(self.config.min_half_spread_vol, base_spread, risk_spread) + net_vega_penalty)

    def _inventory_mid_shift(self, option_greeks, exposure: ExposureReport) -> float:
        penalty_delta = (
            self.config.lambda_delta * (exposure.net_delta / max(self.config.delta_limit, 1e-12)) ** 2
        )
        penalty_gamma = (
            self.config.lambda_gamma * (exposure.net_gamma / max(self.config.gamma_limit, 1e-12)) ** 2
        )
        price_shift = 0.0
        if exposure.net_delta != 0.0 and option_greeks.delta != 0.0:
            price_shift -= np.sign(exposure.net_delta) * np.sign(option_greeks.delta) * penalty_delta
        if exposure.net_gamma != 0.0 and option_greeks.gamma != 0.0:
            price_shift -= np.sign(exposure.net_gamma) * np.sign(option_greeks.gamma) * penalty_gamma
        return float(price_shift / max(abs(option_greeks.vega), 1e-8))

    def _open_interest(self, strike: float, expiry: float, option_type: str) -> float:
        df = self.surface.chain_df
        mask = (
            np.isclose(df["strike"].to_numpy(dtype=np.float64), float(strike), atol=1e-8, rtol=0.0)
            & np.isclose(df["expiry"].to_numpy(dtype=np.float64), float(expiry), atol=1e-8, rtol=0.0)
            & (df["type"].to_numpy(dtype=object) == option_type)
        )
        if np.any(mask):
            oi = float(df.loc[mask, "open_interest"].iloc[0])
            return max(oi, 1.0)

        moneyness = max(float(strike) / max(self.surface.spot, 1e-12), 1e-12)
        x = np.log(moneyness)
        oi = self.config.oi_base * np.exp(-self.config.oi_decay * abs(x))
        if option_type == "put":
            oi *= 1.0 + self.config.oi_put_wing_boost * max(-x, 0.0)
        else:
            oi *= 1.0 + self.config.oi_call_wing_boost * max(x, 0.0)
        return max(float(oi), 1.0)

    def _contract_position(self, strike: float, expiry: float, option_type: str) -> float:
        total = 0.0
        for pos in self.portfolio.positions:
            if not getattr(pos, "inventory", True):
                continue
            if (
                pos.option_type == option_type
                and abs(pos.strike - strike) <= 1e-8
                and abs(pos.expiry - expiry) <= 1e-8
            ):
                total += pos.quantity
        return float(total)

    def _would_breach_limits(
        self,
        strike: float,
        expiry: float,
        option_type: str,
        quantity: float,
        premium: float | None = None,
    ) -> bool:
        current_contract = self._contract_position(strike, expiry, option_type)
        model_iv = float(self.surface.implied_vol(strike, expiry))
        greeks = bs_greeks(
            self.surface.spot,
            strike,
            expiry,
            self.surface.r,
            self.surface.q,
            model_iv,
            option_type,
        )
        contract_cap = self._kelly_position_cap(greeks.gamma)
        if abs(current_contract + quantity) > contract_cap:
            return True

        exposure = self.current_exposure()
        next_delta = exposure.net_delta + quantity * greeks.delta
        next_gamma = exposure.net_gamma + quantity * greeks.gamma
        next_vega = exposure.net_vega + quantity * greeks.vega
        limits_breached = (
            abs(next_delta) > self.config.delta_limit
            or abs(next_gamma) > self.config.gamma_limit
            or abs(next_vega) > self.config.vega_limit
        )
        if limits_breached:
            return True

        if self._stress_guard_active():
            current_worst = float(_portfolio_attr(self.portfolio, "last_worst_stress_pnl", 0.0))
            next_worst = self._trial_worst_stress_pnl(
                strike,
                expiry,
                option_type,
                quantity,
                premium=float(premium) if premium is not None else None,
            )
            if next_worst < current_worst - 1e-8:
                message = (
                    f"stress guard rejected fill strike={float(strike):.4f} expiry={float(expiry):.6f} "
                    f"type={option_type} qty={float(quantity):+.4f} current_worst={current_worst:.4f} "
                    f"next_worst={next_worst:.4f}"
                )
                logger.info(message)
                _append_portfolio_log(self.portfolio, "rejection_log", message)
                return True
        return False

    def _fill_probabilities(
        self,
        model_iv: float,
        bid_vol: float,
        ask_vol: float,
        open_interest: float,
    ) -> tuple[float, float]:
        oi_multiplier = min(
            1.0, (open_interest / max(self.config.oi_reference, 1.0)) ** self.config.fill_oi_exponent
        )
        scale = max(self.config.min_half_spread_vol, 1e-6)
        bid_prob = 0.0
        ask_prob = 0.0
        if np.isfinite(bid_vol):
            bid_distance = max(model_iv - bid_vol, 0.0)
            bid_prob = (
                self.config.fill_base_probability
                * np.exp(-self.config.fill_distance_sensitivity * bid_distance / scale)
                * oi_multiplier
            )
        if np.isfinite(ask_vol):
            ask_distance = max(ask_vol - model_iv, 0.0)
            ask_prob = (
                self.config.fill_base_probability
                * np.exp(-self.config.fill_distance_sensitivity * ask_distance / scale)
                * oi_multiplier
            )
        return float(min(max(bid_prob, 0.0), 0.95)), float(min(max(ask_prob, 0.0), 0.95))

    def _gamma_pull_bps(self, strike: float, expiry: float, book_gamma: float) -> float:
        days_to_expiry = float(expiry) * 365.0
        if days_to_expiry > float(self.config.gamma_target_max_dte_days):
            return 0.0

        spot = max(float(self.surface.spot), 1e-12)
        moneyness_gap = abs(float(strike) / spot - 1.0)
        if moneyness_gap > float(self.config.gamma_target_moneyness_band):
            return 0.0

        gamma_gap = float(self.config.target_net_gamma) - float(book_gamma)
        gamma_pull_bps = gamma_gap * float(self.config.gamma_sensitivity)
        if gamma_pull_bps > 0.0:
            gamma_pull_bps *= float(
                _portfolio_attr(self.portfolio, "gamma_bid_aggressiveness_multiplier", 1.0)
            )
        return float(gamma_pull_bps)

    def _kelly_position_cap(self, option_gamma: float) -> float:
        gamma_abs = abs(float(option_gamma))
        if gamma_abs <= 1e-12:
            return float(self.config.max_position)

        bankroll = _portfolio_bankroll(self.portfolio, float(self.surface.spot))
        cap = (
            float(self.config.kelly_fraction)
            * bankroll
            / max(
                gamma_abs * float(self.surface.spot) * float(self.surface.spot),
                1e-12,
            )
        )
        return float(min(float(self.config.max_position), max(cap, 0.0)))

    def _quote_size(self, strike: float, expiry: float, option_type: str, option_gamma: float) -> float:
        position_cap = self._kelly_position_cap(option_gamma)
        current_inventory = abs(self._contract_position(strike, expiry, option_type))
        remaining_capacity = max(position_cap - current_inventory, 0.0)
        return float(min(float(self.config.quote_size), remaining_capacity))

    def _stress_guard_active(self) -> bool:
        daily_theta = abs(float(_portfolio_attr(self.portfolio, "last_daily_theta", 0.0)))
        worst_stress = float(_portfolio_attr(self.portfolio, "last_worst_stress_pnl", 0.0))
        threshold = -float(self.config.stress_guard_multiple) * max(daily_theta, 1e-12)
        return worst_stress < threshold

    def _trial_worst_stress_pnl(
        self,
        strike: float,
        expiry: float,
        option_type: str,
        quantity: float,
        premium: float | None,
    ) -> float:
        from vol_surface_mm.core.stress import Shock, _shocked_surface

        model_iv = float(self.surface.implied_vol(strike, expiry))
        fill_premium = float(
            bs_price(self.surface.spot, strike, expiry, self.surface.r, self.surface.q, model_iv, option_type)
        )
        if premium is not None and np.isfinite(float(premium)):
            fill_premium = float(premium)

        # Reuse the most recent portfolio-level worst shock instead of running
        # a full scenario grid for every candidate fill. The guard only needs
        # to know whether this incremental trade worsens the already-binding
        # tail loss, so revaluing the option under the cached worst scenario is
        # both sufficient and much cheaper.
        shock = Shock(
            name="cached worst",
            spot_mult=float(_portfolio_attr(self.portfolio, "last_worst_spot_mult", 1.0)),
            vol_shift=float(_portfolio_attr(self.portfolio, "last_worst_vol_shift", 0.0)),
            skew_twist=float(_portfolio_attr(self.portfolio, "last_worst_skew_twist", 0.0)),
            term_slope=float(_portfolio_attr(self.portfolio, "last_worst_term_slope", 0.0)),
            vov_shift=float(_portfolio_attr(self.portfolio, "last_worst_vov_shift", 0.0)),
            rho_shift=float(_portfolio_attr(self.portfolio, "last_worst_rho_shift", 0.0)),
        )
        shock_key = (
            float(shock.spot_mult),
            float(shock.vol_shift),
            float(shock.skew_twist),
            float(shock.term_slope),
            float(shock.vov_shift),
            float(shock.rho_shift),
        )
        if self._cached_worst_shock_key != shock_key or self._cached_worst_shocked_surface is None:
            self._cached_worst_shock_key = shock_key
            self._cached_worst_shocked_surface = _shocked_surface(self.surface, shock)

        shocked_surface = self._cached_worst_shocked_surface
        shocked_spot = float(self.surface.spot) * float(shock.spot_mult)
        shocked_iv = float(shocked_surface.implied_vol(strike, expiry))
        shocked_price = float(
            bs_price(
                shocked_spot, strike, expiry, shocked_surface.r, shocked_surface.q, shocked_iv, option_type
            )
        )
        incremental_stress_pnl = float(quantity) * (shocked_price - fill_premium)
        current_worst = float(_portfolio_attr(self.portfolio, "last_worst_stress_pnl", 0.0))
        return float(current_worst + incremental_stress_pnl)


def quote_option(
    surface: VolSurface,
    strike: float,
    expiry: float,
    option_type: str,
    inventory_vega: float,
    realized_minus_implied_var: float,
    cfg: QuotingConfig,
) -> Quote:
    """Compatibility wrapper returning a price-space quote.

    ``inventory_vega`` and ``realized_minus_implied_var`` are retained for the
    older call path. The legacy wrapper only uses the vega term to emulate the
    previous spread widening behaviour; new code should use ``QuotingEngine``.
    """
    engine = QuotingEngine(surface, BookState(), cfg)
    quote = engine.quote_market(strike, expiry, option_type)
    extra_half_spread = cfg.inventory_penalty * abs(inventory_vega) + 0.25 * abs(realized_minus_implied_var)
    if extra_half_spread <= 0.0:
        return quote

    bid_vol = quote.bid_vol
    ask_vol = quote.ask_vol
    if np.isfinite(bid_vol):
        bid_vol = max(bid_vol - extra_half_spread, 1e-4)
    if np.isfinite(ask_vol):
        ask_vol = max(
            ask_vol + extra_half_spread, (bid_vol if np.isfinite(bid_vol) else quote.model_iv) + 1e-6
        )

    bid = (
        float("nan")
        if not np.isfinite(bid_vol)
        else float(
            bs_price(surface.spot, strike, _expiry_years(expiry), surface.r, surface.q, bid_vol, option_type)
        )
    )
    ask = (
        float("nan")
        if not np.isfinite(ask_vol)
        else float(
            bs_price(surface.spot, strike, _expiry_years(expiry), surface.r, surface.q, ask_vol, option_type)
        )
    )
    return Quote(
        strike=quote.strike,
        expiry=quote.expiry,
        option_type=quote.option_type,
        mid_price=quote.mid_price,
        bid=float(max(bid, 0.0)) if np.isfinite(bid) else float("nan"),
        ask=float(ask),
        model_iv=quote.model_iv,
        half_spread_vol=float(quote.half_spread_vol + extra_half_spread),
        size=quote.size,
        bid_vol=float(bid_vol),
        ask_vol=float(ask_vol),
        open_interest=quote.open_interest,
        fill_probability_bid=quote.fill_probability_bid,
        fill_probability_ask=quote.fill_probability_ask,
    )
