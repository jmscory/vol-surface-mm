"""Delta-hedged portfolio manager with gamma/theta P&L tracking.

Public API
----------
Backward-compatible (used by legacy CLI, quoting.py, display.py):
    OptionPosition, BookState
    aggregate_greeks, rebalance_delta, mark_book
    StepPnL, attribute_pnl

New stateful portfolio-management API:
    Portfolio, BookGreeks, HedgeTrade, PnLReport
    PortfolioManager
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vol_surface_mm.config import HedgingConfig
from vol_surface_mm.core.pricing import Greeks, OptionQuote, bs_greeks
from vol_surface_mm.core.surface import VolSurface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared low-level primitives  (backward-compatible)
# ---------------------------------------------------------------------------


@dataclass
class OptionPosition:
    """A single option leg with signed quantity.

    Parameters
    ----------
    strike:      strike price in dollars
    expiry:      time to expiry in years
    option_type: ``'call'`` or ``'put'``
    quantity:    signed contract count; positive = long
    """

    strike: float
    expiry: float
    option_type: str
    quantity: float
    inventory: bool = True
    label: str = "inventory"


@dataclass
class BookState:
    """Legacy mutable book state used by the quoting engine and display layer.

    ``spot_position`` is the delta-hedge in underlying shares (signed).
    ``transaction_costs`` accumulates all realized hedging costs.
    """

    positions: list[OptionPosition] = field(default_factory=list)
    spot_position: float = 0.0
    cash: float = 0.0
    transaction_costs: float = 0.0

    def add(self, pos: OptionPosition, premium: float) -> None:
        """Append a position and debit/credit the cash account."""
        self.positions.append(pos)
        self.cash -= pos.quantity * premium


# ---------------------------------------------------------------------------
# New rich data-model types
# ---------------------------------------------------------------------------


@dataclass
class Portfolio:
    """Options book plus explicit delta-hedge share tracking.

    ``hedge_shares`` is the signed number of underlying shares held as the
    delta hedge.  The ``cash`` account is debited when shares are bought and
    credited when they are sold (including premium receipts/payments).
    """

    positions: list[OptionPosition] = field(default_factory=list)
    hedge_shares: float = 0.0
    cash: float = 0.0
    transaction_costs: float = 0.0
    bankroll: float = 100000.0
    var_capture_running: float = 0.0
    negative_var_capture_streak: int = 0
    vol_cheapness_boost_periods_remaining: int = 0
    gamma_bid_aggressiveness_multiplier: float = 1.0
    last_realized_vol_30d: float = 0.0
    last_atm_implied_vol: float = 0.0
    last_var_capture_step: float = 0.0
    last_worst_stress_pnl: float = 0.0
    last_worst_spot_mult: float = 1.0
    last_worst_vol_shift: float = 0.0
    last_worst_skew_twist: float = 0.0
    last_worst_term_slope: float = 0.0
    last_worst_vov_shift: float = 0.0
    last_worst_rho_shift: float = 0.0
    last_daily_theta: float = 0.0
    cum_gamma_pnl: float = 0.0
    cum_total_pnl: float = 0.0
    audit_trail: list[str] = field(default_factory=list)
    rejection_log: list[str] = field(default_factory=list)
    alert_log: list[str] = field(default_factory=list)
    sanity_checks_logged: bool = False
    stress_state_dirty: bool = True


@dataclass(frozen=True)
class BookGreeks:
    """Aggregate Greeks of the whole portfolio at a point in time.

    All values are dollar-denominated and signed (long = positive vega/gamma).

    Attributes
    ----------
    theta          : per calendar day (typically negative for long options)
    break_even_move: daily fractional spot move at which gamma P&L exactly
                     offsets the theta cost, computed as::

                         sqrt(2 * |theta_daily| / |gamma|) / spot

                     Zero when the gamma/theta pairing is not meaningful
                     (e.g. flat gamma, or same-sign gamma and theta).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    option_value: float
    total_value: float
    break_even_move: float


@dataclass(frozen=True)
class HedgeTrade:
    """Record of a single delta-hedge rebalancing trade.

    Attributes
    ----------
    shares    : signed share count; positive = bought, negative = sold
    price     : execution price (spot at the time of the trade)
    cost      : realized transaction cost in dollars
    prev_delta: net portfolio delta before the trade
    post_delta: net portfolio delta after the trade (target ≈ 0)
    """

    shares: float
    price: float
    cost: float
    prev_delta: float
    post_delta: float


@dataclass(frozen=True)
class PnLReport:
    """Full daily P&L attribution via second-order Taylor expansion.

    The decomposition is::

        dV ≈ delta·dS + 0.5·gamma·dS² + theta·dt + vega·dσ + residual

    where all greeks are taken at the *start* of the step (post-hedge).

    Attributes
    ----------
    realized_variance  : (d ln S)² / dt — annualised realised spot variance
    implied_variance   : ATM σ² from the surface at the start of the step
    variance_capture   : realized_variance − implied_variance
                         (positive means the realized move exceeded expectation;
                          good for long-gamma positions)
    exceeded_break_even: True when |dS/S| exceeded ``break_even_move``
    """

    step: int
    spot: float
    spot_prev: float
    iv_atm: float  # ATM implied vol, end of step
    iv_atm_prev: float  # ATM implied vol, start of step
    dt: float  # step length in years
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    residual_pnl: float
    realized_variance: float
    implied_variance: float
    variance_capture: float
    hedge_trade: HedgeTrade | None
    break_even_move: float
    exceeded_break_even: bool
    cum_delta_pnl: float
    cum_gamma_pnl: float
    cum_theta_pnl: float
    cum_vega_pnl: float
    cum_total_pnl: float


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------


class PortfolioManager:
    """Delta-hedged options book with step-by-step P&L decomposition.

    Intended usage::

        pm = PortfolioManager(cfg.hedging)
        for tick in feed:
            surface = build_surface(...)
            report  = pm.mark_to_market(surface, tick.spot)
            trade   = pm.delta_hedge(tick.spot)   # 'continuous' or 'discrete'

    ``mark_to_market`` must be called before ``delta_hedge`` on each step.
    The trade recorded by ``delta_hedge`` is attached to the *next* step's
    ``PnLReport``, and the saved MTM baseline is updated post-hedge so that
    the next step's P&L attribution is based on the post-hedge greeks
    (net delta ≈ 0).
    """

    def __init__(self, config: HedgingConfig) -> None:
        self.config = config
        self.portfolio = Portfolio(bankroll=float(config.initial_bankroll))

        # Surface and spot from last mark_to_market call
        self._surface: VolSurface | None = None
        self._spot: float | None = None

        # State for inter-step P&L attribution
        self._step: int = 0
        self._prev_greeks: BookGreeks | None = None
        self._prev_spot: float | None = None
        self._prev_iv_atm: float | None = None
        self._prev_mtm: float | None = None

        # Hedge trade to attach to the next PnLReport
        self._pending_trade: HedgeTrade | None = None

        # Cumulative P&L components
        self._cum_delta_pnl: float = 0.0
        self._cum_gamma_pnl: float = 0.0
        self._cum_theta_pnl: float = 0.0
        self._cum_vega_pnl: float = 0.0
        self._cum_total_pnl: float = 0.0
        self._rv30_returns: list[float] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add_position(
        self,
        quote: OptionQuote,
        quantity: int,
        side: str,
    ) -> None:
        """Add an option leg sourced from a pricing quote.

        Parameters
        ----------
        quote:    ``OptionQuote`` from :func:`~core.pricing.price_chain`
                  or :class:`~core.quoting.QuotingEngine`.
        quantity: Number of contracts; must be a positive integer.
        side:     ``'buy'`` (debit ask) or ``'sell'`` (credit bid).
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        quantity = int(quantity)
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer")

        signed_qty = float(quantity if side == "buy" else -quantity)
        price = float(quote.ask if side == "buy" else quote.bid)
        pos = OptionPosition(
            strike=float(quote.strike),
            expiry=float(quote.expiry_days) / 365.0,
            option_type=str(quote.option_type),
            quantity=signed_qty,
        )
        self.portfolio.positions.append(pos)
        self.portfolio.cash -= signed_qty * price
        self.portfolio.stress_state_dirty = True

    def delta_hedge(
        self,
        spot: float,
        method: Literal["continuous", "discrete"] = "continuous",
    ) -> HedgeTrade | None:
        """Compute and apply a delta-hedge rebalancing trade.

        Parameters
        ----------
        spot:   Current underlying price used as the execution price.
        method: ``'continuous'`` — trade whenever |net delta| exceeds
                ``config.rebalance_threshold``;
                ``'discrete'`` — trade unconditionally every
                ``config.discrete_hedge_interval`` steps.

        Returns
        -------
        :class:`HedgeTrade` if a trade was executed, ``None`` otherwise.

        Side effects
        ------------
        Updates ``portfolio.hedge_shares``, ``portfolio.cash``, and
        ``portfolio.transaction_costs``.  Also overwrites the stored
        previous MTM and greeks to the post-hedge values so that the
        *next* step's :meth:`mark_to_market` attributes P&L against a
        delta-neutral baseline.
        """
        if self._surface is None:
            return None
        if method not in ("continuous", "discrete"):
            raise ValueError(f"method must be 'continuous' or 'discrete', got {method!r}")

        greeks = self.compute_book_greeks()
        self._maybe_add_tail_hedge(greeks.delta, spot)
        self._maybe_add_gamma_protection(greeks.gamma, spot)
        greeks = self.compute_book_greeks()
        net_delta = greeks.delta

        should_hedge: bool
        if method == "continuous":
            should_hedge = abs(net_delta) > self.config.rebalance_threshold
        else:
            should_hedge = self._step % max(self.config.discrete_hedge_interval, 1) == 0

        if not should_hedge:
            return None

        shares_to_trade = -net_delta
        notional = abs(shares_to_trade) * spot
        cost = notional * self.config.transaction_cost_bps * 1e-4

        self.portfolio.hedge_shares += shares_to_trade
        self.portfolio.cash -= shares_to_trade * spot
        self.portfolio.cash -= cost
        self.portfolio.transaction_costs += cost
        self.portfolio.stress_state_dirty = True

        trade = HedgeTrade(
            shares=float(shares_to_trade),
            price=float(spot),
            cost=float(cost),
            prev_delta=float(net_delta),
            post_delta=float(net_delta + shares_to_trade),
        )
        self._pending_trade = trade

        # Update the saved baseline to the post-hedge portfolio so the next
        # mark_to_market P&L decomposition uses post-hedge (near-zero) delta.
        post_greeks = self.compute_book_greeks()
        self._prev_greeks = post_greeks
        self._prev_mtm = post_greeks.total_value
        return trade

    def compute_book_greeks(self) -> BookGreeks:
        """Aggregate Greeks across all positions at the current surface and spot.

        Requires :meth:`mark_to_market` to have been called at least once.

        Returns
        -------
        :class:`BookGreeks` with full first/second-order risk metrics and
        the gamma/theta break-even daily move.

        Notes
        -----
        The break-even daily move is::

            sqrt(2 * |theta_daily| / |gamma|) / spot

        and is non-zero only when the gamma/theta pairing is economically
        consistent (long gamma with negative theta, or short gamma with
        positive theta).
        """
        if self._surface is None:
            raise RuntimeError("No surface available. Call mark_to_market first.")

        surface = self._surface
        spot = float(self._spot if self._spot is not None else surface.spot)
        r, q = surface.r, surface.q

        net_delta = float(self.portfolio.hedge_shares)
        gamma = vega = theta = rho = vanna = volga = option_value = 0.0

        for pos in self.portfolio.positions:
            iv = float(surface.implied_vol(pos.strike, pos.expiry))
            g = bs_greeks(spot, pos.strike, pos.expiry, r, q, iv, pos.option_type)
            net_delta += pos.quantity * g.delta
            gamma += pos.quantity * g.gamma
            vega += pos.quantity * g.vega
            theta += pos.quantity * g.theta
            rho += pos.quantity * g.rho
            vanna += pos.quantity * g.vanna
            volga += pos.quantity * g.volga
            option_value += pos.quantity * g.price

        total_value = option_value + self.portfolio.hedge_shares * spot + self.portfolio.cash

        # Break-even daily move: sqrt(2 * |theta| / |gamma|) / spot
        # Meaningful only for consistent long-gamma or short-gamma positions.
        break_even_move = 0.0
        if abs(gamma) > 1e-12:
            long_gamma = gamma > 0.0 and theta < 0.0
            short_gamma = gamma < 0.0 and theta > 0.0
            if long_gamma or short_gamma:
                break_even_move = float(np.sqrt(2.0 * abs(theta) / abs(gamma)) / max(spot, 1e-12))

        return BookGreeks(
            delta=float(net_delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            vanna=float(vanna),
            volga=float(volga),
            option_value=float(option_value),
            total_value=float(total_value),
            break_even_move=float(break_even_move),
        )

    def mark_to_market(self, surface: VolSurface, spot: float) -> PnLReport:
        """Update the surface, compute P&L attribution, and advance the step.

        Must be called before :meth:`delta_hedge` on each simulation step.

        Parameters
        ----------
        surface: Newly fitted :class:`~core.surface.VolSurface`.
        spot:    Current underlying price.

        Returns
        -------
        :class:`PnLReport` with full P&L decomposition for this step.

        P&L decomposition
        -----------------
        Uses second-order Taylor expansion against start-of-step greeks::

            delta_pnl = prev_delta · dS
            gamma_pnl = 0.5 · prev_gamma · dS²
            theta_pnl = prev_theta · dt
            vega_pnl  = prev_vega  · dσ_atm
            residual  = total_pnl − sum(above)

        The step counter is incremented at the end of each call.
        """
        self._surface = surface
        self._spot = spot

        greeks = self.compute_book_greeks()
        log_ret_for_tracker: float | None = None
        if self._prev_spot is not None:
            log_ret_for_tracker = float(np.log(max(spot, 1e-12) / max(float(self._prev_spot), 1e-12)))

        # ATM vol at the midpoint expiry (stable representative slice).
        mid_idx = len(surface.expiries) // 2
        iv_atm = float(surface.implied_vol(spot, float(surface.expiries[mid_idx])))

        current_mtm = greeks.total_value
        dt = self.config.dt

        # --- build PnL report ---
        if self._prev_greeks is None:
            # First observation: no P&L yet.
            report = PnLReport(
                step=self._step,
                spot=float(spot),
                spot_prev=float(spot),
                iv_atm=float(iv_atm),
                iv_atm_prev=float(iv_atm),
                dt=float(dt),
                total_pnl=0.0,
                delta_pnl=0.0,
                gamma_pnl=0.0,
                theta_pnl=0.0,
                vega_pnl=0.0,
                residual_pnl=0.0,
                realized_variance=0.0,
                implied_variance=float(iv_atm**2),
                variance_capture=0.0,
                hedge_trade=self._pending_trade,
                break_even_move=float(greeks.break_even_move),
                exceeded_break_even=False,
                cum_delta_pnl=0.0,
                cum_gamma_pnl=0.0,
                cum_theta_pnl=0.0,
                cum_vega_pnl=0.0,
                cum_total_pnl=0.0,
            )
        else:
            prev_g = self._prev_greeks
            prev_spot = float(self._prev_spot)  # type: ignore[arg-type]
            prev_iv_atm = float(self._prev_iv_atm)  # type: ignore[arg-type]
            prev_mtm = float(self._prev_mtm)  # type: ignore[arg-type]

            ds = spot - prev_spot
            dv = iv_atm - prev_iv_atm
            total_pnl = current_mtm - prev_mtm

            delta_pnl = prev_g.delta * ds
            gamma_pnl = 0.5 * prev_g.gamma * ds * ds
            theta_pnl = prev_g.theta * dt  # theta already per day; dt converts to step
            vega_pnl = prev_g.vega * dv
            residual_pnl = total_pnl - delta_pnl - gamma_pnl - theta_pnl - vega_pnl

            # Realized variance: (d ln S)² / dt, annualised.
            log_ret = float(np.log(max(spot, 1e-12) / max(prev_spot, 1e-12)))
            realized_variance = (log_ret**2) / max(dt, 1e-12)
            implied_variance = prev_iv_atm**2
            variance_capture = realized_variance - implied_variance

            # Did the realised move exceed the break-even threshold?
            realized_frac = abs(ds) / max(prev_spot, 1e-12)
            exceeded = prev_g.break_even_move > 0.0 and realized_frac > prev_g.break_even_move

            self._cum_delta_pnl += delta_pnl
            self._cum_gamma_pnl += gamma_pnl
            self._cum_theta_pnl += theta_pnl
            self._cum_vega_pnl += vega_pnl
            self._cum_total_pnl += total_pnl

            report = PnLReport(
                step=self._step,
                spot=float(spot),
                spot_prev=float(prev_spot),
                iv_atm=float(iv_atm),
                iv_atm_prev=float(prev_iv_atm),
                dt=float(dt),
                total_pnl=float(total_pnl),
                delta_pnl=float(delta_pnl),
                gamma_pnl=float(gamma_pnl),
                theta_pnl=float(theta_pnl),
                vega_pnl=float(vega_pnl),
                residual_pnl=float(residual_pnl),
                realized_variance=float(realized_variance),
                implied_variance=float(implied_variance),
                variance_capture=float(variance_capture),
                hedge_trade=self._pending_trade,
                break_even_move=float(prev_g.break_even_move),
                exceeded_break_even=bool(exceeded),
                cum_delta_pnl=float(self._cum_delta_pnl),
                cum_gamma_pnl=float(self._cum_gamma_pnl),
                cum_theta_pnl=float(self._cum_theta_pnl),
                cum_vega_pnl=float(self._cum_vega_pnl),
                cum_total_pnl=float(self._cum_total_pnl),
            )

        # Persist current-step state as baseline for next step.
        # (delta_hedge may overwrite these with post-hedge values.)
        self._prev_greeks = greeks
        self._prev_spot = spot
        self._prev_iv_atm = iv_atm
        self._prev_mtm = current_mtm
        self._pending_trade = None
        self.portfolio.cum_gamma_pnl = float(report.cum_gamma_pnl)
        self.portfolio.cum_total_pnl = float(report.cum_total_pnl)
        self.portfolio.last_daily_theta = float(abs(greeks.theta))
        self._update_var_capture_state(iv_atm, greeks, log_ret_for_tracker)
        # Stress refresh is expensive; only recompute it when the book has
        # actually changed so the quoter always sees a current tail snapshot
        # without paying for redundant scenario grids.
        if self._step == 0 or self.portfolio.stress_state_dirty:
            self._refresh_stress_state(surface, spot)
            self.portfolio.stress_state_dirty = False
        self._step += 1

        return report

    def _append_runtime_log(self, field_name: str, message: str) -> None:
        log = getattr(self.portfolio, field_name)
        log.append(message)
        if len(log) > 512:
            del log[:-512]

    def _update_var_capture_state(
        self,
        iv_atm: float,
        greeks: BookGreeks,
        log_ret: float | None,
    ) -> None:
        # Keep a rolling realised-vol window inside the hedger so the quoter
        # can react to variance carry without needing to know about the feed.
        if log_ret is not None:
            self._rv30_returns.append(float(log_ret))
            if len(self._rv30_returns) > self.config.var_capture_window_days:
                self._rv30_returns.pop(0)

        rv30_var = 0.0
        if len(self._rv30_returns) >= 2:
            rv30_var = float(np.var(self._rv30_returns, ddof=1) / max(self.config.dt, 1e-12))
        realized_vol_30d = float(np.sqrt(max(rv30_var, 0.0)))

        self.portfolio.last_atm_implied_vol = float(iv_atm)
        self.portfolio.last_realized_vol_30d = float(realized_vol_30d)

        if log_ret is None:
            self.portfolio.last_var_capture_step = 0.0
            self.portfolio.gamma_bid_aggressiveness_multiplier = (
                float(self.config.cheapness_boost_multiplier)
                if self.portfolio.vol_cheapness_boost_periods_remaining > 0
                else 1.0
            )
            return

        var_capture_step = float((iv_atm**2 - realized_vol_30d**2) * greeks.vega * self.config.dt)
        self.portfolio.last_var_capture_step = var_capture_step
        self.portfolio.var_capture_running += var_capture_step

        if self.portfolio.var_capture_running < 0.0:
            self.portfolio.negative_var_capture_streak += 1
        else:
            self.portfolio.negative_var_capture_streak = 0

        if self.portfolio.vol_cheapness_boost_periods_remaining > 0:
            self.portfolio.negative_var_capture_streak = 0
            self.portfolio.vol_cheapness_boost_periods_remaining -= 1
        elif self.portfolio.negative_var_capture_streak >= self.config.cheapness_negative_streak:
            self.portfolio.vol_cheapness_boost_periods_remaining = self.config.cheapness_boost_periods
            self.portfolio.negative_var_capture_streak = 0
            message = (
                "vol cheapness alert: running variance capture stayed negative for "
                f"{self.config.cheapness_negative_streak} periods; boosting gamma bids by 20%"
            )
            logger.warning(message)
            self._append_runtime_log("alert_log", message)

        self.portfolio.gamma_bid_aggressiveness_multiplier = (
            float(self.config.cheapness_boost_multiplier)
            if self.portfolio.vol_cheapness_boost_periods_remaining > 0
            else 1.0
        )

    def _refresh_stress_state(self, surface: VolSurface, spot: float) -> None:
        from vol_surface_mm.core.stress import run_all_scenarios

        # Refresh one canonical downside snapshot per step. The quoter reuses
        # this cached worst shock when deciding whether an incremental fill
        # would push the book beyond the stress guardrail.
        reports = run_all_scenarios(self.portfolio, surface, spot, surface.r, surface.q)
        worst_report = min(reports, key=lambda report: float(report.pnl_impact), default=None)
        if worst_report is None:
            self.portfolio.last_worst_stress_pnl = 0.0
            self.portfolio.last_worst_spot_mult = 1.0
            self.portfolio.last_worst_vol_shift = 0.0
            self.portfolio.last_worst_skew_twist = 0.0
            self.portfolio.last_worst_term_slope = 0.0
            self.portfolio.last_worst_vov_shift = 0.0
            self.portfolio.last_worst_rho_shift = 0.0
            return

        self.portfolio.last_worst_stress_pnl = float(worst_report.pnl_impact)
        self.portfolio.last_worst_spot_mult = 1.0 + float(worst_report.spot_shock_pct) / 100.0
        self.portfolio.last_worst_vol_shift = float(worst_report.vol_shock_pts) / 100.0
        self.portfolio.last_worst_skew_twist = float(worst_report.skew_twist)
        self.portfolio.last_worst_term_slope = float(worst_report.term_slope)
        self.portfolio.last_worst_vov_shift = float(worst_report.vov_shift)
        self.portfolio.last_worst_rho_shift = float(worst_report.rho_shift)

    def _select_tail_hedge_contract(self, spot: float) -> tuple[float, float, object]:
        if self._surface is None:
            raise RuntimeError("tail hedge selection requires a live surface")

        expiry_days = np.asarray(self._surface.expiries_days, dtype=np.float64)
        expiry_idx = int(np.argmin(np.abs(expiry_days - float(self.config.tail_hedge_dte_days))))
        expiry = float(self._surface.expiries[expiry_idx])

        best_score = float("inf")
        best_strike = float(self._surface.strikes[0])
        best_greeks = None
        for strike in np.asarray(self._surface.strikes, dtype=np.float64):
            iv = float(self._surface.implied_vol(float(strike), expiry))
            greeks = bs_greeks(spot, float(strike), expiry, self._surface.r, self._surface.q, iv, "put")
            score = abs(abs(float(greeks.delta)) - float(self.config.tail_hedge_put_delta))
            if score < best_score:
                best_score = score
                best_strike = float(strike)
                best_greeks = greeks

        if best_greeks is None:
            raise RuntimeError("failed to locate a tail hedge contract")
        return expiry, best_strike, best_greeks

    def _maybe_add_tail_hedge(self, net_delta: float, spot: float) -> None:
        trigger_delta = float(self.config.tail_hedge_delta_trigger_ratio * self.config.delta_limit)
        if net_delta <= trigger_delta or self._surface is None:
            return

        # Tail hedges are booked outside normal inventory limits so the market
        # maker can protect downside delta without consuming quoting capacity.
        expiry, strike, hedge_greeks = self._select_tail_hedge_contract(spot)
        target_qty = float(net_delta / max(self.config.tail_hedge_put_delta, 1e-12))
        existing_qty = sum(
            pos.quantity
            for pos in self.portfolio.positions
            if not pos.inventory and pos.label == "tail_hedge" and pos.option_type == "put"
        )
        qty_to_buy = max(target_qty - existing_qty, 0.0)
        if qty_to_buy <= 1e-8:
            return

        premium = float(hedge_greeks.price)
        self.portfolio.positions.append(
            OptionPosition(
                strike=float(strike),
                expiry=float(expiry),
                option_type="put",
                quantity=float(qty_to_buy),
                inventory=False,
                label="tail_hedge",
            )
        )
        self.portfolio.cash -= qty_to_buy * premium
        self.portfolio.stress_state_dirty = True

        message = (
            f"tail hedge added: qty={qty_to_buy:.2f} strike={strike:.4f} expiry={expiry:.6f} "
            f"trigger_delta={net_delta:.4f}"
        )
        logger.info(message)
        self._append_runtime_log("alert_log", message)

    def _select_gamma_protection_contracts(self, spot: float) -> tuple[float, float, float, Greeks, Greeks]:
        """Pick a long OTM strangle near ``gamma_protection_strangle_delta``."""
        if self._surface is None:
            raise RuntimeError("surface is required for gamma protection")

        target_dte = float(self.config.gamma_protection_dte_days) / 365.0
        expiry_idx = int(np.argmin(np.abs(self._surface.expiries - target_dte)))
        expiry = float(self._surface.expiries[expiry_idx])

        target_delta = float(self.config.gamma_protection_strangle_delta)
        best_put_score = float("inf")
        best_call_score = float("inf")
        best_put_strike = float(self._surface.strikes[0])
        best_call_strike = float(self._surface.strikes[-1])
        best_put_greeks: Greeks | None = None
        best_call_greeks: Greeks | None = None

        for strike in np.asarray(self._surface.strikes, dtype=np.float64):
            iv = float(self._surface.implied_vol(float(strike), expiry))
            put_g = bs_greeks(spot, float(strike), expiry, self._surface.r, self._surface.q, iv, "put")
            call_g = bs_greeks(spot, float(strike), expiry, self._surface.r, self._surface.q, iv, "call")
            put_score = abs(abs(float(put_g.delta)) - target_delta)
            call_score = abs(abs(float(call_g.delta)) - target_delta)
            if put_score < best_put_score and float(strike) < spot:
                best_put_score = put_score
                best_put_strike = float(strike)
                best_put_greeks = put_g
            if call_score < best_call_score and float(strike) > spot:
                best_call_score = call_score
                best_call_strike = float(strike)
                best_call_greeks = call_g

        if best_put_greeks is None or best_call_greeks is None:
            raise RuntimeError("failed to locate gamma protection contracts")
        return expiry, best_put_strike, best_call_strike, best_put_greeks, best_call_greeks

    def _maybe_add_gamma_protection(self, net_gamma: float, spot: float) -> None:
        """Buy a long OTM strangle when the book is significantly short gamma.

        Gamma protection fires when ``net_gamma < -gamma_protection_trigger``.
        Each leg of the strangle is sized so that its combined gamma roughly
        closes the gap to zero, capped at ``gamma_protection_max_lots`` per
        leg.  Hedge positions are tagged ``label="gamma_protection"`` and
        booked outside normal inventory limits, mirroring
        :meth:`_maybe_add_tail_hedge`.
        """
        if self._surface is None:
            return
        trigger = float(self.config.gamma_protection_trigger)
        if trigger <= 0.0:
            return
        threshold = -trigger
        if net_gamma >= threshold:
            return

        expiry, put_strike, call_strike, put_g, call_g = self._select_gamma_protection_contracts(spot)
        leg_gamma = max(float(put_g.gamma) + float(call_g.gamma), 1e-12)
        gamma_gap = max(-net_gamma, 0.0)
        target_qty = float(gamma_gap / leg_gamma)
        max_lots = float(self.config.gamma_protection_max_lots)

        existing_qty = (
            sum(
                pos.quantity
                for pos in self.portfolio.positions
                if not pos.inventory and pos.label == "gamma_protection"
            )
            / 2.0
        )  # two legs per protection trade
        qty_to_buy = max(min(target_qty, max_lots) - existing_qty, 0.0)
        if qty_to_buy <= 1e-8:
            return

        put_premium = float(put_g.price)
        call_premium = float(call_g.price)
        self.portfolio.positions.extend(
            [
                OptionPosition(
                    strike=float(put_strike),
                    expiry=float(expiry),
                    option_type="put",
                    quantity=float(qty_to_buy),
                    inventory=False,
                    label="gamma_protection",
                ),
                OptionPosition(
                    strike=float(call_strike),
                    expiry=float(expiry),
                    option_type="call",
                    quantity=float(qty_to_buy),
                    inventory=False,
                    label="gamma_protection",
                ),
            ]
        )
        self.portfolio.cash -= qty_to_buy * (put_premium + call_premium)
        self.portfolio.stress_state_dirty = True

        message = (
            f"gamma protection added: qty={qty_to_buy:.2f} put_strike={put_strike:.4f} "
            f"call_strike={call_strike:.4f} expiry={expiry:.6f} net_gamma={net_gamma:.4f} "
            f"threshold={threshold:.4f}"
        )
        logger.info(message)
        self._append_runtime_log("alert_log", message)


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers (used by legacy CLI / quoting.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepPnL:
    """Simple P&L decomposition container (legacy, used by attribute_pnl)."""

    total: float
    delta: float
    gamma: float
    theta: float
    vega: float
    residual: float


def aggregate_greeks(
    book: BookState,
    surface: VolSurface,
    spot: float,
) -> dict[str, float]:
    """Sum Greeks across all option positions plus the spot hedge.

    Parameters
    ----------
    book:    :class:`BookState` holding option positions and spot hedge.
    surface: Fitted :class:`~core.surface.VolSurface` for IV lookup.
    spot:    Current underlying price.

    Returns
    -------
    dict with keys ``delta``, ``gamma``, ``vega``, ``theta``, ``value``.
    """
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    value = 0.0
    r, q = surface.r, surface.q
    for pos in book.positions:
        iv = surface.implied_vol(pos.strike, pos.expiry)
        g = bs_greeks(spot, pos.strike, pos.expiry, r, q, iv, pos.option_type)
        delta += pos.quantity * g.delta
        gamma += pos.quantity * g.gamma
        vega += pos.quantity * g.vega
        theta += pos.quantity * g.theta
        value += pos.quantity * g.price
    delta += book.spot_position
    value += book.spot_position * spot
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "value": value}


def rebalance_delta(
    book: BookState,
    surface: VolSurface,
    spot: float,
    cfg: HedgingConfig,
) -> float:
    """Trade spot to bring |net delta| below the rebalance threshold.

    Parameters
    ----------
    book:    Mutable :class:`BookState`; modified in place.
    surface: Current :class:`~core.surface.VolSurface`.
    spot:    Execution price for the hedge trade.
    cfg:     :class:`~config.HedgingConfig` supplying the threshold and costs.

    Returns
    -------
    Number of shares traded (signed; positive = bought).
    """
    greeks = aggregate_greeks(book, surface, spot)
    delta = greeks["delta"]
    if abs(delta) < cfg.rebalance_threshold:
        return 0.0
    shares_to_trade = -delta
    book.spot_position += shares_to_trade
    notional = abs(shares_to_trade) * spot
    cost = notional * cfg.transaction_cost_bps * 1e-4
    book.cash -= shares_to_trade * spot
    book.cash -= cost
    book.transaction_costs += cost
    return shares_to_trade


def attribute_pnl(
    prev_greeks: dict[str, float],
    prev_spot: float,
    prev_iv_mean: float,
    new_spot: float,
    new_iv_mean: float,
    dt: float,
    pnl_total: float,
) -> StepPnL:
    """Decompose realized P&L into first/second-order Greek contributions.

    Parameters
    ----------
    prev_greeks:    dict from :func:`aggregate_greeks` at the previous step.
    prev_spot:      Underlying price at the start of the step.
    prev_iv_mean:   Mean ATM IV at the start of the step.
    new_spot:       Underlying price at the end of the step.
    new_iv_mean:    Mean ATM IV at the end of the step.
    dt:             Step length in years.
    pnl_total:      Realized total P&L for the step.

    Returns
    -------
    :class:`StepPnL` with individual Greek contributions and residual.
    """
    ds = new_spot - prev_spot
    dv = new_iv_mean - prev_iv_mean
    delta_pnl = prev_greeks["delta"] * ds
    gamma_pnl = 0.5 * prev_greeks["gamma"] * ds * ds
    theta_pnl = prev_greeks["theta"] * dt
    vega_pnl = prev_greeks["vega"] * dv
    residual = pnl_total - (delta_pnl + gamma_pnl + theta_pnl + vega_pnl)
    return StepPnL(
        total=pnl_total,
        delta=delta_pnl,
        gamma=gamma_pnl,
        theta=theta_pnl,
        vega=vega_pnl,
        residual=residual,
    )


def mark_book(
    book: BookState,
    surface: VolSurface,
    spot: float,
) -> float:
    """Mark-to-market value of the book: options + spot hedge + cash.

    Parameters
    ----------
    book:    :class:`BookState` to value.
    surface: Fitted :class:`~core.surface.VolSurface`.
    spot:    Current underlying price.

    Returns
    -------
    Total MTM value in dollars.
    """
    g = aggregate_greeks(book, surface, spot)
    return g["value"] + book.cash
