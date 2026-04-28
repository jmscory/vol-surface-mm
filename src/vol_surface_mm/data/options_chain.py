"""Synthetic option-chain builder.

For each spot tick the builder produces a full two-sided chain covering
strikes of +/- 30% around ATM in 2.5% increments and expiries of
``[7, 14, 30, 60, 90, 180]`` days-to-expiry. Each row holds
``(strike, expiry, type, mid_price, bid, ask, open_interest)`` plus
diagnostic columns (``moneyness``, ``iv``) used downstream.

The mid price is computed from Black-Scholes on a parametric smile
anchored at the current realized (or instantaneous) volatility; bid/ask
are a linear function of a half-spread-in-bps that widens into the
wings; open interest follows an exponential decay in ``|log(K/S)|`` so
ATM strikes carry the bulk of the book, with a small put/call skew to
mimic index-hedging flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_surface_mm.config import ChainConfig, MarketConfig
from vol_surface_mm.core.pricing import bs_price

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strike_grid(spot: float, range_pct: float, step_pct: float) -> np.ndarray:
    """Return strikes in ``spot * (1 +/- range_pct)`` at ``step_pct`` spacing."""
    if step_pct <= 0 or range_pct <= 0:
        raise ValueError("range_pct and step_pct must be positive")
    n = int(round(range_pct / step_pct))
    mults = 1.0 + np.arange(-n, n + 1) * step_pct
    return np.asarray(spot * mults, dtype=np.float64)


def _smile_iv(
    base_vol: float,
    moneyness: np.ndarray,
    t_years: np.ndarray,
    skew: float = -0.15,
    smile_curvature: float = 0.40,
    term: float = 0.05,
) -> np.ndarray:
    """Parametric IV as a function of ``(moneyness, T)``.

    ``base_vol`` should be the current realized or instantaneous vol.
    Returns a floor-clipped array matching the shape of the inputs.
    """
    x = np.log(moneyness)
    iv = base_vol + skew * x + smile_curvature * x * x + term * np.sqrt(t_years)
    return np.maximum(iv, 0.02)


def _half_spread_bps(
    moneyness: np.ndarray,
    atm_bps: float,
    wing_bps: float,
) -> np.ndarray:
    """Linear widening of half-spread in ``|log(moneyness)|``, in bps of mid."""
    x = np.abs(np.log(moneyness))
    x_max = np.log(1.30)
    frac = np.clip(x / max(x_max, 1e-12), 0.0, 1.0)
    return atm_bps + frac * (wing_bps - atm_bps)


def _open_interest(
    moneyness: np.ndarray,
    base: float,
    decay: float,
    option_type: str,
) -> np.ndarray:
    """Synthetic OI: exponential decay from ATM with put/call asymmetry.

    Puts get a modest boost in the downside wing; calls, a smaller boost
    in the upside wing - matching typical index/hedging OI patterns.
    """
    x = np.log(moneyness)
    core = base * np.exp(-decay * np.abs(x))
    if option_type == "put":
        core = core * (1.0 + 0.5 * np.clip(-x, 0.0, None))
    else:
        core = core * (1.0 + 0.2 * np.clip(x, 0.0, None))
    return np.round(core).astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_chain(
    spot: float,
    inst_vol: float,
    market: MarketConfig,
    chain_cfg: ChainConfig,
) -> pd.DataFrame:
    """Return an option chain DataFrame for the given spot.

    Parameters
    ----------
    spot:
        Current underlying price.
    inst_vol:
        Anchor volatility for the parametric smile. Pass 30-day realized
        vol in production; ``MarketConfig.sigma`` works as a fallback.
    market:
        :class:`MarketConfig` supplying ``r`` and ``q``.
    chain_cfg:
        :class:`ChainConfig` controlling strikes, expiries, spreads, OI.

    Returns
    -------
    pandas.DataFrame
        Columns: ``strike, expiry, type, mid_price, bid, ask,
        open_interest, moneyness, iv``.
        ``expiry`` is in years (``DTE / 365``). One row per
        ``(strike, expiry, {call, put})`` combination, sorted by
        ``(expiry, strike, type)``.
    """
    strikes = _strike_grid(
        spot,
        chain_cfg.strike_range_pct,
        chain_cfg.strike_step_pct,
    )
    expiries_yr = np.asarray(
        [dte / 365.0 for dte in chain_cfg.expiries_dte],
        dtype=np.float64,
    )

    # (n_expiry, n_strike) grids
    K, T = np.meshgrid(strikes, expiries_yr, indexing="xy")
    M = K / spot
    IV = _smile_iv(inst_vol, M, T)

    call_mid = bs_price(spot, K, T, market.r, market.q, IV, "call")
    put_mid = bs_price(spot, K, T, market.r, market.q, IV, "put")

    hs_bps = _half_spread_bps(
        M,
        chain_cfg.spread_bps_atm,
        chain_cfg.spread_bps_wing,
    )

    def _rows(mid: np.ndarray, option_type: str) -> pd.DataFrame:
        mid_flat = mid.reshape(-1)
        k_flat = K.reshape(-1)
        t_flat = T.reshape(-1)
        m_flat = M.reshape(-1)
        iv_flat = IV.reshape(-1)
        hs_flat = hs_bps.reshape(-1)

        half = mid_flat * (hs_flat * 1e-4)
        bid = np.maximum(mid_flat - half, 0.0)
        ask = mid_flat + half
        oi = _open_interest(
            m_flat,
            chain_cfg.oi_base,
            chain_cfg.oi_decay,
            option_type,
        )
        return pd.DataFrame(
            {
                "strike": k_flat,
                "expiry": t_flat,
                "type": option_type,
                "mid_price": mid_flat,
                "bid": bid,
                "ask": ask,
                "open_interest": oi,
                "moneyness": m_flat,
                "iv": iv_flat,
            }
        )

    df = pd.concat([_rows(call_mid, "call"), _rows(put_mid, "put")], ignore_index=True)
    return df.sort_values(["expiry", "strike", "type"]).reset_index(drop=True)
