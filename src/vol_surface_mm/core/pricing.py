"""Black-Scholes-Merton pricing engine with continuous dividend yield.

All array-level computations are vectorised over numpy arrays; inputs must be
broadcast-compatible.  No external pricing libraries are used.

Public API
----------
bs_price(s, k, t, r, q, sigma, option_type)  -> np.ndarray
bs_greeks(s, k, t, r, q, sigma, option_type) -> Greeks
implied_vol(price, s, k, t, r, q, option_type) -> float
price_chain(...)                              -> list[OptionQuote]

Greeks.vega is the raw dC/dsigma (per 1.0 unit of vol), preserved for
backward compatibility with callers that compare against FD vega.
OptionQuote.vega is scaled to per 1% vol move (raw x 0.01).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

_DAYS_PER_YEAR: float = 365.0


@dataclass(frozen=True)
class Greeks:
    """Closed-form first- and second-order Greeks for a single option.

    ``vega``  -- raw dC/dsigma, per 1.0 unit of vol.
    ``theta`` -- dC/dt per calendar day (typically negative for long options).
    """

    price: float
    delta: float
    gamma: float
    vega: float  # dC/dsigma, per 1.0 unit of vol
    theta: float  # per calendar day
    rho: float
    vanna: float  # d2C/dS dsigma
    volga: float  # d2C/dsigma^2  (Vomma)


# Alias expected by newer calling code.
GreeksResult = Greeks


@dataclass
class OptionQuote:
    """Full pricing snapshot for a single option contract.

    ``vega``  -- per 1% vol move (raw dC/dsigma x 0.01).
    ``theta`` -- per calendar day.
    """

    strike: float
    expiry_days: int
    option_type: str
    theo: float
    bid: float
    ask: float
    delta: float
    gamma: float
    vega: float  # per 1% vol move
    theta: float  # per calendar day
    vanna: float
    volga: float
    implied_vol: float


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _d1_d2(
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: float,
    q: float,
    sigma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d1 and d2 with guarded denominators to avoid NaN at T->0, sigma->0."""
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.maximum(np.asarray(t, dtype=np.float64), 1e-12)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


def bs_price(
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: float,
    q: float,
    sigma: np.ndarray | float,
    option_type: OptionType = "call",
) -> np.ndarray:
    """BSM European price with continuous dividend yield *q*.

    Put price is derived via put-call parity to guarantee consistency:
    P = C - S*e^(-qT) + K*e^(-rT)
    """
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    t_arr = np.maximum(np.asarray(t, dtype=np.float64), 1e-12)
    s_arr = np.asarray(s, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    disc_r = np.exp(-r * t_arr)
    disc_q = np.exp(-q * t_arr)
    call = s_arr * disc_q * norm.cdf(d1) - k_arr * disc_r * norm.cdf(d2)
    if option_type == "call":
        return call
    return call - s_arr * disc_q + k_arr * disc_r  # put-call parity


# ---------------------------------------------------------------------------
# Scalar Greek helpers
# ---------------------------------------------------------------------------


def bs_delta(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """dC/dS (call) or dP/dS (put)."""
    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    disc_q = np.exp(-q * max(t, 1e-12))
    if option_type == "call":
        return float(disc_q * norm.cdf(d1))
    return float(disc_q * (norm.cdf(d1) - 1.0))


def bs_gamma(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """d2C/dS^2 -- identical for calls and puts."""
    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    t_safe = max(t, 1e-12)
    sig_safe = max(sigma, 1e-12)
    disc_q = np.exp(-q * t_safe)
    return float(disc_q * norm.pdf(d1) / (s * sig_safe * np.sqrt(t_safe)))


def bs_vega(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Raw vega dC/dsigma per 1.0 unit of vol -- identical for calls and puts."""
    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    t_safe = max(t, 1e-12)
    disc_q = np.exp(-q * t_safe)
    return float(s * disc_q * norm.pdf(d1) * np.sqrt(t_safe))


def bs_theta(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Theta per calendar day -- typically negative for long options."""
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    t_safe = max(t, 1e-12)
    disc_r = np.exp(-r * t_safe)
    disc_q = np.exp(-q * t_safe)
    decay = -s * disc_q * norm.pdf(d1) * sigma / (2.0 * np.sqrt(t_safe))
    if option_type == "call":
        annual = float(decay - r * k * disc_r * norm.cdf(d2) + q * s * disc_q * norm.cdf(d1))
    else:
        annual = float(decay + r * k * disc_r * norm.cdf(-d2) - q * s * disc_q * norm.cdf(-d1))
    return annual / _DAYS_PER_YEAR


def bs_rho(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """dC/dr or dP/dr."""
    _, d2 = _d1_d2(s, k, t, r, q, sigma)
    t_safe = max(t, 1e-12)
    disc_r = np.exp(-r * t_safe)
    if option_type == "call":
        return float(k * t_safe * disc_r * norm.cdf(d2))
    return float(-k * t_safe * disc_r * norm.cdf(-d2))


def bs_vanna(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Vanna d2C/dS dsigma = -e^(-qT)*N'(d1)*d2/sigma -- same for calls and puts.

    Derivation: dd1/dsigma = -d2/sigma, so
    d(delta_call)/dsigma = e^(-qT)*N'(d1)*(-d2/sigma).
    """
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    t_safe = max(t, 1e-12)
    sig_safe = max(sigma, 1e-12)
    disc_q = np.exp(-q * t_safe)
    return float(-disc_q * norm.pdf(d1) * d2 / sig_safe)


def bs_volga(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Volga (Vomma) d2C/dsigma^2 = vega*d1*d2/sigma -- same for calls and puts."""
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    sig_safe = max(sigma, 1e-12)
    return float(bs_vega(s, k, t, r, q, sigma) * d1 * d2 / sig_safe)


def bs_greeks(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType = "call",
) -> Greeks:
    """All seven closed-form Greeks plus price in a single call.

    ``Greeks.vega`` is raw dC/dsigma (per 1.0 unit of vol) to preserve
    backward-compatibility with callers that compare against FD vega.
    """
    return Greeks(
        price=float(bs_price(s, k, t, r, q, sigma, option_type)),
        delta=bs_delta(s, k, t, r, q, sigma, option_type),
        gamma=bs_gamma(s, k, t, r, q, sigma),
        vega=bs_vega(s, k, t, r, q, sigma),
        theta=bs_theta(s, k, t, r, q, sigma, option_type),
        rho=bs_rho(s, k, t, r, q, sigma, option_type),
        vanna=bs_vanna(s, k, t, r, q, sigma),
        volga=bs_volga(s, k, t, r, q, sigma),
    )


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    option_type: OptionType = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Invert BSM for sigma via Brent's method; bracket [1e-6, 5.0].

    Returns ``nan`` when:
    * the price is outside no-arbitrage bounds, or
    * the root-finder fails to bracket a solution.
    """
    t_safe = max(t, 1e-12)
    disc_r = np.exp(-r * t_safe)
    disc_q = np.exp(-q * t_safe)
    s_f, k_f = float(s), float(k)
    intrinsic = (
        max(s_f * disc_q - k_f * disc_r, 0.0)
        if option_type == "call"
        else max(k_f * disc_r - s_f * disc_q, 0.0)
    )
    upper = s_f * disc_q if option_type == "call" else k_f * disc_r
    if price < intrinsic - tol or price > upper + tol:
        return float("nan")

    def _obj(sigma: float) -> float:
        return float(bs_price(s_f, k_f, t_safe, r, q, sigma, option_type)) - price

    try:
        sigma = brentq(_obj, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return float("nan")

    # Newton polish (improves accuracy near expiry or extreme strikes).
    for _ in range(5):
        v = bs_vega(s_f, k_f, t_safe, r, q, sigma)
        if v < 1e-10:
            break
        diff = _obj(sigma)
        step = diff / v
        sigma_new = sigma - step
        if sigma_new <= 0.0 or abs(step) < tol:
            break
        sigma = sigma_new
    return float(sigma)


# ---------------------------------------------------------------------------
# Vectorised chain pricer
# ---------------------------------------------------------------------------


def price_chain(
    S: float,
    strikes: np.ndarray,
    expiry_years: np.ndarray,
    expiry_days: np.ndarray,
    option_types: np.ndarray,
    vols: np.ndarray,
    r: float,
    q: float,
    spreads: np.ndarray | None = None,
) -> list[OptionQuote]:
    """Price a full option chain in one vectorised numpy pass.

    Parameters
    ----------
    S            : current spot price
    strikes      : (N,) strike prices
    expiry_years : (N,) time to expiry in years
    expiry_days  : (N,) calendar days to expiry (stored verbatim)
    option_types : (N,) object array of ``"call"`` / ``"put"`` strings
    vols         : (N,) model implied volatilities (sigma)
    r            : risk-free rate (continuous, annualised)
    q            : dividend yield (continuous, annualised)
    spreads      : (N,) per-option half-spread in dollars; zero if *None*

    Returns
    -------
    list[OptionQuote]
        One :class:`OptionQuote` per input option.  All greek math is
        vectorised; only the final dataclass construction iterates.
    """
    k = np.asarray(strikes, dtype=np.float64)
    t = np.maximum(np.asarray(expiry_years, dtype=np.float64), 1e-12)
    days = np.asarray(expiry_days, dtype=np.int64)
    sig = np.maximum(np.asarray(vols, dtype=np.float64), 1e-12)
    otypes = np.asarray(option_types)
    is_call = otypes == "call"
    hs = np.zeros(len(k)) if spreads is None else np.asarray(spreads, dtype=np.float64)

    # ---- discount factors ------------------------------------------------
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    sqrt_t = np.sqrt(t)

    # ---- d1, d2 ----------------------------------------------------------
    d1 = (np.log(S / k) + (r - q + 0.5 * sig * sig) * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t

    # ---- CDF / PDF -------------------------------------------------------
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    nnd2 = 1.0 - nd2  # norm.cdf(-d2)
    npd1 = norm.pdf(d1)

    # ---- price (put via put-call parity) ---------------------------------
    call_px = S * disc_q * nd1 - k * disc_r * nd2
    put_px = call_px - S * disc_q + k * disc_r
    theo = np.where(is_call, call_px, put_px)

    # ---- delta -----------------------------------------------------------
    delta = np.where(is_call, disc_q * nd1, disc_q * (nd1 - 1.0))

    # ---- gamma (call = put) ----------------------------------------------
    gamma = disc_q * npd1 / (S * sig * sqrt_t)

    # ---- vega: raw dC/dsigma; OptionQuote uses per-1%-move scaling -------
    vega_raw = S * disc_q * npd1 * sqrt_t
    vega_pct = vega_raw * 0.01

    # ---- theta per calendar day ------------------------------------------
    decay = -S * disc_q * npd1 * sig / (2.0 * sqrt_t)
    call_theta_ann = decay - r * k * disc_r * nd2 + q * S * disc_q * nd1
    put_theta_ann = decay + r * k * disc_r * nnd2 - q * S * disc_q * (1.0 - nd1)
    theta = np.where(is_call, call_theta_ann, put_theta_ann) / _DAYS_PER_YEAR

    # ---- vanna: -e^(-qT)*N'(d1)*d2/sigma  (call = put) ------------------
    vanna = -disc_q * npd1 * d2 / sig

    # ---- volga: vega*d1*d2/sigma  (call = put) ---------------------------
    volga = vega_raw * d1 * d2 / sig

    # ---- bid / ask -------------------------------------------------------
    bid = theo - hs
    ask = theo + hs

    # ---- package into OptionQuote objects --------------------------------
    return [
        OptionQuote(
            strike=float(k[i]),
            expiry_days=int(days[i]),
            option_type=str(otypes[i]),
            theo=float(theo[i]),
            bid=float(bid[i]),
            ask=float(ask[i]),
            delta=float(delta[i]),
            gamma=float(gamma[i]),
            vega=float(vega_pct[i]),
            theta=float(theta[i]),
            vanna=float(vanna[i]),
            volga=float(volga[i]),
            implied_vol=float(sig[i]),
        )
        for i in range(len(k))
    ]
