"""AFL-inspired structured CLI dashboard using ``rich``.

Visual language
---------------
- Black terminal background, white text, monospaced.
- Colours: white / cyan / green / yellow / red only.
- Section headers: ``[[*]] SECTION NAME`` in bold cyan
  (``[[+]]`` good, ``[[-]]`` warn, ``[[!]]`` critical, ``[[*]]`` info).
- Numbers right-aligned in fixed-width columns.
- Tables: ``box=box.SIMPLE`` — no heavy borders.
- Sparklines: ▁▂▃▄▅▆▇█

Backward-compatible API (used by legacy CLI)
-----------------------------------------
``console``, ``banner``, ``feed_panel``, ``surface_panel``, ``book_panel``,
``quotes_panel``, ``risk_panel``, ``stress_panel``

New API
-------
``pnl_panel``, ``diagnostics_panel``, ``LiveDashboard``
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import pandas as pd
from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vol_surface_mm.core.hedging import BookState
from vol_surface_mm.core.quoting import Quote

# ── Console ───────────────────────────────────────────────────────────────────

console = Console(highlight=False)

# ── Constants ─────────────────────────────────────────────────────────────────

_SPARK_CHARS: str = "▁▂▃▄▅▆▇█"
_BORDER: str = "cyan"

# ── Low-level helpers ─────────────────────────────────────────────────────────


def _spark(values: Sequence[float], width: int = 20) -> str:
    """Block-character sparkline for the last ``width`` values."""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    chars = [_SPARK_CHARS[0 if rng < 1e-10 else min(int((v - mn) / rng * 7), 7)] for v in vals]
    return "".join(chars).ljust(width)


def _vol_style(iv: float) -> str:
    """green <20 %, yellow 20-30 %, red ≥30 %."""
    if iv < 0.20:
        return "green"
    if iv < 0.30:
        return "yellow"
    return "red"


def _util_style(ratio: float) -> str:
    """green <50 %, yellow <80 %, red ≥80 %."""
    if ratio < 0.50:
        return "green"
    if ratio < 0.80:
        return "yellow"
    return "red"


def _sign_style(value: float) -> str:
    return "green" if value >= 0.0 else "red"


def _status_tag(ratio: float) -> str:
    """Return an AFL-style ``[+]`` / ``[-]`` / ``[!]`` markup string."""
    if ratio < 0.50:
        return "[bold green][[+]][/bold green]"
    if ratio < 0.80:
        return "[bold yellow][[-]][/bold yellow]"
    return "[bold red][[!]][/bold red]"


def _hdr(tag: str, title: str) -> str:
    """``[[tag]] TITLE`` in bold cyan — tag ``[[`` escapes produce literal ``[``."""
    return f"[bold cyan][[{tag}]] {title}[/bold cyan]"


def _placeholder(title: str) -> Panel:
    return Panel("[dim]no data[/dim]", title=_hdr("*", title), border_style="dim")


# ── 1. MARKET FEED ────────────────────────────────────────────────────────────


def feed_panel(
    step: int,
    t: float,
    spot: float,
    inst_vol: float,
    *,
    spot_history: Sequence[float] | None = None,
    bid: float | None = None,
    ask: float | None = None,
) -> Panel:
    """``[[*]] MARKET FEED`` — spot, vol, optional sparkline."""
    iv_s = _vol_style(inst_vol)
    rows: list[tuple[str, str]] = [
        ("step", f"{step:>6d}"),
        ("t (yrs)", f"{t:>10.4f}"),
        ("spot", f"[white]{spot:>10.4f}[/white]"),
        ("inst vol", f"[{iv_s}]{inst_vol:>10.4f}[/{iv_s}]"),
        ("utc", datetime.now(tz=UTC).strftime("%H:%M:%S")),
    ]
    if bid is not None and ask is not None:
        rows.insert(3, ("bid / ask", f"[green]{bid:.4f}[/green] / [red]{ask:.4f}[/red]"))

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="left", style="bold white")
    tbl.add_column(justify="right")
    for k, v in rows:
        tbl.add_row(k, v)

    parts: list[RenderableType] = [tbl]
    if spot_history:
        spark_str = _spark(spot_history, width=20)
        lo, hi = min(spot_history), max(spot_history)
        parts.append(
            Text(
                f"  [{lo:.2f} {spark_str} {hi:.2f}]",
                style="cyan",
            )
        )

    return Panel(
        Group(*parts),
        title=_hdr("*", "MARKET FEED"),
        border_style=_BORDER,
        expand=False,
    )


# ── 2. VOLATILITY SURFACE ─────────────────────────────────────────────────────


def surface_panel(iv_grid: pd.DataFrame, arb: dict[str, int]) -> Panel:
    """``[[*]] VOLATILITY SURFACE`` — IV grid coloured by level."""
    tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        expand=False,
        padding=(0, 1),
    )
    tbl.add_column("T \\ K", justify="right", style="dim white")
    for col in iv_grid.columns:
        tbl.add_column(f"{col:.2f}", justify="right")

    for idx, row in iv_grid.iterrows():
        cells: list[str] = [f"{idx:.3f}"]
        for v in row.values:
            iv = float(v)
            s = _vol_style(iv)
            cells.append(f"[{s}]{iv:.4f}[/{s}]")
        tbl.add_row(*cells)

    cal_v = int(arb["calendar_violations"])
    fly_v = int(arb["butterfly_violations"])
    cal_s = "bold red" if cal_v > 0 else "green"
    fly_s = "bold red" if fly_v > 0 else "green"
    footer = Text.assemble(
        "  cal  ",
        (f"{'FAIL' if cal_v else 'PASS'}", cal_s),
        f" ({cal_v})   fly  ",
        (f"{'FAIL' if fly_v else 'PASS'}", fly_s),
        f" ({fly_v})",
    )
    return Panel(
        Group(tbl, footer),
        title=_hdr("*", "VOLATILITY SURFACE"),
        border_style=_BORDER,
        expand=False,
    )


# ── 3. PORTFOLIO GREEKS ───────────────────────────────────────────────────────


def book_panel(
    book: BookState,
    greeks: dict[str, float],
    mtm: float,
    *,
    vanna: float | None = None,
    volga: float | None = None,
    break_even_move: float | None = None,
    strategy: dict[str, float] | None = None,
    delta_limit: float = 300.0,
    gamma_limit: float = 8.0,
    vega_limit: float = 4000.0,
) -> Panel:
    """``[[+/-/!]] PORTFOLIO GREEKS`` — utilisation-coloured Greeks + positions."""
    delta = greeks.get("delta", 0.0)
    gamma = greeks.get("gamma", 0.0)
    vega_ = greeks.get("vega", 0.0)
    theta = greeks.get("theta", 0.0)

    d_util = abs(delta) / max(delta_limit, 1e-6)
    g_util = abs(gamma) / max(gamma_limit, 1e-6)
    v_util = abs(vega_) / max(vega_limit, 1e-6)
    worst = max(d_util, g_util, v_util)

    tag_char = "!" if worst >= 0.80 else ("+" if worst < 0.50 else "-")
    hdr_color = "red" if worst >= 0.80 else ("green" if worst < 0.50 else "yellow")

    pos_tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        expand=False,
        padding=(0, 1),
    )
    pos_tbl.add_column("type")
    pos_tbl.add_column("K", justify="right")
    pos_tbl.add_column("T", justify="right")
    pos_tbl.add_column("qty", justify="right")
    for p in book.positions[-6:]:
        qs = "green" if p.quantity > 0 else "red"
        pos_tbl.add_row(
            p.option_type,
            f"[white]{p.strike:.2f}[/white]",
            f"[white]{p.expiry:.3f}[/white]",
            f"[{qs}]{p.quantity:+.2f}[/{qs}]",
        )

    g_tbl = Table.grid(padding=(0, 1))
    g_tbl.add_column(justify="left", min_width=6)
    g_tbl.add_column(justify="left", style="bold white", min_width=9)
    g_tbl.add_column(justify="right", min_width=12)

    def _grow(label: str, val: float, util: float | None = None) -> None:
        tag = _status_tag(util) if util is not None else "      "
        vs = _sign_style(val)
        if util is not None:
            us = _util_style(util)
            g_tbl.add_row(tag, label, f"[{us}]{val:+10.4f}[/{us}]")
        else:
            g_tbl.add_row(tag, label, f"[{vs}]{val:+10.4f}[/{vs}]")

    _grow("delta", delta, d_util)
    _grow("gamma", gamma, g_util)
    _grow("vega", vega_, v_util)
    _grow("theta", theta)
    if vanna is not None:
        _grow("vanna", vanna)
    if volga is not None:
        _grow("volga", volga)
    g_tbl.add_row("      ", "spot pos", f"[white]{book.spot_position:+10.4f}[/white]")
    g_tbl.add_row("      ", "cash", f"[white]{book.cash:+12.4f}[/white]")
    mtm_s = _sign_style(mtm)
    g_tbl.add_row("      ", "mtm", f"[{mtm_s}]{mtm:+12.4f}[/{mtm_s}]")
    g_tbl.add_row("      ", "tcost", f"[white]{book.transaction_costs:12.4f}[/white]")

    parts: list[RenderableType] = [pos_tbl, g_tbl]
    if break_even_move is not None:
        be_pct = break_even_move * 100.0
        parts.append(
            Text(
                f"  break-even move  {be_pct:.3f} %  ({break_even_move:.5f})",
                style="yellow",
            )
        )

    if strategy is not None:
        strat_tbl = Table.grid(padding=(0, 1))
        strat_tbl.add_column(justify="left", style="bold white", min_width=12)
        strat_tbl.add_column(justify="right", min_width=12)

        worst_stress = float(strategy.get("worst_stress", 0.0))
        stress_guard = float(strategy.get("stress_guard", 0.0))
        worst_style = "green" if worst_stress >= stress_guard else "red"
        guard_style = "yellow"

        strat_tbl.add_row("gamma tgt", f"[white]{float(strategy.get('target_gamma', 0.0)):+10.4f}[/white]")
        strat_tbl.add_row("kelly", f"[white]{float(strategy.get('kelly_fraction', 0.0)):10.3f}[/white]")
        strat_tbl.add_row("tail trig", f"[white]{float(strategy.get('tail_trigger', 0.0)):10.3f}[/white]")
        strat_tbl.add_row("tail qty", f"[white]{float(strategy.get('tail_hedge_qty', 0.0)):10.2f}[/white]")
        strat_tbl.add_row("worst stress", f"[{worst_style}]{worst_stress:+10.4f}[/{worst_style}]")
        strat_tbl.add_row("guard bound", f"[{guard_style}]{stress_guard:+10.4f}[/{guard_style}]")
        parts.append(strat_tbl)

    return Panel(
        Group(*parts),
        title=f"[bold {hdr_color}][[{tag_char}]] PORTFOLIO GREEKS[/bold {hdr_color}]",
        border_style=hdr_color,
        expand=False,
    )


# ── 4. QUOTES ─────────────────────────────────────────────────────────────────


def quotes_panel(quotes: Iterable[Quote]) -> Panel:
    """``[[+]] QUOTES`` — two-sided markets with fill probability."""
    tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        expand=False,
        padding=(0, 1),
    )
    for col, just in [
        ("type", "left"),
        ("K", "right"),
        ("T", "right"),
        ("IV", "right"),
        ("bid", "right"),
        ("ask", "right"),
        ("hs(vol)", "right"),
        ("size", "right"),
        ("fp", "right"),
    ]:
        tbl.add_column(col, justify=just)

    for q in quotes:
        iv_s = _vol_style(q.model_iv)
        tbl.add_row(
            f"[white]{q.option_type}[/white]",
            f"[white]{q.strike:.2f}[/white]",
            f"[white]{q.expiry:.3f}[/white]",
            f"[{iv_s}]{q.model_iv:.4f}[/{iv_s}]",
            f"[green]{q.bid:.4f}[/green]",
            f"[red]{q.ask:.4f}[/red]",
            f"[yellow]{q.half_spread_vol:.4f}[/yellow]",
            f"[white]{q.size:.2f}[/white]",
            f"[dim]{q.fill_probability_bid:.2f}[/dim]",
        )
    return Panel(
        tbl,
        title=_hdr("+", "QUOTES"),
        border_style="green",
        expand=False,
    )


# ── 5. RISK ───────────────────────────────────────────────────────────────────


def risk_panel(rv: float, iv: float, residual_attr: float) -> Panel:
    """``[[*]] RISK`` — RV vs IV, cumulative residual attribution."""
    disloc = rv * rv - iv * iv
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="left", style="bold white")
    tbl.add_column(justify="right")

    rv_s = _vol_style(rv)
    iv_s = _vol_style(iv)
    dl_s = _sign_style(disloc)
    ra_s = _sign_style(residual_attr)

    tbl.add_row("RV (ann)", f"[{rv_s}]{rv:>10.4f}[/{rv_s}]")
    tbl.add_row("IV (ann, atm avg)", f"[{iv_s}]{iv:>10.4f}[/{iv_s}]")
    tbl.add_row("RV^2 - IV^2", f"[{dl_s}]{disloc:>10.5f}[/{dl_s}]")
    tbl.add_row("residual attr", f"[{ra_s}]{residual_attr:>+12.4f}[/{ra_s}]")

    return Panel(tbl, title=_hdr("*", "RISK"), border_style=_BORDER, expand=False)


# ── 6. STRESS RESULTS ─────────────────────────────────────────────────────────


def stress_panel(df: pd.DataFrame) -> Panel:
    """``[[!]] STRESS RESULTS`` — worst 5 highlighted, absolute worst in bright red."""
    tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        expand=False,
        padding=(0, 1),
    )
    for col, just in [
        ("scenario", "left"),
        ("spot_mult", "right"),
        ("vol_shift", "right"),
        ("skew_twist", "right"),
        ("term_slope", "right"),
        ("pnl", "right"),
    ]:
        tbl.add_column(col, justify=just)

    if df.empty:
        return Panel(tbl, title=_hdr("!", "STRESS RESULTS"), border_style="red", expand=False)

    sorted_idx = df["pnl"].sort_values().index
    worst_idx = sorted_idx[0]
    worst5_idx = set(sorted_idx[:5])

    for row_idx, row in df.iterrows():
        pnl = float(row["pnl"])
        if row_idx == worst_idx:
            ps, ns = "bold red", "bold red"
        elif row_idx in worst5_idx:
            ps, ns = "red", "red"
        else:
            ps = "green" if pnl >= 0 else "yellow"
            ns = "white"
        tbl.add_row(
            f"[{ns}]{row['scenario']}[/{ns}]",
            f"[white]{row['spot_mult']:.3f}[/white]",
            f"[white]{row['vol_shift']:+.3f}[/white]",
            f"[white]{row['skew_twist']:+.3f}[/white]",
            f"[white]{row['term_slope']:+.3f}[/white]",
            f"[{ps}]{pnl:+.4f}[/{ps}]",
        )
    return Panel(tbl, title=_hdr("!", "STRESS RESULTS"), border_style="red", expand=False)


# ── 7. P&L DECOMPOSITION ─────────────────────────────────────────────────────


def pnl_panel(pnl_report: object | None) -> Panel:
    """``[[*]] P&L DECOMPOSITION`` — step + cumulative attribution from ``PnLReport``."""
    if pnl_report is None:
        return _placeholder("P&L DECOMPOSITION")

    tbl = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        expand=False,
        padding=(0, 1),
    )
    tbl.add_column("component", justify="left", style="bold white")
    tbl.add_column("today", justify="right", min_width=11)
    tbl.add_column("cumulative", justify="right", min_width=11)

    def _row(label: str, today: float, cum: float) -> None:
        ts, cs = _sign_style(today), _sign_style(cum)
        tbl.add_row(
            label,
            f"[{ts}]{today:+10.4f}[/{ts}]",
            f"[{cs}]{cum:+10.4f}[/{cs}]",
        )

    try:
        _row("delta", pnl_report.delta_pnl, pnl_report.cum_delta_pnl)
        _row("gamma", pnl_report.gamma_pnl, pnl_report.cum_gamma_pnl)
        _row("theta", pnl_report.theta_pnl, pnl_report.cum_theta_pnl)
        _row("vega", pnl_report.vega_pnl, pnl_report.cum_vega_pnl)
        resid = pnl_report.residual_pnl
        cum_resid = (
            pnl_report.cum_total_pnl
            - pnl_report.cum_delta_pnl
            - pnl_report.cum_gamma_pnl
            - pnl_report.cum_theta_pnl
            - pnl_report.cum_vega_pnl
        )
        _row("residual", resid, cum_resid)
        _row("total", pnl_report.total_pnl, pnl_report.cum_total_pnl)
    except AttributeError:
        tbl.add_row("—", "—", "—")

    footer = Text()
    try:
        rv_v = pnl_report.realized_variance
        iv_v = pnl_report.implied_variance
        vc = pnl_report.variance_capture
        vc_s = _sign_style(vc)
        footer = Text.assemble(
            "  RV: ",
            (f"{rv_v:.5f}", "white"),
            "  IV²: ",
            (f"{iv_v:.5f}", "white"),
            "  var_capture: ",
            (f"{vc:+.5f}", vc_s),
        )
    except AttributeError:
        pass

    return Panel(
        Group(tbl, footer),
        title=_hdr("*", "P&L DECOMPOSITION"),
        border_style=_BORDER,
        expand=False,
    )


# ── 8. SURFACE DIAGNOSTICS ───────────────────────────────────────────────────


def diagnostics_panel(diag: object | None) -> Panel:
    """``[[*]] SURFACE DIAGNOSTICS`` — RMSE, stability, VRP, hedge-error quality."""
    if diag is None:
        return _placeholder("SURFACE DIAGNOSTICS")

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="left", style="bold white")
    tbl.add_column(justify="right", min_width=12)

    def _metric(
        label: str,
        val: float,
        fmt: str = ".6f",
        lo: float = 0.005,
        hi: float = 0.015,
        good_low: bool = True,
    ) -> None:
        if good_low:
            s = "green" if val < lo else ("yellow" if val < hi else "red")
        else:
            s = _sign_style(val)
        tbl.add_row(label, f"[{s}]{val:{fmt}}[/{s}]")

    try:
        _metric("surface RMSE", diag.surface_rmse, lo=0.001, hi=0.005)
        _metric("ATM vol stab", diag.atm_vol_stability, lo=0.001, hi=0.005)
        _metric("skew stab", diag.skew_stability, lo=0.001, hi=0.008)

        vrp = diag.variance_risk_premium
        vrp30 = diag.vrp_30d_mean
        tbl.add_row("VRP now", f"[{_sign_style(vrp)}]{vrp:+.6f}[/]")
        tbl.add_row("VRP 30d mean", f"[{_sign_style(vrp30)}]{vrp30:+.6f}[/]")

        _metric("resid attr mean", diag.residual_attribution_mean, fmt=".5f", lo=0.0, hi=0.0, good_low=False)
        _metric("resid attr std", diag.residual_attribution_std, lo=0.01, hi=0.05)

        krt = diag.residual_attribution_kurtosis
        krt_s = "green" if abs(krt) < 1.0 else ("yellow" if abs(krt) < 3.0 else "red")
        tbl.add_row("resid attr kurt", f"[{krt_s}]{krt:+.4f}[/{krt_s}]")
    except AttributeError:
        tbl.add_row("—", "no data")

    return Panel(
        tbl,
        title=_hdr("*", "SURFACE DIAGNOSTICS"),
        border_style=_BORDER,
        expand=False,
    )


# ── Live Dashboard ────────────────────────────────────────────────────────────


class LiveDashboard:
    """Full-terminal AFL dashboard — 3-row × 2-col layout refreshed at 1 Hz.

    Slots (keyword names for :meth:`update`)::

        feed_panel        surface_panel
        book_panel        stress_panel
        pnl_panel         diagnostics_panel

    Usage::

        with LiveDashboard() as dash:
            for tick in feed:
                ...
                dash.update(
                    feed_panel=feed_panel(...),
                    surface_panel=surface_panel(...),
                    book_panel=book_panel(...),
                    stress_panel=stress_panel(...),
                    pnl_panel=pnl_panel(...),
                    diagnostics_panel=diagnostics_panel(...),
                )
    """

    _SLOTS: tuple[tuple[str, str], ...] = (
        ("feed_panel", "MARKET FEED"),
        ("surface_panel", "VOLATILITY SURFACE"),
        ("book_panel", "PORTFOLIO GREEKS"),
        ("stress_panel", "STRESS RESULTS"),
        ("pnl_panel", "P&L DECOMPOSITION"),
        ("diagnostics_panel", "SURFACE DIAGNOSTICS"),
    )

    def __init__(self, refresh_rate: float = 1.0, screen: bool = True) -> None:
        self._refresh_rate = refresh_rate
        self._screen = screen
        self._panels: dict[str, RenderableType] = {key: _placeholder(title) for key, title in self._SLOTS}
        self._live: Live | None = None

    def update(self, **panels: RenderableType) -> None:
        """Replace named panels and redraw."""
        self._panels.update(panels)
        if self._live is not None:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="row0", ratio=1),
            Layout(name="row1", ratio=1),
            Layout(name="row2", ratio=1),
        )
        for row_name, (left_key, right_key) in zip(
            ("row0", "row1", "row2"),
            (
                ("feed_panel", "surface_panel"),
                ("book_panel", "stress_panel"),
                ("pnl_panel", "diagnostics_panel"),
            ),
            strict=True,
        ):
            root[row_name].split_row(
                Layout(name=left_key),
                Layout(name=right_key),
            )
            root[row_name][left_key].update(self._panels[left_key])
            root[row_name][right_key].update(self._panels[right_key])

        return root

    def __enter__(self) -> LiveDashboard:
        self._live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=self._refresh_rate,
            screen=self._screen,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None


# ── Utilities ─────────────────────────────────────────────────────────────────


def banner(title: str) -> None:
    """Print a full-width rule."""
    console.rule(f"[bold white]{title}[/bold white]")
