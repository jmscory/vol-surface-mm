"""Render a per-feed P&L heatmap from the grouped sweep CSV.

Reads ``results/param_sweep_grouped.csv`` and writes a 2x2 grid of heatmaps
showing ``total_pnl`` for each feed (gbm, bates, rough, real_aapl) across the
``(kelly_fraction, gamma_target)`` grid, aggregated by median over the other
two sweep dimensions (``tail_hedge_trigger``, ``stress_guard_multiple``).

Usage::

    python -m vol_surface_mm.scripts.plot_sweep
    python -m vol_surface_mm.scripts.plot_sweep --output docs/assets/foo.png

Requires ``matplotlib`` (install via ``pip install matplotlib`` or
``pip install -e ".[plot]"``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface_mm import REPO_ROOT

FEEDS: tuple[str, ...] = ("gbm", "bates", "rough", "real_aapl")
DEFAULT_INPUT = REPO_ROOT / "results" / "param_sweep_grouped.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "sweep_summary.png"


def _build_grid(df: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Pivot the median per-feed total PnL on (gamma_target, kelly_fraction)."""
    column = f"{feed}_total_pnl"
    return (
        df.groupby(["gamma_target", "kelly_fraction"])[column]
        .median()
        .unstack("kelly_fraction")
        .sort_index(ascending=False)
    )


def render(input_path: Path, output_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    df = pd.read_csv(input_path)

    grids = {feed: _build_grid(df, feed) for feed in FEEDS}
    all_values = np.concatenate([g.to_numpy().ravel() for g in grids.values()])
    finite = all_values[np.isfinite(all_values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(
        "Median total P&L per feed across (kelly_fraction, gamma_target)",
        fontsize=12,
        fontweight="bold",
    )

    last_im = None
    for ax, feed in zip(axes.ravel(), FEEDS, strict=True):
        grid = grids[feed]
        last_im = ax.imshow(grid.to_numpy(), cmap="RdBu", norm=norm, aspect="auto")
        ax.set_title(feed, fontsize=10)
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{k:g}" for k in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{g:g}" for g in grid.index])
        ax.set_xlabel("kelly_fraction")
        ax.set_ylabel("gamma_target")
        for i, gamma in enumerate(grid.index):
            for j, kelly in enumerate(grid.columns):
                value = grid.loc[gamma, kelly]
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="black" if abs(value) < vmax * 0.55 else "white",
                    fontsize=9,
                )

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, label="total P&L")
    cbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Grouped sweep CSV (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args(argv)
    out = render(args.input, args.output)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
