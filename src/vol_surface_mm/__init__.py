"""vol_surface_mm — options market-making simulator.

A self-contained simulator that drives a continuously refitted implied
volatility surface from a synthetic (or research) underlying feed, posts
inventory-aware two-sided quotes, books fills, hedges delta, and reports
P&L attribution and stress scenarios.

Sub-packages:

- :mod:`vol_surface_mm.core` — pricing, surface fitting, quoting engine,
  hedging book, and stress scenarios.
- :mod:`vol_surface_mm.data` — feeds and option chain construction.
- :mod:`vol_surface_mm.diagnostics` — metrics and the live Rich dashboard.
- :mod:`vol_surface_mm.scripts` — backtest artifact and parameter sweep
  drivers.

The :mod:`vol_surface_mm.cli` module wires these together behind the
``vol-surface-mm`` console script.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PACKAGE_ROOT", "REPO_ROOT", "__version__"]

__version__ = "0.1.0"

# Package directory (src/vol_surface_mm).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# Repository root (two levels up from src/vol_surface_mm/__init__.py).
REPO_ROOT: Path = PACKAGE_ROOT.parents[1]
