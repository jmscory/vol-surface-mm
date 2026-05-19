from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vol_surface_mm import REPO_ROOT
from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.scripts.artifact_common import RunSummary, run_feed_backtest


def _dominant_risk_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_") if raw else "residual"


def _scenario_name(raw: str) -> str:
    name = raw.strip().lower().replace("worst:", "").replace("%", "").replace("pt", "")
    name = name.replace("/", " ")
    name = name.replace("spot-", "spot_-").replace("spot+", "spot_+")
    name = name.replace("vol-", "vol_-").replace("vol+", "vol_+")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.replace("+", "")
    return name.strip("_")


def _summary_entry(summary: RunSummary) -> dict[str, Any]:
    return {
        "total_pnl": summary.total_pnl,
        "var_capture": summary.var_capture,
        "max_drawdown": summary.max_drawdown,
        "gamma_pnl": summary.gamma_pnl,
        "theta_pnl": summary.theta_pnl,
        "hedge_error_kurtosis": summary.hedge_error_kurtosis,
        "surface_rmse": summary.surface_rmse,
        "survived": summary.survived,
    }


def _write_surface_snapshot(path: Path, summary: RunSummary) -> None:
    surface = summary.final_surface
    spot = summary.final_spot

    strikes = np.linspace(0.80 * spot, 1.20 * spot, 20, dtype=np.float64)
    expiries_days = [int(x) for x in surface.expiries_days[:6]]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strike"] + [f"T_{d}d" for d in expiries_days])
        for strike in strikes:
            row = [float(strike)]
            for expiry_days in expiries_days:
                row.append(float(surface.vol(float(strike), int(expiry_days))))
            writer.writerow(row)


def _print_afl_report(report: dict[str, Any]) -> None:
    print("================ FINAL REVIEW REPORT ================")
    md = report["run_metadata"]
    print(
        f"seed={md['seed']} steps={md['steps']} backend={md['backend']} timestamp={md['timestamp']}"
    )
    print()

    print("PER FEED")
    print("feed      total_pnl   var_capture   max_dd   gamma_pnl   theta_pnl   kurtosis   rmse   survived")
    for feed in ("gbm", "bates", "rough"):
        row = report["per_feed"][feed]
        print(
            f"{feed:<8} {row['total_pnl']:>10.2f} {row['var_capture']:>12.4f} {row['max_drawdown']:>8.2f} "
            f"{row['gamma_pnl']:>10.2f} {row['theta_pnl']:>10.4f} {row['hedge_error_kurtosis']:>10.4f} "
            f"{row['surface_rmse']:>7.5f} {row['survived']!s:>9}"
        )
    print()

    print("WORST STRESS SCENARIOS (TOP 5)")
    for sc in report["stress_scenarios"]:
        print(f"- {sc['name']}: pnl={sc['pnl']:.2f}, dominant_risk={sc['dominant_risk']}")
    print()

    sd = report["surface_diagnostics"]
    print("SURFACE DIAGNOSTICS")
    print(
        f"atm_vol_stability={sd['atm_vol_stability']:.6f} "
        f"skew_stability={sd['skew_stability']:.6f} "
        f"calendar_violations={sd['calendar_violations']} "
        f"butterfly_violations={sd['butterfly_violations']}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic final report artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic feeds.")
    parser.add_argument("--steps", type=int, default=252, help="Number of simulation steps.")
    parser.add_argument(
        "--backend",
        choices=["sabr", "spline"],
        default="sabr",
        help="Surface backend passed to run_feed_backtest.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for JSON/TXT/CSV artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.ERROR)
    args = _parse_args(argv)

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    seed = int(args.seed)
    steps = int(args.steps)
    backend = str(args.backend)

    summaries: dict[str, RunSummary] = {}
    for feed in ("gbm", "bates", "rough"):
        summaries[feed] = run_feed_backtest(
            feed=feed,
            cfg=cfg,
            steps=steps,
            seed=seed,
            backend=backend,
            survival_multiple=3.0,
            frozen_surface=False,
            stress_refresh_interval=20,
            fill_interval=5,
            max_inventory_positions=300,
            simulate_fills=True,
        )

    all_stress: list[tuple[str, Any]] = []
    for feed, summary in summaries.items():
        for report in summary.stress_reports:
            all_stress.append((feed, report))
    all_stress.sort(key=lambda x: float(x[1].pnl_impact))

    top5 = []
    for feed, report in all_stress[:5]:
        top5.append(
            {
                "name": f"{feed}_{_scenario_name(report.scenario_name)}",
                "pnl": float(report.pnl_impact),
                "dominant_risk": _dominant_risk_name(report.dominant_risk),
            }
        )

    # Use the final (rough) surface as the snapshot anchor.
    rough_surface = summaries["rough"].final_surface
    rough_spot = summaries["rough"].final_spot
    arb = rough_surface.arbitrage_report(
        strikes=tuple(np.linspace(0.80 * rough_spot, 1.20 * rough_spot, 20)),
        expiries=tuple(rough_surface.expiries_days[:6]),
    )

    report_obj: dict[str, Any] = {
        "run_metadata": {
            "seed": seed,
            "steps": steps,
            "backend": backend,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        "per_feed": {
            "gbm": _summary_entry(summaries["gbm"]),
            "bates": _summary_entry(summaries["bates"]),
            "rough": _summary_entry(summaries["rough"]),
        },
        "stress_scenarios": top5,
        "surface_diagnostics": {
            "atm_vol_stability": float(summaries["rough"].atm_vol_stability),
            "skew_stability": float(summaries["rough"].skew_stability),
            "calendar_violations": int(arb["calendar_violations"]),
            "butterfly_violations": int(arb["butterfly_violations"]),
        },
    }

    with (results_dir / "final_report.json").open("w") as f:
        json.dump(report_obj, f, indent=2)

    _write_surface_snapshot(results_dir / "surface_snapshot.csv", summaries["rough"])

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_afl_report(report_obj)
    text = buf.getvalue()
    (results_dir / "final_report.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
