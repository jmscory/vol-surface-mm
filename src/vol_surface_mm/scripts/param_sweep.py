from __future__ import annotations

import argparse
import csv
import logging
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from vol_surface_mm import REPO_ROOT
from vol_surface_mm.config import DEFAULT_CONFIG
from vol_surface_mm.scripts.artifact_common import run_feed_backtest

DEFAULT_KELLY_VALUES = [0.05, 0.10, 0.15]
DEFAULT_GAMMA_TARGETS = [0.005, 0.010]
DEFAULT_TAIL_TRIGGER_VALUES = [0.20, 0.30]
DEFAULT_STRESS_GUARD_VALUES = [2.0, 2.5]
DEFAULT_FEEDS = ["gbm", "bates", "rough", "real_aapl"]
SURVIVAL_MULTIPLE = 2.0
FIELDNAMES = [
    "feed",
    "data_source",
    "kelly_fraction",
    "gamma_target",
    "tail_hedge_trigger",
    "stress_guard_multiple",
    "total_pnl",
    "var_capture",
    "max_drawdown",
    "avg_daily_theta",
    "dd_limit",
    "survived",
    "tuning_survived",
    "profitable",
    "final_bankroll",
]
GROUP_FIELDNAMES = [
    "kelly_fraction",
    "gamma_target",
    "tail_hedge_trigger",
    "stress_guard_multiple",
    "feed_count",
    "feeds_covered",
    "all_feeds_present",
    "all_tuning_survived",
    "strict_survived_count",
    "profitable_count",
    "all_profitable",
    "worst_total_pnl",
    "median_total_pnl",
    "worst_max_drawdown",
    "median_var_capture",
    "gbm_total_pnl",
    "bates_total_pnl",
    "rough_total_pnl",
    "real_aapl_total_pnl",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the focused live-path grid around the current low-gamma/low-kelly "
            "winner and optionally merge shard outputs."
        )
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge param_sweep_part_*.csv into the canonical outputs and exit.",
    )
    parser.add_argument(
        "--kelly-shard",
        type=float,
        help="Run only one Kelly fraction shard and write results/param_sweep_part_kXXX.csv.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Number of backtest steps per run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic feeds.")
    parser.add_argument("--backend", default="sabr", help="Surface backend passed to run_feed_backtest.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for CSV and stdout artifacts.",
    )
    return parser.parse_args(argv)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_float(value: Any) -> float:
    return float(value)


def _data_source(feed: str) -> str:
    if feed == "real_aapl":
        return "plotly_finance_charts_apple"
    return "synthetic"


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["feed"]),
        float(row["kelly_fraction"]),
        float(row["gamma_target"]),
        float(row["tail_hedge_trigger"]),
        float(row["stress_guard_multiple"]),
    )


def _bundle_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _as_float(row["kelly_fraction"]),
        _as_float(row["gamma_target"]),
        _as_float(row["tail_hedge_trigger"]),
        _as_float(row["stress_guard_multiple"]),
    )


def _bundle_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(_as_bool(row["all_feeds_present"])),
        int(_as_bool(row["all_tuning_survived"])),
        _as_float(row["worst_total_pnl"]),
        int(row["profitable_count"]),
        _as_float(row["median_total_pnl"]),
        -_as_float(row["worst_max_drawdown"]),
        _as_float(row["median_var_capture"]),
    )


def _is_feasible(row: dict[str, Any]) -> bool:
    tuning_survived = _as_bool(row["tuning_survived"])
    return tuning_survived and _as_float(row["max_drawdown"]) < _as_float(row["dd_limit"])


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Maximize var_capture subject to feasibility, using max_drawdown as the
    # secondary minimization axis for Pareto dominance.
    feasible = [r for r in rows if _is_feasible(r)]
    front: list[dict[str, Any]] = []

    for i, a in enumerate(feasible):
        dominated = False
        for j, b in enumerate(feasible):
            if i == j:
                continue
            no_worse = _as_float(b["var_capture"]) >= _as_float(a["var_capture"]) and _as_float(
                b["max_drawdown"]
            ) <= _as_float(a["max_drawdown"])
            strictly_better = _as_float(b["var_capture"]) > _as_float(a["var_capture"]) or _as_float(
                b["max_drawdown"]
            ) < _as_float(a["max_drawdown"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)

    front.sort(key=lambda x: (_as_float(x["var_capture"]), -_as_float(x["max_drawdown"])), reverse=True)
    return front


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"feed={row['feed']} kelly={_as_float(row['kelly_fraction']):.2f} "
        f"gamma_target={_as_float(row['gamma_target']):.3f} "
        f"tail_trigger={_as_float(row['tail_hedge_trigger']):.2f} "
        f"stress_guard={_as_float(row['stress_guard_multiple']):.2f} "
        f"var_capture={_as_float(row['var_capture']):.4f} total_pnl={_as_float(row['total_pnl']):.2f} "
        f"max_dd={_as_float(row['max_drawdown']):.2f} limit={_as_float(row['dd_limit']):.2f} "
        f"tuning_survived={_as_bool(row['tuning_survived'])} profitable={_as_bool(row['profitable'])}"
    )


def _format_bundle_row(row: dict[str, Any]) -> str:
    return (
        f"kelly={_as_float(row['kelly_fraction']):.2f} gamma_target={_as_float(row['gamma_target']):.3f} "
        f"tail_trigger={_as_float(row['tail_hedge_trigger']):.2f} stress_guard={_as_float(row['stress_guard_multiple']):.2f} "
        f"worst_pnl={_as_float(row['worst_total_pnl']):.2f} profitable_feeds={int(row['profitable_count'])}/{len(DEFAULT_FEEDS)} "
        f"median_pnl={_as_float(row['median_total_pnl']):.2f} worst_dd={_as_float(row['worst_max_drawdown']):.2f} "
        f"median_var_capture={_as_float(row['median_var_capture']):.4f} real_aapl_pnl={_as_float(row['real_aapl_total_pnl']):.2f} "
        f"all_tuning_survived={_as_bool(row['all_tuning_survived'])}"
    )


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (_as_float(row["var_capture"]), -_as_float(row["max_drawdown"])))


def _write_rows(csv_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_group_rows(csv_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_bundle_key(row), []).append(row)

    expected_feeds = list(DEFAULT_FEEDS)
    grouped_rows: list[dict[str, Any]] = []
    for bundle_key, bundle_rows in grouped.items():
        by_feed = {str(row["feed"]): row for row in bundle_rows}
        pnl_values = [_as_float(row["total_pnl"]) for row in bundle_rows]
        max_drawdown_values = [_as_float(row["max_drawdown"]) for row in bundle_rows]
        var_capture_values = [_as_float(row["var_capture"]) for row in bundle_rows]
        feed_count = len(by_feed)
        all_feeds_present = all(feed in by_feed for feed in expected_feeds)
        all_tuning_survived = all_feeds_present and all(
            _as_bool(by_feed[feed]["tuning_survived"]) for feed in expected_feeds
        )
        strict_survived_count = sum(
            1 for feed in expected_feeds if feed in by_feed and _as_bool(by_feed[feed]["survived"])
        )
        profitable_count = sum(
            1 for feed in expected_feeds if feed in by_feed and _as_bool(by_feed[feed]["profitable"])
        )

        grouped_rows.append(
            {
                "kelly_fraction": bundle_key[0],
                "gamma_target": bundle_key[1],
                "tail_hedge_trigger": bundle_key[2],
                "stress_guard_multiple": bundle_key[3],
                "feed_count": feed_count,
                "feeds_covered": ",".join(feed for feed in expected_feeds if feed in by_feed),
                "all_feeds_present": all_feeds_present,
                "all_tuning_survived": all_tuning_survived,
                "strict_survived_count": strict_survived_count,
                "profitable_count": profitable_count,
                "all_profitable": profitable_count == len(expected_feeds),
                "worst_total_pnl": min(pnl_values) if pnl_values else float("nan"),
                "median_total_pnl": median(pnl_values) if pnl_values else float("nan"),
                "worst_max_drawdown": max(max_drawdown_values) if max_drawdown_values else float("nan"),
                "median_var_capture": median(var_capture_values) if var_capture_values else float("nan"),
                "gbm_total_pnl": _as_float(by_feed["gbm"]["total_pnl"]) if "gbm" in by_feed else float("nan"),
                "bates_total_pnl": _as_float(by_feed["bates"]["total_pnl"])
                if "bates" in by_feed
                else float("nan"),
                "rough_total_pnl": _as_float(by_feed["rough"]["total_pnl"])
                if "rough" in by_feed
                else float("nan"),
                "real_aapl_total_pnl": _as_float(by_feed["real_aapl"]["total_pnl"])
                if "real_aapl" in by_feed
                else float("nan"),
            }
        )

    grouped_rows.sort(key=_bundle_sort_key, reverse=True)
    return grouped_rows


def _summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    strict_count = sum(1 for row in rows if _as_bool(row["survived"]))
    tuning_count = sum(1 for row in rows if _as_bool(row["tuning_survived"]))
    profitable_count = sum(1 for row in rows if _as_bool(row["profitable"]))
    grouped_rows = _group_rows(rows)
    grouped_candidates = [row for row in grouped_rows if _as_bool(row["all_tuning_survived"])]

    lines = [
        (
            "Cross-feed tuple ranking (screen: tuning_survived=True on every feed; "
            "rank by worst_total_pnl, profitable_count, median_total_pnl, worst_max_drawdown, median_var_capture)"
        ),
        (
            f"row_count={len(rows)} strict_survived_rows={strict_count} tuning_survived_rows={tuning_count} "
            f"profitable_rows={profitable_count} tuple_count={len(grouped_rows)} candidate_tuples={len(grouped_candidates)}"
        ),
    ]

    if grouped_candidates:
        lines.append("Top cross-feed tuples")
        lines.extend(_format_bundle_row(row) for row in grouped_candidates)
    else:
        lines.append("No parameter tuples satisfied tuning_survived=True on every feed.")
        if grouped_rows:
            lines.append("Top tuples before the cross-feed screen")
            lines.extend(_format_bundle_row(row) for row in grouped_rows[:5])

    best_tuple = grouped_candidates[0] if grouped_candidates else grouped_rows[0]
    lines.append("")
    lines.append("Best cross-feed tuple")
    lines.append(_format_bundle_row(best_tuple))
    if not _as_bool(best_tuple["all_profitable"]):
        lines.append("No tuple achieved profitability on every feed.")

    legacy_best = _best_row(rows)
    best_rows = [
        row
        for row in rows
        if math.isclose(
            _as_float(row["var_capture"]), _as_float(legacy_best["var_capture"]), rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            _as_float(row["max_drawdown"]), _as_float(legacy_best["max_drawdown"]), rel_tol=0.0, abs_tol=1e-12
        )
    ]
    lines.append("")
    lines.append("Legacy best single row by var_capture with drawdown tiebreaker")
    lines.append(_format_row(legacy_best))
    lines.append(f"legacy_tied_best_rows={len(best_rows)}")
    return lines


def _write_summary(stdout_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = _summary_lines(rows)
    stdout_path.write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def _shard_path(results_dir: Path, kelly_fraction: float) -> Path:
    shard_id = int(round(kelly_fraction * 100.0))
    return results_dir / f"param_sweep_part_k{shard_id:03d}.csv"


def _merge_shards(results_dir: Path) -> list[dict[str, Any]]:
    shard_paths = sorted(results_dir.glob("param_sweep_part_k*.csv"))
    if not shard_paths:
        raise FileNotFoundError("no param_sweep_part_k*.csv files found to merge")

    merged_rows: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        with shard_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            merged_rows.extend(reader)

    merged_rows.sort(key=_row_sort_key)
    _write_rows(results_dir / "param_sweep.csv", merged_rows)
    _write_group_rows(results_dir / "param_sweep_grouped.csv", _group_rows(merged_rows))
    _write_summary(results_dir / "param_sweep_stdout.txt", merged_rows)
    return merged_rows


def _run_rows(
    *,
    steps: int,
    seed: int,
    backend: str,
    kelly_values: list[float],
) -> list[dict[str, Any]]:
    cfg = DEFAULT_CONFIG
    rows: list[dict[str, Any]] = []

    for kelly in kelly_values:
        for gamma_target in DEFAULT_GAMMA_TARGETS:
            for tail_trigger in DEFAULT_TAIL_TRIGGER_VALUES:
                for stress_guard in DEFAULT_STRESS_GUARD_VALUES:
                    for feed in DEFAULT_FEEDS:
                        quoting = replace(
                            cfg.quoting,
                            kelly_fraction=float(kelly),
                            target_net_gamma=float(gamma_target),
                            stress_guard_multiple=float(stress_guard),
                        )
                        hedging = replace(cfg.hedging, tail_hedge_delta_trigger_ratio=float(tail_trigger))
                        run_cfg = replace(cfg, quoting=quoting, hedging=hedging)

                        summary = run_feed_backtest(
                            feed=feed,
                            cfg=run_cfg,
                            steps=steps,
                            seed=seed,
                            backend=backend,
                            survival_multiple=SURVIVAL_MULTIPLE,
                            frozen_surface=False,
                            stress_refresh_interval=20,
                            fill_interval=5,
                            max_inventory_positions=300,
                            simulate_fills=True,
                        )
                        rows.append(
                            {
                                "feed": feed,
                                "data_source": _data_source(feed),
                                "kelly_fraction": float(kelly),
                                "gamma_target": float(gamma_target),
                                "tail_hedge_trigger": float(tail_trigger),
                                "stress_guard_multiple": float(stress_guard),
                                "total_pnl": float(summary.total_pnl),
                                "var_capture": float(summary.var_capture),
                                "max_drawdown": float(summary.max_drawdown),
                                "avg_daily_theta": float(summary.avg_daily_theta),
                                "dd_limit": float(summary.dd_limit),
                                "survived": bool(summary.survived),
                                "tuning_survived": bool(summary.tuning_survived),
                                "profitable": bool(summary.profitable),
                                "final_bankroll": float(summary.final_bankroll),
                            }
                        )

    rows.sort(key=_row_sort_key)
    return rows


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.ERROR)

    args = _parse_args(argv)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        _merge_shards(results_dir)
        return

    if args.kelly_shard is None:
        kelly_values = list(DEFAULT_KELLY_VALUES)
    else:
        kelly_values = [float(args.kelly_shard)]

    rows = _run_rows(
        steps=int(args.steps),
        seed=int(args.seed),
        backend=str(args.backend),
        kelly_values=kelly_values,
    )

    if args.kelly_shard is None:
        _write_rows(results_dir / "param_sweep.csv", rows)
        _write_group_rows(results_dir / "param_sweep_grouped.csv", _group_rows(rows))
        _write_summary(results_dir / "param_sweep_stdout.txt", rows)
        return

    shard_path = _shard_path(results_dir, float(args.kelly_shard))
    _write_rows(shard_path, rows)
    print(f"Wrote shard results to {shard_path}")


if __name__ == "__main__":
    main()
