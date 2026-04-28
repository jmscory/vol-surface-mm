from __future__ import annotations

import json
import sys

import pytest

from vol_surface_mm import cli


def test_help_includes_unified_commands(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["vol-surface-mm", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "artifacts" in out
    assert "sweep" in out
    assert "plot-sweep" in out
    assert "config" in out


def test_config_command_prints_seeding_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["vol-surface-mm", "config", "--seeding-mode", "short_straddle"],
    )

    cli.main()

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["seeding_mode"] == "short_straddle"


def test_artifacts_delegates_to_script_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> None:
        called["argv"] = [] if argv is None else list(argv)

    monkeypatch.setattr(cli.generate_artifacts, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vol-surface-mm",
            "artifacts",
            "--seed",
            "99",
            "--steps",
            "123",
            "--backend",
            "spline",
        ],
    )

    cli.main()

    assert called["argv"] == [
        "--seed",
        "99",
        "--steps",
        "123",
        "--backend",
        "spline",
        "--results-dir",
        "results",
    ]


def test_sweep_delegates_shard_and_merge_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> None:
        called["argv"] = [] if argv is None else list(argv)

    monkeypatch.setattr(cli.param_sweep, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vol-surface-mm",
            "sweep",
            "--kelly-shard",
            "0.10",
            "--merge-only",
        ],
    )

    cli.main()

    assert "--kelly-shard" in called["argv"]
    assert "0.1" in called["argv"]
    assert "--merge-only" in called["argv"]


def test_plot_sweep_delegates_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> None:
        called["argv"] = [] if argv is None else list(argv)

    monkeypatch.setattr(cli.plot_sweep, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vol-surface-mm",
            "plot-sweep",
            "--input",
            "results/custom_grouped.csv",
            "--output",
            "docs/assets/custom.png",
        ],
    )

    cli.main()

    assert called["argv"] == [
        "--input",
        "results/custom_grouped.csv",
        "--output",
        "docs/assets/custom.png",
    ]
