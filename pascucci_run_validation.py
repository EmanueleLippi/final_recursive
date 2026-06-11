"""Validation helpers for thesis-level Pascucci run gates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


EXPECTED_T12_N13 = {
    "model_name": "pascucci",
    "M": 10000,
    "N": 13,
    "D": 4,
    "T_total": 12.0,
    "block_size": 2.0,
    "passes": 2,
}
EXPECTED_BLOCK_START = np.asarray([0, 2, 4, 6, 8, 10], dtype=np.float32)
EXPECTED_BLOCK_END = np.asarray([2, 4, 6, 8, 10, 12], dtype=np.float32)
EXPECTED_PAPER_PLOTS = (
    "pascucci_paper_35_S_ou_band.png",
    "pascucci_paper_36_H_ou_band.png",
    "pascucci_paper_37_accumulated_cost.png",
    "pascucci_paper_38_alpha.png",
    "pascucci_paper_39_state_bands_S_V_Q.png",
    "pascucci_paper_40_controlled_uncontrolled.png",
)
MAX_SELECTED_SCORE_RATIO_VS_PASS0 = 1.25
MAX_SELECTED_SCORE_ABS_INCREASE_VS_PASS0 = 0.05
MIN_FINAL_CONTROL_WIN_RATE = 0.5
MAX_FINAL_DELTA_COST_MEAN = 0.0
MAX_FINAL_VIOLATION_RATE = 0.2
MAX_FINAL_VIOLATION_RATE_INCREASE_VS_PASS0 = 0.1


def _read_json(path: Path, failures: Optional[list[str]] = None) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        if failures is not None:
            failures.append(f"malformed JSON at {path}: {exc.msg}")
        return None
    if not isinstance(payload, dict):
        if failures is not None:
            failures.append(f"JSON root at {path} must be an object")
        return None
    return payload


def _is_finite_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(scalar)


def _assert_finite_json(obj: Any, failures: list[str], path: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_finite_json(value, failures, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            _assert_finite_json(value, failures, f"{path}[{index}]")
        return
    if isinstance(obj, float) and not math.isfinite(obj):
        failures.append(f"non-finite scalar at {path}: {obj!r}")


def _check_npz(path: Path, failures: list[str], *, required_shapes: Optional[Dict[str, tuple[int, ...]]] = None) -> None:
    if not path.exists():
        failures.append(f"missing artifact {path}")
        return
    with np.load(path, allow_pickle=False) as data:
        for key, expected_shape in (required_shapes or {}).items():
            if key not in data:
                failures.append(f"{path} missing key {key}")
                continue
            if tuple(data[key].shape) != tuple(expected_shape):
                failures.append(f"{path}:{key} shape {data[key].shape}, expected {expected_shape}")
        for key in data.files:
            arr = data[key]
            if arr.dtype.kind in "fciu" and (arr.size == 0 or not np.isfinite(arr).all()):
                failures.append(f"non-finite array {path}:{key}")


def _check_csv_logs(path: Path, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"missing pass log {path}")
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        failures.append(f"{path} must contain 6 block rows, got {len(rows)}")
    for row in rows:
        for field in ("eval_mean_loss", "eval_mean_loss_per_sample", "eval_mean_y0"):
            try:
                value = float(row.get(field, "nan"))
            except ValueError:
                value = float("nan")
            if not math.isfinite(value):
                failures.append(f"{path}:{field} non-finite in block {row.get('block')}")


def _load_csv_logs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _score_log_rows(rows: list[dict[str, str]], loss_key: str = "eval_mean_loss_per_sample") -> Optional[float]:
    scores: list[float] = []
    for row in rows:
        try:
            value = float(row.get(loss_key, "nan"))
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        scores.append(value)
    if len(scores) == 0:
        return None
    return float(np.mean(scores) + 0.35 * max(scores))


def _nested_float(payload: dict, keys: tuple[str, ...]) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if not _is_finite_scalar(current):
        return None
    return float(current)


def _check_science_gate(rec_dir: Path, results: Optional[dict], failures: list[str]) -> None:
    if results is None:
        return
    selected_pass_id = results.get("selected_pass_id")
    if selected_pass_id not in (1, 2):
        return

    pass0_rows = _load_csv_logs(rec_dir / "pass_00_logs.csv")
    selected_rows = _load_csv_logs(rec_dir / f"pass_{int(selected_pass_id) - 1:02d}_logs.csv")
    pass0_score = _score_log_rows(pass0_rows)
    selected_score = _score_log_rows(selected_rows)
    if pass0_score is not None and selected_score is not None and int(selected_pass_id) != 1:
        allowed_score = max(
            pass0_score * MAX_SELECTED_SCORE_RATIO_VS_PASS0,
            pass0_score + MAX_SELECTED_SCORE_ABS_INCREASE_VS_PASS0,
        )
        if selected_score > allowed_score:
            failures.append(
                "science gate: selected pass is dominated by pass0 loss score "
                f"(selected_pass_id={selected_pass_id}, selected_score={selected_score:.6g}, "
                f"pass0_score={pass0_score:.6g}, allowed={allowed_score:.6g})"
            )

    pass0_metrics = _read_json(rec_dir / "application_metrics_pass00.json")
    final_metrics = _read_json(rec_dir / "application_metrics_final.json")
    if final_metrics is None:
        return
    final_delta = _nested_float(final_metrics, ("comparison", "delta_cost_J_total_mean"))
    if final_delta is None:
        failures.append("science gate: final comparison.delta_cost_J_total_mean missing or non-finite")
    elif final_delta > MAX_FINAL_DELTA_COST_MEAN:
        failures.append(
            "science gate: selected pass controlled cost is worse than uncontrolled "
            f"(delta_cost_J_total_mean={final_delta:.6g}, max={MAX_FINAL_DELTA_COST_MEAN:.6g})"
        )
    final_win_rate = _nested_float(final_metrics, ("comparison", "cost_J_total_control_win_rate"))
    if final_win_rate is None:
        failures.append("science gate: final comparison.cost_J_total_control_win_rate missing or non-finite")
    elif final_win_rate < MIN_FINAL_CONTROL_WIN_RATE:
        failures.append(
            "science gate: selected pass controlled win-rate too low "
            f"(win_rate={final_win_rate:.6g}, min={MIN_FINAL_CONTROL_WIN_RATE:.6g})"
        )

    for key in (
        "q_lower_violation_rate",
        "q_upper_violation_rate",
        "v_lower_violation_rate",
        "v_upper_violation_rate",
    ):
        final_rate = _nested_float(final_metrics, ("diagnostics", key))
        pass0_rate = _nested_float(pass0_metrics or {}, ("diagnostics", key))
        if final_rate is None:
            failures.append(f"science gate: final diagnostics.{key} missing or non-finite")
            continue
        if final_rate > MAX_FINAL_VIOLATION_RATE:
            failures.append(
                "science gate: selected pass physical violation rate too high "
                f"({key}={final_rate:.6g}, max={MAX_FINAL_VIOLATION_RATE:.6g})"
            )
        if pass0_rate is not None and final_rate > pass0_rate + MAX_FINAL_VIOLATION_RATE_INCREASE_VS_PASS0:
            failures.append(
                "science gate: selected pass worsens physical violations versus pass0 "
                f"({key}={final_rate:.6g}, pass0={pass0_rate:.6g}, "
                f"max_increase={MAX_FINAL_VIOLATION_RATE_INCREASE_VS_PASS0:.6g})"
            )


def _resolve_run_root(path: Path) -> Optional[Path]:
    if (path / "run_config.json").exists():
        return path
    candidates = sorted(p for p in path.glob("run_*") if (p / "run_config.json").exists())
    if len(candidates) == 1:
        return candidates[0]
    return None


def _report_status(failures: Iterable[str], incomplete: Iterable[str], inconclusive: Iterable[str]) -> str:
    if list(failures):
        return "FAILED"
    if list(incomplete):
        return "INCOMPLETE"
    if list(inconclusive):
        return "INCONCLUSIVE"
    return "GREEN"


def validate_t12_gate_n13(path: str | Path) -> Dict[str, Any]:
    """Validate the post-run artifact contract for the T12/N13 Pascucci gate."""

    root_input = Path(path).expanduser()
    failures: list[str] = []
    incomplete: list[str] = []
    warnings: list[str] = []
    inconclusive: list[str] = []

    run_root = _resolve_run_root(root_input)
    if run_root is None:
        failures.append(f"no run_config.json found under {root_input}")
        return {
            "status": "FAILED",
            "run_root": None,
            "failures": failures,
            "incomplete": incomplete,
            "inconclusive": inconclusive,
            "warnings": warnings,
        }

    rec_dir = run_root / "recursive"
    cfg_path = run_root / "run_config.json"
    cfg = _read_json(cfg_path, failures)
    if cfg is None:
        if not cfg_path.exists():
            failures.append(f"missing run_config.json at {run_root}")
    else:
        for key, expected in EXPECTED_T12_N13.items():
            if cfg.get(key) != expected:
                failures.append(f"run_config {key}={cfg.get(key)!r}, expected {expected!r}")
        if cfg.get("state_labels") != ["S", "H", "V", "X_state"]:
            failures.append("run_config state_labels mismatch")
        if cfg.get("z_labels") != ["Z_S", "Z_H", "Z_V", "Z_X"]:
            failures.append("run_config z_labels mismatch")
        if cfg.get("application_metric_schema") != "pascucci_application_metrics_v2":
            failures.append("run_config application_metric_schema must be pascucci_application_metrics_v2")

    results_path = rec_dir / "results.json"
    results = _read_json(results_path, failures)
    if results is None:
        if not results_path.exists():
            failures.append(f"missing results.json at {results_path}")
    else:
        _assert_finite_json(results, failures, "results")
        if len(results.get("blocks", [])) != 6:
            failures.append("results.blocks must have length 6")
        if len(results.get("passes", [])) != 2:
            failures.append("results.passes must have length 2")
        if results.get("evaluation_bundle_M") != 10000:
            failures.append("results.evaluation_bundle_M must be 10000")
        if results.get("selected_pass_id") not in (1, 2):
            failures.append("results.selected_pass_id must be 1 or 2")
        if not _is_finite_scalar(results.get("selected_score")):
            failures.append(f"results.selected_score not finite: {results.get('selected_score')!r}")
        if results.get("pass_invalid_reasons"):
            failures.append(f"results contains invalid pass reasons: {results.get('pass_invalid_reasons')!r}")

    _check_npz(
        rec_dir / "evaluation_bundle.npz",
        failures,
        required_shapes={
            "Xi_initial": (10000, 4),
            "t_bundle": (6, 10000, 14, 1),
            "W_bundle": (6, 10000, 14, 4),
        },
    )
    bundle_path = rec_dir / "evaluation_bundle.npz"
    if bundle_path.exists():
        with np.load(bundle_path, allow_pickle=False) as bundle:
            if "block_t_start" in bundle and not np.allclose(bundle["block_t_start"], EXPECTED_BLOCK_START):
                failures.append("evaluation_bundle block_t_start mismatch")
            if "block_t_end" in bundle and not np.allclose(bundle["block_t_end"], EXPECTED_BLOCK_END):
                failures.append("evaluation_bundle block_t_end mismatch")

    for log_name in ("pass_00_logs.csv", "pass_01_logs.csv"):
        _check_csv_logs(rec_dir / log_name, failures)

    for stem in (
        "stitched_predictions_pass00",
        "stitched_predictions_pass01",
        "stitched_predictions_final",
        "application_metrics_pass00",
        "application_metrics_pass01",
        "application_metrics_final",
    ):
        suffix = ".json" if stem.startswith("application_metrics") else ".npz"
        path_to_check = rec_dir / f"{stem}{suffix}"
        if suffix == ".json":
            payload = _read_json(path_to_check, failures)
            if payload is None:
                if not path_to_check.exists():
                    failures.append(f"missing artifact {path_to_check}")
            else:
                _assert_finite_json(payload, failures, str(path_to_check))
        else:
            _check_npz(path_to_check, failures)

    for stem in ("application_metrics_pass00", "application_metrics_pass01", "application_metrics_final"):
        _check_npz(rec_dir / f"{stem}.npz", failures)

    plot_dir = rec_dir / "plots" / "pascucci_paper"
    manifest_path = plot_dir / "pascucci_paper_plots_manifest.json"
    manifest = _read_json(manifest_path, failures)
    if manifest is None:
        if not manifest_path.exists():
            failures.append(f"missing plot manifest {manifest_path}")
    elif manifest.get("schema") != "pascucci_paper_plots_v1":
        failures.append("plot manifest schema mismatch")
    for filename in EXPECTED_PAPER_PLOTS:
        plot_path = plot_dir / filename
        if not plot_path.exists():
            failures.append(f"missing plot {plot_path}")
        elif plot_path.stat().st_size <= 1000:
            failures.append(f"plot {plot_path} too small")

    _check_science_gate(rec_dir, results, failures)

    status = _report_status(failures, incomplete, inconclusive)
    return {
        "status": status,
        "run_root": str(run_root),
        "failures": failures,
        "incomplete": incomplete,
        "inconclusive": inconclusive,
        "warnings": warnings,
    }


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate Pascucci T12/N13 gate artifacts")
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args(argv)
    report = validate_t12_gate_n13(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
