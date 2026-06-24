"""Validation helpers for thesis-level Pascucci run gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
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
    "pascucci_paper_39_forward_components_S_H_V_X.png",
    "pascucci_paper_40_controlled_uncontrolled.png",
)
EXPECTED_PAPER_STORIES = ("#35", "#36", "#37", "#38", "#39", "#40")
EXPECTED_PLOTMAKER_NATIVE_PLOTS = {
    "Y": "pascucci_plotmaker_backward_Y.png",
    "Z": "pascucci_plotmaker_Z_components.png",
}
EXPECTED_PAPER_PLOT_DIMENSIONS = {
    "#35": (1920, 960),
    "#36": (1920, 960),
    "#37": (1600, 960),
    "#38": (1600, 800),
    "#39": (2240, 1600),
    "#40": (1280, 800),
}
EXPECTED_PLOTMAKER_NATIVE_DIMENSIONS = {
    "Y": (1600, 960),
    "Z": (2240, 1600),
}
PAPER_PLOT_MANIFEST = "pascucci_paper_plots_manifest.json"
PAPER_PLOT_SCHEMA = "pascucci_paper_plots_v1"
PAPER_PLOT_DATA_SCHEMA = "pascucci_paper_plot_data_v1"
PAPER_PLOT_DATA_NPZ = "pascucci_paper_plot_data.npz"
APPLICATION_METRIC_SCHEMA = "pascucci_application_metrics_v2"
PASCUCCI_OU_CALIBRATION_SCHEMA = "pascucci_ou_calibration_v1"
MC_CONFIRMATION_SCHEMA = "pascucci_mc_confirmation_v1"
MC_CONFIRMATION_SOURCE = "independent_post_selection"
MC_CONFIRMATION_NPZ = "application_metrics_mc_confirmation.npz"
MIN_MC_CONFIRMATION_SAMPLE_PATHS = 10000
PAPER_PARITY_APPLICATION_KEYS = (
    "controlled_cost_J_total",
    "controlled_cost_J_running_cumulative",
    "controlled_cost_J_trajectory",
    "controlled_cost_J_trajectory_math",
    "controlled_alpha",
    "uncontrolled_cost_J_total",
    "uncontrolled_cost_J_running_cumulative",
    "uncontrolled_cost_J_trajectory",
    "uncontrolled_cost_J_trajectory_math",
    "uncontrolled_alpha",
)
MAX_SELECTED_SCORE_RATIO_VS_PASS0 = 1.25
MAX_SELECTED_SCORE_ABS_INCREASE_VS_PASS0 = 0.05
MIN_FINAL_CONTROL_WIN_RATE = 0.5
MAX_FINAL_DELTA_COST_MEAN = 0.0
MAX_FINAL_VIOLATION_RATE = 0.2
MAX_FINAL_VIOLATION_RATE_INCREASE_VS_PASS0 = 0.1
MC_CONFIRMATION_SUMMARY_ATOL = 5.0e-5
PAPER_PLOT_DATA_ATOL = 2.0e-4
PLOTMAKER_DATASET_NAME = "2025dicembre1"
PLOTMAKER_H_BASENAME = "2025dicembre1.csv"
PLOTMAKER_S_BASENAME = "2025dicembre1.xlsx"
PLOTMAKER_H_SHA256 = "75004dde0cd982f67c547c241ce704a4fc596380d5bcbca18169d8d6bc4b5c44"
PLOTMAKER_S_SHA256 = "7526faeaff806a250dd3736bc2eb7e1202c20d4635638c3b371d85c133eed5ef"
PLOTMAKER_T = 24.0
PLOTMAKER_N = 150
PLOTMAKER_M = 10000
PLOTMAKER_CALIBRATION_K = 2
PLOTMAKER_CALIBRATION_DT = 1.0
PLOTMAKER_PASCUCCI_SCALAR_PARAMS = {
    "l_v": 0.01,
    "l_a": 0.005,
    "c3": 10.0,
    "c4": 10.0,
    "gamma": 1.0,
    "d": 1.0,
    "x_max": 10.0,
    "v_max": 2.0,
    "v_min": -2.0,
    "s3": 0.01,
    "omega": 0.01,
    "c_h": 0.001,
    "c_con": 0.01,
}
PLOTMAKER_PARAM_ATOL = 1.0e-7


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_dimensions(path: Path) -> Optional[tuple[int, int]]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width, height = struct.unpack(">II", header[16:24])
    except struct.error:
        return None
    return int(width), int(height)


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


def _check_plotmaker_pascucci_scalar_params(params: Dict[str, Any], failures: list[str]) -> None:
    for name, expected in PLOTMAKER_PASCUCCI_SCALAR_PARAMS.items():
        value = params.get(name)
        if not _is_finite_scalar(value):
            failures.append(f"run_config params.{name} must be finite for plotmaker paper-parity, got {value!r}")
            continue
        actual = float(value)
        if abs(actual - float(expected)) > PLOTMAKER_PARAM_ATOL:
            failures.append(
                "run_config params."
                f"{name}={actual:.12g}, expected plotmaker.ipynb cell 8 value {float(expected):.12g}"
            )


def _check_npz(path: Path, failures: list[str], *, required_shapes: Optional[Dict[str, tuple[int, ...]]] = None) -> None:
    if not path.exists():
        failures.append(f"missing artifact {path}")
        return
    try:
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
    except (OSError, ValueError) as exc:
        failures.append(f"unreadable npz artifact {path}: {exc}")


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


def _as_flat_float_array(arr: np.ndarray, label: str, failures: list[str]) -> Optional[np.ndarray]:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    else:
        failures.append(f"{label} shape {values.shape}, expected (M,) or (M, 1)")
        return None
    if values.size == 0 or not np.isfinite(values).all():
        failures.append(f"{label} must be non-empty and finite")
        return None
    return values


def _as_pathwise_values(arr: np.ndarray, label: str, sample_paths: int, failures: list[str]) -> Optional[np.ndarray]:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] != int(sample_paths):
        failures.append(f"{label} shape {values.shape}, expected first dimension {int(sample_paths)}")
        return None
    if values.size == 0 or not np.isfinite(values).all():
        failures.append(f"{label} must be non-empty and finite")
        return None
    return values.reshape(-1)


def _check_summary_close(
    *,
    payload_value: Optional[float],
    computed_value: float,
    label: str,
    failures: list[str],
) -> None:
    if payload_value is None:
        failures.append(f"MC confirmation {label} missing or non-finite")
        return
    if abs(float(payload_value) - float(computed_value)) > MC_CONFIRMATION_SUMMARY_ATOL:
        failures.append(
            f"MC confirmation {label} does not match pathwise NPZ "
            f"(json={float(payload_value):.6g}, computed={float(computed_value):.6g})"
        )


def _check_mc_confirmation(
    *,
    rec_dir: Path,
    cfg: Optional[dict],
    results: Optional[dict],
    failures: list[str],
) -> None:
    """Validate an independent post-selection Monte Carlo confirmation artifact."""

    summary_path = rec_dir / "application_metrics_mc_confirmation.json"
    payload = _read_json(summary_path, failures)
    if payload is None:
        if not summary_path.exists():
            failures.append(f"missing independent MC confirmation at {summary_path}")
        return

    _assert_finite_json(payload, failures, "application_metrics_mc_confirmation")
    if payload.get("schema") != MC_CONFIRMATION_SCHEMA:
        failures.append(
            f"MC confirmation schema={payload.get('schema')!r}, expected {MC_CONFIRMATION_SCHEMA!r}"
        )
    if payload.get("source") != MC_CONFIRMATION_SOURCE:
        failures.append(
            "MC confirmation source must be independent_post_selection, "
            f"got {payload.get('source')!r}"
        )
    if payload.get("model_name") != "pascucci":
        failures.append(f"MC confirmation model_name={payload.get('model_name')!r}, expected 'pascucci'")
    if payload.get("independent_of_eval_bundle") is not True:
        failures.append("MC confirmation must declare independent_of_eval_bundle=true")
    if payload.get("eval_bundle_reused") is True:
        failures.append("MC confirmation must not reuse the selected-pass evaluation bundle")

    if isinstance(results, dict) and results.get("selected_pass_id") is not None:
        if payload.get("selected_pass_id") != results.get("selected_pass_id"):
            failures.append(
                "MC confirmation selected_pass_id mismatch "
                f"(json={payload.get('selected_pass_id')!r}, results={results.get('selected_pass_id')!r})"
            )

    horizon = payload.get("horizon")
    if not isinstance(horizon, dict):
        failures.append("MC confirmation horizon must be an object")
        horizon = {}
    cfg_T = cfg.get("T_total") if isinstance(cfg, dict) else None
    if _is_finite_scalar(cfg_T):
        mc_T = horizon.get("T_total")
        if not _is_finite_scalar(mc_T) or abs(float(mc_T) - float(cfg_T)) > 1.0e-6:
            failures.append(
                f"MC confirmation horizon.T_total={mc_T!r}, expected run_config T_total={float(cfg_T):.6g}"
            )

    seed = payload.get("seed", horizon.get("seed"))
    if not isinstance(seed, int) or isinstance(seed, bool):
        failures.append(f"MC confirmation seed must be an integer, got {seed!r}")
    eval_seed = None
    if isinstance(cfg, dict):
        seed_manifest = cfg.get("seed_manifest")
        if isinstance(seed_manifest, dict):
            eval_seed = seed_manifest.get("eval_seed")
    if isinstance(seed, int) and isinstance(eval_seed, int) and seed == eval_seed:
        failures.append(
            f"MC confirmation seed must differ from run_config seed_manifest.eval_seed ({eval_seed})"
        )

    npz_name = payload.get("pathwise_npz", MC_CONFIRMATION_NPZ)
    if not isinstance(npz_name, str) or npz_name.strip() == "":
        failures.append("MC confirmation pathwise_npz must be a non-empty relative path")
        return
    npz_path = Path(npz_name)
    if npz_path.is_absolute():
        failures.append("MC confirmation pathwise_npz must be relative to recursive dir")
        return
    arrays = _load_npz_arrays(rec_dir / npz_path, failures)
    if arrays is None:
        return

    required = ("controlled_cost_J_total", "uncontrolled_cost_J_total")
    for key in required:
        if key not in arrays:
            failures.append(f"MC confirmation NPZ missing key {key}")
    if any(key not in arrays for key in required):
        return

    controlled_total = _as_flat_float_array(
        arrays["controlled_cost_J_total"],
        "MC confirmation NPZ controlled_cost_J_total",
        failures,
    )
    uncontrolled_total = _as_flat_float_array(
        arrays["uncontrolled_cost_J_total"],
        "MC confirmation NPZ uncontrolled_cost_J_total",
        failures,
    )
    if controlled_total is None or uncontrolled_total is None:
        return
    if controlled_total.shape != uncontrolled_total.shape:
        failures.append(
            "MC confirmation controlled/uncontrolled total shapes differ "
            f"({controlled_total.shape} vs {uncontrolled_total.shape})"
        )
        return

    sample_paths = int(controlled_total.shape[0])
    if sample_paths < MIN_MC_CONFIRMATION_SAMPLE_PATHS:
        failures.append(
            f"MC confirmation sample_paths={sample_paths}, expected at least {MIN_MC_CONFIRMATION_SAMPLE_PATHS}"
        )
    horizon_paths = horizon.get("sample_paths")
    if horizon_paths is not None and horizon_paths != sample_paths:
        failures.append(
            f"MC confirmation horizon.sample_paths={horizon_paths!r}, expected NPZ sample_paths={sample_paths}"
        )
    paired_sample_count = _nested_float(payload, ("comparison", "paired_sample_count"))
    if paired_sample_count is None:
        failures.append("MC confirmation comparison.paired_sample_count missing or non-finite")
    elif abs(float(paired_sample_count) - float(sample_paths)) > 1.0e-9:
        failures.append(
            "MC confirmation comparison.paired_sample_count mismatch "
            f"(json={float(paired_sample_count):.6g}, expected NPZ sample_paths={sample_paths})"
        )

    delta = controlled_total - uncontrolled_total
    delta_mean = float(np.mean(delta))
    win_rate = float(np.mean(controlled_total < uncontrolled_total))
    _check_summary_close(
        payload_value=_nested_float(payload, ("comparison", "delta_cost_J_total_mean")),
        computed_value=delta_mean,
        label="comparison.delta_cost_J_total_mean",
        failures=failures,
    )
    _check_summary_close(
        payload_value=_nested_float(payload, ("comparison", "cost_J_total_control_win_rate")),
        computed_value=win_rate,
        label="comparison.cost_J_total_control_win_rate",
        failures=failures,
    )

    if delta_mean > MAX_FINAL_DELTA_COST_MEAN:
        failures.append(
            "MC confirmation: controlled cost is worse than uncontrolled "
            f"(delta_cost_J_total_mean={delta_mean:.6g}, max={MAX_FINAL_DELTA_COST_MEAN:.6g})"
        )
    if win_rate < MIN_FINAL_CONTROL_WIN_RATE:
        failures.append(
            "MC confirmation: controlled win-rate too low "
            f"(win_rate={win_rate:.6g}, min={MIN_FINAL_CONTROL_WIN_RATE:.6g})"
        )

    for key in (
        "q_lower_violation_rate",
        "q_upper_violation_rate",
        "v_lower_violation_rate",
        "v_upper_violation_rate",
    ):
        npz_key = key.removesuffix("_rate")
        if npz_key not in arrays:
            failures.append(f"MC confirmation NPZ missing key {npz_key}")
            continue
        values = _as_pathwise_values(
            arrays[npz_key],
            f"MC confirmation NPZ {npz_key}",
            sample_paths,
            failures,
        )
        if values is None:
            continue
        rate = float(np.mean(values > 0.0))
        _check_summary_close(
            payload_value=_nested_float(payload, ("diagnostics", key)),
            computed_value=rate,
            label=f"diagnostics.{key}",
            failures=failures,
        )
        if rate > MAX_FINAL_VIOLATION_RATE:
            failures.append(
                "MC confirmation: physical violation rate too high "
                f"({key}={rate:.6g}, max={MAX_FINAL_VIOLATION_RATE:.6g})"
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


def _classify_paper_parity_failure(message: str) -> str:
    text = str(message)
    if text.startswith("science gate:"):
        return "science"
    if text.startswith("MC confirmation") or text.startswith("missing independent MC confirmation"):
        return "mc_confirmation"
    return "contract"


def _paper_parity_diagnostics(
    failures: Iterable[str],
    incomplete: Iterable[str],
    inconclusive: Iterable[str],
) -> Dict[str, Any]:
    failure_categories = {
        "contract": [],
        "science": [],
        "mc_confirmation": [],
    }
    incomplete_list = list(incomplete)
    inconclusive_list = list(inconclusive)
    for failure in failures:
        failure_categories[_classify_paper_parity_failure(str(failure))].append(str(failure))
    return {
        "contract_status": _report_status(failure_categories["contract"], incomplete_list, inconclusive_list),
        "science_status": "FAILED" if failure_categories["science"] else "GREEN",
        "mc_confirmation_status": "FAILED" if failure_categories["mc_confirmation"] else "GREEN",
        "failure_counts": {
            "contract": len(failure_categories["contract"]),
            "science": len(failure_categories["science"]),
            "mc_confirmation": len(failure_categories["mc_confirmation"]),
            "incomplete": len(incomplete_list),
            "inconclusive": len(inconclusive_list),
        },
        "failure_categories": failure_categories,
    }


def _paper_parity_report(
    *,
    status: str,
    run_root: Optional[str],
    failures: list[str],
    incomplete: list[str],
    inconclusive: list[str],
    warnings: list[str],
) -> Dict[str, Any]:
    return {
        "status": status,
        "run_root": run_root,
        "failures": failures,
        "incomplete": incomplete,
        "inconclusive": inconclusive,
        "warnings": warnings,
        **_paper_parity_diagnostics(failures, incomplete, inconclusive),
    }


def _load_npz_arrays(path: Path, failures: list[str]) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        failures.append(f"missing artifact {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError) as exc:
        failures.append(f"unreadable npz artifact {path}: {exc}")
        return None
    for key, arr in arrays.items():
        if arr.dtype.kind in "fciu" and (arr.size == 0 or not np.isfinite(arr).all()):
            failures.append(f"non-finite array {path}:{key}")
    return arrays


def _calibration_metadata_source(metadata: dict, key: str, failures: list[str], warnings: list[str]) -> None:
    value = metadata.get("source_path")
    if not isinstance(value, str) or value.strip() == "":
        failures.append(f"run_config params.pascucci_calibration.{key}.source_path missing")
        return
    source_path = Path(value).expanduser()
    if not source_path.exists():
        failures.append(f"calibration source_path not found locally: {source_path}")
        return
    expected_sha = metadata.get("source_sha256")
    if isinstance(expected_sha, str) and expected_sha.strip():
        try:
            import hashlib

            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"cannot hash calibration source_path {source_path}: {exc}")
            return
        if digest != expected_sha:
            failures.append(
                f"calibration source_sha256 mismatch for {key}: "
                f"metadata={expected_sha}, actual={digest}"
            )
    else:
        warnings.append(f"run_config params.pascucci_calibration.{key}.source_sha256 missing")
    prepared_points = metadata.get("prepared_points")
    if prepared_points is not None:
        if not isinstance(prepared_points, int) or prepared_points <= 1:
            failures.append(
                f"run_config params.pascucci_calibration.{key}.prepared_points invalid: "
                f"{prepared_points!r}"
            )


def _ou_param_max_abs_diff(actual: Any, expected: Any) -> tuple[float, str]:
    actual_keys = set(actual) if isinstance(actual, dict) else set()
    expected_keys = set(expected) if isinstance(expected, dict) else set()
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        return float("inf"), "; ".join(detail) or "key mismatch"

    max_diff = 0.0
    max_key = ""
    for key in sorted(expected_keys):
        actual_arr = np.asarray(actual[key], dtype=np.float64)
        expected_arr = np.asarray(expected[key], dtype=np.float64)
        if actual_arr.shape != expected_arr.shape:
            return float("inf"), f"{key} shape {actual_arr.shape}, expected {expected_arr.shape}"
        if actual_arr.size == 0 and expected_arr.size == 0:
            continue
        if not np.isfinite(actual_arr).all() or not np.isfinite(expected_arr).all():
            return float("inf"), f"{key} contains non-finite values"
        diff = float(np.max(np.abs(actual_arr - expected_arr))) if actual_arr.size else 0.0
        if diff > max_diff:
            max_diff = diff
            max_key = str(key)
    return max_diff, max_key


def _check_pascucci_calibration_recomputes(params: dict, calibration: dict, failures: list[str]) -> None:
    """Recompute OU params from source files and compare against run_config params."""

    H_metadata = calibration.get("H_metadata")
    S_metadata = calibration.get("S_metadata")
    if not isinstance(H_metadata, dict) or not isinstance(S_metadata, dict):
        return
    if "params_H" not in params or "params_S" not in params:
        failures.append("run_config params must include params_H and params_S for calibration recompute")
        return

    try:
        from .pascucci_calibration import calibrate_pascucci_ou_inputs, serialize_ou_params
        from .pascucci_data import prepare_H, prepare_S

        H_path = Path(str(H_metadata.get("source_path", ""))).expanduser()
        S_path = Path(str(S_metadata.get("source_path", ""))).expanduser()
        H_series = prepare_H(
            str(H_path),
            n=int(H_metadata.get("n_per_hour", 1)),
            mul_factor=float(H_metadata.get("mul_factor", 1.0)),
        )
        S_series = prepare_S(
            str(S_path),
            n=int(S_metadata.get("n_per_hour", 1)),
            mul_factor=float(S_metadata.get("mul_factor", 1.0)),
        )
        K = int(calibration.get("K"))
        recomputed = calibrate_pascucci_ou_inputs(
            H_series,
            S_series,
            K=K,
            dt=float(calibration.get("dt")),
            start_hour=float(calibration.get("start_hour", 0.0)),
            log_price=bool(calibration.get("log_price", True)),
        )
        expected_H = serialize_ou_params(recomputed["params_H"], K)
        expected_S = serialize_ou_params(recomputed["params_S"], K)
    except Exception as exc:
        failures.append(f"cannot recompute Pascucci calibration from H/S sources: {exc}")
        return

    for label, actual, expected in (
        ("params_H", params.get("params_H"), expected_H),
        ("params_S", params.get("params_S"), expected_S),
    ):
        max_diff, detail = _ou_param_max_abs_diff(actual, expected)
        if not np.isfinite(max_diff) or max_diff > 2.0e-6:
            failures.append(
                f"recomputed Pascucci calibration mismatch for {label}: "
                f"max_abs={max_diff:.6g} at {detail}"
            )


def _check_paper_like_xi_initial(xi: np.ndarray, failures: list[str]) -> None:
    if xi.ndim != 2 or xi.shape[1] < 4:
        failures.append(f"evaluation_bundle Xi_initial shape {xi.shape}, expected (M, >=4)")
        return
    if xi.shape[0] < 32:
        failures.append(f"evaluation_bundle Xi_initial has too few paths for distribution diagnostics: {xi.shape[0]}")
        return

    S = np.asarray(xi[:, 0], dtype=np.float64)
    H = np.asarray(xi[:, 1], dtype=np.float64)
    V = np.asarray(xi[:, 2], dtype=np.float64)
    Q = np.asarray(xi[:, 3], dtype=np.float64)
    checks = (
        ("S mean", float(np.mean(S)), -2.3, 0.20),
        ("H mean", float(np.mean(H)), 0.4, 0.25),
        ("V mean", float(np.mean(V)), 0.0, 0.25),
        ("Q mean", float(np.mean(Q)), 5.0, 0.45),
    )
    for label, value, target, tol in checks:
        if abs(value - target) > tol:
            failures.append(f"Xi_initial paper-like distribution mismatch: {label}={value:.6g}, expected {target:.6g} +/- {tol:.6g}")

    ranges = (
        ("S std", float(np.std(S)), 0.10, 0.35),
        ("H std", float(np.std(H)), 0.25, 0.80),
        ("V std", float(np.std(V)), 0.65, 1.35),
        ("Q std", float(np.std(Q)), 1.85, 2.75),
    )
    for label, value, lower, upper in ranges:
        if value < lower or value > upper:
            failures.append(f"Xi_initial paper-like distribution mismatch: {label}={value:.6g}, expected in [{lower:.6g}, {upper:.6g}]")
    quantile_checks = (
        ("S q05", float(np.quantile(S, 0.05)), -2.3 - 1.645 * 0.2, 0.16),
        ("S q50", float(np.quantile(S, 0.50)), -2.3, 0.08),
        ("S q95", float(np.quantile(S, 0.95)), -2.3 + 1.645 * 0.2, 0.16),
        ("H q05", float(np.quantile(H, 0.05)), 0.4 - 1.645 * 0.5, 0.32),
        ("H q50", float(np.quantile(H, 0.50)), 0.4, 0.16),
        ("H q95", float(np.quantile(H, 0.95)), 0.4 + 1.645 * 0.5, 0.32),
        ("V q05", float(np.quantile(V, 0.05)), -1.645, 0.45),
        ("V q50", float(np.quantile(V, 0.50)), 0.0, 0.20),
        ("V q95", float(np.quantile(V, 0.95)), 1.645, 0.45),
        ("Q q05", float(np.quantile(Q, 0.05)), 1.4, 0.35),
        ("Q q50", float(np.quantile(Q, 0.50)), 5.0, 0.65),
        ("Q q95", float(np.quantile(Q, 0.95)), 8.6, 0.35),
    )
    for label, value, target, tol in quantile_checks:
        if abs(value - target) > tol:
            failures.append(
                "Xi_initial paper-like distribution mismatch: "
                f"{label}={value:.6g}, expected {target:.6g} +/- {tol:.6g}"
            )
    q_min = float(np.min(Q))
    q_max = float(np.max(Q))
    if q_min < 1.0 - 1.0e-6 or q_max > 9.0 + 1.0e-6:
        failures.append(f"Xi_initial Q outside paper support [1, 9]: min={q_min:.6g}, max={q_max:.6g}")
    if q_min > 1.5 or q_max < 8.5:
        failures.append(f"Xi_initial Q does not cover paper-like U(1,9) support: min={q_min:.6g}, max={q_max:.6g}")


def _check_paper_results_contract(
    *,
    cfg: Optional[dict],
    results: Optional[dict],
    results_path: Path,
    failures: list[str],
) -> None:
    if results is None:
        if not results_path.exists():
            failures.append(f"missing results.json at {results_path}")
        return

    _assert_finite_json(results, failures, "results")
    if results.get("selected_pass_id") is None:
        failures.append("results.selected_pass_id missing")
    elif not isinstance(results.get("selected_pass_id"), int):
        failures.append(f"results.selected_pass_id must be an integer, got {results.get('selected_pass_id')!r}")
    if not _is_finite_scalar(results.get("selected_score")):
        failures.append(f"results.selected_score not finite: {results.get('selected_score')!r}")
    if results.get("pass_invalid_reasons"):
        failures.append(f"results contains invalid pass reasons: {results.get('pass_invalid_reasons')!r}")

    blocks = results.get("blocks")
    if not isinstance(blocks, list) or len(blocks) == 0:
        failures.append("results.blocks must be a non-empty list")
        blocks = []
    passes = results.get("passes")
    if not isinstance(passes, list) or len(passes) == 0:
        failures.append("results.passes must be a non-empty list")
        passes = []

    if not isinstance(cfg, dict):
        return
    expected_passes = cfg.get("passes")
    if isinstance(expected_passes, int) and expected_passes > 0 and len(passes) != expected_passes:
        failures.append(f"results.passes length {len(passes)}, expected run_config passes={expected_passes}")

    T_total = cfg.get("T_total")
    block_size = cfg.get("block_size")
    if _is_finite_scalar(T_total) and _is_finite_scalar(block_size) and float(block_size) > 0.0:
        expected_blocks_float = float(T_total) / float(block_size)
        expected_blocks = int(round(expected_blocks_float))
        if abs(expected_blocks_float - expected_blocks) <= 1.0e-6 and len(blocks) != expected_blocks:
            failures.append(
                f"results.blocks length {len(blocks)}, expected {expected_blocks} "
                f"from T_total/block_size"
            )
        if len(blocks) > 0:
            first_start = blocks[0].get("t_start") if isinstance(blocks[0], dict) else None
            last_end = blocks[-1].get("t_end") if isinstance(blocks[-1], dict) else None
            if not _is_finite_scalar(first_start) or abs(float(first_start)) > 1.0e-6:
                failures.append(f"results.blocks first t_start={first_start!r}, expected 0.0")
            if not _is_finite_scalar(last_end) or abs(float(last_end) - float(T_total)) > 1.0e-5:
                failures.append(f"results.blocks final t_end={last_end!r}, expected T_total={float(T_total):.6g}")


def _check_application_cost_accounting(application: dict[str, np.ndarray], failures: list[str]) -> None:
    def _require(key: str) -> Optional[np.ndarray]:
        value = application.get(key)
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64)

    for prefix in ("controlled", "uncontrolled"):
        running = _require(f"{prefix}_cost_J_running")
        terminal = _require(f"{prefix}_cost_J_terminal")
        total = _require(f"{prefix}_cost_J_total")
        cumulative = _require(f"{prefix}_cost_J_running_cumulative")
        trajectory = _require(f"{prefix}_cost_J_trajectory")
        trajectory_math = _require(f"{prefix}_cost_J_trajectory_math")
        if any(value is None for value in (running, terminal, total, cumulative, trajectory, trajectory_math)):
            continue
        if cumulative.ndim != 3 or trajectory.ndim != 3 or trajectory_math.ndim != 3:
            continue
        if cumulative.shape[0] != running.shape[0] or cumulative.shape[2] != 1:
            failures.append(f"application_metrics_final.npz:{prefix}_cost_J_running_cumulative shape is inconsistent")
            continue
        if trajectory.shape[0] != total.shape[0] or trajectory.shape[2] != 1:
            failures.append(f"application_metrics_final.npz:{prefix}_cost_J_trajectory shape is inconsistent")
            continue
        if trajectory_math.shape[0] != total.shape[0] or trajectory_math.shape[2] != 1:
            failures.append(f"application_metrics_final.npz:{prefix}_cost_J_trajectory_math shape is inconsistent")
            continue
        if cumulative.shape[1] == 0 or trajectory.shape[1] == 0 or trajectory_math.shape[1] == 0:
            failures.append(f"application_metrics_final.npz:{prefix} cost traces must be non-empty")
            continue
        running_end = cumulative[:, -1, :]
        trajectory_math_end = trajectory_math[:, -1, :]
        if not np.allclose(running_end, running, rtol=2.0e-5, atol=2.0e-5):
            max_err = float(np.max(np.abs(running_end - running)))
            failures.append(
                f"application cost accounting mismatch: {prefix}_cost_J_running_cumulative[-1] "
                f"!= {prefix}_cost_J_running (max_abs={max_err:.6g})"
            )
        if not np.allclose(running + terminal, total, rtol=2.0e-5, atol=2.0e-5):
            max_err = float(np.max(np.abs((running + terminal) - total)))
            failures.append(
                f"application cost accounting mismatch: {prefix}_running + {prefix}_terminal "
                f"!= {prefix}_total (max_abs={max_err:.6g})"
            )
        if not np.allclose(trajectory_math_end, total, rtol=2.0e-5, atol=2.0e-5):
            max_err = float(np.max(np.abs(trajectory_math_end - total)))
            failures.append(
                f"application cost accounting mismatch: {prefix}_cost_J_trajectory_math[-1] "
                f"!= {prefix}_cost_J_total (max_abs={max_err:.6g})"
            )


def _plot_data_stats(values: np.ndarray, *, quantiles: tuple[float, ...]) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    stats = {"mean": np.mean(arr, axis=0).astype(np.float32)}
    for quantile in quantiles:
        stats[f"q{int(round(float(quantile) * 100)):02d}"] = np.quantile(
            arr,
            float(quantile),
            axis=0,
        ).astype(np.float32)
    return stats


def _check_plot_data_close(
    plot_data_arrays: dict[str, np.ndarray],
    key: str,
    expected: np.ndarray,
    failures: list[str],
    *,
    atol: float = PAPER_PLOT_DATA_ATOL,
) -> None:
    if key not in plot_data_arrays:
        return
    actual = np.asarray(plot_data_arrays[key], dtype=np.float32)
    expected_arr = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected_arr.shape:
        failures.append(f"plot data {key} shape {actual.shape}, expected {expected_arr.shape}")
        return
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(expected_arr).all():
        failures.append(f"plot data {key} must be non-empty and finite")
        return
    if not np.allclose(actual, expected_arr, rtol=2.0e-5, atol=float(atol)):
        max_abs_diff = float(np.max(np.abs(actual - expected_arr)))
        failures.append(f"plot data {key} mismatch against paper-plot source (max_abs_diff={max_abs_diff:.6g})")


def _check_paper_plot_data_consistency(
    *,
    plot_data_arrays: dict[str, np.ndarray],
    cfg: Optional[dict],
    stitched: Optional[dict[str, np.ndarray]],
    application: Optional[dict[str, np.ndarray]],
    failures: list[str],
) -> None:
    if not isinstance(cfg, dict) or stitched is None or application is None:
        return
    params = cfg.get("params")
    if not isinstance(params, dict):
        return
    t = stitched.get("t")
    X = stitched.get("X")
    if t is None or X is None or t.ndim != 3 or X.ndim != 3 or X.shape[2] < 4:
        return
    time = np.asarray(t[0, :, 0], dtype=np.float32)
    if time.size == 0 or not np.isfinite(time).all():
        return
    horizon_T = float(time[-1])

    try:
        from .pascucci_plotting import _load_to_ema_ou_reference, _load_to_ema_uncontrolled_reference
    except ImportError as exc:
        failures.append(f"plot data consistency cannot import Pascucci plot references: {exc}")
        return

    try:
        ou_references, ou_summary = _load_to_ema_ou_reference(params, horizon_T=horizon_T)
    except Exception as exc:  # pragma: no cover - surfaced as a validator failure
        failures.append(f"plot data consistency cannot rebuild to_ema OU references: {exc}")
        ou_references, ou_summary = {}, {}
    if isinstance(ou_summary, dict) and ou_summary.get("source") == "calibration_metadata":
        dt_sim = float((ou_summary.get("simulation") or {}).get("dt_sim", 0.5))
        dt_real = float(ou_summary.get("dt_real", 1.0))
        for story, prefix in (("#35", "plot35"), ("#36", "plot36")):
            ref = ou_references.get(story, {}) if isinstance(ou_references, dict) else {}
            paths = ref.get("paths")
            real = ref.get("real_path")
            if paths is None or real is None:
                continue
            paths_arr = np.asarray(paths, dtype=np.float32)
            real_arr = np.asarray(real, dtype=np.float32)
            sim_time = (np.arange(paths_arr.shape[1], dtype=np.float32) * np.float32(dt_sim)).astype(np.float32)
            real_time = (np.arange(real_arr.shape[0], dtype=np.float32) * np.float32(dt_real)).astype(np.float32)
            real_points = max(1, min(int(float(ref.get("T", 0.0)) / dt_real), int(real_arr.shape[0])))
            sim_stats = _plot_data_stats(paths_arr, quantiles=(0.05, 0.50, 0.95))
            _check_plot_data_close(plot_data_arrays, f"{prefix}_time_sim", sim_time, failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_sim_mean", sim_stats["mean"], failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_sim_q05", sim_stats["q05"], failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_sim_q50", sim_stats["q50"], failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_sim_q95", sim_stats["q95"], failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_time_real", real_time[:real_points], failures)
            _check_plot_data_close(plot_data_arrays, f"{prefix}_real", real_arr[:real_points], failures)

    controlled_trace = application.get("controlled_cost_J_trajectory")
    if controlled_trace is not None and controlled_trace.ndim == 3:
        controlled_stats = _plot_data_stats(controlled_trace[:, :, 0], quantiles=(0.10, 0.90))
        _check_plot_data_close(plot_data_arrays, "plot37_controlled_time", time, failures)
        _check_plot_data_close(plot_data_arrays, "plot37_controlled_mean", controlled_stats["mean"], failures)
        _check_plot_data_close(plot_data_arrays, "plot37_controlled_q10", controlled_stats["q10"], failures)
        _check_plot_data_close(plot_data_arrays, "plot37_controlled_q90", controlled_stats["q90"], failures)

    controlled_total = application.get("controlled_cost_J_total")
    if controlled_total is not None:
        controlled_total_stats = _plot_data_stats(
            np.asarray(controlled_total, dtype=np.float32).reshape(-1, 1),
            quantiles=(0.05, 0.50, 0.95),
        )
        _check_plot_data_close(plot_data_arrays, "plot40_controlled_mean", controlled_total_stats["mean"], failures)
        _check_plot_data_close(plot_data_arrays, "plot40_controlled_q05", controlled_total_stats["q05"], failures)
        _check_plot_data_close(plot_data_arrays, "plot40_controlled_q50", controlled_total_stats["q50"], failures)
        _check_plot_data_close(plot_data_arrays, "plot40_controlled_q95", controlled_total_stats["q95"], failures)

    try:
        to_ema_uncontrolled, to_ema_summary = _load_to_ema_uncontrolled_reference(params, horizon_T=horizon_T)
    except Exception as exc:  # pragma: no cover - surfaced as a validator failure
        failures.append(f"plot data consistency cannot rebuild to_ema uncontrolled reference: {exc}")
        to_ema_uncontrolled, to_ema_summary = {}, {}
    if isinstance(to_ema_summary, dict) and to_ema_summary.get("source") == "calibration_metadata":
        J_paths = np.asarray(to_ema_uncontrolled.get("J_paths"), dtype=np.float32)
        J_time = np.asarray(to_ema_uncontrolled.get("time"), dtype=np.float32)
        if J_paths.ndim == 2 and J_time.ndim == 1:
            uncontrolled_stats = _plot_data_stats(J_paths, quantiles=(0.10, 0.90))
            _check_plot_data_close(plot_data_arrays, "plot37_uncontrolled_time", J_time, failures)
            _check_plot_data_close(plot_data_arrays, "plot37_uncontrolled_mean", uncontrolled_stats["mean"], failures)
            _check_plot_data_close(plot_data_arrays, "plot37_uncontrolled_q10", uncontrolled_stats["q10"], failures)
            _check_plot_data_close(plot_data_arrays, "plot37_uncontrolled_q90", uncontrolled_stats["q90"], failures)
            uncontrolled_final_stats = _plot_data_stats(J_paths[:, -1:].astype(np.float32), quantiles=(0.05, 0.50, 0.95))
            _check_plot_data_close(plot_data_arrays, "plot40_uncontrolled_mean", uncontrolled_final_stats["mean"], failures)
            _check_plot_data_close(plot_data_arrays, "plot40_uncontrolled_q05", uncontrolled_final_stats["q05"], failures)
            _check_plot_data_close(plot_data_arrays, "plot40_uncontrolled_q50", uncontrolled_final_stats["q50"], failures)
            _check_plot_data_close(plot_data_arrays, "plot40_uncontrolled_q95", uncontrolled_final_stats["q95"], failures)

    controlled_alpha = application.get("controlled_alpha")
    uncontrolled_alpha = application.get("uncontrolled_alpha")
    if controlled_alpha is not None and controlled_alpha.ndim == 3:
        alpha_stats = _plot_data_stats(controlled_alpha[:, :, 0], quantiles=(0.05, 0.50, 0.95))
        _check_plot_data_close(plot_data_arrays, "plot38_time", time[:-1], failures)
        _check_plot_data_close(plot_data_arrays, "plot38_controlled_mean", alpha_stats["mean"], failures)
        _check_plot_data_close(plot_data_arrays, "plot38_controlled_q05", alpha_stats["q05"], failures)
        _check_plot_data_close(plot_data_arrays, "plot38_controlled_q50", alpha_stats["q50"], failures)
        _check_plot_data_close(plot_data_arrays, "plot38_controlled_q95", alpha_stats["q95"], failures)
    if uncontrolled_alpha is not None and uncontrolled_alpha.ndim == 3:
        _check_plot_data_close(
            plot_data_arrays,
            "plot38_uncontrolled_mean",
            np.mean(uncontrolled_alpha[:, :, 0], axis=0).astype(np.float32),
            failures,
        )

    state_components = {
        "plot39_S": np.exp(np.clip(np.asarray(X[:, :, 0], dtype=np.float32), -50.0, 50.0)).astype(np.float32),
        "plot39_H": np.asarray(X[:, :, 1], dtype=np.float32),
        "plot39_V": np.asarray(X[:, :, 2], dtype=np.float32),
        "plot39_X": np.asarray(X[:, :, 3], dtype=np.float32),
    }
    _check_plot_data_close(plot_data_arrays, "plot39_time", time, failures)
    if "plot39_sample_indices" in plot_data_arrays:
        sample_idx = np.asarray(plot_data_arrays["plot39_sample_indices"], dtype=np.int64).reshape(-1)
        expected_sample_idx = np.arange(min(3, int(X.shape[0])), dtype=np.int64)
        if sample_idx.shape != expected_sample_idx.shape or not np.array_equal(sample_idx, expected_sample_idx):
            failures.append(
                "plot data plot39_sample_indices must be deterministic first paths "
                f"{expected_sample_idx.tolist()}, got {sample_idx.tolist()}"
            )
        if sample_idx.size > 3:
            failures.append(f"plot data plot39_sample_indices has {sample_idx.size} entries, expected at most 3")
        if sample_idx.size > 0 and (np.min(sample_idx) < 0 or np.max(sample_idx) >= X.shape[0]):
            failures.append("plot data plot39_sample_indices outside stitched path range")
    else:
        sample_idx = np.asarray([], dtype=np.int64)
    for prefix, values in state_components.items():
        stats = _plot_data_stats(values, quantiles=(0.10, 0.50, 0.90))
        _check_plot_data_close(plot_data_arrays, f"{prefix}_mean", stats["mean"], failures)
        _check_plot_data_close(plot_data_arrays, f"{prefix}_q10", stats["q10"], failures)
        _check_plot_data_close(plot_data_arrays, f"{prefix}_q50", stats["q50"], failures)
        _check_plot_data_close(plot_data_arrays, f"{prefix}_q90", stats["q90"], failures)
        if sample_idx.size > 0:
            _check_plot_data_close(plot_data_arrays, f"{prefix}_samples", values[sample_idx], failures)

    Y = stitched.get("Y")
    if Y is not None and Y.ndim == 3 and Y.shape[:2] == X.shape[:2] and Y.shape[2] == 1:
        Y_values = np.asarray(Y[:, :, 0], dtype=np.float32)
        Y_stats = _plot_data_stats(Y_values, quantiles=(0.10, 0.50, 0.90))
        _check_plot_data_close(plot_data_arrays, "plotmaker_Y_time", time, failures)
        _check_plot_data_close(plot_data_arrays, "plotmaker_Y_mean", Y_stats["mean"], failures)
        _check_plot_data_close(plot_data_arrays, "plotmaker_Y_q10", Y_stats["q10"], failures)
        _check_plot_data_close(plot_data_arrays, "plotmaker_Y_q50", Y_stats["q50"], failures)
        _check_plot_data_close(plot_data_arrays, "plotmaker_Y_q90", Y_stats["q90"], failures)
        if "plotmaker_Y_sample_indices" in plot_data_arrays:
            y_sample_idx = np.asarray(plot_data_arrays["plotmaker_Y_sample_indices"], dtype=np.int64).reshape(-1)
            expected_y_sample_idx = np.arange(min(3, int(Y.shape[0])), dtype=np.int64)
            if y_sample_idx.shape != expected_y_sample_idx.shape or not np.array_equal(
                y_sample_idx,
                expected_y_sample_idx,
            ):
                failures.append(
                    "plot data plotmaker_Y_sample_indices must be deterministic first paths "
                    f"{expected_y_sample_idx.tolist()}, got {y_sample_idx.tolist()}"
                )
            if y_sample_idx.size > 0:
                _check_plot_data_close(plot_data_arrays, "plotmaker_Y_samples", Y_values[y_sample_idx], failures)

    Z = stitched.get("Z")
    if Z is not None and Z.ndim == 3 and Z.shape[:2] == X.shape[:2] and Z.shape[2] >= 4:
        Z_arr = np.asarray(Z, dtype=np.float32)
        _check_plot_data_close(plot_data_arrays, "plotmaker_Z_time", time, failures)
        if "plotmaker_Z_sample_indices" in plot_data_arrays:
            z_sample_idx = np.asarray(plot_data_arrays["plotmaker_Z_sample_indices"], dtype=np.int64).reshape(-1)
            expected_z_sample_idx = np.arange(min(3, int(Z.shape[0])), dtype=np.int64)
            if z_sample_idx.shape != expected_z_sample_idx.shape or not np.array_equal(
                z_sample_idx,
                expected_z_sample_idx,
            ):
                failures.append(
                    "plot data plotmaker_Z_sample_indices must be deterministic first paths "
                    f"{expected_z_sample_idx.tolist()}, got {z_sample_idx.tolist()}"
                )
        else:
            z_sample_idx = np.asarray([], dtype=np.int64)
        for component, label in enumerate(("Z_S", "Z_H", "Z_V", "Z_X")):
            values = Z_arr[:, :, component]
            stats = _plot_data_stats(values, quantiles=(0.10, 0.50, 0.90))
            _check_plot_data_close(plot_data_arrays, f"plotmaker_Z_{label}_mean", stats["mean"], failures)
            _check_plot_data_close(plot_data_arrays, f"plotmaker_Z_{label}_q10", stats["q10"], failures)
            _check_plot_data_close(plot_data_arrays, f"plotmaker_Z_{label}_q50", stats["q50"], failures)
            _check_plot_data_close(plot_data_arrays, f"plotmaker_Z_{label}_q90", stats["q90"], failures)
            if z_sample_idx.size > 0:
                _check_plot_data_close(
                    plot_data_arrays,
                    f"plotmaker_Z_{label}_samples",
                    values[z_sample_idx],
                    failures,
                )


def _check_paper_plot_manifest(
    manifest_path: Path,
    failures: list[str],
    *,
    cfg: Optional[dict] = None,
    stitched: Optional[dict[str, np.ndarray]] = None,
    application: Optional[dict[str, np.ndarray]] = None,
) -> None:
    manifest = _read_json(manifest_path, failures)
    if manifest is None:
        if not manifest_path.exists():
            failures.append(f"missing plot manifest {manifest_path}")
        return

    if manifest.get("schema") != PAPER_PLOT_SCHEMA:
        failures.append(f"plot manifest schema={manifest.get('schema')!r}, expected {PAPER_PLOT_SCHEMA!r}")
    if manifest.get("plot_count") != len(EXPECTED_PAPER_STORIES):
        failures.append(f"plot manifest plot_count={manifest.get('plot_count')!r}, expected {len(EXPECTED_PAPER_STORIES)}")
    if isinstance(cfg, dict):
        cfg_params = cfg.get("params", {})
        if isinstance(cfg_params, dict):
            cost_profile = str(cfg_params.get("pascucci_cost_profile", "exp")).strip().lower()
            cost_offset = cfg_params.get("pascucci_cost_offset", 0.0)
            if cost_profile != "exp":
                failures.append(
                    "paper plot parity requires pascucci_cost_profile='exp' "
                    f"to match raw/to_ema final_model3.py, got {cost_profile!r}"
                )
            if not _is_finite_scalar(cost_offset) or abs(float(cost_offset)) > 1.0e-8:
                failures.append(
                    "paper plot parity requires pascucci_cost_offset=0.0 "
                    f"with exp profile, got {cost_offset!r}"
                )
            calibration = cfg_params.get("pascucci_calibration", {})
            if not isinstance(calibration, dict):
                failures.append("paper plot parity requires Pascucci calibration metadata for plotmaker dataset checks")
            else:
                if calibration.get("K") != PLOTMAKER_CALIBRATION_K:
                    failures.append(
                        "paper plot parity requires plotmaker calibration K="
                        f"{PLOTMAKER_CALIBRATION_K}, got {calibration.get('K')!r}"
                    )
                calibration_dt = calibration.get("dt")
                if not _is_finite_scalar(calibration_dt) or abs(float(calibration_dt) - PLOTMAKER_CALIBRATION_DT) > 1.0e-12:
                    failures.append(
                        "paper plot parity requires plotmaker calibration dt="
                        f"{PLOTMAKER_CALIBRATION_DT}, got {calibration_dt!r}"
                    )
                for label, expected_basename, expected_sha256 in (
                    ("H_metadata", PLOTMAKER_H_BASENAME, PLOTMAKER_H_SHA256),
                    ("S_metadata", PLOTMAKER_S_BASENAME, PLOTMAKER_S_SHA256),
                ):
                    metadata = calibration.get(label)
                    if not isinstance(metadata, dict):
                        failures.append(f"paper plot parity requires calibration {label}")
                        continue
                    basename = Path(str(metadata.get("source_path", ""))).name
                    source_sha256 = str(metadata.get("source_sha256", ""))
                    if basename != expected_basename:
                        failures.append(
                            f"paper plot parity requires {label}.source_path basename "
                            f"{expected_basename!r}, got {basename!r}"
                        )
                    if source_sha256 != expected_sha256:
                        failures.append(
                            f"paper plot parity requires {label}.source_sha256 "
                            f"{expected_sha256}, got {source_sha256}"
                        )

    plots = manifest.get("plots")
    if not isinstance(plots, dict):
        failures.append("plot manifest plots must be an object keyed by story id")
        plots = {}
    for story, filename in zip(EXPECTED_PAPER_STORIES, EXPECTED_PAPER_PLOTS):
        entry = plots.get(story)
        if not isinstance(entry, dict):
            failures.append(f"plot manifest missing story {story}")
            continue
        if entry.get("filename") != filename:
            failures.append(f"plot manifest {story}.filename={entry.get('filename')!r}, expected {filename!r}")
        path_value = entry.get("path", filename)
        if not isinstance(path_value, str) or path_value.strip() == "":
            failures.append(f"plot manifest {story}.path missing")
            continue
        plot_path = Path(path_value)
        if plot_path.is_absolute():
            failures.append(f"plot manifest {story}.path must be relative to manifest dir")
        resolved_plot_path = manifest_path.parent / plot_path
        if not resolved_plot_path.exists():
            failures.append(f"missing plot {resolved_plot_path}")
        elif resolved_plot_path.stat().st_size <= 1000:
            failures.append(f"plot {resolved_plot_path} too small")
        else:
            dimensions = _png_dimensions(resolved_plot_path)
            expected_dimensions = EXPECTED_PAPER_PLOT_DIMENSIONS[story]
            if dimensions is None:
                failures.append(f"plot {resolved_plot_path} must be a readable PNG")
            elif dimensions != expected_dimensions:
                failures.append(
                    f"plot {resolved_plot_path} dimensions={dimensions}, expected {expected_dimensions}"
                )

    for filename in EXPECTED_PAPER_PLOTS:
        plot_path = manifest_path.parent / filename
        if not plot_path.exists():
            failures.append(f"missing expected paper plot file {plot_path}")

    native_plots = manifest.get("plotmaker_native_plots")
    if not isinstance(native_plots, dict):
        failures.append("plot manifest plotmaker_native_plots must be an object")
        native_plots = {}
    for key, filename in EXPECTED_PLOTMAKER_NATIVE_PLOTS.items():
        entry = native_plots.get(key)
        if not isinstance(entry, dict):
            failures.append(f"plot manifest plotmaker_native_plots.{key} must be an object")
            continue
        if entry.get("filename") != filename:
            failures.append(
                f"plot manifest plotmaker_native_plots.{key}.filename={entry.get('filename')!r}, "
                f"expected {filename!r}"
            )
        if entry.get("path_relative_to") != "manifest_dir":
            failures.append(f"plot manifest plotmaker_native_plots.{key}.path_relative_to must be manifest_dir")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or path_value.strip() == "":
            failures.append(f"plot manifest plotmaker_native_plots.{key}.path missing")
            continue
        if Path(path_value).is_absolute():
            failures.append(f"plot manifest plotmaker_native_plots.{key}.path must be relative")
            continue
        plot_path = manifest_path.parent / path_value
        if not plot_path.exists():
            failures.append(f"missing plotmaker native plot file {plot_path}")
        elif plot_path.stat().st_size <= 1000:
            failures.append(f"plotmaker native plot file {plot_path} too small")
        else:
            dimensions = _png_dimensions(plot_path)
            expected_dimensions = EXPECTED_PLOTMAKER_NATIVE_DIMENSIONS[key]
            if dimensions is None:
                failures.append(f"plotmaker native plot file {plot_path} must be a readable PNG")
            elif dimensions != expected_dimensions:
                failures.append(
                    f"plotmaker native plot file {plot_path} dimensions={dimensions}, "
                    f"expected {expected_dimensions}"
                )

    plot_data = manifest.get("plot_data")
    plot_data_arrays: Optional[dict[str, np.ndarray]] = None
    if not isinstance(plot_data, dict):
        failures.append("plot manifest plot_data must be an object")
    else:
        if plot_data.get("schema") != PAPER_PLOT_DATA_SCHEMA:
            failures.append(
                f"plot data schema={plot_data.get('schema')!r}, expected {PAPER_PLOT_DATA_SCHEMA!r}"
            )
        plot_data_path_value = plot_data.get("path", PAPER_PLOT_DATA_NPZ)
        if not isinstance(plot_data_path_value, str) or plot_data_path_value.strip() == "":
            failures.append("plot manifest plot_data.path missing")
        else:
            plot_data_path = Path(plot_data_path_value)
            if plot_data_path.is_absolute():
                failures.append("plot manifest plot_data.path must be relative to manifest dir")
            else:
                plot_data_arrays = _load_npz_arrays(manifest_path.parent / plot_data_path, failures)
        declared_keys = plot_data.get("keys")
        if declared_keys is not None and not isinstance(declared_keys, list):
            failures.append("plot manifest plot_data.keys must be a list when present")
    if plot_data_arrays is not None:
        required_plot_data_keys = (
            "plot35_time_sim",
            "plot35_sim_q05",
            "plot35_sim_q95",
            "plot35_time_real",
            "plot35_real",
            "plot36_time_sim",
            "plot36_sim_q05",
            "plot36_sim_q95",
            "plot36_time_real",
            "plot36_real",
            "plot37_controlled_time",
            "plot37_controlled_mean",
            "plot37_controlled_q10",
            "plot37_controlled_q90",
            "plot37_uncontrolled_time",
            "plot37_uncontrolled_mean",
            "plot37_uncontrolled_q10",
            "plot37_uncontrolled_q90",
            "plot38_time",
            "plot38_controlled_mean",
            "plot38_controlled_q05",
            "plot38_controlled_q50",
            "plot38_controlled_q95",
            "plot38_uncontrolled_mean",
            "plot39_time",
            "plot39_sample_indices",
            "plot39_S_mean",
            "plot39_S_q10",
            "plot39_S_q50",
            "plot39_S_q90",
            "plot39_S_samples",
            "plot39_H_mean",
            "plot39_H_q10",
            "plot39_H_q50",
            "plot39_H_q90",
            "plot39_H_samples",
            "plot39_V_mean",
            "plot39_V_q10",
            "plot39_V_q50",
            "plot39_V_q90",
            "plot39_V_samples",
            "plot39_X_mean",
            "plot39_X_q10",
            "plot39_X_q50",
            "plot39_X_q90",
            "plot39_X_samples",
            "plot40_controlled_mean",
            "plot40_controlled_q05",
            "plot40_controlled_q50",
            "plot40_controlled_q95",
            "plot40_uncontrolled_mean",
            "plot40_uncontrolled_q05",
            "plot40_uncontrolled_q50",
            "plot40_uncontrolled_q95",
            "plotmaker_Y_time",
            "plotmaker_Y_sample_indices",
            "plotmaker_Y_mean",
            "plotmaker_Y_q10",
            "plotmaker_Y_q50",
            "plotmaker_Y_q90",
            "plotmaker_Y_samples",
            "plotmaker_Z_time",
            "plotmaker_Z_sample_indices",
            "plotmaker_Z_Z_S_mean",
            "plotmaker_Z_Z_S_q10",
            "plotmaker_Z_Z_S_q50",
            "plotmaker_Z_Z_S_q90",
            "plotmaker_Z_Z_S_samples",
            "plotmaker_Z_Z_H_mean",
            "plotmaker_Z_Z_H_q10",
            "plotmaker_Z_Z_H_q50",
            "plotmaker_Z_Z_H_q90",
            "plotmaker_Z_Z_H_samples",
            "plotmaker_Z_Z_V_mean",
            "plotmaker_Z_Z_V_q10",
            "plotmaker_Z_Z_V_q50",
            "plotmaker_Z_Z_V_q90",
            "plotmaker_Z_Z_V_samples",
            "plotmaker_Z_Z_X_mean",
            "plotmaker_Z_Z_X_q10",
            "plotmaker_Z_Z_X_q50",
            "plotmaker_Z_Z_X_q90",
            "plotmaker_Z_Z_X_samples",
        )
        for key in required_plot_data_keys:
            if key not in plot_data_arrays:
                failures.append(f"plot data NPZ missing key {key}")
        for prefix in ("plot35", "plot36"):
            time_sim = plot_data_arrays.get(f"{prefix}_time_sim")
            q05 = plot_data_arrays.get(f"{prefix}_sim_q05")
            q95 = plot_data_arrays.get(f"{prefix}_sim_q95")
            time_real = plot_data_arrays.get(f"{prefix}_time_real")
            real = plot_data_arrays.get(f"{prefix}_real")
            if time_sim is not None and q05 is not None and q05.shape != time_sim.shape:
                failures.append(f"plot data {prefix}_sim_q05 shape does not match {prefix}_time_sim")
            if time_sim is not None and q95 is not None and q95.shape != time_sim.shape:
                failures.append(f"plot data {prefix}_sim_q95 shape does not match {prefix}_time_sim")
            if time_real is not None and real is not None and real.shape != time_real.shape:
                failures.append(f"plot data {prefix}_real shape does not match {prefix}_time_real")
        for prefix in ("plot37_controlled", "plot37_uncontrolled"):
            time_arr = plot_data_arrays.get(f"{prefix}_time")
            for stat in ("mean", "q10", "q90"):
                values = plot_data_arrays.get(f"{prefix}_{stat}")
                if time_arr is not None and values is not None and values.shape != time_arr.shape:
                    failures.append(f"plot data {prefix}_{stat} shape does not match {prefix}_time")
        _check_paper_plot_data_consistency(
            plot_data_arrays=plot_data_arrays,
            cfg=cfg,
            stitched=stitched,
            application=application,
            failures=failures,
        )

    ou_reference = manifest.get("ou_reference")
    if not isinstance(ou_reference, dict):
        failures.append("plot manifest ou_reference must be an object")
    elif ou_reference.get("source") != "calibration_metadata":
        failures.append(
            "plot manifest ou_reference.source must be calibration_metadata "
            f"for paper parity, got {ou_reference.get('source')!r}"
        )
    else:
        simulation = ou_reference.get("simulation")
        if not isinstance(simulation, dict):
            failures.append("plot manifest ou_reference.simulation missing")
        else:
            if simulation.get("n_sim") != 10000:
                failures.append(f"plot manifest ou_reference.simulation.n_sim={simulation.get('n_sim')!r}, expected 10000")
            dt_sim = simulation.get("dt_sim")
            if not _is_finite_scalar(dt_sim) or float(dt_sim) != 0.5:
                failures.append(f"plot manifest ou_reference.simulation.dt_sim={simulation.get('dt_sim')!r}, expected 0.5")
            if simulation.get("seed") != 42:
                failures.append(f"plot manifest ou_reference.simulation.seed={simulation.get('seed')!r}, expected 42")

    if manifest.get("cost_trace_source") != "cost_J_trajectory":
        failures.append(
            "plot manifest cost_trace_source must be cost_J_trajectory "
            f"for paper parity, got {manifest.get('cost_trace_source')!r}"
        )
    if manifest.get("uncontrolled_cost_trace_source") != "to_ema_raw_uncontrolled_J":
        failures.append(
            "plot manifest uncontrolled_cost_trace_source must be to_ema_raw_uncontrolled_J "
            f"for paper parity, got {manifest.get('uncontrolled_cost_trace_source')!r}"
        )
    to_ema_uncontrolled = manifest.get("to_ema_uncontrolled_reference")
    if not isinstance(to_ema_uncontrolled, dict):
        failures.append("plot manifest to_ema_uncontrolled_reference must be an object")
    elif to_ema_uncontrolled.get("source") != "calibration_metadata":
        failures.append(
            "plot manifest to_ema_uncontrolled_reference.source must be calibration_metadata "
            f"for paper parity, got {to_ema_uncontrolled.get('source')!r}"
        )
    else:
        simulation = to_ema_uncontrolled.get("simulation")
        if not isinstance(simulation, dict):
            failures.append("plot manifest to_ema_uncontrolled_reference.simulation missing")
        else:
            if simulation.get("n_sim") != 10000:
                failures.append(
                    "plot manifest to_ema_uncontrolled_reference.simulation.n_sim="
                    f"{simulation.get('n_sim')!r}, expected 10000"
                )
            dt = simulation.get("dt")
            if not _is_finite_scalar(dt) or abs(float(dt) - 0.1) > 1.0e-12:
                failures.append(
                    "plot manifest to_ema_uncontrolled_reference.simulation.dt="
                    f"{simulation.get('dt')!r}, expected 0.1"
                )
            if simulation.get("seed") != 42:
                failures.append(
                    "plot manifest to_ema_uncontrolled_reference.simulation.seed="
                    f"{simulation.get('seed')!r}, expected 42"
                )
    transforms = manifest.get("state_transforms")
    if not isinstance(transforms, dict):
        failures.append("plot manifest state_transforms missing")
    else:
        if "exp(S) * 1000" not in str(transforms.get("#35", "")):
            failures.append("plot manifest state_transforms.#35 must record exp(S) * 1000 price scale")
        if "exp(S)" not in str(transforms.get("#39", "")):
            failures.append("plot manifest state_transforms.#39 must record exp(S) state transform")

    plot37_inputs = "\n".join(str(item) for item in (plots.get("#37", {}) or {}).get("inputs", []))
    if "controlled_cost_J_trajectory" not in plot37_inputs or "to_ema.raw_uncontrolled_J_paths" not in plot37_inputs:
        failures.append("plot manifest #37 inputs must use controlled cost_J_trajectory and to_ema raw uncontrolled J")
    comparison_sources = manifest.get("comparison_sources")
    if not isinstance(comparison_sources, dict):
        failures.append("plot manifest comparison_sources must be an object")
    else:
        plot37_sources = comparison_sources.get("#37")
        if not isinstance(plot37_sources, dict):
            failures.append("plot manifest comparison_sources.#37 must be an object")
        elif plot37_sources.get("uncontrolled") != "to_ema_raw_uncontrolled_J":
            failures.append(
                "plot manifest comparison_sources.#37.uncontrolled must be to_ema_raw_uncontrolled_J, "
                f"got {plot37_sources.get('uncontrolled')!r}"
            )
        plot40_sources = comparison_sources.get("#40")
        if not isinstance(plot40_sources, dict):
            failures.append("plot manifest comparison_sources.#40 must be an object")
        elif plot40_sources.get("uncontrolled") != "to_ema_raw_uncontrolled_J_final":
            failures.append(
                "plot manifest comparison_sources.#40.uncontrolled must be to_ema_raw_uncontrolled_J_final, "
                f"got {plot40_sources.get('uncontrolled')!r}"
            )
        elif plot40_sources.get("paired_alpha_zero_detail") != "application_metrics.uncontrolled_cost_J_total":
            failures.append("plot manifest comparison_sources.#40 must retain paired alpha_zero application metric detail")
    plot40_inputs = "\n".join(str(item) for item in (plots.get("#40", {}) or {}).get("inputs", []))
    if "to_ema.raw_uncontrolled_J_paths[:, -1]" not in plot40_inputs:
        failures.append("plot manifest #40 inputs must use final raw to_ema uncontrolled J paths")
    if "application_metrics.uncontrolled_cost_J_total" not in plot40_inputs:
        failures.append("plot manifest #40 inputs must retain paired alpha_zero total as application metric detail")
    plot35_inputs = "\n".join(str(item) for item in (plots.get("#35", {}) or {}).get("inputs", []))
    if "prepare_S" not in plot35_inputs or "real_S * 1000" not in plot35_inputs:
        failures.append("plot manifest #35 inputs must use to_ema-style prepared S and real_S * 1000")
    plot36_inputs = "\n".join(str(item) for item in (plots.get("#36", {}) or {}).get("inputs", []))
    if "prepare_H" not in plot36_inputs or "real_H" not in plot36_inputs:
        failures.append("plot manifest #36 inputs must use to_ema-style prepared H and real_H")
    plot39_inputs = "\n".join(str(item) for item in (plots.get("#39", {}) or {}).get("inputs", []))
    if "H,V,X" not in plot39_inputs or "q10-q90" not in plot39_inputs or "sample paths" not in plot39_inputs:
        failures.append("plot manifest #39 inputs must use plotmaker-style forward components S,H,V,X")
    plotmaker_reference = manifest.get("plotmaker_reference")
    if not isinstance(plotmaker_reference, dict):
        failures.append("plot manifest plotmaker_reference must be an object")
    else:
        if plotmaker_reference.get("data") != PLOTMAKER_DATASET_NAME:
            failures.append(
                f"plot manifest plotmaker_reference.data={plotmaker_reference.get('data')!r}, "
                f"expected {PLOTMAKER_DATASET_NAME!r}"
            )
        if plotmaker_reference.get("dataset_status") != "matched":
            failures.append(
                "plot manifest plotmaker_reference.dataset_status must be 'matched', "
                f"got {plotmaker_reference.get('dataset_status')!r}"
            )
        for key, expected_basename, expected_sha256 in (
            ("H", PLOTMAKER_H_BASENAME, PLOTMAKER_H_SHA256),
            ("S", PLOTMAKER_S_BASENAME, PLOTMAKER_S_SHA256),
        ):
            payload = plotmaker_reference.get(key)
            if not isinstance(payload, dict):
                failures.append(f"plot manifest plotmaker_reference.{key} must be an object")
                continue
            if payload.get("expected_basename") != expected_basename or payload.get("actual_basename") != expected_basename:
                failures.append(
                    f"plot manifest plotmaker_reference.{key} must use basename "
                    f"{expected_basename!r}, got expected={payload.get('expected_basename')!r}, "
                    f"actual={payload.get('actual_basename')!r}"
                )
            if payload.get("expected_sha256") != expected_sha256 or payload.get("actual_sha256") != expected_sha256:
                failures.append(
                    f"plot manifest plotmaker_reference.{key} must use source_sha256 "
                    f"{expected_sha256}, got expected={payload.get('expected_sha256')}, "
                    f"actual={payload.get('actual_sha256')}"
                )
    notebook_parity = manifest.get("notebook_parity")
    if not isinstance(notebook_parity, dict):
        failures.append("plot manifest notebook_parity must be an object")
        notebook_stories = {}
    else:
        if notebook_parity.get("exact_all_stories") is not False:
            failures.append("plot manifest notebook_parity.exact_all_stories must be false until #38/#40 have direct plotmaker or paper figure oracles")
        notebook_stories = notebook_parity.get("stories")
        if not isinstance(notebook_stories, dict):
            failures.append("plot manifest notebook_parity.stories must be an object")
            notebook_stories = {}
    expected_notebook_status = {
        "#35": "to_ema_calibration_reference",
        "#36": "to_ema_calibration_reference",
        "#37": "plotmaker_per_time_cost_formula_oracled",
        "#38": "diagnostic_only_no_plotmaker_equivalent",
        "#39": "plotmaker_forward_components_reference",
        "#40": "diagnostic_only_no_plotmaker_equivalent",
    }
    for story, expected_status in expected_notebook_status.items():
        actual_status = notebook_stories.get(story) if isinstance(notebook_stories, dict) else None
        if actual_status != expected_status:
            failures.append(
                f"plot manifest notebook_parity.stories.{story}={actual_status!r}, "
                f"expected {expected_status!r}"
            )
    native_notebook_stories = notebook_parity.get("native_plotmaker_plots") if isinstance(notebook_parity, dict) else None
    if not isinstance(native_notebook_stories, dict):
        failures.append("plot manifest notebook_parity.native_plotmaker_plots must be an object")
        native_notebook_stories = {}
    expected_native_notebook_status = {
        "Y": "plotmaker_backward_component_reference",
        "Z": "plotmaker_z_components_reference",
    }
    for story, expected_status in expected_native_notebook_status.items():
        actual_status = native_notebook_stories.get(story)
        if actual_status != expected_status:
            failures.append(
                f"plot manifest notebook_parity.native_plotmaker_plots.{story}={actual_status!r}, "
                f"expected {expected_status!r}"
            )

    visual_regression = manifest.get("visual_regression")
    if not isinstance(visual_regression, dict):
        failures.append("plot manifest visual_regression must be an object")
        visual_regression = {}
    else:
        visual_status = str(visual_regression.get("status", ""))
        if visual_regression.get("structural_style_contract") is not True:
            failures.append("plot manifest visual_regression.structural_style_contract must be true")
        if visual_status == "structural_style_only_no_golden_images":
            if visual_regression.get("pixel_exact_claim") is not False:
                failures.append("plot manifest visual_regression.pixel_exact_claim must be false without golden images")
            if visual_regression.get("golden_images_available") is not False:
                failures.append("plot manifest visual_regression.golden_images_available must be false until golden images are saved")
            comparison_method = str(visual_regression.get("comparison_method", ""))
            if "numeric_plot_data_recomputation" not in comparison_method:
                failures.append("plot manifest visual_regression.comparison_method must mention numeric_plot_data_recomputation")
            remaining_gaps = visual_regression.get("remaining_gaps")
            if not isinstance(remaining_gaps, list):
                failures.append("plot manifest visual_regression.remaining_gaps must be a list")
            else:
                joined_gaps = "\n".join(str(item) for item in remaining_gaps)
                if "golden image" not in joined_gaps:
                    failures.append("plot manifest visual_regression.remaining_gaps must record missing golden images")
                if "np.random.choice" not in joined_gaps:
                    failures.append("plot manifest visual_regression.remaining_gaps must record plotmaker sample-path randomness")
        elif visual_status == "pixel_hash_exact":
            if visual_regression.get("pixel_exact_claim") is not True:
                failures.append("plot manifest visual_regression.pixel_exact_claim must be true for pixel_hash_exact")
            if visual_regression.get("golden_images_available") is not True:
                failures.append("plot manifest visual_regression.golden_images_available must be true for pixel_hash_exact")
            comparison_method = str(visual_regression.get("comparison_method", ""))
            if "sha256" not in comparison_method:
                failures.append("plot manifest visual_regression.comparison_method must mention sha256 for pixel_hash_exact")
            golden_images = visual_regression.get("golden_images")
            if not isinstance(golden_images, dict):
                failures.append("plot manifest visual_regression.golden_images must be an object for pixel_hash_exact")
                golden_images = {}
            expected_visual_entries: Dict[str, str] = {}
            for story in EXPECTED_PAPER_STORIES:
                entry = plots.get(story)
                if isinstance(entry, dict):
                    expected_visual_entries[story] = str(entry.get("path", ""))
            for key in EXPECTED_PLOTMAKER_NATIVE_PLOTS:
                entry = native_plots.get(key)
                if isinstance(entry, dict):
                    expected_visual_entries[key] = str(entry.get("path", ""))
            for key, expected_current_path in expected_visual_entries.items():
                payload = golden_images.get(key)
                if not isinstance(payload, dict):
                    failures.append(f"plot manifest visual_regression.golden_images.{key} must be an object")
                    continue
                current_path_value = str(payload.get("current_path", ""))
                golden_path_value = str(payload.get("golden_path", ""))
                expected_sha256 = str(payload.get("sha256", "")).lower()
                if current_path_value != expected_current_path:
                    failures.append(
                        "plot manifest visual_regression.golden_images."
                        f"{key}.current_path={current_path_value!r}, expected {expected_current_path!r}"
                    )
                if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
                    failures.append(f"plot manifest visual_regression.golden_images.{key}.sha256 must be a lowercase SHA-256")
                    continue
                current_rel = Path(current_path_value)
                golden_rel = Path(golden_path_value)
                if current_rel.is_absolute():
                    failures.append(f"plot manifest visual_regression.golden_images.{key}.current_path must be relative")
                    continue
                if golden_rel.is_absolute():
                    failures.append(f"plot manifest visual_regression.golden_images.{key}.golden_path must be relative")
                    continue
                current_path = manifest_path.parent / current_rel
                golden_path = manifest_path.parent / golden_rel
                if not current_path.exists():
                    failures.append(f"missing current visual regression image {current_path}")
                    continue
                if not golden_path.exists():
                    failures.append(f"missing golden visual regression image {golden_path}")
                    continue
                current_sha256 = _sha256_file(current_path)
                golden_sha256 = _sha256_file(golden_path)
                if current_sha256 != expected_sha256:
                    failures.append(
                        f"visual_regression.golden_images.{key} current sha256={current_sha256}, "
                        f"expected {expected_sha256}"
                    )
                if golden_sha256 != expected_sha256:
                    failures.append(
                        f"visual_regression.golden_images.{key} golden sha256={golden_sha256}, "
                        f"expected {expected_sha256}"
                    )
        else:
            failures.append(
                "plot manifest visual_regression.status must be "
                "'structural_style_only_no_golden_images' or 'pixel_hash_exact'"
            )

    def _style_for(story: str) -> dict:
        entry = plots.get(story)
        style = entry.get("style") if isinstance(entry, dict) else None
        if not isinstance(style, dict):
            failures.append(f"plot manifest {story}.style missing")
            return {}
        return style

    def _expect_style_value(story: str, style: dict, key: str, expected: Any) -> None:
        actual = style.get(key)
        if actual != expected:
            failures.append(f"plot manifest {story}.style.{key}={actual!r}, expected {expected!r}")

    def _expect_style_float(story: str, style: dict, key: str, expected: float, tol: float = 1.0e-12) -> None:
        actual = style.get(key)
        if not _is_finite_scalar(actual) or abs(float(actual) - float(expected)) > tol:
            failures.append(f"plot manifest {story}.style.{key}={actual!r}, expected {expected!r}")

    def _expect_style_sequence(story: str, style: dict, key: str, expected: tuple[float, ...]) -> None:
        actual = style.get(key)
        if not isinstance(actual, list) or len(actual) != len(expected):
            failures.append(f"plot manifest {story}.style.{key}={actual!r}, expected {list(expected)!r}")
            return
        for got, want in zip(actual, expected):
            if not _is_finite_scalar(got) or abs(float(got) - float(want)) > 1.0e-12:
                failures.append(f"plot manifest {story}.style.{key}={actual!r}, expected {list(expected)!r}")
                return

    style35 = _style_for("#35")
    if style35:
        if "variable_mu_calibration.ipynb::generate_plot" not in str(style35.get("source", "")):
            failures.append("plot manifest #35.style.source must reference raw/to_ema variable_mu_calibration generate_plot")
        if "calibration.py" not in str(style35.get("formula_source", "")):
            failures.append("plot manifest #35.style.formula_source must reference raw/to_ema calibration.py")
        _expect_style_sequence("#35", style35, "figure_size", (12.0, 6.0))
        _expect_style_value("#35", style35, "real_color", "red")
        _expect_style_float("#35", style35, "band_alpha", 0.3)
        _expect_style_value("#35", style35, "band_label", "80% Band")
        _expect_style_sequence("#35", style35, "quantiles", (0.05, 0.95))
        _expect_style_value("#35", style35, "xlabel", "Hours")

    style36 = _style_for("#36")
    if style36:
        if "variable_mu_calibration.ipynb::generate_plot" not in str(style36.get("source", "")):
            failures.append("plot manifest #36.style.source must reference raw/to_ema variable_mu_calibration generate_plot")
        if "calibration.py" not in str(style36.get("formula_source", "")):
            failures.append("plot manifest #36.style.formula_source must reference raw/to_ema calibration.py")
        _expect_style_sequence("#36", style36, "figure_size", (12.0, 6.0))
        _expect_style_value("#36", style36, "real_color", "red")
        _expect_style_float("#36", style36, "band_alpha", 0.3)
        _expect_style_value("#36", style36, "band_label", "80% Band")
        _expect_style_sequence("#36", style36, "quantiles", (0.05, 0.95))
        _expect_style_value("#36", style36, "xlabel", "Hours")

    style37 = _style_for("#37")
    if style37:
        if "plotmaker.ipynb::comparison J cell" not in str(style37.get("source", "")):
            failures.append("plot manifest #37.style.source must reference raw/to_ema plotmaker comparison J cell")
        _expect_style_sequence("#37", style37, "figure_size", (10.0, 6.0))
        _expect_style_value("#37", style37, "controlled_color", "r")
        _expect_style_value("#37", style37, "uncontrolled_color", "b")
        _expect_style_float("#37", style37, "band_alpha", 0.2)
        _expect_style_sequence("#37", style37, "quantiles", (0.10, 0.90))
        if "J_t=" not in str(style37.get("title", "")) or "g(X_t)" not in str(style37.get("title", "")):
            failures.append("plot manifest #37.style.title must match plotmaker J_t comparison title")
        _expect_style_value("#37", style37, "xlabel", "Time(h)")
        _expect_style_value("#37", style37, "ylabel", "Cost")

    style38 = _style_for("#38")
    if style38:
        _expect_style_value("#38", style38, "source", "final_recursive.application_metrics_alpha")
        _expect_style_sequence("#38", style38, "figure_size", (10.0, 5.0))
        _expect_style_value("#38", style38, "controlled_color", "tab:green")
        _expect_style_value("#38", style38, "uncontrolled_color", "tab:gray")
        _expect_style_value("#38", style38, "uncontrolled_linestyle", "--")
        _expect_style_value("#38", style38, "zero_line", True)
        _expect_style_float("#38", style38, "band_alpha", 0.20)
        _expect_style_sequence("#38", style38, "quantiles", (0.05, 0.50, 0.95))
        _expect_style_value("#38", style38, "xlabel", "Time")
        _expect_style_value("#38", style38, "ylabel", "alpha")

    style39 = _style_for("#39")
    if style39:
        if "plotmaker.ipynb::forward_components" not in str(style39.get("source", "")):
            failures.append("plot manifest #39.style.source must reference raw/to_ema plotmaker forward components")
        _expect_style_sequence("#39", style39, "figure_size", (14.0, 10.0))
        _expect_style_value("#39", style39, "layout", "2x2")
        _expect_style_float("#39", style39, "band_alpha", 0.2)
        _expect_style_sequence("#39", style39, "quantiles", (0.10, 0.90))
        _expect_style_value("#39", style39, "sample_paths", 3)
        _expect_style_value("#39", style39, "sample_policy", "deterministic_first3_for_reproducibility")

    style40 = _style_for("#40")
    if style40:
        _expect_style_sequence("#40", style40, "figure_size", (8.0, 5.0))
        _expect_style_value("#40", style40, "bar_stat", "q50")
        _expect_style_value("#40", style40, "marker_stat", "mean")
        _expect_style_value("#40", style40, "marker", "D")
        _expect_style_value("#40", style40, "interval", "q05-q95")
        _expect_style_value("#40", style40, "ylabel", "J total median with q05-q95; diamonds show mean")

    def _native_style_for(story: str) -> dict:
        entry = native_plots.get(story)
        style = entry.get("style") if isinstance(entry, dict) else None
        if not isinstance(style, dict):
            failures.append(f"plot manifest plotmaker_native_plots.{story}.style missing")
            return {}
        return style

    native_y_style = _native_style_for("Y")
    if native_y_style:
        if "plotmaker.ipynb::Backward Component Y cell" not in str(native_y_style.get("source", "")):
            failures.append("plot manifest native Y style source must reference raw/to_ema plotmaker Y cell")
        _expect_style_sequence("plotmaker_native_plots.Y", native_y_style, "figure_size", (10.0, 6.0))
        _expect_style_float("plotmaker_native_plots.Y", native_y_style, "band_alpha", 0.2)
        _expect_style_sequence("plotmaker_native_plots.Y", native_y_style, "quantiles", (0.10, 0.90))
        _expect_style_value("plotmaker_native_plots.Y", native_y_style, "sample_paths", 3)
        _expect_style_value(
            "plotmaker_native_plots.Y",
            native_y_style,
            "sample_policy",
            "deterministic_first3_for_reproducibility",
        )
        _expect_style_value("plotmaker_native_plots.Y", native_y_style, "title", "Backward Component Y")
        _expect_style_value("plotmaker_native_plots.Y", native_y_style, "xlabel", "t")
        _expect_style_value("plotmaker_native_plots.Y", native_y_style, "ylabel", "Y_t")

    native_z_style = _native_style_for("Z")
    if native_z_style:
        if "plotmaker.ipynb::Z components cell" not in str(native_z_style.get("source", "")):
            failures.append("plot manifest native Z style source must reference raw/to_ema plotmaker Z cell")
        _expect_style_sequence("plotmaker_native_plots.Z", native_z_style, "figure_size", (14.0, 10.0))
        _expect_style_value("plotmaker_native_plots.Z", native_z_style, "layout", "2x2")
        _expect_style_value("plotmaker_native_plots.Z", native_z_style, "labels", ["Z_S", "Z_H", "Z_V", "Z_X"])
        _expect_style_float("plotmaker_native_plots.Z", native_z_style, "band_alpha", 0.2)
        _expect_style_sequence("plotmaker_native_plots.Z", native_z_style, "quantiles", (0.10, 0.90))
        _expect_style_value("plotmaker_native_plots.Z", native_z_style, "sample_paths", 3)
        _expect_style_value(
            "plotmaker_native_plots.Z",
            native_z_style,
            "sample_policy",
            "deterministic_first3_for_reproducibility",
        )


def validate_pascucci_paper_plot_parity(path: str | Path) -> Dict[str, Any]:
    """Validate that saved Pascucci artifacts are comparable to the paper plots."""

    root_input = Path(path).expanduser()
    failures: list[str] = []
    incomplete: list[str] = []
    warnings: list[str] = []
    inconclusive: list[str] = []

    run_root = _resolve_run_root(root_input)
    if run_root is None:
        failures.append(f"no run_config.json found under {root_input}")
        return _paper_parity_report(
            status="FAILED",
            run_root=None,
            failures=failures,
            incomplete=incomplete,
            inconclusive=inconclusive,
            warnings=warnings,
        )

    rec_dir = run_root / "recursive"
    cfg_path = run_root / "run_config.json"
    cfg = _read_json(cfg_path, failures)
    if cfg is None:
        if not cfg_path.exists():
            failures.append(f"missing run_config.json at {run_root}")
    else:
        if cfg.get("model_name") != "pascucci":
            failures.append(f"run_config model_name={cfg.get('model_name')!r}, expected 'pascucci'")
        if cfg.get("application_metric_schema") != APPLICATION_METRIC_SCHEMA:
            failures.append(
                "run_config application_metric_schema must be "
                f"{APPLICATION_METRIC_SCHEMA}, got {cfg.get('application_metric_schema')!r}"
            )
        if cfg.get("pascucci_require_calibration") is not True:
            failures.append("run_config pascucci_require_calibration must be true for paper-parity")
        if cfg.get("pascucci_calibration_enabled") is not True:
            warnings.append("run_config pascucci_calibration_enabled is not true; relying on params.pascucci_calibration checks")
        if cfg.get("state_labels") != ["S", "H", "V", "X_state"]:
            failures.append("run_config state_labels mismatch")
        if cfg.get("z_labels") not in (None, ["Z_S", "Z_H", "Z_V", "Z_X"]):
            failures.append("run_config z_labels mismatch")

        params = cfg.get("params")
        if not isinstance(params, dict):
            failures.append("run_config params must be an object")
            params = {}
        _check_plotmaker_pascucci_scalar_params(params, failures)
        calibration = params.get("pascucci_calibration")
        if not isinstance(calibration, dict):
            failures.append("run_config params.pascucci_calibration missing")
        else:
            if calibration.get("schema") != PASCUCCI_OU_CALIBRATION_SCHEMA:
                failures.append(
                    "run_config params.pascucci_calibration.schema must be "
                    f"{PASCUCCI_OU_CALIBRATION_SCHEMA}, got {calibration.get('schema')!r}"
                )
            if calibration.get("log_price") is not True or calibration.get("S_transform") != "log":
                failures.append("run_config Pascucci calibration must use log-price S_transform='log'")
            if calibration.get("H_transform") != "linear":
                failures.append("run_config Pascucci calibration must use H_transform='linear'")
            if not _is_finite_scalar(calibration.get("dt")) or float(calibration.get("dt")) <= 0.0:
                failures.append(f"run_config Pascucci calibration dt invalid: {calibration.get('dt')!r}")
            if not isinstance(calibration.get("K"), int) or calibration.get("K") < 0:
                failures.append(f"run_config Pascucci calibration K invalid: {calibration.get('K')!r}")
            for metadata_key in ("H_metadata", "S_metadata"):
                metadata = calibration.get(metadata_key)
                if not isinstance(metadata, dict):
                    failures.append(f"run_config params.pascucci_calibration.{metadata_key} missing")
                else:
                    _calibration_metadata_source(metadata, metadata_key, failures, warnings)
            _check_pascucci_calibration_recomputes(params, calibration, failures)

        seed_manifest = cfg.get("seed_manifest")
        if not isinstance(seed_manifest, dict) or "eval_seed" not in seed_manifest:
            failures.append("run_config seed_manifest.eval_seed is required for paper-parity reproducibility")

    results_path = rec_dir / "results.json"
    results = _read_json(results_path, failures)
    _check_paper_results_contract(
        cfg=cfg,
        results=results,
        results_path=results_path,
        failures=failures,
    )

    stitched = _load_npz_arrays(rec_dir / "stitched_predictions_final.npz", failures)
    n_paths: Optional[int] = None
    n_time_points: Optional[int] = None
    if stitched is not None:
        for key in ("t", "X"):
            if key not in stitched:
                failures.append(f"stitched_predictions_final.npz missing key {key}")
        t = stitched.get("t")
        X = stitched.get("X")
        if t is not None and X is not None:
            if X.ndim != 3 or X.shape[2] < 4:
                failures.append(f"stitched_predictions_final.npz:X shape {X.shape}, expected (M, time, >=4)")
            else:
                n_paths = int(X.shape[0])
                n_time_points = int(X.shape[1])
            if t.ndim != 3 or t.shape[2] != 1:
                failures.append(f"stitched_predictions_final.npz:t shape {t.shape}, expected (M or 1, time, 1)")
            elif n_paths is not None and n_time_points is not None:
                if int(t.shape[0]) not in (1, n_paths) or int(t.shape[1]) != n_time_points:
                    failures.append(
                        "stitched_predictions_final.npz:t shape does not match X "
                        f"(t={t.shape}, X={X.shape})"
                    )

    application = _load_npz_arrays(rec_dir / "application_metrics_final.npz", failures)
    if application is not None:
        for key in PAPER_PARITY_APPLICATION_KEYS:
            if key not in application:
                failures.append(f"application_metrics_final.npz missing key {key}")
        controlled_traj = application.get("controlled_cost_J_trajectory")
        uncontrolled_traj = application.get("uncontrolled_cost_J_trajectory")
        if controlled_traj is not None and uncontrolled_traj is not None:
            if controlled_traj.ndim != 3 or controlled_traj.shape[2] != 1:
                failures.append(
                    "application_metrics_final.npz:controlled_cost_J_trajectory "
                    f"shape {controlled_traj.shape}, expected (M, time, 1)"
                )
            if uncontrolled_traj.shape != controlled_traj.shape:
                failures.append(
                    "application_metrics_final.npz uncontrolled_cost_J_trajectory shape "
                    f"{uncontrolled_traj.shape}, expected {controlled_traj.shape}"
                )
            if n_paths is not None and int(controlled_traj.shape[0]) != n_paths:
                failures.append(
                    "application_metrics_final.npz cost_J_trajectory path count "
                    f"{controlled_traj.shape[0]}, expected stitched path count {n_paths}"
                )
            if n_time_points is not None and int(controlled_traj.shape[1]) != n_time_points:
                failures.append(
                    "application_metrics_final.npz cost_J_trajectory time count "
                    f"{controlled_traj.shape[1]}, expected stitched time count {n_time_points}"
                )
        for key in ("controlled_cost_J_trajectory_math", "uncontrolled_cost_J_trajectory_math"):
            math_traj = application.get(key)
            if math_traj is not None:
                if math_traj.ndim != 3 or math_traj.shape[2] != 1:
                    failures.append(
                        f"application_metrics_final.npz:{key} shape {math_traj.shape}, expected (M, time, 1)"
                    )
                elif n_paths is not None and int(math_traj.shape[0]) != n_paths:
                    failures.append(
                        f"application_metrics_final.npz:{key} path count {math_traj.shape[0]}, "
                        f"expected stitched path count {n_paths}"
                    )
                elif n_time_points is not None and int(math_traj.shape[1]) != n_time_points:
                    failures.append(
                        f"application_metrics_final.npz:{key} time count {math_traj.shape[1]}, "
                        f"expected stitched time count {n_time_points}"
                    )
        _check_application_cost_accounting(application, failures)
        for key in ("controlled_alpha", "uncontrolled_alpha"):
            alpha = application.get(key)
            if alpha is not None and n_time_points is not None:
                if alpha.ndim != 3 or alpha.shape[2] != 1 or int(alpha.shape[1]) != n_time_points - 1:
                    failures.append(f"application_metrics_final.npz:{key} shape {alpha.shape}, expected (M, time-1, 1)")

    bundle = _load_npz_arrays(rec_dir / "evaluation_bundle.npz", failures)
    if bundle is not None:
        xi = bundle.get("Xi_initial")
        if xi is None:
            failures.append("evaluation_bundle.npz missing key Xi_initial")
        else:
            _check_paper_like_xi_initial(xi, failures)
            if n_paths is not None and xi.ndim == 2 and int(xi.shape[0]) != n_paths:
                failures.append(
                    f"evaluation_bundle Xi_initial path count {xi.shape[0]}, expected stitched path count {n_paths}"
                )

    _check_paper_plot_manifest(
        rec_dir / "plots" / "pascucci_paper" / PAPER_PLOT_MANIFEST,
        failures,
        cfg=cfg,
        stitched=stitched,
        application=application,
    )
    _check_science_gate(rec_dir, results, failures)
    _check_mc_confirmation(
        rec_dir=rec_dir,
        cfg=cfg,
        results=results,
        failures=failures,
    )

    status = _report_status(failures, incomplete, inconclusive)
    return _paper_parity_report(
        status=status,
        run_root=str(run_root),
        failures=failures,
        incomplete=incomplete,
        inconclusive=inconclusive,
        warnings=warnings,
    )


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

    parser = argparse.ArgumentParser(description="Validate Pascucci run artifacts")
    parser.add_argument("run_dir", type=str)
    parser.add_argument(
        "--paper-parity",
        action="store_true",
        help="Validate the Pascucci paper-plot parity contract instead of the T12/N13 gate.",
    )
    args = parser.parse_args(argv)
    if args.paper_parity:
        report = validate_pascucci_paper_plot_parity(args.run_dir)
    else:
        report = validate_t12_gate_n13(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
