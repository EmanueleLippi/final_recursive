from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _flatten_float(values: np.ndarray, *, label: str) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(flat).all():
        raise ValueError(f"{label} contains non-finite values")
    return flat


def _trace_stats(values: np.ndarray) -> Dict[str, float]:
    flat = _flatten_float(values, label="metric traces")
    return {
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "rate": float(np.mean(flat > 0.0)),
    }


def summarize_application_alpha(alpha: np.ndarray, baseline_mode: str) -> Dict[str, float | int | str]:
    flat = _flatten_float(alpha, label="alpha")
    abs_flat = np.abs(flat)
    return {
        "baseline_mode": str(baseline_mode),
        "sample_count": int(flat.size),
        "alpha_mean": float(np.mean(flat)),
        "alpha_std": float(np.std(flat)),
        "alpha_q05": float(np.quantile(flat, 0.05)),
        "alpha_q50": float(np.quantile(flat, 0.50)),
        "alpha_q95": float(np.quantile(flat, 0.95)),
        "alpha_abs_mean": float(np.mean(abs_flat)),
        "alpha_abs_q95": float(np.quantile(abs_flat, 0.95)),
    }


def _pathwise_metric(result: dict, metric_name: str) -> np.ndarray:
    try:
        value = result["pathwise"][metric_name]
    except KeyError as exc:
        raise KeyError(f"pathwise metric '{metric_name}' is required") from exc
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"pathwise metric '{metric_name}' must contain at least one value")
    if not np.isfinite(arr).all():
        raise ValueError(f"pathwise metric '{metric_name}' contains non-finite values")
    return arr


def summarize_controlled_uncontrolled_comparison(
    *,
    controlled: dict,
    uncontrolled: dict,
) -> Dict[str, float | int | bool | str]:
    comparison: Dict[str, float | int | bool | str] = {
        "paired_pathwise_samples": True,
        "controlled_baseline_mode": str(controlled.get("metadata", {}).get("baseline_mode", "")),
        "uncontrolled_baseline_mode": str(uncontrolled.get("metadata", {}).get("baseline_mode", "")),
        "controlled_paired_inputs": str(controlled.get("metadata", {}).get("paired_inputs", "")),
        "uncontrolled_paired_inputs": str(uncontrolled.get("metadata", {}).get("paired_inputs", "")),
    }
    comparison["same_input_source"] = (
        comparison["controlled_paired_inputs"] == comparison["uncontrolled_paired_inputs"]
    )
    sample_count = None
    for metric_name in ("cost_J_running", "cost_J_terminal", "cost_J_total"):
        controlled_values = _pathwise_metric(controlled, metric_name)
        uncontrolled_values = _pathwise_metric(uncontrolled, metric_name)
        if controlled_values.shape != uncontrolled_values.shape:
            raise ValueError(
                f"controlled/uncontrolled '{metric_name}' shapes must match, "
                f"got {controlled_values.shape} and {uncontrolled_values.shape}"
            )
        delta = controlled_values - uncontrolled_values
        flat = _flatten_float(delta, label=f"delta {metric_name}")
        abs_flat = np.abs(flat)
        if sample_count is None:
            sample_count = int(flat.size)
        elif sample_count != int(flat.size):
            raise ValueError("all cost_J comparison metrics must have the same sample count")
        comparison[f"delta_{metric_name}_mean"] = float(np.mean(flat))
        comparison[f"delta_{metric_name}_std"] = float(np.std(flat))
        comparison[f"delta_{metric_name}_q05"] = float(np.quantile(flat, 0.05))
        comparison[f"delta_{metric_name}_q50"] = float(np.quantile(flat, 0.50))
        comparison[f"delta_{metric_name}_q95"] = float(np.quantile(flat, 0.95))
        comparison[f"delta_{metric_name}_abs_q95"] = float(np.quantile(abs_flat, 0.95))
        comparison[f"{metric_name}_control_win_rate"] = float(np.mean(flat < 0.0))
    comparison["paired_sample_count"] = int(sample_count or 0)
    return comparison


def _stitch_boundary_jump_tensor(
    stitched: dict,
    blocks: list,
    *,
    signed: bool,
) -> np.ndarray | None:
    key = "stitch_X_boundary_signed_jump" if signed else "stitch_X_boundary_abs_jump"
    if key in stitched:
        return np.asarray(stitched[key], dtype=np.float32)

    X = np.asarray(stitched["X"], dtype=np.float32)
    t = np.asarray(stitched["t"], dtype=np.float32)[0, :, 0]
    jumps = []
    for block in blocks[:-1]:
        boundary = float(block["t_end"])
        matches = np.where(np.isclose(t, boundary, rtol=1.0e-6, atol=1.0e-6))[0]
        if len(matches) >= 2:
            jump = X[:, matches[1], :] - X[:, matches[0], :]
            jumps.append(jump if signed else np.abs(jump))
    if jumps:
        return np.stack(jumps, axis=0).astype(np.float32)
    if signed:
        return None
    if len(blocks) > 1:
        raise ValueError(
            "stitch_X_boundary_abs_jump is required when stitched outputs do not retain "
            "duplicate boundary times"
        )
    return np.asarray([0.0], dtype=np.float32)


def _stitch_boundary_jump_values(stitched: dict, blocks: list) -> np.ndarray:
    tensor = _stitch_boundary_jump_tensor(stitched, blocks, signed=False)
    if tensor is None:
        return np.asarray([0.0], dtype=np.float32)
    return np.asarray(tensor, dtype=np.float32).reshape(-1)


def _iter_state_labels(state_labels: Iterable[str] | None, D: int) -> tuple[str, ...]:
    if state_labels is None:
        return ()
    labels = tuple(str(label) for label in state_labels)
    if len(labels) != int(D):
        raise ValueError(f"state_labels must contain {D} labels, got {len(labels)}")
    return labels


def _physical_violation_values_from_x(stitched: dict, params: dict | None) -> Dict[str, np.ndarray]:
    if params is None or "X" not in stitched:
        return {}
    required = ("x_max", "v_min", "v_max")
    if any(key not in params for key in required):
        return {}

    X = np.asarray(stitched["X"], dtype=np.float32)
    if X.ndim != 3 or X.shape[2] < 4:
        return {}

    V = X[:, :, 2]
    Q = X[:, :, 3]
    x_max = np.float32(params["x_max"])
    v_min = np.float32(params["v_min"])
    v_max = np.float32(params["v_max"])
    return {
        "q_lower_violation": np.maximum(-Q, 0.0).astype(np.float32),
        "q_upper_violation": np.maximum(Q - x_max, 0.0).astype(np.float32),
        "v_lower_violation": np.maximum(v_min - V, 0.0).astype(np.float32),
        "v_upper_violation": np.maximum(V - v_max, 0.0).astype(np.float32),
    }


def summarize_pascucci_stitched_diagnostics(
    stitched: dict,
    blocks: list,
    params: dict | None = None,
    state_labels: Iterable[str] | None = None,
) -> Dict[str, float]:
    pathwise_violations = _physical_violation_values_from_x(stitched, params)
    summary: Dict[str, float] = {}
    for key in (
        "q_lower_violation",
        "q_upper_violation",
        "v_lower_violation",
        "v_upper_violation",
    ):
        if key in pathwise_violations:
            values = pathwise_violations[key]
        elif key in stitched:
            values = np.asarray(stitched[key], dtype=np.float32)
        else:
            raise KeyError(f"stitched diagnostics require '{key}' or pathwise X+params")
        stats = _trace_stats(values)
        for stat_name, value in stats.items():
            summary[f"{key}_{stat_name}"] = value

    jump_values = _stitch_boundary_jump_values(stitched, blocks)
    summary["stitch_X_boundary_max_abs_jump"] = float(np.max(jump_values))
    summary["stitch_X_boundary_mean_abs_jump"] = float(np.mean(jump_values))
    jump_tensor = _stitch_boundary_jump_tensor(stitched, blocks, signed=False)
    signed_jump_tensor = _stitch_boundary_jump_tensor(stitched, blocks, signed=True)
    if jump_tensor is not None and np.asarray(jump_tensor).ndim == 3:
        labels = _iter_state_labels(state_labels, np.asarray(jump_tensor).shape[2])
        for idx, label in enumerate(labels):
            component_abs = np.abs(np.asarray(jump_tensor, dtype=np.float32)[:, :, idx]).reshape(-1)
            summary[f"stitch_X_boundary_max_abs_jump_{label}"] = float(np.max(component_abs))
            summary[f"stitch_X_boundary_abs_q95_jump_{label}"] = float(np.quantile(component_abs, 0.95))
            if signed_jump_tensor is not None:
                component_signed = np.asarray(signed_jump_tensor, dtype=np.float32)[:, :, idx].reshape(-1)
                summary[f"stitch_X_boundary_signed_mean_jump_{label}"] = float(np.mean(component_signed))
    for key in ("Y", "Z"):
        jump_key = f"stitch_{key}_boundary_abs_jump"
        if jump_key in stitched:
            values = np.asarray(stitched[jump_key], dtype=np.float32).reshape(-1)
            summary[f"stitch_{key}_boundary_max_abs_jump"] = float(np.max(values))
            summary[f"stitch_{key}_boundary_mean_abs_jump"] = float(np.mean(values))
    return summary


def summarize_application_pass_stability(
    *,
    previous_stitched: dict,
    current_stitched: dict,
    previous_pathwise: dict,
    current_pathwise: dict,
    z_v_index: int = 2,
) -> Dict[str, float]:
    previous_t = np.asarray(previous_stitched["t"], dtype=np.float32)
    current_t = np.asarray(current_stitched["t"], dtype=np.float32)
    if previous_t.shape != current_t.shape or not np.allclose(previous_t, current_t, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("pass stability requires the same stitched time grid")

    previous_Y = np.asarray(previous_stitched["Y"], dtype=np.float32)
    current_Y = np.asarray(current_stitched["Y"], dtype=np.float32)
    previous_Z = np.asarray(previous_stitched["Z"], dtype=np.float32)
    current_Z = np.asarray(current_stitched["Z"], dtype=np.float32)
    if previous_Y.shape != current_Y.shape:
        raise ValueError("pass stability requires matching Y shapes")
    if previous_Z.shape != current_Z.shape:
        raise ValueError("pass stability requires matching Z shapes")
    if int(z_v_index) < 0 or int(z_v_index) >= previous_Z.shape[-1]:
        raise ValueError(f"z_v_index {z_v_index} is outside Z dimension {previous_Z.shape[-1]}")

    previous_cost = np.asarray(previous_pathwise["controlled_cost_J_total"], dtype=np.float32)
    current_cost = np.asarray(current_pathwise["controlled_cost_J_total"], dtype=np.float32)
    previous_alpha = np.asarray(previous_pathwise["controlled_alpha"], dtype=np.float32)
    current_alpha = np.asarray(current_pathwise["controlled_alpha"], dtype=np.float32)
    if previous_cost.shape != current_cost.shape:
        raise ValueError("pass stability requires matching controlled_cost_J_total shapes")
    if previous_alpha.shape != current_alpha.shape:
        raise ValueError("pass stability requires matching controlled_alpha shapes")

    return {
        "pass_vs_prev_Y_mae": float(np.mean(np.abs(current_Y - previous_Y))),
        "pass_vs_prev_Z_V_mae": float(
            np.mean(np.abs(current_Z[:, :, int(z_v_index)] - previous_Z[:, :, int(z_v_index)]))
        ),
        "pass_vs_prev_cost_J_total_mean_abs_delta": float(np.mean(np.abs(current_cost - previous_cost))),
        "pass_vs_prev_alpha_abs_mean_delta": float(np.mean(np.abs(current_alpha - previous_alpha))),
    }
