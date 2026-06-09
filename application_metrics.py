from __future__ import annotations

from typing import Dict

import numpy as np


def _trace_stats(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("metric traces must contain at least one value")
    return {
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "rate": float(np.mean(flat > 0.0)),
    }


def _stitch_boundary_jump_values(stitched: dict, blocks: list) -> np.ndarray:
    if "stitch_X_boundary_abs_jump" in stitched:
        return np.asarray(stitched["stitch_X_boundary_abs_jump"], dtype=np.float32).reshape(-1)

    X = np.asarray(stitched["X"], dtype=np.float32)
    t = np.asarray(stitched["t"], dtype=np.float32)[0, :, 0]
    jumps = []
    for block in blocks[:-1]:
        boundary = float(block["t_end"])
        matches = np.where(np.isclose(t, boundary, rtol=1.0e-6, atol=1.0e-6))[0]
        if len(matches) >= 2:
            jumps.append(np.abs(X[:, matches[1], :] - X[:, matches[0], :]))
    if jumps:
        return np.concatenate([j.reshape(-1) for j in jumps])
    if len(blocks) > 1:
        raise ValueError(
            "stitch_X_boundary_abs_jump is required when stitched outputs do not retain "
            "duplicate boundary times"
        )
    return np.asarray([0.0], dtype=np.float32)


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
) -> Dict[str, float]:
    pathwise_violations = _physical_violation_values_from_x(stitched, params)
    summary: Dict[str, float] = {}
    for key in (
        "q_lower_violation",
        "q_upper_violation",
        "v_lower_violation",
        "v_upper_violation",
    ):
        values = pathwise_violations.get(key, np.asarray(stitched[key], dtype=np.float32))
        stats = _trace_stats(values)
        for stat_name, value in stats.items():
            summary[f"{key}_{stat_name}"] = value

    jump_values = _stitch_boundary_jump_values(stitched, blocks)
    summary["stitch_X_boundary_max_abs_jump"] = float(np.max(jump_values))
    summary["stitch_X_boundary_mean_abs_jump"] = float(np.mean(jump_values))
    for key in ("Y", "Z"):
        jump_key = f"stitch_{key}_boundary_abs_jump"
        if jump_key in stitched:
            values = np.asarray(stitched[jump_key], dtype=np.float32).reshape(-1)
            summary[f"stitch_{key}_boundary_max_abs_jump"] = float(np.max(values))
            summary[f"stitch_{key}_boundary_mean_abs_jump"] = float(np.mean(values))
    return summary
