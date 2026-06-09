from __future__ import annotations

from typing import Dict

import numpy as np


def _trace_stats(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
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


def summarize_pascucci_stitched_diagnostics(
    stitched: dict,
    blocks: list,
    params: dict | None = None,
) -> Dict[str, float]:
    del params
    summary: Dict[str, float] = {}
    for key in (
        "q_lower_violation",
        "q_upper_violation",
        "v_lower_violation",
        "v_upper_violation",
    ):
        stats = _trace_stats(np.asarray(stitched[key], dtype=np.float32))
        for stat_name, value in stats.items():
            summary[f"{key}_{stat_name}"] = value

    jump_values = _stitch_boundary_jump_values(stitched, blocks)
    summary["stitch_X_boundary_max_abs_jump"] = float(np.max(jump_values))
    summary["stitch_X_boundary_mean_abs_jump"] = float(np.mean(jump_values))
    return summary
