"""Exact quadratic-coupled formulas and diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .io_utils import save_rows_csv
from .naming import _z_component_labels
from .sampling import build_stitched_rollout_inputs

def build_exact_solution_functions(
    solution_name: str,
    params: Dict[str, np.ndarray],
    D: int,
) -> Optional[Dict[str, Any]]:
    name = str(solution_name or "none").strip().lower()
    if name in ("", "none", "off", "false", "0"):
        return None

    if name in ("quadratic_coupled", "quadratic", "qc4d"):
        if int(D) != 4:
            raise ValueError(
                f"exact_solution='quadratic_coupled' requires D=4, found D={int(D)}"
            )

        gamma = float(params["gamma"])
        s1 = float(params["s1"])
        s3 = float(params["s3"])

        def u_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
            _ = t_arr
            Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
            S = Xi_arr[:, 0:1]
            V = Xi_arr[:, 2:3]
            X_state = Xi_arr[:, 3:4]
            return (-gamma * np.exp(S) * X_state + V ** 2 + V * X_state).astype(np.float32)

        def z_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
            _ = t_arr
            Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
            S = Xi_arr[:, 0:1]
            V = Xi_arr[:, 2:3]
            X_state = Xi_arr[:, 3:4]

            z_s = -gamma * np.exp(S) * X_state * s1
            z_h = np.zeros_like(z_s)
            z_v = (2.0 * V + X_state) * s3
            z_x = np.zeros_like(z_s)
            return np.concatenate([z_s, z_h, z_v, z_x], axis=1).astype(np.float32)

        return {
            "name": "quadratic_coupled",
            "u_exact": u_exact,
            "z_exact": z_exact,
        }

    raise ValueError(
        "Unknown exact_solution profile "
        f"'{solution_name}'. Supported: none, quadratic_coupled"
    )

def _clip_unit_interval_np(arr: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, arr)).astype(np.float32)

def _quadratic_coupled_psi_x_np(X_state: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    d = float(params["d"])
    x_max = float(params["x_max"])
    return np.maximum(
        0.0,
        np.minimum(
            1.0,
            np.minimum(X_state / d, (x_max - X_state) / d),
        ),
    ).astype(np.float32)

def _quadratic_coupled_psi3_np(V: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    d = float(params["d"])
    v_max = float(params["v_max"])
    return _clip_unit_interval_np((v_max - V) / d)

def _quadratic_coupled_psi4_np(V: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    d = float(params["d"])
    v_min = float(params["v_min"])
    return _clip_unit_interval_np((V - v_min) / d)

def quadratic_coupled_exact_z_np(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    S = X[:, 0:1]
    V = X[:, 2:3]
    X_state = X[:, 3:4]
    gamma = float(params["gamma"])
    s1 = float(params["s1"])
    s3 = float(params["s3"])
    z_s = -gamma * np.exp(S) * X_state * s1
    z_h = np.zeros_like(z_s)
    z_v = (2.0 * V + X_state) * s3
    z_x = np.zeros_like(z_s)
    return np.concatenate([z_s, z_h, z_v, z_x], axis=1).astype(np.float32)

def quadratic_coupled_mu_np(
    X: np.ndarray,
    Z: np.ndarray,
    params: Dict[str, np.ndarray],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)

    S = X[:, 0:1]
    H = X[:, 1:2]
    V = X[:, 2:3]
    X_state = X[:, 3:4]
    Z_S = Z[:, 0:1]

    mu1 = float(params["mu1"])
    mu2 = float(params["mu2"])
    c1 = float(params["c1"])
    c2 = float(params["c2"])
    c3 = float(params["c3"])
    c4 = float(params["c4"])
    gamma = float(params["gamma"])
    s1 = float(params["s1"])
    x_max = float(params["x_max"])

    psi_x = _quadratic_coupled_psi_x_np(X_state, params)
    psi_neg_x = _quadratic_coupled_psi_x_np(-X_state, params)
    psi_x_minus_xmax = _quadratic_coupled_psi_x_np(X_state - x_max, params)
    psi3 = _quadratic_coupled_psi3_np(V, params)
    psi4 = _quadratic_coupled_psi4_np(V, params)
    exp_neg_S = np.exp(-S).astype(np.float32)
    control = _quadratic_coupled_psi_x_np(-exp_neg_S * Z_S / (gamma * s1), params)
    f_val = (-0.5 * V * control).astype(np.float32)

    dS = mu1 * (c1 - S)
    dH = mu2 * (c2 - H)
    dV = f_val * psi_x + c3 * psi_neg_x * psi3 - c4 * psi_x_minus_xmax * psi4
    dX = V
    return np.concatenate([dS, dH, dV, dX], axis=1).astype(np.float32)

def build_exact_initial_boundary_samples(
    Xi_generator,
    exact_solution: Dict[str, Any],
    params: Dict[str, np.ndarray],
    blocks: List[Dict[str, float]],
    M_rollout: int,
    N_per_block: int,
    D: int,
    seed: int = 1234,
) -> List[np.ndarray]:
    if exact_solution is None:
        raise ValueError("Exact initialization requested but exact_solution is disabled")
    if str(exact_solution.get("name", "")).strip().lower() != "quadratic_coupled":
        raise ValueError(
            "Exact initialization is currently supported only for exact_solution='quadratic_coupled'"
        )

    Xi_curr = Xi_generator(M_rollout, D).astype(np.float32)
    rollout_inputs = build_stitched_rollout_inputs(
        blocks=blocks,
        M=int(M_rollout),
        N_per_block=int(N_per_block),
        D=int(D),
        seed=int(seed),
    )
    boundary_samples = [Xi_curr.copy()]

    s1 = float(params["s1"])
    s2 = float(params["s2"])
    s3 = float(params["s3"])

    for block_idx, block in enumerate(blocks):
        t_b, W_b = rollout_inputs[block_idx]
        X_state = Xi_curr.copy()
        for n in range(int(N_per_block)):
            dt = (t_b[:, n + 1, :] - t_b[:, n, :]).astype(np.float32)
            dW = (W_b[:, n + 1, :] - W_b[:, n, :]).astype(np.float32)
            Z_exact = quadratic_coupled_exact_z_np(X_state, params)
            mu = quadratic_coupled_mu_np(X_state, Z_exact, params)

            X_next = X_state.copy()
            X_next[:, 0:1] = X_state[:, 0:1] + mu[:, 0:1] * dt + s1 * dW[:, 0:1]
            X_next[:, 1:2] = X_state[:, 1:2] + mu[:, 1:2] * dt + s2 * dW[:, 1:2]
            X_next[:, 2:3] = X_state[:, 2:3] + mu[:, 2:3] * dt + s3 * dW[:, 2:3]
            X_next[:, 3:4] = X_state[:, 3:4] + mu[:, 3:4] * dt
            X_state = X_next.astype(np.float32)

        Xi_curr = X_state.astype(np.float32)
        boundary_samples.append(Xi_curr.copy())

    return boundary_samples

def compute_stitched_exact_bundle(
    stitched: Dict[str, np.ndarray],
    exact_solution: Dict[str, Any],
    eps: float = 1.0e-8,
) -> Dict[str, Any]:
    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]

    M_paths = int(X_all.shape[0])
    T_points = int(X_all.shape[1])
    D = int(X_all.shape[2])

    X_flat = X_all.reshape(-1, D)
    t_flat = t_all.reshape(-1, 1)

    Y_exact = exact_solution["u_exact"](t_flat, X_flat).reshape(M_paths, T_points, 1).astype(np.float32)
    Z_exact = exact_solution["z_exact"](t_flat, X_flat).reshape(M_paths, T_points, D).astype(np.float32)

    abs_err_Y = np.abs(Y_pred - Y_exact)
    abs_err_Z = np.abs(Z_pred - Z_exact)

    # Legacy relative error (kept for backward compatibility / diagnostics).
    rel_err_Z_legacy = abs_err_Z / (np.abs(Z_exact) + float(eps))

    # Robust relative error: ignore components where exact Z is (near) zero.
    valid_mask = np.abs(Z_exact) > float(eps)
    rel_err_Z = np.zeros_like(abs_err_Z, dtype=np.float32)
    np.divide(
        abs_err_Z,
        np.abs(Z_exact) + float(eps),
        out=rel_err_Z,
        where=valid_mask,
    )

    y0_pred = Y_pred[:, 0, 0]
    y0_exact = Y_exact[:, 0, 0]

    mean_abs_err_Y_t = np.mean(abs_err_Y[:, :, 0], axis=0)
    mean_abs_err_Z_t = np.mean(abs_err_Z, axis=0)
    mean_rel_err_Z_legacy_t = np.mean(rel_err_Z_legacy, axis=0)
    valid_count_t = np.maximum(np.sum(valid_mask, axis=0), 1.0).astype(np.float32)
    mean_rel_err_Z_t = (np.sum(rel_err_Z, axis=0) / valid_count_t).astype(np.float32)
    valid_count_all = float(max(np.sum(valid_mask), 1.0))
    valid_count_comp = np.maximum(np.sum(valid_mask, axis=(0, 1)), 1.0).astype(np.float32)

    summary = {
        "solution_name": exact_solution.get("name", "unknown"),
        "n_paths": int(M_paths),
        "n_time_points": int(T_points),
        "mean_pred_y0": float(np.mean(y0_pred)),
        "mean_exact_y0": float(np.mean(y0_exact)),
        "abs_error_mean_y0": float(np.mean(np.abs(y0_pred - y0_exact))),
        "rmse_y0": float(np.sqrt(np.mean((y0_pred - y0_exact) ** 2))),
        "mean_abs_error_y": float(np.mean(abs_err_Y)),
        "rmse_y": float(np.sqrt(np.mean((Y_pred - Y_exact) ** 2))),
        "mean_abs_error_z": float(np.mean(abs_err_Z)),
        "mean_rel_error_z": float(np.sum(rel_err_Z) / valid_count_all),
        "mean_rel_error_z_legacy": float(np.mean(rel_err_Z_legacy)),
        "mean_abs_error_z_by_component": np.mean(abs_err_Z, axis=(0, 1)).astype(np.float32),
        "mean_rel_error_z_by_component": (np.sum(rel_err_Z, axis=(0, 1)) / valid_count_comp).astype(np.float32),
        "mean_rel_error_z_by_component_legacy": np.mean(rel_err_Z_legacy, axis=(0, 1)).astype(np.float32),
        "valid_rel_error_fraction_z_by_component": np.mean(valid_mask.astype(np.float32), axis=(0, 1)).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    timeseries = {
        "t": t_all[0, :, 0].astype(np.float32),
        "mean_abs_error_y": mean_abs_err_Y_t.astype(np.float32),
        "mean_abs_error_z": mean_abs_err_Z_t.astype(np.float32),
        "mean_rel_error_z": mean_rel_err_Z_t.astype(np.float32),
        "mean_rel_error_z_legacy": mean_rel_err_Z_legacy_t.astype(np.float32),
        "valid_rel_error_fraction_z": np.mean(valid_mask.astype(np.float32), axis=0).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    return {
        "summary": summary,
        "timeseries": timeseries,
        "Y_exact": Y_exact,
        "Z_exact": Z_exact,
    }

def save_exact_error_timeseries_csv(timeseries: Dict[str, np.ndarray], path: str) -> None:
    t = np.asarray(timeseries["t"])
    abs_y = np.asarray(timeseries["mean_abs_error_y"])
    abs_z = np.asarray(timeseries["mean_abs_error_z"])
    rel_z = np.asarray(timeseries["mean_rel_error_z"])
    rel_z_legacy = np.asarray(timeseries["mean_rel_error_z_legacy"]) if "mean_rel_error_z_legacy" in timeseries else None
    labels = timeseries.get("z_component_labels", _z_component_labels(abs_z.shape[1]))

    rows = []
    for i in range(int(t.shape[0])):
        row = {
            "t": float(t[i]),
            "mean_abs_error_y": float(abs_y[i]),
        }
        for d, label in enumerate(labels):
            row[f"mean_abs_error_{label}"] = float(abs_z[i, d])
            row[f"mean_rel_error_{label}"] = float(rel_z[i, d])
            if rel_z_legacy is not None:
                row[f"mean_rel_error_{label}_legacy"] = float(rel_z_legacy[i, d])
        rows.append(row)

    save_rows_csv(rows, path)
