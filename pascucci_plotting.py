"""Artifact-driven Pascucci paper plot helpers."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .io_utils import _as_blob_dict, save_blob_npz, save_json
from .plotting import _PLOTTING_AVAILABLE, plt


PAPER_PLOT_SCHEMA = "pascucci_paper_plots_v1"
PAPER_PLOT_MANIFEST = "pascucci_paper_plots_manifest.json"
PAPER_PLOT_DATA_SCHEMA = "pascucci_paper_plot_data_v1"
PAPER_PLOT_DATA_NPZ = "pascucci_paper_plot_data.npz"
PLOTMAKER_NATIVE_Y_FILENAME = "pascucci_plotmaker_backward_Y.png"
PLOTMAKER_NATIVE_Z_FILENAME = "pascucci_plotmaker_Z_components.png"
APPLICATION_METRIC_SCHEMA = "pascucci_application_metrics_v2"
TO_EMA_OU_REFERENCE_SCHEMA = "to_ema_ou_reference_v1"
TO_EMA_OU_NSIM = 10000
TO_EMA_OU_DT_SIM = 0.5
TO_EMA_OU_SEED = 42
TO_EMA_UNCONTROLLED_REFERENCE_SCHEMA = "to_ema_uncontrolled_reference_v1"
TO_EMA_UNCONTROLLED_NSIM = 10000
TO_EMA_UNCONTROLLED_DT = 0.1
TO_EMA_UNCONTROLLED_SEED = 42
TO_EMA_UNCONTROLLED_EPS = 0.01
TO_EMA_UNCONTROLLED_XMAX = 10.0
TO_EMA_UNCONTROLLED_LAMBDA_V = 1.0e-2
TO_EMA_UNCONTROLLED_GAMMA = 1.0
TO_EMA_UNCONTROLLED_OMEGA = 1.0e-2
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


def _require_plotting() -> None:
    if not _PLOTTING_AVAILABLE:
        raise RuntimeError("matplotlib is required for Pascucci paper plots")


def _as_finite_array(value: Any, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if ndim is not None and arr.ndim != int(ndim):
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_stitched(stitched: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in ("t", "X") if key not in stitched]
    if missing:
        raise ValueError(f"stitched artifact is missing keys: {', '.join(missing)}")
    t = _as_finite_array(stitched["t"], name="stitched.t", ndim=3)
    X = _as_finite_array(stitched["X"], name="stitched.X", ndim=3)
    if t.shape[0] != X.shape[0] or t.shape[1] != X.shape[1]:
        raise ValueError(f"stitched.t and stitched.X time grids differ: {t.shape} vs {X.shape}")
    if X.shape[2] < 4:
        raise ValueError("Pascucci paper plots require state columns S,H,V,Q")
    t0 = t[0, :, 0]
    if not np.allclose(t[:, :, 0], t0.reshape(1, -1), rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("Pascucci paper plots require a shared time grid across paths")
    if np.any(np.diff(t0) <= 0.0):
        raise ValueError("Pascucci paper plot time grid must be strictly increasing")
    return t0.astype(np.float32), X.astype(np.float32)


def _require_pathwise(application_pathwise: Dict[str, Any], key: str, *, ndim: Optional[int] = None) -> np.ndarray:
    if key not in application_pathwise:
        if key.endswith("cost_J_running_cumulative"):
            raise ValueError(
                "application_metrics pathwise artifact is missing "
                f"'{key}'. Pascucci paper plot schema {PAPER_PLOT_SCHEMA} "
                "requires Sprint 19 cumulative running-cost traces; regenerate "
                "application metrics with the current code."
            )
        raise ValueError(f"application_metrics pathwise artifact is missing '{key}'")
    return _as_finite_array(application_pathwise[key], name=f"application.{key}", ndim=ndim)


def _band(values: np.ndarray) -> Dict[str, np.ndarray]:
    arr = _as_finite_array(values, name="band values", ndim=2)
    return {
        "mean": np.mean(arr, axis=0),
        "q05": np.quantile(arr, 0.05, axis=0).astype(np.float32),
        "q50": np.quantile(arr, 0.50, axis=0).astype(np.float32),
        "q95": np.quantile(arr, 0.95, axis=0).astype(np.float32),
    }


def _paper_j_band(values: np.ndarray) -> Dict[str, np.ndarray]:
    arr = _as_finite_array(values, name="paper J values", ndim=2)
    return {
        "mean": np.mean(arr, axis=0),
        "q10": np.quantile(arr, 0.10, axis=0).astype(np.float32),
        "q90": np.quantile(arr, 0.90, axis=0).astype(np.float32),
    }


def _pascucci_price_eur_mwh(log_price: np.ndarray) -> np.ndarray:
    values = _as_finite_array(log_price, name="Pascucci log-price values")
    return (np.exp(np.clip(values, -50.0, 50.0)) * np.float32(1000.0)).astype(np.float32)


def _resolve_metadata_source_path(path_value: Any, *, kind: str) -> Optional[str]:
    text = str(path_value or "").strip()
    if text == "":
        return None
    expanded = Path(os.path.expanduser(text))
    if expanded.exists():
        return str(expanded.resolve())

    basename = expanded.name
    if basename == "":
        return str(expanded)
    kind_dir = "casa" if str(kind) == "H" else "prezzi"
    code_path = Path(__file__).resolve()
    for parent_index in (2, 3):
        local_to_ema = code_path.parents[parent_index] / "to_ema" / "dataset" / kind_dir / basename
        if local_to_ema.exists():
            return str(local_to_ema.resolve())
    return str(expanded)


def _source_sha256_mismatch(path_value: str, metadata: Dict[str, Any], *, label: str) -> Optional[Dict[str, str]]:
    expected = str(metadata.get("source_sha256", "") or "").strip().lower()
    if expected == "":
        return None
    try:
        actual = hashlib.sha256(Path(path_value).read_bytes()).hexdigest()
    except OSError as exc:
        return {
            "label": str(label),
            "source_path": str(metadata.get("source_path", "")),
            "resolved_source_path": str(path_value),
            "expected_source_sha256": expected,
            "error": str(exc),
        }
    if actual == expected:
        return None
    return {
        "label": str(label),
        "source_path": str(metadata.get("source_path", "")),
        "resolved_source_path": str(path_value),
        "expected_source_sha256": expected,
        "resolved_source_sha256": actual,
    }


def _plotmaker_source_match(metadata: Dict[str, Any], *, expected_basename: str, expected_sha256: str) -> Dict[str, Any]:
    source_path = str(metadata.get("source_path", ""))
    source_sha256 = str(metadata.get("source_sha256", ""))
    basename = os.path.basename(source_path)
    return {
        "expected_basename": expected_basename,
        "expected_sha256": expected_sha256,
        "actual_basename": basename,
        "actual_sha256": source_sha256,
        "matched": basename == expected_basename and source_sha256 == expected_sha256,
    }


def _plotmaker_reference_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    calibration = params.get("pascucci_calibration", {})
    if not isinstance(calibration, dict):
        H_match = _plotmaker_source_match(
            {},
            expected_basename=PLOTMAKER_H_BASENAME,
            expected_sha256=PLOTMAKER_H_SHA256,
        )
        S_match = _plotmaker_source_match(
            {},
            expected_basename=PLOTMAKER_S_BASENAME,
            expected_sha256=PLOTMAKER_S_SHA256,
        )
        dataset_status = "missing_calibration"
    else:
        H_match = _plotmaker_source_match(
            dict(calibration.get("H_metadata", {})),
            expected_basename=PLOTMAKER_H_BASENAME,
            expected_sha256=PLOTMAKER_H_SHA256,
        )
        S_match = _plotmaker_source_match(
            dict(calibration.get("S_metadata", {})),
            expected_basename=PLOTMAKER_S_BASENAME,
            expected_sha256=PLOTMAKER_S_SHA256,
        )
        try:
            dt_ok = abs(float(calibration.get("dt")) - PLOTMAKER_CALIBRATION_DT) <= 1.0e-12
        except (TypeError, ValueError):
            dt_ok = False
        k_ok = int(calibration.get("K", -1)) == PLOTMAKER_CALIBRATION_K
        dataset_status = "matched" if H_match["matched"] and S_match["matched"] and dt_ok and k_ok else "mismatch"
    return {
        "source": "raw/to_ema/plotmaker.ipynb cells 8, 16, 18",
        "data": PLOTMAKER_DATASET_NAME,
        "T": PLOTMAKER_T,
        "N": PLOTMAKER_N,
        "M": PLOTMAKER_M,
        "calibration_K": PLOTMAKER_CALIBRATION_K,
        "calibration_dt": PLOTMAKER_CALIBRATION_DT,
        "H": H_match,
        "S": S_match,
        "dataset_status": dataset_status,
    }


def _is_day(t: np.ndarray) -> np.ndarray:
    hour = np.mod(np.asarray(t, dtype=np.float32), 24.0)
    return (hour >= 7.0) & (hour < 19.0)


def _harmonic_mean(t: np.ndarray, params: Dict[str, Any], prefix: str) -> np.ndarray:
    alpha = np.asarray(params[f"alpha_{prefix}"], dtype=np.float32).reshape(-1)
    beta = np.asarray(params[f"beta_{prefix}"], dtype=np.float32).reshape(-1)
    if alpha.shape != beta.shape:
        raise ValueError(f"OU alpha_{prefix}/beta_{prefix} shapes differ")
    mean = np.full_like(t, np.float32(params[f"a0_{prefix}"]), dtype=np.float32)
    for idx, (alpha_k, beta_k) in enumerate(zip(alpha, beta), start=1):
        omega = np.float32(2.0 * np.pi * float(idx) / 24.0)
        mean = mean + np.float32(alpha_k) * np.cos(omega * t) + np.float32(beta_k) * np.sin(omega * t)
    return mean.astype(np.float32)


def _ou_mean_band(t: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    day = _is_day(t)
    mean_day = _harmonic_mean(t, params, "day")
    mean_night = _harmonic_mean(t, params, "night")
    mean = np.where(day, mean_day, mean_night).astype(np.float32)
    kappa_day = max(float(params["kappa_day"]), 1.0e-12)
    kappa_night = max(float(params["kappa_night"]), 1.0e-12)
    sigma_day = float(params["sigma_day"])
    sigma_night = float(params["sigma_night"])
    std_day = sigma_day / np.sqrt(2.0 * kappa_day)
    std_night = sigma_night / np.sqrt(2.0 * kappa_night)
    std = np.where(day, std_day, std_night).astype(np.float32)
    return {
        "mean": mean,
        "lower": (mean - np.float32(1.96) * std).astype(np.float32),
        "upper": (mean + np.float32(1.96) * std).astype(np.float32),
    }


def _simulate_ou_day_night_paths(
    *,
    initial_value: Any,
    params: Dict[str, Any],
    T: float,
    n_sim: int = TO_EMA_OU_NSIM,
    dt: float = TO_EMA_OU_DT_SIM,
    seed: int = TO_EMA_OU_SEED,
    start_hour: float = 0.0,
) -> np.ndarray:
    T_float = float(T)
    dt_float = float(dt)
    n_sim_int = int(n_sim)
    if not np.isfinite(T_float) or T_float <= 0.0:
        raise ValueError("OU simulation T must be positive and finite")
    if not np.isfinite(dt_float) or dt_float <= 0.0:
        raise ValueError("OU simulation dt must be positive and finite")
    if n_sim_int <= 0:
        raise ValueError("OU simulation n_sim must be positive")

    n_steps = int(T_float / dt_float)
    if n_steps <= 0:
        raise ValueError("OU simulation needs at least one step")
    paths = np.zeros((n_sim_int, n_steps + 1), dtype=np.float32)
    initial = np.asarray(initial_value, dtype=np.float32).reshape(-1)
    if initial.size == 1:
        paths[:, 0] = float(initial[0])
    elif initial.size == n_sim_int:
        paths[:, 0] = initial.astype(np.float32)
    else:
        raise ValueError(f"OU initial_value must be scalar or length n_sim, got shape {initial.shape}")

    rng = np.random.RandomState(int(seed))
    for i in range(n_steps):
        t_actual = np.asarray([float(start_hour) + float(i) * dt_float], dtype=np.float32)
        if bool(_is_day(t_actual)[0]):
            prefix = "day"
        else:
            prefix = "night"
        mu = float(_harmonic_mean(t_actual, params, prefix)[0])
        kappa = float(params[f"kappa_{prefix}"])
        sigma = float(params[f"sigma_{prefix}"])
        z = rng.randn(n_sim_int).astype(np.float32)
        paths[:, i + 1] = (
            paths[:, i]
            + np.float32(kappa) * (np.float32(mu) - paths[:, i]) * np.float32(dt_float)
            + np.float32(sigma) * z * np.float32(np.sqrt(dt_float))
        )
    return paths.astype(np.float32)


def _load_to_ema_ou_reference(params: Dict[str, Any], *, horizon_T: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
    calibration = params.get("pascucci_calibration", {})
    if not isinstance(calibration, dict) or calibration.get("schema") != "pascucci_ou_calibration_v1":
        return {}, {
            "schema": TO_EMA_OU_REFERENCE_SCHEMA,
            "source": "stitched_fallback",
            "reason": "missing_pascucci_calibration_metadata",
        }

    H_metadata = dict(calibration.get("H_metadata", {}))
    S_metadata = dict(calibration.get("S_metadata", {}))
    H_path = _resolve_metadata_source_path(H_metadata.get("source_path", ""), kind="H")
    S_path = _resolve_metadata_source_path(S_metadata.get("source_path", ""), kind="S")
    missing_paths = [
        str(path)
        for path in (H_path, S_path)
        if path is None or not os.path.exists(str(path))
    ]
    if missing_paths:
        return {}, {
            "schema": TO_EMA_OU_REFERENCE_SCHEMA,
            "source": "stitched_fallback",
            "reason": "calibration_source_path_not_found",
            "missing_paths": missing_paths,
        }
    sha_mismatches = [
        mismatch
        for mismatch in (
            _source_sha256_mismatch(str(H_path), H_metadata, label="H_metadata"),
            _source_sha256_mismatch(str(S_path), S_metadata, label="S_metadata"),
        )
        if mismatch is not None
    ]
    if sha_mismatches:
        return {}, {
            "schema": TO_EMA_OU_REFERENCE_SCHEMA,
            "source": "stitched_fallback",
            "reason": "calibration_source_sha256_mismatch",
            "mismatches": sha_mismatches,
        }

    from .pascucci_data import prepare_H, prepare_S

    H_n = int(H_metadata.get("n_per_hour", 1))
    S_n = int(S_metadata.get("n_per_hour", 1))
    H_mul = float(H_metadata.get("mul_factor", 1.0))
    S_mul = float(S_metadata.get("mul_factor", 1.0))
    H = prepare_H(str(H_path), n=H_n, mul_factor=H_mul)
    S = prepare_S(str(S_path), n=S_n, mul_factor=S_mul)
    if H.size < 2 or S.size < 2:
        raise ValueError("Pascucci calibration source series must contain at least two hourly points")
    if np.any(S <= 0.0):
        raise ValueError("Pascucci price source must be strictly positive for log-price OU plots")

    dt_real = float(calibration.get("dt", 1.0))
    start_hour = float(calibration.get("start_hour", 0.0))
    T_requested = float(horizon_T)
    T_H = float(H.shape[0] - 1) * dt_real
    T_S = float(S.shape[0] - 1) * dt_real
    if T_H <= 0.0 or T_S <= 0.0:
        raise ValueError("Pascucci calibration source series are too short for OU plots")

    H_paths = _simulate_ou_day_night_paths(
        initial_value=float(H[0]),
        params=dict(params["params_H"]),
        T=T_H,
        n_sim=TO_EMA_OU_NSIM,
        dt=TO_EMA_OU_DT_SIM,
        seed=TO_EMA_OU_SEED,
        start_hour=start_hour,
    )
    S_log_paths = _simulate_ou_day_night_paths(
        initial_value=float(np.log(S[0])),
        params=dict(params["params_S"]),
        T=T_S,
        n_sim=TO_EMA_OU_NSIM,
        dt=TO_EMA_OU_DT_SIM,
        seed=TO_EMA_OU_SEED,
        start_hour=start_hour,
    )
    references = {
        "#35": {
            "paths": _pascucci_price_eur_mwh(S_log_paths),
            "real_path": (np.asarray(S, dtype=np.float32) * np.float32(1000.0)).astype(np.float32),
            "T": float(T_S),
            "name": "PUN Price",
            "ylabel": "EUR/MWh",
            "source_path": str(S_metadata.get("source_path", "")),
            "resolved_source_path": str(S_path),
        },
        "#36": {
            "paths": H_paths,
            "real_path": np.asarray(H, dtype=np.float32),
            "T": float(T_H),
            "name": "Net-Load",
            "ylabel": "kW",
            "source_path": str(H_metadata.get("source_path", "")),
            "resolved_source_path": str(H_path),
        },
    }
    summary = {
        "schema": TO_EMA_OU_REFERENCE_SCHEMA,
        "source": "calibration_metadata",
        "simulation": {
            "n_sim": int(TO_EMA_OU_NSIM),
            "dt_sim": float(TO_EMA_OU_DT_SIM),
            "seed": int(TO_EMA_OU_SEED),
            "quantiles": [0.05, 0.95],
            "band_label": "80% Band",
        },
        "dt_real": float(dt_real),
        "start_hour": float(start_hour),
        "run_horizon_T": float(T_requested),
        "source_horizon_policy": "full_calibration_series_like_to_ema",
        "#35": {
            "source_path": references["#35"]["source_path"],
            "resolved_source_path": references["#35"]["resolved_source_path"],
            "real_path_points": int(S.shape[0]),
            "simulation_points": int(references["#35"]["paths"].shape[1]),
            "T": float(T_S),
            "transform": "exp(logS) * 1000",
        },
        "#36": {
            "source_path": references["#36"]["source_path"],
            "resolved_source_path": references["#36"]["resolved_source_path"],
            "real_path_points": int(H.shape[0]),
            "simulation_points": int(references["#36"]["paths"].shape[1]),
            "T": float(T_H),
            "transform": "identity",
        },
    }
    return references, summary


def _to_ema_asymmetric_quadratic(values: np.ndarray, *, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(arr < 0.0, np.float32(scale) * arr * arr, np.float32(2.0 * scale) * arr * arr)


def _simulate_to_ema_x_v_uncontrolled(
    *,
    X_0: np.ndarray,
    H_paths: np.ndarray,
    T: float,
    n_sim: int,
    dt: float,
    eps: float,
    x_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(float(T) / float(dt))
    n_time = n_steps + 1
    X_paths = np.zeros((int(n_sim), n_time), dtype=np.float32)
    V_paths = np.zeros((int(n_sim), n_time), dtype=np.float32)
    X_paths[:, 0] = np.asarray(X_0, dtype=np.float32).reshape(int(n_sim))
    H_arr = _as_finite_array(H_paths, name="to_ema uncontrolled H paths", ndim=2)
    if H_arr.shape[0] != int(n_sim) or H_arr.shape[1] < n_time:
        raise ValueError(f"to_ema uncontrolled H paths shape {H_arr.shape}, expected ({n_sim}, >={n_time})")

    V_paths[:, 0] = np.where(
        X_paths[:, 0] <= 0.0,
        -np.minimum(H_arr[:, 0], 0.0),
        np.where(X_paths[:, 0] >= float(x_max), -np.maximum(H_arr[:, 0], 0.0), -H_arr[:, 0]),
    )
    rng = np.random.RandomState(int(seed))
    W_paths = np.cumsum(
        rng.randn(int(n_sim), n_time).astype(np.float32) * np.float32(np.sqrt(float(dt))),
        axis=-1,
    ).astype(np.float32)
    for step in range(n_time - 1):
        X_paths[:, step + 1] = X_paths[:, step] + V_paths[:, step] * np.float32(dt)
        V_paths[:, step + 1] = (
            np.where(
                X_paths[:, step + 1] <= 0.0,
                -np.minimum(H_arr[:, step + 1], 0.0),
                np.where(
                    X_paths[:, step + 1] >= float(x_max),
                    -np.maximum(H_arr[:, step + 1], 0.0),
                    -H_arr[:, step + 1],
                ),
            )
            + np.float32(eps) * W_paths[:, step + 1]
        )
    return X_paths.astype(np.float32), V_paths.astype(np.float32)


def _to_ema_uncontrolled_cost_paths(
    *,
    H_paths: np.ndarray,
    S_paths: np.ndarray,
    V_paths: np.ndarray,
    X_paths: np.ndarray,
    dt: float,
) -> np.ndarray:
    H_arr = _as_finite_array(H_paths, name="to_ema uncontrolled H paths", ndim=2)
    S_arr = _as_finite_array(S_paths, name="to_ema uncontrolled S paths", ndim=2)
    V_arr = _as_finite_array(V_paths, name="to_ema uncontrolled V paths", ndim=2)
    X_arr = _as_finite_array(X_paths, name="to_ema uncontrolled X paths", ndim=2)
    if H_arr.shape != S_arr.shape or H_arr.shape != V_arr.shape or H_arr.shape != X_arr.shape:
        raise ValueError(
            "to_ema uncontrolled path shapes must match "
            f"(H={H_arr.shape}, S={S_arr.shape}, V={V_arr.shape}, X={X_arr.shape})"
        )
    mean_X = np.mean(X_arr, axis=0)
    mean_H = np.mean(H_arr, axis=0)
    mean_V = np.mean(V_arr, axis=0)
    f_paths = (
        S_arr * (H_arr + V_arr)
        + np.float32(TO_EMA_UNCONTROLLED_LAMBDA_V) * V_arr**2
        + _to_ema_asymmetric_quadratic(X_arr - mean_X.reshape(1, -1), scale=0.001)
        + _to_ema_asymmetric_quadratic(
            H_arr + V_arr - (mean_H + mean_V).reshape(1, -1),
            scale=0.1,
        )
    )
    f_integral = np.cumsum(f_paths * np.float32(dt), axis=-1)
    g_paths = (
        -np.float32(TO_EMA_UNCONTROLLED_GAMMA) * S_arr * X_arr
        + np.float32(0.5 * TO_EMA_UNCONTROLLED_OMEGA) * (X_arr - mean_X.reshape(1, -1)) ** 2
    )
    return (f_integral + g_paths).astype(np.float32)


def _load_to_ema_uncontrolled_reference(params: Dict[str, Any], *, horizon_T: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
    calibration = params.get("pascucci_calibration", {})
    if not isinstance(calibration, dict) or calibration.get("schema") != "pascucci_ou_calibration_v1":
        return {}, {
            "schema": TO_EMA_UNCONTROLLED_REFERENCE_SCHEMA,
            "source": "application_metrics_fallback",
            "reason": "missing_pascucci_calibration_metadata",
        }
    if "params_H" not in params or "params_S" not in params:
        return {}, {
            "schema": TO_EMA_UNCONTROLLED_REFERENCE_SCHEMA,
            "source": "application_metrics_fallback",
            "reason": "missing_params_H_or_params_S",
        }

    T_float = float(horizon_T)
    if not np.isfinite(T_float) or T_float <= 0.0:
        raise ValueError("to_ema uncontrolled reference horizon_T must be positive and finite")
    n_sim = int(TO_EMA_UNCONTROLLED_NSIM)
    dt = float(TO_EMA_UNCONTROLLED_DT)
    seed = int(TO_EMA_UNCONTROLLED_SEED)
    start_hour = float(calibration.get("start_hour", 0.0))

    rng = np.random.RandomState(seed)
    X_0 = rng.uniform(1.0, 9.0, n_sim).astype(np.float32)
    H_0 = rng.normal(0.4, 0.5, n_sim).astype(np.float32)
    S_0 = rng.normal(0.1, 0.02, n_sim).astype(np.float32)
    if np.any(S_0 <= 0.0):
        raise ValueError("to_ema uncontrolled initial S_0 contains non-positive values")

    H_paths = _simulate_ou_day_night_paths(
        initial_value=H_0,
        params=dict(params["params_H"]),
        T=T_float,
        n_sim=n_sim,
        dt=dt,
        seed=seed,
        start_hour=start_hour,
    )
    S_log_paths = _simulate_ou_day_night_paths(
        initial_value=np.log(S_0).astype(np.float32),
        params=dict(params["params_S"]),
        T=T_float,
        n_sim=n_sim,
        dt=dt,
        seed=seed,
        start_hour=start_hour,
    )
    S_paths = np.exp(np.clip(S_log_paths, -50.0, 50.0)).astype(np.float32)
    X_paths, V_paths = _simulate_to_ema_x_v_uncontrolled(
        X_0=X_0,
        H_paths=H_paths,
        T=T_float,
        n_sim=n_sim,
        dt=dt,
        eps=float(TO_EMA_UNCONTROLLED_EPS),
        x_max=float(TO_EMA_UNCONTROLLED_XMAX),
        seed=seed,
    )
    J_paths = _to_ema_uncontrolled_cost_paths(
        H_paths=H_paths,
        S_paths=S_paths,
        V_paths=V_paths,
        X_paths=X_paths,
        dt=dt,
    )
    references = {
        "time": (np.arange(J_paths.shape[1], dtype=np.float32) * np.float32(dt)).astype(np.float32),
        "J_paths": J_paths,
    }
    summary = {
        "schema": TO_EMA_UNCONTROLLED_REFERENCE_SCHEMA,
        "source": "calibration_metadata",
        "simulation": {
            "n_sim": n_sim,
            "dt": dt,
            "seed": seed,
            "eps": float(TO_EMA_UNCONTROLLED_EPS),
            "x_max": float(TO_EMA_UNCONTROLLED_XMAX),
            "lambda_v": float(TO_EMA_UNCONTROLLED_LAMBDA_V),
            "gamma": float(TO_EMA_UNCONTROLLED_GAMMA),
            "omega": float(TO_EMA_UNCONTROLLED_OMEGA),
            "quantiles": [0.10, 0.90],
        },
        "initial_law": {
            "X_0": "Uniform(1, 9)",
            "H_0": "Normal(0.4, 0.5)",
            "S_0": "Normal(0.1, 0.02)",
        },
        "horizon_T": T_float,
        "time_points": int(J_paths.shape[1]),
        "start_hour": start_hour,
        "formula": "raw_to_ema_simulate_uncontrolled_J",
    }
    return references, summary


def _plot_to_ema_ou_simulation(
    *,
    paths: np.ndarray,
    real_path: np.ndarray,
    T: float,
    name: str,
    ylabel: str,
    path: str,
    dt_real: float = 1.0,
    dt_sim: float = TO_EMA_OU_DT_SIM,
) -> Dict[str, np.ndarray]:
    simulated = _as_finite_array(paths, name=f"{name} simulated OU paths", ndim=2)
    real = _as_finite_array(real_path, name=f"{name} real path", ndim=1)
    lower = np.percentile(simulated, 5, axis=0).astype(np.float32)
    upper = np.percentile(simulated, 95, axis=0).astype(np.float32)
    mean = np.mean(simulated, axis=0).astype(np.float32)
    median = np.percentile(simulated, 50, axis=0).astype(np.float32)
    time_sim = np.arange(simulated.shape[1], dtype=np.float32) * np.float32(dt_sim)
    time_real = np.arange(real.shape[0], dtype=np.float32) * np.float32(dt_real)
    real_points = max(1, min(int(float(T) / float(dt_real)), int(real.shape[0])))

    plt.figure(figsize=(12, 6))
    plt.plot(time_real[:real_points], real[:real_points], label=f"Real {name}", color="red")
    plt.fill_between(time_sim, lower[: time_sim.shape[0]], upper[: time_sim.shape[0]], alpha=0.3, label="80% Band")
    plt.legend()
    plt.title(f"OU Simulation with periodic mean of {name}")
    plt.xlabel("Hours")
    plt.ylabel(ylabel)
    plt.savefig(path, dpi=160)
    plt.close()
    return {
        "time_sim": time_sim,
        "sim_mean": mean,
        "sim_q05": lower,
        "sim_q50": median,
        "sim_q95": upper,
        "time_real": time_real[:real_points].astype(np.float32),
        "real": real[:real_points].astype(np.float32),
    }


def _plot_state_with_ou(
    *,
    time: np.ndarray,
    values: np.ndarray,
    ou_params: Dict[str, Any],
    ylabel: str,
    title: str,
    path: str,
    transform=None,
) -> Dict[str, np.ndarray]:
    transform_fn = (lambda x: x) if transform is None else transform
    empirical = _band(transform_fn(values))
    ou = _ou_mean_band(time, ou_params)
    ou = {key: transform_fn(value) for key, value in ou.items()}
    plt.figure(figsize=(10, 5))
    plt.fill_between(time, empirical["q05"], empirical["q95"], color="tab:blue", alpha=0.18, label="empirical q05-q95")
    plt.plot(time, empirical["q50"], color="tab:blue", linewidth=1.8, label="empirical median")
    plt.fill_between(time, ou["lower"], ou["upper"], color="tab:orange", alpha=0.16, label="OU +/-1.96 sigma_inf envelope")
    plt.plot(time, ou["mean"], color="tab:orange", linewidth=1.5, linestyle="--", label="OU mean")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {
        "time": np.asarray(time, dtype=np.float32),
        "empirical_mean": empirical["mean"].astype(np.float32),
        "empirical_q05": empirical["q05"].astype(np.float32),
        "empirical_q50": empirical["q50"].astype(np.float32),
        "empirical_q95": empirical["q95"].astype(np.float32),
        "ou_mean": ou["mean"].astype(np.float32),
        "ou_lower": ou["lower"].astype(np.float32),
        "ou_upper": ou["upper"].astype(np.float32),
    }


def _plot_accumulated_cost(
    *,
    time: np.ndarray,
    controlled: np.ndarray,
    uncontrolled: np.ndarray,
    trace_source: str,
    path: str,
    uncontrolled_time: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    controlled_band = _paper_j_band(np.asarray(controlled, dtype=np.float32)[:, :, 0])
    uncontrolled_band = _paper_j_band(np.asarray(uncontrolled, dtype=np.float32)[:, :, 0])
    controlled_time = _as_finite_array(time, name="controlled J time", ndim=1)
    uncontrolled_time_arr = controlled_time if uncontrolled_time is None else _as_finite_array(
        uncontrolled_time,
        name="uncontrolled J time",
        ndim=1,
    )
    if controlled_band["mean"].shape[0] != controlled_time.shape[0]:
        raise ValueError(
            f"controlled J trace length {controlled_band['mean'].shape[0]} does not match time {controlled_time.shape[0]}"
        )
    if uncontrolled_band["mean"].shape[0] != uncontrolled_time_arr.shape[0]:
        raise ValueError(
            "uncontrolled J trace length "
            f"{uncontrolled_band['mean'].shape[0]} does not match time {uncontrolled_time_arr.shape[0]}"
        )
    plt.figure(figsize=(10, 6))
    plt.plot(controlled_time, controlled_band["mean"], linewidth=2, color="r", label="Optimal $J_t$")
    plt.fill_between(
        uncontrolled_time_arr,
        uncontrolled_band["q10"],
        uncontrolled_band["q90"],
        color="b",
        alpha=0.2,
    )
    plt.fill_between(controlled_time, controlled_band["q10"], controlled_band["q90"], color="r", alpha=0.2)
    plt.plot(uncontrolled_time_arr, uncontrolled_band["mean"], linewidth=2, color="b", label="Benchmark Unoptimal $J_t$")
    if trace_source == "cost_J_trajectory":
        plt.title(r"Comparison of $J_t= \int_0^t f_s ds + g(X_t)$")
    else:
        plt.title("Pascucci accumulated running cost (legacy trace)")
    plt.xlabel("Time(h)")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(path, dpi=160)
    plt.close()
    return {
        "controlled_time": controlled_time.astype(np.float32),
        "controlled_mean": controlled_band["mean"].astype(np.float32),
        "controlled_q10": controlled_band["q10"].astype(np.float32),
        "controlled_q90": controlled_band["q90"].astype(np.float32),
        "uncontrolled_time": uncontrolled_time_arr.astype(np.float32),
        "uncontrolled_mean": uncontrolled_band["mean"].astype(np.float32),
        "uncontrolled_q10": uncontrolled_band["q10"].astype(np.float32),
        "uncontrolled_q90": uncontrolled_band["q90"].astype(np.float32),
    }


def _plot_alpha(
    *,
    time: np.ndarray,
    controlled_alpha: np.ndarray,
    uncontrolled_alpha: np.ndarray,
    path: str,
) -> Dict[str, np.ndarray]:
    controlled_band = _band(np.asarray(controlled_alpha, dtype=np.float32)[:, :, 0])
    uncontrolled_mean = np.mean(np.asarray(uncontrolled_alpha, dtype=np.float32)[:, :, 0], axis=0)
    plt.figure(figsize=(10, 5))
    plt.fill_between(time, controlled_band["q05"], controlled_band["q95"], color="tab:green", alpha=0.20)
    plt.plot(time, controlled_band["mean"], color="tab:green", linewidth=1.8, label="controlled alpha mean")
    plt.plot(time, uncontrolled_mean, color="tab:gray", linewidth=1.4, linestyle="--", label="uncontrolled alpha")
    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    plt.title("Pascucci control alpha")
    plt.xlabel("Time")
    plt.ylabel("alpha")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {
        "time": np.asarray(time, dtype=np.float32),
        "controlled_mean": controlled_band["mean"].astype(np.float32),
        "controlled_q05": controlled_band["q05"].astype(np.float32),
        "controlled_q50": controlled_band["q50"].astype(np.float32),
        "controlled_q95": controlled_band["q95"].astype(np.float32),
        "uncontrolled_mean": uncontrolled_mean.astype(np.float32),
    }


def _plot_state_bands(*, time: np.ndarray, X: np.ndarray, path: str) -> Dict[str, np.ndarray]:
    components = (
        (np.exp(np.clip(X[:, :, 0], -50.0, 50.0)).astype(np.float32), "S"),
        (X[:, :, 1], "H"),
        (X[:, :, 2], "V"),
        (X[:, :, 3], "X"),
    )
    n_paths = int(X.shape[0])
    sample_count = min(3, n_paths)
    sample_idx = np.arange(sample_count, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (values, label) in zip(axes.flatten(), components):
        arr = np.asarray(values, dtype=np.float32)
        q10 = np.quantile(arr, 0.10, axis=0).astype(np.float32)
        q90 = np.quantile(arr, 0.90, axis=0).astype(np.float32)
        mean = np.mean(arr, axis=0).astype(np.float32)
        ax.fill_between(time, q10, q90, alpha=0.2)
        ax.plot(time, mean, linewidth=2, label="Mean")
        for idx in sample_idx:
            ax.plot(time, arr[int(idx)], linestyle="--", alpha=0.7)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)

    data: Dict[str, np.ndarray] = {
        "time": np.asarray(time, dtype=np.float32),
        "sample_indices": sample_idx.astype(np.float32),
    }
    for values, label in components:
        arr = np.asarray(values, dtype=np.float32)
        data[f"{label}_mean"] = np.mean(arr, axis=0).astype(np.float32)
        data[f"{label}_q10"] = np.quantile(arr, 0.10, axis=0).astype(np.float32)
        data[f"{label}_q50"] = np.quantile(arr, 0.50, axis=0).astype(np.float32)
        data[f"{label}_q90"] = np.quantile(arr, 0.90, axis=0).astype(np.float32)
        data[f"{label}_samples"] = arr[sample_idx].astype(np.float32)
    return data


def _plot_y_component(*, time: np.ndarray, Y: np.ndarray, path: str) -> Dict[str, np.ndarray]:
    arr = _as_finite_array(Y, name="stitched.Y", ndim=3)[:, :, 0]
    if arr.shape[1] != np.asarray(time).shape[0]:
        raise ValueError(f"stitched.Y time dimension {arr.shape[1]} does not match t {np.asarray(time).shape[0]}")
    n_paths = int(arr.shape[0])
    sample_count = min(3, n_paths)
    sample_idx = np.arange(sample_count, dtype=np.int32)
    q10 = np.quantile(arr, 0.10, axis=0).astype(np.float32)
    q50 = np.quantile(arr, 0.50, axis=0).astype(np.float32)
    q90 = np.quantile(arr, 0.90, axis=0).astype(np.float32)
    mean = np.mean(arr, axis=0).astype(np.float32)

    plt.figure(figsize=(10, 6))
    plt.fill_between(time, q10, q90, alpha=0.2)
    plt.plot(time, mean, linewidth=2, label="Mean Y")
    for idx in sample_idx:
        plt.plot(time, arr[int(idx)], linestyle="--", alpha=0.7)
    plt.title("Backward Component Y")
    plt.xlabel("t")
    plt.ylabel("Y_t")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return {
        "time": np.asarray(time, dtype=np.float32),
        "sample_indices": sample_idx.astype(np.float32),
        "mean": mean,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "samples": arr[sample_idx].astype(np.float32),
    }


def _plot_z_components(*, time: np.ndarray, Z: np.ndarray, path: str) -> Dict[str, np.ndarray]:
    arr = _as_finite_array(Z, name="stitched.Z", ndim=3)
    if arr.shape[1] != np.asarray(time).shape[0]:
        raise ValueError(f"stitched.Z time dimension {arr.shape[1]} does not match t {np.asarray(time).shape[0]}")
    if arr.shape[2] < 4:
        raise ValueError(f"stitched.Z must contain at least 4 components, got {arr.shape}")
    n_paths = int(arr.shape[0])
    sample_count = min(3, n_paths)
    sample_idx = np.arange(sample_count, dtype=np.int32)
    labels = ("Z_S", "Z_H", "Z_V", "Z_X")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    data: Dict[str, np.ndarray] = {
        "time": np.asarray(time, dtype=np.float32),
        "sample_indices": sample_idx.astype(np.float32),
    }
    for ax, label, component in zip(axes.flatten(), labels, range(4)):
        values = arr[:, :, component].astype(np.float32)
        q10 = np.quantile(values, 0.10, axis=0).astype(np.float32)
        q50 = np.quantile(values, 0.50, axis=0).astype(np.float32)
        q90 = np.quantile(values, 0.90, axis=0).astype(np.float32)
        mean = np.mean(values, axis=0).astype(np.float32)
        ax.fill_between(time, q10, q90, alpha=0.2)
        ax.plot(time, mean, linewidth=2, label="Mean")
        for idx in sample_idx:
            ax.plot(time, values[int(idx)], linestyle="--", alpha=0.7)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        data[f"{label}_mean"] = mean
        data[f"{label}_q10"] = q10
        data[f"{label}_q50"] = q50
        data[f"{label}_q90"] = q90
        data[f"{label}_samples"] = values[sample_idx].astype(np.float32)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return data


def _plot_controlled_uncontrolled(
    *,
    controlled_total: np.ndarray,
    uncontrolled_total: np.ndarray,
    path: str,
    uncontrolled_label: str = "uncontrolled",
    title: str = "Pascucci controlled vs uncontrolled total cost distribution",
) -> Dict[str, np.ndarray]:
    controlled_flat = np.asarray(controlled_total, dtype=np.float32).reshape(-1)
    uncontrolled_flat = np.asarray(uncontrolled_total, dtype=np.float32).reshape(-1)
    values = [controlled_flat, uncontrolled_flat]
    means = [float(np.mean(v)) for v in values]
    q05 = [float(np.quantile(v, 0.05)) for v in values]
    q50 = [float(np.quantile(v, 0.50)) for v in values]
    q95 = [float(np.quantile(v, 0.95)) for v in values]
    yerr = np.asarray(
        [[q50[i] - q05[i] for i in range(2)], [q95[i] - q50[i] for i in range(2)]],
        dtype=np.float32,
    )
    plt.figure(figsize=(8, 5))
    x = np.arange(2)
    plt.bar(x, q50, yerr=yerr, capsize=5, color=["tab:blue", "tab:red"], alpha=0.82, label="median q05-q95")
    plt.scatter(x, means, color="black", marker="D", s=28, zorder=3, label="mean")
    plt.xticks(x, ["controlled", str(uncontrolled_label)])
    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    plt.title(str(title))
    plt.ylabel("J total median with q05-q95; diamonds show mean")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {
        "controlled_values": controlled_flat.astype(np.float32),
        "uncontrolled_values": uncontrolled_flat.astype(np.float32),
        "controlled_mean": np.asarray([means[0]], dtype=np.float32),
        "uncontrolled_mean": np.asarray([means[1]], dtype=np.float32),
        "controlled_q05": np.asarray([q05[0]], dtype=np.float32),
        "uncontrolled_q05": np.asarray([q05[1]], dtype=np.float32),
        "controlled_q50": np.asarray([q50[0]], dtype=np.float32),
        "uncontrolled_q50": np.asarray([q50[1]], dtype=np.float32),
        "controlled_q95": np.asarray([q95[0]], dtype=np.float32),
        "uncontrolled_q95": np.asarray([q95[1]], dtype=np.float32),
    }


def _story_entry(
    filename: str,
    title: str,
    inputs: Iterable[str],
    *,
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "filename": filename,
        "path": filename,
        "path_relative_to": "manifest_dir",
        "title": title,
        "inputs": list(inputs),
    }
    if style is not None:
        entry["style"] = dict(style)
    return entry


def _story_path(entry: Dict[str, Any], out_dir: str) -> str:
    return os.path.join(out_dir, str(entry["path"]))


def _visual_regression_contract() -> Dict[str, Any]:
    return {
        "status": "structural_style_only_no_golden_images",
        "structural_style_contract": True,
        "pixel_exact_claim": False,
        "golden_images_available": False,
        "golden_image_source": "raw/to_ema has plotmaker.ipynb but no embedded or saved golden PNG/PDF images",
        "comparison_method": "manifest_style_contract_plus_numeric_plot_data_recomputation",
        "remaining_gaps": [
            "no golden image hashes from an executed raw/to_ema/plotmaker.ipynb run",
            "plotmaker sample paths use np.random.choice without a local seed; final_recursive records deterministic first3 samples for reproducibility",
            "#38 and #40 remain final_recursive diagnostics without direct plotmaker figure equivalents",
        ],
    }


def _flatten_plot_data(plot_data_by_story: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    flattened: Dict[str, np.ndarray] = {}
    for story, payload in plot_data_by_story.items():
        prefix = f"plot{str(story).lstrip('#')}"
        for key, value in payload.items():
            flattened[f"{prefix}_{key}"] = np.asarray(value, dtype=np.float32)
    return flattened


def _prefix_plot_data(prefix: str, payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {f"{prefix}_{key}": np.asarray(value, dtype=np.float32) for key, value in payload.items()}


def plot_pascucci_paper_bundle(
    *,
    stitched: Dict[str, Any],
    application_pathwise: Dict[str, Any],
    params: Dict[str, Any],
    out_dir: str,
    blocks: Optional[list] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render smoke-testable Pascucci paper plots #35-#40 from saved artifacts."""

    _require_plotting()
    time, X = _validate_stitched(stitched)
    Y = _as_finite_array(stitched.get("Y"), name="stitched.Y", ndim=3)
    Z = _as_finite_array(stitched.get("Z"), name="stitched.Z", ndim=3)
    if Y.shape[0] != X.shape[0] or Y.shape[1] != X.shape[1] or Y.shape[2] != 1:
        raise ValueError(f"stitched.Y shape {Y.shape} is incompatible with stitched.X {X.shape}")
    if Z.shape[0] != X.shape[0] or Z.shape[1] != X.shape[1] or Z.shape[2] < 4:
        raise ValueError(f"stitched.Z shape {Z.shape} is incompatible with stitched.X {X.shape}")
    params_S = dict(params.get("params_S", {}))
    params_H = dict(params.get("params_H", {}))
    if not params_S or not params_H:
        raise ValueError("Pascucci paper plots require params_S and params_H in run config params")
    ou_references, ou_reference_summary = _load_to_ema_ou_reference(params, horizon_T=float(time[-1]))

    controlled_alpha = _require_pathwise(application_pathwise, "controlled_alpha", ndim=3)
    uncontrolled_alpha = _require_pathwise(application_pathwise, "uncontrolled_alpha", ndim=3)
    controlled_cumulative = _require_pathwise(
        application_pathwise,
        "controlled_cost_J_running_cumulative",
        ndim=3,
    )
    uncontrolled_cumulative = _require_pathwise(
        application_pathwise,
        "uncontrolled_cost_J_running_cumulative",
        ndim=3,
    )
    controlled_total = _require_pathwise(application_pathwise, "controlled_cost_J_total", ndim=2)
    uncontrolled_total = _require_pathwise(application_pathwise, "uncontrolled_cost_J_total", ndim=2)
    if "controlled_cost_J_trajectory" in application_pathwise or "uncontrolled_cost_J_trajectory" in application_pathwise:
        controlled_cost_trace = _require_pathwise(application_pathwise, "controlled_cost_J_trajectory", ndim=3)
        uncontrolled_cost_trace = _require_pathwise(application_pathwise, "uncontrolled_cost_J_trajectory", ndim=3)
        cost_trace_time = time
        cost_trace_source = "cost_J_trajectory"
    else:
        controlled_cost_trace = controlled_cumulative
        uncontrolled_cost_trace = uncontrolled_cumulative
        cost_trace_time = time[1:]
        cost_trace_source = "cost_J_running_cumulative"
    steps = time.shape[0] - 1
    for name, arr in (
        ("controlled_alpha", controlled_alpha),
        ("uncontrolled_alpha", uncontrolled_alpha),
        ("controlled_cost_J_running_cumulative", controlled_cumulative),
        ("uncontrolled_cost_J_running_cumulative", uncontrolled_cumulative),
    ):
        if arr.shape[0] != X.shape[0] or arr.shape[1] != steps:
            raise ValueError(f"{name} must have shape (M, n_steps, 1), got {arr.shape}")
    if cost_trace_source == "cost_J_trajectory":
        for name, arr in (
            ("controlled_cost_J_trajectory", controlled_cost_trace),
            ("uncontrolled_cost_J_trajectory", uncontrolled_cost_trace),
        ):
            if arr.shape[0] != X.shape[0] or arr.shape[1] != time.shape[0]:
                raise ValueError(f"{name} must have shape (M, n_time_points, 1), got {arr.shape}")
    if controlled_total.shape[0] != X.shape[0] or uncontrolled_total.shape[0] != X.shape[0]:
        raise ValueError("controlled/uncontrolled total costs must match stitched path count")

    to_ema_uncontrolled, to_ema_uncontrolled_summary = _load_to_ema_uncontrolled_reference(
        params,
        horizon_T=float(time[-1]),
    )
    uncontrolled_plot_trace = uncontrolled_cost_trace
    uncontrolled_plot_time = cost_trace_time
    uncontrolled_cost_trace_source = f"application_metrics.uncontrolled_{cost_trace_source}"
    uncontrolled_total_for_plot = uncontrolled_total
    uncontrolled_total_source = "alpha_zero_same_model"
    uncontrolled_total_label = "alpha=0"
    uncontrolled_total_title = "Controlled vs alpha=0 same-model total cost"
    if (
        cost_trace_source == "cost_J_trajectory"
        and to_ema_uncontrolled_summary.get("source") == "calibration_metadata"
    ):
        uncontrolled_plot_trace = to_ema_uncontrolled["J_paths"][:, :, None]
        uncontrolled_plot_time = to_ema_uncontrolled["time"]
        uncontrolled_cost_trace_source = "to_ema_raw_uncontrolled_J"
        uncontrolled_total_for_plot = to_ema_uncontrolled["J_paths"][:, -1:]
        uncontrolled_total_source = "to_ema_raw_uncontrolled_J_final"
        uncontrolled_total_label = "raw benchmark"
        uncontrolled_total_title = "Controlled vs raw to_ema final cost"

    os.makedirs(out_dir, exist_ok=True)
    plot_data_by_story: Dict[str, Dict[str, np.ndarray]] = {}
    plotmaker_native_plots = {
        "Y": _story_entry(
            PLOTMAKER_NATIVE_Y_FILENAME,
            "Backward Component Y",
            (
                "stitched.t",
                "stitched.Y[:, :, 0]",
                "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
            ),
            style={
                "source": "raw/to_ema/plotmaker.ipynb::Backward Component Y cell",
                "figure_size": [10, 6],
                "band_alpha": 0.2,
                "quantiles": [0.10, 0.90],
                "sample_paths": 3,
                "sample_policy": "deterministic_first3_for_reproducibility",
                "title": "Backward Component Y",
                "xlabel": "t",
                "ylabel": "Y_t",
            },
        ),
        "Z": _story_entry(
            PLOTMAKER_NATIVE_Z_FILENAME,
            "Z components",
            (
                "stitched.t",
                "stitched.Z[:, :, Z_S,Z_H,Z_V,Z_X]",
                "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
            ),
            style={
                "source": "raw/to_ema/plotmaker.ipynb::Z components cell",
                "figure_size": [14, 10],
                "layout": "2x2",
                "labels": ["Z_S", "Z_H", "Z_V", "Z_X"],
                "band_alpha": 0.2,
                "quantiles": [0.10, 0.90],
                "sample_paths": 3,
                "sample_policy": "deterministic_first3_for_reproducibility",
            },
        ),
    }
    plot_specs = {
        "#35": _story_entry(
            "pascucci_paper_35_S_ou_band.png",
            "PUN price with OU envelope",
            (
                "stitched.t",
                "exp(stitched.X[:, :, S]) * 1000",
                "exp(run_config.params.params_S OU quantities) * 1000",
            ),
            style={
                "source": "raw/to_ema/variable_mu_calibration.ipynb::generate_plot",
                "formula_source": "raw/to_ema/calibration.py",
                "figure_size": [12, 6],
                "real_color": "red",
                "band_alpha": 0.3,
                "band_label": "80% Band",
                "quantiles": [0.05, 0.95],
                "xlabel": "Hours",
            },
        ),
        "#36": _story_entry(
            "pascucci_paper_36_H_ou_band.png",
            "H with OU envelope",
            ("stitched.t", "stitched.X[:, :, H]", "run_config.params.params_H"),
            style={
                "source": "raw/to_ema/variable_mu_calibration.ipynb::generate_plot",
                "formula_source": "raw/to_ema/calibration.py",
                "figure_size": [12, 6],
                "real_color": "red",
                "band_alpha": 0.3,
                "band_label": "80% Band",
                "quantiles": [0.05, 0.95],
                "xlabel": "Hours",
            },
        ),
        "#37": _story_entry(
            "pascucci_paper_37_accumulated_cost.png",
            "Comparison of J_t",
            (
                "application_metrics.controlled_cost_J_trajectory",
                "application_metrics.uncontrolled_cost_J_trajectory",
            ),
            style={
                "source": "raw/to_ema/plotmaker.ipynb::comparison J cell",
                "figure_size": [10, 6],
                "controlled_color": "r",
                "uncontrolled_color": "b",
                "band_alpha": 0.2,
                "quantiles": [0.10, 0.90],
                "title": r"Comparison of $J_t= \int_0^t f_s ds + g(X_t)$",
                "xlabel": "Time(h)",
                "ylabel": "Cost",
            },
        ),
        "#38": _story_entry(
            "pascucci_paper_38_alpha.png",
            "Control alpha",
            ("application_metrics.controlled_alpha", "application_metrics.uncontrolled_alpha"),
            style={
                "source": "final_recursive.application_metrics_alpha",
                "figure_size": [10, 5],
                "controlled_color": "tab:green",
                "uncontrolled_color": "tab:gray",
                "uncontrolled_linestyle": "--",
                "zero_line": True,
                "band_alpha": 0.20,
                "quantiles": [0.05, 0.50, 0.95],
                "xlabel": "Time",
                "ylabel": "alpha",
            },
        ),
        "#39": _story_entry(
            "pascucci_paper_39_forward_components_S_H_V_X.png",
            "Forward components S,H,V,X",
            (
                "stitched.t",
                "exp(stitched.X[:, :, S])",
                "stitched.X[:, :, H,V,X]",
                "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
            ),
            style={
                "source": "raw/to_ema/plotmaker.ipynb::forward_components",
                "figure_size": [14, 10],
                "layout": "2x2",
                "band_alpha": 0.2,
                "quantiles": [0.10, 0.90],
                "sample_paths": 3,
                "sample_policy": "deterministic_first3_for_reproducibility",
            },
        ),
        "#40": _story_entry(
            "pascucci_paper_40_controlled_uncontrolled.png",
            uncontrolled_total_title,
            (
                "application_metrics.controlled_cost_J_total",
                "application_metrics.uncontrolled_cost_J_total",
                "uncontrolled baseline source=alpha_zero_same_model",
            ),
            style={
                "source": "final_recursive.total_cost_distribution_summary",
                "figure_size": [8, 5],
                "bar_stat": "q50",
                "marker_stat": "mean",
                "marker": "D",
                "interval": "q05-q95",
                "controlled_color": "tab:blue",
                "uncontrolled_color": "tab:red",
                "ylabel": "J total median with q05-q95; diamonds show mean",
            },
        ),
    }
    if ou_reference_summary.get("source") == "calibration_metadata":
        plot_specs["#35"]["inputs"] = [
            "pascucci_data.prepare_S(source_path, n, mul_factor)",
            "run_config.params.params_S",
            "simulate_ou_day_night(logS0=log(real_S[0]), dt=0.5, Nsim=10000, seed=42)",
            "exp(simulated_logS) * 1000",
            "real_S * 1000",
        ]
        plot_specs["#36"]["inputs"] = [
            "pascucci_data.prepare_H(source_path, n, mul_factor)",
            "run_config.params.params_H",
            "simulate_ou_day_night(H0=real_H[0], dt=0.5, Nsim=10000, seed=42)",
            "real_H",
        ]
    if cost_trace_source == "cost_J_running_cumulative":
        plot_specs["#37"]["title"] = "Accumulated running cost"
        plot_specs["#37"]["inputs"] = [
            "application_metrics.controlled_cost_J_running_cumulative",
            "application_metrics.uncontrolled_cost_J_running_cumulative",
        ]
    elif uncontrolled_cost_trace_source == "to_ema_raw_uncontrolled_J":
        plot_specs["#37"]["inputs"] = [
            "application_metrics.controlled_cost_J_trajectory",
            "to_ema.raw_uncontrolled_J_paths",
            "to_ema.simulate_x_v_paths_uncontrolled",
            "to_ema.simulate_uncontrolled_J(dt=0.1, Nsim=10000, seed=42)",
        ]
        plot_specs["#40"]["inputs"] = [
            "application_metrics.controlled_cost_J_total",
            "to_ema.raw_uncontrolled_J_paths[:, -1]",
            "paired alpha=0 total still saved as application_metrics.uncontrolled_cost_J_total",
            "uncontrolled total source=to_ema_raw_uncontrolled_J_final",
        ]

    if ou_reference_summary.get("source") == "calibration_metadata":
        plot_data_by_story["#35"] = _plot_to_ema_ou_simulation(
            paths=ou_references["#35"]["paths"],
            real_path=ou_references["#35"]["real_path"],
            T=float(ou_references["#35"]["T"]),
            name=str(ou_references["#35"]["name"]),
            ylabel=str(ou_references["#35"]["ylabel"]),
            path=_story_path(plot_specs["#35"], out_dir),
            dt_real=float(ou_reference_summary["dt_real"]),
            dt_sim=float(ou_reference_summary["simulation"]["dt_sim"]),
        )
        plot_data_by_story["#36"] = _plot_to_ema_ou_simulation(
            paths=ou_references["#36"]["paths"],
            real_path=ou_references["#36"]["real_path"],
            T=float(ou_references["#36"]["T"]),
            name=str(ou_references["#36"]["name"]),
            ylabel=str(ou_references["#36"]["ylabel"]),
            path=_story_path(plot_specs["#36"], out_dir),
            dt_real=float(ou_reference_summary["dt_real"]),
            dt_sim=float(ou_reference_summary["simulation"]["dt_sim"]),
        )
    else:
        plot_data_by_story["#35"] = _plot_state_with_ou(
            time=time,
            values=X[:, :, 0],
            ou_params=params_S,
            ylabel="PUN price (EUR/MWh)",
            title="Pascucci PUN price with calibrated OU envelope",
            path=_story_path(plot_specs["#35"], out_dir),
            transform=_pascucci_price_eur_mwh,
        )
        plot_data_by_story["#36"] = _plot_state_with_ou(
            time=time,
            values=X[:, :, 1],
            ou_params=params_H,
            ylabel="H",
            title="Pascucci H with calibrated OU envelope",
            path=_story_path(plot_specs["#36"], out_dir),
        )
    plot_data_by_story["#37"] = _plot_accumulated_cost(
        time=cost_trace_time,
        controlled=controlled_cost_trace,
        uncontrolled=uncontrolled_plot_trace,
        trace_source=cost_trace_source,
        uncontrolled_time=uncontrolled_plot_time,
        path=_story_path(plot_specs["#37"], out_dir),
    )
    plot_data_by_story["#38"] = _plot_alpha(
        time=time[:-1],
        controlled_alpha=controlled_alpha,
        uncontrolled_alpha=uncontrolled_alpha,
        path=_story_path(plot_specs["#38"], out_dir),
    )
    plot_data_by_story["#39"] = _plot_state_bands(time=time, X=X, path=_story_path(plot_specs["#39"], out_dir))
    plot_data_by_story["#40"] = _plot_controlled_uncontrolled(
        controlled_total=controlled_total,
        uncontrolled_total=uncontrolled_total_for_plot,
        path=_story_path(plot_specs["#40"], out_dir),
        uncontrolled_label=uncontrolled_total_label,
        title=uncontrolled_total_title,
    )
    plot_data = _flatten_plot_data(plot_data_by_story)
    plot_data.update(
        _prefix_plot_data(
            "plotmaker_Y",
            _plot_y_component(time=time, Y=Y, path=_story_path(plotmaker_native_plots["Y"], out_dir)),
        )
    )
    plot_data.update(
        _prefix_plot_data(
            "plotmaker_Z",
            _plot_z_components(time=time, Z=Z, path=_story_path(plotmaker_native_plots["Z"], out_dir)),
        )
    )
    save_blob_npz(plot_data, os.path.join(out_dir, PAPER_PLOT_DATA_NPZ))

    source = dict(source_metadata or {})
    manifest = {
        "schema": PAPER_PLOT_SCHEMA,
        "model_name": str(source.get("model_name", "pascucci")),
        "plots": plot_specs,
        "plot_count": int(len(plot_specs)),
        "plotmaker_native_plots": plotmaker_native_plots,
        "plot_data": {
            "schema": PAPER_PLOT_DATA_SCHEMA,
            "path": PAPER_PLOT_DATA_NPZ,
            "path_relative_to": "manifest_dir",
            "keys": sorted(plot_data.keys()),
            "purpose": "numeric series and statistics used to render paper plots #35-#40",
        },
        "source": source,
        "horizon": {
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "n_time_points": int(time.shape[0]),
            "n_steps": int(steps),
            "sample_paths": int(X.shape[0]),
        },
        "state_column_map": {"S": 0, "H": 1, "V": 2, "X": 3},
        "cost_trace_source": cost_trace_source,
        "uncontrolled_cost_trace_source": uncontrolled_cost_trace_source,
        "to_ema_uncontrolled_reference": to_ema_uncontrolled_summary,
        "plotmaker_reference": _plotmaker_reference_summary(params),
        "comparison_sources": {
            "#37": {
                "controlled": "application_metrics.controlled_cost_J_trajectory",
                "uncontrolled": uncontrolled_cost_trace_source,
            },
            "#40": {
                "controlled": "application_metrics.controlled_cost_J_total",
                "uncontrolled": uncontrolled_total_source,
                "uncontrolled_detail": (
                    "to_ema.raw_uncontrolled_J_paths[:, -1]"
                    if uncontrolled_total_source == "to_ema_raw_uncontrolled_J_final"
                    else "application_metrics.uncontrolled_cost_J_total"
                ),
                "paired_alpha_zero_detail": "application_metrics.uncontrolled_cost_J_total",
            },
        },
        "notebook_parity": {
            "exact_all_stories": False,
            "reason": "#38 and #40 remain final_recursive diagnostic summaries in the historical #35-#40 contract; native plotmaker Y/Z figures are emitted separately under plotmaker_native_plots. #37 is oracled against the per-time plotmaker formula; the raw flattened helper is not used because it mixes mean-field moments across time.",
            "stories": {
                "#35": "to_ema_calibration_reference",
                "#36": "to_ema_calibration_reference",
                "#37": "plotmaker_per_time_cost_formula_oracled",
                "#38": "diagnostic_only_no_plotmaker_equivalent",
                "#39": "plotmaker_forward_components_reference",
                "#40": "diagnostic_only_no_plotmaker_equivalent",
            },
            "native_plotmaker_plots": {
                "Y": "plotmaker_backward_component_reference",
                "Z": "plotmaker_z_components_reference",
            },
        },
        "visual_regression": _visual_regression_contract(),
        "controlled_uncontrolled_available": True,
        "plot_path_policy": "relative_to_manifest_dir",
        "state_transforms": {
            "#35": "PUN price = exp(S) * 1000, matching to_ema calibration plots",
            "#39": "forward components S,H,V,X in plotmaker 2x2 layout; S is plotted as exp(S)",
        },
        "ou_reference": ou_reference_summary,
        "blocks": blocks or [],
    }
    save_json(manifest, os.path.join(out_dir, PAPER_PLOT_MANIFEST))
    return manifest


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_application_metric_schema(run_config: Dict[str, Any]) -> None:
    schema = str(run_config.get("application_metric_schema", ""))
    if schema != APPLICATION_METRIC_SCHEMA:
        raise ValueError(
            "Pascucci paper plots require run_config "
            f"application_metric_schema='{APPLICATION_METRIC_SCHEMA}', got {schema!r}"
        )


def plot_pascucci_paper_bundle_from_artifacts(
    *,
    stitched_npz_path: str,
    application_npz_path: str,
    run_config_path: str,
    out_dir: str,
    blocks: Optional[list] = None,
    source_label: str = "artifact",
) -> Dict[str, Any]:
    """Load saved run artifacts and render Pascucci paper plots #35-#40."""

    stitched_path = os.path.abspath(os.path.expanduser(stitched_npz_path))
    application_path = os.path.abspath(os.path.expanduser(application_npz_path))
    config_path = os.path.abspath(os.path.expanduser(run_config_path))
    run_config = _load_json(config_path)
    model_name = str(run_config.get("model_name", ""))
    if model_name != "pascucci":
        raise ValueError(f"Pascucci paper plots require run_config model_name='pascucci', got {model_name!r}")
    _require_application_metric_schema(run_config)
    artifact_blocks = blocks
    if artifact_blocks is None:
        artifact_blocks = run_config.get("blocks", None)
    if artifact_blocks is None:
        results_path = os.path.join(str(Path(application_path).parent), "results.json")
        if os.path.exists(results_path):
            artifact_blocks = _load_json(results_path).get("blocks", [])
    source = {
        "source_label": str(source_label),
        "run_dir": str(Path(config_path).parent),
        "run_config_path": config_path,
        "stitched_npz_path": stitched_path,
        "application_npz_path": application_path,
        "run_config_sha256": run_config.get("run_config_sha256", ""),
        "seed_manifest": run_config.get("seed_manifest", {}),
        "application_metric_schema": run_config.get("application_metric_schema", ""),
        "state_labels": run_config.get("state_labels", []),
        "z_labels": run_config.get("z_labels", []),
        "model_name": model_name,
    }
    return plot_pascucci_paper_bundle(
        stitched=_as_blob_dict(stitched_path) or {},
        application_pathwise=_as_blob_dict(application_path) or {},
        params=dict(run_config.get("params", {})),
        out_dir=os.path.abspath(os.path.expanduser(out_dir)),
        blocks=artifact_blocks or [],
        source_metadata=source,
    )
