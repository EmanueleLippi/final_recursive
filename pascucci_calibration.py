"""OU calibration helpers for Pascucci model inputs."""

from __future__ import annotations

from typing import Dict

import numpy as np


OU_PARAM_KEYS = {
    "kappa_day",
    "kappa_night",
    "a0_day",
    "a0_night",
    "alpha_day",
    "alpha_night",
    "beta_day",
    "beta_night",
    "sigma_day",
    "sigma_night",
}
HARMONIC_KEYS = ("alpha_day", "alpha_night", "beta_day", "beta_night")
SCALAR_KEYS = tuple(sorted(OU_PARAM_KEYS - set(HARMONIC_KEYS)))


def _validate_k_dt(K: int, dt: float) -> tuple[int, float]:
    K_int = int(K)
    dt_float = float(dt)
    if K_int < 0:
        raise ValueError("K must be non-negative")
    if not np.isfinite(dt_float) or dt_float <= 0.0:
        raise ValueError("dt must be a positive finite value")
    return K_int, dt_float


def is_day(t) -> np.ndarray:
    hour = np.mod(np.asarray(t, dtype=np.float64), 24.0)
    return (hour >= 7.0) & (hour < 19.0)


def _design_matrix(values: np.ndarray, t: np.ndarray, K: int) -> np.ndarray:
    day = is_day(t).astype(np.float64)
    night = 1.0 - day
    columns = [
        -values * day,
        -values * night,
        day,
        night,
    ]
    for k in range(1, int(K) + 1):
        omega = 2.0 * np.pi * float(k) / 24.0
        cos = np.cos(omega * t)
        sin = np.sin(omega * t)
        columns.extend([cos * day, cos * night, sin * day, sin * night])
    return np.column_stack(columns)


def _regime_design_matrix(values: np.ndarray, t: np.ndarray, K: int, mask: np.ndarray) -> np.ndarray:
    values_regime = values[mask]
    t_regime = t[mask]
    columns = [-values_regime, np.ones_like(values_regime)]
    for k in range(1, int(K) + 1):
        omega = 2.0 * np.pi * float(k) / 24.0
        columns.extend([np.cos(omega * t_regime), np.sin(omega * t_regime)])
    return np.column_stack(columns)


def _validate_regime_design(
    values: np.ndarray,
    t: np.ndarray,
    K: int,
    mask: np.ndarray,
    label: str,
    condition_max: float,
) -> int:
    p_regime = 2 + 2 * int(K)
    n_regime = int(np.sum(mask))
    if n_regime <= p_regime:
        raise ValueError(
            f"Not enough {label} observations for residual degrees of freedom: "
            f"{n_regime} observations for {p_regime} regime parameters"
        )
    X_regime = _regime_design_matrix(values, t, K, mask)
    rank = np.linalg.matrix_rank(X_regime)
    if rank < p_regime:
        raise ValueError(f"Rank deficient {label} OU regression: rank {rank} for {p_regime} parameters")
    condition = float(np.linalg.cond(X_regime))
    if not np.isfinite(condition) or condition > float(condition_max):
        raise ValueError(f"Ill-conditioned {label} OU regression: condition={condition:.3e}")
    return n_regime - p_regime


def _residual_sigma(residuals: np.ndarray, mask: np.ndarray, dt: float, dof: int, label: str) -> float:
    selected = residuals[mask]
    if int(dof) <= 0:
        raise ValueError(f"Not enough {label} residual degrees of freedom to estimate sigma")
    sse = float(np.sum(selected * selected))
    sigma = float(np.sqrt(max(sse, 0.0) / float(dof) / float(dt)))
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"Invalid {label} sigma estimate")
    return sigma


def validate_ou_params(params: Dict[str, object], K: int) -> None:
    """Validate the params_S/params_H schema consumed by the Pascucci model."""

    K_int, _ = _validate_k_dt(K, 1.0)
    keys = set(params)
    missing = OU_PARAM_KEYS - keys
    extra = keys - OU_PARAM_KEYS
    if missing:
        raise ValueError(f"OU params missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"OU params contain unexpected keys: {', '.join(sorted(extra))}")

    for key in HARMONIC_KEYS:
        value = np.asarray(params[key])
        if value.shape != (K_int,):
            raise ValueError(f"{key} must have shape ({K_int},), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must contain finite values")

    for key in SCALAR_KEYS:
        value = np.asarray(params[key])
        if value.shape != ():
            raise ValueError(f"{key} must be a scalar")
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{key} must be finite")

    for key in ("kappa_day", "kappa_night"):
        if float(params[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")
    for key in ("sigma_day", "sigma_night"):
        if float(params[key]) < 0.0:
            raise ValueError(f"{key} must be non-negative")
    return None


def calibrate_OU_variable(
    values,
    K: int,
    dt: float = 1.0,
    start_hour: float = 0.0,
    kappa_min: float = 1.0e-8,
    condition_max: float = 1.0e10,
) -> Dict[str, object]:
    """Fit day/night OU drift and diffusion parameters by least squares."""

    K_int, dt_float = _validate_k_dt(K, dt)
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.shape[0] < 2:
        raise ValueError("calibrate_OU_variable requires at least two observations")
    if not np.all(np.isfinite(series)):
        raise ValueError("calibrate_OU_variable values must be finite")

    Y = series[1:] - series[:-1]
    t = float(start_hour) + np.arange(Y.shape[0], dtype=np.float64) * dt_float
    day_idx = is_day(t)
    night_idx = ~day_idx
    if not np.any(day_idx):
        raise ValueError("calibrate_OU_variable requires at least one day observation")
    if not np.any(night_idx):
        raise ValueError("calibrate_OU_variable requires at least one night observation")
    if not np.isfinite(float(kappa_min)) or float(kappa_min) <= 0.0:
        raise ValueError("kappa_min must be a positive finite value")
    if not np.isfinite(float(condition_max)) or float(condition_max) <= 0.0:
        raise ValueError("condition_max must be a positive finite value")

    X = _design_matrix(series[:-1], t, K_int)
    n_obs, n_cols = X.shape
    if n_obs < n_cols:
        raise ValueError(f"Underdetermined OU regression: {n_obs} rows for {n_cols} parameters")
    rank = np.linalg.matrix_rank(X)
    if rank < n_cols:
        raise ValueError(f"Rank deficient OU regression: rank {rank} for {n_cols} parameters")
    condition = float(np.linalg.cond(X))
    if not np.isfinite(condition) or condition > float(condition_max):
        raise ValueError(f"Ill-conditioned OU regression: condition={condition:.3e}")
    day_dof = _validate_regime_design(series[:-1], t, K_int, day_idx, "day", float(condition_max))
    night_dof = _validate_regime_design(series[:-1], t, K_int, night_idx, "night", float(condition_max))

    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    theta_day = float(theta[0])
    theta_night = float(theta[1])
    kappa_day = theta_day / dt_float
    kappa_night = theta_night / dt_float
    if kappa_day <= float(kappa_min):
        raise ValueError("kappa_day must be positive and mean-reverting")
    if kappa_night <= float(kappa_min):
        raise ValueError("kappa_night must be positive and mean-reverting")

    params: Dict[str, object] = {
        "kappa_day": np.float64(kappa_day),
        "kappa_night": np.float64(kappa_night),
        "a0_day": np.float64(theta[2] / theta_day),
        "a0_night": np.float64(theta[3] / theta_night),
        "alpha_day": np.zeros(K_int, dtype=np.float64),
        "alpha_night": np.zeros(K_int, dtype=np.float64),
        "beta_day": np.zeros(K_int, dtype=np.float64),
        "beta_night": np.zeros(K_int, dtype=np.float64),
        "sigma_day": np.float64(0.0),
        "sigma_night": np.float64(0.0),
    }
    if K_int:
        theta_k = theta[4 : 4 + 4 * K_int].reshape(K_int, 4)
        params["alpha_day"] = theta_k[:, 0] / theta_day
        params["alpha_night"] = theta_k[:, 1] / theta_night
        params["beta_day"] = theta_k[:, 2] / theta_day
        params["beta_night"] = theta_k[:, 3] / theta_night

    residuals = Y - X @ theta
    params["sigma_day"] = np.float64(_residual_sigma(residuals, day_idx, dt_float, day_dof, "day"))
    params["sigma_night"] = np.float64(_residual_sigma(residuals, night_idx, dt_float, night_dof, "night"))
    validate_ou_params(params, K_int)
    return params


def calibrate_pascucci_ou_inputs(
    H_series,
    S_series,
    K: int,
    dt: float = 1.0,
    start_hour: float = 0.0,
    log_price: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Calibrate Pascucci H and S OU parameter dictionaries."""

    H = np.asarray(H_series, dtype=np.float64).reshape(-1)
    S = np.asarray(S_series, dtype=np.float64).reshape(-1)
    if H.shape != S.shape:
        raise ValueError("H_series and S_series must have the same shape")
    if not np.all(np.isfinite(H)):
        raise ValueError("H_series must contain finite values")
    if not np.all(np.isfinite(S)):
        raise ValueError("S_series must contain finite values")

    if bool(log_price):
        if np.any(S <= 0.0):
            raise ValueError("log_price=True requires strictly positive S prices")
        S_for_calibration = np.log(S)
    else:
        S_for_calibration = S

    params_H = calibrate_OU_variable(H, K=K, dt=dt, start_hour=start_hour)
    params_S = calibrate_OU_variable(S_for_calibration, K=K, dt=dt, start_hour=start_hour)
    return {"params_H": params_H, "params_S": params_S}