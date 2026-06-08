"""Independent NumPy oracle formulas for Pascucci equation-level tests."""

from __future__ import annotations

from typing import Any

import numpy as np


ORACLE_VERSION = 1
DEFAULT_TOLERANCES = {"rtol": 1.0e-5, "atol": 1.0e-6}
HISTORICAL_REFERENCES = ["final_model3.py", "final_model_modifiche_f.py", "calibration.py"]


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, found {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _scalar(params: dict[str, Any], key: str) -> float:
    return float(np.asarray(params[key], dtype=np.float32))


def _day_mask(t: np.ndarray) -> np.ndarray:
    hour = np.mod(t, 24.0)
    return (hour >= 7.0) & (hour < 19.0)


def _regime_switch(t: np.ndarray, day_value: np.ndarray | float, night_value: np.ndarray | float) -> np.ndarray:
    day = np.broadcast_to(np.asarray(day_value, dtype=np.float32), t.shape)
    night = np.broadcast_to(np.asarray(night_value, dtype=np.float32), t.shape)
    return np.where(_day_mask(t), day, night).astype(np.float32)


def _ou_omega(params: dict[str, Any]) -> np.ndarray:
    k = int(np.asarray(params["alpha_day"], dtype=np.float32).shape[0])
    return (2.0 * np.pi * np.arange(1, k + 1, dtype=np.float32) / 24.0).reshape(1, -1)


def _seasonal_mean(t: np.ndarray, params: dict[str, Any], *, regime: str) -> np.ndarray:
    omega = _ou_omega(params)
    t_flat = t.reshape(-1, 1)
    alpha = np.asarray(params[f"alpha_{regime}"], dtype=np.float32).reshape(1, -1)
    beta = np.asarray(params[f"beta_{regime}"], dtype=np.float32).reshape(1, -1)
    seasonal = np.sum(alpha * np.cos(omega * t_flat) + beta * np.sin(omega * t_flat), axis=1, keepdims=True)
    return (_scalar(params, f"a0_{regime}") + seasonal).astype(np.float32)


def _mu_daynight(t: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return _regime_switch(
        t,
        _seasonal_mean(t, params, regime="day"),
        _seasonal_mean(t, params, regime="night"),
    )


def _kappa_daynight(t: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return _regime_switch(t, _scalar(params, "kappa_day"), _scalar(params, "kappa_night"))


def _sigma_daynight(t: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return _regime_switch(t, _scalar(params, "sigma_day"), _scalar(params, "sigma_night"))


def _psi(q: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    d = _scalar(params, "d")
    x_max = _scalar(params, "x_max")
    return np.maximum(0.0, np.minimum(1.0, np.minimum(q / d, (x_max - q) / d))).astype(np.float32)


def _psi1(q: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (q - _scalar(params, "x_max")) / _scalar(params, "d"))).astype(np.float32)


def _psi2(q: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, -q / _scalar(params, "d"))).astype(np.float32)


def _psi3(v: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (_scalar(params, "v_max") - v) / _scalar(params, "d"))).astype(np.float32)


def _psi4(v: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (v - _scalar(params, "v_min")) / _scalar(params, "d"))).astype(np.float32)


def _h_with_mean(value: np.ndarray, mean_value: np.ndarray) -> np.ndarray:
    return np.where(value < mean_value, (value - mean_value) ** 2, 2.0 * (value - mean_value) ** 2).astype(np.float32)


def _running_cost_price(S: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    profile = str(params.get("pascucci_cost_profile", "exp")).strip().lower()
    offset = _scalar(params, "pascucci_cost_offset")
    price = np.exp(S).astype(np.float32)
    if profile == "exp_minus_offset":
        return (price - np.float32(offset)).astype(np.float32)
    if profile == "exp" and abs(offset) <= 1.0e-8:
        return price
    raise ValueError(f"unsupported Pascucci oracle cost profile/offset: {profile!r}, {offset!r}")


def _sigma_v(X: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    H = X[:, [1]]
    V = X[:, [2]]
    mean_v = moments["mean_v"]
    return (
        _scalar(params, "s3") * np.ones_like(V, dtype=np.float32)
        + _scalar(params, "s3h") * np.abs(H)
        + _scalar(params, "s3v") * np.abs(V)
        + _scalar(params, "s3k") * np.abs(V - mean_v)
    ).astype(np.float32)


def _alpha(X: np.ndarray, Z: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    q = X[:, [3]]
    z_v = Z[:, [2]]
    denom = 2.0 * _scalar(params, "l_a") * np.maximum(_sigma_v(X, params, moments), np.float32(1.0e-7))
    return (-(_psi(q, params) * z_v) / denom).astype(np.float32)


def _mu(t: np.ndarray, X: np.ndarray, Z: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    S = X[:, [0]]
    H = X[:, [1]]
    V = X[:, [2]]
    q = X[:, [3]]
    dS = _kappa_daynight(t, params["params_S"]) * (_mu_daynight(t, params["params_S"]) - S)
    dH = _kappa_daynight(t, params["params_H"]) * (_mu_daynight(t, params["params_H"]) - H)
    alpha = _alpha(X, Z, params, moments)
    dV = (
        alpha * _psi(q, params)
        + _scalar(params, "c3") * _psi2(q, params) * _psi3(V, params)
        - _scalar(params, "c4") * _psi1(q, params) * _psi4(V, params)
    )
    return np.concatenate([dS, dH, dV, V], axis=1).astype(np.float32)


def _sigma(t: np.ndarray, X: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    M = X.shape[0]
    out = np.zeros((M, 4, 4), dtype=np.float32)
    out[:, 0, 0] = _sigma_daynight(t, params["params_S"])[:, 0]
    out[:, 1, 1] = _sigma_daynight(t, params["params_H"])[:, 0]
    out[:, 2, 2] = _sigma_v(X, params, moments)[:, 0]
    return out


def _f(X: np.ndarray, Z: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    S = X[:, [0]]
    H = X[:, [1]]
    V = X[:, [2]]
    q = X[:, [3]]
    H_plus_V = H + V
    alpha = _alpha(X, Z, params, moments)
    return (
        _running_cost_price(S, params) * H_plus_V
        + _scalar(params, "l_v") * V ** 2
        + _scalar(params, "l_a") * alpha ** 2
        + _scalar(params, "c_h") * _h_with_mean(q, moments["mean_q"])
        + _scalar(params, "c_con") * _h_with_mean(H_plus_V, moments["mean_h_plus_v"])
    ).astype(np.float32)


def _g(X: np.ndarray, params: dict[str, Any], moments: dict[str, np.ndarray]) -> np.ndarray:
    S = X[:, [0]]
    q = X[:, [3]]
    return (
        -_scalar(params, "gamma") * q * np.exp(S)
        + 0.5 * _scalar(params, "omega") * (q - moments["mean_q"]) ** 2
    ).astype(np.float32)


def _validate_fixture(fixture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    metadata = dict(fixture["metadata"])
    inputs = {
        "t": _array(fixture["inputs"]["t"], name="inputs.t", ndim=2),
        "X": _array(fixture["inputs"]["X"], name="inputs.X", ndim=2),
        "Y": _array(fixture["inputs"]["Y"], name="inputs.Y", ndim=2),
        "Z": _array(fixture["inputs"]["Z"], name="inputs.Z", ndim=2),
    }
    params = dict(fixture["params"])
    moments = {
        "mean_v": _array(fixture["moments"]["mean_v"], name="moments.mean_v", ndim=2),
        "mean_q": _array(fixture["moments"]["mean_q"], name="moments.mean_q", ndim=2),
        "mean_h_plus_v": _array(fixture["moments"]["mean_h_plus_v"], name="moments.mean_h_plus_v", ndim=2),
    }
    if inputs["X"].shape[1] != 4 or inputs["Z"].shape[1] != 4:
        raise ValueError("Pascucci oracle fixture expects X and Z with 4 columns")
    if inputs["t"].shape[0] != inputs["X"].shape[0] or inputs["Y"].shape[0] != inputs["X"].shape[0]:
        raise ValueError("Pascucci oracle fixture input batch dimensions do not match")
    return metadata, inputs, params, moments


def evaluate_pascucci_equation_oracle(fixture: dict[str, Any]) -> dict[str, Any]:
    """Evaluate independent expected values for Pascucci ``mu``, ``sigma``, ``alpha``, ``f``, and ``g``."""

    fixture_metadata, inputs, params, moments = _validate_fixture(fixture)
    variant = str(fixture_metadata["oracle_source_variant"])
    try:
        source_provenance = fixture_metadata["source_variants"][variant]
    except KeyError as exc:
        raise ValueError(f"oracle source variant metadata missing for {variant!r}") from exc
    source_file = source_provenance["source_file"]
    if "oracle_validation_mode" not in fixture_metadata:
        raise ValueError("Pascucci oracle fixture metadata missing oracle_validation_mode")
    if "historical_tf1_runtime_parity" not in fixture_metadata:
        raise ValueError("Pascucci oracle fixture metadata missing historical_tf1_runtime_parity")
    if "historical_reference_provenance" not in fixture_metadata:
        raise ValueError("Pascucci oracle fixture metadata missing historical_reference_provenance")
    X = inputs["X"]
    t = inputs["t"]
    Z = inputs["Z"]
    outputs = {
        "mu": _mu(t, X, Z, params, moments),
        "sigma": _sigma(t, X, params, moments),
        "alpha": _alpha(X, Z, params, moments),
        "f": _f(X, Z, params, moments),
        "g": _g(X, params, moments),
    }
    metadata = {
        "oracle_version": ORACLE_VERSION,
        "model_name": "pascucci",
        "fixture_version": int(fixture_metadata["fixture_version"]),
        "fixture_seed": int(fixture_metadata["seed"]),
        "oracle_source_variant": variant,
        "source_file": source_file,
        "source_provenance": dict(source_provenance),
        "historical_references": list(HISTORICAL_REFERENCES),
        "historical_reference_provenance": dict(fixture_metadata["historical_reference_provenance"]),
        "oracle_validation_mode": str(fixture_metadata["oracle_validation_mode"]),
        "historical_tf1_runtime_parity": bool(fixture_metadata["historical_tf1_runtime_parity"]),
        "equation_scope": ["mu", "sigma", "alpha", "f", "g"],
        "moment_policy": "explicit_fixture_moments",
        "pascucci_cost_profile": str(params["pascucci_cost_profile"]),
        "pascucci_cost_offset": _scalar(params, "pascucci_cost_offset"),
        "tolerances": dict(DEFAULT_TOLERANCES),
    }
    return {"metadata": metadata, "outputs": outputs}
