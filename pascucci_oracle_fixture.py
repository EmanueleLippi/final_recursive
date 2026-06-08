"""Deterministic pointwise fixtures for future Pascucci oracle tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model_specs import get_model_spec


FIXTURE_VERSION = 1

SOURCE_VARIANTS = {
    "final_model3": {
        "source_file": "final_model3.py",
        "pascucci_cost_profile": "exp",
        "pascucci_cost_offset": 0.0,
    },
    "final_model_modifiche_f": {
        "source_file": "final_model_modifiche_f.py",
        "pascucci_cost_profile": "exp_minus_offset",
        "pascucci_cost_offset": 0.12,
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _normalise_cost_profile(profile: str, offset: float) -> tuple[str, float]:
    resolved_profile = str(profile).strip().lower()
    resolved_offset = float(offset)
    if resolved_profile not in ("exp", "exp_minus_offset"):
        raise ValueError(f"unsupported pascucci_cost_profile={resolved_profile!r}")
    if not np.isfinite(resolved_offset) or resolved_offset < 0.0:
        raise ValueError("pascucci_cost_offset must be finite and >= 0")
    if resolved_profile == "exp" and abs(resolved_offset) > 1.0e-8:
        raise ValueError("pascucci_cost_offset must be 0.0 when pascucci_cost_profile='exp'")
    return resolved_profile, resolved_offset


def _build_oracle_ou_params(prefix_shift: float) -> dict[str, Any]:
    return {
        "kappa_day": np.float32(0.31 + prefix_shift),
        "kappa_night": np.float32(0.17 + prefix_shift),
        "a0_day": np.float32(0.80 - prefix_shift),
        "a0_night": np.float32(-0.20 + prefix_shift),
        "sigma_day": np.float32(0.13 + 0.5 * prefix_shift),
        "sigma_night": np.float32(0.09 + 0.5 * prefix_shift),
        "alpha_day": np.asarray([0.05 + prefix_shift, -0.02], dtype=np.float32),
        "alpha_night": np.asarray([-0.04, 0.015 + prefix_shift], dtype=np.float32),
        "beta_day": np.asarray([0.01, 0.03 + prefix_shift], dtype=np.float32),
        "beta_night": np.asarray([0.02 - prefix_shift, -0.01], dtype=np.float32),
    }


def _build_pointwise_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    t = np.asarray([[6.0], [7.0], [12.0], [18.0], [19.0], [30.0]], dtype=np.float32)
    X = np.asarray(
        [
            [0.25, -0.40, -2.50, -0.50],
            [1.10, 0.00, -2.00, 0.00],
            [-0.60, 0.70, 0.00, 4.50],
            [0.40, -0.80, 2.00, 10.00],
            [1.80, 0.50, 2.50, 10.60],
            [-1.00, -0.20, 0.80, 7.00],
        ],
        dtype=np.float32,
    )
    X[:, 0] += rng.uniform(-0.05, 0.05, size=X.shape[0]).astype(np.float32)
    X[:, 1] += rng.uniform(-0.03, 0.03, size=X.shape[0]).astype(np.float32)
    Y = rng.uniform(-0.30, 0.30, size=(6, 1)).astype(np.float32)
    Z = np.asarray(
        [
            [0.10, 0.20, -0.30, 0.40],
            [0.20, -0.10, 0.40, -0.50],
            [0.30, 0.00, 0.20, -0.10],
            [0.40, -0.20, -0.10, 0.20],
            [-0.15, 0.25, 0.55, -0.35],
            [0.05, -0.30, -0.45, 0.15],
        ],
        dtype=np.float32,
    )
    Z += rng.uniform(-0.02, 0.02, size=Z.shape).astype(np.float32)
    return {"t": t, "X": X.astype(np.float32), "Y": Y, "Z": Z.astype(np.float32)}


def _build_moments(X: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean_v": np.mean(X[:, [2]], axis=0, keepdims=True).astype(np.float32),
        "mean_q": np.mean(X[:, [3]], axis=0, keepdims=True).astype(np.float32),
        "mean_h_plus_v": np.mean(X[:, [1]] + X[:, [2]], axis=0, keepdims=True).astype(np.float32),
    }


def build_pascucci_oracle_fixture(
    *,
    seed: int = 20260608,
    oracle_source_variant: str = "final_model3",
    pascucci_cost_profile: str | None = None,
    pascucci_cost_offset: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic pointwise Pascucci fixture for equation-level oracles."""

    variant = str(oracle_source_variant).strip()
    if variant not in SOURCE_VARIANTS:
        raise ValueError(f"unsupported oracle_source_variant={variant!r}")
    variant_contract = SOURCE_VARIANTS[variant]
    profile = (
        variant_contract["pascucci_cost_profile"]
        if pascucci_cost_profile is None
        else pascucci_cost_profile
    )
    offset = (
        variant_contract["pascucci_cost_offset"]
        if pascucci_cost_offset is None
        else pascucci_cost_offset
    )
    profile, offset = _normalise_cost_profile(profile, offset)
    if profile != variant_contract["pascucci_cost_profile"] or not np.isclose(
        offset,
        variant_contract["pascucci_cost_offset"],
    ):
        raise ValueError("pascucci cost profile must match oracle_source_variant")

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.77)
    params["params_S"] = _build_oracle_ou_params(0.0)
    params["params_H"] = _build_oracle_ou_params(0.07)
    params["pascucci_cost_profile"] = profile
    params["pascucci_cost_offset"] = np.float32(offset)

    inputs = _build_pointwise_inputs(int(seed))
    moments = _build_moments(inputs["X"])
    metadata = {
        "fixture_version": FIXTURE_VERSION,
        "model_name": "pascucci",
        "seed": int(seed),
        "dtype": "float32",
        "state_labels": list(spec.state_labels),
        "z_labels": list(spec.z_labels),
        "moment_names": list(spec.moment_names),
        "equation_scope": ["mu", "sigma", "alpha", "f", "g"],
        "oracle_source_variant": variant,
        "source_variants": _json_safe(SOURCE_VARIANTS),
        "recursive_terminal_blob": None,
        "coverage": {
            "day_night_hours": [6.0, 7.0, 18.0, 19.0],
            "q_values": _json_safe(inputs["X"][:, 3]),
            "v_values": _json_safe(inputs["X"][:, 2]),
            "z_v_values": _json_safe(inputs["Z"][:, 2]),
        },
        "notes": {
            "pointwise_only": True,
            "no_training_run": True,
            "future_oracle_issue": "#21",
        },
    }
    return {
        "metadata": metadata,
        "inputs": inputs,
        "params": _json_safe(params),
        "moments": moments,
    }


def save_pascucci_oracle_fixture(fixture: dict[str, Any], path: str | Path) -> None:
    """Persist a Pascucci oracle fixture as a pickle-free NPZ artifact."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in fixture["inputs"].items()
    }
    arrays.update(
        {
            f"moment_{key}": np.asarray(value, dtype=np.float32)
            for key, value in fixture["moments"].items()
        }
    )
    np.savez(
        out_path,
        metadata_json=np.asarray(json.dumps(fixture["metadata"], sort_keys=True)),
        params_json=np.asarray(json.dumps(fixture["params"], sort_keys=True)),
        **arrays,
    )


def load_pascucci_oracle_fixture(path: str | Path) -> dict[str, Any]:
    """Load a fixture saved by :func:`save_pascucci_oracle_fixture`."""

    with np.load(Path(path), allow_pickle=False) as data:
        required = {
            "metadata_json",
            "params_json",
            "t",
            "X",
            "Y",
            "Z",
            "moment_mean_v",
            "moment_mean_q",
            "moment_mean_h_plus_v",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing Pascucci oracle fixture keys: {missing}")
        metadata = json.loads(str(data["metadata_json"].item()))
        params = json.loads(str(data["params_json"].item()))
        inputs = {
            "t": np.asarray(data["t"], dtype=np.float32),
            "X": np.asarray(data["X"], dtype=np.float32),
            "Y": np.asarray(data["Y"], dtype=np.float32),
            "Z": np.asarray(data["Z"], dtype=np.float32),
        }
        moments = {
            "mean_v": np.asarray(data["moment_mean_v"], dtype=np.float32),
            "mean_q": np.asarray(data["moment_mean_q"], dtype=np.float32),
            "mean_h_plus_v": np.asarray(data["moment_mean_h_plus_v"], dtype=np.float32),
        }
    return {
        "metadata": metadata,
        "inputs": inputs,
        "params": params,
        "moments": moments,
    }
