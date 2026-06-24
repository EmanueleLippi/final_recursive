"""In-package test runner used by `python -m final_recursive test`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Callable, List
from xml.sax.saxutils import escape

import numpy as np

from .exact import (
    build_exact_solution_functions,
    quadratic_coupled_exact_z_np,
    quadratic_coupled_mu_np,
)
from .sampling import build_blocks, build_stitched_rollout_inputs, file_sha256, Xi_generator_default
from .schedules import parse_float_sequence_arg, resolve_coarse_curriculum_schedule


def _write_minimal_png(path: Path, width: int, height: int) -> None:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", int(width), int(height), 8, 2, 0, 0, 0)
    row = b"\x00" + (b"\xff\xff\xff" * int(width))
    image = zlib.compress(row * int(height))
    payload = header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", image) + chunk(b"IEND", b"")
    if len(payload) <= 1001:
        payload += b"\0" * (1002 - len(payload))
    path.write_bytes(payload)


def _default_params(const: float = 1.0):
    return {
        "mu1": np.float32(1.0),
        "mu2": np.float32(1.0),
        "c1": np.float32(1.0),
        "c2": np.float32(1.0),
        "c3": np.float32(10.0),
        "c4": np.float32(10.0),
        "gamma": np.float32(1.0),
        "d": np.float32(1.0),
        "x_max": np.float32(10.0),
        "v_max": np.float32(2.0),
        "v_min": np.float32(-2.0),
        "s1": np.float32(0.5),
        "s2": np.float32(0.5),
        "s3": np.float32(0.5),
        "const": np.float32(const),
    }


def _build_default_pascucci_ou_params():
    return {
        "kappa_day": np.float32(0.40),
        "kappa_night": np.float32(0.40),
        "a0_day": np.float32(-1.0),
        "a0_night": np.float32(-1.0),
        "sigma_day": np.float32(0.10),
        "sigma_night": np.float32(0.15),
        "alpha_day": np.asarray([np.float32(0.0)], dtype=np.float32),
        "alpha_night": np.asarray([np.float32(0.0)], dtype=np.float32),
        "beta_day": np.asarray([np.float32(0.0)], dtype=np.float32),
        "beta_night": np.asarray([np.float32(0.0)], dtype=np.float32),
    }


def _default_pascucci_params(const: float = 1.0):
    return {
        "l_v": np.float32(0.01),
        "l_a": np.float32(0.005),
        "c3": np.float32(10.0),
        "c4": np.float32(10.0),
        "gamma": np.float32(1.0),
        "d": np.float32(1.0),
        "x_max": np.float32(10.0),
        "v_max": np.float32(2.0),
        "v_min": np.float32(-2.0),
        "s3": np.float32(0.01),
        "s3h": np.float32(0.001),
        "s3v": np.float32(0.001),
        "s3k": np.float32(0.001),
        "omega": np.float32(0.01),
        "c_h": np.float32(0.001),
        "c_con": np.float32(0.01),
        "const": np.float32(const),
        "params_S": _build_default_pascucci_ou_params(),
        "params_H": _build_default_pascucci_ou_params(),
    }


def _make_blob(layers: List[int]) -> dict:
    rng = np.random.RandomState(2026)
    blob = {
        "n_layers": np.array(len(layers) - 1, dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int32),
        "t_start": np.asarray(0.0, dtype=np.float32),
        "t_end": np.asarray(0.25, dtype=np.float32),
        "T_total": np.asarray(0.25, dtype=np.float32),
        "normalize_time_input": np.asarray(1, dtype=np.int32),
        "x_norm_mean": np.zeros((1, 4), dtype=np.float32),
        "x_norm_std": np.ones((1, 4), dtype=np.float32),
    }
    for i, (src, dst) in enumerate(zip(layers[:-1], layers[1:])):
        blob[f"W_{i}"] = (0.15 * rng.normal(size=(src, dst))).astype(np.float32)
        blob[f"b_{i}"] = (0.03 * rng.normal(size=(1, dst))).astype(np.float32)
    return blob


def _make_block_blob(layers: List[int], t_start: float, t_end: float, T_total: float) -> dict:
    blob = _make_blob(layers)
    blob["t_start"] = np.asarray(t_start, dtype=np.float32)
    blob["t_end"] = np.asarray(t_end, dtype=np.float32)
    blob["T_total"] = np.asarray(T_total, dtype=np.float32)
    return blob


def _fixed_validation_batch(
    *,
    M: int = 4,
    N: int = 2,
    D: int = 4,
    T: float = 0.25,
    t_start: float = 0.0,
    seed: int = 123,
    v_shift: float = 0.0,
):
    blocks = [{"idx": 0, "t_start": float(t_start), "t_end": float(t_start + T), "T_block": float(T)}]
    rollout = build_stitched_rollout_inputs(blocks, M=int(M), N_per_block=int(N), D=int(D), seed=int(seed))
    t_batch, W_batch = rollout[0]
    grid = np.linspace(0.0, 1.0, int(M), dtype=np.float32)
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    Xi[:, 0] = 0.5 + 0.1 * grid
    Xi[:, 1] = -0.2 + 0.05 * grid
    Xi[:, 2] = np.float32(v_shift) + 0.2 + 0.03 * grid
    Xi[:, 3] = 3.0 + grid
    return t_batch.astype(np.float32), W_batch.astype(np.float32), Xi.astype(np.float32)


def _schedule_fixed_eval_batches(
    *,
    M: int = 2,
    N: int = 1,
    D: int = 4,
    t_start: float = 0.0,
    t_end: float = 0.25,
):
    t_batch = (
        np.linspace(float(t_start), float(t_end), int(N) + 1, dtype=np.float32)
        .reshape(1, int(N) + 1, 1)
        .repeat(int(M), axis=0)
    )
    W_batch = np.zeros((int(M), int(N) + 1, int(D)), dtype=np.float32)
    Xi_batch = np.zeros((int(M), int(D)), dtype=np.float32)
    return [(t_batch, W_batch, Xi_batch)]


def _pascucci_physical_violation_traces_from_x(X: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    V = X[:, :, [2]]
    q = X[:, :, [3]]
    x_max = np.float32(params["x_max"])
    v_min = np.float32(params["v_min"])
    v_max = np.float32(params["v_max"])
    return {
        "q_lower_violation": np.mean(np.maximum(-q, 0.0), axis=0).astype(np.float32),
        "q_upper_violation": np.mean(np.maximum(q - x_max, 0.0), axis=0).astype(np.float32),
        "v_lower_violation": np.mean(np.maximum(v_min - V, 0.0), axis=0).astype(np.float32),
        "v_upper_violation": np.mean(np.maximum(V - v_max, 0.0), axis=0).astype(np.float32),
    }


def _assert_source_provenance_contract(provenance: dict, *, expected_file: str) -> None:
    import hashlib

    assert provenance["source_file"] == expected_file
    assert provenance["source_path"].endswith(expected_file)
    assert provenance["source_available"] is True
    assert isinstance(provenance["source_sha256"], str)
    assert len(provenance["source_sha256"]) == 64
    int(provenance["source_sha256"], 16)
    assert int(provenance["source_size_bytes"]) > 0
    path = (Path(__file__).resolve().parent / provenance["source_path"]).resolve()
    payload = path.read_bytes()
    assert provenance["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["source_size_bytes"] == len(payload)


def _run_case(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
    except RuntimeError as exc:
        if "TensorFlow could not be imported" in str(exc) or "TensorFlow is not installed" in str(exc):
            print(f"SKIP {name}: {exc}")
            return True
        print(f"FAIL {name}: {exc}")
        return False
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"PASS {name}")
    return True


def _run_subprocess_case(name: str, function_name: str) -> bool:
    repo_code = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_code + os.pathsep + env.get("PYTHONPATH", "")
    script = f"from final_recursive.tests import {function_name}; {function_name}()"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode == 0:
        if output.strip():
            print(output.strip())
        print(f"PASS {name}")
        return True
    if "TensorFlow could not be imported" in output or "TensorFlow is not installed" in output:
        print(f"SKIP {name}: TensorFlow could not be imported")
        return True
    print(f"FAIL {name}: subprocess exited with code {completed.returncode}")
    if output.strip():
        print(output.strip())
    return False


def test_numpy_math_and_schedules() -> None:
    blocks = build_blocks(T_total=1.0, block_size=0.4)
    assert len(blocks) == 3
    assert np.isclose(blocks[-1]["t_end"], 1.0)

    values = parse_float_sequence_arg("0.0, 0.5, 1.0", "--example")
    assert values == [0.0, 0.5, 1.0]
    consts, scales = resolve_coarse_curriculum_schedule(values, [0.2], terminal_const=0.75)
    assert consts == [0.0, 0.5, 0.75]
    assert scales == [0.2, 0.2, 0.2]

    params = _default_params()
    X = np.array(
        [
            [1.0, 1.0, 0.2, 4.0],
            [0.5, 1.2, -0.3, 7.0],
        ],
        dtype=np.float32,
    )
    Z = quadratic_coupled_exact_z_np(X, params)
    mu = quadratic_coupled_mu_np(X, Z, params)
    assert Z.shape == X.shape
    assert mu.shape == X.shape
    assert np.all(np.isfinite(mu))

    exact = build_exact_solution_functions("quadratic_coupled", params, D=4)
    assert exact is not None
    Y = exact["u_exact"](np.zeros((2, 1), dtype=np.float32), X)
    assert Y.shape == (2, 1)


def test_rollout_inputs_are_antithetic() -> None:
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    rollout = build_stitched_rollout_inputs(blocks, M=4, N_per_block=3, D=4, seed=99)
    assert len(rollout) == 2
    for t, W in rollout:
        assert t.shape == (4, 4, 1)
        assert W.shape == (4, 4, 4)
        dW = W[:, 1:, :] - W[:, :-1, :]
        np.testing.assert_allclose(dW[0], -dW[2], atol=1.0e-6)
        np.testing.assert_allclose(dW[1], -dW[3], atol=1.0e-6)


def test_evaluation_bundle_rejects_block_metadata_mismatch() -> None:
    from .sampling import load_evaluation_bundle, save_evaluation_bundle

    saved_blocks = build_blocks(T_total=0.5, block_size=0.25)
    shifted_blocks = [
        {
            "idx": int(block["idx"]),
            "t_start": float(block["t_start"]) + 0.125,
            "t_end": float(block["t_end"]) + 0.125,
            "T_block": float(block["T_block"]),
        }
        for block in saved_blocks
    ]
    Xi = np.zeros((4, 4), dtype=np.float32)
    rollout = build_stitched_rollout_inputs(
        blocks=saved_blocks,
        M=Xi.shape[0],
        N_per_block=2,
        D=4,
        seed=123,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation_bundle.npz"
        save_evaluation_bundle(
            path=str(path),
            Xi_initial=Xi,
            rollout_inputs=rollout,
            blocks=saved_blocks,
        )
        loaded_xi, loaded_rollout = load_evaluation_bundle(
            path=str(path),
            n_blocks_expected=len(saved_blocks),
            N_per_block_expected=2,
            D_expected=4,
        )
        np.testing.assert_allclose(loaded_xi, Xi)
        assert len(loaded_rollout) == len(saved_blocks)
        try:
            load_evaluation_bundle(
                path=str(path),
                n_blocks_expected=len(shifted_blocks),
                N_per_block_expected=2,
                D_expected=4,
                blocks_expected=shifted_blocks,
                T_total_expected=0.625,
            )
        except ValueError as exc:
            message = str(exc).lower()
            assert "block" in message or "t_total" in message or "metadata" in message
        else:
            raise AssertionError("mismatched evaluation bundle metadata should be rejected")


def test_evaluation_bundle_rejects_wrong_bundle_kind() -> None:
    from .sampling import load_evaluation_bundle, save_evaluation_bundle

    blocks = build_blocks(T_total=0.5, block_size=0.25)
    Xi = np.zeros((4, 4), dtype=np.float32)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=Xi.shape[0],
        N_per_block=2,
        D=4,
        seed=124,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation_bundle.npz"
        save_evaluation_bundle(
            path=str(path),
            Xi_initial=Xi,
            rollout_inputs=rollout,
            blocks=blocks,
            bundle_kind="evaluation",
        )
        try:
            load_evaluation_bundle(
                path=str(path),
                n_blocks_expected=len(blocks),
                N_per_block_expected=2,
                D_expected=4,
                blocks_expected=blocks,
                T_total_expected=0.5,
                bundle_kind_expected="boundary_rollout",
            )
        except ValueError as exc:
            message = str(exc).lower()
            assert "bundle_kind" in message
            assert "boundary_rollout" in message
        else:
            raise AssertionError("evaluation bundle should not load as a boundary rollout bundle")


def test_evaluation_bundle_loads_legacy_no_kind_as_evaluation_only() -> None:
    from .sampling import load_evaluation_bundle

    blocks = build_blocks(T_total=0.5, block_size=0.25)
    Xi = np.zeros((4, 4), dtype=np.float32)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=Xi.shape[0],
        N_per_block=2,
        D=4,
        seed=125,
    )
    t_bundle = np.stack([pair[0] for pair in rollout], axis=0).astype(np.float32)
    W_bundle = np.stack([pair[1] for pair in rollout], axis=0).astype(np.float32)
    block_t_start = np.array([float(block["t_start"]) for block in blocks], dtype=np.float32)
    block_t_end = np.array([float(block["t_end"]) for block in blocks], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy_evaluation_bundle.npz"
        np.savez(
            path,
            Xi_initial=Xi,
            t_bundle=t_bundle,
            W_bundle=W_bundle,
            block_t_start=block_t_start,
            block_t_end=block_t_end,
        )
        loaded_xi, loaded_rollout = load_evaluation_bundle(
            path=str(path),
            n_blocks_expected=len(blocks),
            N_per_block_expected=2,
            D_expected=4,
            blocks_expected=blocks,
            T_total_expected=0.5,
            bundle_kind_expected="evaluation",
        )
        np.testing.assert_allclose(loaded_xi, Xi)
        assert len(loaded_rollout) == len(blocks)

        try:
            load_evaluation_bundle(
                path=str(path),
                n_blocks_expected=len(blocks),
                N_per_block_expected=2,
                D_expected=4,
                blocks_expected=blocks,
                T_total_expected=0.5,
                bundle_kind_expected="boundary_rollout",
            )
        except ValueError as exc:
            assert "missing bundle_kind" in str(exc)
        else:
            raise AssertionError("legacy evaluation bundle should not load as boundary rollout")


def test_model_spec_contract() -> None:
    from .model_specs import get_model_spec

    spec = get_model_spec()
    assert spec.name == "quadratic_coupled"
    assert spec.state_dim == 4
    assert spec.state_labels == ("S", "H", "V", "X_state")
    assert spec.z_labels == ("Z_S", "Z_H", "Z_V", "Z_X")
    assert spec.build_layers(4) == [5, 256, 256, 256, 256, 1]
    assert spec.exclude_bootstrap_pass_from_selection is True

    params = spec.build_default_params(const=0.75)
    expected = _default_params(const=0.75)
    assert set(params) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(params[key], value)

    pascucci_spec = get_model_spec("pascucci")
    assert pascucci_spec.name == "pascucci"
    assert pascucci_spec.state_dim == 4
    assert pascucci_spec.state_labels == ("S", "H", "V", "X_state")
    assert pascucci_spec.z_labels == ("Z_S", "Z_H", "Z_V", "Z_X")
    assert pascucci_spec.build_layers(4) == [5, 256, 256, 256, 256, 1]
    assert pascucci_spec.exclude_bootstrap_pass_from_selection is False

    pascucci_params = pascucci_spec.build_default_params(const=0.55)
    assert pascucci_params["const"] == np.float32(0.55)
    assert pascucci_params["l_a"] == np.float32(0.005)
    assert pascucci_params["c_h"] == np.float32(0.001)
    for required in ("params_S", "params_H", "l_v", "l_a", "s3", "s3h", "s3v", "s3k", "omega", "c_h", "c_con"):
        assert required in pascucci_params
    assert pascucci_spec.build_exact_initial_boundary_samples is None
    assert pascucci_spec.build_exact_solution("none", pascucci_params, 4) is None
    try:
        pascucci_spec.build_exact_solution("quadratic_coupled", pascucci_params, 4)
    except ValueError as exc:
        assert "pascucci" in str(exc)
        assert "--exact_solution none" in str(exc)
    else:
        raise AssertionError("pascucci should reject exact profiles until an oracle exists")

    spec.validate_state_dim(4)
    for action in (lambda: spec.validate_state_dim(3), lambda: spec.build_layers(3)):
        try:
            action()
        except ValueError as exc:
            assert "D=4" in str(exc)
        else:
            raise AssertionError("D != 4 should be rejected")

    try:
        get_model_spec("unsupported_model_xyz")
    except ValueError as exc:
        assert "Supported:" in str(exc)
        assert "quadratic_coupled" in str(exc)
    else:
        raise AssertionError("unknown model should be rejected")

    assert spec.build_exact_solution("none", params, 4) is None
    exact = spec.build_exact_solution("quadratic_coupled", params, 4)
    assert exact is not None
    assert exact["name"] == "quadratic_coupled"

    Xi = spec.deterministic_xi(4, 4, seed=11)
    assert Xi.shape == (4, 4)
    assert Xi.dtype == np.float32


def test_pascucci_model_spec_uses_paper_like_initial_distribution() -> None:
    from .model_specs import get_model_spec

    quadratic_spec = get_model_spec("quadratic_coupled")
    pascucci_spec = get_model_spec("pascucci")
    assert pascucci_spec.xi_generator is not quadratic_spec.xi_generator
    assert pascucci_spec.deterministic_xi is not quadratic_spec.deterministic_xi

    Xi = pascucci_spec.deterministic_xi(10000, 4, seed=123)
    Xi_repeat = pascucci_spec.deterministic_xi(10000, 4, seed=123)
    assert Xi.shape == (10000, 4)
    assert Xi.dtype == np.float32
    np.testing.assert_allclose(Xi, Xi_repeat)

    means = np.mean(Xi, axis=0)
    stds = np.std(Xi, axis=0)
    assert abs(float(means[0]) - (-2.3)) < 0.02
    assert abs(float(stds[0]) - 0.2) < 0.01
    assert abs(float(means[1]) - 0.4) < 0.03
    assert abs(float(stds[1]) - 0.5) < 0.02
    assert abs(float(means[2])) < 0.03
    assert abs(float(stds[2]) - 1.0) < 0.03
    assert float(np.min(Xi[:, 3])) >= 1.0
    assert float(np.max(Xi[:, 3])) <= 9.0
    assert abs(float(means[3]) - 5.0) < 0.08

    quadratic_Xi = quadratic_spec.deterministic_xi(10000, 4, seed=123)
    assert abs(float(np.mean(quadratic_Xi[:, 0]) - means[0])) > 3.0
    assert float(np.min(quadratic_Xi[:, 3])) >= 3.0
    assert float(np.max(quadratic_Xi[:, 3])) <= 7.0


PASCUCCI_CALIBRATION_TDD_CONTRACTS = {
    "pascucci_prepare_H_hourly_mean_net_power_and_scale": {
        "type": "unit",
        "target": "pascucci_data.prepare_H",
        "purpose": "Protect historical H ingestion: hourly mean net power, not energy sum.",
        "expected": "1D float array with truncated full blocks and mul_factor applied.",
        "failure": "Catches unit-scale drift, tail handling drift, or net-power sign swap.",
    },
    "pascucci_prepare_H_missing_columns_raise": {
        "type": "negative-unit",
        "target": "pascucci_data.prepare_H",
        "purpose": "Fail closed when required home-load CSV columns are missing.",
        "expected": "ValueError naming the missing required columns.",
        "failure": "Catches silent acceptance of malformed load/production files.",
    },
    "pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log": {
        "type": "unit",
        "target": "pascucci_data.prepare_S",
        "purpose": "Protect price ingestion: comma decimals, hourly averaging, no log transform.",
        "expected": "1D float array of linear prices after mul_factor, not log prices.",
        "failure": "Catches locale parsing errors, double-log risk, and scale drift.",
    },
    "pascucci_prepare_S_missing_or_non_numeric_values_raise": {
        "type": "negative-unit",
        "target": "pascucci_data.prepare_S",
        "purpose": "Fail closed on missing price column or non-numeric price cells.",
        "expected": "ValueError for missing columns and ValueError for invalid numeric cells.",
        "failure": "Catches silent bad-price ingestion before OU calibration.",
    },
    "pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling": {
        "type": "oracle-unit",
        "target": "pascucci_calibration.calibrate_OU_variable",
        "purpose": "Protect continuous-time OU parameter recovery and the dt scaling fix.",
        "expected": "Known kappa/a0 recovered and continuous-time sigma normalized by sqrt(dt).",
        "failure": "Catches recurrence of the historical dt scaling bug.",
    },
    "pascucci_calibrate_ou_variable_start_hour_controls_phase": {
        "type": "oracle-unit",
        "target": "pascucci_calibration.calibrate_OU_variable",
        "purpose": "Make physical 24h clock phase explicit through start_hour.",
        "expected": "Shifted synthetic series recovers parameters only with matching start_hour.",
        "failure": "Catches hidden reset of day/night and Fourier phase at block boundaries.",
    },
    "pascucci_calibrate_ou_variable_rejects_degenerate_inputs": {
        "type": "negative-unit",
        "target": "pascucci_calibration.calibrate_OU_variable",
        "purpose": "Fail fast on underdetermined, missing-regime, or non-mean-reverting fits.",
        "expected": "ValueError for too few rows, missing regimes, and kappa <= kappa_min.",
        "failure": "Catches singular OLS, NaN sigma, and anti-mean-reverting drift.",
    },
    "pascucci_calibration_output_contract_shapes": {
        "type": "unit",
        "target": "pascucci_calibration.calibrate_OU_variable/validate_ou_params schema",
        "purpose": "Freeze params_S/params_H schema consumed by the Pascucci model abstraction.",
        "expected": "10 exact keys; validate_ou_params returns None; Fourier arrays shape (K,); finite scalars; sigma >= 0.",
        "failure": "Catches schema drift before malformed params reach the model layer.",
    },
    "pascucci_calibrate_inputs_log_price_guard_and_parity": {
        "type": "regression-unit",
        "target": "pascucci_calibration.calibrate_pascucci_ou_inputs",
        "purpose": "Make prepare_S no-log and log-price calibration guard explicit.",
        "expected": "S <= 0 rejected; positive S path equals direct np.log(S) calibration.",
        "failure": "Catches double-log/no-log mistakes in Pascucci price calibration.",
    },
    "quadratic_spec_unaffected_by_pascucci_calibration_import": {
        "type": "regression-unit",
        "target": "model_specs.get_model_spec plus Pascucci calibration imports",
        "purpose": "Protect frozen quadratic_coupled baseline from calibration module side effects.",
        "expected": "Quadratic metadata, params, deterministic Xi, and RNG stream unchanged.",
        "failure": "Catches hidden import-time coupling to the benchmark path.",
    },
    "pascucci_ou_params_json_safe_after_serialization": {
        "type": "unit",
        "target": "pascucci_calibration.serialize_ou_params",
        "purpose": "Make params_S/params_H safe for run_config JSON without relying on ad hoc encoder side effects.",
        "expected": "OU params validate after serialization and contain only JSON-native scalars/lists.",
        "failure": "Catches numpy scalar/array leakage into persisted run configs.",
    },
    "pascucci_day_night_boundary_semantics_are_explicit": {
        "type": "unit",
        "target": "pascucci_calibration.is_day",
        "purpose": "Freeze physical day/night regime boundaries used by OU calibration.",
        "expected": "Day is [7, 19) modulo 24, so 7.0 is day and 19.0 is night.",
        "failure": "Catches off-by-one or hidden timezone/phase changes in day/night fits.",
    },
    "pascucci_log_price_false_calibrates_linear_prices": {
        "type": "regression-unit",
        "target": "pascucci_calibration.calibrate_pascucci_ou_inputs",
        "purpose": "Protect both log-price and linear-price branches from collapsing to the same scale.",
        "expected": "log_price=False matches direct calibration on linear S, while log_price=True matches direct calibration on log(S).",
        "failure": "Catches silent double-log or no-log mistakes.",
    },
    "pascucci_calibration_config_records_units_log_price_dt_and_sources": {
        "type": "unit",
        "target": "pascucci_calibration.build_pascucci_calibration_config",
        "purpose": "Freeze metadata for units, log-price policy, physical clock phase, and data sources.",
        "expected": "JSON-safe config includes params_H, params_S, K, dt, start_hour, log_price, transforms, and source metadata.",
        "failure": "Catches untraceable calibration assumptions before micro-runs.",
    },
    "pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags": {
        "type": "regression-unit",
        "target": "pascucci_calibration.build_pascucci_run_config_params",
        "purpose": "Ensure calibrated OU params can replace defaults in Pascucci params while preserving solver/training flags.",
        "expected": "Returned params are JSON-safe, contain calibrated params_S/params_H and calibration metadata, and do not mutate defaults.",
        "failure": "Catches non-reproducible run configs or accidental solver flag drops.",
    },
    "pascucci_minimal_fixture_pipeline_builds_json_run_params": {
        "type": "acceptance-unit",
        "target": "pascucci_data + pascucci_calibration fixture pipeline",
        "purpose": "Exercise the minimum CSV/XLSX -> prepare -> calibrate -> run params path without starting a run.",
        "expected": "Fixture pipeline produces finite validated params and a JSON-dumpable Pascucci run params dict.",
        "failure": "Catches broken end-to-end calibration wiring before T12/T24 gates.",
    },
    "pascucci_cli_calibration_paths_inject_run_config_params": {
        "type": "acceptance-cli",
        "target": "cli.run --pascucci_H_path/--pascucci_S_path",
        "purpose": "Prevent T12/T24 plot-parity runs from silently using placeholder OU params.",
        "expected": "CLI prepares, calibrates, injects params_H/params_S, and persists calibration metadata in run_config.json.",
        "failure": "Catches non-comparable Pascucci runs with default H/S dynamics.",
    },
    "pascucci_cli_require_calibration_fails_before_training_without_paths": {
        "type": "negative-cli",
        "target": "cli.run --pascucci_require_calibration",
        "purpose": "Fail fast before expensive T12/T24 paper-parity runs when H/S calibration sources are missing.",
        "expected": "ValueError before run directory creation or training when either calibration path is absent, and rejection for non-Pascucci models.",
        "failure": "Catches accidental launch of diagnostic-only Pascucci runs under a paper-parity label.",
    },
}


PASCUCCI_MODEL_LAYER_TDD_CONTRACTS = {
    "pascucci_cost_profile_default_is_exp_and_json_safe": {
        "type": "regression-unit",
        "target": "model_specs.get_model_spec('pascucci').build_default_params",
        "purpose": "Freeze the current Pascucci running-cost profile as explicit config instead of an implicit formula.",
        "expected": "Default params record pascucci_cost_profile='exp' and pascucci_cost_offset=0.0 in JSON-safe form.",
        "failure": "Catches silent flips from exp(S) to exp(S)-0.12 or untracked cost-profile choices.",
    },
    "pascucci_cost_profile_exp_minus_offset_changes_only_running_cost": {
        "type": "oracle-unit",
        "target": "models.NN_Pascucci.f_tf/phi_tf/g_tf",
        "purpose": "Make the historical exp(S)-0.12 variant selectable while preserving the terminal cost.",
        "expected": "The offset profile shifts f and phi by -offset*(H+V), while g remains tied to exp(S).",
        "failure": "Catches applying the 0.12 offset to the wrong equation or losing the exp(S) baseline.",
    },
    "pascucci_cost_profile_rejects_unknown_profile": {
        "type": "negative-unit",
        "target": "models.NN_Pascucci.__init__",
        "purpose": "Fail fast on unsupported Pascucci cost profiles before any training run.",
        "expected": "Constructing a Pascucci model with an unknown cost profile raises ValueError.",
        "failure": "Catches silent fallback to an unintended economic objective.",
    },
    "pascucci_cli_records_cost_profile_params_in_run_config": {
        "type": "integration-unit",
        "target": "cli.run_program/run_config.json",
        "purpose": "Ensure the selected Pascucci cost profile is persisted in reproducible run metadata.",
        "expected": "A stubbed tiny CLI run records pascucci_cost_profile and pascucci_cost_offset in params/run_config.",
        "failure": "Catches a configurable model choice that is not traceable in output artifacts.",
    },
    "pascucci_cli_rejects_cost_profile_for_quadratic_model": {
        "type": "negative-integration-unit",
        "target": "cli.run_program",
        "purpose": "Protect the frozen quadratic benchmark from Pascucci-specific config pollution.",
        "expected": "Passing Pascucci cost-profile args with quadratic_coupled raises ValueError before any run.",
        "failure": "Catches benchmark run_config drift or accidental cross-model parameter injection.",
    },
    "pascucci_cli_rejects_calibration_paths_for_quadratic_model": {
        "type": "negative-integration-unit",
        "target": "cli.run_program",
        "purpose": "Protect the frozen quadratic benchmark from Pascucci-specific calibration metadata.",
        "expected": "Passing Pascucci H/S calibration paths with quadratic_coupled raises ValueError before any file read or run.",
        "failure": "Catches benchmark run_config drift or accidental cross-model calibration injection.",
    },
    "pascucci_recursive_cost_profile_matches_standard_formula": {
        "type": "regression-unit",
        "target": "models.NN_Pascucci_Recursive.f_tf/phi_tf/g_tf",
        "purpose": "Keep the recursive Pascucci block model on the same running-cost formula as the standard model.",
        "expected": "The recursive model has the same offset delta for f/phi and leaves g unchanged.",
        "failure": "Catches standard/recursive model divergence before micro-runs.",
    },
    "pascucci_cli_records_cost_profile_params_in_recursive_run_config": {
        "type": "integration-unit",
        "target": "cli.run_program recursive/run_config.json",
        "purpose": "Ensure recursive Pascucci runs persist the selected cost profile before training.",
        "expected": "A stubbed tiny recursive CLI run records the selected cost profile and forwards it to orchestration.",
        "failure": "Catches recursive-only config drops before time-stitching validation.",
    },
    "pascucci_physical_constraint_diagnostics_q_v_are_model_owned": {
        "type": "unit",
        "target": "models.NN_Pascucci.physical_constraint_diagnostics_tf",
        "purpose": "Expose Q/V physical violations from the Pascucci model layer without solver branching.",
        "expected": "Per-sample non-negative Q/V lower/upper violations match x_max, v_min, and v_max.",
        "failure": "Catches hidden physical diagnostics or accidental coupling to generic solver code.",
    },
    "pascucci_q_v_barrier_drift_pushes_toward_physical_domain": {
        "type": "regression-unit",
        "target": "models.NN_Pascucci.mu_tf",
        "purpose": "Protect the existing barrier signs for Q/V before adding later physical diagnostics.",
        "expected": "With zero Z_V, dV is positive below Q=0, negative above Q=x_max, and zero in the interior.",
        "failure": "Catches sign inversions in the Pascucci Q/V barrier terms.",
    },
    "pascucci_q_v_barrier_sweep_with_nonzero_z_v": {
        "type": "regression-unit",
        "target": "models.NN_Pascucci.mu_tf",
        "purpose": "Protect Q/V barrier and control interaction when the active control component Z_V is nonzero.",
        "expected": "Boundary pushes have the correct sign and dV matches the closed-form Pascucci barrier formula.",
        "failure": "Catches ignored Z_V terms, barrier sign flips, or scaling drift in the Q/V physical-domain guard.",
    },
}


PASCUCCI_ORACLE_FIXTURE_TDD_CONTRACTS = {
    "pascucci_oracle_fixture_generation_contract": {
        "type": "acceptance-unit",
        "target": "pascucci_oracle_fixture.build_pascucci_oracle_fixture",
        "purpose": "Create a deterministic pointwise Pascucci fixture for future TF1/TF2 oracle tests.",
        "expected": "Fixture contains JSON-safe metadata/params plus finite float32 t, X, Y, Z arrays and explicit moments.",
        "failure": "Catches untraceable, non-serializable, or under-specified oracle fixture inputs before #21.",
    },
    "pascucci_oracle_fixture_reproducible_and_seed_sensitive": {
        "type": "unit",
        "target": "pascucci_oracle_fixture.build_pascucci_oracle_fixture",
        "purpose": "Pin reproducibility while keeping seed changes observable.",
        "expected": "Same seed gives identical arrays and metadata; different seed changes pointwise non-time inputs.",
        "failure": "Catches hidden RNG, unstable ordering, or a seed parameter that is recorded but ignored.",
    },
    "pascucci_oracle_fixture_save_load_roundtrip": {
        "type": "io-unit",
        "target": "pascucci_oracle_fixture.save/load_pascucci_oracle_fixture",
        "purpose": "Make fixture artifacts reusable across future TF1/TF2 subprocess oracle checks.",
        "expected": "NPZ roundtrip preserves arrays, metadata, params, and explicit moments without pickle.",
        "failure": "Catches non-portable fixture artifacts or lossy metadata serialization.",
    },
    "pascucci_oracle_fixture_cost_profile_variants_are_explicit": {
        "type": "regression-unit",
        "target": "pascucci_oracle_fixture.build_pascucci_oracle_fixture metadata",
        "purpose": "Avoid ambiguity between historical final_model3 and final_model_modifiche_f running-cost variants.",
        "expected": "Fixture metadata records source variant, cost profile, and offset; unsupported profiles fail fast.",
        "failure": "Catches selecting the wrong historical oracle source by implication.",
    },
    "pascucci_oracle_provenance_metadata_roundtrip": {
        "type": "acceptance-unit",
        "target": "pascucci_oracle_fixture + pascucci_equation_oracle provenance metadata",
        "purpose": "Make oracle source provenance reproducible and auditable before thesis-level oracle evidence.",
        "expected": "Fixture/oracle metadata record validation mode, TF1-runtime parity status, source paths, sizes, and SHA-256 hashes.",
        "failure": "Catches stale, missing, or lossy provenance metadata in pointwise oracle artifacts.",
    },
    "pascucci_oracle_fixture_missing_historical_reference_fails_fast": {
        "type": "negative-unit",
        "target": "pascucci_oracle_fixture.build_pascucci_oracle_fixture",
        "purpose": "Document that oracle provenance requires the read-only historical Pascucci source files.",
        "expected": "Fixture generation raises FileNotFoundError with an actionable message when a historical reference is unavailable.",
        "failure": "Catches silent downgrade from content-addressed provenance to filename-only oracle metadata.",
    },
    "quadratic_spec_unaffected_by_pascucci_oracle_fixture_import": {
        "type": "regression-unit",
        "target": "model_specs.get_model_spec plus pascucci_oracle_fixture import",
        "purpose": "Protect the frozen quadratic benchmark from Pascucci oracle-fixture side effects.",
        "expected": "Quadratic params and deterministic Xi remain unchanged after importing/building Pascucci fixtures.",
        "failure": "Catches hidden import-time coupling or benchmark config pollution.",
    },
}


PASCUCCI_EQUATION_ORACLE_TDD_CONTRACTS = {
    "pascucci_equation_oracle_tdd_metadata": {
        "type": "acceptance-unit",
        "target": "pascucci_equation_oracle.evaluate_pascucci_equation_oracle metadata",
        "purpose": "Make #21 an explicit equation-level oracle slice, not an informal fixture comparison.",
        "expected": "Contract documents mu/sigma/alpha/f/g scope, historical source variant, and explicit tolerances.",
        "failure": "Catches untracked oracle assumptions before using the tests as thesis evidence.",
    },
    "pascucci_equation_oracle_final_model3_matches_tf2_fixture": {
        "type": "oracle-unit",
        "target": "models.NN_Pascucci mu/sigma/alpha/f/g on fixture #20",
        "purpose": "Compare TF2 Pascucci equations against independent NumPy formulas matching final_model3.py.",
        "expected": "mu, sigma, alpha, f, and g match expected values on deterministic fixture inputs.",
        "failure": "Catches equation drift in OU day/night, Z_V control, barriers, running cost, or terminal cost.",
    },
    "pascucci_equation_oracle_exp_minus_offset_variant": {
        "type": "oracle-unit",
        "target": "pascucci_equation_oracle final_model_modifiche_f variant",
        "purpose": "Preserve the historical exp(S)-0.12 running-cost variant without changing other equations.",
        "expected": "Only f shifts by -0.12*(H+V); mu, sigma, alpha, and g stay on the same formulas.",
        "failure": "Catches applying the historical offset to the wrong equation or source variant.",
    },
    "pascucci_equation_oracle_uses_explicit_fixture_moments": {
        "type": "regression-unit",
        "target": "pascucci_equation_oracle moment_state handling",
        "purpose": "Ensure oracle expected values consume fixture moments instead of hidden reduce_mean side effects.",
        "expected": "Changing explicit moments changes the expected equations and still matches TF2 with moment_state.",
        "failure": "Catches oracle code that silently recomputes mean-field quantities from X.",
    },
    "quadratic_spec_unaffected_by_pascucci_equation_oracle_import": {
        "type": "regression-unit",
        "target": "model_specs.get_model_spec plus pascucci_equation_oracle import",
        "purpose": "Protect the frozen quadratic benchmark from Pascucci oracle side effects.",
        "expected": "Quadratic params and deterministic Xi remain unchanged after importing/evaluating Pascucci oracle code.",
        "failure": "Catches hidden import-time coupling or benchmark config pollution.",
    },
}


def _assert_pascucci_tdd_contract(name: str) -> None:
    contract = PASCUCCI_CALIBRATION_TDD_CONTRACTS[name]
    for key in ("type", "target", "purpose", "expected", "failure"):
        value = str(contract.get(key, "")).strip()
        assert value, f"{name} missing TDD contract field {key}"


def _assert_pascucci_model_tdd_contract(name: str) -> None:
    contract = PASCUCCI_MODEL_LAYER_TDD_CONTRACTS[name]
    for key in ("type", "target", "purpose", "expected", "failure"):
        value = str(contract.get(key, "")).strip()
        assert value, f"{name} missing TDD contract field {key}"


def _assert_pascucci_oracle_fixture_tdd_contract(name: str) -> None:
    contract = PASCUCCI_ORACLE_FIXTURE_TDD_CONTRACTS[name]
    for key in ("type", "target", "purpose", "expected", "failure"):
        value = str(contract.get(key, "")).strip()
        assert value, f"{name} missing TDD contract field {key}"


def _assert_pascucci_equation_oracle_tdd_contract(name: str) -> None:
    contract = PASCUCCI_EQUATION_ORACLE_TDD_CONTRACTS[name]
    for key in ("type", "target", "purpose", "expected", "failure"):
        value = str(contract.get(key, "")).strip()
        assert value, f"{name} missing TDD contract field {key}"


def _assert_json_native_tree(value, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str), f"{path} contains non-string key {key!r}"
            _assert_json_native_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_json_native_tree(child, path=f"{path}[{idx}]")
        return
    assert value is None or isinstance(value, (str, int, float, bool)), (
        f"{path} contains non-JSON-native value {type(value).__name__}: {value!r}"
    )
    if isinstance(value, float):
        assert np.isfinite(value), f"{path} contains non-finite float"


def _assert_json_roundtrip(value) -> None:
    _assert_json_native_tree(value)
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == value


def _write_rows_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    import csv

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _xlsx_col_name(index: int) -> str:
    index = int(index)
    name = ""
    while index >= 0:
        name = chr(ord("A") + (index % 26)) + name
        index = index // 26 - 1
    return name


def _write_minimal_xlsx(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    xml_rows = []
    all_rows = [headers] + rows
    for row_idx, row in enumerate(all_rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
            text = escape(str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{row_idx}">' + "".join(cells) + "</row>")

    last_ref = f"{_xlsx_col_name(len(all_rows[0]) - 1)}{len(all_rows)}"
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_ref}"/>'
        "<sheetData>"
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _is_day_np(t: np.ndarray) -> np.ndarray:
    hour = np.mod(np.asarray(t, dtype=np.float64), 24.0)
    return (hour >= 7.0) & (hour < 19.0)


def _as_harmonic_coeffs(values) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _periodic_mean_np(t: np.ndarray, a0: float, alpha, beta) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    alpha_arr = _as_harmonic_coeffs(alpha)
    beta_arr = _as_harmonic_coeffs(beta)
    if alpha_arr.shape != beta_arr.shape:
        raise ValueError("alpha and beta test coefficients must have the same length")
    mean = np.full_like(t, float(a0), dtype=np.float64)
    for idx, (alpha_k, beta_k) in enumerate(zip(alpha_arr, beta_arr), start=1):
        omega = 2.0 * np.pi * float(idx) / 24.0
        mean += float(alpha_k) * np.cos(omega * t)
        mean += float(beta_k) * np.sin(omega * t)
    return mean


def _pascucci_test_design_matrix(values: np.ndarray, t: np.ndarray, K: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    day = _is_day_np(t).astype(np.float64)
    night = 1.0 - day
    columns = [-values * day, -values * night, day, night]
    for k in range(1, int(K) + 1):
        omega = 2.0 * np.pi * float(k) / 24.0
        cos = np.cos(omega * t)
        sin = np.sin(omega * t)
        columns.extend([cos * day, cos * night, sin * day, sin * night])
    return np.column_stack(columns)


def _expected_ou_regression_sigmas(series: np.ndarray, *, K: int, dt: float, start_hour: float) -> tuple[float, float]:
    series = np.asarray(series, dtype=np.float64)
    Y = series[1:] - series[:-1]
    t = float(start_hour) + np.arange(Y.shape[0], dtype=np.float64) * float(dt)
    X = _pascucci_test_design_matrix(series[:-1], t, int(K))
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ theta
    p_regime = 2 + 2 * int(K)
    day = _is_day_np(t)
    night = ~day
    sigma_day = np.sqrt(np.sum(residuals[day] ** 2) / (int(np.sum(day)) - p_regime) / float(dt))
    sigma_night = np.sqrt(np.sum(residuals[night] ** 2) / (int(np.sum(night)) - p_regime) / float(dt))
    return float(sigma_day), float(sigma_night)


def _simulate_piecewise_ou_series(
    *,
    n_points: int,
    dt: float,
    start_hour: float = 0.0,
    kappa_day: float = 0.30,
    kappa_night: float = 0.15,
    a0_day: float = 1.20,
    a0_night: float = -0.80,
    alpha_day=None,
    alpha_night=None,
    beta_day=None,
    beta_night=None,
    sigma_day: float = 0.0,
    sigma_night: float = 0.0,
    seed: int | None = None,
    x0: float = 0.35,
) -> np.ndarray:
    alpha_day_arr = _as_harmonic_coeffs(alpha_day)
    alpha_night_arr = _as_harmonic_coeffs(alpha_night)
    beta_day_arr = np.zeros_like(alpha_day_arr) if beta_day is None else _as_harmonic_coeffs(beta_day)
    beta_night_arr = np.zeros_like(alpha_night_arr) if beta_night is None else _as_harmonic_coeffs(beta_night)
    if alpha_day_arr.shape != beta_day_arr.shape:
        raise ValueError("alpha_day and beta_day test coefficients must have the same length")
    if alpha_night_arr.shape != beta_night_arr.shape:
        raise ValueError("alpha_night and beta_night test coefficients must have the same length")
    rng = np.random.RandomState(seed) if seed is not None else None
    values = np.zeros(int(n_points), dtype=np.float64)
    values[0] = float(x0)
    for i in range(int(n_points) - 1):
        t = float(start_hour) + float(i) * float(dt)
        if _is_day_np(np.asarray([t]))[0]:
            kappa = float(kappa_day)
            a0 = float(a0_day)
            alpha = alpha_day_arr
            beta = beta_day_arr
            sigma = float(sigma_day)
        else:
            kappa = float(kappa_night)
            a0 = float(a0_night)
            alpha = alpha_night_arr
            beta = beta_night_arr
            sigma = float(sigma_night)
        diffusion = 0.0
        if sigma != 0.0:
            if rng is None:
                raise ValueError("seed is required for noisy synthetic OU test series")
            diffusion = sigma * np.sqrt(float(dt)) * rng.normal()
        mu = _periodic_mean_np(np.asarray([t]), a0, alpha, beta)[0]
        values[i + 1] = values[i] + kappa * (mu - values[i]) * float(dt) + diffusion
    return values.astype(np.float64)


def _assert_ou_params_close_to_piecewise_truth(
    params: dict,
    *,
    kappa_day: float = 0.30,
    kappa_night: float = 0.15,
    a0_day: float = 1.20,
    a0_night: float = -0.80,
    alpha_day=None,
    alpha_night=None,
    beta_day=None,
    beta_night=None,
    atol: float = 1.0e-8,
) -> None:
    alpha_day_arr = _as_harmonic_coeffs(alpha_day)
    alpha_night_arr = _as_harmonic_coeffs(alpha_night)
    beta_day_arr = np.zeros_like(alpha_day_arr) if beta_day is None else _as_harmonic_coeffs(beta_day)
    beta_night_arr = np.zeros_like(alpha_night_arr) if beta_night is None else _as_harmonic_coeffs(beta_night)
    np.testing.assert_allclose(params["kappa_day"], kappa_day, rtol=1.0e-7, atol=atol)
    np.testing.assert_allclose(params["kappa_night"], kappa_night, rtol=1.0e-7, atol=atol)
    np.testing.assert_allclose(params["a0_day"], a0_day, rtol=1.0e-7, atol=atol)
    np.testing.assert_allclose(params["a0_night"], a0_night, rtol=1.0e-7, atol=atol)
    np.testing.assert_allclose(params["alpha_day"], alpha_day_arr, atol=atol)
    np.testing.assert_allclose(params["alpha_night"], alpha_night_arr, atol=atol)
    np.testing.assert_allclose(params["beta_day"], beta_day_arr, atol=atol)
    np.testing.assert_allclose(params["beta_night"], beta_night_arr, atol=atol)
    assert float(params["sigma_day"]) <= 1.0e-8
    assert float(params["sigma_night"]) <= 1.0e-8


def test_pascucci_tdd_contract_metadata() -> None:
    expected_names = {
        "pascucci_prepare_H_hourly_mean_net_power_and_scale",
        "pascucci_prepare_H_missing_columns_raise",
        "pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log",
        "pascucci_prepare_S_missing_or_non_numeric_values_raise",
        "pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling",
        "pascucci_calibrate_ou_variable_start_hour_controls_phase",
        "pascucci_calibrate_ou_variable_rejects_degenerate_inputs",
        "pascucci_calibration_output_contract_shapes",
        "pascucci_calibrate_inputs_log_price_guard_and_parity",
        "quadratic_spec_unaffected_by_pascucci_calibration_import",
        "pascucci_ou_params_json_safe_after_serialization",
        "pascucci_day_night_boundary_semantics_are_explicit",
        "pascucci_log_price_false_calibrates_linear_prices",
        "pascucci_calibration_config_records_units_log_price_dt_and_sources",
        "pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags",
        "pascucci_minimal_fixture_pipeline_builds_json_run_params",
        "pascucci_cli_calibration_paths_inject_run_config_params",
        "pascucci_cli_require_calibration_fails_before_training_without_paths",
    }
    assert set(PASCUCCI_CALIBRATION_TDD_CONTRACTS) == expected_names
    for name in expected_names:
        _assert_pascucci_tdd_contract(name)


def test_pascucci_model_layer_tdd_contract_metadata() -> None:
    expected_names = {
        "pascucci_cost_profile_default_is_exp_and_json_safe",
        "pascucci_cost_profile_exp_minus_offset_changes_only_running_cost",
        "pascucci_cost_profile_rejects_unknown_profile",
        "pascucci_cli_records_cost_profile_params_in_run_config",
        "pascucci_cli_rejects_cost_profile_for_quadratic_model",
        "pascucci_cli_rejects_calibration_paths_for_quadratic_model",
        "pascucci_recursive_cost_profile_matches_standard_formula",
        "pascucci_cli_records_cost_profile_params_in_recursive_run_config",
        "pascucci_physical_constraint_diagnostics_q_v_are_model_owned",
        "pascucci_q_v_barrier_drift_pushes_toward_physical_domain",
        "pascucci_q_v_barrier_sweep_with_nonzero_z_v",
    }
    assert set(PASCUCCI_MODEL_LAYER_TDD_CONTRACTS) == expected_names
    for name in expected_names:
        _assert_pascucci_model_tdd_contract(name)


def test_pascucci_oracle_fixture_tdd_contract_metadata() -> None:
    expected_names = {
        "pascucci_oracle_fixture_generation_contract",
        "pascucci_oracle_fixture_reproducible_and_seed_sensitive",
        "pascucci_oracle_fixture_save_load_roundtrip",
        "pascucci_oracle_fixture_cost_profile_variants_are_explicit",
        "pascucci_oracle_provenance_metadata_roundtrip",
        "pascucci_oracle_fixture_missing_historical_reference_fails_fast",
        "quadratic_spec_unaffected_by_pascucci_oracle_fixture_import",
    }
    assert set(PASCUCCI_ORACLE_FIXTURE_TDD_CONTRACTS) == expected_names
    for name in expected_names:
        _assert_pascucci_oracle_fixture_tdd_contract(name)


def test_pascucci_equation_oracle_tdd_contract_metadata() -> None:
    expected_names = {
        "pascucci_equation_oracle_tdd_metadata",
        "pascucci_equation_oracle_final_model3_matches_tf2_fixture",
        "pascucci_equation_oracle_exp_minus_offset_variant",
        "pascucci_equation_oracle_uses_explicit_fixture_moments",
        "quadratic_spec_unaffected_by_pascucci_equation_oracle_import",
    }
    assert set(PASCUCCI_EQUATION_ORACLE_TDD_CONTRACTS) == expected_names
    for name in expected_names:
        _assert_pascucci_equation_oracle_tdd_contract(name)


def test_pascucci_prepare_H_hourly_mean_net_power_and_scale() -> None:
    _assert_pascucci_tdd_contract("pascucci_prepare_H_hourly_mean_net_power_and_scale")
    from .pascucci_data import prepare_H

    consumo = np.asarray([100, 120, 140, 160, 200, 220, 240, 260, 900, 901, 902], dtype=np.float64)
    produzione = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3], dtype=np.float64)
    rows = [
        {"Consumo (W)": str(c), "Produzione (W)": str(p)}
        for c, p in zip(consumo, produzione)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "home.csv"
        _write_rows_csv(path, ["Consumo (W)", "Produzione (W)"], rows)
        prepared = prepare_H(str(path), n=4, mul_factor=0.001)

    expected = np.asarray(
        [
            np.mean((consumo - produzione)[:4]),
            np.mean((consumo - produzione)[4:8]),
        ],
        dtype=np.float64,
    ) * 0.001
    assert prepared.shape == (2,)
    assert np.all(np.isfinite(prepared))
    np.testing.assert_allclose(prepared, expected, rtol=0.0, atol=1.0e-12)


def test_pascucci_prepare_H_missing_columns_raise() -> None:
    _assert_pascucci_tdd_contract("pascucci_prepare_H_missing_columns_raise")
    from .pascucci_data import prepare_H

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_home.csv"
        _write_rows_csv(path, ["Consumo (W)"], [{"Consumo (W)": "1.0"}])
        try:
            prepare_H(str(path), n=1, mul_factor=1.0)
        except ValueError as exc:
            message = str(exc)
            assert "Produzione (W)" in message
        else:
            raise AssertionError("prepare_H should reject missing Produzione (W)")


def test_pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log() -> None:
    _assert_pascucci_tdd_contract("pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log")
    from .pascucci_data import prepare_S

    values = ["10,0", "20,0", "30,0", "40,0", "50,0", "60,0", "999,0"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prices.xlsx"
        _write_minimal_xlsx(path, ["Data", "Ora", "€/MWh"], [["d", i, value] for i, value in enumerate(values)])
        prepared = prepare_S(str(path), n=3, mul_factor=0.01)

    expected = np.asarray([0.2, 0.5], dtype=np.float64)
    assert prepared.shape == (2,)
    assert np.all(np.isfinite(prepared))
    np.testing.assert_allclose(prepared, expected, rtol=0.0, atol=1.0e-12)


def test_pascucci_prepare_S_missing_or_non_numeric_values_raise() -> None:
    _assert_pascucci_tdd_contract("pascucci_prepare_S_missing_or_non_numeric_values_raise")
    from .pascucci_data import prepare_S

    with tempfile.TemporaryDirectory() as tmp:
        missing_path = Path(tmp) / "missing_price.xlsx"
        _write_minimal_xlsx(missing_path, ["Data", "Ora"], [["d", "1"]])
        try:
            prepare_S(str(missing_path), n=1, mul_factor=1.0)
        except ValueError as exc:
            assert "€/MWh" in str(exc)
        else:
            raise AssertionError("prepare_S should reject missing €/MWh")

        bad_value_path = Path(tmp) / "bad_price.xlsx"
        _write_minimal_xlsx(bad_value_path, ["Data", "Ora", "€/MWh"], [["d", "1", "not-a-number"]])
        try:
            prepare_S(str(bad_value_path), n=1, mul_factor=1.0)
        except ValueError as exc:
            message = str(exc)
            assert "€/MWh" in message
            assert "numeric" in message.lower() or "not-a-number" in message
        else:
            raise AssertionError("prepare_S should reject non-numeric price values")


def test_pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling")
    from .pascucci_calibration import calibrate_OU_variable

    for dt in (1.0, 0.5):
        series = _simulate_piecewise_ou_series(n_points=240, dt=dt, start_hour=0.0)
        params = calibrate_OU_variable(series, K=0, dt=dt, start_hour=0.0)
        _assert_ou_params_close_to_piecewise_truth(params)

    recovered_sigmas = {}
    for dt, n_points in ((1.0, 6000), (0.5, 6000)):
        noisy_series = _simulate_piecewise_ou_series(
            n_points=n_points,
            dt=dt,
            start_hour=0.0,
            sigma_day=0.04,
            sigma_night=0.07,
            seed=314159,
        )
        noisy_params = calibrate_OU_variable(noisy_series, K=0, dt=dt, start_hour=0.0)
        expected_sigma_day, expected_sigma_night = _expected_ou_regression_sigmas(
            noisy_series,
            K=0,
            dt=dt,
            start_hour=0.0,
        )
        recovered_sigmas[dt] = (float(noisy_params["sigma_day"]), float(noisy_params["sigma_night"]))
        np.testing.assert_allclose(noisy_params["sigma_day"], expected_sigma_day, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(noisy_params["sigma_night"], expected_sigma_night, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(noisy_params["sigma_day"], 0.04, rtol=0.08, atol=0.004)
        np.testing.assert_allclose(noisy_params["sigma_night"], 0.07, rtol=0.08, atol=0.004)
    np.testing.assert_allclose(recovered_sigmas[0.5], recovered_sigmas[1.0], rtol=0.08, atol=0.004)


def test_pascucci_calibrate_ou_variable_start_hour_controls_phase() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibrate_ou_variable_start_hour_controls_phase")
    from .pascucci_calibration import calibrate_OU_variable

    harmonic_truth = {
        "alpha_day": np.asarray([0.20], dtype=np.float64),
        "alpha_night": np.asarray([-0.15], dtype=np.float64),
        "beta_day": np.asarray([0.10], dtype=np.float64),
        "beta_night": np.asarray([-0.05], dtype=np.float64),
    }
    series = _simulate_piecewise_ou_series(n_points=240, dt=1.0, start_hour=6.0, **harmonic_truth)
    correct_phase = calibrate_OU_variable(series, K=1, dt=1.0, start_hour=6.0)
    _assert_ou_params_close_to_piecewise_truth(correct_phase, **harmonic_truth)

    wrong_phase = calibrate_OU_variable(series, K=1, dt=1.0, start_hour=0.0)
    mismatch_terms = [
        abs(float(wrong_phase["kappa_day"]) - 0.30),
        abs(float(wrong_phase["kappa_night"]) - 0.15),
        abs(float(wrong_phase["a0_day"]) - 1.20),
        abs(float(wrong_phase["a0_night"]) + 0.80),
    ]
    for key, expected in harmonic_truth.items():
        mismatch_terms.extend(np.abs(np.asarray(wrong_phase[key], dtype=np.float64) - expected).reshape(-1))
    mismatch = max(float(value) for value in mismatch_terms)
    assert mismatch > 1.0e-3, "wrong start_hour should not silently recover the oracle"


def test_pascucci_calibrate_ou_variable_rejects_degenerate_inputs() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibrate_ou_variable_rejects_degenerate_inputs")
    from .pascucci_calibration import calibrate_OU_variable

    invalid_cases = [
        (
            "too few regression rows",
            np.linspace(0.0, 1.0, 8),
            {"K": 1, "dt": 3.0, "start_hour": 4.0},
            ("underdetermined",),
        ),
        (
            "insufficient per-regime residual dof",
            _simulate_piecewise_ou_series(n_points=13, dt=1.0, start_hour=1.0),
            {"K": 2, "dt": 1.0, "start_hour": 1.0},
            ("residual degrees of freedom",),
        ),
        (
            "missing night regime",
            np.linspace(0.0, 1.0, 8),
            {"K": 0, "dt": 1.0, "start_hour": 7.0},
            ("night",),
        ),
        (
            "ill-conditioned regression",
            1.0 + 1.0e-10 * np.sin(np.arange(240, dtype=np.float64)),
            {"K": 0, "dt": 1.0, "start_hour": 0.0},
            ("ill-conditioned", "condition"),
        ),
        (
            "tiny non-identifiable kappa",
            _simulate_piecewise_ou_series(
                n_points=240,
                dt=1.0,
                start_hour=0.0,
                kappa_day=1.0e-5,
                kappa_night=1.0e-5,
            ),
            {"K": 0, "dt": 1.0, "start_hour": 0.0, "kappa_min": 1.0e-4},
            ("mean-reverting", "kappa"),
        ),
        (
            "anti mean reverting kappa",
            _simulate_piecewise_ou_series(
                n_points=240,
                dt=1.0,
                start_hour=0.0,
                kappa_day=-0.20,
                kappa_night=-0.10,
            ),
            {"K": 0, "dt": 1.0, "start_hour": 0.0},
            ("mean-reverting", "ill-conditioned", "condition"),
        ),
    ]
    for label, series, kwargs, expected_tokens in invalid_cases:
        try:
            calibrate_OU_variable(series, **kwargs)
        except ValueError as exc:
            message = str(exc).lower()
            assert message.strip(), label
            assert any(token in message for token in expected_tokens), (label, message)
        else:
            raise AssertionError(f"calibrate_OU_variable should reject {label}")


def test_pascucci_calibration_output_contract_shapes() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibration_output_contract_shapes")
    from .pascucci_calibration import calibrate_OU_variable, validate_ou_params

    K = 2
    series = _simulate_piecewise_ou_series(n_points=240, dt=1.0, start_hour=0.0)
    params = calibrate_OU_variable(series, K=K, dt=1.0, start_hour=0.0)
    params_snapshot = {key: np.asarray(value).copy() for key, value in params.items()}
    assert validate_ou_params(params, K=K) is None

    expected_keys = {
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
    assert set(params) == expected_keys
    for key, value in params_snapshot.items():
        np.testing.assert_allclose(params[key], value)
    for key in ("alpha_day", "alpha_night", "beta_day", "beta_night"):
        value = np.asarray(params[key])
        assert value.shape == (K,), f"{key} shape {value.shape}"
        assert np.all(np.isfinite(value)), key
    for key in expected_keys - {"alpha_day", "alpha_night", "beta_day", "beta_night"}:
        assert np.asarray(params[key]).shape == (), key
        assert np.isfinite(float(params[key])), key
    assert float(params["sigma_day"]) >= 0.0
    assert float(params["sigma_night"]) >= 0.0

    bad_params = dict(params)
    bad_params["alpha_day"] = np.asarray([0.0], dtype=np.float32)
    try:
        validate_ou_params(bad_params, K=K)
    except ValueError as exc:
        assert "alpha_day" in str(exc) or "K" in str(exc)
    else:
        raise AssertionError("validate_ou_params should reject wrong harmonic length")

    bad_params = dict(params)
    bad_params.pop("sigma_day")
    try:
        validate_ou_params(bad_params, K=K)
    except ValueError as exc:
        assert "sigma_day" in str(exc)
    else:
        raise AssertionError("validate_ou_params should reject missing sigma_day")

    bad_params = dict(params)
    bad_params["beta_night"] = np.asarray([0.0, np.nan], dtype=np.float32)
    try:
        validate_ou_params(bad_params, K=K)
    except ValueError as exc:
        assert "beta_night" in str(exc) or "finite" in str(exc).lower()
    else:
        raise AssertionError("validate_ou_params should reject NaN harmonic coefficients")

    bad_params = dict(params)
    bad_params["sigma_night"] = np.float32(-1.0)
    try:
        validate_ou_params(bad_params, K=K)
    except ValueError as exc:
        assert "sigma_night" in str(exc) or "non-negative" in str(exc).lower()
    else:
        raise AssertionError("validate_ou_params should reject negative sigma")


def test_pascucci_calibrate_inputs_log_price_guard_and_parity() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibrate_inputs_log_price_guard_and_parity")
    from .pascucci_calibration import calibrate_OU_variable, calibrate_pascucci_ou_inputs

    H_series = _simulate_piecewise_ou_series(n_points=240, dt=1.0, start_hour=0.0)
    log_S_series = _simulate_piecewise_ou_series(
        n_points=240,
        dt=1.0,
        start_hour=0.0,
        x0=0.15,
        a0_day=0.25,
        a0_night=-0.10,
    )
    S_prices = np.exp(log_S_series)
    bundle = calibrate_pascucci_ou_inputs(H_series, S_prices, K=0, dt=1.0, start_hour=0.0, log_price=True)
    assert set(bundle) == {"params_H", "params_S"}
    direct_log_params = calibrate_OU_variable(np.log(S_prices), K=0, dt=1.0, start_hour=0.0)
    for key, value in direct_log_params.items():
        np.testing.assert_allclose(bundle["params_S"][key], value, rtol=1.0e-7, atol=1.0e-8)

    bad_prices = S_prices.copy()
    bad_prices[3] = 0.0
    try:
        calibrate_pascucci_ou_inputs(H_series, bad_prices, K=0, dt=1.0, start_hour=0.0, log_price=True)
    except ValueError as exc:
        assert "positive" in str(exc).lower() or "log" in str(exc).lower()
    else:
        raise AssertionError("log_price=True should reject non-positive prices")


def test_quadratic_spec_unaffected_by_pascucci_calibration_import() -> None:
    _assert_pascucci_tdd_contract("quadratic_spec_unaffected_by_pascucci_calibration_import")
    from .model_specs import get_model_spec

    spec_before = get_model_spec("quadratic_coupled")
    params_before = spec_before.build_default_params(const=0.75)
    layers_before = spec_before.build_layers(4)
    xi_before = spec_before.deterministic_xi(4, 4, seed=2026)
    for module_name in ("final_recursive.pascucci_calibration", "final_recursive.pascucci_data"):
        sys.modules.pop(module_name, None)
    np.random.seed(2468)
    expected_random_before = np.random.random(4)
    expected_random_after_import = np.random.random(4)
    np.random.seed(2468)
    random_before = np.random.random(4)

    import importlib

    importlib.import_module("final_recursive.pascucci_calibration")
    importlib.import_module("final_recursive.pascucci_data")

    spec_after = get_model_spec("quadratic_coupled")
    params_after = spec_after.build_default_params(const=0.75)
    layers_after = spec_after.build_layers(4)
    xi_after = spec_after.deterministic_xi(4, 4, seed=2026)
    random_after = np.random.random(4)

    assert spec_after.name == spec_before.name == "quadratic_coupled"
    assert spec_after.state_labels == spec_before.state_labels
    assert spec_after.z_labels == spec_before.z_labels
    assert layers_after == layers_before
    assert set(params_after) == set(params_before)
    for key in params_before:
        np.testing.assert_allclose(params_after[key], params_before[key])
    np.testing.assert_allclose(xi_after, xi_before)
    np.testing.assert_allclose(random_before, expected_random_before)
    np.testing.assert_allclose(random_after, expected_random_after_import)


def _build_sprint12_calibration_config(K: int = 1) -> dict:
    from .pascucci_calibration import build_pascucci_calibration_config, calibrate_pascucci_ou_inputs

    H_series = _simulate_piecewise_ou_series(
        n_points=240,
        dt=0.5,
        start_hour=6.0,
        kappa_day=0.22,
        kappa_night=0.18,
        a0_day=1.10,
        a0_night=-0.45,
        alpha_day=np.asarray([0.08], dtype=np.float64)[:K],
        alpha_night=np.asarray([-0.04], dtype=np.float64)[:K],
        beta_day=np.asarray([0.03], dtype=np.float64)[:K],
        beta_night=np.asarray([-0.02], dtype=np.float64)[:K],
        x0=0.20,
    )
    log_S_series = _simulate_piecewise_ou_series(
        n_points=240,
        dt=0.5,
        start_hour=6.0,
        kappa_day=0.20,
        kappa_night=0.16,
        a0_day=0.22,
        a0_night=-0.05,
        alpha_day=np.asarray([0.05], dtype=np.float64)[:K],
        alpha_night=np.asarray([-0.02], dtype=np.float64)[:K],
        beta_day=np.asarray([0.02], dtype=np.float64)[:K],
        beta_night=np.asarray([-0.01], dtype=np.float64)[:K],
        x0=0.10,
    )
    calibration = calibrate_pascucci_ou_inputs(
        H_series,
        np.exp(log_S_series),
        K=K,
        dt=0.5,
        start_hour=6.0,
        log_price=True,
    )
    return build_pascucci_calibration_config(
        calibration,
        K=K,
        dt=0.5,
        start_hour=6.0,
        log_price=True,
        H_metadata={
            "source_path": "fixture_H.csv",
            "units": "kW",
            "n_per_hour": 2,
            "mul_factor": 0.001,
        },
        S_metadata={
            "source_path": "fixture_S.xlsx",
            "units": "EUR_per_MWh",
            "n_per_hour": 2,
            "mul_factor": 0.01,
        },
    )


def test_pascucci_ou_params_json_safe_after_serialization() -> None:
    _assert_pascucci_tdd_contract("pascucci_ou_params_json_safe_after_serialization")
    from .pascucci_calibration import calibrate_OU_variable, serialize_ou_params, validate_ou_params

    K = 1
    series = _simulate_piecewise_ou_series(
        n_points=240,
        dt=1.0,
        start_hour=0.0,
        alpha_day=np.asarray([0.10], dtype=np.float64),
        alpha_night=np.asarray([-0.06], dtype=np.float64),
        beta_day=np.asarray([0.04], dtype=np.float64),
        beta_night=np.asarray([-0.03], dtype=np.float64),
    )
    params = calibrate_OU_variable(series, K=K, dt=1.0, start_hour=0.0)
    json_params = serialize_ou_params(params, K=K)

    assert json_params is not params
    assert set(json_params) == set(params)
    assert validate_ou_params(json_params, K=K) is None
    for key in ("alpha_day", "alpha_night", "beta_day", "beta_night"):
        assert isinstance(json_params[key], list), key
        assert len(json_params[key]) == K
    for key in ("kappa_day", "kappa_night", "a0_day", "a0_night", "sigma_day", "sigma_night"):
        assert isinstance(json_params[key], float), key
    _assert_json_roundtrip(json_params)

    json_params["alpha_day"][0] += 1.0
    assert not np.allclose(json_params["alpha_day"], params["alpha_day"])


def test_pascucci_day_night_boundary_semantics_are_explicit() -> None:
    _assert_pascucci_tdd_contract("pascucci_day_night_boundary_semantics_are_explicit")
    from .pascucci_calibration import is_day

    times = np.asarray([6.999, 7.0, 18.999, 19.0, 30.999, 31.0, 42.999, 43.0], dtype=np.float64)
    expected = np.asarray([False, True, True, False, False, True, True, False], dtype=bool)
    result = is_day(times)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_pascucci_log_price_false_calibrates_linear_prices() -> None:
    _assert_pascucci_tdd_contract("pascucci_log_price_false_calibrates_linear_prices")
    from .pascucci_calibration import calibrate_OU_variable, calibrate_pascucci_ou_inputs

    H_series = _simulate_piecewise_ou_series(n_points=240, dt=1.0, start_hour=0.0)
    log_S_series = _simulate_piecewise_ou_series(
        n_points=240,
        dt=1.0,
        start_hour=0.0,
        x0=0.15,
        a0_day=0.25,
        a0_night=-0.10,
    )
    S_prices = np.exp(log_S_series)
    linear_bundle = calibrate_pascucci_ou_inputs(H_series, S_prices, K=0, dt=1.0, start_hour=0.0, log_price=False)
    direct_linear_params = calibrate_OU_variable(S_prices, K=0, dt=1.0, start_hour=0.0)
    direct_log_params = calibrate_OU_variable(np.log(S_prices), K=0, dt=1.0, start_hour=0.0)
    for key, value in direct_linear_params.items():
        np.testing.assert_allclose(linear_bundle["params_S"][key], value, rtol=1.0e-7, atol=1.0e-8)

    differences = [
        float(abs(np.asarray(direct_linear_params[key]).reshape(-1)[0] - np.asarray(direct_log_params[key]).reshape(-1)[0]))
        for key in ("kappa_day", "kappa_night", "a0_day", "a0_night")
    ]
    assert max(differences) > 1.0e-3, "linear and log price calibration should differ on this fixture"


def test_pascucci_calibration_config_records_units_log_price_dt_and_sources() -> None:
    _assert_pascucci_tdd_contract("pascucci_calibration_config_records_units_log_price_dt_and_sources")
    from .pascucci_calibration import validate_ou_params

    config = _build_sprint12_calibration_config(K=1)
    assert set(config) == {"params_H", "params_S", "calibration"}
    metadata = config["calibration"]
    assert metadata["schema"] == "pascucci_ou_calibration_v1"
    assert metadata["K"] == 1
    assert metadata["dt"] == 0.5
    assert metadata["start_hour"] == 6.0
    assert metadata["day_window_hours"] == [7.0, 19.0]
    assert metadata["log_price"] is True
    assert metadata["H_transform"] == "linear"
    assert metadata["S_transform"] == "log"
    assert metadata["H_metadata"]["units"] == "kW"
    assert metadata["H_metadata"]["n_per_hour"] == 2
    assert metadata["S_metadata"]["units"] == "EUR_per_MWh"
    assert metadata["S_metadata"]["mul_factor"] == 0.01
    assert validate_ou_params(config["params_H"], K=1) is None
    assert validate_ou_params(config["params_S"], K=1) is None
    _assert_json_roundtrip(config)


def test_pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags() -> None:
    _assert_pascucci_tdd_contract(
        "pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags"
    )
    from .model_specs import get_model_spec
    from .pascucci_calibration import build_pascucci_run_config_params

    defaults = get_model_spec("pascucci").build_default_params(const=0.75)
    defaults.update(
        {
            "same_xi_antithetic_sampling": True,
            "dynamic_loss_dt_normalization": True,
            "terminal_z_component_weights": [1.0, 2.0, 3.0, 4.0],
        }
    )
    defaults_snapshot = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in defaults.items()
    }
    config = _build_sprint12_calibration_config(K=1)
    params = build_pascucci_run_config_params(defaults, config)

    assert params is not defaults
    assert params["const"] == 0.75
    assert params["same_xi_antithetic_sampling"] is True
    assert params["dynamic_loss_dt_normalization"] is True
    assert params["terminal_z_component_weights"] == [1.0, 2.0, 3.0, 4.0]
    assert params["params_H"] == config["params_H"]
    assert params["params_S"] == config["params_S"]
    assert params["pascucci_calibration"] == config["calibration"]
    assert np.asarray(defaults_snapshot["params_H"]["alpha_day"]).shape == (1,)
    assert np.asarray(defaults["params_H"]["alpha_day"]).shape == (1,)
    _assert_json_roundtrip(params)


def test_pascucci_minimal_fixture_pipeline_builds_json_run_params() -> None:
    _assert_pascucci_tdd_contract("pascucci_minimal_fixture_pipeline_builds_json_run_params")
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
        validate_ou_params,
    )
    from .pascucci_data import prepare_H, prepare_S
    from .pascucci_plotting import _load_to_ema_ou_reference, _load_to_ema_uncontrolled_reference

    n_per_hour = 2
    H_hours = _simulate_piecewise_ou_series(n_points=240, dt=1.0, start_hour=0.0)
    log_S_hours = _simulate_piecewise_ou_series(
        n_points=240,
        dt=1.0,
        start_hour=0.0,
        x0=0.08,
        a0_day=0.18,
        a0_night=-0.08,
    )
    S_hours = np.exp(log_S_hours)
    H_rows = []
    for value in H_hours:
        for _ in range(n_per_hour):
            production = 5000.0
            consumption = production + float(value) / 0.001
            H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
    S_rows = []
    for value in S_hours:
        for _ in range(n_per_hour):
            S_rows.append(["fixture-day", "fixture-hour", f"{float(value) / 0.01:.12f}"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_fixture.csv"
        S_path = tmp_path / "S_fixture.xlsx"
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        prepared_H = prepare_H(str(H_path), n=n_per_hour, mul_factor=0.001)
        prepared_S = prepare_S(str(S_path), n=n_per_hour, mul_factor=0.01)

        np.testing.assert_allclose(prepared_H, H_hours, rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(prepared_S, S_hours, rtol=0.0, atol=1.0e-10)
        calibration = calibrate_pascucci_ou_inputs(
            prepared_H,
            prepared_S,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
        )
        config = build_pascucci_calibration_config(
            calibration,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
            H_metadata={
                "source_path": str(H_path),
                "units": "kW",
                "n_per_hour": n_per_hour,
                "mul_factor": 0.001,
            },
            S_metadata={
                "source_path": str(S_path),
                "units": "EUR_per_MWh",
                "n_per_hour": n_per_hour,
                "mul_factor": 0.01,
            },
        )

    assert validate_ou_params(config["params_H"], K=0) is None
    assert validate_ou_params(config["params_S"], K=0) is None
    params = build_pascucci_run_config_params(get_model_spec("pascucci").build_default_params(), config)
    assert params["pascucci_calibration"]["H_metadata"]["source_path"].endswith("H_fixture.csv")
    assert params["pascucci_calibration"]["S_metadata"]["source_path"].endswith("S_fixture.xlsx")
    _assert_json_roundtrip(params)


def test_cli_pascucci_calibration_paths_inject_run_config_params() -> None:
    _assert_pascucci_tdd_contract("pascucci_cli_calibration_paths_inject_run_config_params")
    import types

    from . import cli

    n_points = 120
    H_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.25,
        a0_day=0.80,
        a0_night=-0.30,
    )
    log_S_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.05,
        a0_day=0.30,
        a0_night=-0.12,
    )
    S_hours = np.exp(log_S_hours)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_cli_fixture.csv"
        S_path = tmp_path / "S_cli_fixture.xlsx"
        out_dir = tmp_path / "out"
        H_rows = []
        for value in H_hours:
            for _ in range(4):
                production = 5000.0
                consumption = production + float(value) / 0.001
                H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
        S_rows = [["fixture-day", "fixture-hour", f"{float(value) / 0.01:.12f}"] for value in S_hours]
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        calls = {"recursive": 0, "print": 0}
        captured_params = {}

        def fake_run_recursive_training(**kwargs):
            calls["recursive"] += 1
            params = kwargs["params"]
            captured_params.update(params)
            assert kwargs["model_spec"].name == "pascucci"
            assert params["pascucci_calibration"]["schema"] == "pascucci_ou_calibration_v1"
            assert params["pascucci_calibration"]["K"] == 0
            assert params["pascucci_calibration"]["log_price"] is True
            assert params["pascucci_calibration"]["S_transform"] == "log"
            assert params["params_H"]["a0_day"] != -1.0
            assert params["params_S"]["a0_day"] != -1.0
            assert params["pascucci_calibration"]["S_metadata"]["mul_factor"] == 0.01
            blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
            logs = [
                {
                    "pass": 1,
                    "block": 0,
                    "t_start": blocks[0]["t_start"],
                    "t_end": blocks[0]["t_end"],
                    "T_block": blocks[0]["T_block"],
                    "eval_mean_loss": 1.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.25,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "precision_target": None,
                    "refine_rounds": 0,
                }
            ]
            return {
                "blocks": blocks,
                "passes": [
                    {
                        "pass_id": 1,
                        "reference_loss": 1.0,
                        "logs": logs,
                        "blobs": [_make_blob(kwargs["layers"]) for _ in blocks],
                        "models_dir": kwargs["output_dir"],
                        "pass_init_mode": kwargs["pass1_init_mode"],
                        "boundary_source": "base_xi",
                        "is_bootstrap_pass": True,
                        "active_set_summary": {},
                    }
                ],
                "boundary_samples": [
                    np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                    for _ in range(len(blocks) + 1)
                ],
            }

        def fake_print_recursive_pass(**kwargs):
            calls["print"] += 1
            assert kwargs["model_spec"].name == "pascucci"
            assert kwargs["params"]["pascucci_calibration"]["schema"] == "pascucci_ou_calibration_v1"
            return {
                "processed_pass_ids": [1],
                "exact_summary_by_pass": {},
                "exact_summary_by_pass_index": {},
                "eval_bundle_path": str(Path(kwargs["rec_dir"]) / "evaluation_bundle.npz"),
                "eval_bundle_sha256": "",
                "eval_bundle_status": "fake",
                "evaluation_bundle_M": 4,
                "excluded_pass_ids_from_selection": [],
                "excluded_pass_indices_from_selection": [],
                "selected_pass_id": 1,
                "selected_pass_index": 0,
                "selected_score_metric": "loss.eval_mean_loss_per_sample",
                "selected_score": 0.25,
                "selected_scores_by_pass": {"1": 0.25},
                "selected_scores_by_pass_index": {"0": 0.25},
                "score_key": "eval_mean_loss_per_sample",
                "pass_scores_loss": {1: 0.25},
                "pass_scores_loss_by_index": {0: 0.25},
            }

        fake_orchestration = types.ModuleType("final_recursive.orchestration")
        fake_orchestration.run_recursive_training = fake_run_recursive_training
        fake_orchestration.print_recursive_pass = fake_print_recursive_pass
        fake_orchestration.run_recursive_coarse_prepass = lambda **kwargs: None
        fake_orchestration.run_standard_reference = lambda **kwargs: None

        module_name = f"{cli.__package__}.orchestration"
        original_orchestration = sys.modules.get(module_name)
        original_set_tf_seed = cli.set_tf_seed
        sys.modules[module_name] = fake_orchestration
        cli.set_tf_seed = lambda seed: None
        try:
            exit_code = cli.main(
                [
                    "run",
                    "--model",
                    "pascucci",
                    "--mode",
                    "recursive",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--pascucci_H_path",
                    str(H_path),
                    "--pascucci_S_path",
                    str(S_path),
                    "--pascucci_require_calibration",
                    "--pascucci_calibration_K",
                    "0",
                    "--pascucci_S_mul_factor",
                    "0.01",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
        finally:
            cli.set_tf_seed = original_set_tf_seed
            if original_orchestration is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_orchestration

        assert exit_code == 0
        assert calls["recursive"] == 1
        assert calls["print"] >= 1
        assert captured_params["pascucci_calibration"]["schema"] == "pascucci_ou_calibration_v1"
        runs = sorted(out_dir.glob("run_*"))
        assert len(runs) == 1
        config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
        assert config["model_name"] == "pascucci"
        assert config["pascucci_require_calibration"] is True
        assert config["pascucci_calibration_enabled"] is True
        assert config["params"]["pascucci_calibration"]["schema"] == "pascucci_ou_calibration_v1"
        assert config["params"]["pascucci_calibration"]["K"] == 0
        assert config["params"]["pascucci_calibration"]["S_transform"] == "log"
        assert config["params"]["params_H"]["a0_day"] != -1.0
        assert config["params"]["params_S"]["a0_day"] != -1.0
        assert config["params"]["pascucci_calibration"]["H_metadata"]["source_path"] == os.path.abspath(str(H_path))
        assert config["params"]["pascucci_calibration"]["S_metadata"]["source_path"] == os.path.abspath(str(S_path))
        assert config["params"]["pascucci_calibration"]["H_metadata"]["source_sha256"] == file_sha256(str(H_path))
        assert config["params"]["pascucci_calibration"]["S_metadata"]["source_sha256"] == file_sha256(str(S_path))
        assert config["params"]["pascucci_calibration"]["H_metadata"]["prepared_points"] == n_points
        assert config["params"]["pascucci_calibration"]["S_metadata"]["prepared_points"] == n_points
        assert config["params"]["pascucci_calibration"]["S_metadata"]["mul_factor"] == 0.01


def test_cli_pascucci_require_calibration_fails_before_training_without_paths() -> None:
    _assert_pascucci_tdd_contract("pascucci_cli_require_calibration_fails_before_training_without_paths")
    import types

    from . import cli

    calls = {"recursive": 0}

    def forbidden_run_recursive_training(**kwargs):
        del kwargs
        calls["recursive"] += 1
        raise AssertionError("paper-parity calibration guard must fail before training starts")

    fake_orchestration = types.ModuleType("final_recursive.orchestration")
    fake_orchestration.run_recursive_training = forbidden_run_recursive_training
    fake_orchestration.print_recursive_pass = lambda **kwargs: {}
    fake_orchestration.run_recursive_coarse_prepass = lambda **kwargs: None
    fake_orchestration.run_standard_reference = lambda **kwargs: None

    module_name = f"{cli.__package__}.orchestration"
    original_orchestration = sys.modules.get(module_name)
    original_set_tf_seed = cli.set_tf_seed
    sys.modules[module_name] = fake_orchestration
    cli.set_tf_seed = lambda seed: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            try:
                cli.main(
                    [
                        "run",
                        "--model",
                        "pascucci",
                        "--mode",
                        "recursive",
                        "--M",
                        "4",
                        "--N",
                        "2",
                        "--T_total",
                        "0.25",
                        "--block_size",
                        "0.25",
                        "--passes",
                        "1",
                        "--pascucci_require_calibration",
                        "--output_dir",
                        str(out_dir),
                    ]
                )
            except ValueError as exc:
                message = str(exc)
                assert "--pascucci_require_calibration" in message
                assert "--pascucci_H_path" in message
                assert "--pascucci_S_path" in message
            else:
                raise AssertionError("--pascucci_require_calibration must reject missing H/S paths")

            assert calls["recursive"] == 0
            assert not list(out_dir.glob("run_*"))

            try:
                cli.main(
                    [
                        "run",
                        "--model",
                        "quadratic_coupled",
                        "--mode",
                        "recursive",
                        "--M",
                        "4",
                        "--N",
                        "2",
                        "--T_total",
                        "0.25",
                        "--block_size",
                        "0.25",
                        "--passes",
                        "1",
                        "--pascucci_require_calibration",
                        "--output_dir",
                        str(out_dir),
                    ]
                )
            except ValueError as exc:
                assert "supported only with --model pascucci" in str(exc)
            else:
                raise AssertionError("--pascucci_require_calibration must not apply to quadratic_coupled")
    finally:
        cli.set_tf_seed = original_set_tf_seed
        if original_orchestration is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_orchestration


def _pascucci_psi(x: np.ndarray, d: float, x_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, np.minimum(x / d, (x_max - x) / d))).astype(np.float32)


def _pascucci_psi2(x: np.ndarray, d: float, x_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, -x / d)).astype(np.float32)


def _pascucci_psi1(x: np.ndarray, d: float, x_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (x - x_max) / d)).astype(np.float32)


def _pascucci_psi3(v: np.ndarray, d: float, v_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (v_max - v) / d)).astype(np.float32)


def _pascucci_psi4(v: np.ndarray, d: float, v_min: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (v - v_min) / d)).astype(np.float32)


def _pascucci_h(x: np.ndarray) -> np.ndarray:
    x_mean = np.mean(x, axis=0, keepdims=True)
    return np.where(x < x_mean, (x - x_mean) ** 2, 2.0 * (x - x_mean) ** 2).astype(np.float32)


def _pascucci_h_with_mean(x: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
    return np.where(x < x_mean, (x - x_mean) ** 2, 2.0 * (x - x_mean) ** 2).astype(np.float32)


def _pascucci_ou_mu_daynight(t: np.ndarray, params: dict) -> np.ndarray:
    t = t.astype(np.float32)
    omega = (2.0 * np.pi * np.arange(1, len(np.asarray(params["alpha_day"])) + 1, dtype=np.float32) / 24.0).reshape(1, -1)
    t_exp = t.reshape(-1, 1, 1)
    cos_part = np.cos(omega * t_exp)
    sin_part = np.sin(omega * t_exp)

    alpha_d = np.asarray(params["alpha_day"], dtype=np.float32).reshape(1, 1, -1)
    beta_d = np.asarray(params["beta_day"], dtype=np.float32).reshape(1, 1, -1)
    mu_d = params["a0_day"] + np.sum(alpha_d * cos_part + beta_d * sin_part, axis=-1)

    alpha_n = np.asarray(params["alpha_night"], dtype=np.float32).reshape(1, 1, -1)
    beta_n = np.asarray(params["beta_night"], dtype=np.float32).reshape(1, 1, -1)
    mu_n = params["a0_night"] + np.sum(alpha_n * cos_part + beta_n * sin_part, axis=-1)

    hour = np.mod(t, 24.0)
    return np.where((hour >= 7.0) & (hour < 19.0), mu_d, mu_n).astype(np.float32)


def _pascucci_sigmaV(X: np.ndarray, params: dict) -> np.ndarray:
    H = X[:, [1]]
    V = X[:, [2]]
    return (
        float(params["s3"]) * np.ones_like(V, dtype=np.float32)
        + float(params["s3h"]) * np.abs(H)
        + float(params["s3v"]) * np.abs(V)
        + float(params["s3k"]) * np.abs(V - np.mean(V, axis=0, keepdims=True))
    )


def _pascucci_alpha(X: np.ndarray, Z_V: np.ndarray, params: dict) -> np.ndarray:
    X_state = X[:, [3]]
    return (
        -( _pascucci_psi(X_state, float(params["d"]), float(params["x_max"])) * Z_V )
        / (2.0 * float(params["l_a"]) * np.maximum(_pascucci_sigmaV(X, params), 1.0e-7))
    ).astype(np.float32)


def test_pascucci_equation_fixtures() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.77)
    model = NN_Pascucci(
        Xi_generator_default,
        0.25,
        4,
        2,
        4,
        [5, 8, 1],
        params,
    )

    X = np.array(
        [
            [1.2, 0.4, 0.2, 4.5],
            [-0.4, -0.2, -0.1, 7.7],
            [5.2, 1.0, 0.5, -0.2],
            [3.1, -0.5, -0.3, 0.8],
        ],
        dtype=np.float32,
    )
    Y = np.array(
        [[0.1], [0.2], [-0.1], [0.3]],
        dtype=np.float32,
    )
    Z = np.array(
        [[0.1, 0.2, -0.3, 0.4], [0.2, -0.1, 0.4, -0.5], [0.3, 0.0, 0.2, -0.1], [0.4, -0.2, -0.1, 0.2]],
        dtype=np.float32,
    )
    t = np.zeros((4, 1), dtype=np.float32)

    alpha_np = _pascucci_alpha(X, Z[:, [2]], params)
    sigma_v_np = _pascucci_sigmaV(X, params)

    alpha_tf = model.alpha_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X), tf.convert_to_tensor(Z[:, [2]])).numpy()
    np.testing.assert_allclose(alpha_tf, alpha_np, rtol=1.0e-6, atol=1.0e-6)

    sigma_tf = model.sigmaV_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X)).numpy()
    np.testing.assert_allclose(sigma_tf, sigma_v_np, rtol=1.0e-6, atol=1.0e-6)

    mu_tf = model.mu_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X), tf.convert_to_tensor(Y), tf.convert_to_tensor(Z)).numpy()
    H = X[:, [1]]
    V = X[:, [2]]
    X_state = X[:, [3]]
    mu_S_mean = _pascucci_ou_mu_daynight(np.zeros((4, 1), dtype=np.float32), params["params_S"])
    mu_H_mean = _pascucci_ou_mu_daynight(np.zeros((4, 1), dtype=np.float32), params["params_H"])
    kappa_day = np.array(float(params["params_S"]["kappa_day"]))
    kappa_night = np.array(float(params["params_S"]["kappa_night"]))
    kappa = np.where((np.mod(np.zeros((4, 1), dtype=np.float32), 24.0) >= 7.0) & (np.mod(np.zeros((4, 1), dtype=np.float32), 24.0) < 19.0), kappa_day, kappa_night)
    dS_np = kappa * (mu_S_mean - X[:, [0]])

    kappa_h = np.array(float(params["params_H"]["kappa_day"]))
    kappa_night_h = np.array(float(params["params_H"]["kappa_night"]))
    kappa_h_t = np.where((np.mod(np.zeros((4, 1), dtype=np.float32), 24.0) >= 7.0) & (np.mod(np.zeros((4, 1), dtype=np.float32), 24.0) < 19.0), kappa_h, kappa_night_h)
    dH_np = kappa_h_t * (mu_H_mean - H)
    dV_np = alpha_np * _pascucci_psi(X_state, float(params["d"]), float(params["x_max"])) + params["c3"] * _pascucci_psi2(X_state, float(params["d"]), float(params["x_max"])) * _pascucci_psi3(V, float(params["d"]), float(params["v_max"])) - params["c4"] * _pascucci_psi1(X_state, float(params["d"]), float(params["x_max"])) * _pascucci_psi4(V, float(params["d"]), float(params["v_min"]))
    dX_np = V
    mu_np = np.concatenate([dS_np, dH_np, dV_np, dX_np], axis=1)
    np.testing.assert_allclose(mu_tf, mu_np, rtol=1.0e-6, atol=1.0e-6)

    sigma_tf = model.sigma_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X), tf.convert_to_tensor(Y)).numpy()
    assert sigma_tf.shape == (4, 4, 4)
    assert np.isfinite(sigma_tf).all()

    f_tf = model.f_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X), tf.convert_to_tensor(Y), tf.convert_to_tensor(Z)).numpy()
    term1 = np.exp(X[:, [0]]) * (H + V)
    term2 = float(params["l_v"]) * (V ** 2)
    term3 = float(params["l_a"]) * (alpha_np ** 2)
    term4 = float(params["c_h"]) * _pascucci_h(X_state)
    term5 = float(params["c_con"]) * _pascucci_h(H + V)
    f_np = term1 + term2 + term3 + term4 + term5
    np.testing.assert_allclose(f_tf, f_np, rtol=1.0e-6, atol=1.0e-6)

    g_tf = model.g_tf(tf.convert_to_tensor(X)).numpy()
    g_np = -float(params["gamma"]) * X_state * np.exp(X[:, [0]]) + 0.5 * float(params["omega"]) * (X_state - np.mean(X_state, axis=0, keepdims=True)) ** 2
    np.testing.assert_allclose(g_tf, g_np, rtol=1.0e-6, atol=1.0e-6)

    phi = model.phi_tf(tf.convert_to_tensor(t), tf.convert_to_tensor(X), tf.convert_to_tensor(Y), tf.convert_to_tensor(Z)).numpy()
    np.testing.assert_allclose(phi, -f_np, rtol=1.0e-6, atol=1.0e-6)
    model.close()


def test_model_spec_mean_field_moment_names() -> None:
    from .model_specs import get_model_spec

    assert get_model_spec("quadratic_coupled").moment_names == ()
    assert get_model_spec("pascucci").moment_names == (
        "mean_v",
        "mean_q",
        "mean_h_plus_v",
    )


def test_model_spec_application_metric_names() -> None:
    from .model_specs import get_model_spec

    quadratic = get_model_spec("quadratic_coupled")
    pascucci = get_model_spec("pascucci")

    assert getattr(quadratic, "application_metric_names", None) == (), (
        "quadratic_coupled must declare an empty application metric contract"
    )
    assert getattr(quadratic, "application_metric_schema", None) == "none", (
        "quadratic_coupled must keep application metrics disabled"
    )
    assert getattr(pascucci, "application_metric_schema", None) == "pascucci_application_metrics_v2", (
        "Pascucci must declare the application metric schema used by artifacts"
    )
    assert getattr(pascucci, "application_metric_names", None) == (
        "cost_J_running",
        "cost_J_terminal",
        "cost_J_total",
        "cost_J_running_cumulative",
        "cost_J_trajectory",
        "cost_J_trajectory_math",
    ), "Pascucci must declare cost J components before metrics are emitted"
    assert set(pascucci.moment_names).isdisjoint(pascucci.application_metric_names)


def _build_pascucci_unit_model(params: dict, *, M: int = 4):
    from .models import NN_Pascucci

    return NN_Pascucci(
        Xi_generator_default,
        0.25,
        M,
        2,
        4,
        [5, 8, 1],
        params,
    )


def _build_pascucci_recursive_unit_model(params: dict, *, M: int = 4):
    from .models import NN_Pascucci_Recursive

    return NN_Pascucci_Recursive(
        Xi_generator_default,
        0.25,
        M,
        2,
        4,
        [5, 8, 1],
        params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
    )


def _numpy_cost_result_pathwise(result: dict, key: str) -> np.ndarray:
    pathwise = result.get("pathwise", {})
    assert key in pathwise, f"{key} missing from pathwise cost result: {sorted(pathwise)}"
    value = pathwise[key]
    value_np = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    value_np = np.asarray(value_np, dtype=np.float32)
    assert value_np.ndim == 2 and value_np.shape[1] == 1, (key, value_np.shape)
    assert np.isfinite(value_np).all(), key
    return value_np


def test_pascucci_application_cost_functional_decomposes_to_f_plus_g() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed, tf

    reset_backend_state()
    set_seed(97)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        assert hasattr(model, "application_cost_functional"), (
            "Pascucci application cost J must be explicit; eval loss is not J"
        )
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=97)
        result = model.application_cost_functional(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            baseline_mode="controlled",
        )

        _, X, Y, Z = model.loss_function(t_batch, W_batch, Xi_batch, const_value=0.75)
        t_tf = tf.convert_to_tensor(t_batch, dtype=tf.float32)
        dt_np = t_batch[:, 1:, :] - t_batch[:, :-1, :]
        running_terms = []
        for step in range(t_batch.shape[1] - 1):
            X_step = X[:, step, :]
            Y_step = Y[:, step, :]
            Z_step = Z[:, step, :]
            t_step = t_tf[:, step, :]
            moments = model.mean_field_moments_tf(X_step)
            f_step = model.f_tf(t_step, X_step, Y_step, Z_step, moment_state=moments).numpy()
            running_terms.append(f_step * dt_np[:, step, :])
        expected_running = np.sum(np.stack(running_terms, axis=0), axis=0).astype(np.float32)
        terminal_moments = model.mean_field_moments_tf(X[:, -1, :])
        expected_terminal = model.g_tf(X[:, -1, :], moment_state=terminal_moments).numpy().astype(np.float32)
        expected_total = expected_running + expected_terminal

        np.testing.assert_allclose(
            _numpy_cost_result_pathwise(result, "cost_J_running"),
            expected_running,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _numpy_cost_result_pathwise(result, "cost_J_terminal"),
            expected_terminal,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _numpy_cost_result_pathwise(result, "cost_J_total"),
            expected_total,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        expected_cumulative = np.cumsum(np.stack(running_terms, axis=1), axis=1).astype(np.float32)
        cumulative = np.asarray(result["pathwise"]["cost_J_running_cumulative"], dtype=np.float32)
        assert cumulative.shape == (t_batch.shape[0], t_batch.shape[1] - 1, 1)
        np.testing.assert_allclose(cumulative, expected_cumulative, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(cumulative[:, -1, :], expected_running, rtol=1.0e-6, atol=1.0e-6)

        trajectory_terms = []
        terminal_terms = []
        for step in range(t_batch.shape[1]):
            X_step = X[:, step, :]
            Z_step = Z[:, step, :]
            t_step = t_tf[:, step, :]
            moments = model.mean_field_moments_tf(X_step)
            f_step = model.f_tf(
                t_step,
                X_step,
                tf.zeros_like(Y[:, step, :]),
                Z_step,
                moment_state=moments,
            ).numpy()
            if t_batch.shape[1] == 1:
                dt_step = np.zeros_like(t_batch[:, step, :])
            elif step == 0:
                dt_step = t_batch[:, 1, :] - t_batch[:, 0, :]
            else:
                dt_step = t_batch[:, step, :] - t_batch[:, step - 1, :]
            trajectory_terms.append(f_step * dt_step)
            terminal_terms.append(model.g_tf(X_step, moment_state=moments).numpy())
        expected_trajectory = (
            np.cumsum(np.stack(trajectory_terms, axis=1), axis=1)
            + np.stack(terminal_terms, axis=1)
        ).astype(np.float32)
        trajectory = np.asarray(result["pathwise"]["cost_J_trajectory"], dtype=np.float32)
        assert trajectory.shape == (t_batch.shape[0], t_batch.shape[1], 1)
        np.testing.assert_allclose(trajectory, expected_trajectory, rtol=1.0e-6, atol=1.0e-6)

        terminal_trace = np.stack(terminal_terms, axis=1).astype(np.float32)
        integral_to_time = np.concatenate(
            [np.zeros((t_batch.shape[0], 1, 1), dtype=np.float32), expected_cumulative],
            axis=1,
        )
        expected_trajectory_math = (integral_to_time + terminal_trace).astype(np.float32)
        trajectory_math = np.asarray(result["pathwise"]["cost_J_trajectory_math"], dtype=np.float32)
        assert trajectory_math.shape == (t_batch.shape[0], t_batch.shape[1], 1)
        np.testing.assert_allclose(trajectory_math, expected_trajectory_math, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(trajectory_math[:, 0, :], terminal_trace[:, 0, :], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(trajectory_math[:, -1, :], expected_total, rtol=1.0e-6, atol=1.0e-6)
        assert not np.allclose(trajectory[:, -1, :], expected_total, rtol=1.0e-6, atol=1.0e-6), (
            "cost_J_trajectory intentionally follows raw/to_ema plotmaker.ipynb; "
            "use cost_J_trajectory_math for cost accounting"
        )
    finally:
        model.close()


def test_pascucci_application_cost_trajectory_matches_plotmaker_nonuniform_grid() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import require_tensorflow, reset_backend_state, set_seed

    tf = require_tensorflow()
    reset_backend_state()
    set_seed(303)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=3)

    try:
        M = 3
        t_grid = np.asarray([0.0, 0.05, 0.20, 0.25], dtype=np.float32)
        t = np.tile(t_grid.reshape(1, -1, 1), (M, 1, 1)).astype(np.float32)
        X = np.zeros((M, len(t_grid), 4), dtype=np.float32)
        for i in range(M):
            X[i, :, 0] = 0.20 + 0.04 * i + np.asarray([0.00, 0.01, 0.03, 0.06], dtype=np.float32)
            X[i, :, 1] = -0.10 + 0.03 * i + np.asarray([0.00, 0.02, 0.02, 0.04], dtype=np.float32)
            X[i, :, 2] = 0.10 + 0.02 * i + np.asarray([0.00, 0.05, 0.04, 0.07], dtype=np.float32)
            X[i, :, 3] = 3.00 + 0.20 * i + np.asarray([0.00, 0.10, 0.15, 0.25], dtype=np.float32)
        Y = np.zeros((M, len(t_grid), 1), dtype=np.float32)
        Z = np.zeros((M, len(t_grid), 4), dtype=np.float32)
        Z[:, :, 2] = 0.05

        result = model.application_cost_from_path(t, X, Y, Z)

        f_slices = []
        g_slices = []
        t_tf = tf.constant(t, dtype=tf.float32)
        X_tf = tf.constant(X, dtype=tf.float32)
        Y_tf = tf.constant(Y, dtype=tf.float32)
        Z_tf = tf.constant(Z, dtype=tf.float32)
        for step in range(len(t_grid)):
            X_step = X_tf[:, step, :]
            moments = model.mean_field_moments_tf(X_step)
            f_slices.append(
                model.f_tf(
                    t_tf[:, step, :],
                    X_step,
                    Y_tf[:, step, :],
                    Z_tf[:, step, :],
                    moment_state=moments,
                ).numpy()[:, 0]
            )
            g_slices.append(model.g_tf(X_step, moment_state=moments).numpy()[:, 0])
        f_vals = np.stack(f_slices, axis=1)
        g_vals = np.stack(g_slices, axis=1)
        plotmaker_dt = np.concatenate([[np.diff(t_grid)[0]], np.diff(t_grid)]).astype(np.float32)
        expected_plotmaker = (
            np.cumsum(f_vals * plotmaker_dt.reshape(1, -1), axis=1)
            + g_vals
        ).reshape(M, len(t_grid), 1)

        trajectory = np.asarray(result["pathwise"]["cost_J_trajectory"], dtype=np.float32)
        trajectory_math = np.asarray(result["pathwise"]["cost_J_trajectory_math"], dtype=np.float32)
        total = np.asarray(result["pathwise"]["cost_J_total"], dtype=np.float32)

        np.testing.assert_allclose(trajectory, expected_plotmaker, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(trajectory_math[:, -1, :], total, rtol=1.0e-6, atol=1.0e-6)
        assert not np.allclose(trajectory[:, -1, :], total, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_pascucci_application_cost_trajectory_uses_per_time_mean_field_oracle() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import require_tensorflow, reset_backend_state, set_seed

    tf = require_tensorflow()
    reset_backend_state()
    set_seed(404)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        M = 4
        t_grid = np.asarray([0.0, 0.04, 0.12, 0.25], dtype=np.float32)
        t = np.tile(t_grid.reshape(1, -1, 1), (M, 1, 1)).astype(np.float32)
        X = np.zeros((M, len(t_grid), 4), dtype=np.float32)
        path_offsets = np.arange(M, dtype=np.float32)
        q_offsets = np.asarray([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
        for step in range(len(t_grid)):
            X[:, step, 0] = 0.10 + 0.03 * step + 0.02 * path_offsets
            X[:, step, 1] = -2.00 + 2.00 * step + 0.50 * path_offsets
            X[:, step, 2] = 0.10 + 1.00 * step + 0.40 * path_offsets
            X[:, step, 3] = 1.00 + 3.00 * step + q_offsets
        Y = np.zeros((M, len(t_grid), 1), dtype=np.float32)
        Z = np.zeros((M, len(t_grid), 4), dtype=np.float32)
        Z[:, :, 2] = 0.10 + 0.02 * np.arange(len(t_grid), dtype=np.float32).reshape(1, -1)

        result = model.application_cost_from_path(t, X, Y, Z)

        t_tf = tf.constant(t, dtype=tf.float32)
        X_tf = tf.constant(X, dtype=tf.float32)
        Y_tf = tf.constant(Y, dtype=tf.float32)
        Z_tf = tf.constant(Z, dtype=tf.float32)
        f_slices = []
        g_slices = []
        for step in range(len(t_grid)):
            X_step = X_tf[:, step, :]
            moments = model.mean_field_moments_tf(X_step)
            f_slices.append(
                model.f_tf(
                    t_tf[:, step, :],
                    X_step,
                    Y_tf[:, step, :],
                    Z_tf[:, step, :],
                    moment_state=moments,
                ).numpy()[:, 0]
            )
            g_slices.append(model.g_tf(X_step, moment_state=moments).numpy()[:, 0])
        f_vals = np.stack(f_slices, axis=1)
        g_vals = np.stack(g_slices, axis=1)
        plotmaker_dt = np.concatenate([[np.diff(t_grid)[0]], np.diff(t_grid)]).astype(np.float32)
        per_time_oracle = (
            np.cumsum(f_vals * plotmaker_dt.reshape(1, -1), axis=1)
            + g_vals
        ).reshape(M, len(t_grid), 1)

        t_flat = t.reshape(-1, 1)
        X_flat = X.reshape(-1, 4)
        Y_flat = Y.reshape(-1, 1)
        Z_flat = Z.reshape(-1, 4)
        f_flat = model.f_tf(
            tf.constant(t_flat, dtype=tf.float32),
            tf.constant(X_flat, dtype=tf.float32),
            tf.constant(Y_flat, dtype=tf.float32),
            tf.constant(Z_flat, dtype=tf.float32),
        ).numpy().reshape(M, len(t_grid))
        g_flat = model.g_tf(tf.constant(X_flat, dtype=tf.float32)).numpy().reshape(M, len(t_grid))
        flattened_helper = (
            np.cumsum(f_flat * plotmaker_dt.reshape(1, -1), axis=1)
            + g_flat
        ).reshape(M, len(t_grid), 1)

        trajectory = np.asarray(result["pathwise"]["cost_J_trajectory"], dtype=np.float32)
        np.testing.assert_allclose(trajectory, per_time_oracle, rtol=1.0e-6, atol=1.0e-6)
        assert np.max(np.abs(flattened_helper - per_time_oracle)) > 1.0e-3, (
            "The raw notebook helper cannot be used naively after flattening "
            "(path, time) because mean-field moments must be evaluated per time slice."
        )
    finally:
        model.close()


def test_pascucci_application_cost_summary_has_quantiles_and_metadata() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(101)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        assert hasattr(model, "application_cost_functional"), (
            "Pascucci application cost J must expose a pathwise summary"
        )
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=101)
        result = model.application_cost_functional(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            baseline_mode="controlled",
        )
        assert result["schema"] == "pascucci_application_metrics_v2"
        assert result["metadata"]["baseline_mode"] == "controlled"
        assert result["metadata"]["aggregation"] == "left_riemann_f_plus_terminal_g"
        assert result["metadata"]["control_law"] == "alpha_tf"

        summary = result["summary"]
        for metric_name in (
            "cost_J_running",
            "cost_J_terminal",
            "cost_J_total",
            "cost_J_running_cumulative",
            "cost_J_trajectory",
            "cost_J_trajectory_math",
        ):
            if metric_name in ("cost_J_running_cumulative", "cost_J_trajectory", "cost_J_trajectory_math"):
                trace = np.asarray(result["pathwise"][metric_name], dtype=np.float32)
                assert trace.ndim == 3
                assert trace.shape[1] > 0
                flat = trace[:, -1, :].reshape(-1)
            else:
                pathwise = _numpy_cost_result_pathwise(result, metric_name)
                flat = pathwise[:, 0]
            expected = {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "q05": float(np.quantile(flat, 0.05)),
                "q50": float(np.quantile(flat, 0.50)),
                "q95": float(np.quantile(flat, 0.95)),
            }
            for suffix, expected_value in expected.items():
                key = f"{metric_name}_{suffix}"
                assert key in summary, f"{key} missing from summary: {sorted(summary)}"
                assert isinstance(summary[key], float)
                assert np.isfinite(summary[key])
                np.testing.assert_allclose(summary[key], expected_value, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_pascucci_application_cost_summary_handles_zero_step_cumulative() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(1011)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=3)

    try:
        pathwise = {
            "cost_J_running": np.zeros((3, 1), dtype=np.float32),
            "cost_J_terminal": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "cost_J_total": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "cost_J_running_cumulative": np.zeros((3, 0, 1), dtype=np.float32),
            "cost_J_trajectory": np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32),
            "cost_J_trajectory_math": np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32),
        }
        summary = model._application_summary_np(pathwise)
        for suffix in ("mean", "std", "q05", "q50", "q95"):
            key = f"cost_J_running_cumulative_{suffix}"
            assert key in summary
            assert isinstance(summary[key], float)
            np.testing.assert_allclose(summary[key], 0.0, rtol=0.0, atol=0.0)
        assert summary["cost_J_trajectory_mean"] == 2.0
    finally:
        model.close()


def test_pascucci_application_cost_functional_restores_const_state() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(102)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=102)
        before_const = np.float32(model.const)
        before_const_tf = np.float32(model.const_tf.numpy())
        model.application_cost_functional(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.25,
            baseline_mode="uncontrolled",
        )
        np.testing.assert_allclose(np.float32(model.const), before_const, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.float32(model.const_tf.numpy()), before_const_tf, rtol=0.0, atol=0.0)
    finally:
        model.close()


def test_pascucci_application_cost_from_path_rejects_invalid_baseline_mode() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(104)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=104)
        _, X, Y, Z = model.loss_function(t_batch, W_batch, Xi_batch, const_value=0.75)
        try:
            model.application_cost_from_path(
                t_batch,
                X,
                Y,
                Z,
                const_value=0.75,
                baseline_mode="bad-mode",
            )
        except ValueError as exc:
            assert "baseline_mode must be 'controlled' or 'uncontrolled'" in str(exc)
        else:
            raise AssertionError("application_cost_from_path should reject invalid baseline_mode")
    finally:
        model.close()


def test_pascucci_uncontrolled_baseline_is_paired_and_alpha_zero() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(103)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = _build_pascucci_recursive_unit_model(params, M=4)

    try:
        assert hasattr(model, "application_cost_functional"), (
            "Controlled/uncontrolled baseline requires an explicit application cost helper"
        )
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=103)
        controlled = model.application_cost_functional(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            baseline_mode="controlled",
        )
        uncontrolled = model.application_cost_functional(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            baseline_mode="uncontrolled",
        )

        assert controlled["metadata"]["paired_inputs"] == "same_t_W_Xi"
        assert uncontrolled["metadata"]["paired_inputs"] == "same_t_W_Xi"
        assert controlled["metadata"]["baseline_mode"] == "controlled"
        assert uncontrolled["metadata"]["baseline_mode"] == "uncontrolled"
        assert uncontrolled["metadata"]["control_law"] == "alpha_zero"
        alpha_pathwise = np.asarray(uncontrolled["pathwise"]["alpha"], dtype=np.float32)
        assert alpha_pathwise.shape == (Xi_batch.shape[0], t_batch.shape[1] - 1, 1)
        np.testing.assert_allclose(alpha_pathwise, 0.0, rtol=0.0, atol=0.0)
        assert np.isfinite(_numpy_cost_result_pathwise(controlled, "cost_J_total")).all()
        assert np.isfinite(_numpy_cost_result_pathwise(uncontrolled, "cost_J_total")).all()
    finally:
        model.close()


def test_pascucci_physical_tail_and_stitching_diagnostics_contract() -> None:
    try:
        from .application_metrics import summarize_pascucci_stitched_diagnostics
    except ImportError as exc:
        raise AssertionError(
            "Pascucci application diagnostics must have an explicit summary helper"
        ) from exc

    params = _default_pascucci_params(const=0.75)
    stitched = {
        "t": np.asarray([[[0.0], [0.25], [0.50], [0.75]]], dtype=np.float32),
        "X": np.asarray(
            [
                [
                    [0.0, 0.0, -2.50, -0.50],
                    [0.0, 0.0, -2.10, 0.00],
                    [0.0, 0.0, 2.25, 10.50],
                    [0.0, 0.0, 2.50, 11.00],
                ]
            ],
            dtype=np.float32,
        ),
        "q_lower_violation": np.asarray([[0.50], [0.00], [0.00], [0.00]], dtype=np.float32),
        "q_upper_violation": np.asarray([[0.00], [0.00], [0.50], [1.00]], dtype=np.float32),
        "v_lower_violation": np.asarray([[0.50], [0.10], [0.00], [0.00]], dtype=np.float32),
        "v_upper_violation": np.asarray([[0.00], [0.00], [0.25], [0.50]], dtype=np.float32),
        "stitch_X_boundary_abs_jump": np.asarray(
            [
                [[0.00, 0.00, 0.10, 0.20]],
                [[0.00, 0.00, 0.30, 1.25]],
            ],
            dtype=np.float32,
        ),
    }
    blocks = [
        {"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25},
        {"idx": 1, "t_start": 0.25, "t_end": 0.50, "T_block": 0.25},
        {"idx": 2, "t_start": 0.50, "t_end": 0.75, "T_block": 0.25},
    ]
    summary = summarize_pascucci_stitched_diagnostics(
        stitched=stitched,
        blocks=blocks,
        params=params,
    )
    for key in (
        "q_lower_violation_mean",
        "q_lower_violation_max",
        "q_lower_violation_q95",
        "q_lower_violation_rate",
        "v_upper_violation_mean",
        "v_upper_violation_max",
        "v_upper_violation_q95",
        "v_upper_violation_rate",
        "stitch_X_boundary_max_abs_jump",
        "stitch_X_boundary_mean_abs_jump",
    ):
        assert key in summary, f"{key} missing from diagnostic summary: {sorted(summary)}"
        assert isinstance(summary[key], float)
        assert np.isfinite(summary[key])
    assert summary["q_upper_violation_max"] == 1.0
    assert summary["v_lower_violation_rate"] == 0.5
    np.testing.assert_allclose(summary["stitch_X_boundary_max_abs_jump"], 1.25)
    np.testing.assert_allclose(summary["stitch_X_boundary_mean_abs_jump"], 0.23125)

    duplicate_boundary = dict(stitched)
    duplicate_boundary.pop("stitch_X_boundary_abs_jump")
    duplicate_boundary["t"] = np.asarray([[[0.0], [0.25], [0.25], [0.50]]], dtype=np.float32)
    duplicate_boundary["X"] = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.5, 3.0],
                [0.0, 0.0, 0.5, 4.0],
            ]
        ],
        dtype=np.float32,
    )
    duplicate_summary = summarize_pascucci_stitched_diagnostics(
        stitched=duplicate_boundary,
        blocks=blocks[:2],
        params=params,
    )
    np.testing.assert_allclose(duplicate_summary["stitch_X_boundary_max_abs_jump"], 1.0)
    np.testing.assert_allclose(duplicate_summary["stitch_X_boundary_mean_abs_jump"], 0.375)

    pathwise_tail = dict(stitched)
    pathwise_tail["X"] = np.asarray(
        [
            [[0.0, 0.0, 0.0, 9.0], [0.0, 0.0, 0.0, 9.0]],
            [[0.0, 0.0, 0.0, 9.0], [0.0, 0.0, 0.0, 20.0]],
        ],
        dtype=np.float32,
    )
    for key in (
        "q_lower_violation",
        "q_upper_violation",
        "v_lower_violation",
        "v_upper_violation",
    ):
        pathwise_tail[key] = np.zeros((2, 1), dtype=np.float32)
    tail_summary = summarize_pascucci_stitched_diagnostics(
        stitched=pathwise_tail,
        blocks=[{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}],
        params=params,
    )
    np.testing.assert_allclose(tail_summary["q_upper_violation_max"], 10.0)
    np.testing.assert_allclose(tail_summary["q_upper_violation_rate"], 0.25)

    missing_jump = dict(stitched)
    missing_jump.pop("stitch_X_boundary_abs_jump")
    try:
        summarize_pascucci_stitched_diagnostics(
            stitched=missing_jump,
            blocks=blocks,
            params=params,
        )
    except ValueError as exc:
        assert "stitch_X_boundary_abs_jump" in str(exc)
        assert "duplicate boundary times" in str(exc)
    else:
        raise AssertionError("unique stitched boundaries must not imply zero boundary jump")


def test_pascucci_controlled_uncontrolled_comparison_is_paired_and_sign_safe() -> None:
    try:
        from .application_metrics import summarize_controlled_uncontrolled_comparison
    except ImportError as exc:
        raise AssertionError(
            "Controlled/uncontrolled artifacts need an explicit paired comparison helper"
        ) from exc

    controlled = {
        "metadata": {"baseline_mode": "controlled", "paired_inputs": "stitched_XYZ"},
        "pathwise": {
            "cost_J_running": np.asarray([[1.0], [3.0], [-2.0]], dtype=np.float32),
            "cost_J_terminal": np.asarray([[0.5], [-4.0], [1.0]], dtype=np.float32),
            "cost_J_total": np.asarray([[1.5], [-1.0], [-1.0]], dtype=np.float32),
        },
    }
    uncontrolled = {
        "metadata": {"baseline_mode": "uncontrolled", "paired_inputs": "same_t_W_Xi"},
        "pathwise": {
            "cost_J_running": np.asarray([[2.0], [1.0], [-3.0]], dtype=np.float32),
            "cost_J_terminal": np.asarray([[1.5], [-5.0], [3.0]], dtype=np.float32),
            "cost_J_total": np.asarray([[3.5], [-4.0], [0.0]], dtype=np.float32),
        },
    }

    comparison = summarize_controlled_uncontrolled_comparison(
        controlled=controlled,
        uncontrolled=uncontrolled,
    )
    assert comparison["paired_pathwise_samples"] is True
    assert comparison["paired_sample_count"] == 3
    assert comparison["controlled_baseline_mode"] == "controlled"
    assert comparison["uncontrolled_baseline_mode"] == "uncontrolled"
    assert comparison["controlled_paired_inputs"] == "stitched_XYZ"
    assert comparison["uncontrolled_paired_inputs"] == "same_t_W_Xi"
    assert comparison["same_input_source"] is False

    delta_total = controlled["pathwise"]["cost_J_total"] - uncontrolled["pathwise"]["cost_J_total"]
    np.testing.assert_allclose(
        comparison["delta_cost_J_total_mean"],
        float(np.mean(delta_total)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        comparison["delta_cost_J_total_q50"],
        float(np.quantile(delta_total.reshape(-1), 0.50)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        comparison["delta_cost_J_total_abs_q95"],
        float(np.quantile(np.abs(delta_total.reshape(-1)), 0.95)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        comparison["cost_J_total_control_win_rate"],
        float(np.mean(delta_total.reshape(-1) < 0.0)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert not any("ratio" in key or "percent" in key for key in comparison), sorted(comparison)


def test_pascucci_application_alpha_summary_is_controlled_only_and_plot_ready() -> None:
    try:
        from .application_metrics import summarize_application_alpha
    except ImportError as exc:
        raise AssertionError("Pascucci alpha diagnostics need an explicit summary helper") from exc

    alpha = np.asarray(
        [
            [[-2.0], [-1.0], [0.0]],
            [[1.0], [2.0], [3.0]],
        ],
        dtype=np.float32,
    )
    summary = summarize_application_alpha(alpha, baseline_mode="controlled")
    flat = alpha.reshape(-1)
    expected = {
        "baseline_mode": "controlled",
        "sample_count": int(flat.size),
        "alpha_mean": float(np.mean(flat)),
        "alpha_std": float(np.std(flat)),
        "alpha_q05": float(np.quantile(flat, 0.05)),
        "alpha_q50": float(np.quantile(flat, 0.50)),
        "alpha_q95": float(np.quantile(flat, 0.95)),
        "alpha_abs_mean": float(np.mean(np.abs(flat))),
        "alpha_abs_q95": float(np.quantile(np.abs(flat), 0.95)),
    }
    for key, value in expected.items():
        assert key in summary, f"{key} missing from alpha summary: {sorted(summary)}"
        if isinstance(value, str):
            assert summary[key] == value
        else:
            np.testing.assert_allclose(summary[key], value, rtol=1.0e-6, atol=1.0e-6)

    uncontrolled = summarize_application_alpha(np.zeros_like(alpha), baseline_mode="uncontrolled")
    assert uncontrolled["baseline_mode"] == "uncontrolled"
    np.testing.assert_allclose(uncontrolled["alpha_abs_mean"], 0.0, rtol=0.0, atol=0.0)


def test_pascucci_stitched_diagnostics_report_component_boundary_drift() -> None:
    from .application_metrics import summarize_pascucci_stitched_diagnostics

    params = _default_pascucci_params(const=0.75)
    stitched = {
        "t": np.asarray([[[0.0], [0.25], [0.50]]], dtype=np.float32),
        "X": np.zeros((2, 3, 4), dtype=np.float32),
        "q_lower_violation": np.zeros((3, 1), dtype=np.float32),
        "q_upper_violation": np.zeros((3, 1), dtype=np.float32),
        "v_lower_violation": np.zeros((3, 1), dtype=np.float32),
        "v_upper_violation": np.zeros((3, 1), dtype=np.float32),
        "stitch_X_boundary_signed_jump": np.asarray(
            [
                [[0.50, -0.25, 0.00, 2.00], [-0.50, 0.25, 1.00, 0.00]],
            ],
            dtype=np.float32,
        ),
        "stitch_X_boundary_abs_jump": np.asarray(
            [
                [[0.50, 0.25, 0.00, 2.00], [0.50, 0.25, 1.00, 0.00]],
            ],
            dtype=np.float32,
        ),
    }
    blocks = [
        {"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25},
        {"idx": 1, "t_start": 0.25, "t_end": 0.50, "T_block": 0.25},
    ]
    summary = summarize_pascucci_stitched_diagnostics(
        stitched=stitched,
        blocks=blocks,
        params=params,
        state_labels=("S", "H", "V", "Q"),
    )

    np.testing.assert_allclose(summary["stitch_X_boundary_signed_mean_jump_S"], 0.0)
    np.testing.assert_allclose(summary["stitch_X_boundary_max_abs_jump_V"], 1.0)
    np.testing.assert_allclose(summary["stitch_X_boundary_max_abs_jump_Q"], 2.0)
    np.testing.assert_allclose(
        summary["stitch_X_boundary_abs_q95_jump_Q"],
        float(np.quantile(np.asarray([2.0, 0.0], dtype=np.float32), 0.95)),
    )


def test_pascucci_application_pass_stability_uses_same_grid_pathwise_deltas() -> None:
    try:
        from .application_metrics import summarize_application_pass_stability
    except ImportError as exc:
        raise AssertionError("Pascucci pass stability needs an explicit summary helper") from exc

    t = np.asarray([[[0.0], [0.5], [1.0]], [[0.0], [0.5], [1.0]]], dtype=np.float32)
    previous_stitched = {
        "t": t,
        "Y": np.zeros((2, 3, 1), dtype=np.float32),
        "Z": np.zeros((2, 3, 4), dtype=np.float32),
    }
    current_stitched = {
        "t": t.copy(),
        "Y": np.asarray([[[0.0], [0.5], [1.0]], [[0.0], [0.0], [0.0]]], dtype=np.float32),
        "Z": np.zeros((2, 3, 4), dtype=np.float32),
    }
    current_stitched["Z"][:, :, 2] = np.asarray(
        [[0.0, 0.25, 0.50], [0.0, 0.00, 0.25]], dtype=np.float32
    )
    previous_pathwise = {
        "controlled_cost_J_total": np.asarray([[1.0], [2.0]], dtype=np.float32),
        "controlled_alpha": np.asarray([[[-1.0], [0.0]], [[1.0], [2.0]]], dtype=np.float32),
    }
    current_pathwise = {
        "controlled_cost_J_total": np.asarray([[1.5], [1.0]], dtype=np.float32),
        "controlled_alpha": np.asarray([[[-0.5], [0.5]], [[1.5], [1.0]]], dtype=np.float32),
    }

    stability = summarize_application_pass_stability(
        previous_stitched=previous_stitched,
        current_stitched=current_stitched,
        previous_pathwise=previous_pathwise,
        current_pathwise=current_pathwise,
        z_v_index=2,
    )
    np.testing.assert_allclose(
        stability["pass_vs_prev_Y_mae"],
        float(np.mean(np.abs(current_stitched["Y"] - previous_stitched["Y"]))),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        stability["pass_vs_prev_Z_V_mae"],
        float(np.mean(np.abs(current_stitched["Z"][:, :, 2] - previous_stitched["Z"][:, :, 2]))),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        stability["pass_vs_prev_cost_J_total_mean_abs_delta"],
        float(np.mean(np.abs(current_pathwise["controlled_cost_J_total"] - previous_pathwise["controlled_cost_J_total"]))),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        stability["pass_vs_prev_alpha_abs_mean_delta"],
        float(np.mean(np.abs(current_pathwise["controlled_alpha"] - previous_pathwise["controlled_alpha"]))),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_pascucci_cost_profile_default_is_exp_and_json_safe() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cost_profile_default_is_exp_and_json_safe")
    from .model_specs import get_model_spec

    quadratic_params = get_model_spec("quadratic_coupled").build_default_params()
    assert "pascucci_cost_profile" not in quadratic_params
    assert "pascucci_cost_offset" not in quadratic_params

    params = get_model_spec("pascucci").build_default_params()
    assert params["pascucci_cost_profile"] == "exp"
    assert float(params["pascucci_cost_offset"]) == 0.0
    _assert_json_roundtrip(
        {
            "pascucci_cost_profile": params["pascucci_cost_profile"],
            "pascucci_cost_offset": float(params["pascucci_cost_offset"]),
        }
    )


def test_pascucci_cost_profile_exp_minus_offset_changes_only_running_cost() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cost_profile_exp_minus_offset_changes_only_running_cost")
    from .model_specs import get_model_spec
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    base_params = spec.build_default_params(const=0.77)
    base_params["pascucci_cost_profile"] = "exp"
    base_params["pascucci_cost_offset"] = np.float32(0.0)
    offset_params = spec.build_default_params(const=0.77)
    offset_params["pascucci_cost_profile"] = "exp_minus_offset"
    offset_params["pascucci_cost_offset"] = np.float32(0.12)

    base_model = _build_pascucci_unit_model(base_params)
    offset_model = _build_pascucci_unit_model(offset_params)
    try:
        X = np.array(
            [
                [1.2, 0.4, 0.2, 4.5],
                [-0.4, -0.2, -0.1, 7.7],
                [0.3, 1.0, -0.5, 2.2],
                [0.0, -0.5, 0.3, 0.8],
            ],
            dtype=np.float32,
        )
        Y = np.array([[0.1], [0.2], [-0.1], [0.3]], dtype=np.float32)
        Z = np.zeros((4, 4), dtype=np.float32)
        t = np.zeros((4, 1), dtype=np.float32)

        t_tf = tf.convert_to_tensor(t)
        X_tf = tf.convert_to_tensor(X)
        Y_tf = tf.convert_to_tensor(Y)
        Z_tf = tf.convert_to_tensor(Z)
        base_f = base_model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy()
        offset_f = offset_model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy()

        H_plus_V = X[:, [1]] + X[:, [2]]
        expected_delta = -float(offset_params["pascucci_cost_offset"]) * H_plus_V
        np.testing.assert_allclose(offset_f - base_f, expected_delta, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            offset_model.phi_tf(t_tf, X_tf, Y_tf, Z_tf).numpy(),
            -offset_f,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            offset_model.g_tf(X_tf).numpy(),
            base_model.g_tf(X_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    finally:
        base_model.close()
        offset_model.close()


def test_pascucci_cost_profile_rejects_unknown_profile() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cost_profile_rejects_unknown_profile")
    from .model_specs import get_model_spec

    params = get_model_spec("pascucci").build_default_params()
    params["pascucci_cost_profile"] = "exp_plus_untracked_magic"
    params["pascucci_cost_offset"] = np.float32(0.0)
    model = None
    try:
        model = _build_pascucci_unit_model(params)
    except ValueError as exc:
        assert "pascucci_cost_profile" in str(exc)
    else:
        raise AssertionError("unknown Pascucci cost profile should fail before model construction succeeds")
    finally:
        if model is not None:
            model.close()


def test_pascucci_recursive_cost_profile_matches_standard_formula() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_recursive_cost_profile_matches_standard_formula")
    from .model_specs import get_model_spec
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    base_params = spec.build_default_params(const=0.77)
    base_params["pascucci_cost_profile"] = "exp"
    base_params["pascucci_cost_offset"] = np.float32(0.0)
    offset_params = spec.build_default_params(const=0.77)
    offset_params["pascucci_cost_profile"] = "exp_minus_offset"
    offset_params["pascucci_cost_offset"] = np.float32(0.12)

    base_model = _build_pascucci_recursive_unit_model(base_params)
    offset_model = _build_pascucci_recursive_unit_model(offset_params)
    try:
        X = np.array(
            [
                [0.2, 0.4, 0.2, 4.5],
                [-0.4, -0.2, -0.1, 7.7],
                [0.3, 1.0, -0.5, 2.2],
                [0.0, -0.5, 0.3, 0.8],
            ],
            dtype=np.float32,
        )
        Y = np.array([[0.1], [0.2], [-0.1], [0.3]], dtype=np.float32)
        Z = np.zeros((4, 4), dtype=np.float32)
        t = np.zeros((4, 1), dtype=np.float32)
        t_tf = tf.convert_to_tensor(t)
        X_tf = tf.convert_to_tensor(X)
        Y_tf = tf.convert_to_tensor(Y)
        Z_tf = tf.convert_to_tensor(Z)
        base_f = base_model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy()
        offset_f = offset_model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy()

        expected_delta = -float(offset_params["pascucci_cost_offset"]) * (X[:, [1]] + X[:, [2]])
        np.testing.assert_allclose(offset_f - base_f, expected_delta, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            offset_model.phi_tf(t_tf, X_tf, Y_tf, Z_tf).numpy(),
            -offset_f,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            offset_model.g_tf(X_tf).numpy(),
            base_model.g_tf(X_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    finally:
        base_model.close()
        offset_model.close()


def test_pascucci_physical_constraint_diagnostics_q_v_are_model_owned() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_physical_constraint_diagnostics_q_v_are_model_owned")
    from .model_specs import get_model_spec
    from .tf_backend import tf

    params = get_model_spec("pascucci").build_default_params()
    model = _build_pascucci_unit_model(params)
    try:
        X = np.array(
            [
                [0.0, 0.0, -2.5, -0.25],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 2.25, 10.5],
                [0.0, 0.0, -1.0, 10.0],
            ],
            dtype=np.float32,
        )
        diagnostics = model.physical_constraint_diagnostics_tf(tf.convert_to_tensor(X))
        assert set(diagnostics) == {
            "q_lower_violation",
            "q_upper_violation",
            "v_lower_violation",
            "v_upper_violation",
        }
        expected = {
            "q_lower_violation": np.array([[0.25], [0.0], [0.0], [0.0]], dtype=np.float32),
            "q_upper_violation": np.array([[0.0], [0.0], [0.5], [0.0]], dtype=np.float32),
            "v_lower_violation": np.array([[0.5], [0.0], [0.0], [0.0]], dtype=np.float32),
            "v_upper_violation": np.array([[0.0], [0.0], [0.25], [0.0]], dtype=np.float32),
        }
        for key, expected_value in expected.items():
            value = diagnostics[key].numpy()
            assert value.shape == (4, 1), key
            assert np.isfinite(value).all(), key
            assert np.all(value >= 0.0), key
            np.testing.assert_allclose(value, expected_value, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_pascucci_q_v_barrier_drift_pushes_toward_physical_domain() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_q_v_barrier_drift_pushes_toward_physical_domain")
    from .model_specs import get_model_spec
    from .tf_backend import tf

    params = get_model_spec("pascucci").build_default_params()
    model = _build_pascucci_unit_model(params, M=3)
    try:
        X = np.array(
            [
                [0.0, 0.0, -2.0, -0.5],
                [0.0, 0.0, 0.0, 10.5],
                [0.0, 0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        t = np.zeros((3, 1), dtype=np.float32)
        Y = np.zeros((3, 1), dtype=np.float32)
        Z = np.zeros((3, 4), dtype=np.float32)
        mu = model.mu_tf(
            tf.convert_to_tensor(t),
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(Y),
            tf.convert_to_tensor(Z),
        ).numpy()
        dV = mu[:, 2]
        assert dV[0] > 0.0, dV
        assert dV[1] < 0.0, dV
        assert abs(float(dV[2])) <= 1.0e-7, dV
    finally:
        model.close()


def test_pascucci_q_v_barrier_sweep_with_nonzero_z_v() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_q_v_barrier_sweep_with_nonzero_z_v")
    from .model_specs import get_model_spec
    from .tf_backend import tf

    params = get_model_spec("pascucci").build_default_params()

    def expected_dv_np(X_np: np.ndarray, Z_np: np.ndarray) -> np.ndarray:
        H = X_np[:, [1]]
        V = X_np[:, [2]]
        q = X_np[:, [3]]
        mean_v = np.mean(V, axis=0, keepdims=True)
        d = np.float32(params["d"])
        x_max = np.float32(params["x_max"])
        v_min = np.float32(params["v_min"])
        v_max = np.float32(params["v_max"])
        psi = np.maximum(0.0, np.minimum(1.0, np.minimum(q / d, (x_max - q) / d))).astype(np.float32)
        psi1 = np.maximum(0.0, np.minimum(1.0, (q - x_max) / d)).astype(np.float32)
        psi2 = np.maximum(0.0, np.minimum(1.0, -q / d)).astype(np.float32)
        psi3 = np.maximum(0.0, np.minimum(1.0, (v_max - V) / d)).astype(np.float32)
        psi4 = np.maximum(0.0, np.minimum(1.0, (V - v_min) / d)).astype(np.float32)
        sigma_v = (
            np.float32(params["s3"])
            + np.float32(params["s3h"]) * np.abs(H)
            + np.float32(params["s3v"]) * np.abs(V)
            + np.float32(params["s3k"]) * np.abs(V - mean_v)
        ).astype(np.float32)
        alpha = -(psi * Z_np[:, [2]]) / (
            2.0 * np.float32(params["l_a"]) * np.maximum(sigma_v, np.float32(1.0e-7))
        )
        return (
            alpha * psi
            + np.float32(params["c3"]) * psi2 * psi3
            - np.float32(params["c4"]) * psi1 * psi4
        )[:, 0].astype(np.float32)

    model = _build_pascucci_unit_model(params, M=12)
    try:
        q_lower = -0.25
        q_upper = float(params["x_max"]) + 0.25
        q_mid = 0.5 * float(params["x_max"])
        v_mid = 0.0
        cases = []
        expected_signs = []
        for z_v in (-2.0, -1.0, 1.0, 2.0):
            cases.append([0.0, 0.0, v_mid, q_lower])
            expected_signs.append(1.0)
            cases.append([0.0, 0.0, v_mid, q_upper])
            expected_signs.append(-1.0)
            cases.append([0.0, 0.0, v_mid, q_mid])
            expected_signs.append(-np.sign(z_v))

        X = np.asarray(cases, dtype=np.float32)
        t = np.zeros((X.shape[0], 1), dtype=np.float32)
        Y = np.zeros((X.shape[0], 1), dtype=np.float32)
        Z = np.zeros((X.shape[0], 4), dtype=np.float32)
        Z[:, 2] = np.repeat(np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float32), 3)
        mu = model.mu_tf(
            tf.convert_to_tensor(t),
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(Y),
            tf.convert_to_tensor(Z),
        ).numpy()
        dV = mu[:, 2]
        assert np.isfinite(dV).all(), dV
        np.testing.assert_allclose(dV, expected_dv_np(X, Z), rtol=1.0e-6, atol=1.0e-6)
        for actual, expected_sign in zip(dV, expected_signs):
            assert np.sign(actual) == expected_sign, (dV, expected_signs)

        q_faces = np.asarray(
            [
                [0.0, 0.0, float(params["v_min"]) + 0.1, -0.1],
                [0.0, 0.0, float(params["v_max"]) - 0.1, -0.1],
                [0.0, 0.0, float(params["v_min"]) + 0.1, float(params["x_max"]) + 0.1],
                [0.0, 0.0, float(params["v_max"]) - 0.1, float(params["x_max"]) + 0.1],
            ],
            dtype=np.float32,
        )
        face_Z = np.zeros((q_faces.shape[0], 4), dtype=np.float32)
        face_Z[:, 2] = np.asarray([-3.0, 3.0, -3.0, 3.0], dtype=np.float32)
        face_mu = model.mu_tf(
            tf.zeros((q_faces.shape[0], 1), dtype=tf.float32),
            tf.convert_to_tensor(q_faces),
            tf.zeros((q_faces.shape[0], 1), dtype=tf.float32),
            tf.convert_to_tensor(face_Z),
        ).numpy()
        np.testing.assert_allclose(
            face_mu[:, 2],
            expected_dv_np(q_faces, face_Z),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert np.all(face_mu[:2, 2] > 0.0), face_mu[:, 2]
        assert np.all(face_mu[2:, 2] < 0.0), face_mu[:, 2]
    finally:
        model.close()


def _assert_pascucci_oracle_fixture_arrays(fixture: dict, *, expected_seed: int) -> None:
    metadata = fixture["metadata"]
    inputs = fixture["inputs"]
    moments = fixture["moments"]

    assert metadata["fixture_version"] == 1
    assert metadata["model_name"] == "pascucci"
    assert metadata["seed"] == int(expected_seed)
    assert metadata["dtype"] == "float32"
    assert tuple(metadata["state_labels"]) == ("S", "H", "V", "X_state")
    assert tuple(metadata["z_labels"]) == ("Z_S", "Z_H", "Z_V", "Z_X")
    assert tuple(metadata["moment_names"]) == ("mean_v", "mean_q", "mean_h_plus_v")
    assert metadata["equation_scope"] == ["mu", "sigma", "alpha", "f", "g"]
    assert metadata["recursive_terminal_blob"] is None
    assert metadata["oracle_validation_mode"] == "tf2_numpy_formula_regression"
    assert metadata["historical_tf1_runtime_parity"] is False
    assert metadata["source_variants"]["final_model3"]["pascucci_cost_profile"] == "exp"
    assert metadata["source_variants"]["final_model_modifiche_f"]["pascucci_cost_profile"] == "exp_minus_offset"
    assert np.isclose(metadata["source_variants"]["final_model_modifiche_f"]["pascucci_cost_offset"], 0.12)
    _assert_source_provenance_contract(
        metadata["source_variants"]["final_model3"],
        expected_file="final_model3.py",
    )
    _assert_source_provenance_contract(
        metadata["source_variants"]["final_model_modifiche_f"],
        expected_file="final_model_modifiche_f.py",
    )
    assert set(metadata["historical_reference_provenance"]) == {
        "final_model3.py",
        "final_model_modifiche_f.py",
        "calibration.py",
    }
    for source_file, provenance in metadata["historical_reference_provenance"].items():
        _assert_source_provenance_contract(provenance, expected_file=source_file)
    assert set(metadata["coverage"]) == {"day_night_hours", "q_values", "v_values", "z_v_values"}
    assert metadata["coverage"]["day_night_hours"] == [6.0, 7.0, 18.0, 19.0]

    assert set(inputs) == {"t", "X", "Y", "Z"}
    expected_shapes = {
        "t": (6, 1),
        "X": (6, 4),
        "Y": (6, 1),
        "Z": (6, 4),
    }
    for key, expected_shape in expected_shapes.items():
        value = inputs[key]
        assert isinstance(value, np.ndarray), key
        assert value.shape == expected_shape, key
        assert value.dtype == np.float32, key
        assert np.isfinite(value).all(), key

    assert np.array_equal(inputs["t"][:, 0], np.asarray([6.0, 7.0, 12.0, 18.0, 19.0, 30.0], dtype=np.float32))
    assert np.any(inputs["Z"][:, 2] != 0.0)
    assert np.any(inputs["Z"][:, 0] != inputs["Z"][:, 2])
    assert np.any(inputs["X"][:, 3] < 0.0)
    assert np.any(inputs["X"][:, 3] > float(fixture["params"]["x_max"]))
    assert np.any(inputs["X"][:, 2] < float(fixture["params"]["v_min"]))
    assert np.any(inputs["X"][:, 2] > float(fixture["params"]["v_max"]))

    expected_moments = {
        "mean_v": np.mean(inputs["X"][:, [2]], axis=0, keepdims=True),
        "mean_q": np.mean(inputs["X"][:, [3]], axis=0, keepdims=True),
        "mean_h_plus_v": np.mean(inputs["X"][:, [1]] + inputs["X"][:, [2]], axis=0, keepdims=True),
    }
    assert set(moments) == set(expected_moments)
    for key, expected in expected_moments.items():
        value = moments[key]
        assert value.shape == (1, 1), key
        assert value.dtype == np.float32, key
        np.testing.assert_allclose(value, expected.astype(np.float32), rtol=0.0, atol=1.0e-7)


def test_pascucci_oracle_fixture_generation_contract() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("pascucci_oracle_fixture_generation_contract")
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    fixture = build_pascucci_oracle_fixture(seed=20260608)
    assert set(fixture) == {"metadata", "inputs", "params", "moments"}
    _assert_pascucci_oracle_fixture_arrays(fixture, expected_seed=20260608)
    assert fixture["metadata"]["oracle_source_variant"] == "final_model3"
    assert fixture["params"]["pascucci_cost_profile"] == "exp"
    assert float(fixture["params"]["pascucci_cost_offset"]) == 0.0
    _assert_json_roundtrip(fixture["metadata"])
    _assert_json_roundtrip(fixture["params"])


def test_pascucci_oracle_fixture_reproducible_and_seed_sensitive() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("pascucci_oracle_fixture_reproducible_and_seed_sensitive")
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    first = build_pascucci_oracle_fixture(seed=12345)
    second = build_pascucci_oracle_fixture(seed=12345)
    changed = build_pascucci_oracle_fixture(seed=12346)

    assert first["metadata"] == second["metadata"]
    assert first["params"] == second["params"]
    for section in ("inputs", "moments"):
        for key, value in first[section].items():
            assert np.array_equal(value, second[section][key]), (section, key)

    assert np.array_equal(first["inputs"]["t"], changed["inputs"]["t"])
    assert not np.array_equal(first["inputs"]["X"], changed["inputs"]["X"])
    assert not np.array_equal(first["inputs"]["Z"], changed["inputs"]["Z"])
    assert changed["metadata"]["seed"] == 12346


def test_pascucci_oracle_fixture_save_load_roundtrip() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("pascucci_oracle_fixture_save_load_roundtrip")
    from .pascucci_oracle_fixture import (
        build_pascucci_oracle_fixture,
        load_pascucci_oracle_fixture,
        save_pascucci_oracle_fixture,
    )

    fixture = build_pascucci_oracle_fixture(
        seed=9876,
        oracle_source_variant="final_model_modifiche_f",
        pascucci_cost_profile="exp_minus_offset",
        pascucci_cost_offset=0.12,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pascucci_oracle_fixture.npz"
        save_pascucci_oracle_fixture(fixture, path)
        with np.load(path, allow_pickle=False) as raw:
            assert "metadata_json" in raw.files
            assert "params_json" in raw.files
            assert "t" in raw.files
            assert "X" in raw.files
            assert "Y" in raw.files
            assert "Z" in raw.files
            assert "moment_mean_v" in raw.files
            assert "moment_mean_q" in raw.files
            assert "moment_mean_h_plus_v" in raw.files
        loaded = load_pascucci_oracle_fixture(path)

        malformed_path = Path(tmp) / "malformed_pascucci_oracle_fixture.npz"
        np.savez(
            malformed_path,
            metadata_json=np.asarray(json.dumps(fixture["metadata"], sort_keys=True)),
            params_json=np.asarray(json.dumps(fixture["params"], sort_keys=True)),
            t=fixture["inputs"]["t"],
            X=fixture["inputs"]["X"],
            Y=fixture["inputs"]["Y"],
            Z=fixture["inputs"]["Z"],
        )
        try:
            load_pascucci_oracle_fixture(malformed_path)
        except ValueError as exc:
            assert "missing Pascucci oracle fixture keys" in str(exc)
            assert "moment_mean_v" in str(exc)
        else:
            raise AssertionError("malformed oracle fixture should fail schema validation")

    assert loaded["metadata"] == fixture["metadata"]
    assert loaded["params"] == fixture["params"]
    for section in ("inputs", "moments"):
        for key, value in fixture[section].items():
            assert np.array_equal(loaded[section][key], value), (section, key)


def test_pascucci_oracle_provenance_metadata_roundtrip() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("pascucci_oracle_provenance_metadata_roundtrip")
    from .pascucci_equation_oracle import evaluate_pascucci_equation_oracle
    from .pascucci_oracle_fixture import (
        build_pascucci_oracle_fixture,
        load_pascucci_oracle_fixture,
        save_pascucci_oracle_fixture,
    )

    fixture = build_pascucci_oracle_fixture(
        seed=2468,
        oracle_source_variant="final_model_modifiche_f",
        pascucci_cost_profile="exp_minus_offset",
        pascucci_cost_offset=0.12,
    )
    oracle = evaluate_pascucci_equation_oracle(fixture)

    assert fixture["metadata"]["oracle_validation_mode"] == "tf2_numpy_formula_regression"
    assert fixture["metadata"]["historical_tf1_runtime_parity"] is False
    assert oracle["metadata"]["oracle_validation_mode"] == fixture["metadata"]["oracle_validation_mode"]
    assert oracle["metadata"]["historical_tf1_runtime_parity"] is False
    assert (
        oracle["metadata"]["source_provenance"]
        == fixture["metadata"]["source_variants"]["final_model_modifiche_f"]
    )
    assert (
        oracle["metadata"]["historical_reference_provenance"]
        == fixture["metadata"]["historical_reference_provenance"]
    )
    _assert_json_roundtrip(fixture["metadata"])
    _assert_json_roundtrip(oracle["metadata"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pascucci_oracle_fixture.npz"
        save_pascucci_oracle_fixture(fixture, path)
        loaded = load_pascucci_oracle_fixture(path)
    assert loaded["metadata"] == fixture["metadata"]

    for missing_key in (
        "oracle_validation_mode",
        "historical_tf1_runtime_parity",
        "historical_reference_provenance",
    ):
        legacy_fixture = dict(fixture)
        legacy_fixture["metadata"] = dict(fixture["metadata"])
        del legacy_fixture["metadata"][missing_key]
        try:
            evaluate_pascucci_equation_oracle(legacy_fixture)
        except ValueError as exc:
            assert missing_key in str(exc)
        else:
            raise AssertionError(f"legacy fixture without {missing_key} should fail fast")


def test_pascucci_oracle_fixture_missing_historical_reference_fails_fast() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract(
        "pascucci_oracle_fixture_missing_historical_reference_fails_fast"
    )
    from . import pascucci_oracle_fixture

    original_paths = dict(pascucci_oracle_fixture.HISTORICAL_REFERENCE_PATHS)
    try:
        pascucci_oracle_fixture.HISTORICAL_REFERENCE_PATHS["calibration.py"] = (
            "../../../to_ema/__missing_pascucci_reference_for_test__.py"
        )
        try:
            pascucci_oracle_fixture.build_pascucci_oracle_fixture(seed=2468)
        except FileNotFoundError as exc:
            assert "historical Pascucci reference not found" in str(exc)
            assert "__missing_pascucci_reference_for_test__.py" in str(exc)
        else:
            raise AssertionError("missing historical reference should fail fixture provenance")
    finally:
        pascucci_oracle_fixture.HISTORICAL_REFERENCE_PATHS.clear()
        pascucci_oracle_fixture.HISTORICAL_REFERENCE_PATHS.update(original_paths)


def test_pascucci_oracle_fixture_cost_profile_variants_are_explicit() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("pascucci_oracle_fixture_cost_profile_variants_are_explicit")
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    base = build_pascucci_oracle_fixture(seed=77, oracle_source_variant="final_model3")
    assert base["metadata"]["oracle_source_variant"] == "final_model3"
    assert base["params"]["pascucci_cost_profile"] == "exp"
    assert float(base["params"]["pascucci_cost_offset"]) == 0.0

    offset = build_pascucci_oracle_fixture(
        seed=77,
        oracle_source_variant="final_model_modifiche_f",
        pascucci_cost_profile="exp_minus_offset",
        pascucci_cost_offset=0.12,
    )
    assert offset["metadata"]["oracle_source_variant"] == "final_model_modifiche_f"
    assert offset["params"]["pascucci_cost_profile"] == "exp_minus_offset"
    assert np.isclose(offset["params"]["pascucci_cost_offset"], 0.12)

    for kwargs in (
        {"oracle_source_variant": "unknown_source"},
        {"pascucci_cost_profile": "exp_plus_hidden_magic"},
        {"pascucci_cost_profile": "exp", "pascucci_cost_offset": 0.12},
    ):
        try:
            build_pascucci_oracle_fixture(seed=77, **kwargs)
        except ValueError as exc:
            assert "pascucci" in str(exc) or "oracle_source_variant" in str(exc)
        else:
            raise AssertionError(f"invalid fixture args should fail: {kwargs}")


def test_quadratic_spec_unaffected_by_pascucci_oracle_fixture_import() -> None:
    _assert_pascucci_oracle_fixture_tdd_contract("quadratic_spec_unaffected_by_pascucci_oracle_fixture_import")
    from .model_specs import get_model_spec

    spec_before = get_model_spec("quadratic_coupled")
    params_before = spec_before.build_default_params()
    xi_before = spec_before.deterministic_xi(4, 4, seed=2026)
    np.random.seed(8642)
    rng_before = np.random.rand(4).astype(np.float32)

    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    fixture = build_pascucci_oracle_fixture(seed=2026)
    assert fixture["metadata"]["model_name"] == "pascucci"

    spec_after = get_model_spec("quadratic_coupled")
    params_after = spec_after.build_default_params()
    xi_after = spec_after.deterministic_xi(4, 4, seed=2026)
    np.random.seed(8642)
    rng_after = np.random.rand(4).astype(np.float32)

    assert spec_after.name == spec_before.name
    assert "pascucci_cost_profile" not in params_after
    assert "pascucci_cost_offset" not in params_after
    assert sorted(params_after) == sorted(params_before)
    np.testing.assert_array_equal(xi_after, xi_before)
    np.testing.assert_array_equal(rng_after, rng_before)


def _evaluate_pascucci_tf_equations_from_fixture(fixture: dict) -> dict[str, np.ndarray]:
    from .models import PascucciMeanFieldMoments
    from .tf_backend import tf

    model = _build_pascucci_unit_model(fixture["params"], M=fixture["inputs"]["X"].shape[0])
    try:
        t_tf = tf.convert_to_tensor(fixture["inputs"]["t"], dtype=tf.float32)
        X_tf = tf.convert_to_tensor(fixture["inputs"]["X"], dtype=tf.float32)
        Y_tf = tf.convert_to_tensor(fixture["inputs"]["Y"], dtype=tf.float32)
        Z_tf = tf.convert_to_tensor(fixture["inputs"]["Z"], dtype=tf.float32)
        moment_state = PascucciMeanFieldMoments(
            mean_v=tf.convert_to_tensor(fixture["moments"]["mean_v"], dtype=tf.float32),
            mean_q=tf.convert_to_tensor(fixture["moments"]["mean_q"], dtype=tf.float32),
            mean_h_plus_v=tf.convert_to_tensor(fixture["moments"]["mean_h_plus_v"], dtype=tf.float32),
        )
        return {
            "mu": model.mu_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=moment_state).numpy(),
            "sigma": model.sigma_tf(t_tf, X_tf, Y_tf, moment_state=moment_state).numpy(),
            "alpha": model.alpha_tf(t_tf, X_tf, Z_tf[:, 2:3], moment_state=moment_state).numpy(),
            "f": model.f_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=moment_state).numpy(),
            "g": model.g_tf(X_tf, moment_state=moment_state).numpy(),
        }
    finally:
        model.close()


def _assert_pascucci_equation_oracle_bundle(oracle: dict, fixture: dict) -> None:
    metadata = oracle["metadata"]
    outputs = oracle["outputs"]
    assert metadata["oracle_version"] == 1
    assert metadata["model_name"] == "pascucci"
    assert metadata["fixture_version"] == fixture["metadata"]["fixture_version"]
    assert metadata["fixture_seed"] == fixture["metadata"]["seed"]
    assert metadata["oracle_source_variant"] == fixture["metadata"]["oracle_source_variant"]
    assert metadata["source_file"] == fixture["metadata"]["source_variants"][metadata["oracle_source_variant"]]["source_file"]
    assert metadata["oracle_validation_mode"] == fixture["metadata"]["oracle_validation_mode"]
    assert metadata["historical_tf1_runtime_parity"] is False
    assert (
        metadata["source_provenance"]
        == fixture["metadata"]["source_variants"][metadata["oracle_source_variant"]]
    )
    assert (
        metadata["historical_reference_provenance"]
        == fixture["metadata"]["historical_reference_provenance"]
    )
    assert metadata["pascucci_cost_profile"] == fixture["params"]["pascucci_cost_profile"]
    assert np.isclose(metadata["pascucci_cost_offset"], fixture["params"]["pascucci_cost_offset"])
    assert metadata["equation_scope"] == ["mu", "sigma", "alpha", "f", "g"]
    assert metadata["moment_policy"] == "explicit_fixture_moments"
    assert set(metadata["tolerances"]) == {"rtol", "atol"}
    assert metadata["historical_references"] == ["final_model3.py", "final_model_modifiche_f.py", "calibration.py"]
    _assert_json_roundtrip(metadata)

    expected_shapes = {
        "mu": (6, 4),
        "sigma": (6, 4, 4),
        "alpha": (6, 1),
        "f": (6, 1),
        "g": (6, 1),
    }
    assert set(outputs) == set(expected_shapes)
    for key, expected_shape in expected_shapes.items():
        value = outputs[key]
        assert value.shape == expected_shape, key
        assert value.dtype == np.float32, key
        assert np.isfinite(value).all(), key


def test_pascucci_equation_oracle_final_model3_matches_tf2_fixture() -> None:
    _assert_pascucci_equation_oracle_tdd_contract("pascucci_equation_oracle_final_model3_matches_tf2_fixture")
    from .pascucci_equation_oracle import evaluate_pascucci_equation_oracle
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    fixture = build_pascucci_oracle_fixture(seed=20260608, oracle_source_variant="final_model3")
    oracle = evaluate_pascucci_equation_oracle(fixture)
    _assert_pascucci_equation_oracle_bundle(oracle, fixture)

    tf_outputs = _evaluate_pascucci_tf_equations_from_fixture(fixture)
    for key, expected in oracle["outputs"].items():
        np.testing.assert_allclose(tf_outputs[key], expected, rtol=1.0e-5, atol=1.0e-6, err_msg=key)


def test_pascucci_equation_oracle_exp_minus_offset_variant() -> None:
    _assert_pascucci_equation_oracle_tdd_contract("pascucci_equation_oracle_exp_minus_offset_variant")
    from .pascucci_equation_oracle import evaluate_pascucci_equation_oracle
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    base = build_pascucci_oracle_fixture(seed=13579, oracle_source_variant="final_model3")
    offset = build_pascucci_oracle_fixture(seed=13579, oracle_source_variant="final_model_modifiche_f")
    assert np.array_equal(base["inputs"]["X"], offset["inputs"]["X"])
    assert np.array_equal(base["inputs"]["Z"], offset["inputs"]["Z"])

    base_oracle = evaluate_pascucci_equation_oracle(base)
    offset_oracle = evaluate_pascucci_equation_oracle(offset)
    _assert_pascucci_equation_oracle_bundle(offset_oracle, offset)
    assert offset_oracle["metadata"]["source_file"] == "final_model_modifiche_f.py"

    tf_outputs = _evaluate_pascucci_tf_equations_from_fixture(offset)
    for key, expected in offset_oracle["outputs"].items():
        np.testing.assert_allclose(tf_outputs[key], expected, rtol=1.0e-5, atol=1.0e-6, err_msg=key)

    H_plus_V = offset["inputs"]["X"][:, [1]] + offset["inputs"]["X"][:, [2]]
    expected_delta = -float(offset["params"]["pascucci_cost_offset"]) * H_plus_V
    np.testing.assert_allclose(
        offset_oracle["outputs"]["f"] - base_oracle["outputs"]["f"],
        expected_delta,
        rtol=1.0e-6,
        atol=6.0e-3,
    )
    for key in ("mu", "sigma", "alpha", "g"):
        np.testing.assert_allclose(
            offset_oracle["outputs"][key],
            base_oracle["outputs"][key],
            rtol=1.0e-6,
            atol=1.0e-6,
            err_msg=key,
        )


def test_pascucci_equation_oracle_uses_explicit_fixture_moments() -> None:
    _assert_pascucci_equation_oracle_tdd_contract("pascucci_equation_oracle_uses_explicit_fixture_moments")
    from .pascucci_equation_oracle import evaluate_pascucci_equation_oracle
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    fixture = build_pascucci_oracle_fixture(seed=24680)
    shifted = {
        "metadata": dict(fixture["metadata"]),
        "inputs": {key: value.copy() for key, value in fixture["inputs"].items()},
        "params": dict(fixture["params"]),
        "moments": {key: value.copy() for key, value in fixture["moments"].items()},
    }
    shifted["moments"]["mean_v"] = shifted["moments"]["mean_v"] + np.asarray([[0.75]], dtype=np.float32)
    shifted["moments"]["mean_q"] = shifted["moments"]["mean_q"] + np.asarray([[1.25]], dtype=np.float32)
    shifted["moments"]["mean_h_plus_v"] = shifted["moments"]["mean_h_plus_v"] - np.asarray([[0.50]], dtype=np.float32)

    base_oracle = evaluate_pascucci_equation_oracle(fixture)
    shifted_oracle = evaluate_pascucci_equation_oracle(shifted)
    shifted_tf = _evaluate_pascucci_tf_equations_from_fixture(shifted)
    for key, expected in shifted_oracle["outputs"].items():
        np.testing.assert_allclose(shifted_tf[key], expected, rtol=1.0e-5, atol=1.0e-6, err_msg=key)

    for key in ("mu", "sigma", "alpha", "f", "g"):
        assert not np.allclose(base_oracle["outputs"][key], shifted_oracle["outputs"][key]), key


def test_quadratic_spec_unaffected_by_pascucci_equation_oracle_import() -> None:
    _assert_pascucci_equation_oracle_tdd_contract("quadratic_spec_unaffected_by_pascucci_equation_oracle_import")
    from .model_specs import get_model_spec

    spec_before = get_model_spec("quadratic_coupled")
    params_before = spec_before.build_default_params()
    xi_before = spec_before.deterministic_xi(4, 4, seed=2027)
    np.random.seed(97531)
    rng_before = np.random.rand(4).astype(np.float32)

    from .pascucci_equation_oracle import evaluate_pascucci_equation_oracle
    from .pascucci_oracle_fixture import build_pascucci_oracle_fixture

    fixture = build_pascucci_oracle_fixture(seed=2027)
    oracle = evaluate_pascucci_equation_oracle(fixture)
    assert oracle["metadata"]["model_name"] == "pascucci"

    spec_after = get_model_spec("quadratic_coupled")
    params_after = spec_after.build_default_params()
    xi_after = spec_after.deterministic_xi(4, 4, seed=2027)
    np.random.seed(97531)
    rng_after = np.random.rand(4).astype(np.float32)

    assert sorted(params_after) == sorted(params_before)
    assert "pascucci_cost_profile" not in params_after
    assert "pascucci_cost_offset" not in params_after
    np.testing.assert_array_equal(xi_after, xi_before)
    np.testing.assert_array_equal(rng_after, rng_before)


def test_pascucci_mean_field_moments_contract() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci, PascucciMeanFieldMoments
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.77)
    model = NN_Pascucci(
        Xi_generator_default,
        0.25,
        4,
        2,
        4,
        [5, 8, 1],
        params,
    )

    try:
        X = np.array(
            [
                [1.2, 0.4, 0.2, 4.5],
                [-0.4, -0.2, -0.1, 7.7],
                [5.2, 1.0, 0.5, 2.2],
                [3.1, -0.5, -0.3, 0.8],
            ],
            dtype=np.float32,
        )
        moments = model.mean_field_moments_tf(tf.convert_to_tensor(X))

        assert isinstance(moments, PascucciMeanFieldMoments)
        for name in ("mean_v", "mean_q", "mean_h_plus_v"):
            value_np = getattr(moments, name).numpy()
            assert value_np.shape == (1, 1), f"{name} shape {value_np.shape}"
            assert value_np.dtype == np.float32, f"{name} dtype {value_np.dtype}"
            assert np.isfinite(value_np).all(), f"{name} contains non-finite values"

        np.testing.assert_allclose(moments.mean_v.numpy(), np.mean(X[:, [2]], axis=0, keepdims=True))
        np.testing.assert_allclose(moments.mean_q.numpy(), np.mean(X[:, [3]], axis=0, keepdims=True))
        np.testing.assert_allclose(
            moments.mean_h_plus_v.numpy(),
            np.mean(X[:, [1]] + X[:, [2]], axis=0, keepdims=True),
        )
    finally:
        model.close()


def test_pascucci_equations_accept_explicit_mean_field_moments() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci, PascucciMeanFieldMoments
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    z_v_index = spec.z_labels.index("Z_V")
    params = spec.build_default_params(const=0.77)
    model = NN_Pascucci(
        Xi_generator_default,
        0.25,
        4,
        2,
        4,
        [5, 8, 1],
        params,
    )

    try:
        X = np.array(
            [
                [1.2, 0.4, 0.2, 4.5],
                [-0.4, -0.2, -0.1, 7.7],
                [5.2, 1.0, 0.5, 2.2],
                [3.1, -0.5, -0.3, 0.8],
            ],
            dtype=np.float32,
        )
        Y = np.array([[0.1], [0.2], [-0.1], [0.3]], dtype=np.float32)
        Z = np.array(
            [
                [0.1, 0.2, -0.3, 0.4],
                [0.2, -0.1, 0.4, -0.5],
                [0.3, 0.0, 0.2, -0.1],
                [0.4, -0.2, -0.1, 0.2],
            ],
            dtype=np.float32,
        )
        t = np.zeros((4, 1), dtype=np.float32)

        t_tf = tf.convert_to_tensor(t)
        X_tf = tf.convert_to_tensor(X)
        Y_tf = tf.convert_to_tensor(Y)
        Z_tf = tf.convert_to_tensor(Z)

        fallback_moments = model.mean_field_moments_tf(X_tf)
        custom_moments = PascucciMeanFieldMoments(
            mean_v=tf.constant([[1.25]], dtype=tf.float32),
            mean_q=tf.constant([[3.0]], dtype=tf.float32),
            mean_h_plus_v=tf.constant([[0.75]], dtype=tf.float32),
        )

        np.testing.assert_allclose(
            model.sigmaV_tf(t_tf, X_tf, moment_state=fallback_moments).numpy(),
            model.sigmaV_tf(t_tf, X_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.sigma_tf(t_tf, X_tf, Y_tf, moment_state=fallback_moments).numpy(),
            model.sigma_tf(t_tf, X_tf, Y_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.f_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=fallback_moments).numpy(),
            model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.g_tf(X_tf, moment_state=fallback_moments).numpy(),
            model.g_tf(X_tf).numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        S, H, V, Q = X[:, [0]], X[:, [1]], X[:, [2]], X[:, [3]]
        mean_v = custom_moments.mean_v.numpy()
        mean_q = custom_moments.mean_q.numpy()
        mean_h_plus_v = custom_moments.mean_h_plus_v.numpy()

        sigma_custom_np = (
            float(params["s3"]) * np.ones_like(V, dtype=np.float32)
            + float(params["s3h"]) * np.abs(H)
            + float(params["s3v"]) * np.abs(V)
            + float(params["s3k"]) * np.abs(V - mean_v)
        )
        alpha_custom_np = (
            -(_pascucci_psi(Q, float(params["d"]), float(params["x_max"])) * Z[:, [z_v_index]])
            / (2.0 * float(params["l_a"]) * np.maximum(sigma_custom_np, 1.0e-7))
        ).astype(np.float32)
        f_custom_np = (
            np.exp(S) * (H + V)
            + float(params["l_v"]) * V ** 2
            + float(params["l_a"]) * alpha_custom_np ** 2
            + float(params["c_h"]) * _pascucci_h_with_mean(Q, mean_q)
            + float(params["c_con"]) * _pascucci_h_with_mean(H + V, mean_h_plus_v)
        ).astype(np.float32)
        g_custom_np = (
            -float(params["gamma"]) * Q * np.exp(S)
            + 0.5 * float(params["omega"]) * (Q - mean_q) ** 2
        ).astype(np.float32)
        mu_S_mean = _pascucci_ou_mu_daynight(t, params["params_S"])
        mu_H_mean = _pascucci_ou_mu_daynight(t, params["params_H"])
        kappa_S = float(params["params_S"]["kappa_night"])
        kappa_H = float(params["params_H"]["kappa_night"])
        dS_np = kappa_S * (mu_S_mean - S)
        dH_np = kappa_H * (mu_H_mean - H)
        dV_np = (
            alpha_custom_np * _pascucci_psi(Q, float(params["d"]), float(params["x_max"]))
            + float(params["c3"]) * _pascucci_psi2(Q, float(params["d"]), float(params["x_max"])) * _pascucci_psi3(V, float(params["d"]), float(params["v_max"]))
            - float(params["c4"]) * _pascucci_psi1(Q, float(params["d"]), float(params["x_max"])) * _pascucci_psi4(V, float(params["d"]), float(params["v_min"]))
        )
        mu_custom_np = np.concatenate([dS_np, dH_np, dV_np, V], axis=1).astype(np.float32)

        np.testing.assert_allclose(
            model.alpha_tf(
                t_tf,
                X_tf,
                Z_tf[:, z_v_index : z_v_index + 1],
                moment_state=custom_moments,
            ).numpy(),
            alpha_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.sigmaV_tf(t_tf, X_tf, moment_state=custom_moments).numpy(),
            sigma_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        sigma_custom_tf = model.sigma_tf(t_tf, X_tf, Y_tf, moment_state=custom_moments).numpy()
        np.testing.assert_allclose(
            sigma_custom_tf[:, 2, 2:3],
            sigma_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert np.max(np.abs(sigma_custom_tf - model.sigma_tf(t_tf, X_tf, Y_tf).numpy())) > 1.0e-6
        np.testing.assert_allclose(
            model.f_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=custom_moments).numpy(),
            f_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.mu_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=custom_moments).numpy(),
            mu_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.g_tf(X_tf, moment_state=custom_moments).numpy(),
            g_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            model.phi_tf(t_tf, X_tf, Y_tf, Z_tf, moment_state=custom_moments).numpy(),
            -f_custom_np,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        assert np.max(np.abs(sigma_custom_np - model.sigmaV_tf(t_tf, X_tf).numpy())) > 1.0e-6
        assert np.max(np.abs(f_custom_np - model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy())) > 1.0e-6
        assert np.max(np.abs(g_custom_np - model.g_tf(X_tf).numpy())) > 1.0e-6
    finally:
        model.close()


def test_quadratic_loss_context_hook_is_noop() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import reset_backend_state, set_seed, tf

    class RecordingQuadratic(NN_Quadratic_Coupled_Recursive):
        def __init__(self, *args, **kwargs):
            self.sigma_contexts = []
            self.mu_contexts = []
            self.phi_contexts = []
            self.g_contexts = []
            self.dg_contexts = []
            super().__init__(*args, **kwargs)

        def sigma_tf(self, t, X, Y, moment_state=None):
            self.sigma_contexts.append(moment_state)
            return super().sigma_tf(t, X, Y, moment_state=moment_state)

        def mu_tf(self, t, X, Y, Z, moment_state=None):
            self.mu_contexts.append(moment_state)
            return super().mu_tf(t, X, Y, Z, moment_state=moment_state)

        def phi_tf(self, t, X, Y, Z, moment_state=None):
            self.phi_contexts.append(moment_state)
            return super().phi_tf(t, X, Y, Z, moment_state=moment_state)

        def g_tf(self, X, moment_state=None):
            self.g_contexts.append(moment_state)
            return super().g_tf(X, moment_state=moment_state)

        def Dg_tf(self, X, moment_state=None):
            self.dg_contexts.append(moment_state)
            return super().Dg_tf(X, moment_state=moment_state)

    reset_backend_state()
    set_seed(61)

    spec = get_model_spec("quadratic_coupled")
    params = spec.build_default_params(const=1.0)
    model = RecordingQuadratic(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        t0 = tf.convert_to_tensor(t_batch[:, 0, :])
        X0 = tf.convert_to_tensor(Xi_batch)
        Y0, _ = model.net_u(t0, X0)

        assert hasattr(model, "build_loss_context_tf")
        assert model.build_loss_context_tf(t0, X0, Y0) is None
        loss, X, Y, Z = model.loss_function(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=1.0,
        )
        del loss, X, Y, Z
        for name, contexts in (
            ("sigma_tf", model.sigma_contexts),
            ("mu_tf", model.mu_contexts),
            ("phi_tf", model.phi_contexts),
            ("g_tf", model.g_contexts),
            ("Dg_tf", model.dg_contexts),
        ):
            assert contexts, f"{name} was not exercised"
            assert all(context is None for context in contexts), (
                f"{name} received a non-null runtime context"
            )
    finally:
        model.close()


def test_pascucci_loss_function_forwards_model_owned_context() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed, tf

    class RecordingPascucci(NN_Pascucci_Recursive):
        def __init__(self, *args, **kwargs):
            self.context_states = []
            self.built_contexts = []
            self.sigma_contexts = []
            self.mu_contexts = []
            self.phi_contexts = []
            self.g_contexts = []
            self.dg_contexts = []
            super().__init__(*args, **kwargs)

        def build_loss_context_tf(self, t, X, Y):
            del t, Y
            self.context_states.append(tf.identity(tf.cast(X, tf.float32)))
            context = self.mean_field_moments_tf(X)
            self.built_contexts.append(context)
            return context

        def sigma_tf(self, t, X, Y, moment_state=None):
            self.sigma_contexts.append(moment_state)
            return super().sigma_tf(t, X, Y, moment_state=moment_state)

        def mu_tf(self, t, X, Y, Z, moment_state=None):
            self.mu_contexts.append(moment_state)
            return super().mu_tf(t, X, Y, Z, moment_state=moment_state)

        def phi_tf(self, t, X, Y, Z, moment_state=None):
            self.phi_contexts.append(moment_state)
            return super().phi_tf(t, X, Y, Z, moment_state=moment_state)

        def g_tf(self, X, moment_state=None):
            self.g_contexts.append(moment_state)
            return super().g_tf(X, moment_state=moment_state)

        def Dg_tf(self, X, moment_state=None):
            self.dg_contexts.append(moment_state)
            return tf.zeros_like(tf.cast(X, tf.float32))

    reset_backend_state()
    set_seed(67)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = RecordingPascucci(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=3,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        _, X, _, _ = model.loss_function(t_batch, W_batch, Xi_batch, const_value=0.75)
        X_np = X.numpy()

        assert len(model.context_states) >= X_np.shape[1]
        for step, context_state in enumerate(model.context_states[: X_np.shape[1]]):
            np.testing.assert_allclose(
                context_state.numpy(),
                X_np[:, step, :],
                rtol=1.0e-6,
                atol=1.0e-6,
            )

        terminal_context = model.built_contexts[X_np.shape[1] - 1]
        assert model.g_contexts[-1] is terminal_context
        assert model.dg_contexts[-1] is terminal_context

        for name, contexts in (
            ("sigma_tf", model.sigma_contexts),
            ("mu_tf", model.mu_contexts),
            ("phi_tf", model.phi_contexts),
            ("g_tf", model.g_contexts),
            ("Dg_tf", model.dg_contexts),
        ):
            assert contexts, f"{name} did not receive any runtime context"
            assert all(context is not None for context in contexts), (
                f"{name} received an implicit moment context"
            )
    finally:
        model.close()


def test_pascucci_runtime_moment_diagnostics_follow_loss_context() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed, tf

    class RecordingPascucci(NN_Pascucci_Recursive):
        def __init__(self, *args, **kwargs):
            self.built_contexts = []
            self.diagnostic_contexts = []
            super().__init__(*args, **kwargs)

        def build_loss_context_tf(self, t, X, Y):
            del t, Y
            context = self.mean_field_moments_tf(X)
            self.built_contexts.append(context)
            return context

        def loss_context_diagnostics_tf(self, loss_context):
            self.diagnostic_contexts.append(loss_context)
            return super().loss_context_diagnostics_tf(loss_context)

    reset_backend_state()
    set_seed(71)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = RecordingPascucci(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=3,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        result = model.loss_function(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            return_runtime_diagnostics=True,
        )
        assert len(result) == 5
        loss, X, Y, Z, diagnostics = result
        del loss, Y, Z

        X_np = X.numpy()
        expected = {
            "mean_v": np.mean(X_np[:, :, [2]], axis=0),
            "mean_q": np.mean(X_np[:, :, [3]], axis=0),
            "mean_h_plus_v": np.mean(X_np[:, :, [1]] + X_np[:, :, [2]], axis=0),
        }
        expected.update(_pascucci_physical_violation_traces_from_x(X_np, params))
        assert set(diagnostics) == set(expected)
        assert len(model.diagnostic_contexts) >= X_np.shape[1]
        for step in range(X_np.shape[1]):
            assert model.diagnostic_contexts[step] is model.built_contexts[step]
        for key, expected_trace in expected.items():
            value = diagnostics[key]
            value_np = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            assert value_np.shape == (X_np.shape[1], 1), (key, value_np.shape)
            assert value_np.dtype == np.float32, (key, value_np.dtype)
            assert np.isfinite(value_np).all(), key
            np.testing.assert_allclose(value_np, expected_trace, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_pascucci_runtime_physical_q_v_diagnostics_follow_loss_context() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(75)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = NN_Pascucci_Recursive(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=75)
        Xi_batch[:, 2] = np.asarray([-2.50, -2.05, 2.25, 0.00], dtype=np.float32)
        Xi_batch[:, 3] = np.asarray([-0.25, 0.00, 10.50, 5.00], dtype=np.float32)
        _, X, _, _, diagnostics = model.loss_function(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=0.75,
            return_runtime_diagnostics=True,
        )
        X_np = X.numpy()
        expected = _pascucci_physical_violation_traces_from_x(X_np, params)
        for key, expected_trace in expected.items():
            assert key in diagnostics, f"{key} missing from diagnostics: {sorted(diagnostics)}"
            value = diagnostics[key].numpy()
            assert value.shape == (X_np.shape[1], 1), (key, value.shape)
            assert value.dtype == np.float32, (key, value.dtype)
            assert np.isfinite(value).all(), key
            assert np.all(value >= 0.0), key
            np.testing.assert_allclose(value, expected_trace, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_quadratic_runtime_moment_diagnostics_are_noop() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(73)

    spec = get_model_spec("quadratic_coupled")
    params = spec.build_default_params(const=1.0)
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        loss, X, Y, Z = model.loss_function(t_batch, W_batch, Xi_batch, const_value=1.0)
        diagnostic_result = model.loss_function(
            t_batch,
            W_batch,
            Xi_batch,
            const_value=1.0,
            return_runtime_diagnostics=True,
        )
        assert len(diagnostic_result) == 5
        loss_diag, X_diag, Y_diag, Z_diag, diagnostics = diagnostic_result
        np.testing.assert_allclose(loss_diag.numpy(), loss.numpy(), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(X_diag.numpy(), X.numpy(), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(Y_diag.numpy(), Y.numpy(), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(Z_diag.numpy(), Z.numpy(), rtol=1.0e-6, atol=1.0e-6)
        assert diagnostics == {}

        pred = model.predict(Xi_batch, t_batch, W_batch, const_value=1.0)
        pred_diag = model.predict(
            Xi_batch,
            t_batch,
            W_batch,
            const_value=1.0,
            return_runtime_diagnostics=True,
        )
        assert len(pred_diag) == 4
        for actual, expected in zip(pred_diag[:3], pred):
            np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
        assert pred_diag[3] == {}

        stats = model.evaluate(const_value=1.0, n_batches=1)
        assert not any("block_end_mean" in key for key in stats)
    finally:
        model.close()


def test_pascucci_evaluate_reports_block_end_moment_scalars() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed

    class DeterministicDiagnosticPascucci(NN_Pascucci_Recursive):
        def __init__(self, *args, **kwargs):
            self.diagnostic_call = 0
            super().__init__(*args, **kwargs)

        def loss_context_diagnostics_tf(self, loss_context):
            del loss_context
            self.diagnostic_call += 1
            offset = np.float32(self.diagnostic_call)
            return {
                "mean_v": np.asarray([[offset]], dtype=np.float32),
                "mean_q": np.asarray([[offset + 1.0]], dtype=np.float32),
                "mean_h_plus_v": np.asarray([[offset + 2.0]], dtype=np.float32),
            }

    reset_backend_state()
    set_seed(79)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = DeterministicDiagnosticPascucci(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=1,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        stats = model.evaluate(const_value=0.75, n_batches=2)
        expected_values = {
            "mean_block_end_mean_v": np.mean([2.0, 4.0]),
            "std_block_end_mean_v": np.std([2.0, 4.0]),
            "mean_block_end_mean_q": np.mean([3.0, 5.0]),
            "std_block_end_mean_q": np.std([3.0, 5.0]),
            "mean_block_end_mean_h_plus_v": np.mean([4.0, 6.0]),
            "std_block_end_mean_h_plus_v": np.std([4.0, 6.0]),
        }
        for key, expected in expected_values.items():
            assert key in stats, f"{key} missing from evaluate stats: {sorted(stats)}"
            assert isinstance(stats[key], float)
            assert np.isfinite(stats[key])
            np.testing.assert_allclose(stats[key], expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()


def test_pascucci_fixed_eval_bundle_recomputes_moments_deterministically() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(81)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = NN_Pascucci_Recursive(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        fixed_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=81)
        _, X, _, _, diagnostics = model.loss_function(
            fixed_batch[0],
            fixed_batch[1],
            fixed_batch[2],
            const_value=0.75,
            return_runtime_diagnostics=True,
        )
        X_np = X.numpy()
        expected_block_end = {
            "mean_block_end_mean_v": float(np.mean(X_np[:, -1, [2]])),
            "mean_block_end_mean_q": float(np.mean(X_np[:, -1, [3]])),
            "mean_block_end_mean_h_plus_v": float(
                np.mean(X_np[:, -1, [1]] + X_np[:, -1, [2]])
            ),
        }
        expected_block_end.update(
            {
                f"mean_block_end_{key}": float(value[-1, 0])
                for key, value in _pascucci_physical_violation_traces_from_x(X_np, params).items()
            }
        )
        for key, trace in diagnostics.items():
            trace_np = trace.numpy() if hasattr(trace, "numpy") else np.asarray(trace)
            np.testing.assert_allclose(
                trace_np[-1],
                expected_block_end[f"mean_block_end_{key}"],
                rtol=1.0e-6,
                atol=1.0e-6,
            )

        first = model.evaluate(
            const_value=0.75,
            n_batches=1,
            evaluation_batches=[fixed_batch],
            moment_policy="fixed_eval_recompute",
        )
        np.random.seed(999)
        _ = np.random.normal(size=128)
        set_seed(999)
        second = model.evaluate(
            const_value=0.75,
            n_batches=1,
            evaluation_batches=[fixed_batch],
            moment_policy="fixed_eval_recompute",
        )

        assert first["moment_policy"] == "fixed_eval_recompute"
        assert first["evaluation_batches"] == 1
        for key in (
            "mean_loss",
            "std_loss",
            "mean_loss_per_sample",
            "std_loss_per_sample",
            "mean_y0",
            "std_y0",
            *expected_block_end.keys(),
        ):
            assert key in first, f"{key} missing from fixed eval stats: {sorted(first)}"
            assert key in second
            np.testing.assert_allclose(first[key], second[key], rtol=1.0e-7, atol=1.0e-7)
        for key, expected in expected_block_end.items():
            np.testing.assert_allclose(first[key], expected, rtol=1.0e-6, atol=1.0e-6)
            std_key = key.replace("mean_", "std_", 1)
            assert std_key in first, f"{std_key} missing from fixed eval stats: {sorted(first)}"
            np.testing.assert_allclose(first[std_key], 0.0, rtol=0.0, atol=1.0e-7)
    finally:
        model.close()


def test_pascucci_fixed_eval_does_not_reuse_prior_live_batch_context() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed

    class RecordingPascucci(NN_Pascucci_Recursive):
        def __init__(self, *args, **kwargs):
            self.context_snapshots = []
            super().__init__(*args, **kwargs)

        def build_loss_context_tf(self, t, X, Y):
            context = super().build_loss_context_tf(t, X, Y)
            self.context_snapshots.append(
                {
                    "t": float(np.mean(t.numpy())),
                    "mean_v": float(context.mean_v.numpy()[0, 0]),
                    "mean_q": float(context.mean_q.numpy()[0, 0]),
                    "mean_h_plus_v": float(context.mean_h_plus_v.numpy()[0, 0]),
                }
            )
            return context

    reset_backend_state()
    set_seed(82)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    model = RecordingPascucci(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        train_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=82, v_shift=0.0)
        eval_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=83, v_shift=5.0)

        model.loss_function(
            train_batch[0],
            train_batch[1],
            train_batch[2],
            const_value=0.75,
        )
        first_train_context = dict(model.context_snapshots[0])
        model.context_snapshots.clear()

        stats = model.evaluate(
            const_value=0.75,
            n_batches=1,
            evaluation_batches=[eval_batch],
            moment_policy="fixed_eval_recompute",
        )
        del stats
        assert model.context_snapshots, "fixed evaluation did not build any moment context"
        first_eval_context = model.context_snapshots[0]

        expected_train_mean_v = float(np.mean(train_batch[2][:, 2]))
        expected_eval_mean_v = float(np.mean(eval_batch[2][:, 2]))
        np.testing.assert_allclose(first_train_context["mean_v"], expected_train_mean_v)
        np.testing.assert_allclose(first_eval_context["mean_v"], expected_eval_mean_v)
        assert abs(first_eval_context["mean_v"] - first_train_context["mean_v"]) > 1.0
    finally:
        model.close()


def test_quadratic_fixed_eval_bundle_keeps_moment_policy_noop() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(84)

    spec = get_model_spec("quadratic_coupled")
    params = spec.build_default_params(const=1.0)
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        fixed_batch = _fixed_validation_batch(M=4, N=2, D=4, T=0.25, seed=84)
        stats = model.evaluate(
            const_value=1.0,
            n_batches=1,
            evaluation_batches=[fixed_batch],
            moment_policy="fixed_eval_recompute",
        )
        assert stats["moment_policy"] == "fixed_eval_recompute"
        assert stats["evaluation_batches"] == 1
        assert not any("block_end_mean" in key for key in stats)
    finally:
        model.close()


def test_pascucci_f_mu_select_z_v_from_full_z() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    assert spec.z_labels == ("Z_S", "Z_H", "Z_V", "Z_X")
    z_v_index = spec.z_labels.index("Z_V")

    params = spec.build_default_params(const=0.77)
    model = NN_Pascucci(
        Xi_generator_default,
        0.25,
        4,
        2,
        4,
        [5, 8, 1],
        params,
    )

    try:
        X = np.array(
            [
                [1.2, 0.4, 0.2, 4.5],
                [-0.4, -0.2, -0.1, 7.7],
                [5.2, 1.0, 0.5, 2.2],
                [3.1, -0.5, -0.3, 0.8],
            ],
            dtype=np.float32,
        )
        Y = np.array([[0.1], [0.2], [-0.1], [0.3]], dtype=np.float32)
        Z = np.array(
            [
                [0.1, 0.2, -0.3, 0.4],
                [0.2, -0.1, 0.4, -0.5],
                [0.3, 0.0, 0.2, -0.1],
                [0.4, -0.2, -0.1, 0.2],
            ],
            dtype=np.float32,
        )
        t = np.zeros((4, 1), dtype=np.float32)

        psi = _pascucci_psi(X[:, [3]], float(params["d"]), float(params["x_max"]))
        assert np.all(psi > 0.0)

        t_tf = tf.convert_to_tensor(t)
        X_tf = tf.convert_to_tensor(X)
        Y_tf = tf.convert_to_tensor(Y)

        def eval_alpha_f_mu(z_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            Z_tf = tf.convert_to_tensor(z_np)
            return (
                model.alpha_tf(t_tf, X_tf, tf.convert_to_tensor(z_np[:, [z_v_index]])).numpy(),
                model.f_tf(t_tf, X_tf, Y_tf, Z_tf).numpy(),
                model.mu_tf(t_tf, X_tf, Y_tf, Z_tf).numpy(),
            )

        alpha_base, f_base, mu_base = eval_alpha_f_mu(Z)

        Z_non_v_changed = Z.copy()
        Z_non_v_changed[:, 0] += 100.0
        Z_non_v_changed[:, 1] -= 200.0
        Z_non_v_changed[:, 3] += np.array([50.0, -60.0, 70.0, -80.0], dtype=np.float32)
        alpha_non_v, f_non_v, mu_non_v = eval_alpha_f_mu(Z_non_v_changed)

        np.testing.assert_allclose(alpha_non_v, alpha_base, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(f_non_v, f_base, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(mu_non_v, mu_base, rtol=1.0e-6, atol=1.0e-6)

        Z_v_changed = Z.copy()
        Z_v_changed[:, z_v_index] += np.array([1.25, -1.5, 2.0, -2.25], dtype=np.float32)
        alpha_v, f_v, mu_v = eval_alpha_f_mu(Z_v_changed)

        assert np.max(np.abs(alpha_v - alpha_base)) > 1.0e-4
        assert np.max(np.abs(f_v - f_base)) > 1.0e-4
        assert np.max(np.abs(mu_v[:, [2]] - mu_base[:, [2]])) > 1.0e-4
        np.testing.assert_allclose(mu_v[:, [0, 1, 3]], mu_base[:, [0, 1, 3]], rtol=1.0e-6, atol=1.0e-6)
    finally:
        model.close()

def test_pascucci_equation_shapes_dtypes_finite() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci
    from .tf_backend import tf

    spec = get_model_spec("pascucci")
    z_v_index = spec.z_labels.index("Z_V")
    params = spec.build_default_params(const=0.77)
    model = NN_Pascucci(
        Xi_generator_default,
        0.25,
        4,
        2,
        4,
        [5, 8, 1],
        params,
    )

    try:
        X = np.array(
            [
                [0.2, 0.1, 0.05, 0.5],
                [-0.3, -0.2, -0.1, 1.5],
                [0.4, 0.3, 0.2, 8.5],
                [0.0, -0.4, 0.3, 10.5],
            ],
            dtype=np.float32,
        )
        Y = np.array([[0.1], [-0.2], [0.3], [0.0]], dtype=np.float32)
        Z = np.array(
            [
                [0.1, 0.2, -0.3, 0.4],
                [-0.2, 0.1, 0.4, -0.5],
                [0.3, -0.1, 0.2, 0.0],
                [0.0, 0.2, -0.1, 0.1],
            ],
            dtype=np.float32,
        )
        t = np.array([[0.0], [8.0], [20.0], [23.5]], dtype=np.float32)

        t_tf = tf.convert_to_tensor(t)
        X_tf = tf.convert_to_tensor(X)
        Y_tf = tf.convert_to_tensor(Y)
        Z_tf = tf.convert_to_tensor(Z)

        def assert_contract(name: str, value, expected_shape: tuple[int, ...]) -> None:
            value_np = value.numpy()
            assert value_np.shape == expected_shape, f"{name} shape {value_np.shape}"
            assert value_np.dtype == np.float32, f"{name} dtype {value_np.dtype}"
            assert np.isfinite(value_np).all(), f"{name} contains non-finite values"

        assert_contract("alpha_tf", model.alpha_tf(t_tf, X_tf, Z_tf[:, z_v_index : z_v_index + 1]), (4, 1))
        assert_contract("sigmaV_tf", model.sigmaV_tf(t_tf, X_tf), (4, 1))
        assert_contract("mu_tf", model.mu_tf(t_tf, X_tf, Y_tf, Z_tf), (4, 4))
        assert_contract("sigma_tf", model.sigma_tf(t_tf, X_tf, Y_tf), (4, 4, 4))
        assert_contract("f_tf", model.f_tf(t_tf, X_tf, Y_tf, Z_tf), (4, 1))
        assert_contract("g_tf", model.g_tf(X_tf), (4, 1))
        assert_contract("phi_tf", model.phi_tf(t_tf, X_tf, Y_tf, Z_tf), (4, 1))
    finally:
        model.close()


def test_pascucci_recursive_loss_shapes_dtypes_finite() -> None:
    from .model_specs import get_model_spec
    from .tf_backend import reset_backend_state, set_seed

    reset_backend_state()
    set_seed(101)

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    params["same_xi_antithetic_sampling"] = True

    model = spec.build_recursive_model(
        Xi_generator=spec.xi_generator,
        T=0.25,
        M=4,
        N=3,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        terminal_blob=None,
        normalize_time_input=True,
    )

    try:
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        assert t_batch.shape == (4, 4, 1)
        assert W_batch.shape == (4, 4, 4)
        assert Xi_batch.shape == (4, 4)
        assert t_batch.dtype == np.float32
        assert W_batch.dtype == np.float32
        assert Xi_batch.dtype == np.float32
        assert np.isfinite(t_batch).all()
        assert np.isfinite(W_batch).all()
        assert np.isfinite(Xi_batch).all()

        loss, X, Y, Z = model.loss_function(t_batch, W_batch, Xi_batch, const_value=0.75)
        loss_np = np.asarray(loss.numpy())
        X_np = X.numpy()
        Y_np = Y.numpy()
        Z_np = Z.numpy()

        assert loss_np.dtype == np.float32
        assert X_np.shape == (4, 4, 4)
        assert Y_np.shape == (4, 4, 1)
        assert Z_np.shape == (4, 4, 4)
        assert X_np.dtype == np.float32
        assert Y_np.dtype == np.float32
        assert Z_np.dtype == np.float32
        assert np.isfinite(float(loss_np))
        assert np.isfinite(X_np).all()
        assert np.isfinite(Y_np).all()
        assert np.isfinite(Z_np).all()
    finally:
        model.close()


def test_model_spec_params_overlay_preserves_solver_flags() -> None:
    from .model_specs import get_model_spec

    spec = get_model_spec()
    params = spec.build_default_params(const=0.5)
    runtime_keys = {
        "same_xi_antithetic_sampling",
        "dynamic_loss_dt_normalization",
        "dynamic_loss_weight",
        "terminal_y_loss_weight",
        "terminal_z_loss_weight",
        "terminal_z_component_weights",
        "structural_z_loss_weight",
        "structural_z_component_weights",
    }
    assert runtime_keys.isdisjoint(params)

    overlay = {
        "same_xi_antithetic_sampling": True,
        "dynamic_loss_dt_normalization": True,
        "dynamic_loss_weight": np.float32(0.7),
        "terminal_y_loss_weight": np.float32(1.2),
        "terminal_z_loss_weight": np.float32(2.0),
        "terminal_z_component_weights": [1.0, 0.5, 0.0, 3.0],
        "structural_z_loss_weight": np.float32(0.25),
        "structural_z_component_weights": [0.0, 1.0, 0.0, 0.0],
    }
    params.update(overlay)

    assert np.isclose(params["const"], np.float32(0.5))
    for key, value in overlay.items():
        assert params[key] == value


def test_exact_path_plot_outputs() -> None:
    from .plotting import _PLOTTING_AVAILABLE, plot_recursive_exact_comparison

    if not _PLOTTING_AVAILABLE:
        return

    M = 3
    T_steps = 5
    D = 4
    t = np.tile(np.linspace(0.0, 1.0, T_steps, dtype=np.float32).reshape(1, T_steps, 1), (M, 1, 1))
    X = np.zeros((M, T_steps, D), dtype=np.float32)
    Y_pred = np.zeros((M, T_steps, 1), dtype=np.float32)
    Z_pred = np.zeros((M, T_steps, D), dtype=np.float32)
    Y_exact = np.zeros_like(Y_pred)
    Z_exact = np.zeros_like(Z_pred)
    for i in range(M):
        Y_pred[i, :, 0] = i + t[i, :, 0]
        Y_exact[i, :, 0] = i + 0.9 * t[i, :, 0]
        for d in range(D):
            Z_pred[i, :, d] = (d + 1) * (i + 1) * t[i, :, 0]
            Z_exact[i, :, d] = 0.8 * Z_pred[i, :, d]
    stitched = {"t": t, "X": X, "Y": Y_pred, "Z": Z_pred}
    blocks = [{"t_start": 0.0, "t_end": 0.5}, {"t_start": 0.5, "t_end": 1.0}]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        plot_recursive_exact_comparison(
            stitched=stitched,
            Y_exact=Y_exact,
            Z_exact=Z_exact,
            blocks=blocks,
            out_dir=str(out_dir),
            sample_paths=2,
            file_suffix="_unit",
            include_path_plots=True,
            include_error_plots=True,
        )
        expected = [
            "recursive_stitched_Y_exact_unit.png",
            "recursive_stitched_Z_S_exact_unit.png",
            "recursive_stitched_Z_H_exact_unit.png",
            "recursive_stitched_Z_V_exact_unit.png",
            "recursive_stitched_Z_X_exact_unit.png",
            "recursive_stitched_Z_rel_error_unit.png",
            "recursive_stitched_abs_error_unit.png",
        ]
        for name in expected:
            path = out_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name


def _pascucci_paper_plot_smoke_inputs(M: int = 5):
    M = int(M)
    T_steps = 6
    t_grid = np.linspace(0.0, 5.0, T_steps, dtype=np.float32)
    t = np.tile(t_grid.reshape(1, T_steps, 1), (M, 1, 1)).astype(np.float32)
    path_offsets = np.linspace(-0.08, 0.08, M, dtype=np.float32).reshape(M, 1)
    X = np.zeros((M, T_steps, 4), dtype=np.float32)
    X[:, :, 0] = 1.0 + 0.04 * t_grid.reshape(1, -1) + path_offsets
    X[:, :, 1] = -0.4 + 0.05 * np.sin(t_grid.reshape(1, -1)) + path_offsets
    X[:, :, 2] = 0.2 + 0.03 * t_grid.reshape(1, -1) + 0.5 * path_offsets
    X[:, :, 3] = 4.0 + 0.15 * t_grid.reshape(1, -1) + 2.0 * path_offsets
    Y = np.zeros((M, T_steps, 1), dtype=np.float32)
    Z = np.zeros((M, T_steps, 4), dtype=np.float32)
    Z[:, :, 2] = 0.1 + 0.02 * t_grid.reshape(1, -1)
    steps = T_steps - 1
    alpha_time = np.linspace(-0.4, 0.4, steps, dtype=np.float32).reshape(1, steps, 1)
    controlled_alpha = alpha_time + path_offsets.reshape(M, 1, 1)
    uncontrolled_alpha = np.zeros_like(controlled_alpha)
    controlled_increment = (0.2 + np.abs(controlled_alpha)).astype(np.float32)
    uncontrolled_increment = (0.3 + 0.1 * np.ones_like(uncontrolled_alpha)).astype(np.float32)
    controlled_cumulative = np.cumsum(controlled_increment, axis=1).astype(np.float32)
    uncontrolled_cumulative = np.cumsum(uncontrolled_increment, axis=1).astype(np.float32)
    controlled_terminal = np.full((M, 1), 0.25, dtype=np.float32)
    uncontrolled_terminal = np.full((M, 1), 0.35, dtype=np.float32)
    controlled_trajectory = (
        np.concatenate([np.zeros((M, 1, 1), dtype=np.float32), controlled_cumulative], axis=1)
        + controlled_terminal[:, None, :]
    ).astype(np.float32)
    uncontrolled_trajectory = (
        np.concatenate([np.zeros((M, 1, 1), dtype=np.float32), uncontrolled_cumulative], axis=1)
        + uncontrolled_terminal[:, None, :]
    ).astype(np.float32)
    application_pathwise = {
        "controlled_cost_J_running": controlled_cumulative[:, -1, :],
        "controlled_cost_J_terminal": controlled_terminal,
        "controlled_cost_J_total": controlled_cumulative[:, -1, :] + controlled_terminal,
        "controlled_cost_J_running_cumulative": controlled_cumulative,
        "controlled_cost_J_trajectory": controlled_trajectory,
        "controlled_cost_J_trajectory_math": controlled_trajectory,
        "controlled_alpha": controlled_alpha,
        "uncontrolled_cost_J_running": uncontrolled_cumulative[:, -1, :],
        "uncontrolled_cost_J_terminal": uncontrolled_terminal,
        "uncontrolled_cost_J_total": uncontrolled_cumulative[:, -1, :] + uncontrolled_terminal,
        "uncontrolled_cost_J_running_cumulative": uncontrolled_cumulative,
        "uncontrolled_cost_J_trajectory": uncontrolled_trajectory,
        "uncontrolled_cost_J_trajectory_math": uncontrolled_trajectory,
        "uncontrolled_alpha": uncontrolled_alpha,
    }
    params = _default_pascucci_params(const=0.75)
    params["params_S"]["a0_day"] = np.float32(1.1)
    params["params_S"]["a0_night"] = np.float32(0.9)
    params["params_H"]["a0_day"] = np.float32(-0.3)
    params["params_H"]["a0_night"] = np.float32(-0.5)
    return {
        "stitched": {"t": t, "X": X, "Y": Y, "Z": Z},
        "application_pathwise": application_pathwise,
        "params": params,
        "blocks": [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "T_block": 5.0}],
    }


def test_pascucci_paper_plot_bundle_from_artifacts_smoke() -> None:
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE, plt

    if not _PLOTTING_AVAILABLE:
        return

    try:
        from .pascucci_plotting import plot_pascucci_paper_bundle_from_artifacts
    except ImportError as exc:
        raise AssertionError("Sprint 19 needs a Pascucci paper-plot artifact pipeline") from exc

    fixture = _pascucci_paper_plot_smoke_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stitched_path = tmp_path / "stitched_predictions_final.npz"
        application_npz_path = tmp_path / "application_metrics_final.npz"
        run_config_path = tmp_path / "run_config.json"
        save_blob_npz(fixture["stitched"], str(stitched_path))
        save_blob_npz(fixture["application_pathwise"], str(application_npz_path))
        save_json(
            {
                "model_name": "pascucci",
                "T_total": 5.0,
                "params": fixture["params"],
                "state_labels": ["S", "H", "V", "X_state"],
                "application_metric_schema": "pascucci_application_metrics_v2",
                "run_config_sha256": "0" * 64,
                "seed_manifest": {"eval_seed": 1234, "visual_seed_effective": 5678},
                "blocks": fixture["blocks"],
            },
            str(run_config_path),
        )

        manifest = plot_pascucci_paper_bundle_from_artifacts(
            stitched_npz_path=str(stitched_path),
            application_npz_path=str(application_npz_path),
            run_config_path=str(run_config_path),
            out_dir=str(tmp_path / "paper_plots"),
            source_label="synthetic_smoke",
        )

        assert manifest["schema"] == "pascucci_paper_plots_v1"
        assert manifest["model_name"] == "pascucci"
        assert manifest["source"]["source_label"] == "synthetic_smoke"
        assert manifest["source"]["run_config_sha256"] == "0" * 64
        assert manifest["cost_trace_source"] == "cost_J_trajectory"
        assert manifest["controlled_uncontrolled_available"] is True
        assert manifest["plot_path_policy"] == "relative_to_manifest_dir"
        assert manifest["state_transforms"]["#35"] == "PUN price = exp(S) * 1000, matching to_ema calibration plots"
        assert "exp(stitched.X[:, :, S]) * 1000" in manifest["plots"]["#35"]["inputs"]
        assert manifest["plots"]["#35"]["style"]["source"] == "raw/to_ema/variable_mu_calibration.ipynb::generate_plot"
        assert manifest["plots"]["#35"]["style"]["real_color"] == "red"
        assert manifest["plots"]["#35"]["style"]["figure_size"] == [12, 6]
        assert manifest["plots"]["#37"]["style"]["source"] == "raw/to_ema/plotmaker.ipynb::comparison J cell"
        assert manifest["plots"]["#37"]["style"]["figure_size"] == [10, 6]
        assert manifest["plots"]["#37"]["style"]["controlled_color"] == "r"
        assert "J_t=" in manifest["plots"]["#37"]["style"]["title"]
        assert manifest["notebook_parity"]["exact_all_stories"] is False
        assert manifest["notebook_parity"]["stories"]["#37"] == "plotmaker_per_time_cost_formula_oracled"
        assert manifest["notebook_parity"]["stories"]["#38"] == "diagnostic_only_no_plotmaker_equivalent"
        assert manifest["notebook_parity"]["stories"]["#40"] == "diagnostic_only_no_plotmaker_equivalent"
        assert manifest["visual_regression"]["status"] == "structural_style_only_no_golden_images"
        assert manifest["visual_regression"]["pixel_exact_claim"] is False
        assert manifest["visual_regression"]["golden_images_available"] is False
        assert "np.random.choice" in "\n".join(manifest["visual_regression"]["remaining_gaps"])
        assert manifest["plotmaker_reference"]["data"] == "2025dicembre1"
        assert manifest["plotmaker_reference"]["dataset_status"] == "mismatch"
        assert manifest["plotmaker_reference"]["H"]["expected_basename"] == "2025dicembre1.csv"
        assert manifest["plotmaker_reference"]["S"]["expected_basename"] == "2025dicembre1.xlsx"
        assert manifest["plotmaker_native_plots"]["Y"]["filename"] == "pascucci_plotmaker_backward_Y.png"
        assert manifest["plotmaker_native_plots"]["Y"]["style"]["source"] == (
            "raw/to_ema/plotmaker.ipynb::Backward Component Y cell"
        )
        assert manifest["plotmaker_native_plots"]["Z"]["filename"] == "pascucci_plotmaker_Z_components.png"
        assert manifest["plotmaker_native_plots"]["Z"]["style"]["labels"] == ["Z_S", "Z_H", "Z_V", "Z_X"]
        assert manifest["notebook_parity"]["native_plotmaker_plots"]["Y"] == "plotmaker_backward_component_reference"
        assert manifest["notebook_parity"]["native_plotmaker_plots"]["Z"] == "plotmaker_z_components_reference"
        assert manifest["state_transforms"]["#39"] == (
            "forward components S,H,V,X in plotmaker 2x2 layout; S is plotted as exp(S)"
        )
        assert "exp(stitched.X[:, :, S])" in manifest["plots"]["#39"]["inputs"]
        assert "stitched.X[:, :, H,V,X]" in manifest["plots"]["#39"]["inputs"]
        assert "plotmaker-style q10-q90 bands, mean, and 3 sample paths" in manifest["plots"]["#39"]["inputs"]
        assert manifest["plots"]["#39"]["style"]["sample_policy"] == "deterministic_first3_for_reproducibility"
        assert "stitched.X[:, :, S,V,Q]" not in manifest["plots"]["#39"]["inputs"]
        assert "application_metrics.controlled_cost_J_trajectory" in manifest["plots"]["#37"]["inputs"]
        assert "application_metrics.uncontrolled_cost_J_trajectory" in manifest["plots"]["#37"]["inputs"]
        assert manifest["plots"]["#40"]["title"] == "Controlled vs alpha=0 same-model total cost"
        assert "uncontrolled baseline source=alpha_zero_same_model" in manifest["plots"]["#40"]["inputs"]
        assert manifest["plots"]["#40"]["style"]["bar_stat"] == "q50"
        assert manifest["plots"]["#40"]["style"]["marker_stat"] == "mean"
        assert manifest["uncontrolled_cost_trace_source"] == "application_metrics.uncontrolled_cost_J_trajectory"
        assert manifest["to_ema_uncontrolled_reference"]["source"] == "application_metrics_fallback"
        assert manifest["comparison_sources"]["#40"]["uncontrolled"] == "alpha_zero_same_model"
        assert manifest["ou_reference"]["source"] == "stitched_fallback"
        assert manifest["ou_reference"]["reason"] == "missing_pascucci_calibration_metadata"
        assert manifest["blocks"] == fixture["blocks"]
        expected_stories = {"#35", "#36", "#37", "#38", "#39", "#40"}
        assert set(manifest["plots"]) == expected_stories
        assert manifest["plot_data"]["schema"] == "pascucci_paper_plot_data_v1"
        assert manifest["plot_data"]["path"] == "pascucci_paper_plot_data.npz"
        manifest_path = tmp_path / "paper_plots" / "pascucci_paper_plots_manifest.json"
        assert manifest_path.exists()
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved_manifest["plots"].keys() == manifest["plots"].keys()
        plot_data_path = manifest_path.parent / manifest["plot_data"]["path"]
        assert plot_data_path.exists()
        with np.load(plot_data_path, allow_pickle=False) as plot_data:
            assert "plot37_controlled_mean" in plot_data.files
            assert "plot37_uncontrolled_q90" in plot_data.files
            assert "plot39_S_q10" in plot_data.files
            assert "plot39_H_mean" in plot_data.files
            assert "plot39_X_samples" in plot_data.files
            assert "plotmaker_Y_mean" in plot_data.files
            assert "plotmaker_Y_samples" in plot_data.files
            assert "plotmaker_Z_Z_V_mean" in plot_data.files
            assert "plotmaker_Z_Z_X_samples" in plot_data.files
            assert plot_data["plot37_controlled_time"].shape == plot_data["plot37_controlled_mean"].shape
            assert plot_data["plot40_controlled_q50"].shape == (1,)
            assert plot_data["plot39_sample_indices"].shape == (3,)
            assert plot_data["plot39_X_samples"].shape == (3, fixture["stitched"]["X"].shape[1])
            assert plot_data["plotmaker_Y_samples"].shape == (3, fixture["stitched"]["Y"].shape[1])
            assert plot_data["plotmaker_Z_Z_S_samples"].shape == (3, fixture["stitched"]["Z"].shape[1])
        plot39_image = plt.imread(str(manifest_path.parent / manifest["plots"]["#39"]["path"]))
        assert plot39_image.shape[1] > plot39_image.shape[0], "plotmaker #39 forward-components figure must be landscape"
        plotmaker_z_image = plt.imread(str(manifest_path.parent / manifest["plotmaker_native_plots"]["Z"]["path"]))
        assert plotmaker_z_image.shape[1] > plotmaker_z_image.shape[0], "plotmaker Z components figure must be landscape"
        for story, entry in manifest["plots"].items():
            assert not Path(entry["path"]).is_absolute(), story
            assert entry["path_relative_to"] == "manifest_dir", story
            plot_path = manifest_path.parent / entry["path"]
            assert plot_path.exists(), story
            assert plot_path.stat().st_size > 0, story
            assert entry["filename"].startswith("pascucci_paper_"), story
        for story, entry in manifest["plotmaker_native_plots"].items():
            assert not Path(entry["path"]).is_absolute(), story
            assert entry["path_relative_to"] == "manifest_dir", story
            plot_path = manifest_path.parent / entry["path"]
            assert plot_path.exists(), story
            assert plot_path.stat().st_size > 0, story


def test_pascucci_paper_plot_bundle_legacy_cost_trace_fallback() -> None:
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle

    fixture = _pascucci_paper_plot_smoke_inputs()
    legacy_pathwise = dict(fixture["application_pathwise"])
    legacy_pathwise.pop("controlled_cost_J_trajectory")
    legacy_pathwise.pop("uncontrolled_cost_J_trajectory")
    legacy_pathwise.pop("controlled_cost_J_trajectory_math")
    legacy_pathwise.pop("uncontrolled_cost_J_trajectory_math")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = plot_pascucci_paper_bundle(
            stitched=fixture["stitched"],
            application_pathwise=legacy_pathwise,
            params=fixture["params"],
            out_dir=str(Path(tmp) / "paper_plots"),
            blocks=fixture["blocks"],
            source_metadata={"model_name": "pascucci"},
        )
    assert manifest["cost_trace_source"] == "cost_J_running_cumulative"
    assert manifest["uncontrolled_cost_trace_source"] == "application_metrics.uncontrolled_cost_J_running_cumulative"
    assert "application_metrics.controlled_cost_J_running_cumulative" in manifest["plots"]["#37"]["inputs"]
    assert "application_metrics.uncontrolled_cost_J_running_cumulative" in manifest["plots"]["#37"]["inputs"]


def test_pascucci_paper_plot_bundle_uses_to_ema_calibration_sources() -> None:
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
    )
    from .pascucci_data import prepare_H, prepare_S
    from .pascucci_plotting import _load_to_ema_ou_reference, _load_to_ema_uncontrolled_reference
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle

    n_points = 120
    H_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.25,
        a0_day=0.80,
        a0_night=-0.30,
    )
    log_S_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.05,
        a0_day=0.30,
        a0_night=-0.12,
    )
    S_hours = np.exp(log_S_hours)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_plot_fixture.csv"
        S_path = tmp_path / "S_plot_fixture.xlsx"
        H_rows = []
        for value in H_hours:
            for _ in range(4):
                production = 5000.0
                consumption = production + float(value) / 0.001
                H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
        S_rows = [["fixture-day", "fixture-hour", f"{float(value) / 0.01:.12f}"] for value in S_hours]
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        prepared_H = prepare_H(str(H_path), n=4, mul_factor=0.001)
        prepared_S = prepare_S(str(S_path), n=1, mul_factor=0.01)
        calibration = calibrate_pascucci_ou_inputs(
            prepared_H,
            prepared_S,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
        )
        config = build_pascucci_calibration_config(
            calibration,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
            H_metadata={
                "source_path": str(H_path),
                "units": "kW",
                "n_per_hour": 4,
                "mul_factor": 0.001,
            },
            S_metadata={
                "source_path": str(S_path),
                "units": "EUR_per_MWh",
                "n_per_hour": 1,
                "mul_factor": 0.01,
            },
        )
        params = build_pascucci_run_config_params(
            get_model_spec("pascucci").build_default_params(),
            config,
        )
        fixture = _pascucci_paper_plot_smoke_inputs(M=5)
        manifest = plot_pascucci_paper_bundle(
            stitched=fixture["stitched"],
            application_pathwise=fixture["application_pathwise"],
            params=params,
            out_dir=str(tmp_path / "paper_plots"),
            blocks=fixture["blocks"],
            source_metadata={"model_name": "pascucci"},
        )

        assert manifest["ou_reference"]["source"] == "calibration_metadata"
        assert manifest["ou_reference"]["simulation"]["dt_sim"] == 0.5
        assert manifest["ou_reference"]["simulation"]["n_sim"] == 10000
        assert manifest["ou_reference"]["simulation"]["seed"] == 42
        assert manifest["ou_reference"]["#35"]["real_path_points"] == n_points
        assert manifest["ou_reference"]["#36"]["real_path_points"] == n_points
        assert manifest["ou_reference"]["#35"]["transform"] == "exp(logS) * 1000"
        assert manifest["uncontrolled_cost_trace_source"] == "to_ema_raw_uncontrolled_J"
        assert manifest["to_ema_uncontrolled_reference"]["source"] == "calibration_metadata"
        assert manifest["to_ema_uncontrolled_reference"]["simulation"]["dt"] == 0.1
        assert manifest["to_ema_uncontrolled_reference"]["simulation"]["n_sim"] == 10000
        assert manifest["comparison_sources"]["#37"]["uncontrolled"] == "to_ema_raw_uncontrolled_J"
        assert manifest["comparison_sources"]["#40"]["uncontrolled"] == "to_ema_raw_uncontrolled_J_final"
        assert (
            manifest["comparison_sources"]["#40"]["paired_alpha_zero_detail"]
            == "application_metrics.uncontrolled_cost_J_total"
        )
        assert "to_ema.raw_uncontrolled_J_paths" in manifest["plots"]["#37"]["inputs"]
        assert manifest["plots"]["#40"]["title"] == "Controlled vs raw to_ema final cost"
        assert "to_ema.raw_uncontrolled_J_paths[:, -1]" in manifest["plots"]["#40"]["inputs"]
        assert "paired alpha=0 total still saved as application_metrics.uncontrolled_cost_J_total" in manifest[
            "plots"
        ]["#40"]["inputs"]
        assert "pascucci_data.prepare_S" in manifest["plots"]["#35"]["inputs"][0]
        assert "simulate_ou_day_night" in manifest["plots"]["#35"]["inputs"][2]
        assert "real_S * 1000" in manifest["plots"]["#35"]["inputs"]
        assert "pascucci_data.prepare_H" in manifest["plots"]["#36"]["inputs"][0]
        assert "real_H" in manifest["plots"]["#36"]["inputs"]
        plot_data_path = tmp_path / "paper_plots" / manifest["plot_data"]["path"]
        assert plot_data_path.exists()
        with np.load(plot_data_path, allow_pickle=False) as plot_data:
            assert plot_data["plot35_time_sim"].shape == plot_data["plot35_sim_q05"].shape
            assert plot_data["plot36_time_sim"].shape == plot_data["plot36_sim_q95"].shape
            assert plot_data["plot37_uncontrolled_mean"].shape[0] > plot_data["plot37_controlled_mean"].shape[0]
        for story in ("#35", "#36"):
            plot_path = tmp_path / "paper_plots" / manifest["plots"][story]["path"]
            assert plot_path.exists(), story
            assert plot_path.stat().st_size > 0, story


def test_pascucci_paper_plot_ou_calibration_sources_are_pass_invariant() -> None:
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
    )
    from .pascucci_data import prepare_H, prepare_S
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from . import pascucci_plotting as plotting

    n_points = 72
    H_hours = _simulate_piecewise_ou_series(n_points=n_points, dt=1.0, x0=0.25, a0_day=0.8, a0_night=-0.3)
    S_hours = np.exp(
        _simulate_piecewise_ou_series(n_points=n_points, dt=1.0, x0=0.05, a0_day=0.3, a0_night=-0.12)
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_pass_invariant.csv"
        S_path = tmp_path / "S_pass_invariant.xlsx"
        H_rows = []
        for value in H_hours:
            for _ in range(4):
                production = 5000.0
                consumption = production + float(value) / 0.001
                H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
        S_rows = [["fixture-day", "fixture-hour", f"{float(value) / 0.01:.12f}"] for value in S_hours]
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        prepared_H = prepare_H(str(H_path), n=4, mul_factor=0.001)
        prepared_S = prepare_S(str(S_path), n=1, mul_factor=0.01)
        calibration = calibrate_pascucci_ou_inputs(prepared_H, prepared_S, K=0, dt=1.0, start_hour=0.0, log_price=True)
        config = build_pascucci_calibration_config(
            calibration,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
            H_metadata={"source_path": str(H_path), "n_per_hour": 4, "mul_factor": 0.001},
            S_metadata={"source_path": str(S_path), "n_per_hour": 1, "mul_factor": 0.01},
        )
        params = build_pascucci_run_config_params(get_model_spec("pascucci").build_default_params(), config)
        fixture = _pascucci_paper_plot_smoke_inputs(M=4)
        altered = {
            **fixture["stitched"],
            "X": np.array(fixture["stitched"]["X"], copy=True),
        }
        altered["X"][:, :, 0] += np.float32(100.0)
        altered["X"][:, :, 1] -= np.float32(50.0)

        calls: Dict[str, List[Dict[str, np.ndarray]]] = {"first": [], "second": []}
        active = {"label": "first"}
        original_ou = plotting._plot_to_ema_ou_simulation
        original_state = plotting._plot_state_with_ou
        original_cost = plotting._plot_accumulated_cost
        original_alpha = plotting._plot_alpha
        original_bands = plotting._plot_state_bands
        original_total = plotting._plot_controlled_uncontrolled

        def fake_ou(**kwargs):
            paths = np.asarray(kwargs["paths"], dtype=np.float32)
            real_path = np.asarray(kwargs["real_path"], dtype=np.float32)
            time_sim = np.arange(paths.shape[1], dtype=np.float32) * np.float32(kwargs["dt_sim"])
            time_real = np.arange(real_path.shape[0], dtype=np.float32) * np.float32(kwargs["dt_real"])
            calls[active["label"]].append(
                {
                    "paths": paths.copy(),
                    "real_path": real_path.copy(),
                    "T": np.asarray([float(kwargs["T"])], dtype=np.float32),
                    "dt_real": np.asarray([float(kwargs["dt_real"])], dtype=np.float32),
                    "dt_sim": np.asarray([float(kwargs["dt_sim"])], dtype=np.float32),
                }
            )
            Path(str(kwargs["path"])).write_bytes(b"fake-png")
            return {
                "time_sim": time_sim,
                "sim_mean": np.mean(paths, axis=0).astype(np.float32),
                "sim_q05": np.percentile(paths, 5, axis=0).astype(np.float32),
                "sim_q50": np.percentile(paths, 50, axis=0).astype(np.float32),
                "sim_q95": np.percentile(paths, 95, axis=0).astype(np.float32),
                "time_real": time_real.astype(np.float32),
                "real": real_path.astype(np.float32),
            }

        def fail_state_fallback(**_kwargs):
            raise AssertionError("#35/#36 must use calibration_metadata, not stitched fallback")

        def fake_other(**kwargs):
            Path(str(kwargs["path"])).write_bytes(b"fake-png")
            return {"dummy": np.asarray([1.0], dtype=np.float32)}

        try:
            plotting._plot_to_ema_ou_simulation = fake_ou
            plotting._plot_state_with_ou = fail_state_fallback
            plotting._plot_accumulated_cost = fake_other
            plotting._plot_alpha = fake_other
            plotting._plot_state_bands = fake_other
            plotting._plot_controlled_uncontrolled = fake_other

            manifest_first = plotting.plot_pascucci_paper_bundle(
                stitched=fixture["stitched"],
                application_pathwise=fixture["application_pathwise"],
                params=params,
                out_dir=str(tmp_path / "plots_first"),
                blocks=fixture["blocks"],
                source_metadata={"model_name": "pascucci", "pass_label": "pass0"},
            )
            active["label"] = "second"
            manifest_second = plotting.plot_pascucci_paper_bundle(
                stitched=altered,
                application_pathwise=fixture["application_pathwise"],
                params=params,
                out_dir=str(tmp_path / "plots_second"),
                blocks=fixture["blocks"],
                source_metadata={"model_name": "pascucci", "pass_label": "pass1"},
            )
        finally:
            plotting._plot_to_ema_ou_simulation = original_ou
            plotting._plot_state_with_ou = original_state
            plotting._plot_accumulated_cost = original_cost
            plotting._plot_alpha = original_alpha
            plotting._plot_state_bands = original_bands
            plotting._plot_controlled_uncontrolled = original_total

    assert manifest_first["ou_reference"]["source"] == "calibration_metadata"
    assert manifest_second["ou_reference"]["source"] == "calibration_metadata"
    assert len(calls["first"]) == 2
    assert len(calls["second"]) == 2
    for first, second in zip(calls["first"], calls["second"]):
        for key in ("paths", "real_path", "T", "dt_real", "dt_sim"):
            np.testing.assert_array_equal(first[key], second[key])


def test_pascucci_to_ema_ou_reference_rejects_basename_fallback_sha_mismatch() -> None:
    from .model_specs import get_model_spec
    from .pascucci_plotting import _load_to_ema_ou_reference

    params = get_model_spec("pascucci").build_default_params()
    params["pascucci_calibration"] = {
        "schema": "pascucci_ou_calibration_v1",
        "K": 0,
        "dt": 1.0,
        "start_hour": 0.0,
        "log_price": True,
        "S_transform": "log",
        "H_transform": "linear",
        "H_metadata": {
            "source_path": "/definitely/missing/2025dicembre1.csv",
            "source_sha256": "0" * 64,
            "n_per_hour": 4,
            "mul_factor": 0.001,
        },
        "S_metadata": {
            "source_path": "/definitely/missing/2025dicembre1.xlsx",
            "source_sha256": "0" * 64,
            "n_per_hour": 1,
            "mul_factor": 0.001,
        },
    }

    references, summary = _load_to_ema_ou_reference(params, horizon_T=24.0)

    assert references == {}
    assert summary["source"] == "stitched_fallback"
    assert summary["reason"] == "calibration_source_sha256_mismatch"
    labels = {entry["label"] for entry in summary["mismatches"]}
    assert labels == {"H_metadata", "S_metadata"}
    for entry in summary["mismatches"]:
        assert "resolved_source_path" in entry
        assert entry["expected_source_sha256"] == "0" * 64


def test_pascucci_to_ema_ou_reference_matches_legacy_formula() -> None:
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
    )
    from .pascucci_data import prepare_H, prepare_S
    from .pascucci_plotting import _load_to_ema_ou_reference

    def legacy_is_day(t):
        hour = np.mod(t, 24.0)
        return (hour >= 7.0) & (hour < 19.0)

    def legacy_mu_t(t, a0, alpha, beta):
        value = a0
        for k, (alpha_k, beta_k) in enumerate(zip(alpha, beta), start=1):
            omega = 2.0 * np.pi * float(k) / 24.0
            value += float(alpha_k) * np.cos(omega * t)
            value += float(beta_k) * np.sin(omega * t)
        return value

    def legacy_mu_daynight(t, params):
        return np.where(
            legacy_is_day(t),
            legacy_mu_t(t, params["a0_day"], params["alpha_day"], params["beta_day"]),
            legacy_mu_t(t, params["a0_night"], params["alpha_night"], params["beta_night"]),
        )

    def legacy_simulate_ou(H0, params, *, T, Nsim=10000, dt=0.5, seed=42):
        np.random.seed(int(seed))
        Nt = int(float(T) / float(dt))
        paths = np.zeros((int(Nsim), Nt + 1), dtype=np.float64)
        paths[:, 0] = float(H0)
        for i in range(Nt):
            t_actual = float(i) * float(dt)
            kappa = np.where(legacy_is_day(t_actual), params["kappa_day"], params["kappa_night"])
            sigma = np.where(legacy_is_day(t_actual), params["sigma_day"], params["sigma_night"])
            mu = legacy_mu_daynight(t_actual, params)
            Z = np.random.randn(int(Nsim))
            paths[:, i + 1] = paths[:, i] + kappa * (mu - paths[:, i]) * float(dt) + sigma * Z * np.sqrt(float(dt))
        return paths

    n_points = 72
    H_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.25,
        a0_day=0.80,
        a0_night=-0.30,
        sigma_day=0.01,
        sigma_night=0.02,
        seed=111,
    )
    log_S_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=0.0,
        x0=0.05,
        a0_day=0.30,
        a0_night=-0.12,
        sigma_day=0.005,
        sigma_night=0.006,
        seed=222,
    )
    S_hours = np.exp(log_S_hours)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_legacy_oracle.csv"
        S_path = tmp_path / "S_legacy_oracle.xlsx"
        H_rows = []
        for value in H_hours:
            for _ in range(4):
                production = 5000.0
                consumption = production + float(value) / 0.001
                H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
        S_rows = [["fixture-day", "fixture-hour", f"{float(value) / 0.01:.12f}"] for value in S_hours]
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        H = prepare_H(str(H_path), n=4, mul_factor=0.001)
        S = prepare_S(str(S_path), n=1, mul_factor=0.01)
        calibration = calibrate_pascucci_ou_inputs(H, S, K=0, dt=1.0, start_hour=0.0, log_price=True)
        config = build_pascucci_calibration_config(
            calibration,
            K=0,
            dt=1.0,
            start_hour=0.0,
            log_price=True,
            H_metadata={"source_path": str(H_path), "n_per_hour": 4, "mul_factor": 0.001},
            S_metadata={"source_path": str(S_path), "n_per_hour": 1, "mul_factor": 0.01},
        )
        params = build_pascucci_run_config_params(get_model_spec("pascucci").build_default_params(), config)
        references, summary = _load_to_ema_ou_reference(params, horizon_T=24.0)

    assert summary["source"] == "calibration_metadata"
    assert summary["source_horizon_policy"] == "full_calibration_series_like_to_ema"
    T_H = float(H.shape[0] - 1)
    T_S = float(S.shape[0] - 1)
    legacy_H = legacy_simulate_ou(H[0], params["params_H"], T=T_H)
    legacy_S = np.exp(legacy_simulate_ou(np.log(S[0]), params["params_S"], T=T_S)) * 1000.0

    np.testing.assert_allclose(references["#36"]["real_path"], H.astype(np.float32), rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(references["#35"]["real_path"], (S * 1000.0).astype(np.float32), rtol=0.0, atol=2.0e-4)
    np.testing.assert_allclose(references["#36"]["paths"], legacy_H, rtol=2.0e-6, atol=5.0e-6)
    np.testing.assert_allclose(references["#35"]["paths"], legacy_S, rtol=2.0e-6, atol=5.0e-4)
    np.testing.assert_allclose(
        np.percentile(references["#36"]["paths"], [5, 95], axis=0),
        np.percentile(legacy_H, [5, 95], axis=0),
        rtol=2.0e-6,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        np.percentile(references["#35"]["paths"], [5, 95], axis=0),
        np.percentile(legacy_S, [5, 95], axis=0),
        rtol=2.0e-6,
        atol=5.0e-4,
    )


def test_pascucci_to_ema_ou_reference_honors_harmonics_and_start_hour() -> None:
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
    )
    from .pascucci_data import prepare_H, prepare_S
    from . import pascucci_plotting as plotting

    def legacy_is_day(t):
        hour = np.mod(t, 24.0)
        return (hour >= 7.0) & (hour < 19.0)

    def legacy_mu_t(t, a0, alpha, beta):
        value = a0
        for k, (alpha_k, beta_k) in enumerate(zip(alpha, beta), start=1):
            omega = 2.0 * np.pi * float(k) / 24.0
            value += float(alpha_k) * np.cos(omega * t)
            value += float(beta_k) * np.sin(omega * t)
        return value

    def legacy_mu_daynight(t, params):
        return np.where(
            legacy_is_day(t),
            legacy_mu_t(t, params["a0_day"], params["alpha_day"], params["beta_day"]),
            legacy_mu_t(t, params["a0_night"], params["alpha_night"], params["beta_night"]),
        )

    def legacy_simulate_ou(initial, params, *, T, Nsim, dt, seed, start_hour):
        np.random.seed(int(seed))
        Nt = int(float(T) / float(dt))
        paths = np.zeros((int(Nsim), Nt + 1), dtype=np.float64)
        paths[:, 0] = float(initial)
        for i in range(Nt):
            t_actual = float(start_hour) + float(i) * float(dt)
            kappa = np.where(legacy_is_day(t_actual), params["kappa_day"], params["kappa_night"])
            sigma = np.where(legacy_is_day(t_actual), params["sigma_day"], params["sigma_night"])
            mu = legacy_mu_daynight(t_actual, params)
            Z = np.random.randn(int(Nsim))
            paths[:, i + 1] = paths[:, i] + kappa * (mu - paths[:, i]) * float(dt) + sigma * Z * np.sqrt(float(dt))
        return paths

    n_points = 144
    start_hour = 6.0
    H_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=start_hour,
        x0=0.15,
        kappa_day=0.22,
        kappa_night=0.17,
        a0_day=0.70,
        a0_night=-0.35,
        alpha_day=[0.25, -0.08],
        alpha_night=[-0.11, 0.05],
        beta_day=[0.09, 0.04],
        beta_night=[0.07, -0.03],
        sigma_day=0.012,
        sigma_night=0.018,
        seed=333,
    )
    log_S_hours = _simulate_piecewise_ou_series(
        n_points=n_points,
        dt=1.0,
        start_hour=start_hour,
        x0=0.03,
        kappa_day=0.19,
        kappa_night=0.14,
        a0_day=0.18,
        a0_night=-0.06,
        alpha_day=[0.04, -0.02],
        alpha_night=[-0.03, 0.01],
        beta_day=[0.02, 0.015],
        beta_night=[0.025, -0.01],
        sigma_day=0.004,
        sigma_night=0.006,
        seed=444,
    )
    S_hours = np.exp(log_S_hours)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        H_path = tmp_path / "H_k2_start6_oracle.csv"
        S_path = tmp_path / "S_k2_start6_oracle.xlsx"
        H_rows = []
        for value in H_hours:
            for _ in range(3):
                production = 7000.0
                consumption = production + float(value) / 0.002
                H_rows.append({"Consumo (W)": f"{consumption:.12f}", "Produzione (W)": f"{production:.12f}"})
        S_rows = [["fixture-day", "fixture-hour", f"{float(value) / 0.02:.12f}"] for value in S_hours]
        _write_rows_csv(H_path, ["Consumo (W)", "Produzione (W)"], H_rows)
        _write_minimal_xlsx(S_path, ["Data", "Ora", "€/MWh"], S_rows)

        H = prepare_H(str(H_path), n=3, mul_factor=0.002)
        S = prepare_S(str(S_path), n=1, mul_factor=0.02)
        calibration = calibrate_pascucci_ou_inputs(H, S, K=2, dt=1.0, start_hour=start_hour, log_price=True)
        config = build_pascucci_calibration_config(
            calibration,
            K=2,
            dt=1.0,
            start_hour=start_hour,
            log_price=True,
            H_metadata={"source_path": str(H_path), "n_per_hour": 3, "mul_factor": 0.002},
            S_metadata={"source_path": str(S_path), "n_per_hour": 1, "mul_factor": 0.02},
        )
        params = build_pascucci_run_config_params(get_model_spec("pascucci").build_default_params(), config)
        old_values = (
            plotting.TO_EMA_OU_NSIM,
            plotting.TO_EMA_OU_DT_SIM,
            plotting.TO_EMA_OU_SEED,
        )
        try:
            plotting.TO_EMA_OU_NSIM = 16
            plotting.TO_EMA_OU_DT_SIM = 0.5
            plotting.TO_EMA_OU_SEED = 123
            references, summary = plotting._load_to_ema_ou_reference(params, horizon_T=24.0)
        finally:
            (
                plotting.TO_EMA_OU_NSIM,
                plotting.TO_EMA_OU_DT_SIM,
                plotting.TO_EMA_OU_SEED,
            ) = old_values

    assert summary["source"] == "calibration_metadata"
    assert summary["start_hour"] == start_hour
    assert summary["simulation"]["n_sim"] == 16
    assert summary["simulation"]["seed"] == 123
    assert len(params["params_H"]["alpha_day"]) == 2
    assert len(params["params_S"]["alpha_day"]) == 2

    T_H = float(H.shape[0] - 1)
    T_S = float(S.shape[0] - 1)
    legacy_H = legacy_simulate_ou(H[0], params["params_H"], T=T_H, Nsim=16, dt=0.5, seed=123, start_hour=start_hour)
    legacy_S_log = legacy_simulate_ou(
        np.log(S[0]),
        params["params_S"],
        T=T_S,
        Nsim=16,
        dt=0.5,
        seed=123,
        start_hour=start_hour,
    )
    wrong_phase_H = legacy_simulate_ou(H[0], params["params_H"], T=T_H, Nsim=16, dt=0.5, seed=123, start_hour=0.0)

    np.testing.assert_allclose(references["#36"]["paths"], legacy_H, rtol=2.0e-6, atol=5.0e-6)
    np.testing.assert_allclose(references["#35"]["paths"], np.exp(legacy_S_log) * 1000.0, rtol=2.0e-6, atol=5.0e-4)
    assert float(np.max(np.abs(references["#36"]["paths"] - wrong_phase_H))) > 1.0e-3


def test_pascucci_to_ema_uncontrolled_reference_matches_legacy_formula() -> None:
    from .model_specs import get_model_spec
    from . import pascucci_plotting as plotting

    def legacy_is_day(t):
        hour = np.mod(t, 24.0)
        return (hour >= 7.0) & (hour < 19.0)

    def legacy_mu_t(t, a0, alpha, beta):
        value = a0
        for k, (alpha_k, beta_k) in enumerate(zip(alpha, beta), start=1):
            omega = 2.0 * np.pi * float(k) / 24.0
            value += float(alpha_k) * np.cos(omega * t)
            value += float(beta_k) * np.sin(omega * t)
        return value

    def legacy_mu_daynight(t, params):
        return np.where(
            legacy_is_day(t),
            legacy_mu_t(t, params["a0_day"], params["alpha_day"], params["beta_day"]),
            legacy_mu_t(t, params["a0_night"], params["alpha_night"], params["beta_night"]),
        )

    def legacy_simulate_ou(H0, params, *, T, Nsim, dt, seed):
        np.random.seed(int(seed))
        Nt = int(float(T) / float(dt))
        paths = np.zeros((int(Nsim), Nt + 1), dtype=np.float64)
        paths[:, 0] = np.asarray(H0, dtype=np.float64).reshape(int(Nsim))
        for i in range(Nt):
            t_actual = float(i) * float(dt)
            kappa = np.where(legacy_is_day(t_actual), params["kappa_day"], params["kappa_night"])
            sigma = np.where(legacy_is_day(t_actual), params["sigma_day"], params["sigma_night"])
            mu = legacy_mu_daynight(t_actual, params)
            Z = np.random.randn(int(Nsim))
            paths[:, i + 1] = paths[:, i] + kappa * (mu - paths[:, i]) * float(dt) + sigma * Z * np.sqrt(float(dt))
        return paths

    def legacy_make_h(values, scale):
        arr = np.asarray(values, dtype=np.float64)
        return np.where(arr < 0.0, float(scale) * arr * arr, 2.0 * float(scale) * arr * arr)

    def legacy_x_v(X_0, H_paths, *, T, Nsim, dt, eps, xmax, seed):
        np.random.seed(int(seed))
        Nt = int(float(T) / float(dt)) + 1
        X_paths = np.zeros((int(Nsim), Nt), dtype=np.float64)
        V_paths = np.zeros((int(Nsim), Nt), dtype=np.float64)
        X_paths[:, 0] = X_0
        V_paths[:, 0] = np.where(
            X_paths[:, 0] <= 0.0,
            -np.minimum(H_paths[:, 0], 0.0),
            np.where(X_paths[:, 0] >= float(xmax), -np.maximum(H_paths[:, 0], 0.0), -H_paths[:, 0]),
        )
        W_paths = np.cumsum(np.random.randn(int(Nsim), Nt) * np.sqrt(float(dt)), axis=-1)
        for step in range(Nt - 1):
            X_paths[:, step + 1] = X_paths[:, step] + V_paths[:, step] * float(dt)
            V_paths[:, step + 1] = (
                np.where(
                    X_paths[:, step + 1] <= 0.0,
                    -np.minimum(H_paths[:, step + 1], 0.0),
                    np.where(
                        X_paths[:, step + 1] >= float(xmax),
                        -np.maximum(H_paths[:, step + 1], 0.0),
                        -H_paths[:, step + 1],
                    ),
                )
                + float(eps) * W_paths[:, step + 1]
            )
        return X_paths, V_paths

    params = get_model_spec("pascucci").build_default_params()
    params["pascucci_calibration"] = {
        "schema": "pascucci_ou_calibration_v1",
        "start_hour": 0.0,
    }
    old_values = (
        plotting.TO_EMA_UNCONTROLLED_NSIM,
        plotting.TO_EMA_UNCONTROLLED_DT,
        plotting.TO_EMA_UNCONTROLLED_SEED,
    )
    try:
        plotting.TO_EMA_UNCONTROLLED_NSIM = 8
        plotting.TO_EMA_UNCONTROLLED_DT = 0.5
        plotting.TO_EMA_UNCONTROLLED_SEED = 7
        references, summary = plotting._load_to_ema_uncontrolled_reference(params, horizon_T=1.0)
    finally:
        (
            plotting.TO_EMA_UNCONTROLLED_NSIM,
            plotting.TO_EMA_UNCONTROLLED_DT,
            plotting.TO_EMA_UNCONTROLLED_SEED,
        ) = old_values

    n_sim = 8
    dt = 0.5
    seed = 7
    rng = np.random.RandomState(seed)
    X_0 = rng.uniform(1.0, 9.0, n_sim)
    H_0 = rng.normal(0.4, 0.5, n_sim)
    S_0 = rng.normal(0.1, 0.02, n_sim)
    legacy_H = legacy_simulate_ou(H_0, params["params_H"], T=1.0, Nsim=n_sim, dt=dt, seed=seed)
    legacy_S = np.exp(legacy_simulate_ou(np.log(S_0), params["params_S"], T=1.0, Nsim=n_sim, dt=dt, seed=seed))
    legacy_X, legacy_V = legacy_x_v(
        X_0,
        legacy_H,
        T=1.0,
        Nsim=n_sim,
        dt=dt,
        eps=0.01,
        xmax=10.0,
        seed=seed,
    )
    mean_X = np.mean(legacy_X, axis=0)
    mean_H = np.mean(legacy_H, axis=0)
    mean_V = np.mean(legacy_V, axis=0)
    legacy_f = (
        legacy_S * (legacy_H + legacy_V)
        + 1.0e-2 * legacy_V**2
        + legacy_make_h(legacy_X - mean_X.reshape(1, -1), 0.001)
        + legacy_make_h(legacy_H + legacy_V - (mean_H + mean_V).reshape(1, -1), 0.1)
    )
    legacy_g = -legacy_S * legacy_X + 0.5e-2 * (legacy_X - mean_X.reshape(1, -1)) ** 2
    legacy_J = np.cumsum(legacy_f * dt, axis=-1) + legacy_g

    assert summary["source"] == "calibration_metadata"
    assert summary["simulation"]["n_sim"] == n_sim
    assert summary["simulation"]["dt"] == dt
    assert summary["simulation"]["seed"] == seed
    np.testing.assert_allclose(references["time"], np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(references["J_paths"], legacy_J, rtol=2.0e-6, atol=5.0e-6)


def test_pascucci_paper_plot_bundle_handles_skewed_total_costs() -> None:
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle

    controlled = np.concatenate(
        [
            np.zeros((96, 1), dtype=np.float32),
            np.full((4, 1), 1000.0, dtype=np.float32),
        ],
        axis=0,
    )
    uncontrolled = np.concatenate(
        [
            np.full((4, 1), -1000.0, dtype=np.float32),
            np.zeros((96, 1), dtype=np.float32),
        ],
        axis=0,
    )
    assert float(np.mean(controlled)) > float(np.quantile(controlled.reshape(-1), 0.95))
    assert float(np.mean(uncontrolled)) < float(np.quantile(uncontrolled.reshape(-1), 0.05))

    fixture = _pascucci_paper_plot_smoke_inputs(M=100)
    pathwise = dict(fixture["application_pathwise"])
    pathwise["controlled_cost_J_total"] = controlled
    pathwise["uncontrolled_cost_J_total"] = uncontrolled
    with tempfile.TemporaryDirectory() as tmp:
        manifest = plot_pascucci_paper_bundle(
            stitched=fixture["stitched"],
            application_pathwise=pathwise,
            params=fixture["params"],
            out_dir=str(Path(tmp) / "paper_plots"),
            blocks=fixture["blocks"],
            source_metadata={"model_name": "pascucci"},
        )
        plot_path = Path(tmp) / "paper_plots" / manifest["plots"]["#40"]["path"]
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


def test_pascucci_paper_plot_bundle_loads_blocks_from_recursive_results() -> None:
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle_from_artifacts

    fixture = _pascucci_paper_plot_smoke_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_20260610_000000"
        rec_dir = run_dir / "recursive"
        rec_dir.mkdir(parents=True)
        save_json(
            {
                "model_name": "pascucci",
                "T_total": 5.0,
                "params": fixture["params"],
                "state_labels": ["S", "H", "V", "X_state"],
                "application_metric_schema": "pascucci_application_metrics_v2",
                "run_config_sha256": "2" * 64,
                "seed_manifest": {"eval_seed": 11, "visual_seed_effective": 22},
            },
            str(run_dir / "run_config.json"),
        )
        save_json({"blocks": fixture["blocks"]}, str(rec_dir / "results.json"))
        save_blob_npz(fixture["stitched"], str(rec_dir / "stitched_predictions_final.npz"))
        save_blob_npz(fixture["application_pathwise"], str(rec_dir / "application_metrics_final.npz"))

        manifest = plot_pascucci_paper_bundle_from_artifacts(
            stitched_npz_path=str(rec_dir / "stitched_predictions_final.npz"),
            application_npz_path=str(rec_dir / "application_metrics_final.npz"),
            run_config_path=str(run_dir / "run_config.json"),
            out_dir=str(rec_dir / "plots" / "pascucci_paper"),
        )

        assert manifest["blocks"] == fixture["blocks"]


def test_pascucci_paper_plot_bundle_rejects_non_pascucci_run_config() -> None:
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle_from_artifacts

    fixture = _pascucci_paper_plot_smoke_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stitched_path = tmp_path / "stitched_predictions_final.npz"
        application_npz_path = tmp_path / "application_metrics_final.npz"
        run_config_path = tmp_path / "run_config.json"
        save_blob_npz(fixture["stitched"], str(stitched_path))
        save_blob_npz(fixture["application_pathwise"], str(application_npz_path))
        save_json(
            {
                "model_name": "quadratic_coupled",
                "params": fixture["params"],
            },
            str(run_config_path),
        )
        try:
            plot_pascucci_paper_bundle_from_artifacts(
                stitched_npz_path=str(stitched_path),
                application_npz_path=str(application_npz_path),
                run_config_path=str(run_config_path),
                out_dir=str(tmp_path / "paper_plots"),
            )
        except ValueError as exc:
            assert "model_name='pascucci'" in str(exc)
            assert "quadratic_coupled" in str(exc)
        else:
            raise AssertionError("Pascucci paper plots must reject non-Pascucci run configs")


def test_pascucci_paper_plot_bundle_rejects_incompatible_application_schema() -> None:
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle_from_artifacts

    fixture = _pascucci_paper_plot_smoke_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stitched_path = tmp_path / "stitched_predictions_final.npz"
        application_npz_path = tmp_path / "application_metrics_final.npz"
        run_config_path = tmp_path / "run_config.json"
        save_blob_npz(fixture["stitched"], str(stitched_path))
        save_blob_npz(fixture["application_pathwise"], str(application_npz_path))
        save_json(
            {
                "model_name": "pascucci",
                "params": fixture["params"],
                "application_metric_schema": "pascucci_application_metrics_v1",
            },
            str(run_config_path),
        )
        try:
            plot_pascucci_paper_bundle_from_artifacts(
                stitched_npz_path=str(stitched_path),
                application_npz_path=str(application_npz_path),
                run_config_path=str(run_config_path),
                out_dir=str(tmp_path / "paper_plots"),
            )
        except ValueError as exc:
            message = str(exc)
            assert "application_metric_schema" in message
            assert "pascucci_application_metrics_v2" in message
            assert "pascucci_application_metrics_v1" in message
        else:
            raise AssertionError("Pascucci paper plots must reject incompatible application metric schemas")


def test_pascucci_paper_plot_bundle_rejects_legacy_artifacts_without_cumulative_cost() -> None:
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    from .pascucci_plotting import plot_pascucci_paper_bundle

    fixture = _pascucci_paper_plot_smoke_inputs()
    legacy_pathwise = dict(fixture["application_pathwise"])
    del legacy_pathwise["controlled_cost_J_running_cumulative"]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            plot_pascucci_paper_bundle(
                stitched=fixture["stitched"],
                application_pathwise=legacy_pathwise,
                params=fixture["params"],
                out_dir=str(Path(tmp) / "paper_plots"),
                blocks=fixture["blocks"],
                source_metadata={"model_name": "pascucci"},
            )
        except ValueError as exc:
            message = str(exc)
            assert "controlled_cost_J_running_cumulative" in message
            assert "pascucci_paper_plots_v1" in message
            assert "Sprint 19 cumulative running-cost traces" in message
        else:
            raise AssertionError("legacy Pascucci artifacts without cumulative cost must fail clearly")


def test_cli_plot_pascucci_paper_from_artifacts_does_not_train() -> None:
    from . import cli
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    fixture = _pascucci_paper_plot_smoke_inputs()
    original_run_program = cli.run_program

    def forbidden_run_program(argv=None):
        del argv
        raise AssertionError("plot-only command must not call run_program or start training")

    cli.run_program = forbidden_run_program
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_20260610_000000"
            rec_dir = run_dir / "recursive"
            rec_dir.mkdir(parents=True)
            save_json(
                {
                    "model_name": "pascucci",
                    "T_total": 5.0,
                    "params": fixture["params"],
                    "state_labels": ["S", "H", "V", "X_state"],
                    "application_metric_schema": "pascucci_application_metrics_v2",
                    "run_config_sha256": "1" * 64,
                    "seed_manifest": {"eval_seed": 11, "visual_seed_effective": 22},
                    "blocks": fixture["blocks"],
                },
                str(run_dir / "run_config.json"),
            )
            save_blob_npz(fixture["stitched"], str(rec_dir / "stitched_predictions_final.npz"))
            save_blob_npz(fixture["application_pathwise"], str(rec_dir / "application_metrics_final.npz"))
            exit_code = cli.main(["plot", "--run_dir", str(run_dir)])
            assert exit_code == 0
            manifest_path = rec_dir / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert set(manifest["plots"]) == {"#35", "#36", "#37", "#38", "#39", "#40"}
            assert manifest["source"]["run_dir"] == str(run_dir)
            assert manifest["blocks"] == fixture["blocks"]
            for entry in manifest["plots"].values():
                assert not Path(entry["path"]).is_absolute()
                assert (manifest_path.parent / entry["path"]).exists()
    finally:
        cli.run_program = original_run_program


def test_cli_plot_pascucci_paper_resolves_single_nested_run_dir() -> None:
    from . import cli
    from .io_utils import save_blob_npz, save_json
    from .plotting import _PLOTTING_AVAILABLE

    if not _PLOTTING_AVAILABLE:
        return

    fixture = _pascucci_paper_plot_smoke_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        wrapper_dir = Path(tmp) / "pascucci_T12_wrapper_job999999"
        run_dir = wrapper_dir / "run_20260611_153158"
        rec_dir = run_dir / "recursive"
        rec_dir.mkdir(parents=True)
        save_json(
            {
                "model_name": "pascucci",
                "T_total": 5.0,
                "params": fixture["params"],
                "state_labels": ["S", "H", "V", "X_state"],
                "application_metric_schema": "pascucci_application_metrics_v2",
                "run_config_sha256": "3" * 64,
                "seed_manifest": {"eval_seed": 11, "visual_seed_effective": 22},
                "blocks": fixture["blocks"],
            },
            str(run_dir / "run_config.json"),
        )
        save_blob_npz(fixture["stitched"], str(rec_dir / "stitched_predictions_final.npz"))
        save_blob_npz(fixture["application_pathwise"], str(rec_dir / "application_metrics_final.npz"))

        exit_code = cli.main(["plot", "--run_dir", str(wrapper_dir), "--source_label", "wrapper_smoke"])

        assert exit_code == 0
        manifest_path = rec_dir / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source"]["source_label"] == "wrapper_smoke"
        assert manifest["source"]["run_dir"] == str(run_dir)


def test_tf2_model_smoke_and_blob_roundtrip() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed, tf

    set_seed(7)
    params = _default_params()
    layers = [5, 8, 1]
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=layers,
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    blob = _make_blob(layers)
    model.import_parameter_blob(blob, strict=True)
    t_batch, W_batch, Xi_batch = model.fetch_minibatch()
    loss, X, Y, Z = model.loss_function(t_batch, W_batch, Xi_batch, const_value=1.0)
    assert np.isfinite(float(loss.numpy()))
    assert X.shape == (4, 3, 4)
    assert Y.shape == (4, 3, 1)
    assert Z.shape == (4, 3, 4)

    exported = model.export_parameter_blob()
    model2 = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=layers,
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    model2.import_parameter_blob(exported, strict=True)
    X1, Y1, Z1 = model.predict(Xi_batch, t_batch, W_batch, const_value=1.0)
    X2, Y2, Z2 = model2.predict(Xi_batch, t_batch, W_batch, const_value=1.0)
    np.testing.assert_allclose(X1, X2, atol=1.0e-6)
    np.testing.assert_allclose(Y1, Y2, atol=1.0e-6)
    np.testing.assert_allclose(Z1, Z2, atol=1.0e-6)

    Z_np = quadratic_coupled_exact_z_np(Xi_batch, params)
    mu_np = quadratic_coupled_mu_np(Xi_batch, Z_np, params)
    model.const_tf.assign(np.float32(1.0))
    mu_tf = model.mu_tf(
        tf.zeros((Xi_batch.shape[0], 1), dtype=tf.float32),
        tf.convert_to_tensor(Xi_batch),
        tf.zeros((Xi_batch.shape[0], 1), dtype=tf.float32),
        tf.convert_to_tensor(Z_np),
    ).numpy()
    np.testing.assert_allclose(mu_tf, mu_np, rtol=1.0e-5, atol=1.0e-5)

    stats = model.train(
        N_Iter=1,
        learning_rate=1.0e-3,
        const_value=1.0,
        eval_every=1,
        val_batches=1,
    )
    assert stats["best_iter"] == 0
    assert np.isfinite(stats["best_score"])

    model.close()
    model2.close()


def test_model_train_rejects_nonfinite_loss() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed, tf

    set_seed(23)
    params = _default_params()
    layers = [5, 8, 1]
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=layers,
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )

    def nonfinite_loss_function(
        t,
        W,
        Xi,
        const_value=None,
        return_components=False,
        return_runtime_diagnostics=False,
    ):
        del W, const_value
        batch_size = tf.shape(Xi)[0]
        n_steps = tf.shape(t)[1]
        loss = tf.reduce_sum(model.trainable_variables[0] * 0.0) + tf.constant(float("nan"), dtype=tf.float32)
        X = tf.zeros((batch_size, n_steps, 4), dtype=tf.float32)
        Y = tf.zeros((batch_size, n_steps, 1), dtype=tf.float32)
        Z = tf.zeros((batch_size, n_steps, 4), dtype=tf.float32)
        components = {"loss_total": loss}
        diagnostics = {}
        if return_components and return_runtime_diagnostics:
            return loss, X, Y, Z, components, diagnostics
        if return_components:
            return loss, X, Y, Z, components
        if return_runtime_diagnostics:
            return loss, X, Y, Z, diagnostics
        return loss, X, Y, Z

    original_loss_function = model.loss_function
    model.loss_function = nonfinite_loss_function
    try:
        try:
            model.train(
                N_Iter=2,
                learning_rate=1.0e-3,
                const_value=1.0,
                eval_every=1,
                val_batches=1,
            )
        except RuntimeError as exc:
            assert "non-finite" in str(exc).lower()
        else:
            raise AssertionError("training should reject a non-finite loss")
    finally:
        model.loss_function = original_loss_function
        model.close()


def test_model_train_reports_gradient_norm_diagnostics() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(31)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )

    try:
        stats = model.train(
            N_Iter=2,
            learning_rate=1.0e-3,
            const_value=1.0,
            eval_every=1,
            val_batches=1,
            return_gradient_diagnostics=True,
        )
        for key in (
            "train_grad_global_norm",
            "train_grad_global_norm_clipped",
            "train_grad_clip_ratio",
            "train_grad_clip_norm",
            "train_grad_clip_enabled",
            "train_grad_global_norm_max",
            "train_grad_global_norm_clipped_max",
            "train_grad_clip_ratio_max",
        ):
            assert key in stats, f"{key} missing from train diagnostics"
            assert np.isfinite(float(stats[key])), f"{key} must be finite"
        assert stats["train_grad_global_norm"] >= 0.0
        assert stats["train_grad_global_norm_clipped"] >= 0.0
        assert stats["train_grad_global_norm_clipped"] <= stats["train_grad_global_norm"] + 1.0e-6
        assert stats["train_grad_clip_ratio"] >= 0.0
        assert stats["train_grad_clip_enabled"] == 1.0
        assert stats["train_grad_global_norm_max"] >= stats["train_grad_global_norm"]
        assert stats["train_grad_global_norm_clipped_max"] >= stats["train_grad_global_norm_clipped"]
        assert stats["train_grad_clip_ratio_max"] >= stats["train_grad_clip_ratio"]
    finally:
        model.close()


def test_model_train_gradient_diagnostics_marks_disabled_clipping() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(32)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    model.clip_grad_norm = None

    try:
        stats = model.train(
            N_Iter=1,
            learning_rate=1.0e-3,
            const_value=1.0,
            eval_every=1,
            val_batches=1,
            return_gradient_diagnostics=True,
        )
        assert stats["train_grad_clip_enabled"] == 0.0
        assert stats["train_grad_clip_norm"] == 0.0
        assert stats["train_grad_clip_ratio"] == 1.0
        np.testing.assert_allclose(
            stats["train_grad_global_norm_clipped"],
            stats["train_grad_global_norm"],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    finally:
        model.close()


def test_model_train_uses_fixed_evaluation_batches_for_best_snapshot() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(34)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    fixed_batches = [
        (
            np.linspace(0.0, 0.25, 3, dtype=np.float32).reshape(1, 3, 1).repeat(4, axis=0),
            np.zeros((4, 3, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )
    ]
    eval_calls = []
    eval_scores = [0.75, 0.25]

    def fixed_evaluate(
        const_value=None,
        n_batches=5,
        evaluation_batches=None,
        moment_policy="batch_current",
    ):
        eval_calls.append(
            {
                "n_batches": n_batches,
                "evaluation_batches": evaluation_batches,
                "moment_policy": moment_policy,
            }
        )
        score = eval_scores[len(eval_calls) - 1]
        return {
            "mean_loss": score,
            "std_loss": 0.0,
            "mean_loss_per_sample": score / 4.0,
            "std_loss_per_sample": 0.0,
            "mean_y0": 0.0,
            "std_y0": 0.0,
        }

    model.evaluate = fixed_evaluate

    try:
        stats = model.train(
            N_Iter=2,
            learning_rate=1.0e-3,
            const_value=1.0,
            eval_every=1,
            val_batches=7,
            early_stopping_metric="loss",
            restore_best=True,
            evaluation_batches=fixed_batches,
            moment_policy="fixed_evaluation_bundle",
        )
        assert stats["best_iter"] == 1
        assert stats["best_score"] == 0.25
        assert len(eval_calls) == 2
        for call in eval_calls:
            assert call["n_batches"] == len(fixed_batches)
            assert call["moment_policy"] == "fixed_evaluation_bundle"
            assert len(call["evaluation_batches"]) == len(fixed_batches)
            for actual_batch, expected_batch in zip(call["evaluation_batches"], fixed_batches):
                for actual, expected in zip(actual_batch, expected_batch):
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    finally:
        model.close()


def test_model_train_rejects_malformed_fixed_evaluation_batches() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(35)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    t_batch = np.linspace(0.0, 0.25, 3, dtype=np.float32).reshape(1, 3, 1).repeat(4, axis=0)
    W_batch = np.zeros((4, 3, 4), dtype=np.float32)
    Xi_batch = np.zeros((4, 4), dtype=np.float32)

    def assert_rejected(evaluation_batches, expected):
        try:
            model.train(
                N_Iter=1,
                learning_rate=1.0e-3,
                const_value=1.0,
                eval_every=1,
                evaluation_batches=evaluation_batches,
                moment_policy="fixed_evaluation_bundle",
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"train should reject malformed evaluation_batches: {expected}")

    try:
        assert_rejected([(t_batch, W_batch)], "triple")
        assert_rejected([(t_batch, W_batch[:, :1, :], Xi_batch)], "time dimension")
        assert_rejected([(t_batch, W_batch, Xi_batch[:, :3])], "state dimension")
        assert_rejected([(t_batch[:2], W_batch[:2], Xi_batch[:2])], "batch size")
        bad_t = t_batch.copy()
        bad_t[:, 1, 0] = bad_t[:, 0, 0]
        assert_rejected([(bad_t, W_batch, Xi_batch)], "strictly increasing")
    finally:
        model.close()


def test_model_train_rejects_wrong_time_horizon_fixed_evaluation_batches() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(37)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    Xi_batch = np.zeros((4, 4), dtype=np.float32)

    def assert_rejected(time_points: int) -> None:
        t_batch = (
            np.linspace(0.0, 0.25, time_points, dtype=np.float32)
            .reshape(1, time_points, 1)
            .repeat(4, axis=0)
        )
        W_batch = np.zeros((4, time_points, 4), dtype=np.float32)
        try:
            model.train(
                N_Iter=1,
                learning_rate=1.0e-3,
                const_value=1.0,
                eval_every=1,
                evaluation_batches=[(t_batch, W_batch, Xi_batch)],
                moment_policy="fixed_evaluation_bundle",
            )
        except ValueError as exc:
            assert "N+1=3" in str(exc)
        else:
            raise AssertionError("train should reject a fixed evaluation bundle with the wrong time horizon")

    try:
        assert_rejected(2)
        assert_rejected(4)
    finally:
        model.close()


def test_model_train_rejects_wrong_time_horizon_before_update() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(38)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    t_batch = np.linspace(0.0, 0.25, 4, dtype=np.float32).reshape(1, 4, 1).repeat(4, axis=0)
    W_batch = np.zeros((4, 4, 4), dtype=np.float32)
    Xi_batch = np.zeros((4, 4), dtype=np.float32)
    loss_calls = {"count": 0}
    original_loss_function = model.loss_function

    def fail_if_training_starts(*args, **kwargs):
        loss_calls["count"] += 1
        raise AssertionError("fixed evaluation validation should fail before the training loss is evaluated")

    model.loss_function = fail_if_training_starts
    try:
        try:
            model.train(
                N_Iter=1,
                learning_rate=1.0e-3,
                const_value=1.0,
                eval_every=1,
                evaluation_batches=[(t_batch, W_batch, Xi_batch)],
                moment_policy="fixed_evaluation_bundle",
                return_gradient_diagnostics=True,
            )
        except ValueError as exc:
            assert "N+1=3" in str(exc)
        else:
            raise AssertionError("train should fail fast on fixed evaluation time-horizon mismatch")
        assert loss_calls["count"] == 0
    finally:
        model.loss_function = original_loss_function
        model.close()


def test_model_spec_recursive_factory_matches_direct_constructor() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import reset_backend_state, set_seed

    spec = get_model_spec()
    params = _default_params()
    layers = [5, 8, 1]
    kwargs = {
        "Xi_generator": Xi_generator_default,
        "T": 0.25,
        "M": 4,
        "N": 2,
        "D": 4,
        "layers": layers,
        "parameters": params,
        "t_start": 0.0,
        "t_end": 0.25,
        "T_total": 0.25,
        "terminal_blob": None,
        "normalize_time_input": True,
    }

    reset_backend_state()
    set_seed(41)
    direct_initial_model = NN_Quadratic_Coupled_Recursive(**kwargs)
    direct_initial = direct_initial_model.export_parameter_blob()
    direct_initial_model.close()

    reset_backend_state()
    set_seed(41)
    spec_initial_model = spec.build_recursive_model(**kwargs)
    spec_initial = spec_initial_model.export_parameter_blob()
    spec_initial_model.close()

    for key in direct_initial:
        if key.startswith(("W_", "b_")):
            np.testing.assert_allclose(spec_initial[key], direct_initial[key], atol=1.0e-7)

    blob = _make_blob(layers)
    reset_backend_state()
    set_seed(43)
    direct = NN_Quadratic_Coupled_Recursive(**kwargs)
    via_spec = spec.build_recursive_model(**kwargs)
    direct.import_parameter_blob(blob, strict=True)
    via_spec.import_parameter_blob(blob, strict=True)
    t_batch, W_batch, Xi_batch = direct.fetch_minibatch()

    loss_direct, X_direct, Y_direct, Z_direct = direct.loss_function(
        t_batch,
        W_batch,
        Xi_batch,
        const_value=1.0,
    )
    loss_spec, X_spec, Y_spec, Z_spec = via_spec.loss_function(
        t_batch,
        W_batch,
        Xi_batch,
        const_value=1.0,
    )
    np.testing.assert_allclose(loss_spec.numpy(), loss_direct.numpy(), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(X_spec.numpy(), X_direct.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(Y_spec.numpy(), Y_direct.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(Z_spec.numpy(), Z_direct.numpy(), atol=1.0e-6)
    direct.close()
    via_spec.close()


def test_pascucci_recursive_factory_matches_direct_constructor() -> None:
    from .model_specs import get_model_spec
    from .models import NN_Pascucci_Recursive
    from .tf_backend import reset_backend_state, set_seed

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    layers = [5, 8, 1]
    kwargs = {
        "Xi_generator": Xi_generator_default,
        "T": 0.25,
        "M": 4,
        "N": 2,
        "D": 4,
        "layers": layers,
        "parameters": params,
        "t_start": 0.0,
        "t_end": 0.25,
        "T_total": 0.25,
        "terminal_blob": None,
        "normalize_time_input": True,
    }

    reset_backend_state()
    set_seed(51)
    direct_initial_model = NN_Pascucci_Recursive(**kwargs)
    direct_initial = direct_initial_model.export_parameter_blob()
    direct_initial_model.close()

    reset_backend_state()
    set_seed(51)
    spec_initial_model = spec.build_recursive_model(**kwargs)
    spec_initial = spec_initial_model.export_parameter_blob()
    spec_initial_model.close()

    for key in direct_initial:
        if key.startswith(("W_", "b_")):
            np.testing.assert_allclose(spec_initial[key], direct_initial[key], atol=1.0e-7)

    blob = _make_blob(layers)
    reset_backend_state()
    set_seed(53)
    direct = NN_Pascucci_Recursive(**kwargs)
    via_spec = spec.build_recursive_model(**kwargs)
    direct.import_parameter_blob(blob, strict=True)
    via_spec.import_parameter_blob(blob, strict=True)
    t_batch, W_batch, Xi_batch = direct.fetch_minibatch()

    loss_direct, X_direct, Y_direct, Z_direct = direct.loss_function(
        t_batch,
        W_batch,
        Xi_batch,
        const_value=0.75,
    )
    loss_spec, X_spec, Y_spec, Z_spec = via_spec.loss_function(
        t_batch,
        W_batch,
        Xi_batch,
        const_value=0.75,
    )
    np.testing.assert_allclose(loss_spec.numpy(), loss_direct.numpy(), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(X_spec.numpy(), X_direct.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(Y_spec.numpy(), Y_direct.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(Z_spec.numpy(), Z_direct.numpy(), atol=1.0e-6)
    assert X_spec.dtype.name == "float32"
    assert Y_spec.dtype.name == "float32"
    assert Z_spec.dtype.name == "float32"
    assert np.isfinite(loss_spec.numpy())
    direct.close()
    via_spec.close()


def test_predict_recursive_stitched_two_block_model_spec() -> None:
    from .model_specs import get_model_spec
    from .orchestration import predict_recursive_stitched
    from .tf_backend import set_seed

    set_seed(31)
    spec = get_model_spec()
    params = _default_params()
    layers = [5, 8, 1]
    D = 4
    M = 4
    N_per_block = 2
    T_total = 0.5
    blocks = build_blocks(T_total=T_total, block_size=0.25)
    blobs = [
        _make_block_blob(layers, block["t_start"], block["t_end"], T_total)
        for block in blocks
    ]
    Xi = spec.deterministic_xi(M, D, seed=111)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=M,
        N_per_block=N_per_block,
        D=D,
        seed=321,
    )

    explicit = predict_recursive_stitched(
        block_blobs=blobs,
        blocks=blocks,
        Xi_initial=Xi,
        params=params,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=rollout,
        model_spec=spec,
    )
    default = predict_recursive_stitched(
        block_blobs=blobs,
        blocks=blocks,
        Xi_initial=Xi,
        params=params,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=rollout,
        model_spec=None,
    )

    for key in ("t", "X", "Y", "Z"):
        np.testing.assert_allclose(explicit[key], default[key], atol=1.0e-6)
    assert explicit["t"].shape == (M, 5, 1)
    assert explicit["X"].shape == (M, 5, D)
    assert explicit["Y"].shape == (M, 5, 1)
    assert explicit["Z"].shape == (M, 5, D)
    np.testing.assert_allclose(
        explicit["t"][0, :, 0],
        np.asarray([0.0, 0.125, 0.25, 0.375, 0.5], dtype=np.float32),
        atol=1.0e-7,
    )

    first = predict_recursive_stitched(
        block_blobs=[blobs[0]],
        blocks=[blocks[0]],
        Xi_initial=Xi,
        params=params,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=[rollout[0]],
        model_spec=spec,
    )
    second = predict_recursive_stitched(
        block_blobs=[blobs[1]],
        blocks=[blocks[1]],
        Xi_initial=first["X"][:, -1, :],
        params=params,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=[rollout[1]],
        model_spec=spec,
    )
    np.testing.assert_allclose(explicit["X"][:, :3, :], first["X"], atol=1.0e-6)
    np.testing.assert_allclose(second["X"][:, 0, :], first["X"][:, -1, :], atol=1.0e-6)
    np.testing.assert_allclose(explicit["X"][:, 3:, :], second["X"][:, 1:, :], atol=1.0e-6)


def test_prefixed_eval_diagnostics_forwards_scalar_model_moments_only() -> None:
    from .orchestration import _prefixed_eval_diagnostics

    diagnostics = _prefixed_eval_diagnostics(
        {
            "mean_loss": 1.0,
            "std_loss": 0.0,
            "mean_loss_dynamic": 0.25,
            "std_loss_dynamic": 0.01,
            "mean_block_end_mean_v": np.float32(1.25),
            "std_block_end_mean_v": np.float32(0.05),
            "mean_block_end_q_lower_violation": np.float32(0.10),
            "std_block_end_q_lower_violation": np.float32(0.02),
            "mean_trace_mean_v": np.asarray([1.0, 2.0], dtype=np.float32),
            "mean_unrelated_scalar": np.float32(0.33),
            "std_bad": np.float32(np.inf),
            "mean_bad": "not-a-number",
            "terminal_mean_v": np.float32(9.0),
        }
    )

    np.testing.assert_allclose(diagnostics["eval_mean_loss_dynamic"], 0.25)
    np.testing.assert_allclose(diagnostics["eval_std_loss_dynamic"], 0.01)
    np.testing.assert_allclose(diagnostics["eval_mean_block_end_mean_v"], 1.25)
    np.testing.assert_allclose(diagnostics["eval_std_block_end_mean_v"], 0.05)
    np.testing.assert_allclose(diagnostics["eval_mean_block_end_q_lower_violation"], 0.10)
    np.testing.assert_allclose(diagnostics["eval_std_block_end_q_lower_violation"], 0.02)
    assert "eval_mean_loss" not in diagnostics
    assert "eval_std_loss" not in diagnostics
    assert "eval_mean_trace_mean_v" not in diagnostics
    assert "eval_mean_unrelated_scalar" not in diagnostics
    assert "eval_std_bad" not in diagnostics
    assert "eval_mean_bad" not in diagnostics
    assert "eval_terminal_mean_v" not in diagnostics


def test_predict_recursive_stitched_pascucci_moment_traces() -> None:
    from .model_specs import get_model_spec
    from .orchestration import predict_recursive_stitched
    from .tf_backend import set_seed

    set_seed(83)
    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    layers = [5, 8, 1]
    D = 4
    M = 4
    N_per_block = 2
    T_total = 0.5
    blocks = build_blocks(T_total=T_total, block_size=0.25)
    blobs = [
        _make_block_blob(layers, block["t_start"], block["t_end"], T_total)
        for block in blocks
    ]
    Xi = spec.deterministic_xi(M, D, seed=83)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=M,
        N_per_block=N_per_block,
        D=D,
        seed=84,
    )

    stitched = predict_recursive_stitched(
        block_blobs=blobs,
        blocks=blocks,
        Xi_initial=Xi,
        params=params,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=rollout,
        coupling_const=0.75,
        model_spec=spec,
    )

    expected = {
        "mean_v": np.mean(stitched["X"][:, :, [2]], axis=0),
        "mean_q": np.mean(stitched["X"][:, :, [3]], axis=0),
        "mean_h_plus_v": np.mean(
            stitched["X"][:, :, [1]] + stitched["X"][:, :, [2]],
            axis=0,
        ),
    }
    expected.update(_pascucci_physical_violation_traces_from_x(stitched["X"], params))
    for key, expected_trace in expected.items():
        assert key in stitched, f"{key} missing from stitched keys: {sorted(stitched)}"
        assert stitched[key].shape == (len(blocks) * N_per_block + 1, 1)
        assert stitched[key].dtype == np.float32
        assert np.isfinite(stitched[key]).all(), key
        np.testing.assert_allclose(stitched[key], expected_trace, rtol=1.0e-6, atol=1.0e-6)

    quadratic_spec = get_model_spec("quadratic_coupled")
    quadratic = predict_recursive_stitched(
        block_blobs=blobs,
        blocks=blocks,
        Xi_initial=quadratic_spec.deterministic_xi(M, D, seed=83),
        params=quadratic_spec.build_default_params(const=1.0),
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
        rollout_inputs=rollout,
        model_spec=quadratic_spec,
    )
    assert set(quadratic) == {"t", "X", "Y", "Z"}
    for key in expected:
        assert key not in quadratic


def test_predict_recursive_stitched_carries_model_owned_diagnostics_without_recomputing() -> None:
    from .orchestration import predict_recursive_stitched

    class FakeSession:
        def close(self) -> None:
            pass

    class FakeModel:
        def __init__(self, block_idx: int, D: int) -> None:
            self.block_idx = int(block_idx)
            self.D = int(D)
            self.sess = FakeSession()

        def import_parameter_blob(self, blob, strict=True) -> None:
            del blob, strict

        def predict(self, Xi_star, t_star, W_star, const_value=None, return_runtime_diagnostics=False):
            del W_star, const_value
            assert return_runtime_diagnostics is True, (
                "predict_recursive_stitched must request model-owned runtime diagnostics"
            )
            M = int(Xi_star.shape[0])
            steps = int(t_star.shape[1])
            X = np.tile(Xi_star[:, None, :], (1, steps, 1)).astype(np.float32)
            if self.block_idx == 0:
                X[:, -1, :] = np.float32(2.0)
            else:
                X[:, 0, :] = X[:, 0, :] + np.float32(0.5)
            Y = np.zeros((M, steps, 1), dtype=np.float32)
            Z = np.zeros((M, steps, self.D), dtype=np.float32)
            start = np.float32(100 + self.block_idx * 2)
            diagnostics = {
                "mean_v": np.asarray([[start], [start + 1.0], [start + 2.0]], dtype=np.float32),
                "mean_q": np.asarray([[start + 10.0], [start + 11.0], [start + 12.0]], dtype=np.float32),
                "mean_h_plus_v": np.asarray([[start + 20.0], [start + 21.0], [start + 22.0]], dtype=np.float32),
            }
            return X, Y, Z, diagnostics

    class FakeSpec:
        name = "sentinel_model"
        application_metric_schema = "sentinel_diagnostics_v1"

        def __init__(self) -> None:
            self.calls = 0

        def validate_state_dim(self, D: int) -> None:
            assert int(D) == 4

        def build_recursive_model(self, **kwargs):
            model = FakeModel(block_idx=self.calls, D=kwargs["D"])
            self.calls += 1
            return model

    D = 4
    M = 4
    N_per_block = 2
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=M,
        N_per_block=N_per_block,
        D=D,
        seed=91,
    )
    stitched = predict_recursive_stitched(
        block_blobs=[{}, {}],
        blocks=blocks,
        Xi_initial=np.zeros((M, D), dtype=np.float32),
        params={},
        N_per_block=N_per_block,
        D=D,
        layers=[5, 8, 1],
        T_total=0.5,
        rollout_inputs=rollout,
        model_spec=FakeSpec(),
    )

    np.testing.assert_allclose(
        stitched["mean_v"],
        np.asarray([[100.0], [101.0], [102.0], [103.0], [104.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        stitched["mean_q"],
        np.asarray([[110.0], [111.0], [112.0], [113.0], [114.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        stitched["mean_h_plus_v"],
        np.asarray([[120.0], [121.0], [122.0], [123.0], [124.0]], dtype=np.float32),
    )
    assert "stitch_X_boundary_abs_jump" in stitched
    assert stitched["stitch_X_boundary_abs_jump"].shape == (1, M, D)
    np.testing.assert_allclose(stitched["stitch_X_boundary_abs_jump"], 0.5)


def test_predict_recursive_stitched_schema_none_diagnostics_do_not_change_boundary_keyset() -> None:
    from .orchestration import predict_recursive_stitched

    class FakeSession:
        def close(self) -> None:
            pass

    class DiagnosticOnlyModel:
        def __init__(self, block_idx: int, D: int) -> None:
            self.block_idx = int(block_idx)
            self.D = int(D)
            self.sess = FakeSession()

        def import_parameter_blob(self, blob, strict=True) -> None:
            del blob, strict

        def predict(self, Xi_star, t_star, W_star, const_value=None, return_runtime_diagnostics=False):
            del W_star, const_value
            assert return_runtime_diagnostics is True
            M = int(Xi_star.shape[0])
            steps = int(t_star.shape[1])
            X = np.tile(Xi_star[:, None, :], (1, steps, 1)).astype(np.float32)
            X[:, -1, :] = np.float32(self.block_idx + 1)
            Y = np.zeros((M, steps, 1), dtype=np.float32)
            Z = np.zeros((M, steps, self.D), dtype=np.float32)
            diagnostics = {
                "mean_probe": np.arange(steps, dtype=np.float32).reshape(steps, 1)
            }
            return X, Y, Z, diagnostics

    class SchemaNoneSpec:
        name = "schema_none_with_diagnostics"
        application_metric_schema = "none"

        def __init__(self) -> None:
            self.calls = 0

        def validate_state_dim(self, D: int) -> None:
            assert int(D) == 4

        def build_recursive_model(self, **kwargs):
            model = DiagnosticOnlyModel(block_idx=self.calls, D=kwargs["D"])
            self.calls += 1
            return model

    D = 4
    M = 4
    N_per_block = 2
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=M,
        N_per_block=N_per_block,
        D=D,
        seed=94,
    )
    spec = SchemaNoneSpec()
    stitched = predict_recursive_stitched(
        block_blobs=[{}, {}],
        blocks=blocks,
        Xi_initial=np.zeros((M, D), dtype=np.float32),
        params={},
        N_per_block=N_per_block,
        D=D,
        layers=[5, 8, 1],
        T_total=0.5,
        rollout_inputs=rollout,
        model_spec=spec,
    )

    assert spec.calls == 2
    assert set(stitched) == {"t", "X", "Y", "Z", "mean_probe"}
    np.testing.assert_allclose(
        stitched["mean_probe"],
        np.asarray([[0.0], [1.0], [2.0], [1.0], [2.0]], dtype=np.float32),
    )


def test_predict_recursive_stitched_accepts_legacy_three_value_predict() -> None:
    from .orchestration import predict_recursive_stitched

    class FakeSession:
        def close(self) -> None:
            pass

    class LegacyModel:
        def __init__(self, block_idx: int, D: int) -> None:
            self.block_idx = int(block_idx)
            self.D = int(D)
            self.sess = FakeSession()

        def import_parameter_blob(self, blob, strict=True) -> None:
            del blob, strict

        def predict(self, Xi_star, t_star, W_star, const_value=None):
            del W_star
            assert np.isclose(float(const_value), 1.0)
            M = int(Xi_star.shape[0])
            steps = int(t_star.shape[1])
            X = np.tile(Xi_star[:, None, :], (1, steps, 1)).astype(np.float32)
            X = X + np.float32(self.block_idx)
            Y = np.zeros((M, steps, 1), dtype=np.float32)
            Z = np.zeros((M, steps, self.D), dtype=np.float32)
            return X, Y, Z

    class LegacySpec:
        name = "legacy_sentinel"
        application_metric_schema = "legacy_boundary_diagnostics_v1"

        def __init__(self) -> None:
            self.calls = 0

        def validate_state_dim(self, D: int) -> None:
            assert int(D) == 4

        def build_recursive_model(self, **kwargs):
            model = LegacyModel(block_idx=self.calls, D=kwargs["D"])
            self.calls += 1
            return model

    D = 4
    M = 4
    N_per_block = 2
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=M,
        N_per_block=N_per_block,
        D=D,
        seed=93,
    )
    spec = LegacySpec()
    stitched = predict_recursive_stitched(
        block_blobs=[{}, {}],
        blocks=blocks,
        Xi_initial=np.zeros((M, D), dtype=np.float32),
        params={},
        N_per_block=N_per_block,
        D=D,
        layers=[5, 8, 1],
        T_total=0.5,
        rollout_inputs=rollout,
        model_spec=spec,
    )

    assert spec.calls == 2
    assert set(stitched) == {
        "t",
        "X",
        "Y",
        "Z",
        "stitch_X_boundary_abs_jump",
        "stitch_Y_boundary_abs_jump",
        "stitch_Z_boundary_abs_jump",
    }
    assert stitched["t"].shape == (M, 5, 1)
    assert stitched["X"].shape == (M, 5, D)
    assert stitched["Y"].shape == (M, 5, 1)
    assert stitched["Z"].shape == (M, 5, D)
    assert stitched["stitch_X_boundary_abs_jump"].shape == (1, M, D)
    assert stitched["stitch_Y_boundary_abs_jump"].shape == (1, M, 1)
    assert stitched["stitch_Z_boundary_abs_jump"].shape == (1, M, D)
    np.testing.assert_allclose(stitched["stitch_X_boundary_abs_jump"], 1.0)
    np.testing.assert_allclose(stitched["stitch_Y_boundary_abs_jump"], 0.0)
    np.testing.assert_allclose(stitched["stitch_Z_boundary_abs_jump"], 0.0)


def test_print_recursive_pass_saves_stitched_moment_traces() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    layers = [5, 8, 1]
    D = 4
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    logs = [
        {
            "pass": 1,
            "block": 0,
            "t_start": 0.0,
            "t_end": 0.25,
            "T_block": 0.25,
            "eval_mean_loss": 1.0,
            "eval_std_loss": 0.0,
            "eval_mean_loss_per_sample": 0.25,
            "eval_std_loss_per_sample": 0.0,
            "eval_mean_y0": 0.0,
            "precision_target": None,
            "refine_rounds": 0,
        }
    ]

    def fake_predict_recursive_stitched(**kwargs):
        M = int(kwargs["Xi_initial"].shape[0])
        t = np.tile(
            np.asarray([0.0, 0.125, 0.25], dtype=np.float32).reshape(1, 3, 1),
            (M, 1, 1),
        )
        X = np.zeros((M, 3, D), dtype=np.float32)
        X[:, :, 1] = np.asarray([0.5, 0.25, 0.0], dtype=np.float32)
        X[:, :, 2] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        X[:, :, 3] = np.asarray([4.0, 4.5, 5.0], dtype=np.float32)
        return {
            "t": t,
            "X": X,
            "Y": np.zeros((M, 3, 1), dtype=np.float32),
            "Z": np.zeros((M, 3, D), dtype=np.float32),
            "mean_v": np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
            "mean_q": np.asarray([[4.0], [4.5], [5.0]], dtype=np.float32),
            "mean_h_plus_v": np.asarray([[0.6], [0.45], [0.3]], dtype=np.float32),
            "q_lower_violation": np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32),
            "q_upper_violation": np.asarray([[0.0], [0.0], [0.3]], dtype=np.float32),
            "v_lower_violation": np.asarray([[0.4], [0.0], [0.0]], dtype=np.float32),
            "v_upper_violation": np.asarray([[0.0], [0.5], [0.0]], dtype=np.float32),
        }

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_stitched = orchestration.plot_recursive_stitched_predictions
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence
    orchestration.predict_recursive_stitched = fake_predict_recursive_stitched
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_predictions = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp)
            orchestration.print_recursive_pass(
                pass_entries=[{"pass_id": 1, "logs": logs, "blobs": [_make_blob(layers)]}],
                blocks=blocks,
                rec_dir=str(rec_dir),
                params=params,
                N_per_block=2,
                D=D,
                layers=layers,
                T_total=0.25,
                exact_solution=None,
                selection_metric="loss",
                eval_bundle_path=str(rec_dir / "evaluation_bundle.npz"),
                eval_seed=101,
                eval_min_paths=4,
                sample_paths=0,
                print_compact_logs=False,
                model_spec=spec,
            )
            for filename in (
                "stitched_predictions_pass00.npz",
                "stitched_predictions_final.npz",
            ):
                path = rec_dir / filename
                assert path.exists(), filename
                with np.load(path) as data:
                    keys = set(data.files)
                    assert {"t", "X", "Y", "Z"}.issubset(keys)
                    for key in (
                        "mean_v",
                        "mean_q",
                        "mean_h_plus_v",
                        "q_lower_violation",
                        "q_upper_violation",
                        "v_lower_violation",
                        "v_upper_violation",
                    ):
                        assert key in keys, f"{key} missing from {filename}: {sorted(keys)}"
                        assert data[key].shape == (3, 1)
                        assert data[key].dtype == np.float32
                        assert np.isfinite(data[key]).all(), key
                    assert "block_params" not in keys
                    assert not any(key.startswith(("params_", "weights_")) for key in keys)
            for filename in ("application_metrics_pass00.json", "application_metrics_final.json"):
                payload = json.loads((rec_dir / filename).read_text(encoding="utf-8"))
                assert payload["schema"] == "pascucci_application_metrics_v2"
                assert payload["model_name"] == "pascucci"
                assert payload["controlled"]["metadata"]["baseline_mode"] == "controlled"
                assert payload["uncontrolled"]["metadata"]["baseline_mode"] == "uncontrolled"
                for metric_name in (
                    "cost_J_running_mean",
                    "cost_J_terminal_mean",
                    "cost_J_total_mean",
                ):
                    assert metric_name in payload["controlled"]["summary"]
                    assert np.isfinite(payload["controlled"]["summary"][metric_name])
                assert "stitch_X_boundary_max_abs_jump" in payload["diagnostics"]
                assert Path(payload["pathwise_npz_path"]).exists()
            for filename in ("application_metrics_pass00.npz", "application_metrics_final.npz"):
                with np.load(rec_dir / filename) as data:
                    keys = set(data.files)
                    assert {
                        "controlled_cost_J_running",
                        "controlled_cost_J_terminal",
                        "controlled_cost_J_total",
                        "controlled_cost_J_running_cumulative",
                        "controlled_cost_J_trajectory",
                        "controlled_cost_J_trajectory_math",
                        "controlled_alpha",
                        "uncontrolled_cost_J_running",
                        "uncontrolled_cost_J_terminal",
                        "uncontrolled_cost_J_total",
                        "uncontrolled_cost_J_running_cumulative",
                        "uncontrolled_cost_J_trajectory",
                        "uncontrolled_cost_J_trajectory_math",
                        "uncontrolled_alpha",
                    }.issubset(keys)
                    assert data["controlled_cost_J_total"].shape == (4, 1)
                    assert data["uncontrolled_cost_J_total"].shape == (4, 1)
                    assert data["controlled_cost_J_running_cumulative"].shape == (4, 2, 1)
                    assert data["uncontrolled_cost_J_running_cumulative"].shape == (4, 2, 1)
                    assert data["controlled_cost_J_trajectory"].shape == (4, 3, 1)
                    assert data["uncontrolled_cost_J_trajectory"].shape == (4, 3, 1)
                    assert data["controlled_cost_J_trajectory_math"].shape == (4, 3, 1)
                    assert data["uncontrolled_cost_J_trajectory_math"].shape == (4, 3, 1)
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_print_recursive_pass_application_metrics_emit_comparison_and_stability() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("pascucci")
    M = 4
    D = 4
    layers = [5, 8, 1]
    z_v_index = 1

    class FakeApplicationModel:
        def _summary(self, running, terminal, total, running_cumulative, trajectory, trajectory_math):
            summary = {}
            for key, values in (
                ("cost_J_running", running),
                ("cost_J_terminal", terminal),
                ("cost_J_total", total),
                ("cost_J_running_cumulative", running_cumulative),
                ("cost_J_trajectory", trajectory),
                ("cost_J_trajectory_math", trajectory_math),
            ):
                values_np = np.asarray(values, dtype=np.float32)
                if key in ("cost_J_running_cumulative", "cost_J_trajectory", "cost_J_trajectory_math"):
                    flat = values_np[:, -1, :].reshape(-1) if values_np.shape[1] > 0 else np.zeros((values_np.shape[0],), dtype=np.float32)
                else:
                    flat = values_np.reshape(-1)
                summary[f"{key}_mean"] = float(np.mean(flat))
                summary[f"{key}_std"] = float(np.std(flat))
                summary[f"{key}_q05"] = float(np.quantile(flat, 0.05))
                summary[f"{key}_q50"] = float(np.quantile(flat, 0.50))
                summary[f"{key}_q95"] = float(np.quantile(flat, 0.95))
            return summary

        def _result(self, *, t, cost_seed, alpha, baseline_mode, control_law, paired_inputs):
            M_local = int(alpha.shape[0])
            steps = int(alpha.shape[1])
            running = np.full((M_local, 1), np.float32(cost_seed), dtype=np.float32)
            if steps > 0:
                running_step = np.full((M_local, steps, 1), np.float32(cost_seed / steps), dtype=np.float32)
                running_cumulative = np.cumsum(running_step, axis=1).astype(np.float32)
            else:
                running_cumulative = np.zeros((M_local, 0, 1), dtype=np.float32)
            terminal = np.full((M_local, 1), np.float32(0.25), dtype=np.float32)
            total = running + terminal
            trajectory = (
                np.concatenate([np.zeros((M_local, 1, 1), dtype=np.float32), running_cumulative], axis=1)
                + terminal[:, None, :]
            ).astype(np.float32)
            trajectory_math = trajectory.copy()
            return {
                "schema": "pascucci_application_metrics_v2",
                "metadata": {
                    "baseline_mode": baseline_mode,
                    "aggregation": "left_riemann_f_plus_terminal_g",
                    "trajectory_aggregation": "plotmaker_cumsum_f_dt_plus_g_at_time",
                    "trajectory_math_aggregation": "left_riemann_integral_to_time_plus_g_at_time",
                    "control_law": control_law,
                    "paired_inputs": paired_inputs,
                },
                "pathwise": {
                    "cost_J_running": running,
                    "cost_J_terminal": terminal,
                    "cost_J_total": total,
                    "cost_J_running_cumulative": running_cumulative,
                    "cost_J_trajectory": trajectory,
                    "cost_J_trajectory_math": trajectory_math,
                    "alpha": alpha.astype(np.float32),
                },
                "summary": self._summary(running, terminal, total, running_cumulative, trajectory, trajectory_math),
            }

        def application_cost_from_path(self, t, X, Y, Z, const_value=None, baseline_mode="controlled", **kwargs):
            del Y, const_value, kwargs
            cost_seed = float(np.asarray(X, dtype=np.float32)[0, 0, 0]) + 1.0
            alpha = np.asarray(Z, dtype=np.float32)[:, :-1, [z_v_index]]
            return self._result(
                t=t,
                cost_seed=cost_seed,
                alpha=alpha,
                baseline_mode=str(baseline_mode),
                control_law="alpha_tf",
                paired_inputs="stitched_XYZ",
            )

        def application_cost_functional(self, t, W, Xi, const_value=None, baseline_mode="uncontrolled"):
            del W, const_value
            steps = int(np.asarray(t).shape[1]) - 1
            alpha = np.zeros((int(np.asarray(Xi).shape[0]), steps, 1), dtype=np.float32)
            return self._result(
                t=t,
                cost_seed=3.0,
                alpha=alpha,
                baseline_mode=str(baseline_mode),
                control_law="alpha_zero",
                paired_inputs="same_t_W_Xi",
            )

        def close(self):
            pass

    spec = ModelSpec(
        name="pascucci_stability_stub",
        state_dim=4,
        state_labels=("S", "H", "V", "X_state"),
        z_labels=("Z_S", "Z_V", "Z_H", "Z_X"),
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: FakeApplicationModel(),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
        moment_names=base.moment_names,
        application_metric_schema="pascucci_application_metrics_v2",
        application_metric_names=base.application_metric_names,
        application_metric_aggregation=base.application_metric_aggregation,
    )
    params = spec.build_default_params(const=0.75)
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    pass_entries = []
    for pass_id, loss in ((1, 0.50), (2, 0.25)):
        pass_entries.append(
            {
                "pass_id": pass_id,
                "logs": [
                    {
                        "pass": pass_id,
                        "block": 0,
                        "t_start": 0.0,
                        "t_end": 0.25,
                        "T_block": 0.25,
                        "eval_mean_loss": float(loss),
                        "eval_std_loss": 0.0,
                        "eval_mean_loss_per_sample": float(loss),
                        "eval_std_loss_per_sample": 0.0,
                        "eval_mean_y0": 0.0,
                        "precision_target": None,
                        "refine_rounds": 0,
                    }
                ],
                "blobs": [_make_blob(layers)],
            }
        )

    predict_call = {"count": 0}

    def fake_predict_recursive_stitched(**kwargs):
        predict_call["count"] += 1
        rollout_inputs = kwargs["rollout_inputs"]
        t = np.asarray(rollout_inputs[0][0], dtype=np.float32)
        M_local = int(t.shape[0])
        pass_offset = np.float32(predict_call["count"] - 1)
        X = np.zeros((M_local, t.shape[1], D), dtype=np.float32)
        X[:, :, 0] = pass_offset
        X[:, :, 3] = 5.0
        Y = np.full((M_local, t.shape[1], 1), pass_offset, dtype=np.float32)
        Z = np.zeros((M_local, t.shape[1], D), dtype=np.float32)
        Z[:, :, z_v_index] = pass_offset * np.float32(2.0)
        return {"t": t, "X": X, "Y": Y, "Z": Z}

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_stitched = orchestration.plot_recursive_stitched_predictions
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence
    orchestration.predict_recursive_stitched = fake_predict_recursive_stitched
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_predictions = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp)
            summary = orchestration.print_recursive_pass(
                pass_entries=pass_entries,
                blocks=blocks,
                rec_dir=str(rec_dir),
                params=params,
                N_per_block=2,
                D=D,
                layers=layers,
                T_total=0.25,
                exact_solution=None,
                selection_metric="loss",
                eval_bundle_path=str(rec_dir / "evaluation_bundle.npz"),
                eval_seed=101,
                eval_min_paths=M,
                sample_paths=0,
                print_compact_logs=False,
                model_spec=spec,
                mc_confirmation_paths=6,
                mc_confirmation_seed=202,
            )

            assert summary["selected_pass_id"] == 2
            assert 2 in summary["application_stability_by_pass"]
            stability = summary["application_stability_by_pass"][2]
            np.testing.assert_allclose(stability["pass_vs_prev_Z_V_mae"], 2.0, rtol=1.0e-6, atol=1.0e-6)
            np.testing.assert_allclose(stability["pass_vs_prev_Y_mae"], 1.0, rtol=1.0e-6, atol=1.0e-6)
            assert summary["application_stability_by_pass_index"] == {"1": stability}

            pass0 = json.loads((rec_dir / "application_metrics_pass00.json").read_text(encoding="utf-8"))
            pass1 = json.loads((rec_dir / "application_metrics_pass01.json").read_text(encoding="utf-8"))
            for payload in (pass0, pass1):
                assert "comparison" in payload
                assert payload["comparison"]["paired_pathwise_samples"] is True
                assert payload["comparison"]["same_input_source"] is False
                assert "alpha_summary" in payload["controlled"]
            assert "stability_vs_previous_pass" not in pass0
            assert pass1["stability_vs_previous_pass"] == stability
            mc_summary = summary["mc_confirmation_summary"]
            assert mc_summary["seed"] == 202
            assert mc_summary["sample_paths"] == 6
            assert Path(mc_summary["summary_path"]).exists()
            assert Path(mc_summary["pathwise_npz_path"]).exists()
            assert Path(mc_summary["bundle_path"]).exists()
            mc_payload = json.loads(Path(mc_summary["summary_path"]).read_text(encoding="utf-8"))
            assert mc_payload["schema"] == "pascucci_mc_confirmation_v1"
            assert mc_payload["source"] == "independent_post_selection"
            assert mc_payload["selected_pass_id"] == 2
            assert mc_payload["seed"] == 202
            assert mc_payload["horizon"]["sample_paths"] == 6
            assert mc_payload["comparison"]["paired_sample_count"] == 6
            assert mc_payload["bundle_kind"] == "mc_confirmation"
            with np.load(mc_summary["pathwise_npz_path"], allow_pickle=False) as data:
                assert data["controlled_cost_J_total"].shape == (6, 1)
                assert data["uncontrolled_cost_J_total"].shape == (6, 1)
                assert data["q_lower_violation"].shape[0] == 6
                assert data["v_lower_violation"].shape[0] == 6
            from .sampling import load_evaluation_bundle

            Xi_loaded, rollout_loaded = load_evaluation_bundle(
                path=mc_summary["bundle_path"],
                n_blocks_expected=len(blocks),
                N_per_block_expected=2,
                D_expected=D,
                blocks_expected=blocks,
                T_total_expected=0.25,
                bundle_kind_expected="mc_confirmation",
            )
            assert Xi_loaded.shape == (6, D)
            assert len(rollout_loaded) == len(blocks)

            predict_call["count"] = 0
            rec_dir_repeat = rec_dir / "repeat_same_seed"
            summary_repeat = orchestration.print_recursive_pass(
                pass_entries=pass_entries,
                blocks=blocks,
                rec_dir=str(rec_dir_repeat),
                params=params,
                N_per_block=2,
                D=D,
                layers=layers,
                T_total=0.25,
                exact_solution=None,
                selection_metric="loss",
                eval_bundle_path=str(rec_dir_repeat / "evaluation_bundle.npz"),
                eval_seed=101,
                eval_min_paths=M,
                sample_paths=0,
                print_compact_logs=False,
                model_spec=spec,
                mc_confirmation_paths=6,
                mc_confirmation_seed=202,
            )
            mc_repeat = summary_repeat["mc_confirmation_summary"]
            mc_payload_repeat = json.loads(Path(mc_repeat["summary_path"]).read_text(encoding="utf-8"))
            assert mc_payload_repeat == mc_payload
            assert Path(mc_repeat["summary_path"]).read_bytes() == Path(mc_summary["summary_path"]).read_bytes()
            assert Path(mc_repeat["pathwise_npz_path"]).read_bytes() == Path(mc_summary["pathwise_npz_path"]).read_bytes()
            assert Path(mc_repeat["bundle_path"]).read_bytes() == Path(mc_summary["bundle_path"]).read_bytes()
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_rollout_boundaries_accepts_fixed_initial_and_rollout_inputs() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    M = 3
    D = 4
    N = 2
    layers = [5, 8, 1]
    blocks = [
        {"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25},
        {"idx": 1, "t_start": 0.25, "t_end": 0.50, "T_block": 0.25},
    ]
    Xi_initial = np.arange(M * D, dtype=np.float32).reshape(M, D) / np.float32(10.0)
    rollout_inputs = []
    for idx, block in enumerate(blocks):
        t_block = (
            np.linspace(block["t_start"], block["t_end"], N + 1, dtype=np.float32)
            .reshape(1, N + 1, 1)
            .repeat(M, axis=0)
        )
        W_block = np.full((M, N + 1, D), np.float32(idx + 1), dtype=np.float32)
        W_block[:, 0, :] = 0.0
        rollout_inputs.append((t_block, W_block))
    predict_calls = []

    class _Session:
        def close(self):
            pass

    class _BoundaryModel:
        def __init__(self, *, model_index: int, **kwargs):
            self.model_index = int(model_index)
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            self.blob = dict(blob)
            self.strict = bool(strict)

        def fetch_minibatch(self):
            raise AssertionError("fixed rollout_inputs should be used instead of fetch_minibatch")

        def predict(self, Xi_star, t_star, W_star, const_value=None):
            predict_calls.append(
                {
                    "idx": self.model_index,
                    "Xi": np.asarray(Xi_star, dtype=np.float32).copy(),
                    "t": np.asarray(t_star, dtype=np.float32).copy(),
                    "W": np.asarray(W_star, dtype=np.float32).copy(),
                    "const": float(const_value),
                }
            )
            X = np.repeat(np.asarray(Xi_star, dtype=np.float32)[:, None, :], np.asarray(t_star).shape[1], axis=1)
            X[:, -1, :] += np.float32(10.0 * (self.model_index + 1))
            Y = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.float32)
            Z = np.zeros((X.shape[0], X.shape[1], D), dtype=np.float32)
            return X.astype(np.float32), Y, Z

    model_counter = {"count": 0}

    def build_model(**kwargs):
        idx = model_counter["count"]
        model_counter["count"] += 1
        return _BoundaryModel(model_index=idx, **kwargs)

    spec = ModelSpec(
        name="boundary_replay_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=build_model,
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def forbidden_generator(M_arg, D_arg):
        raise AssertionError("fixed Xi_initial should be used instead of Xi_generator")

    boundaries = orchestration.rollout_boundaries(
        block_blobs=[_make_blob(layers), _make_blob(layers)],
        blocks=blocks,
        Xi_generator=forbidden_generator,
        params=base.build_default_params(),
        M_rollout=M,
        N_per_block=N,
        D=D,
        layers=layers,
        T_total=0.50,
        coupling_const=0.75,
        model_spec=spec,
        Xi_initial=Xi_initial,
        rollout_inputs=rollout_inputs,
    )

    assert len(boundaries) == 3
    np.testing.assert_allclose(boundaries[0], Xi_initial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(boundaries[1], Xi_initial + np.float32(10.0), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(boundaries[2], Xi_initial + np.float32(30.0), rtol=0.0, atol=1.0e-6)
    assert [call["idx"] for call in predict_calls] == [0, 1]
    np.testing.assert_allclose(predict_calls[0]["Xi"], boundaries[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(predict_calls[1]["Xi"], boundaries[1], rtol=0.0, atol=0.0)
    for idx, call in enumerate(predict_calls):
        np.testing.assert_allclose(call["t"], rollout_inputs[idx][0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(call["W"], rollout_inputs[idx][1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(call["const"], 0.75, rtol=0.0, atol=0.0)


def test_rollout_boundaries_rejects_fixed_input_shape_mismatch() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    base = get_model_spec("quadratic_coupled")
    M = 2
    D = 4
    N = 1
    layers = [5, 8, 1]
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    rollout_inputs = [
        (
            np.asarray([[[0.0], [0.25]], [[0.0], [0.25]]], dtype=np.float32),
            np.zeros((M, N + 1, D), dtype=np.float32),
        )
    ]

    try:
        orchestration.rollout_boundaries(
            block_blobs=[_make_blob(layers)],
            blocks=blocks,
            Xi_generator=base.xi_generator,
            params=base.build_default_params(),
            M_rollout=M,
            N_per_block=N,
            D=D,
            layers=layers,
            T_total=0.25,
            model_spec=base,
            Xi_initial=np.zeros((M, D + 1), dtype=np.float32),
            rollout_inputs=rollout_inputs,
        )
    except ValueError as exc:
        assert "Xi_initial" in str(exc)
    else:
        raise AssertionError("rollout_boundaries should reject invalid fixed Xi_initial shape")


def test_rollout_boundaries_rejects_malformed_fixed_rollout_inputs() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    base = get_model_spec("quadratic_coupled")
    M = 2
    D = 4
    N = 1
    layers = [5, 8, 1]
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    Xi_initial = np.zeros((M, D), dtype=np.float32)
    valid_t = np.asarray([[[0.0], [0.25]], [[0.0], [0.25]]], dtype=np.float32)
    valid_W = np.zeros((M, N + 1, D), dtype=np.float32)

    def assert_rejected(rollout_inputs, expected):
        try:
            orchestration.rollout_boundaries(
                block_blobs=[_make_blob(layers)],
                blocks=blocks,
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M_rollout=M,
                N_per_block=N,
                D=D,
                layers=layers,
                T_total=0.25,
                model_spec=base,
                Xi_initial=Xi_initial,
                rollout_inputs=rollout_inputs,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"rollout_boundaries should reject malformed rollout_inputs: {expected}")

    assert_rejected([], "one (t, W) pair per block")
    assert_rejected([(valid_t[:, :1, :], valid_W)], "shape")
    assert_rejected([(valid_t, valid_W[:, :, :3])], "shape")
    bad_t = valid_t.copy()
    bad_t[:, -1, 0] = np.float32(0.50)
    assert_rejected([(bad_t, valid_W)], "t_end")
    bad_W = valid_W.copy()
    bad_W[:, 0, :] = np.float32(1.0)
    assert_rejected([(valid_t, bad_W)], "W_start")


def test_rollout_boundaries_rejects_nonmonotonic_rollout_times() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    base = get_model_spec("quadratic_coupled")
    M = 2
    D = 4
    N = 2
    layers = [5, 8, 1]
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    Xi_initial = np.zeros((M, D), dtype=np.float32)
    bad_t = np.asarray([[[0.0], [0.25], [0.25]], [[0.0], [0.25], [0.25]]], dtype=np.float32)
    W = np.zeros((M, N + 1, D), dtype=np.float32)

    try:
        orchestration.rollout_boundaries(
            block_blobs=[_make_blob(layers)],
            blocks=blocks,
            Xi_generator=base.xi_generator,
            params=base.build_default_params(),
            M_rollout=M,
            N_per_block=N,
            D=D,
            layers=layers,
            T_total=0.25,
            model_spec=base,
            Xi_initial=Xi_initial,
            rollout_inputs=[(bad_t, W)],
        )
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("rollout_boundaries should reject nonmonotonic fixed rollout times")


def test_rollout_boundaries_legacy_path_still_uses_generator_and_minibatch() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    M = 2
    D = 4
    N = 1
    layers = [5, 8, 1]
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    Xi_from_generator = np.full((M, D), np.float32(3.0), dtype=np.float32)
    calls = {"generator": 0, "fetch": 0}

    class _Session:
        def close(self):
            pass

    class _LegacyBoundaryModel:
        def __init__(self, **kwargs):
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            pass

        def fetch_minibatch(self):
            calls["fetch"] += 1
            t = np.asarray([[[0.0], [0.25]], [[0.0], [0.25]]], dtype=np.float32)
            W = np.zeros((M, N + 1, D), dtype=np.float32)
            Xi = np.zeros((M, D), dtype=np.float32)
            return t, W, Xi

        def predict(self, Xi_star, t_star, W_star, const_value=None):
            X = np.repeat(np.asarray(Xi_star, dtype=np.float32)[:, None, :], np.asarray(t_star).shape[1], axis=1)
            X[:, -1, :] += np.float32(1.0)
            Y = np.zeros((M, N + 1, 1), dtype=np.float32)
            Z = np.zeros((M, N + 1, D), dtype=np.float32)
            return X, Y, Z

    spec = ModelSpec(
        name="boundary_legacy_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _LegacyBoundaryModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def generator(M_arg, D_arg):
        calls["generator"] += 1
        assert int(M_arg) == M
        assert int(D_arg) == D
        return Xi_from_generator.copy()

    boundaries = orchestration.rollout_boundaries(
        block_blobs=[_make_blob(layers)],
        blocks=blocks,
        Xi_generator=generator,
        params=base.build_default_params(),
        M_rollout=M,
        N_per_block=N,
        D=D,
        layers=layers,
        T_total=0.25,
        model_spec=spec,
    )

    assert calls == {"generator": 1, "fetch": 1}
    np.testing.assert_allclose(boundaries[0], Xi_from_generator, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(boundaries[1], Xi_from_generator + np.float32(1.0), rtol=0.0, atol=1.0e-6)


def test_load_evaluation_bundle_rejects_nonfinite_payload() -> None:
    from .sampling import load_evaluation_bundle, save_evaluation_bundle

    D = 4
    M = 3
    N = 2
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    Xi = np.zeros((M, D), dtype=np.float32)
    rollout = build_stitched_rollout_inputs(blocks, M=M, N_per_block=N, D=D, seed=41)

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = Path(tmp) / "evaluation_bundle.npz"
        save_evaluation_bundle(
            path=str(bundle_path),
            Xi_initial=Xi,
            rollout_inputs=rollout,
            blocks=blocks,
        )
        with np.load(bundle_path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}

        for key, index in (
            ("Xi_initial", (0, 0)),
            ("t_bundle", (0, 0, 1, 0)),
            ("W_bundle", (0, 0, 1, 0)),
        ):
            corrupt = {name: value.copy() for name, value in payload.items()}
            corrupt[key][index] = np.float32(np.nan)
            corrupt_path = Path(tmp) / f"corrupt_{key}.npz"
            np.savez(corrupt_path, **corrupt)
            try:
                load_evaluation_bundle(
                    path=str(corrupt_path),
                    n_blocks_expected=len(blocks),
                    N_per_block_expected=N,
                    D_expected=D,
                    blocks_expected=blocks,
                    T_total_expected=0.5,
                )
            except ValueError as exc:
                assert "non-finite" in str(exc).lower()
            else:
                raise AssertionError(f"load_evaluation_bundle should reject non-finite {key}")


def test_print_recursive_pass_fails_fast_on_nonfinite_eval_bundle_before_predict() -> None:
    from . import orchestration
    from .model_specs import get_model_spec
    from .sampling import save_evaluation_bundle

    spec = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    layers = [5, 8, 1]
    blocks = build_blocks(T_total=0.5, block_size=0.25)
    Xi = spec.deterministic_xi(M, D, seed=71)
    rollout = build_stitched_rollout_inputs(blocks, M=M, N_per_block=N, D=D, seed=72)
    bad_rollout = [(t.copy(), W.copy()) for t, W in rollout]
    bad_rollout[0][1][0, 1, 0] = np.float32(np.nan)
    logs = [
        {
            "pass": 1,
            "block": idx,
            "t_start": block["t_start"],
            "t_end": block["t_end"],
            "T_block": block["T_block"],
            "eval_mean_loss": 1.0,
            "eval_std_loss": 0.0,
            "eval_mean_loss_per_sample": 0.25,
            "eval_std_loss_per_sample": 0.0,
            "eval_mean_y0": 0.0,
            "precision_target": None,
            "refine_rounds": 0,
        }
        for idx, block in enumerate(blocks)
    ]
    predict_calls = {"count": 0}

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence

    def forbidden_predict(*args, **kwargs):
        predict_calls["count"] += 1
        raise AssertionError("corrupt evaluation bundle should fail before prediction")

    orchestration.predict_recursive_stitched = forbidden_predict
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "evaluation_bundle.npz"
            save_evaluation_bundle(
                path=str(bundle_path),
                Xi_initial=Xi,
                rollout_inputs=bad_rollout,
                blocks=blocks,
            )
            try:
                orchestration.print_recursive_pass(
                    pass_entries=[
                        {
                            "pass_id": 1,
                            "logs": logs,
                            "blobs": [
                                _make_block_blob(layers, block["t_start"], block["t_end"], 0.5)
                                for block in blocks
                            ],
                        }
                    ],
                    blocks=blocks,
                    rec_dir=str(Path(tmp) / "recursive"),
                    params=spec.build_default_params(),
                    N_per_block=N,
                    D=D,
                    layers=layers,
                    T_total=0.5,
                    exact_solution=None,
                    selection_metric="loss",
                    eval_bundle_path=str(bundle_path),
                    eval_seed=71,
                    eval_min_paths=M,
                    sample_paths=0,
                    print_compact_logs=False,
                    model_spec=spec,
                )
            except ValueError as exc:
                assert "non-finite" in str(exc).lower()
            else:
                raise AssertionError("print_recursive_pass should reject non-finite evaluation bundles")
            assert predict_calls["count"] == 0
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_run_recursive_training_wires_fixed_eval_and_gradient_diagnostics_to_logs() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    T_total = 0.5
    block_size = 0.25
    layers = [5, 8, 1]
    train_calls = []
    frozen_eval_calls = []

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.M = int(kwargs["M"])
            self.N = int(kwargs["N"])
            self.D = int(kwargs["D"])
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            self.blob = dict(blob)

        def export_parameter_blob(self):
            return _make_block_blob(
                layers,
                self.kwargs["t_start"],
                self.kwargs["t_end"],
                self.kwargs["T_total"],
            )

        def save_model(self, path: str) -> None:
            Path(path).write_text("fake checkpoint\n", encoding="utf-8")

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            frozen_eval_calls.append(
                {
                    "n_batches": n_batches,
                    "evaluation_batches": evaluation_batches,
                    "moment_policy": moment_policy,
                }
            )
            return {
                "mean_loss": 0.20,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.05,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
                "evaluation_batches": 1 if evaluation_batches is not None else 0,
                "moment_policy": moment_policy,
            }

    spec = ModelSpec(
        name="recursive_fixed_eval_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def fake_train_with_standard_schedule(**kwargs):
        train_calls.append(kwargs)
        eval_batches = kwargs["eval_batches"]
        assert not isinstance(eval_batches, (int, np.integer))
        assert kwargs["return_gradient_diagnostics"] is True
        assert len(eval_batches) == 1
        t_batch, W_batch, Xi_batch = eval_batches[0]
        assert t_batch.shape == (M, N + 1, 1)
        assert W_batch.shape == (M, N + 1, D)
        assert Xi_batch.shape == (M, D)
        assert np.isfinite(t_batch).all()
        assert np.isfinite(W_batch).all()
        assert np.isfinite(Xi_batch).all()
        diagnostics = {
            "train_grad_global_norm": 3.0,
            "train_grad_global_norm_clipped": 2.0,
            "train_grad_clip_ratio": 2.0 / 3.0,
            "train_grad_clip_enabled": 1.0,
            "train_gradient_variable_count": 4.0,
        }
        return {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 0.10,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.025,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
                "evaluation_batches": 1,
                "moment_policy": "fixed_evaluation_bundle",
            },
            "refine_rounds": 0,
            "precision_target": kwargs.get("precision_target"),
            "gradient_diagnostics": diagnostics,
        }

    def fake_rollout_boundaries(**kwargs):
        blocks = kwargs["blocks"]
        return [
            np.full((int(kwargs["M_rollout"]), int(kwargs["D"])), np.float32(idx), dtype=np.float32)
            for idx in range(len(blocks) + 1)
        ]

    original_train = orchestration.train_with_standard_schedule
    original_rollout = orchestration.rollout_boundaries
    orchestration.train_with_standard_schedule = fake_train_with_standard_schedule
    orchestration.rollout_boundaries = fake_rollout_boundaries
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=T_total,
                block_size=block_size,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=5,
                save_tf_checkpoints=False,
                n_passes=2,
                empirical_jitter_scale=0.0,
                freeze_stable_blocks_after_pass=1,
                freeze_loss_threshold=999.0,
                freeze_neighbor_radius=0,
                fixed_eval_seed=91,
                return_gradient_diagnostics=True,
                model_spec=spec,
            )
    finally:
        orchestration.train_with_standard_schedule = original_train
        orchestration.rollout_boundaries = original_rollout

    assert len(train_calls) == 3
    assert len(frozen_eval_calls) == 1
    train_batches_by_label = {
        call["label"]: call["eval_batches"]
        for call in train_calls
    }
    for left, right in (
        ("pass0:block0", "pass1:block0"),
    ):
        for actual_batch, expected_batch in zip(
            train_batches_by_label[left],
            train_batches_by_label[right],
        ):
            for actual, expected in zip(actual_batch, expected_batch):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    frozen_call = frozen_eval_calls[0]
    assert frozen_call["moment_policy"] == "fixed_evaluation_bundle"
    assert frozen_call["n_batches"] == 1
    assert len(frozen_call["evaluation_batches"]) == 1
    for actual, expected in zip(
        frozen_call["evaluation_batches"][0],
        train_batches_by_label["pass0:block1"][0],
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for pass_entry in result["passes"]:
        for row in pass_entry["logs"]:
            if bool(row.get("was_frozen", False)):
                continue
            for key in (
                "train_grad_global_norm",
                "train_grad_global_norm_clipped",
                "train_grad_clip_ratio",
                "train_grad_clip_enabled",
                "train_gradient_variable_count",
            ):
                assert key in row, f"{key} missing from recursive block log"
                assert np.isfinite(float(row[key]))


def test_run_recursive_training_warm_start_norm_policy_blob_uses_blob_normalization() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    layers = [5, 8, 1]
    model_inits = []
    import_calls = []

    blob = _make_block_blob(layers, 0.0, 0.25, 0.25)
    blob["x_norm_mean"] = np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    blob["x_norm_std"] = np.asarray([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.sess = _Session()
            model_inits.append(self.kwargs)

        def import_parameter_blob(self, imported_blob, strict=True):
            import_calls.append((dict(imported_blob), bool(strict)))

        def export_parameter_blob(self):
            return _make_block_blob(
                layers,
                self.kwargs["t_start"],
                self.kwargs["t_end"],
                self.kwargs["T_total"],
            )

    spec = ModelSpec(
        name="recursive_blob_norm_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def forbidden_estimate_generator_stats(*args, **kwargs):
        raise AssertionError("blob norm policy should not re-estimate warm-start normalization")

    def fake_train_with_standard_schedule(**kwargs):
        return {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 0.10,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.025,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
            "precision_target": kwargs.get("precision_target"),
            "gradient_diagnostics": {},
        }

    def fake_rollout_boundaries(**kwargs):
        return [
            np.full((int(kwargs["M_rollout"]), int(kwargs["D"])), np.float32(idx), dtype=np.float32)
            for idx in range(len(kwargs["blocks"]) + 1)
        ]

    original_estimate = orchestration.estimate_generator_stats
    original_train = orchestration.train_with_standard_schedule
    original_rollout = orchestration.rollout_boundaries
    orchestration.estimate_generator_stats = forbidden_estimate_generator_stats
    orchestration.train_with_standard_schedule = fake_train_with_standard_schedule
    orchestration.rollout_boundaries = fake_rollout_boundaries
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=0.25,
                block_size=0.25,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=4,
                save_tf_checkpoints=False,
                n_passes=1,
                empirical_jitter_scale=0.0,
                pass1_init_mode="coarse",
                initial_boundary_samples=[
                    np.zeros((M, D), dtype=np.float32),
                    np.ones((M, D), dtype=np.float32),
                ],
                initial_warm_start_blobs=[blob],
                warm_start_norm_policy="blob",
                model_spec=spec,
            )
    finally:
        orchestration.estimate_generator_stats = original_estimate
        orchestration.train_with_standard_schedule = original_train
        orchestration.rollout_boundaries = original_rollout

    assert len(model_inits) == 1
    np.testing.assert_allclose(model_inits[0]["x_norm_mean"], blob["x_norm_mean"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_inits[0]["x_norm_std"], blob["x_norm_std"], rtol=0.0, atol=0.0)
    assert len(import_calls) == 1
    assert import_calls[0][1] is False
    row = result["pass1"]["logs"][0]
    assert row["warm_start_norm_policy"] == "blob"
    assert row["warm_start_norm_source"] == "blob"
    assert row["warm_start_blob_used"] is True


def test_run_recursive_training_blob_norm_policy_rejects_blob_without_norm_stats() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    layers = [5, 8, 1]
    blob = _make_block_blob(layers, 0.0, 0.25, 0.25)
    blob.pop("x_norm_mean", None)

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.sess = _Session()

        def export_parameter_blob(self):
            return _make_block_blob(layers, 0.0, 0.25, 0.25)

    spec = ModelSpec(
        name="recursive_missing_blob_norm_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=0.25,
                block_size=0.25,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=4,
                save_tf_checkpoints=False,
                n_passes=1,
                empirical_jitter_scale=0.0,
                pass1_init_mode="coarse",
                initial_boundary_samples=[
                    np.zeros((M, D), dtype=np.float32),
                    np.ones((M, D), dtype=np.float32),
                ],
                initial_warm_start_blobs=[blob],
                warm_start_norm_policy="blob",
                model_spec=spec,
            )
    except ValueError as exc:
        message = str(exc)
        assert "warm_start_norm_policy='blob'" in message
        assert "x_norm_mean/x_norm_std" in message
    else:
        raise AssertionError("blob norm policy should reject warm-start blobs without norm stats")


def test_run_recursive_training_blob_norm_policy_rejects_invalid_norm_stats() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    layers = [5, 8, 1]
    blob = _make_block_blob(layers, 0.0, 0.25, 0.25)
    blob["x_norm_std"] = np.asarray([[1.0, 0.0, np.inf, 2.0]], dtype=np.float32)

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.sess = _Session()

        def export_parameter_blob(self):
            return _make_block_blob(layers, 0.0, 0.25, 0.25)

    spec = ModelSpec(
        name="recursive_invalid_blob_norm_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=0.25,
                block_size=0.25,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=4,
                save_tf_checkpoints=False,
                n_passes=1,
                empirical_jitter_scale=0.0,
                pass1_init_mode="coarse",
                initial_boundary_samples=[
                    np.zeros((M, D), dtype=np.float32),
                    np.ones((M, D), dtype=np.float32),
                ],
                initial_warm_start_blobs=[blob],
                warm_start_norm_policy="blob",
                model_spec=spec,
            )
    except ValueError as exc:
        message = str(exc)
        assert "warm_start_norm_policy='blob'" in message
        assert "finite positive x_norm_std" in message
    else:
        raise AssertionError("blob norm policy should reject invalid x_norm_std")


def test_run_recursive_training_rejects_unknown_warm_start_norm_policy() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    spec = get_model_spec("quadratic_coupled")
    try:
        orchestration.run_recursive_training(
            Xi_generator=spec.xi_generator,
            params=spec.build_default_params(),
            M=2,
            N_per_block=1,
            D=4,
            T_total=0.25,
            block_size=0.25,
            layers=[5, 8, 1],
            stage_plan=[],
            final_plan=[],
            output_dir="unused",
            warm_start_norm_policy="mismatched",
            model_spec=spec,
        )
    except ValueError as exc:
        assert "warm_start_norm_policy" in str(exc)
    else:
        raise AssertionError("run_recursive_training should reject unknown warm_start_norm_policy")


def test_run_recursive_training_pass_boundary_rollouts_use_fixed_rollout_inputs() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    T_total = 0.5
    block_size = 0.25
    rollout_M = 5
    layers = [5, 8, 1]
    blocks = build_blocks(T_total=T_total, block_size=block_size)
    fixed_Xi = base.deterministic_xi(rollout_M, D, seed=61)
    fixed_rollout = build_stitched_rollout_inputs(
        blocks=blocks,
        M=rollout_M,
        N_per_block=N,
        D=D,
        seed=62,
    )
    rollout_calls = []

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.M = int(kwargs["M"])
            self.N = int(kwargs["N"])
            self.D = int(kwargs["D"])
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            pass

        def export_parameter_blob(self):
            return _make_block_blob(
                layers,
                self.kwargs["t_start"],
                self.kwargs["t_end"],
                self.kwargs["T_total"],
            )

        def save_model(self, path: str) -> None:
            Path(path).write_text("fake checkpoint\n", encoding="utf-8")

    spec = ModelSpec(
        name="recursive_fixed_rollout_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def fake_train_with_standard_schedule(**kwargs):
        return {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 0.10,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.025,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
            "precision_target": kwargs.get("precision_target"),
        }

    def fake_rollout_boundaries(**kwargs):
        rollout_calls.append(kwargs)
        np.testing.assert_allclose(kwargs["Xi_initial"], fixed_Xi, rtol=0.0, atol=0.0)
        assert len(kwargs["rollout_inputs"]) == len(fixed_rollout)
        for actual, expected in zip(kwargs["rollout_inputs"], fixed_rollout):
            np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
        return [
            np.full((int(kwargs["M_rollout"]), int(kwargs["D"])), np.float32(idx), dtype=np.float32)
            for idx in range(len(kwargs["blocks"]) + 1)
        ]

    original_train = orchestration.train_with_standard_schedule
    original_rollout = orchestration.rollout_boundaries
    orchestration.train_with_standard_schedule = fake_train_with_standard_schedule
    orchestration.rollout_boundaries = fake_rollout_boundaries
    try:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=T_total,
                block_size=block_size,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=rollout_M,
                save_tf_checkpoints=False,
                n_passes=2,
                empirical_jitter_scale=0.0,
                fixed_boundary_Xi_initial=fixed_Xi,
                fixed_boundary_rollout_inputs=fixed_rollout,
                model_spec=spec,
            )
    finally:
        orchestration.train_with_standard_schedule = original_train
        orchestration.rollout_boundaries = original_rollout

    assert len(rollout_calls) == 2


def test_run_recursive_training_negative_fixed_eval_seed_keeps_legacy_batches() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 3
    N = 2
    layers = [5, 8, 1]
    train_eval_batches = []

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.M = int(kwargs["M"])
            self.N = int(kwargs["N"])
            self.D = int(kwargs["D"])
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            pass

        def export_parameter_blob(self):
            return _make_block_blob(
                layers,
                self.kwargs["t_start"],
                self.kwargs["t_end"],
                self.kwargs["T_total"],
            )

    spec = ModelSpec(
        name="recursive_negative_seed_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def fake_train_with_standard_schedule(**kwargs):
        train_eval_batches.append(kwargs["eval_batches"])
        return {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 0.10,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.025,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
            "precision_target": kwargs.get("precision_target"),
        }

    def fake_rollout_boundaries(**kwargs):
        return [
            np.zeros((int(kwargs["M_rollout"]), int(kwargs["D"])), dtype=np.float32)
            for _ in range(len(kwargs["blocks"]) + 1)
        ]

    original_train = orchestration.train_with_standard_schedule
    original_rollout = orchestration.rollout_boundaries
    orchestration.train_with_standard_schedule = fake_train_with_standard_schedule
    orchestration.rollout_boundaries = fake_rollout_boundaries
    try:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=0.25,
                block_size=0.25,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=5,
                save_tf_checkpoints=False,
                n_passes=1,
                fixed_eval_seed=-1,
                model_spec=spec,
            )
    finally:
        orchestration.train_with_standard_schedule = original_train
        orchestration.rollout_boundaries = original_rollout

    assert train_eval_batches == [5]


def test_run_recursive_training_copies_external_fixed_eval_batches_by_block() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("quadratic_coupled")
    D = 4
    M = 2
    N = 1
    layers = [5, 8, 1]
    external_batches = _schedule_fixed_eval_batches(M=M, N=N, D=D, t_start=0.0, t_end=0.25)
    fixed_eval_batches_by_block = [external_batches]
    train_calls = []

    class _Session:
        def close(self):
            pass

    class _RecursiveModel:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.M = int(kwargs["M"])
            self.N = int(kwargs["N"])
            self.D = int(kwargs["D"])
            self.sess = _Session()

        def import_parameter_blob(self, blob, strict=True):
            pass

        def export_parameter_blob(self):
            return _make_block_blob(
                layers,
                self.kwargs["t_start"],
                self.kwargs["t_end"],
                self.kwargs["T_total"],
            )

    spec = ModelSpec(
        name="recursive_external_fixed_eval_copy_stub",
        state_dim=D,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: _RecursiveModel(**kwargs),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
    )

    def fake_train_with_standard_schedule(**kwargs):
        eval_batches = kwargs["eval_batches"]
        assert not isinstance(eval_batches, (int, np.integer))
        _, _, Xi_batch = eval_batches[0]
        np.testing.assert_allclose(Xi_batch, 0.0, rtol=0.0, atol=0.0)
        Xi_batch[:, :] = np.float32(123.0)
        train_calls.append(eval_batches)
        return {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 0.10,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.05,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
            "precision_target": kwargs.get("precision_target"),
        }

    def fake_rollout_boundaries(**kwargs):
        return [
            np.zeros((int(kwargs["M_rollout"]), int(kwargs["D"])), dtype=np.float32)
            for _ in range(len(kwargs["blocks"]) + 1)
        ]

    original_train = orchestration.train_with_standard_schedule
    original_rollout = orchestration.rollout_boundaries
    orchestration.train_with_standard_schedule = fake_train_with_standard_schedule
    orchestration.rollout_boundaries = fake_rollout_boundaries
    try:
        with tempfile.TemporaryDirectory() as tmp:
            orchestration.run_recursive_training(
                Xi_generator=base.xi_generator,
                params=base.build_default_params(),
                M=M,
                N_per_block=N,
                D=D,
                T_total=0.25,
                block_size=0.25,
                layers=layers,
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                output_dir=tmp,
                max_refine_rounds=0,
                rollout_M=4,
                save_tf_checkpoints=False,
                n_passes=2,
                fixed_eval_batches_by_block=fixed_eval_batches_by_block,
                model_spec=spec,
            )
    finally:
        orchestration.train_with_standard_schedule = original_train
        orchestration.rollout_boundaries = original_rollout

    assert len(train_calls) == 2
    np.testing.assert_allclose(external_batches[0][2], 0.0, rtol=0.0, atol=0.0)


def test_score_pass_logs_rejects_nonfinite_rows() -> None:
    from .plotting import score_pass_logs

    finite_rows = [
        {"eval_mean_loss_per_sample": 0.20},
        {"eval_mean_loss_per_sample": 0.10},
        {"eval_mean_loss_per_sample": 0.30},
    ]
    finite_score = score_pass_logs(finite_rows)
    np.testing.assert_allclose(finite_score, np.mean([0.20, 0.10, 0.30]) + 0.35 * 0.30)

    mixed_rows = [
        {"eval_mean_loss_per_sample": 0.0001},
        {"eval_mean_loss_per_sample": np.nan},
        {"eval_mean_loss_per_sample": 0.0002},
    ]
    assert np.isinf(score_pass_logs(mixed_rows))
    assert np.isinf(score_pass_logs([{"eval_mean_loss_per_sample": np.nan}]))
    assert np.isinf(score_pass_logs([{}]))


def test_resolve_pass_selection_rejects_no_finite_loss_scores() -> None:
    from .orchestration import resolve_pass_selection

    try:
        resolve_pass_selection(
            pass_scores_by_loss={1: float("inf"), 2: np.nan},
            exact_summary_by_pass={},
            selection_metric="loss",
        )
    except RuntimeError as exc:
        assert "finite" in str(exc).lower()
        assert "pass" in str(exc).lower()
    else:
        raise AssertionError("selection should reject all-nonfinite loss candidates")


def test_print_recursive_pass_rejects_invalid_candidate_without_rewriting_final_artifacts() -> None:
    from . import orchestration
    from .model_specs import ModelSpec, get_model_spec

    base = get_model_spec("pascucci")
    M = 4
    D = 4
    layers = [5, 8, 1]
    z_v_index = 2

    class FakeApplicationModel:
        def _summary(self, running, terminal, total, running_cumulative, trajectory, trajectory_math):
            summary = {}
            for key, values in (
                ("cost_J_running", running),
                ("cost_J_terminal", terminal),
                ("cost_J_total", total),
                ("cost_J_running_cumulative", running_cumulative),
                ("cost_J_trajectory", trajectory),
                ("cost_J_trajectory_math", trajectory_math),
            ):
                values_np = np.asarray(values, dtype=np.float32)
                if key in ("cost_J_running_cumulative", "cost_J_trajectory", "cost_J_trajectory_math"):
                    flat = (
                        values_np[:, -1, :].reshape(-1)
                        if values_np.shape[1] > 0
                        else np.zeros((values_np.shape[0],), dtype=np.float32)
                    )
                else:
                    flat = values_np.reshape(-1)
                summary[f"{key}_mean"] = float(np.mean(flat))
                summary[f"{key}_std"] = float(np.std(flat))
                summary[f"{key}_q05"] = float(np.quantile(flat, 0.05))
                summary[f"{key}_q50"] = float(np.quantile(flat, 0.50))
                summary[f"{key}_q95"] = float(np.quantile(flat, 0.95))
            return summary

        def _result(self, *, t, cost_seed, alpha, baseline_mode, control_law, paired_inputs):
            del t
            M_local = int(alpha.shape[0])
            steps = int(alpha.shape[1])
            running = np.full((M_local, 1), np.float32(cost_seed), dtype=np.float32)
            running_step = np.full(
                (M_local, steps, 1),
                np.float32(cost_seed / max(steps, 1)),
                dtype=np.float32,
            )
            running_cumulative = np.cumsum(running_step, axis=1).astype(np.float32)
            terminal = np.full((M_local, 1), np.float32(0.25), dtype=np.float32)
            total = running + terminal
            trajectory = (
                np.concatenate([np.zeros((M_local, 1, 1), dtype=np.float32), running_cumulative], axis=1)
                + terminal[:, None, :]
            ).astype(np.float32)
            trajectory_math = trajectory.copy()
            return {
                "schema": "pascucci_application_metrics_v2",
                "metadata": {
                    "baseline_mode": baseline_mode,
                    "aggregation": "left_riemann_f_plus_terminal_g",
                    "trajectory_aggregation": "plotmaker_cumsum_f_dt_plus_g_at_time",
                    "trajectory_math_aggregation": "left_riemann_integral_to_time_plus_g_at_time",
                    "control_law": control_law,
                    "paired_inputs": paired_inputs,
                },
                "pathwise": {
                    "cost_J_running": running,
                    "cost_J_terminal": terminal,
                    "cost_J_total": total,
                    "cost_J_running_cumulative": running_cumulative,
                    "cost_J_trajectory": trajectory,
                    "cost_J_trajectory_math": trajectory_math,
                    "alpha": alpha.astype(np.float32),
                },
                "summary": self._summary(running, terminal, total, running_cumulative, trajectory, trajectory_math),
            }

        def application_cost_from_path(self, t, X, Y, Z, const_value=None, baseline_mode="controlled", **kwargs):
            del Y, const_value, kwargs
            cost_seed = float(np.asarray(X, dtype=np.float32)[0, 0, 0]) + 1.0
            alpha = np.asarray(Z, dtype=np.float32)[:, :-1, [z_v_index]]
            return self._result(
                t=t,
                cost_seed=cost_seed,
                alpha=alpha,
                baseline_mode=str(baseline_mode),
                control_law="alpha_tf",
                paired_inputs="stitched_XYZ",
            )

        def application_cost_functional(self, t, W, Xi, const_value=None, baseline_mode="uncontrolled"):
            del W, const_value
            steps = int(np.asarray(t).shape[1]) - 1
            alpha = np.zeros((int(np.asarray(Xi).shape[0]), steps, 1), dtype=np.float32)
            return self._result(
                t=t,
                cost_seed=3.0,
                alpha=alpha,
                baseline_mode=str(baseline_mode),
                control_law="alpha_zero",
                paired_inputs="same_t_W_Xi",
            )

        def close(self):
            pass

    spec = ModelSpec(
        name="pascucci_invalid_pass_stub",
        state_dim=4,
        state_labels=base.state_labels,
        z_labels=base.z_labels,
        build_default_params=base.build_default_params,
        build_layers=base.build_layers,
        xi_generator=base.xi_generator,
        deterministic_xi=base.deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: FakeApplicationModel(),
        build_exact_solution=lambda *args, **kwargs: None,
        build_exact_initial_boundary_samples=None,
        moment_names=base.moment_names,
        application_metric_schema=base.application_metric_schema,
        application_metric_names=base.application_metric_names,
        application_metric_aggregation=base.application_metric_aggregation,
    )
    params = spec.build_default_params(const=0.75)
    blocks = [
        {"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25},
        {"idx": 1, "t_start": 0.25, "t_end": 0.50, "T_block": 0.25},
    ]
    pass_entries = [
        {
            "pass_id": 1,
            "logs": [
                {
                    "pass": 1,
                    "block": 0,
                    "t_start": 0.0,
                    "t_end": 0.25,
                    "T_block": 0.25,
                    "eval_mean_loss": 100.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.10,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "precision_target": None,
                    "refine_rounds": 0,
                },
                {
                    "pass": 1,
                    "block": 1,
                    "t_start": 0.25,
                    "t_end": 0.50,
                    "T_block": 0.25,
                    "eval_mean_loss": 120.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.12,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "precision_target": None,
                    "refine_rounds": 0,
                },
            ],
            "blobs": [_make_blob(layers), _make_blob(layers)],
        },
        {
            "pass_id": 2,
            "logs": [
                {
                    "pass": 2,
                    "block": 0,
                    "t_start": 0.0,
                    "t_end": 0.25,
                    "T_block": 0.25,
                    "eval_mean_loss": np.nan,
                    "eval_std_loss": np.nan,
                    "eval_mean_loss_per_sample": np.nan,
                    "eval_std_loss_per_sample": np.nan,
                    "eval_mean_y0": np.nan,
                    "precision_target": None,
                    "refine_rounds": 0,
                },
                {
                    "pass": 2,
                    "block": 1,
                    "t_start": 0.25,
                    "t_end": 0.50,
                    "T_block": 0.25,
                    "eval_mean_loss": 1.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.0001,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": -0.9,
                    "precision_target": None,
                    "refine_rounds": 0,
                },
            ],
            "blobs": [_make_blob(layers), _make_blob(layers)],
        },
    ]

    predict_call = {"count": 0}

    def fake_predict_recursive_stitched(**kwargs):
        predict_call["count"] += 1
        rollout_inputs = kwargs["rollout_inputs"]
        t_segments = []
        for idx, (t_block, _W_block) in enumerate(rollout_inputs):
            t_np = np.asarray(t_block, dtype=np.float32)
            t_segments.append(t_np if idx == 0 else t_np[:, 1:, :])
        t = np.concatenate(t_segments, axis=1).astype(np.float32)
        X = np.zeros((M, t.shape[1], D), dtype=np.float32)
        X[:, :, 3] = 5.0
        Y = np.full((M, t.shape[1], 1), np.float32(predict_call["count"] - 1), dtype=np.float32)
        Z = np.zeros((M, t.shape[1], D), dtype=np.float32)
        if predict_call["count"] == 2:
            Y[:] = np.nan
            Z[:] = np.nan
        jumps_x = np.zeros((len(blocks) - 1, M, D), dtype=np.float32)
        jumps_y = np.zeros((len(blocks) - 1, M, 1), dtype=np.float32)
        jumps_z = np.zeros((len(blocks) - 1, M, D), dtype=np.float32)
        return {
            "t": t,
            "X": X,
            "Y": Y,
            "Z": Z,
            "stitch_X_boundary_signed_jump": jumps_x,
            "stitch_X_boundary_abs_jump": jumps_x,
            "stitch_Y_boundary_signed_jump": jumps_y,
            "stitch_Y_boundary_abs_jump": jumps_y,
            "stitch_Z_boundary_signed_jump": jumps_z,
            "stitch_Z_boundary_abs_jump": jumps_z,
        }

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_stitched = orchestration.plot_recursive_stitched_predictions
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence
    orchestration.predict_recursive_stitched = fake_predict_recursive_stitched
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_predictions = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp)
            stale_json = rec_dir / "application_metrics_final.json"
            stale_npz = rec_dir / "stitched_predictions_final.npz"
            stale_json.write_text('{"stale": true}\n', encoding="utf-8")
            np.savez(stale_npz, sentinel=np.asarray([123], dtype=np.int32))

            try:
                orchestration.print_recursive_pass(
                    pass_entries=pass_entries,
                    blocks=blocks,
                    rec_dir=str(rec_dir),
                    params=params,
                    N_per_block=2,
                    D=D,
                    layers=layers,
                    T_total=0.50,
                    exact_solution=None,
                    selection_metric="loss",
                    eval_bundle_path=str(rec_dir / "evaluation_bundle.npz"),
                    eval_seed=101,
                    eval_min_paths=M,
                    sample_paths=0,
                    print_compact_logs=False,
                    exclude_pass_ids_from_selection=[1],
                    model_spec=spec,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                assert "finite" in message
                assert "pass" in message
            else:
                raise AssertionError("invalid pass candidate should stop final artifact selection")

            assert json.loads(stale_json.read_text(encoding="utf-8")) == {"stale": True}
            with np.load(stale_npz) as data:
                np.testing.assert_array_equal(data["sentinel"], np.asarray([123], dtype=np.int32))
            assert not (rec_dir / "application_metrics_pass01.json").exists()
            assert not (rec_dir / "application_metrics_final.npz").exists()
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_print_recursive_pass_can_skip_intermediate_final_promotion() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    M = 4
    D = 4
    layers = [5, 8, 1]
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    pass_entries = [
        {
            "pass_id": 1,
            "logs": [
                {
                    "pass": 1,
                    "block": 0,
                    "t_start": 0.0,
                    "t_end": 0.25,
                    "T_block": 0.25,
                    "eval_mean_loss": 1.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.25,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "precision_target": None,
                    "refine_rounds": 0,
                }
            ],
            "blobs": [_make_blob(layers)],
        }
    ]

    def fake_predict_recursive_stitched(**kwargs):
        rollout_inputs = kwargs["rollout_inputs"]
        t = np.asarray(rollout_inputs[0][0], dtype=np.float32)
        X = np.zeros((M, t.shape[1], D), dtype=np.float32)
        X[:, :, 3] = 5.0
        Y = np.zeros((M, t.shape[1], 1), dtype=np.float32)
        Z = np.zeros((M, t.shape[1], D), dtype=np.float32)
        return {"t": t, "X": X, "Y": Y, "Z": Z}

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_stitched = orchestration.plot_recursive_stitched_predictions
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence
    orchestration.predict_recursive_stitched = fake_predict_recursive_stitched
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_predictions = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp)
            summary = orchestration.print_recursive_pass(
                pass_entries=pass_entries,
                blocks=blocks,
                rec_dir=str(rec_dir),
                params=params,
                N_per_block=2,
                D=D,
                layers=layers,
                T_total=0.25,
                exact_solution=None,
                selection_metric="loss",
                eval_bundle_path=str(rec_dir / "evaluation_bundle.npz"),
                eval_seed=101,
                eval_min_paths=M,
                sample_paths=0,
                print_compact_logs=False,
                promote_final_artifacts=False,
                exclude_pass_ids_from_selection=[1],
                model_spec=spec,
            )

            assert summary["selection_finalized"] is False
            assert summary["selected_pass_id"] is None
            assert summary["excluded_pass_ids_from_selection"] == []
            assert summary["promoted_final_artifacts"] is False
            assert (rec_dir / "stitched_predictions_pass00.npz").exists()
            assert not (rec_dir / "stitched_predictions_final.npz").exists()
            assert not (rec_dir / "application_metrics_final.json").exists()
            assert not (rec_dir / "application_metrics_final.npz").exists()
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_pascucci_t12_gate_validator_requires_results_manifest_without_nameerror() -> None:
    from .pascucci_run_validation import validate_t12_gate_n13

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        rec_dir = run_root / "recursive"
        rec_dir.mkdir(parents=True)
        (run_root / "run_config.json").write_text(
            json.dumps(
                {
                    "model_name": "pascucci",
                    "M": 10000,
                    "N": 13,
                    "D": 4,
                    "T_total": 12.0,
                    "block_size": 2.0,
                    "passes": 2,
                    "state_labels": ["S", "H", "V", "X_state"],
                    "z_labels": ["Z_S", "Z_H", "Z_V", "Z_X"],
                    "application_metric_schema": "pascucci_application_metrics_v2",
                }
            ),
            encoding="utf-8",
        )
        np.savez(
            rec_dir / "evaluation_bundle.npz",
            Xi_initial=np.zeros((10000, 4), dtype=np.float32),
            t_bundle=np.zeros((6, 10000, 14, 1), dtype=np.float32),
            W_bundle=np.zeros((6, 10000, 14, 4), dtype=np.float32),
            block_t_start=np.asarray([0, 2, 4, 6, 8, 10], dtype=np.float32),
            block_t_end=np.asarray([2, 4, 6, 8, 10, 12], dtype=np.float32),
        )

        report = validate_t12_gate_n13(run_root)

    assert report["status"] == "FAILED"
    joined = "\n".join(report["failures"] + report["incomplete"])
    assert "results.json" in joined
    assert "Xi_initial" not in joined


def _write_minimal_paper_parity_artifacts(
    run_root: Path,
    *,
    legacy_artifacts: bool = False,
    broken_cost_accounting: bool = False,
    science_red: bool = False,
    missing_mc_confirmation: bool = False,
    mc_science_red: bool = False,
    mc_not_independent: bool = False,
    mc_paired_count_drift: bool = False,
    mc_violation_magnitude: float = 1.0,
) -> None:
    from .io_utils import save_blob_npz, save_json
    from .model_specs import get_model_spec
    from .pascucci_calibration import (
        build_pascucci_calibration_config,
        build_pascucci_run_config_params,
        calibrate_pascucci_ou_inputs,
    )
    from .pascucci_data import prepare_H, prepare_S
    from .pascucci_plotting import _load_to_ema_ou_reference, _load_to_ema_uncontrolled_reference

    M = 256
    fixture = _pascucci_paper_plot_smoke_inputs(M=M)
    t_steps = int(fixture["stitched"]["t"].shape[1])
    t_grid = np.linspace(0.0, 24.0, t_steps, dtype=np.float32).reshape(1, t_steps, 1)
    fixture["stitched"]["t"] = np.tile(t_grid, (M, 1, 1)).astype(np.float32)
    rec_dir = run_root / "recursive"
    plot_dir = rec_dir / "plots" / "pascucci_paper"
    source_dir = run_root / "calibration_sources"
    plot_dir.mkdir(parents=True)
    H_source = source_dir / "dataset" / "casa" / "2025dicembre1.csv"
    S_source = source_dir / "dataset" / "prezzi" / "2025dicembre1.xlsx"
    H_source.parent.mkdir(parents=True)
    S_source.parent.mkdir(parents=True)
    to_ema_dataset = Path(__file__).resolve().parents[3] / "to_ema" / "dataset"
    shutil.copyfile(to_ema_dataset / "casa" / "2025dicembre1.csv", H_source)
    shutil.copyfile(to_ema_dataset / "prezzi" / "2025dicembre1.xlsx", S_source)

    prepared_H = prepare_H(str(H_source), n=4, mul_factor=0.001)
    prepared_S = prepare_S(str(S_source), n=1, mul_factor=0.001)
    calibration = calibrate_pascucci_ou_inputs(
        prepared_H,
        prepared_S,
        K=2,
        dt=1.0,
        start_hour=0.0,
        log_price=True,
    )
    calibration_config = build_pascucci_calibration_config(
        calibration,
        K=2,
        dt=1.0,
        start_hour=0.0,
        log_price=True,
        H_metadata={
            "source_path": str(H_source),
            "source_sha256": file_sha256(str(H_source)),
            "n_per_hour": 4,
            "mul_factor": 0.001,
            "prepared_points": int(prepared_H.shape[0]),
        },
        S_metadata={
            "source_path": str(S_source),
            "source_sha256": file_sha256(str(S_source)),
            "n_per_hour": 1,
            "mul_factor": 0.001,
            "prepared_points": int(prepared_S.shape[0]),
        },
    )
    params = build_pascucci_run_config_params(fixture["params"], calibration_config)
    if legacy_artifacts:
        params.pop("pascucci_calibration")

    save_json(
        {
            "model_name": "pascucci",
            "M": M,
            "N": 13,
            "D": 4,
            "T_total": 24.0,
            "block_size": 2.0,
            "passes": 1,
            "state_labels": ["S", "H", "V", "X_state"],
            "z_labels": ["Z_S", "Z_H", "Z_V", "Z_X"],
            "params": params,
            "application_metric_schema": "pascucci_application_metrics_v2",
            "pascucci_require_calibration": not legacy_artifacts,
            "pascucci_calibration_enabled": not legacy_artifacts,
            "run_config_sha256": "4" * 64,
            "seed_manifest": {"train_seed": 1, "eval_seed": 2, "visual_seed_effective": 3},
        },
        str(run_root / "run_config.json"),
    )
    save_blob_npz(fixture["stitched"], str(rec_dir / "stitched_predictions_final.npz"))
    application_pathwise = dict(fixture["application_pathwise"])
    if legacy_artifacts:
        application_pathwise.pop("controlled_cost_J_trajectory")
        application_pathwise.pop("uncontrolled_cost_J_trajectory")
        application_pathwise.pop("controlled_cost_J_trajectory_math")
        application_pathwise.pop("uncontrolled_cost_J_trajectory_math")
    if broken_cost_accounting and not legacy_artifacts:
        application_pathwise["controlled_cost_J_trajectory_math"] = application_pathwise[
            "controlled_cost_J_trajectory_math"
        ].copy()
        application_pathwise["controlled_cost_J_trajectory_math"][:, -1, :] += np.float32(7.0)
    save_blob_npz(application_pathwise, str(rec_dir / "application_metrics_final.npz"))

    pass0_metrics = _application_summary_fixture(
        controlled_total_mean=-3.0,
        uncontrolled_total_mean=-2.0,
        control_win_rate=0.60,
        v_lower_rate=0.02,
    )
    final_metrics = _application_summary_fixture(
        controlled_total_mean=4.0 if science_red else -4.0,
        uncontrolled_total_mean=-2.0,
        control_win_rate=0.20 if science_red else 0.70,
        q_lower_rate=0.30 if science_red else 0.01,
        v_lower_rate=0.28 if science_red else 0.01,
    )
    final_metrics = {
        **final_metrics,
        "selected_from_pass_id": 1,
        "selected_from_pass_index": 0,
    }
    save_json(pass0_metrics, str(rec_dir / "application_metrics_pass00.json"))
    save_json(final_metrics, str(rec_dir / "application_metrics_final.json"))

    if not missing_mc_confirmation and not legacy_artifacts:
        mc_M = 10000
        uncontrolled_total = np.zeros((mc_M, 1), dtype=np.float32)
        controlled_total = np.ones((mc_M, 1), dtype=np.float32)
        mc_wins = int((0.20 if mc_science_red else 0.70) * mc_M)
        controlled_total[:mc_wins, :] = np.float32(-1.0)
        delta = controlled_total - uncontrolled_total
        q_lower_violation = np.zeros((mc_M, 2), dtype=np.float32)
        q_upper_violation = np.zeros((mc_M, 2), dtype=np.float32)
        v_lower_violation = np.zeros((mc_M, 2), dtype=np.float32)
        v_upper_violation = np.zeros((mc_M, 2), dtype=np.float32)
        violation_magnitude = np.float32(mc_violation_magnitude)
        if mc_science_red:
            q_lower_violation[:3000, :] = violation_magnitude
            v_lower_violation[:2800, :] = violation_magnitude
        else:
            q_lower_violation[:100, :] = violation_magnitude
            v_lower_violation[:100, :] = violation_magnitude
        save_blob_npz(
            {
                "controlled_cost_J_total": controlled_total,
                "uncontrolled_cost_J_total": uncontrolled_total,
                "q_lower_violation": q_lower_violation,
                "q_upper_violation": q_upper_violation,
                "v_lower_violation": v_lower_violation,
                "v_upper_violation": v_upper_violation,
            },
            str(rec_dir / "application_metrics_mc_confirmation.npz"),
        )
        mc_seed = 2 if mc_not_independent else 987654
        save_json(
            {
                "schema": "pascucci_mc_confirmation_v1",
                "source": "independent_post_selection",
                "model_name": "pascucci",
                "selected_pass_id": 1,
                "seed": mc_seed,
                "independent_of_eval_bundle": not mc_not_independent,
                "eval_bundle_reused": bool(mc_not_independent),
                "pathwise_npz": "application_metrics_mc_confirmation.npz",
                "horizon": {
                    "T_total": 24.0,
                    "sample_paths": mc_M,
                    "seed": mc_seed,
                },
                "comparison": {
                    "delta_cost_J_total_mean": float(np.mean(delta)),
                    "cost_J_total_control_win_rate": float(np.mean(controlled_total < uncontrolled_total)),
                    "paired_sample_count": mc_M - 1 if mc_paired_count_drift else mc_M,
                },
                "diagnostics": {
                    "q_lower_violation_rate": float(np.mean(q_lower_violation > 0.0)),
                    "q_upper_violation_rate": float(np.mean(q_upper_violation > 0.0)),
                    "v_lower_violation_rate": float(np.mean(v_lower_violation > 0.0)),
                    "v_upper_violation_rate": float(np.mean(v_upper_violation > 0.0)),
                },
            },
            str(rec_dir / "application_metrics_mc_confirmation.json"),
        )
    save_json(
        {
            "blocks": [{"idx": i, "t_start": 2.0 * i, "t_end": 2.0 * (i + 1)} for i in range(12)],
            "passes": [{"pass_id": 1}],
            "evaluation_bundle_M": M,
            "selected_pass_id": 1,
            "selected_score": 0.25,
            "loss_pass_scores": {"1": 0.25},
            "pass_invalid_reasons": {},
        },
        str(rec_dir / "results.json"),
    )

    if legacy_artifacts:
        xi_initial = get_model_spec("quadratic_coupled").deterministic_xi(M, 4, seed=123)
    else:
        xi_initial = get_model_spec("pascucci").deterministic_xi(M, 4, seed=123)
    np.savez_compressed(rec_dir / "evaluation_bundle.npz", Xi_initial=xi_initial.astype(np.float32))

    plot_inputs = {
        "#35": [
            "pascucci_data.prepare_S(source_path, n, mul_factor)",
            "simulate_ou_day_night(logS0=log(real_S[0]), dt=0.5, Nsim=10000, seed=42)",
            "real_S * 1000",
        ],
        "#36": [
            "pascucci_data.prepare_H(source_path, n, mul_factor)",
            "simulate_ou_day_night(H0=real_H[0], dt=0.5, Nsim=10000, seed=42)",
            "real_H",
        ],
        "#37": [
            "application_metrics.controlled_cost_J_trajectory",
            "to_ema.raw_uncontrolled_J_paths",
            "to_ema.simulate_uncontrolled_J(dt=0.1, Nsim=10000, seed=42)",
        ],
        "#38": ["application_metrics.controlled_alpha", "application_metrics.uncontrolled_alpha"],
        "#39": [
            "stitched.t",
            "exp(stitched.X[:, :, S])",
            "stitched.X[:, :, H,V,X]",
            "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
        ],
        "#40": [
            "application_metrics.controlled_cost_J_total",
            "application_metrics.uncontrolled_cost_J_total"
            if legacy_artifacts
            else "to_ema.raw_uncontrolled_J_paths[:, -1]",
            "uncontrolled baseline source=alpha_zero_same_model"
            if legacy_artifacts
            else "paired alpha=0 total still saved as application_metrics.uncontrolled_cost_J_total",
        ],
    }
    if legacy_artifacts:
        plot_inputs["#37"] = [
            "application_metrics.controlled_cost_J_running_cumulative",
            "application_metrics.uncontrolled_cost_J_running_cumulative",
        ]
    plot_styles = {
        "#35": {
            "source": "raw/to_ema/variable_mu_calibration.ipynb::generate_plot",
            "formula_source": "raw/to_ema/calibration.py",
            "figure_size": [12, 6],
            "real_color": "red",
            "band_alpha": 0.3,
            "band_label": "80% Band",
            "quantiles": [0.05, 0.95],
            "xlabel": "Hours",
        },
        "#36": {
            "source": "raw/to_ema/variable_mu_calibration.ipynb::generate_plot",
            "formula_source": "raw/to_ema/calibration.py",
            "figure_size": [12, 6],
            "real_color": "red",
            "band_alpha": 0.3,
            "band_label": "80% Band",
            "quantiles": [0.05, 0.95],
            "xlabel": "Hours",
        },
        "#37": {
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
        "#38": {
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
        "#39": {
            "source": "raw/to_ema/plotmaker.ipynb::forward_components",
            "figure_size": [14, 10],
            "layout": "2x2",
            "band_alpha": 0.2,
            "quantiles": [0.10, 0.90],
            "sample_paths": 3,
            "sample_policy": "deterministic_first3_for_reproducibility",
        },
        "#40": {
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
    }
    plot_dimensions = {
        "#35": (1920, 960),
        "#36": (1920, 960),
        "#37": (1600, 960),
        "#38": (1600, 800),
        "#39": (2240, 1600),
        "#40": (1280, 800),
    }
    plots = {}
    for story, filename in zip(("#35", "#36", "#37", "#38", "#39", "#40"), (
        "pascucci_paper_35_S_ou_band.png",
        "pascucci_paper_36_H_ou_band.png",
        "pascucci_paper_37_accumulated_cost.png",
        "pascucci_paper_38_alpha.png",
        "pascucci_paper_39_forward_components_S_H_V_X.png",
        "pascucci_paper_40_controlled_uncontrolled.png",
    )):
        width, height = plot_dimensions[story]
        _write_minimal_png(plot_dir / filename, width, height)
        plots[story] = {
            "filename": filename,
            "path": filename,
            "path_relative_to": "manifest_dir",
            "title": f"fixture {story}",
            "inputs": plot_inputs[story],
        }
        if story in plot_styles:
            plots[story]["style"] = plot_styles[story]

    plotmaker_native_plots = {
        "Y": {
            "filename": "pascucci_plotmaker_backward_Y.png",
            "path": "pascucci_plotmaker_backward_Y.png",
            "path_relative_to": "manifest_dir",
            "title": "Backward Component Y",
            "inputs": [
                "stitched.t",
                "stitched.Y[:, :, 0]",
                "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
            ],
            "style": {
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
        },
        "Z": {
            "filename": "pascucci_plotmaker_Z_components.png",
            "path": "pascucci_plotmaker_Z_components.png",
            "path_relative_to": "manifest_dir",
            "title": "Z components",
            "inputs": [
                "stitched.t",
                "stitched.Z[:, :, Z_S,Z_H,Z_V,Z_X]",
                "plotmaker-style q10-q90 bands, mean, and 3 sample paths",
            ],
            "style": {
                "source": "raw/to_ema/plotmaker.ipynb::Z components cell",
                "figure_size": [14, 10],
                "layout": "2x2",
                "labels": ["Z_S", "Z_H", "Z_V", "Z_X"],
                "band_alpha": 0.2,
                "quantiles": [0.10, 0.90],
                "sample_paths": 3,
                "sample_policy": "deterministic_first3_for_reproducibility",
            },
        },
    }
    native_plot_dimensions = {"Y": (1600, 960), "Z": (2240, 1600)}
    for story, entry in plotmaker_native_plots.items():
        width, height = native_plot_dimensions[story]
        _write_minimal_png(plot_dir / entry["filename"], width, height)

    def _axis_stats(values: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, np.ndarray]:
        arr = np.asarray(values, dtype=np.float32)
        stats = {"mean": np.mean(arr, axis=0).astype(np.float32)}
        for quantile in quantiles:
            stats[f"q{int(round(float(quantile) * 100)):02d}"] = np.quantile(
                arr,
                float(quantile),
                axis=0,
            ).astype(np.float32)
        return stats

    time = np.asarray(fixture["stitched"]["t"][0, :, 0], dtype=np.float32)
    plot_data_payload: dict[str, np.ndarray] = {}
    if legacy_artifacts:
        for prefix in ("plot35", "plot36"):
            plot_data_payload[f"{prefix}_time_sim"] = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_sim_mean"] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_sim_q05"] = np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_sim_q50"] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_sim_q95"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_time_real"] = np.asarray([0.0, 1.0], dtype=np.float32)
            plot_data_payload[f"{prefix}_real"] = np.asarray([0.0, 0.0], dtype=np.float32)
    else:
        ou_references, ou_summary = _load_to_ema_ou_reference(params, horizon_T=float(time[-1]))
        dt_sim = float(ou_summary["simulation"]["dt_sim"])
        dt_real = float(ou_summary["dt_real"])
        for story, prefix in (("#35", "plot35"), ("#36", "plot36")):
            ref = ou_references[story]
            paths = np.asarray(ref["paths"], dtype=np.float32)
            real = np.asarray(ref["real_path"], dtype=np.float32)
            stats = _axis_stats(paths, (0.05, 0.50, 0.95))
            real_points = max(1, min(int(float(ref["T"]) / dt_real), int(real.shape[0])))
            plot_data_payload[f"{prefix}_time_sim"] = (
                np.arange(paths.shape[1], dtype=np.float32) * np.float32(dt_sim)
            ).astype(np.float32)
            plot_data_payload[f"{prefix}_sim_mean"] = stats["mean"]
            plot_data_payload[f"{prefix}_sim_q05"] = stats["q05"]
            plot_data_payload[f"{prefix}_sim_q50"] = stats["q50"]
            plot_data_payload[f"{prefix}_sim_q95"] = stats["q95"]
            plot_data_payload[f"{prefix}_time_real"] = (
                np.arange(real.shape[0], dtype=np.float32) * np.float32(dt_real)
            )[:real_points].astype(np.float32)
            plot_data_payload[f"{prefix}_real"] = real[:real_points].astype(np.float32)

    controlled_trace_for_plot = application_pathwise.get(
        "controlled_cost_J_trajectory",
        application_pathwise["controlled_cost_J_running_cumulative"],
    )
    controlled_trace_time = time if "controlled_cost_J_trajectory" in application_pathwise else time[1:]
    controlled_J_stats = _axis_stats(np.asarray(controlled_trace_for_plot, dtype=np.float32)[:, :, 0], (0.10, 0.90))
    plot_data_payload["plot37_controlled_time"] = controlled_trace_time.astype(np.float32)
    plot_data_payload["plot37_controlled_mean"] = controlled_J_stats["mean"]
    plot_data_payload["plot37_controlled_q10"] = controlled_J_stats["q10"]
    plot_data_payload["plot37_controlled_q90"] = controlled_J_stats["q90"]

    if legacy_artifacts:
        uncontrolled_trace_for_plot = application_pathwise["uncontrolled_cost_J_running_cumulative"]
        uncontrolled_trace_time = time[1:]
        uncontrolled_total_for_plot = application_pathwise["uncontrolled_cost_J_total"]
    else:
        to_ema_uncontrolled, _ = _load_to_ema_uncontrolled_reference(params, horizon_T=float(time[-1]))
        uncontrolled_trace_for_plot = to_ema_uncontrolled["J_paths"][:, :, None]
        uncontrolled_trace_time = to_ema_uncontrolled["time"]
        uncontrolled_total_for_plot = to_ema_uncontrolled["J_paths"][:, -1:]
    uncontrolled_J_stats = _axis_stats(
        np.asarray(uncontrolled_trace_for_plot, dtype=np.float32)[:, :, 0],
        (0.10, 0.90),
    )
    plot_data_payload["plot37_uncontrolled_time"] = np.asarray(uncontrolled_trace_time, dtype=np.float32)
    plot_data_payload["plot37_uncontrolled_mean"] = uncontrolled_J_stats["mean"]
    plot_data_payload["plot37_uncontrolled_q10"] = uncontrolled_J_stats["q10"]
    plot_data_payload["plot37_uncontrolled_q90"] = uncontrolled_J_stats["q90"]

    controlled_alpha_stats = _axis_stats(application_pathwise["controlled_alpha"][:, :, 0], (0.05, 0.50, 0.95))
    plot_data_payload["plot38_time"] = time[:-1].astype(np.float32)
    plot_data_payload["plot38_controlled_mean"] = controlled_alpha_stats["mean"]
    plot_data_payload["plot38_controlled_q05"] = controlled_alpha_stats["q05"]
    plot_data_payload["plot38_controlled_q50"] = controlled_alpha_stats["q50"]
    plot_data_payload["plot38_controlled_q95"] = controlled_alpha_stats["q95"]
    plot_data_payload["plot38_uncontrolled_mean"] = np.mean(
        application_pathwise["uncontrolled_alpha"][:, :, 0],
        axis=0,
    ).astype(np.float32)

    plot_data_payload["plot39_time"] = time.astype(np.float32)
    plot_data_payload["plot39_sample_indices"] = np.asarray([0, 1, 2], dtype=np.float32)
    state_values = {
        "plot39_S": np.exp(np.clip(fixture["stitched"]["X"][:, :, 0], -50.0, 50.0)).astype(np.float32),
        "plot39_H": fixture["stitched"]["X"][:, :, 1].astype(np.float32),
        "plot39_V": fixture["stitched"]["X"][:, :, 2].astype(np.float32),
        "plot39_X": fixture["stitched"]["X"][:, :, 3].astype(np.float32),
    }
    for prefix, values in state_values.items():
        stats = _axis_stats(values, (0.10, 0.50, 0.90))
        plot_data_payload[f"{prefix}_mean"] = stats["mean"]
        plot_data_payload[f"{prefix}_q10"] = stats["q10"]
        plot_data_payload[f"{prefix}_q50"] = stats["q50"]
        plot_data_payload[f"{prefix}_q90"] = stats["q90"]
        plot_data_payload[f"{prefix}_samples"] = values[[0, 1, 2]].astype(np.float32)

    controlled_total_stats = _axis_stats(application_pathwise["controlled_cost_J_total"].reshape(-1, 1), (0.05, 0.50, 0.95))
    uncontrolled_total_stats = _axis_stats(np.asarray(uncontrolled_total_for_plot, dtype=np.float32).reshape(-1, 1), (0.05, 0.50, 0.95))
    plot_data_payload["plot40_controlled_mean"] = controlled_total_stats["mean"]
    plot_data_payload["plot40_controlled_q05"] = controlled_total_stats["q05"]
    plot_data_payload["plot40_controlled_q50"] = controlled_total_stats["q50"]
    plot_data_payload["plot40_controlled_q95"] = controlled_total_stats["q95"]
    plot_data_payload["plot40_uncontrolled_mean"] = uncontrolled_total_stats["mean"]
    plot_data_payload["plot40_uncontrolled_q05"] = uncontrolled_total_stats["q05"]
    plot_data_payload["plot40_uncontrolled_q50"] = uncontrolled_total_stats["q50"]
    plot_data_payload["plot40_uncontrolled_q95"] = uncontrolled_total_stats["q95"]
    native_sample_idx = np.asarray([0, 1, 2], dtype=np.float32)
    Y_values = fixture["stitched"]["Y"][:, :, 0].astype(np.float32)
    Y_stats = _axis_stats(Y_values, (0.10, 0.50, 0.90))
    plot_data_payload["plotmaker_Y_time"] = time.astype(np.float32)
    plot_data_payload["plotmaker_Y_sample_indices"] = native_sample_idx
    plot_data_payload["plotmaker_Y_mean"] = Y_stats["mean"]
    plot_data_payload["plotmaker_Y_q10"] = Y_stats["q10"]
    plot_data_payload["plotmaker_Y_q50"] = Y_stats["q50"]
    plot_data_payload["plotmaker_Y_q90"] = Y_stats["q90"]
    plot_data_payload["plotmaker_Y_samples"] = Y_values[[0, 1, 2]].astype(np.float32)
    for component, label in enumerate(("Z_S", "Z_H", "Z_V", "Z_X")):
        Z_values = fixture["stitched"]["Z"][:, :, component].astype(np.float32)
        Z_stats = _axis_stats(Z_values, (0.10, 0.50, 0.90))
        plot_data_payload[f"plotmaker_Z_{label}_mean"] = Z_stats["mean"]
        plot_data_payload[f"plotmaker_Z_{label}_q10"] = Z_stats["q10"]
        plot_data_payload[f"plotmaker_Z_{label}_q50"] = Z_stats["q50"]
        plot_data_payload[f"plotmaker_Z_{label}_q90"] = Z_stats["q90"]
        plot_data_payload[f"plotmaker_Z_{label}_samples"] = Z_values[[0, 1, 2]].astype(np.float32)
    plot_data_payload["plotmaker_Z_time"] = time.astype(np.float32)
    plot_data_payload["plotmaker_Z_sample_indices"] = native_sample_idx
    save_blob_npz(plot_data_payload, str(plot_dir / "pascucci_paper_plot_data.npz"))

    plotmaker_reference = {
        "source": "raw/to_ema/plotmaker.ipynb cells 8, 16, 18",
        "data": "2025dicembre1",
        "T": 24.0,
        "N": 150,
        "M": 10000,
        "calibration_K": 2,
        "calibration_dt": 1.0,
        "H": {
            "expected_basename": "2025dicembre1.csv",
            "expected_sha256": "75004dde0cd982f67c547c241ce704a4fc596380d5bcbca18169d8d6bc4b5c44",
            "actual_basename": H_source.name,
            "actual_sha256": file_sha256(str(H_source)),
            "matched": True,
        },
        "S": {
            "expected_basename": "2025dicembre1.xlsx",
            "expected_sha256": "7526faeaff806a250dd3736bc2eb7e1202c20d4635638c3b371d85c133eed5ef",
            "actual_basename": S_source.name,
            "actual_sha256": file_sha256(str(S_source)),
            "matched": True,
        },
        "dataset_status": "matched",
    }

    save_json(
        {
            "schema": "pascucci_paper_plots_v1",
            "model_name": "pascucci",
            "plot_count": 6,
            "plots": plots,
            "plotmaker_native_plots": plotmaker_native_plots,
            "plot_data": {
                "schema": "pascucci_paper_plot_data_v1",
                "path": "pascucci_paper_plot_data.npz",
                "path_relative_to": "manifest_dir",
                "keys": sorted(plot_data_payload.keys()),
            },
            "cost_trace_source": "cost_J_running_cumulative" if legacy_artifacts else "cost_J_trajectory",
            "uncontrolled_cost_trace_source": (
                "application_metrics.uncontrolled_cost_J_running_cumulative"
                if legacy_artifacts
                else "to_ema_raw_uncontrolled_J"
            ),
            "to_ema_uncontrolled_reference": (
                {"source": "application_metrics_fallback", "reason": "missing_pascucci_calibration_metadata"}
                if legacy_artifacts
                else {
                    "source": "calibration_metadata",
                    "simulation": {"n_sim": 10000, "dt": 0.1, "seed": 42},
                }
            ),
            "comparison_sources": (
                {
                    "#37": {"uncontrolled": "application_metrics.uncontrolled_cost_J_running_cumulative"},
                    "#40": {"uncontrolled": "alpha_zero_same_model"},
                }
                if legacy_artifacts
                else {
                    "#37": {"uncontrolled": "to_ema_raw_uncontrolled_J"},
                    "#40": {
                        "uncontrolled": "to_ema_raw_uncontrolled_J_final",
                        "paired_alpha_zero_detail": "application_metrics.uncontrolled_cost_J_total",
                    },
                }
            ),
            "plotmaker_reference": plotmaker_reference,
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
            "visual_regression": {
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
            },
            "state_transforms": {
                "#35": "raw S" if legacy_artifacts else "PUN price = exp(S) * 1000",
                "#39": "raw S"
                if legacy_artifacts
                else "forward components S,H,V,X in plotmaker 2x2 layout; S is plotted as exp(S)",
            },
            "ou_reference": (
                {"source": "stitched_fallback", "reason": "missing_pascucci_calibration_metadata"}
                if legacy_artifacts
                else {"source": "calibration_metadata", "simulation": {"n_sim": 10000, "dt_sim": 0.5, "seed": 42}}
            ),
        },
        str(plot_dir / "pascucci_paper_plots_manifest.json"),
    )


def test_pascucci_paper_plot_parity_validator_green_contract_and_wrapper_resolution() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        wrapper_dir = Path(tmp) / "pascucci_T24_wrapper_job999999"
        run_root = wrapper_dir / "run_20260613_120000"
        _write_minimal_paper_parity_artifacts(run_root)

        report = validate_pascucci_paper_plot_parity(wrapper_dir)

    assert report["status"] == "GREEN"
    assert report["run_root"].endswith("run_20260613_120000")
    assert report["failures"] == []
    assert report["incomplete"] == []
    assert report["inconclusive"] == []
    assert report["warnings"] == []
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    assert report["failure_counts"] == {
        "contract": 0,
        "science": 0,
        "mc_confirmation": 0,
        "incomplete": 0,
        "inconclusive": 0,
    }
    assert report["failure_categories"] == {"contract": [], "science": [], "mc_confirmation": []}


def test_pascucci_paper_plot_parity_validator_rejects_t24_legacy_semantics() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_legacy"
        _write_minimal_paper_parity_artifacts(run_root, legacy_artifacts=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["failure_counts"]["contract"] > 0
    joined = "\n".join(report["failures"])
    assert "pascucci_calibration missing" in joined
    assert "controlled_cost_J_trajectory" in joined
    assert "cost_trace_source" in joined
    assert "ou_reference.source" in joined
    assert "Xi_initial paper-like distribution mismatch" in joined


def test_pascucci_paper_plot_parity_validator_rejects_non_plotmaker_pascucci_params() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_bad_paper_params"
        _write_minimal_paper_parity_artifacts(run_root)
        cfg_path = run_root / "run_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["params"]["l_a"] = 0.01
        cfg["params"]["c_h"] = 0.0001
        save_json(cfg, str(cfg_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "params.l_a" in joined
    assert "expected plotmaker.ipynb cell 8 value 0.005" in joined
    assert "params.c_h" in joined
    assert "expected plotmaker.ipynb cell 8 value 0.001" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_require_calibration_guard() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_require_calibration"
        _write_minimal_paper_parity_artifacts(run_root)
        cfg_path = run_root / "run_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["pascucci_require_calibration"] = False
        save_json(cfg, str(cfg_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    assert "pascucci_require_calibration must be true" in "\n".join(report["failures"])


def test_pascucci_paper_plot_parity_validator_rejects_xi_initial_quantile_drift() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_bad_xi_quantiles"
        _write_minimal_paper_parity_artifacts(run_root)
        M = 10000
        Xi = np.zeros((M, 4), dtype=np.float32)
        Xi[: M // 2, 0] = -2.5
        Xi[M // 2 :, 0] = -2.1
        Xi[: M // 2, 1] = -0.1
        Xi[M // 2 :, 1] = 0.9
        Xi[: M // 2, 2] = -1.0
        Xi[M // 2 :, 2] = 1.0
        Xi[: M // 2, 3] = 3.0
        Xi[M // 2 :, 3] = 7.0
        np.savez_compressed(run_root / "recursive" / "evaluation_bundle.npz", Xi_initial=Xi)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "Xi_initial paper-like distribution mismatch: Q q05" in joined
    assert "Xi_initial paper-like distribution mismatch: Q q95" in joined


def test_pascucci_paper_plot_parity_validator_rejects_science_red_results() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_science_red"
        _write_minimal_paper_parity_artifacts(run_root, science_red=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "FAILED"
    assert report["mc_confirmation_status"] == "GREEN"
    assert report["failure_counts"]["science"] >= 3
    joined = "\n".join(report["failures"])
    assert "science gate" in joined
    assert "controlled cost is worse" in joined
    assert "win-rate too low" in joined
    assert "physical violation rate too high" in joined


def test_pascucci_paper_plot_parity_validator_rejects_cost_accounting_mismatch() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_bad_cost_accounting"
        _write_minimal_paper_parity_artifacts(run_root, broken_cost_accounting=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "application cost accounting mismatch" in joined
    assert "controlled_cost_J_trajectory_math[-1]" in joined
    assert "controlled_cost_J_total" in joined


def test_pascucci_paper_plot_parity_validator_rejects_alpha_zero_as_plot40_paper_benchmark() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_alpha_zero_plot40"
        _write_minimal_paper_parity_artifacts(run_root)
        manifest_path = run_root / "recursive" / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["comparison_sources"]["#40"] = {"uncontrolled": "alpha_zero_same_model"}
        manifest["plots"]["#40"]["inputs"] = [
            "application_metrics.controlled_cost_J_total",
            "application_metrics.uncontrolled_cost_J_total",
            "uncontrolled baseline source=alpha_zero_same_model",
        ]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "comparison_sources.#40.uncontrolled" in joined
    assert "to_ema_raw_uncontrolled_J_final" in joined
    assert "plot manifest #40 inputs must use final raw to_ema uncontrolled J paths" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_plot_style_contract() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_plot_style"
        _write_minimal_paper_parity_artifacts(run_root)
        manifest_path = run_root / "recursive" / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["plots"]["#37"].pop("style", None)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    assert "plot manifest #37.style missing" in "\n".join(report["failures"])


def test_pascucci_paper_plot_parity_validator_rejects_plot_dimension_drift() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_plot_dimension_drift"
        _write_minimal_paper_parity_artifacts(run_root)
        plot_path = (
            run_root
            / "recursive"
            / "plots"
            / "pascucci_paper"
            / "pascucci_paper_39_forward_components_S_H_V_X.png"
        )
        _write_minimal_png(plot_path, 1600, 800)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "pascucci_paper_39_forward_components_S_H_V_X.png dimensions=(1600, 800)" in joined
    assert "expected (2240, 1600)" in joined


def test_pascucci_paper_plot_parity_validator_rejects_unsupported_pixel_exact_claim() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_unsupported_pixel_exact"
        _write_minimal_paper_parity_artifacts(run_root)
        manifest_path = run_root / "recursive" / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["visual_regression"]["pixel_exact_claim"] = True
        manifest["visual_regression"]["golden_images_available"] = True
        manifest["visual_regression"]["remaining_gaps"] = []
        save_json(manifest, str(manifest_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "visual_regression.pixel_exact_claim" in joined
    assert "visual_regression.golden_images_available" in joined
    assert "remaining_gaps" in joined


def _write_pixel_hash_visual_regression_contract(run_root: Path, *, corrupt_key: str = "") -> None:
    from .io_utils import save_json

    manifest_path = run_root / "recursive" / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot_dir = manifest_path.parent
    golden_dir = plot_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_images = {}
    entries = {story: entry for story, entry in manifest["plots"].items()}
    entries.update({story: entry for story, entry in manifest["plotmaker_native_plots"].items()})
    for story, entry in sorted(entries.items()):
        current_rel = str(entry["path"])
        current_path = plot_dir / current_rel
        golden_rel = f"golden/{str(story).replace('#', 'plot')}_{Path(current_rel).name}"
        golden_path = plot_dir / golden_rel
        payload = current_path.read_bytes()
        if story == corrupt_key:
            payload = b"corrupt-golden" + payload
        golden_path.write_bytes(payload)
        golden_images[story] = {
            "current_path": current_rel,
            "golden_path": golden_rel,
            "sha256": file_sha256(str(current_path)),
        }
    manifest["visual_regression"] = {
        "status": "pixel_hash_exact",
        "structural_style_contract": True,
        "pixel_exact_claim": True,
        "golden_images_available": True,
        "comparison_method": "sha256_png_pixel_exact",
        "golden_images": golden_images,
    }
    save_json(manifest, str(manifest_path))


def test_pascucci_paper_plot_parity_validator_accepts_pixel_hash_golden_contract() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_pixel_hash_golden"
        _write_minimal_paper_parity_artifacts(run_root)
        _write_pixel_hash_visual_regression_contract(run_root)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    assert not any("visual_regression" in failure for failure in report["failures"])


def test_pascucci_paper_plot_parity_validator_rejects_pixel_hash_golden_mismatch() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_pixel_hash_mismatch"
        _write_minimal_paper_parity_artifacts(run_root)
        _write_pixel_hash_visual_regression_contract(run_root, corrupt_key="#37")

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "visual_regression.golden_images.#37 golden sha256" in joined


def test_pascucci_paper_plot_parity_validator_rejects_nonpaper_cost_profile() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_nonpaper_cost_profile"
        _write_minimal_paper_parity_artifacts(run_root)
        run_config_path = run_root / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["params"]["pascucci_cost_profile"] = "exp_minus_offset"
        run_config["params"]["pascucci_cost_offset"] = 1.0
        run_config_path.write_text(json.dumps(run_config, sort_keys=True), encoding="utf-8")

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "pascucci_cost_profile='exp'" in joined
    assert "pascucci_cost_offset=0.0" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_plot_data_npz() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_plot_data"
        _write_minimal_paper_parity_artifacts(run_root)
        plot_data_path = (
            run_root
            / "recursive"
            / "plots"
            / "pascucci_paper"
            / "pascucci_paper_plot_data.npz"
        )
        plot_data_path.unlink()

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "pascucci_paper_plot_data.npz" in joined
    assert "missing artifact" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_yz() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_plotmaker_native_yz"
        _write_minimal_paper_parity_artifacts(run_root)
        manifest_path = run_root / "recursive" / "plots" / "pascucci_paper" / "pascucci_paper_plots_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["plotmaker_native_plots"].pop("Y", None)
        save_json(manifest, str(manifest_path))
        (manifest_path.parent / "pascucci_plotmaker_backward_Y.png").unlink()

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "plotmaker_native_plots.Y" in joined
    assert "native_plotmaker_plots" not in joined or "Y" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_plot_data() -> None:
    from .io_utils import save_blob_npz
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_plotmaker_native_plot_data"
        _write_minimal_paper_parity_artifacts(run_root)
        plot_data_path = (
            run_root
            / "recursive"
            / "plots"
            / "pascucci_paper"
            / "pascucci_paper_plot_data.npz"
        )
        with np.load(plot_data_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files if key != "plotmaker_Y_mean"}
        save_blob_npz(payload, str(plot_data_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "plot data NPZ missing key plotmaker_Y_mean" in joined


def test_pascucci_paper_plot_parity_validator_rejects_wrong_plotmaker_dataset() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_wrong_plotmaker_dataset"
        _write_minimal_paper_parity_artifacts(run_root)
        H_source = run_root / "calibration_sources" / "dataset" / "casa" / "2025dicembre1.csv"
        wrong_H_source = H_source.with_name("2025dicembre2.csv")
        shutil.copyfile(H_source, wrong_H_source)
        config_path = run_root / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["params"]["pascucci_calibration"]["H_metadata"]["source_path"] = str(wrong_H_source)
        save_json(config, str(config_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "H_metadata.source_path basename" in joined
    assert "2025dicembre1.csv" in joined
    assert "2025dicembre2.csv" in joined


def test_pascucci_paper_plot_parity_validator_rejects_plot_data_stat_drift() -> None:
    from .io_utils import save_blob_npz
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_plot_data_drift"
        _write_minimal_paper_parity_artifacts(run_root)
        plot_data_path = (
            run_root
            / "recursive"
            / "plots"
            / "pascucci_paper"
            / "pascucci_paper_plot_data.npz"
        )
        with np.load(plot_data_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["plot37_controlled_mean"] = payload["plot37_controlled_mean"].copy()
        payload["plot37_controlled_mean"][0] += np.float32(10.0)
        save_blob_npz(payload, str(plot_data_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "plot37_controlled_mean mismatch" in joined
    assert "paper-plot source" in joined


def test_pascucci_paper_plot_parity_validator_rejects_calibration_source_hash_drift() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_source_hash_drift"
        _write_minimal_paper_parity_artifacts(run_root)
        H_source = run_root / "calibration_sources" / "dataset" / "casa" / "2025dicembre1.csv"
        H_source.write_text("Consumo (W),Produzione (W)\n2000,900\n", encoding="utf-8")

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "source_sha256 mismatch" in joined
    assert "H_metadata" in joined


def test_pascucci_paper_plot_parity_validator_rejects_recomputed_calibration_param_drift() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_calibration_param_drift"
        _write_minimal_paper_parity_artifacts(run_root)
        config_path = run_root / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["params"]["params_H"]["a0_day"] = float(config["params"]["params_H"]["a0_day"]) + 0.25
        save_json(config, str(config_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "recomputed Pascucci calibration mismatch" in joined
    assert "params_H" in joined


def test_pascucci_paper_plot_parity_validator_rejects_missing_mc_confirmation() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_missing_mc_confirmation"
        _write_minimal_paper_parity_artifacts(run_root, missing_mc_confirmation=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "FAILED"
    assert report["failure_counts"]["mc_confirmation"] == 1
    joined = "\n".join(report["failures"])
    assert "missing independent MC confirmation" in joined
    assert "application_metrics_mc_confirmation.json" in joined


def test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_schema_drift() -> None:
    from .io_utils import save_json
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_mc_schema_drift"
        _write_minimal_paper_parity_artifacts(run_root)
        summary_path = run_root / "recursive" / "application_metrics_mc_confirmation.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["schema"] = "pascucci_mc_confirmation_v2"
        save_json(payload, str(summary_path))

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "MC confirmation schema" in joined
    assert "pascucci_mc_confirmation_v1" in joined


def test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_not_independent() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_mc_not_independent"
        _write_minimal_paper_parity_artifacts(run_root, mc_not_independent=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "independent_of_eval_bundle=true" in joined
    assert "must not reuse" in joined
    assert "seed must differ" in joined


def test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_sample_count_drift() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_mc_sample_count_drift"
        _write_minimal_paper_parity_artifacts(run_root, mc_paired_count_drift=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "comparison.paired_sample_count mismatch" in joined
    assert "expected NPZ sample_paths=10000" in joined


def test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_science_red() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_mc_science_red"
        _write_minimal_paper_parity_artifacts(run_root, mc_science_red=True)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "FAILED"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "MC confirmation" in joined
    assert "controlled cost is worse" in joined
    assert "controlled win-rate too low" in joined
    assert "physical violation rate too high" in joined


def test_pascucci_paper_plot_parity_validator_mc_violation_rate_uses_positive_indicator() -> None:
    from .pascucci_run_validation import validate_pascucci_paper_plot_parity

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run_20260613_mc_violation_magnitude"
        _write_minimal_paper_parity_artifacts(run_root, mc_violation_magnitude=3.0)

        report = validate_pascucci_paper_plot_parity(run_root)

    assert report["status"] == "GREEN"
    assert report["contract_status"] == "GREEN"
    assert report["science_status"] == "GREEN"
    assert report["mc_confirmation_status"] == "GREEN"
    joined = "\n".join(report["failures"])
    assert "diagnostics.q_lower_violation_rate does not match" not in joined
    assert "diagnostics.v_lower_violation_rate does not match" not in joined


def _application_summary_fixture(
    *,
    controlled_total_mean: float,
    uncontrolled_total_mean: float,
    control_win_rate: float,
    q_lower_rate: float = 0.0,
    q_upper_rate: float = 0.0,
    v_lower_rate: float = 0.0,
    v_upper_rate: float = 0.0,
) -> Dict[str, object]:
    delta = float(controlled_total_mean) - float(uncontrolled_total_mean)
    return {
        "schema": "pascucci_application_metrics_v2",
        "model_name": "pascucci",
        "horizon": {
            "T_total": 12.0,
            "n_steps": 78,
            "n_time_points": 79,
            "sample_paths": 10000,
            "t_start": 0.0,
            "t_end": 12.0,
        },
        "controlled": {
            "summary": {
                "cost_J_total_mean": float(controlled_total_mean),
                "cost_J_total_q05": float(controlled_total_mean) - 1.0,
                "cost_J_total_q50": float(controlled_total_mean),
                "cost_J_total_q95": float(controlled_total_mean) + 1.0,
                "cost_J_running_mean": float(controlled_total_mean),
                "cost_J_terminal_mean": 0.0,
            },
            "alpha_summary": {
                "alpha_mean": 0.0,
                "alpha_std": 0.1,
                "alpha_abs_mean": 0.1,
                "alpha_abs_q95": 0.2,
                "sample_count": 10000,
            },
        },
        "uncontrolled": {
            "summary": {
                "cost_J_total_mean": float(uncontrolled_total_mean),
                "cost_J_total_q05": float(uncontrolled_total_mean) - 1.0,
                "cost_J_total_q50": float(uncontrolled_total_mean),
                "cost_J_total_q95": float(uncontrolled_total_mean) + 1.0,
                "cost_J_running_mean": float(uncontrolled_total_mean),
                "cost_J_terminal_mean": 0.0,
            },
            "alpha_summary": {
                "alpha_mean": 0.0,
                "alpha_std": 0.0,
                "alpha_abs_mean": 0.0,
                "alpha_abs_q95": 0.0,
                "sample_count": 10000,
            },
        },
        "comparison": {
            "delta_cost_J_total_mean": delta,
            "delta_cost_J_total_q05": delta - 1.0,
            "delta_cost_J_total_q50": delta,
            "delta_cost_J_total_q95": delta + 1.0,
            "cost_J_total_control_win_rate": float(control_win_rate),
            "delta_cost_J_running_mean": delta,
            "delta_cost_J_terminal_mean": 0.0,
            "paired_sample_count": 10000,
        },
        "diagnostics": {
            "q_lower_violation_mean": float(q_lower_rate),
            "q_lower_violation_rate": float(q_lower_rate),
            "q_upper_violation_mean": float(q_upper_rate),
            "q_upper_violation_rate": float(q_upper_rate),
            "v_lower_violation_mean": float(v_lower_rate),
            "v_lower_violation_rate": float(v_lower_rate),
            "v_upper_violation_mean": float(v_upper_rate),
            "v_upper_violation_rate": float(v_upper_rate),
            "stitch_Y_boundary_mean_abs_jump": 0.01,
            "stitch_Y_boundary_max_abs_jump": 0.1,
            "stitch_Z_boundary_mean_abs_jump": 0.001,
            "stitch_Z_boundary_max_abs_jump": 0.01,
        },
        "pathwise_npz_keys": [],
    }


def _write_minimal_t12_gate_artifacts(run_root: Path, *, dominated_final_pass: bool = False) -> None:
    rec_dir = run_root / "recursive"
    plot_dir = rec_dir / "plots" / "pascucci_paper"
    plot_dir.mkdir(parents=True)
    (run_root / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": "pascucci",
                "M": 10000,
                "N": 13,
                "D": 4,
                "T_total": 12.0,
                "block_size": 2.0,
                "passes": 2,
                "state_labels": ["S", "H", "V", "X_state"],
                "z_labels": ["Z_S", "Z_H", "Z_V", "Z_X"],
                "application_metric_schema": "pascucci_application_metrics_v2",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if dominated_final_pass:
        pass0_per = [0.07] * 6
        pass1_per = [5.5, 0.13, 0.001, 0.001, 0.001, 0.001]
        selected_score = 2.893916666666667
        pass0_metrics = _application_summary_fixture(
            controlled_total_mean=-4.0,
            uncontrolled_total_mean=-2.0,
            control_win_rate=0.70,
            v_lower_rate=0.01,
            v_upper_rate=0.01,
        )
        pass1_metrics = _application_summary_fixture(
            controlled_total_mean=3.6,
            uncontrolled_total_mean=-2.0,
            control_win_rate=0.15,
            q_lower_rate=0.28,
            q_upper_rate=0.06,
            v_lower_rate=0.23,
            v_upper_rate=0.27,
        )
    else:
        pass0_per = [0.10] * 6
        pass1_per = [0.08] * 6
        selected_score = 0.108
        pass0_metrics = _application_summary_fixture(
            controlled_total_mean=-3.0,
            uncontrolled_total_mean=-2.0,
            control_win_rate=0.60,
            v_lower_rate=0.02,
        )
        pass1_metrics = _application_summary_fixture(
            controlled_total_mean=-4.0,
            uncontrolled_total_mean=-2.0,
            control_win_rate=0.70,
            v_lower_rate=0.01,
        )

    (rec_dir / "results.json").write_text(
        json.dumps(
            {
                "blocks": [{"idx": i, "t_start": 2.0 * i, "t_end": 2.0 * (i + 1)} for i in range(6)],
                "passes": [{"pass_id": 1}, {"pass_id": 2}],
                "evaluation_bundle_M": 10000,
                "selected_pass_id": 2,
                "selected_score": selected_score,
                "loss_pass_scores": {
                    "1": float(np.mean(pass0_per) + 0.35 * max(pass0_per)),
                    "2": float(np.mean(pass1_per) + 0.35 * max(pass1_per)),
                },
                "pass_invalid_reasons": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        rec_dir / "evaluation_bundle.npz",
        Xi_initial=np.zeros((10000, 4), dtype=np.float32),
        t_bundle=np.zeros((6, 10000, 14, 1), dtype=np.float32),
        W_bundle=np.zeros((6, 10000, 14, 4), dtype=np.float32),
        block_t_start=np.asarray([0, 2, 4, 6, 8, 10], dtype=np.float32),
        block_t_end=np.asarray([2, 4, 6, 8, 10, 12], dtype=np.float32),
    )

    header = "pass,block,t_start,t_end,eval_mean_loss,eval_mean_loss_per_sample,eval_mean_y0\n"
    pass0_rows = "".join(
        f"1,{idx},{2.0 * idx:.1f},{2.0 * (idx + 1):.1f},{per * 10000.0:.6f},{per:.6f},0.0\n"
        for idx, per in enumerate(pass0_per)
    )
    pass1_rows = "".join(
        f"2,{idx},{2.0 * idx:.1f},{2.0 * (idx + 1):.1f},{per * 10000.0:.6f},{per:.6f},0.0\n"
        for idx, per in enumerate(pass1_per)
    )
    (rec_dir / "pass_00_logs.csv").write_text(header + pass0_rows, encoding="utf-8")
    (rec_dir / "pass_01_logs.csv").write_text(header + pass1_rows, encoding="utf-8")

    stitched = {
        "t": np.linspace(0.0, 12.0, 7, dtype=np.float32).reshape(1, 7, 1),
        "X": np.zeros((1, 7, 4), dtype=np.float32),
        "Y": np.zeros((1, 7, 1), dtype=np.float32),
        "Z": np.zeros((1, 7, 4), dtype=np.float32),
    }
    application = {
        "running_cost": np.zeros((1, 6), dtype=np.float32),
        "terminal_cost": np.zeros((1,), dtype=np.float32),
        "cumulative_cost": np.zeros((1, 7), dtype=np.float32),
    }
    for stem in ("stitched_predictions_pass00", "stitched_predictions_pass01", "stitched_predictions_final"):
        np.savez_compressed(rec_dir / f"{stem}.npz", **stitched)
    application_summaries = {
        "application_metrics_pass00": pass0_metrics,
        "application_metrics_pass01": pass1_metrics,
        "application_metrics_final": {
            **pass1_metrics,
            "selected_from_pass_id": 2,
            "selected_from_pass_index": 1,
        },
    }
    for stem, summary in application_summaries.items():
        (rec_dir / f"{stem}.json").write_text(
            json.dumps(summary, sort_keys=True),
            encoding="utf-8",
        )
        np.savez_compressed(rec_dir / f"{stem}.npz", **application)

    (plot_dir / "pascucci_paper_plots_manifest.json").write_text(
        json.dumps({"schema": "pascucci_paper_plots_v1", "plots": []}, sort_keys=True),
        encoding="utf-8",
    )
    for filename in (
        "pascucci_paper_35_S_ou_band.png",
        "pascucci_paper_36_H_ou_band.png",
        "pascucci_paper_37_accumulated_cost.png",
        "pascucci_paper_38_alpha.png",
        "pascucci_paper_39_forward_components_S_H_V_X.png",
        "pascucci_paper_40_controlled_uncontrolled.png",
    ):
        (plot_dir / filename).write_bytes(b"0" * 1001)


def test_pascucci_t12_gate_validator_green_contract_and_numeric_scalars() -> None:
    from .pascucci_run_validation import _is_finite_scalar, validate_t12_gate_n13

    assert _is_finite_scalar(np.float64(0.25))
    assert _is_finite_scalar(np.int64(2))
    assert not _is_finite_scalar(np.float64(np.nan))
    assert not _is_finite_scalar(True)

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_t12_gate_artifacts(run_root)

        report = validate_t12_gate_n13(run_root)

    assert report["status"] == "GREEN"
    assert report["failures"] == []
    assert report["incomplete"] == []
    assert report["inconclusive"] == []


def test_pascucci_t12_gate_validator_rejects_science_red_selected_pass() -> None:
    from .pascucci_run_validation import validate_t12_gate_n13

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_t12_gate_artifacts(run_root, dominated_final_pass=True)

        report = validate_t12_gate_n13(run_root)

    assert report["status"] == "FAILED"
    joined = "\n".join(report["failures"])
    assert "science" in joined.lower()
    assert "selected pass" in joined.lower()
    assert "pass0" in joined.lower()


def test_pascucci_t12_gate_validator_reports_malformed_json_without_exception() -> None:
    from .pascucci_run_validation import validate_t12_gate_n13

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_t12_gate_artifacts(run_root)
        (run_root / "recursive" / "results.json").write_text("{", encoding="utf-8")

        report = validate_t12_gate_n13(run_root)

    assert report["status"] == "FAILED"
    assert any("malformed JSON" in failure for failure in report["failures"])


def test_print_recursive_pass_rejects_mismatched_eval_bundle_metadata() -> None:
    from . import orchestration
    from .model_specs import get_model_spec
    from .sampling import save_evaluation_bundle

    spec = get_model_spec("pascucci")
    params = spec.build_default_params(const=0.75)
    layers = [5, 8, 1]
    D = 4
    saved_blocks = build_blocks(T_total=0.5, block_size=0.25)
    shifted_blocks = [
        {
            "idx": int(block["idx"]),
            "t_start": float(block["t_start"]) + 0.125,
            "t_end": float(block["t_end"]) + 0.125,
            "T_block": float(block["T_block"]),
        }
        for block in saved_blocks
    ]
    logs = [
        {
            "pass": 1,
            "block": int(block["idx"]),
            "t_start": float(block["t_start"]),
            "t_end": float(block["t_end"]),
            "T_block": float(block["T_block"]),
            "eval_mean_loss": 1.0,
            "eval_std_loss": 0.0,
            "eval_mean_loss_per_sample": 0.25,
            "eval_std_loss_per_sample": 0.0,
            "eval_mean_y0": 0.0,
            "precision_target": None,
            "refine_rounds": 0,
        }
        for block in shifted_blocks
    ]

    predict_calls = []

    def fake_predict_recursive_stitched(**kwargs):
        predict_calls.append(kwargs)
        M = int(kwargs["Xi_initial"].shape[0])
        steps = len(shifted_blocks) * 2 + 1
        t = np.tile(
            np.linspace(0.125, 0.625, steps, dtype=np.float32).reshape(1, steps, 1),
            (M, 1, 1),
        )
        return {
            "t": t,
            "X": np.zeros((M, steps, D), dtype=np.float32),
            "Y": np.zeros((M, steps, 1), dtype=np.float32),
            "Z": np.zeros((M, steps, D), dtype=np.float32),
        }

    original_predict = orchestration.predict_recursive_stitched
    original_plot_logs = orchestration.plot_recursive_pass_logs_multi
    original_plot_stitched = orchestration.plot_recursive_stitched_predictions
    original_plot_convergence = orchestration.plot_recursive_stitched_y_convergence
    orchestration.predict_recursive_stitched = fake_predict_recursive_stitched
    orchestration.plot_recursive_pass_logs_multi = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_predictions = lambda *args, **kwargs: None
    orchestration.plot_recursive_stitched_y_convergence = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp)
            bundle_path = rec_dir / "evaluation_bundle.npz"
            Xi = spec.deterministic_xi(4, D, seed=87)
            rollout = build_stitched_rollout_inputs(
                blocks=saved_blocks,
                M=Xi.shape[0],
                N_per_block=2,
                D=D,
                seed=87,
            )
            save_evaluation_bundle(
                path=str(bundle_path),
                Xi_initial=Xi,
                rollout_inputs=rollout,
                blocks=saved_blocks,
            )

            try:
                orchestration.print_recursive_pass(
                    pass_entries=[
                        {
                            "pass_id": 1,
                            "logs": logs,
                            "blobs": [
                                _make_block_blob(layers, block["t_start"], block["t_end"], 0.625)
                                for block in shifted_blocks
                            ],
                        }
                    ],
                    blocks=shifted_blocks,
                    rec_dir=str(rec_dir),
                    params=params,
                    N_per_block=2,
                    D=D,
                    layers=layers,
                    T_total=0.625,
                    exact_solution=None,
                    selection_metric="loss",
                    eval_bundle_path=str(bundle_path),
                    eval_seed=87,
                    eval_min_paths=4,
                    sample_paths=0,
                    print_compact_logs=False,
                    model_spec=spec,
                )
            except ValueError as exc:
                message = str(exc).lower()
                assert "evaluation bundle" in message
                assert "block" in message or "t_total" in message or "metadata" in message
            else:
                raise AssertionError("print_recursive_pass should reject a stale evaluation bundle")
            assert predict_calls == []
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


def test_recursive_coarse_prepass_model_spec_argument() -> None:
    from . import orchestration
    from .model_specs import get_model_spec

    spec = get_model_spec()
    calls = []
    original_run_recursive_training = orchestration.run_recursive_training

    def fake_run_recursive_training(**kwargs):
        calls.append(kwargs)
        assert kwargs["model_spec"] is spec
        assert kwargs["D"] == 4
        return {
            "boundary_samples": [
                np.zeros((2, 4), dtype=np.float32),
                np.ones((2, 4), dtype=np.float32),
            ],
            "pass1": {
                "blobs": [_make_blob([5, 8, 1])],
                "logs": [{"eval_mean_loss_per_sample": 1.0}],
                "reference_loss": 1.0,
            },
        }

    orchestration.run_recursive_training = fake_run_recursive_training
    try:
        result = orchestration.run_recursive_coarse_prepass(
            Xi_generator=spec.xi_generator,
            params=_default_params(),
            M=4,
            N_per_block=2,
            D=4,
            T_total=0.25,
            block_size=0.25,
            layers=[5, 8, 1],
            stage_plan=[(1, 1.0e-3)],
            final_plan=[(1, 1.0e-4)],
            output_dir="unused",
            prepass_M=4,
            prepass_N=2,
            rollout_M=4,
            curriculum_consts=[],
            curriculum_stage_scales=[],
            coupling_const=1.0,
            model_spec=spec,
        )
    finally:
        orchestration.run_recursive_training = original_run_recursive_training

    assert len(calls) == 1
    assert result["summary"]["n_curriculum_stages"] == 1
    assert result["boundary_samples"][0].shape == (2, 4)


def test_cli_coarse_boundary_only_and_blob_norm_flags_wire_to_recursive() -> None:
    import json

    from . import cli, orchestration
    from .sampling import build_blocks, file_sha256
    from .tf_backend import require_tensorflow

    require_tensorflow()

    coarse_calls = []
    recursive_calls = []
    print_calls = []

    def fake_run_recursive_coarse_prepass(**kwargs):
        coarse_calls.append(kwargs)
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        return {
            "boundary_samples": [
                np.full((int(kwargs["rollout_M"]), int(kwargs["D"])), np.float32(idx), dtype=np.float32)
                for idx in range(len(blocks) + 1)
            ],
            "pass1_blobs": [
                _make_block_blob(kwargs["layers"], block["t_start"], block["t_end"], kwargs["T_total"])
                for block in blocks
            ],
            "summary": {
                "M": int(kwargs["prepass_M"]),
                "N": int(kwargs["prepass_N"]),
                "iter_scale": float(kwargs["iter_scale"]),
                "rollout_M": int(kwargs["rollout_M"]),
                "curriculum_consts": [float(kwargs["coupling_const"])],
                "curriculum_stage_scales": [1.0],
                "n_curriculum_stages": 1,
                "stage_summaries": [],
            },
        }

    def fake_run_recursive_training(**kwargs):
        recursive_calls.append(kwargs)
        assert kwargs["initial_boundary_samples"] is not None
        assert kwargs["initial_warm_start_blobs"] is None
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        logs = [
            {
                "pass": 1,
                "block": 0,
                "t_start": blocks[0]["t_start"],
                "t_end": blocks[0]["t_end"],
                "T_block": blocks[0]["T_block"],
                "eval_mean_loss": 1.0,
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25,
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
                "warm_start_norm_policy": kwargs["warm_start_norm_policy"],
                "warm_start_blob_used": False,
            }
        ]
        return {
            "blocks": blocks,
            "passes": [
                {
                    "pass_id": 1,
                    "reference_loss": 1.0,
                    "logs": logs,
                    "blobs": [_make_block_blob(kwargs["layers"], block["t_start"], block["t_end"], kwargs["T_total"]) for block in blocks],
                    "models_dir": kwargs["output_dir"],
                    "pass_init_mode": kwargs["pass1_init_mode"],
                    "boundary_source": "coarse_prepass",
                    "is_bootstrap_pass": False,
                    "active_set_summary": {},
                }
            ],
            "pass1": {
                "logs": logs,
                "reference_loss": 1.0,
                "blobs": [_make_block_blob(kwargs["layers"], block["t_start"], block["t_end"], kwargs["T_total"]) for block in blocks],
            },
            "boundary_samples": [
                np.zeros((kwargs["rollout_M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        print_calls.append(kwargs)
        eval_bundle_path = Path(kwargs["eval_bundle_path"])
        eval_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            eval_bundle_path,
            Xi_initial=np.zeros((4, kwargs["D"]), dtype=np.float32),
        )
        return {
            "processed_pass_ids": [1],
            "exact_summary_by_pass": {},
            "exact_summary_by_pass_index": {},
            "eval_bundle_path": str(eval_bundle_path),
            "eval_bundle_status": "created",
            "eval_bundle_sha256": file_sha256(str(eval_bundle_path)),
            "evaluation_bundle_M": 4,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": 1,
            "selected_pass_index": 0,
            "selected_score_metric": "loss.eval_mean_loss_per_sample",
            "selected_score": 0.25,
            "selected_scores_by_pass": {"1": 0.25},
            "selected_scores_by_pass_index": {"0": 0.25},
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {1: 0.25},
            "pass_scores_loss_by_index": {0: 0.25},
        }

    original_run_recursive_coarse_prepass = orchestration.run_recursive_coarse_prepass
    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    orchestration.run_recursive_coarse_prepass = fake_run_recursive_coarse_prepass
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            exit_code = cli.main(
                [
                    "run",
                    "--mode",
                    "recursive",
                    "--pass1_init",
                    "coarse",
                    "--coarse_disable_weight_warm_start",
                    "--warm_start_norm_policy",
                    "blob",
                    "--coarse_prepass_M",
                    "3",
                    "--coarse_prepass_N",
                    "2",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
            assert exit_code == 0
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            assert config["pass1_init"] == "coarse"
            assert config["coarse_disable_weight_warm_start"] is True
            assert config["warm_start_norm_policy"] == "blob"
            coarse_summary = json.loads(
                (runs[0] / "coarse_prepass" / "summary.json").read_text(encoding="utf-8")
            )
            assert coarse_summary["weight_warm_start_enabled"] is False
            assert coarse_summary["warm_start_norm_policy"] == "blob"
    finally:
        orchestration.run_recursive_coarse_prepass = original_run_recursive_coarse_prepass
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass

    assert len(coarse_calls) == 1
    assert coarse_calls[0]["warm_start_norm_policy"] == "blob"
    assert len(recursive_calls) == 1
    assert recursive_calls[0]["pass1_init_mode"] == "coarse"
    assert recursive_calls[0]["warm_start_norm_policy"] == "blob"
    assert recursive_calls[0]["initial_boundary_samples"] is not None
    assert recursive_calls[0]["initial_warm_start_blobs"] is None
    assert len(print_calls) == 1


def test_cli_rejects_coarse_boundary_only_with_pass1_warm_start_from_next() -> None:
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            cli.main(
                [
                    "run",
                    "--mode",
                    "recursive",
                    "--pass1_init",
                    "coarse",
                    "--coarse_disable_weight_warm_start",
                    "--pass1_warm_start_from_next",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(Path(tmp) / "out"),
                ]
            )
        except ValueError as exc:
            message = str(exc)
            assert "--coarse_disable_weight_warm_start" in message
            assert "--pass1_warm_start_from_next" in message
        else:
            raise AssertionError("boundary-only coarse diagnostic should reject pass1 warm-start from next")


def test_cli_rejects_mc_confirmation_for_quadratic_model() -> None:
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            cli.main(
                [
                    "run",
                    "--mode",
                    "recursive",
                    "--model",
                    "quadratic_coupled",
                    "--mc_confirmation_paths",
                    "4",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(Path(tmp) / "out"),
                ]
            )
        except ValueError as exc:
            message = str(exc)
            assert "--mc_confirmation_paths" in message
            assert "application metrics" in message
            assert "quadratic_coupled" in message
        else:
            raise AssertionError("quadratic_coupled should reject MC confirmation requests")


def test_cli_tiny_standard_and_both_model_spec_outputs() -> None:
    import json

    from . import cli, orchestration
    from .sampling import build_blocks, file_sha256
    from .tf_backend import require_tensorflow

    require_tensorflow()

    class FakeSession:
        def close(self) -> None:
            pass

    class FakeStandardModel:
        def __init__(self) -> None:
            self.sess = FakeSession()

        def save_model(self, path: str) -> None:
            marker = Path(path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("fake checkpoint\n", encoding="utf-8")

        def application_cost_functional(self, t, W, Xi, const_value=None, baseline_mode="controlled"):
            del W, const_value
            mode = str(baseline_mode)
            M = int(np.asarray(Xi).shape[0])
            steps = int(np.asarray(t).shape[1]) - 1
            base = np.float32(1.0 if mode == "controlled" else 2.0)
            running = np.full((M, 1), base, dtype=np.float32)
            terminal = np.full((M, 1), base + np.float32(0.25), dtype=np.float32)
            total = running + terminal
            alpha = np.zeros((M, steps, 1), dtype=np.float32)
            if steps > 0:
                running_cumulative = np.cumsum(
                    np.full((M, steps, 1), base / np.float32(steps), dtype=np.float32),
                    axis=1,
                ).astype(np.float32)
            else:
                running_cumulative = np.zeros((M, 0, 1), dtype=np.float32)
            trajectory = (
                np.concatenate([np.zeros((M, 1, 1), dtype=np.float32), running_cumulative], axis=1)
                + terminal[:, None, :]
            ).astype(np.float32)
            pathwise = {
                "cost_J_running": running,
                "cost_J_terminal": terminal,
                "cost_J_total": total,
                "cost_J_running_cumulative": running_cumulative,
                "cost_J_trajectory": trajectory,
                "cost_J_trajectory_math": trajectory.copy(),
                "alpha": alpha,
            }
            summary = {}
            for metric, values in pathwise.items():
                if not metric.startswith("cost_J_"):
                    continue
                values_np = np.asarray(values, dtype=np.float32)
                if metric in ("cost_J_running_cumulative", "cost_J_trajectory", "cost_J_trajectory_math"):
                    flat = values_np[:, -1, :].reshape(-1) if values_np.shape[1] > 0 else np.zeros((values_np.shape[0],), dtype=np.float32)
                else:
                    flat = values_np.reshape(-1)
                summary[f"{metric}_mean"] = float(flat.mean())
            return {
                "schema": "pascucci_application_metrics_v2",
                "metadata": {
                    "baseline_mode": mode,
                    "aggregation": "left_riemann_f_plus_terminal_g",
                    "trajectory_aggregation": "plotmaker_cumsum_f_dt_plus_g_at_time",
                    "trajectory_math_aggregation": "left_riemann_integral_to_time_plus_g_at_time",
                    "control_law": "alpha_tf" if mode == "controlled" else "alpha_zero",
                    "paired_inputs": "same_t_W_Xi",
                },
                "pathwise": pathwise,
                "summary": summary,
            }

    standard_calls = []
    recursive_calls = []
    print_calls = []
    current_expected_model = {"name": "quadratic_coupled"}

    def fake_run_standard_reference(**kwargs):
        standard_calls.append(kwargs)
        assert kwargs["model_spec"].name == current_expected_model["name"]
        assert kwargs["model_spec"].state_dim == kwargs["D"]
        return FakeStandardModel(), {
            "stage_logs": [
                {
                    "phase": "curriculum",
                    "const": 1.0,
                    "lr": 1.0e-3,
                    "n_iter": 1,
                    "train_last_loss": 1.0,
                    "eval_mean_loss": 1.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.25,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "eval_std_y0": 0.0,
                    "elapsed_sec": 0.0,
                }
            ],
            "eval_stats": {
                "mean_loss": 1.0,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.25,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
        }

    def fake_run_recursive_training(**kwargs):
        recursive_calls.append(kwargs)
        assert kwargs["model_spec"].name == current_expected_model["name"]
        assert kwargs["model_spec"].state_dim == kwargs["D"]
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        logs = [
            {
                "pass": 1,
                "block": 0,
                "t_start": blocks[0]["t_start"],
                "t_end": blocks[0]["t_end"],
                "T_block": blocks[0]["T_block"],
                "eval_mean_loss": 1.0,
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25,
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
            }
        ]
        return {
            "blocks": blocks,
            "passes": [
                {
                    "pass_id": 1,
                    "reference_loss": 1.0,
                    "logs": logs,
                    "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
                    "models_dir": kwargs["output_dir"],
                    "pass_init_mode": kwargs["pass1_init_mode"],
                    "boundary_source": "base_xi",
                    "is_bootstrap_pass": True,
                    "active_set_summary": {},
                }
            ],
            "pass1": {
                "logs": logs,
                "reference_loss": 1.0,
                "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
            },
            "boundary_samples": [
                np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        print_calls.append(kwargs)
        eval_bundle_path = Path(kwargs["eval_bundle_path"])
        eval_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            eval_bundle_path,
            Xi_initial=np.zeros((4, kwargs["D"]), dtype=np.float32),
        )
        return {
            "processed_pass_ids": [1],
            "exact_summary_by_pass": {},
            "exact_summary_by_pass_index": {},
            "eval_bundle_path": str(eval_bundle_path),
            "eval_bundle_status": "created",
            "eval_bundle_sha256": file_sha256(str(eval_bundle_path)),
            "evaluation_bundle_M": 4,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": 1,
            "selected_pass_index": 0,
            "selected_score_metric": "loss.eval_mean_loss_per_sample",
            "selected_score": 0.25,
            "selected_scores_by_pass": {"1": 0.25},
            "selected_scores_by_pass_index": {"0": 0.25},
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {1: 0.25},
            "pass_scores_loss_by_index": {0: 0.25},
        }

    original_run_standard_reference = orchestration.run_standard_reference
    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    original_export_standard_parameter_blob = cli.export_standard_parameter_blob
    original_plot_stage_logs = cli.plot_stage_logs
    orchestration.run_standard_reference = fake_run_standard_reference
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass
    cli.export_standard_parameter_blob = lambda model: _make_blob([5, 8, 1])
    cli.plot_stage_logs = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("standard_default", "standard", [], "quadratic_coupled"),
                ("both_default", "both", [], "quadratic_coupled"),
                ("standard_explicit", "standard", ["--model", "quadratic_coupled"], "quadratic_coupled"),
                ("standard_pascucci", "standard", ["--model", "pascucci"], "pascucci"),
                ("both_pascucci", "both", ["--model", "pascucci"], "pascucci"),
                ("recursive_pascucci", "recursive", ["--model", "pascucci"], "pascucci"),
            ]
            for case_name, mode, model_args, expected_model in cases:
                current_expected_model["name"] = expected_model
                out_dir = root / case_name
                exit_code = cli.main(
                    [
                        "run",
                        "--mode",
                        mode,
                        *model_args,
                        "--M",
                        "4",
                        "--N",
                        "2",
                        "--T_standard",
                        "0.25",
                        "--T_total",
                        "0.25",
                        "--block_size",
                        "0.25",
                        "--passes",
                        "1",
                        "--visual_sample_paths",
                        "1",
                        "--output_dir",
                        str(out_dir),
                    ]
                )
                assert exit_code == 0
                runs = sorted(out_dir.glob("run_*"))
                assert len(runs) == 1
                run_root = runs[0]
                config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
                assert config["mode"] == mode
                assert config["model_requested"] == expected_model
                assert config["model_name"] == expected_model
                assert config["state_labels"] == ["S", "H", "V", "X_state"]
                assert config["z_labels"] == ["Z_S", "Z_H", "Z_V", "Z_X"]
                assert config["M"] == 4
                assert config["N"] == 2
                if expected_model == "quadratic_coupled":
                    assert config["application_metric_schema"] == "none"
                    assert config["application_metric_names"] == []
                    assert config["application_metric_aggregation"] == "none"
                else:
                    assert config["application_metric_schema"] == "pascucci_application_metrics_v2"
                    assert config["application_metric_names"] == [
                        "cost_J_running",
                        "cost_J_terminal",
                        "cost_J_total",
                        "cost_J_running_cumulative",
                        "cost_J_trajectory",
                        "cost_J_trajectory_math",
                    ]
                    assert config["application_metric_aggregation"] == "left_riemann_f_plus_terminal_g"

                if mode in ("standard", "both"):
                    assert (run_root / "standard" / "results.json").exists(), case_name
                    assert (run_root / "standard" / "model_weights.npz").exists(), case_name
                else:
                    assert not (run_root / "standard").exists(), case_name

                if mode == "standard":
                    assert not (run_root / "recursive").exists(), case_name
                else:
                    assert (run_root / "recursive" / "results.json").exists(), case_name
                    assert (run_root / "recursive" / "evaluation_bundle.npz").exists(), case_name

        assert len(standard_calls) == 5
        assert len(recursive_calls) == 3
        for call in recursive_calls:
            assert call["fixed_eval_seed"] is None
            assert call["fixed_boundary_Xi_initial"] is None
            assert call["fixed_boundary_rollout_inputs"] is None
            assert call["return_gradient_diagnostics"] is False
        assert len(print_calls) == 3
    finally:
        orchestration.run_standard_reference = original_run_standard_reference
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass
        cli.export_standard_parameter_blob = original_export_standard_parameter_blob
        cli.plot_stage_logs = original_plot_stage_logs


def test_cli_wires_recursive_diagnostic_flags_and_separate_rollout_bundle() -> None:
    import json

    from . import cli, orchestration
    from .sampling import build_blocks, file_sha256
    from .tf_backend import require_tensorflow

    require_tensorflow()

    recursive_calls = []
    print_calls = []

    def fake_run_recursive_training(**kwargs):
        recursive_calls.append(kwargs)
        assert kwargs["fixed_eval_seed"] == 777
        assert kwargs["return_gradient_diagnostics"] is True
        assert kwargs["fixed_boundary_Xi_initial"] is not None
        assert kwargs["fixed_boundary_Xi_initial"].shape == (2000, kwargs["D"])
        assert kwargs["fixed_boundary_rollout_inputs"] is not None
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        assert len(kwargs["fixed_boundary_rollout_inputs"]) == len(blocks)
        logs = [
            {
                "pass": 1,
                "block": int(block["idx"]),
                "t_start": block["t_start"],
                "t_end": block["t_end"],
                "T_block": block["T_block"],
                "eval_mean_loss": 1.0,
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25,
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
            }
            for block in blocks
        ]
        pass_entry = {
            "pass_id": 1,
            "reference_loss": 1.0,
            "logs": logs,
            "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
            "models_dir": kwargs["output_dir"],
            "pass_init_mode": kwargs["pass1_init_mode"],
            "boundary_source": "base_xi",
            "is_bootstrap_pass": True,
            "active_set_summary": {},
        }
        return {
            "blocks": blocks,
            "passes": [pass_entry],
            "pass1": {
                "logs": logs,
                "reference_loss": 1.0,
                "blobs": pass_entry["blobs"],
            },
            "boundary_samples": [
                np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        print_calls.append(kwargs)
        eval_bundle_path = Path(kwargs["eval_bundle_path"])
        eval_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(eval_bundle_path, Xi_initial=np.zeros((4, kwargs["D"]), dtype=np.float32))
        return {
            "processed_pass_ids": [1],
            "exact_summary_by_pass": {},
            "exact_summary_by_pass_index": {},
            "eval_bundle_path": str(eval_bundle_path),
            "eval_bundle_status": "created",
            "eval_bundle_sha256": file_sha256(str(eval_bundle_path)),
            "evaluation_bundle_M": 4,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": 1,
            "selected_pass_index": 0,
            "selected_score_metric": "loss.eval_mean_loss_per_sample",
            "selected_score": 0.25,
            "selected_scores_by_pass": {"1": 0.25},
            "selected_scores_by_pass_index": {"0": 0.25},
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {1: 0.25},
            "pass_scores_loss_by_index": {0: 0.25},
        }

    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            exit_code = cli.main(
                [
                    "run",
                    "--mode",
                    "recursive",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--recursive_fixed_eval_seed",
                    "777",
                    "--recursive_gradient_diagnostics",
                    "--use_recursive_rollout_bundle",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
            assert exit_code == 0
            assert len(recursive_calls) == 1
            assert len(print_calls) == 1
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            eval_path = Path(config["eval_bundle_path_effective"])
            rollout_path = Path(config["recursive_rollout_bundle_path_effective"])
            assert config["recursive_fixed_eval_seed"] == 777
            assert config["seed_manifest"]["recursive_fixed_eval_seed"] == 777
            assert config["recursive_gradient_diagnostics"] is True
            assert config["use_recursive_rollout_bundle"] is True
            assert "use_eval_bundle_for_recursive_rollout" not in config
            assert eval_path.name == "evaluation_bundle.npz"
            assert rollout_path.name == "boundary_rollout_bundle.npz"
            assert eval_path != rollout_path
            assert eval_path.exists()
            assert rollout_path.exists()
            assert config["eval_bundle_status"] == "created"
            assert isinstance(config["eval_bundle_sha256"], str)
            assert len(config["eval_bundle_sha256"]) == 64
            assert isinstance(config["recursive_rollout_bundle_sha256"], str)
            assert len(config["recursive_rollout_bundle_sha256"]) == 64
            assert config["recursive_rollout_bundle_status"] == "created"
            rec_results = json.loads(
                (runs[0] / "recursive" / "results.json").read_text(encoding="utf-8")
            )
            assert rec_results["evaluation_bundle_status"] == "created"
            assert rec_results["evaluation_bundle_sha256"] == config["eval_bundle_sha256"]
            assert rec_results["recursive_rollout_bundle_status"] == "created"
            assert (
                rec_results["recursive_rollout_bundle_sha256"]
                == config["recursive_rollout_bundle_sha256"]
            )
    finally:
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass


def test_cli_recursive_two_pass_callbacks_finalize_selection_only_at_last_pass() -> None:
    import json

    from . import cli, orchestration
    from .naming import _pass_index
    from .sampling import build_blocks, file_sha256
    from .tf_backend import require_tensorflow

    require_tensorflow()

    print_calls = []

    def make_pass_entry(pass_id: int, blocks: list[dict]) -> dict:
        logs = [
            {
                "pass": pass_id,
                "block": int(block["idx"]),
                "t_start": float(block["t_start"]),
                "t_end": float(block["t_end"]),
                "T_block": float(block["T_block"]),
                "eval_mean_loss": 1.0 / float(pass_id),
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25 / float(pass_id),
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
            }
            for block in blocks
        ]
        return {
            "pass_id": int(pass_id),
            "reference_loss": float(pass_id),
            "logs": logs,
            "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
            "models_dir": "unused",
            "pass_init_mode": "base",
            "boundary_source": "base_xi" if int(pass_id) == 1 else "previous_pass",
            "is_bootstrap_pass": int(pass_id) == 1,
            "active_set_summary": {},
        }

    def fake_run_recursive_training(**kwargs):
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        pass1 = make_pass_entry(1, blocks)
        pass2 = make_pass_entry(2, blocks)
        on_pass_end = kwargs.get("on_pass_end")
        assert on_pass_end is not None
        on_pass_end({"pass_id": 1, "passes": [pass1], "blocks": blocks})
        on_pass_end({"pass_id": 2, "passes": [pass1, pass2], "blocks": blocks})
        return {
            "blocks": blocks,
            "passes": [pass1, pass2],
            "pass1": {
                "logs": pass1["logs"],
                "reference_loss": pass1["reference_loss"],
                "blobs": pass1["blobs"],
            },
            "boundary_samples": [
                np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        print_calls.append(kwargs)
        eval_bundle_path = Path(kwargs["eval_bundle_path"])
        eval_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(eval_bundle_path, Xi_initial=np.zeros((4, kwargs["D"]), dtype=np.float32))
        pass_ids = [int(entry["pass_id"]) for entry in kwargs["pass_entries"]]
        is_final = bool(kwargs.get("promote_final_artifacts", True))
        excluded_pass_ids = list(kwargs.get("exclude_pass_ids_from_selection", []) or [])
        selected_pass_id = 1 if is_final and 1 not in excluded_pass_ids else (2 if is_final else None)
        return {
            "processed_pass_ids": pass_ids,
            "processed_pass_indices": [_pass_index(pid) for pid in pass_ids],
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {pid: 0.25 / float(pid) for pid in pass_ids},
            "pass_scores_loss_by_index": {
                str(_pass_index(pid)): 0.25 / float(pid) for pid in pass_ids
            },
            "pass_invalid_reasons": {},
            "pass_invalid_reasons_by_index": {},
            "selection_finalized": is_final,
            "promoted_final_artifacts": is_final,
            "excluded_pass_ids_from_selection": excluded_pass_ids if is_final else [],
            "excluded_pass_indices_from_selection": [
                _pass_index(pid) for pid in excluded_pass_ids
            ] if is_final else [],
            "selected_pass_id": selected_pass_id,
            "selected_pass_index": _pass_index(selected_pass_id) if selected_pass_id is not None else None,
            "selected_score_metric": "loss.eval_mean_loss_per_sample" if is_final else None,
            "selected_score": (0.25 / float(selected_pass_id)) if is_final else None,
            "selected_scores_by_pass": (
                {selected_pass_id: 0.25 / float(selected_pass_id)}
                if is_final
                else {}
            ),
            "selected_scores_by_pass_index": (
                {str(_pass_index(selected_pass_id)): 0.25 / float(selected_pass_id)}
                if is_final
                else {}
            ),
            "exact_summary_by_pass": {},
            "exact_summary_by_pass_index": {},
            "application_summary_by_pass": {},
            "application_summary_by_pass_index": {},
            "application_stability_by_pass": {},
            "application_stability_by_pass_index": {},
            "selected_application_summary": None,
            "eval_bundle_path": str(eval_bundle_path),
            "eval_bundle_status": "created",
            "eval_bundle_sha256": file_sha256(str(eval_bundle_path)),
            "evaluation_bundle_M": 4,
            "visual_sample_paths": int(kwargs.get("visual_sample_paths", 0)),
            "visual_seed": int(kwargs.get("visual_seed") or 0),
        }

    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cases = [
                ("pascucci", [], 1, 0, 5),
                ("quadratic_coupled", [1], 2, 1, 0),
            ]
            for (
                model_name,
                expected_excluded,
                expected_selected_id,
                expected_selected_index,
                requested_mc_paths,
            ) in cases:
                print_calls.clear()
                out_dir = Path(tmp) / model_name
                args = [
                    "run",
                    "--mode",
                    "recursive",
                    "--model",
                    model_name,
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "2",
                    "--pass1_init",
                    "base",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
                if requested_mc_paths > 0:
                    args.extend(["--mc_confirmation_paths", str(requested_mc_paths)])
                exit_code = cli.main(args)
                assert exit_code == 0
                assert len(print_calls) == 2
                first_call, final_call = print_calls
                assert first_call["promote_final_artifacts"] is False
                assert first_call["mc_confirmation_paths"] == 0
                assert first_call["exclude_pass_ids_from_selection"] == []
                assert [int(entry["pass_id"]) for entry in first_call["pass_entries"]] == [1]
                assert final_call["promote_final_artifacts"] is True
                assert final_call["mc_confirmation_paths"] == requested_mc_paths
                assert final_call["exclude_pass_ids_from_selection"] == expected_excluded
                assert [int(entry["pass_id"]) for entry in final_call["pass_entries"]] == [1, 2]

                runs = sorted(out_dir.glob("run_*"))
                assert len(runs) == 1
                result_path = runs[0] / "recursive" / "results.json"
                assert result_path.exists()
                results = json.loads(result_path.read_text(encoding="utf-8"))
                assert results["selection_excluded_pass_ids"] == expected_excluded
                assert results["selected_pass_id"] == expected_selected_id
                assert results["selected_pass_index"] == expected_selected_index
                assert results["promoted_final_artifacts"] is True
    finally:
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass


def test_cli_records_pascucci_cost_profile_params_in_run_config() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cli_records_cost_profile_params_in_run_config")
    from . import cli, orchestration
    from .tf_backend import require_tensorflow

    require_tensorflow()

    class FakeSession:
        def close(self) -> None:
            pass

    class FakeStandardModel:
        def __init__(self) -> None:
            self.sess = FakeSession()

        def save_model(self, path: str) -> None:
            marker = Path(path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("fake checkpoint\n", encoding="utf-8")

    standard_calls = []

    def fake_run_standard_reference(**kwargs):
        standard_calls.append(kwargs)
        assert kwargs["model_spec"].name == "pascucci"
        return FakeStandardModel(), {
            "stage_logs": [
                {
                    "phase": "curriculum",
                    "const": 1.0,
                    "lr": 1.0e-3,
                    "n_iter": 1,
                    "train_last_loss": 1.0,
                    "eval_mean_loss": 1.0,
                    "eval_std_loss": 0.0,
                    "eval_mean_loss_per_sample": 0.25,
                    "eval_std_loss_per_sample": 0.0,
                    "eval_mean_y0": 0.0,
                    "eval_std_y0": 0.0,
                    "elapsed_sec": 0.0,
                }
            ],
            "eval_stats": {
                "mean_loss": 1.0,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.25,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
        }

    original_run_standard_reference = orchestration.run_standard_reference
    original_export_standard_parameter_blob = cli.export_standard_parameter_blob
    original_plot_stage_logs = cli.plot_stage_logs
    orchestration.run_standard_reference = fake_run_standard_reference
    cli.export_standard_parameter_blob = lambda model: _make_blob([5, 8, 1])
    cli.plot_stage_logs = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            try:
                exit_code = cli.main(
                    [
                        "run",
                        "--mode",
                        "standard",
                        "--model",
                        "pascucci",
                        "--pascucci_cost_profile",
                        "exp_minus_offset",
                        "--pascucci_cost_offset",
                        "0.12",
                        "--M",
                        "4",
                        "--N",
                        "2",
                        "--T_standard",
                        "0.25",
                        "--T_total",
                        "0.25",
                        "--block_size",
                        "0.25",
                        "--passes",
                        "1",
                        "--visual_sample_paths",
                        "1",
                        "--output_dir",
                        str(out_dir),
                    ]
                )
            except SystemExit as exc:
                raise AssertionError("CLI should accept Pascucci cost-profile arguments") from exc
            assert exit_code == 0
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            assert config["model_name"] == "pascucci"
            assert config["params"]["pascucci_cost_profile"] == "exp_minus_offset"
            assert np.isclose(config["params"]["pascucci_cost_offset"], 0.12)
            assert standard_calls[0]["params"]["pascucci_cost_profile"] == "exp_minus_offset"
            assert np.isclose(float(standard_calls[0]["params"]["pascucci_cost_offset"]), 0.12)
    finally:
        orchestration.run_standard_reference = original_run_standard_reference
        cli.export_standard_parameter_blob = original_export_standard_parameter_blob
        cli.plot_stage_logs = original_plot_stage_logs


def test_cli_records_application_metric_manifest_in_run_config() -> None:
    from . import cli, orchestration
    from .tf_backend import require_tensorflow

    require_tensorflow()

    class FakeSession:
        def close(self) -> None:
            pass

    class FakeStandardModel:
        def __init__(self) -> None:
            self.sess = FakeSession()

        def save_model(self, path: str) -> None:
            marker = Path(path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("fake checkpoint\n", encoding="utf-8")

        def application_cost_functional(self, t, W, Xi, const_value=None, baseline_mode="controlled"):
            del W, const_value
            mode = str(baseline_mode)
            M = int(np.asarray(Xi).shape[0])
            steps = int(np.asarray(t).shape[1]) - 1
            base = np.float32(1.0 if mode == "controlled" else 2.0)
            running = np.full((M, 1), base, dtype=np.float32)
            terminal = np.full((M, 1), base + np.float32(0.25), dtype=np.float32)
            total = running + terminal
            if steps > 0:
                running_cumulative = np.cumsum(
                    np.full((M, steps, 1), base / np.float32(steps), dtype=np.float32),
                    axis=1,
                ).astype(np.float32)
            else:
                running_cumulative = np.zeros((M, 0, 1), dtype=np.float32)
            trajectory = (
                np.concatenate([np.zeros((M, 1, 1), dtype=np.float32), running_cumulative], axis=1)
                + terminal[:, None, :]
            ).astype(np.float32)
            pathwise = {
                "cost_J_running": running,
                "cost_J_terminal": terminal,
                "cost_J_total": total,
                "cost_J_running_cumulative": running_cumulative,
                "cost_J_trajectory": trajectory,
                "cost_J_trajectory_math": trajectory.copy(),
                "alpha": np.zeros((M, steps, 1), dtype=np.float32),
            }
            summary = {}
            for metric, values in pathwise.items():
                if not metric.startswith("cost_J_"):
                    continue
                values_np = np.asarray(values, dtype=np.float32)
                if metric in ("cost_J_running_cumulative", "cost_J_trajectory", "cost_J_trajectory_math"):
                    flat = values_np[:, -1, :].reshape(-1) if values_np.shape[1] > 0 else np.zeros((values_np.shape[0],), dtype=np.float32)
                else:
                    flat = values_np.reshape(-1)
                summary[f"{metric}_mean"] = float(flat.mean())
            return {
                "schema": "pascucci_application_metrics_v2",
                "metadata": {
                    "baseline_mode": mode,
                    "aggregation": "left_riemann_f_plus_terminal_g",
                    "trajectory_aggregation": "plotmaker_cumsum_f_dt_plus_g_at_time",
                    "trajectory_math_aggregation": "left_riemann_integral_to_time_plus_g_at_time",
                    "control_law": "alpha_tf" if mode == "controlled" else "alpha_zero",
                    "paired_inputs": "same_t_W_Xi",
                },
                "pathwise": pathwise,
                "summary": summary,
            }

    def fake_run_standard_reference(**kwargs):
        assert kwargs["model_spec"].name == "pascucci"
        return FakeStandardModel(), {
            "stage_logs": [],
            "eval_stats": {
                "mean_loss": 1.0,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.25,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            },
            "refine_rounds": 0,
        }

    original_run_standard_reference = orchestration.run_standard_reference
    original_export_standard_parameter_blob = cli.export_standard_parameter_blob
    original_plot_stage_logs = cli.plot_stage_logs
    orchestration.run_standard_reference = fake_run_standard_reference
    cli.export_standard_parameter_blob = lambda model: _make_blob([5, 8, 1])
    cli.plot_stage_logs = lambda *args, **kwargs: None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            exit_code = cli.main(
                [
                    "run",
                    "--mode",
                    "standard",
                    "--model",
                    "pascucci",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--eval_seed",
                    "222",
                    "--visual_seed",
                    "333",
                    "--coarse_prepass_seed",
                    "444",
                    "--exact_init_seed",
                    "555",
                    "--output_dir",
                    str(out_dir),
                ]
            )
            assert exit_code == 0
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            assert config["application_metric_schema"] == "pascucci_application_metrics_v2"
            assert config["application_metric_names"] == [
                "cost_J_running",
                "cost_J_terminal",
                "cost_J_total",
                "cost_J_running_cumulative",
                "cost_J_trajectory",
                "cost_J_trajectory_math",
            ]
            assert config["application_metric_aggregation"] == "left_riemann_f_plus_terminal_g"
            assert config["seed_manifest"]["global_seed"] == 1234
            assert config["seed_manifest"]["eval_seed"] == 222
            assert config["seed_manifest"]["recursive_fixed_eval_seed"] is None
            assert config["seed_manifest"]["visual_seed"] == 333
            assert config["seed_manifest"]["visual_seed_effective"] == 333
            assert config["seed_manifest"]["coarse_prepass_seed"] == 444
            assert config["seed_manifest"]["exact_init_seed"] == 555
            assert config["visual_seed_effective"] == 333
            assert isinstance(config["run_config_sha256"], str)
            assert len(config["run_config_sha256"]) == 64
            int(config["run_config_sha256"], 16)
            app_json = runs[0] / "standard" / "application_metrics.json"
            app_npz = runs[0] / "standard" / "application_metrics.npz"
            assert app_json.exists()
            assert app_npz.exists()
            payload = json.loads(app_json.read_text(encoding="utf-8"))
            assert payload["schema"] == "pascucci_application_metrics_v2"
            assert payload["controlled"]["metadata"]["baseline_mode"] == "controlled"
            assert payload["uncontrolled"]["metadata"]["baseline_mode"] == "uncontrolled"
            assert payload["horizon"]["eval_seed"] == 222
            with np.load(app_npz) as data:
                keys = set(data.files)
                assert "controlled_cost_J_total" in keys
                assert "uncontrolled_cost_J_total" in keys
                assert "controlled_cost_J_trajectory" in keys
                assert "uncontrolled_cost_J_trajectory" in keys
                assert "controlled_cost_J_trajectory_math" in keys
                assert "uncontrolled_cost_J_trajectory_math" in keys
                assert data["controlled_cost_J_total"].shape == (4, 1)
                assert data["controlled_cost_J_trajectory"].shape == (4, 3, 1)
                assert data["controlled_cost_J_trajectory_math"].shape == (4, 3, 1)
    finally:
        orchestration.run_standard_reference = original_run_standard_reference
        cli.export_standard_parameter_blob = original_export_standard_parameter_blob
        cli.plot_stage_logs = original_plot_stage_logs


def test_run_config_sha256_excludes_timestamp_metadata() -> None:
    from .cli import _run_config_sha256

    config_a = {
        "timestamp": "run_20260609_120000",
        "model_name": "pascucci",
        "M": 4,
        "N": 2,
        "application_metric_schema": "pascucci_application_metrics_v2",
        "eval_bundle_path_effective": "/tmp/run_a/recursive/evaluation_bundle.npz",
        "eval_bundle_sha256": "0" * 64,
        "eval_bundle_status": "created",
        "recursive_rollout_bundle_path_effective": "/tmp/run_a/recursive/boundary_rollout_bundle.npz",
        "recursive_rollout_bundle_sha256": "1" * 64,
        "recursive_rollout_bundle_status": "created",
    }
    config_b = dict(config_a)
    config_b["timestamp"] = "run_20260609_120001"
    config_b["eval_bundle_path_effective"] = "/tmp/run_b/recursive/evaluation_bundle.npz"
    config_b["eval_bundle_sha256"] = "2" * 64
    config_b["eval_bundle_status"] = "loaded"
    config_b["recursive_rollout_bundle_path_effective"] = "/tmp/run_b/recursive/boundary_rollout_bundle.npz"
    config_b["recursive_rollout_bundle_sha256"] = "3" * 64
    config_b["recursive_rollout_bundle_status"] = "loaded"
    config_c = dict(config_a)
    config_c["M"] = 8

    assert _run_config_sha256(config_a) == _run_config_sha256(config_b)
    assert _run_config_sha256(config_a) != _run_config_sha256(config_c)


def test_cli_rejects_pascucci_cost_profile_for_quadratic_model() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cli_rejects_cost_profile_for_quadratic_model")
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            cli.main(
                [
                    "run",
                    "--mode",
                    "standard",
                    "--model",
                    "quadratic_coupled",
                    "--pascucci_cost_profile",
                    "exp_minus_offset",
                    "--pascucci_cost_offset",
                    "0.12",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(Path(tmp)),
                ]
            )
        except SystemExit as exc:
            raise AssertionError("CLI should parse Pascucci args and reject them semantically for quadratic_coupled") from exc
        except ValueError as exc:
            message = str(exc)
            assert "--pascucci_cost_profile" in message
            assert "pascucci" in message
        else:
            raise AssertionError("quadratic_coupled should reject Pascucci-specific cost-profile args")


def test_cli_rejects_pascucci_calibration_paths_for_quadratic_model() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cli_rejects_calibration_paths_for_quadratic_model")
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            cli.main(
                [
                    "run",
                    "--mode",
                    "standard",
                    "--model",
                    "quadratic_coupled",
                    "--pascucci_H_path",
                    str(Path(tmp) / "H.csv"),
                    "--pascucci_S_path",
                    str(Path(tmp) / "S.xlsx"),
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(Path(tmp) / "out"),
                ]
            )
        except SystemExit as exc:
            raise AssertionError("CLI should parse Pascucci calibration args before semantic rejection") from exc
        except ValueError as exc:
            message = str(exc)
            assert "--pascucci_H_path/--pascucci_S_path" in message
            assert "--model pascucci" in message
        else:
            raise AssertionError("quadratic_coupled should reject Pascucci-specific calibration paths")


def test_cli_records_pascucci_cost_profile_params_in_recursive_run_config() -> None:
    _assert_pascucci_model_tdd_contract("pascucci_cli_records_cost_profile_params_in_recursive_run_config")
    from . import cli, orchestration
    from .sampling import build_blocks, file_sha256
    from .tf_backend import require_tensorflow

    require_tensorflow()

    recursive_calls = []
    print_calls = []

    def fake_run_recursive_training(**kwargs):
        recursive_calls.append(kwargs)
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        logs = [
            {
                "pass": 1,
                "block": 0,
                "t_start": blocks[0]["t_start"],
                "t_end": blocks[0]["t_end"],
                "T_block": blocks[0]["T_block"],
                "eval_mean_loss": 1.0,
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25,
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
            }
        ]
        return {
            "blocks": blocks,
            "passes": [
                {
                    "pass_id": 1,
                    "reference_loss": 1.0,
                    "logs": logs,
                    "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
                    "models_dir": kwargs["output_dir"],
                    "pass_init_mode": kwargs["pass1_init_mode"],
                    "boundary_source": "base_xi",
                    "is_bootstrap_pass": True,
                    "active_set_summary": {},
                }
            ],
            "pass1": {
                "logs": logs,
                "reference_loss": 1.0,
                "blobs": [_make_blob([5, 8, 1]) for _ in blocks],
            },
            "boundary_samples": [
                np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        print_calls.append(kwargs)
        eval_bundle_path = Path(kwargs["eval_bundle_path"])
        eval_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            eval_bundle_path,
            Xi_initial=np.zeros((4, kwargs["D"]), dtype=np.float32),
        )
        return {
            "processed_pass_ids": [1],
            "exact_summary_by_pass": {},
            "exact_summary_by_pass_index": {},
            "eval_bundle_path": str(eval_bundle_path),
            "eval_bundle_status": "created",
            "eval_bundle_sha256": file_sha256(str(eval_bundle_path)),
            "evaluation_bundle_M": 4,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": 1,
            "selected_pass_index": 0,
            "selected_score_metric": "loss.eval_mean_loss_per_sample",
            "selected_score": 0.25,
            "selected_scores_by_pass": {"1": 0.25},
            "selected_scores_by_pass_index": {"0": 0.25},
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {1: 0.25},
            "pass_scores_loss_by_index": {0: 0.25},
        }

    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            try:
                exit_code = cli.main(
                    [
                        "run",
                        "--mode",
                        "recursive",
                        "--model",
                        "pascucci",
                        "--pascucci_cost_profile",
                        "exp_minus_offset",
                        "--pascucci_cost_offset",
                        "0.12",
                        "--M",
                        "4",
                        "--N",
                        "2",
                        "--T_standard",
                        "0.25",
                        "--T_total",
                        "0.25",
                        "--block_size",
                        "0.25",
                        "--passes",
                        "1",
                        "--visual_sample_paths",
                        "1",
                        "--output_dir",
                        str(out_dir),
                    ]
                )
            except SystemExit as exc:
                raise AssertionError("CLI should accept Pascucci cost-profile arguments in recursive mode") from exc
            assert exit_code == 0
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            assert config["mode"] == "recursive"
            assert config["application_metric_schema"] == "pascucci_application_metrics_v2"
            assert config["application_metric_names"] == [
                "cost_J_running",
                "cost_J_terminal",
                "cost_J_total",
                "cost_J_running_cumulative",
                "cost_J_trajectory",
                "cost_J_trajectory_math",
            ]
            assert config["application_metric_aggregation"] == "left_riemann_f_plus_terminal_g"
            assert config["seed_manifest"]["eval_seed"] == 1234
            assert config["seed_manifest"]["visual_seed"] is None
            assert config["seed_manifest"]["visual_seed_effective"] == 9153
            assert config["visual_seed_effective"] == 9153
            assert config["params"]["pascucci_cost_profile"] == "exp_minus_offset"
            assert np.isclose(config["params"]["pascucci_cost_offset"], 0.12)
            assert recursive_calls[0]["params"]["pascucci_cost_profile"] == "exp_minus_offset"
            assert np.isclose(float(recursive_calls[0]["params"]["pascucci_cost_offset"]), 0.12)
            assert len(print_calls) == 1
    finally:
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass


def test_cli_rejects_unsupported_model_argument() -> None:
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            cli.main(
                [
                    "run",
                    "--model",
                    "unsupported_model_xyz",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
        except ValueError as exc:
            assert "Unknown model 'unsupported_model_xyz'" in str(exc)
            assert "Supported: quadratic_coupled, pascucci" in str(exc)
            assert list(out_dir.glob("run_*")) == []
        else:
            raise AssertionError("unsupported CLI model should fail before running")


def test_cli_rejects_pascucci_exact_profile() -> None:
    from . import cli
    from .tf_backend import require_tensorflow

    require_tensorflow()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            cli.main(
                [
                    "run",
                    "--model",
                    "pascucci",
                    "--exact_solution",
                    "quadratic_coupled",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
        except ValueError as exc:
            assert "pascucci does not provide an exact solution profile yet" in str(exc)
            assert "--exact_solution none" in str(exc)
            assert list(out_dir.glob("run_*")) == []
        else:
            raise AssertionError("pascucci exact profile should fail before running")


def test_cli_uses_resolved_model_spec_for_runtime_wiring() -> None:
    import json

    from . import cli, orchestration
    from .model_specs import ModelSpec
    from .sampling import build_blocks
    from .tf_backend import require_tensorflow

    require_tensorflow()

    calls = {
        "params": 0,
        "layers": 0,
        "xi": 0,
        "exact": 0,
        "recursive": 0,
        "print": 0,
    }
    sentinel_exact = {"name": "sentinel"}

    def sentinel_build_default_params(const: float = 1.0) -> dict:
        calls["params"] += 1
        return {
            "sentinel_param": np.float32(42.0),
            "const": np.float32(const),
        }

    def sentinel_build_layers(D: int) -> list[int]:
        calls["layers"] += 1
        assert int(D) == 4
        return [5, 7, 1]

    def sentinel_xi_generator(M: int, D: int) -> np.ndarray:
        calls["xi"] += 1
        return np.full((int(M), int(D)), 3.25, dtype=np.float32)

    def sentinel_deterministic_xi(M: int, D: int, seed: int = 1234) -> np.ndarray:
        return np.full((int(M), int(D)), float(seed % 7), dtype=np.float32)

    def sentinel_build_exact_solution(profile: str, params: dict, D: int):
        calls["exact"] += 1
        assert profile == "sentinel"
        assert params["sentinel_param"] == np.float32(42.0)
        assert int(D) == 4
        return sentinel_exact

    sentinel_spec = ModelSpec(
        name="sentinel_model",
        state_dim=4,
        state_labels=("A", "B", "C", "D"),
        z_labels=("ZA", "ZB", "ZC", "ZD"),
        build_default_params=sentinel_build_default_params,
        build_layers=sentinel_build_layers,
        xi_generator=sentinel_xi_generator,
        deterministic_xi=sentinel_deterministic_xi,
        standard_model_factory=lambda **kwargs: None,
        recursive_model_factory=lambda **kwargs: None,
        build_exact_solution=sentinel_build_exact_solution,
        build_exact_initial_boundary_samples=None,
    )

    # Distinctive sentinel hooks make stale benchmark wiring fail loudly.
    def fake_run_recursive_training(**kwargs):
        calls["recursive"] += 1
        assert kwargs["model_spec"] is sentinel_spec
        assert kwargs["Xi_generator"] is sentinel_xi_generator
        xi_probe = kwargs["Xi_generator"](2, kwargs["D"])
        np.testing.assert_allclose(xi_probe, np.full((2, 4), 3.25, dtype=np.float32))
        assert kwargs["params"]["sentinel_param"] == np.float32(42.0)
        assert kwargs["params"]["const"] == np.float32(0.5)
        assert kwargs["layers"] == [5, 7, 1]
        assert kwargs["D"] == 4
        blocks = build_blocks(T_total=kwargs["T_total"], block_size=kwargs["block_size"])
        logs = [
            {
                "pass": 1,
                "block": 0,
                "t_start": blocks[0]["t_start"],
                "t_end": blocks[0]["t_end"],
                "T_block": blocks[0]["T_block"],
                "eval_mean_loss": 1.0,
                "eval_std_loss": 0.0,
                "eval_mean_loss_per_sample": 0.25,
                "eval_std_loss_per_sample": 0.0,
                "eval_mean_y0": 0.0,
                "precision_target": None,
                "refine_rounds": 0,
            }
        ]
        return {
            "blocks": blocks,
            "passes": [
                {
                    "pass_id": 1,
                    "reference_loss": 1.0,
                    "logs": logs,
                    "blobs": [_make_blob([5, 7, 1]) for _ in blocks],
                    "models_dir": kwargs["output_dir"],
                    "pass_init_mode": kwargs["pass1_init_mode"],
                    "boundary_source": "base_xi",
                    "is_bootstrap_pass": True,
                    "active_set_summary": {},
                }
            ],
            "boundary_samples": [
                np.zeros((kwargs["M"], kwargs["D"]), dtype=np.float32)
                for _ in range(len(blocks) + 1)
            ],
        }

    def fake_print_recursive_pass(**kwargs):
        calls["print"] += 1
        assert kwargs["model_spec"] is sentinel_spec
        assert kwargs["params"]["sentinel_param"] == np.float32(42.0)
        assert kwargs["layers"] == [5, 7, 1]
        assert kwargs["exact_solution"] is sentinel_exact
        return {
            "processed_pass_ids": [1],
            "exact_summary_by_pass": {1: {"sentinel": True}},
            "exact_summary_by_pass_index": {0: {"sentinel": True}},
            "eval_bundle_path": str(Path(kwargs["rec_dir"]) / "evaluation_bundle.npz"),
            "evaluation_bundle_M": 4,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": 1,
            "selected_pass_index": 0,
            "selected_score_metric": "loss.eval_mean_loss_per_sample",
            "selected_score": 0.25,
            "selected_scores_by_pass": {"1": 0.25},
            "selected_scores_by_pass_index": {"0": 0.25},
            "score_key": "eval_mean_loss_per_sample",
            "pass_scores_loss": {1: 0.25},
            "pass_scores_loss_by_index": {0: 0.25},
        }

    original_get_model_spec = cli.get_model_spec
    original_run_recursive_training = orchestration.run_recursive_training
    original_print_recursive_pass = orchestration.print_recursive_pass
    cli.get_model_spec = lambda name=None: sentinel_spec
    orchestration.run_recursive_training = fake_run_recursive_training
    orchestration.print_recursive_pass = fake_print_recursive_pass

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            exit_code = cli.main(
                [
                    "run",
                    "--mode",
                    "recursive",
                    "--M",
                    "4",
                    "--N",
                    "2",
                    "--T_standard",
                    "0.25",
                    "--T_total",
                    "0.25",
                    "--block_size",
                    "0.25",
                    "--passes",
                    "1",
                    "--const_override",
                    "0.5",
                    "--exact_solution",
                    "sentinel",
                    "--visual_sample_paths",
                    "1",
                    "--output_dir",
                    str(out_dir),
                ]
            )
            assert exit_code == 0
            runs = sorted(out_dir.glob("run_*"))
            assert len(runs) == 1
            config = json.loads((runs[0] / "run_config.json").read_text(encoding="utf-8"))
            assert config["model_name"] == "sentinel_model"
            assert config["model_requested"] == "quadratic_coupled"
            assert config["state_labels"] == ["A", "B", "C", "D"]
            assert config["z_labels"] == ["ZA", "ZB", "ZC", "ZD"]
            assert config["layers"] == [5, 7, 1]
            assert config["exact_solution"] == "sentinel"
            assert config["params"]["sentinel_param"] == 42.0
            assert config["params"]["const"] == 0.5
            assert (runs[0] / "recursive" / "results.json").exists()

        assert calls["params"] == 1
        assert calls["layers"] == 1
        assert calls["xi"] == 1
        assert calls["exact"] == 1
        assert calls["recursive"] == 1
        assert calls["print"] >= 1
    finally:
        cli.get_model_spec = original_get_model_spec
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass


def test_loss_improvement_flags_are_finite() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(17)
    params = _default_params()
    params.update(
        {
            "dynamic_loss_dt_normalization": True,
            "dynamic_loss_weight": np.float32(1.0),
            "terminal_y_loss_weight": np.float32(1.0),
            "terminal_z_loss_weight": np.float32(2.0),
            "terminal_z_component_weights": [3.0, 0.25, 2.0, 0.0],
            "structural_z_loss_weight": np.float32(0.5),
            "structural_z_component_weights": [0.0, 1.0, 0.0, 0.0],
        }
    )
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    t_batch, W_batch, Xi_batch = model.fetch_minibatch()
    loss, _, _, _, components = model.loss_function(
        t_batch,
        W_batch,
        Xi_batch,
        const_value=1.0,
        return_components=True,
    )
    assert np.isfinite(float(loss.numpy()))
    assert np.isfinite(float(components["loss_total"].numpy()))
    assert np.isfinite(float(components["loss_dynamic"].numpy()))
    assert np.isfinite(float(components["loss_terminal_z"].numpy()))
    np.testing.assert_allclose(
        float(loss.numpy()),
        float(components["loss_total"].numpy()),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    stats = model.evaluate(const_value=1.0, n_batches=1)
    for key in (
        "mean_loss_dynamic",
        "mean_loss_terminal_y",
        "mean_loss_terminal_z",
        "mean_loss_structural_z",
        "mean_loss_weighted_terminal_z_component_0",
        "mean_loss_weighted_structural_z_component_1",
    ):
        assert key in stats
        assert np.isfinite(stats[key])
    train_stats = model.train(
        N_Iter=1,
        learning_rate=1.0e-3,
        const_value=1.0,
        eval_every=1,
        val_batches=1,
    )
    assert train_stats["best_iter"] == 0
    assert np.isfinite(train_stats["best_score"])
    model.close()


def test_terminal_blob_constants_survive_train_graph() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(19)
    params = _default_params()
    layers = [5, 8, 1]
    terminal_blob = _make_blob(layers)
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=layers,
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.5,
        terminal_blob=terminal_blob,
        normalize_time_input=True,
    )
    train_stats = model.train(
        N_Iter=1,
        learning_rate=1.0e-3,
        const_value=1.0,
        eval_every=1,
        val_batches=1,
    )
    assert train_stats["best_iter"] == 0
    assert np.isfinite(train_stats["best_score"])

    eval_stats = model.evaluate(const_value=1.0, n_batches=1)
    assert np.isfinite(eval_stats["mean_loss"])
    t_batch, W_batch, Xi_batch = model.fetch_minibatch()
    X, Y, Z = model.predict(Xi_batch, t_batch, W_batch, const_value=1.0)
    assert X.shape == (4, 3, 4)
    assert Y.shape == (4, 3, 1)
    assert Z.shape == (4, 3, 4)
    model.close()


def test_train_with_standard_schedule_uses_fixed_evaluation_batches() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()
    train_calls = []
    evaluate_calls = []

    class FixedEvalModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            train_calls.append(dict(kwargs))
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            evaluate_calls.append(
                {
                    "const_value": const_value,
                    "n_batches": n_batches,
                    "evaluation_batches": evaluation_batches,
                    "moment_policy": moment_policy,
                }
            )
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    train_with_standard_schedule(
        model=FixedEvalModel(),
        stage_plan=[(1, 1.0e-3)],
        final_plan=[],
        eval_batches=fixed_batches,
        precision_target=None,
        label="pass1:block5",
        coupling_const=1.0,
    )

    assert len(evaluate_calls) >= 2
    for call in evaluate_calls:
        assert len(call["evaluation_batches"]) == len(fixed_batches)
        for actual_batch, expected_batch in zip(call["evaluation_batches"], fixed_batches):
            for actual, expected in zip(actual_batch, expected_batch):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        assert call["n_batches"] == len(fixed_batches)
        assert call["moment_policy"] == "fixed_evaluation_bundle"
    assert len(train_calls) == 1
    train_call = train_calls[0]
    assert len(train_call["evaluation_batches"]) == len(fixed_batches)
    for actual_batch, expected_batch in zip(train_call["evaluation_batches"], fixed_batches):
        for actual, expected in zip(actual_batch, expected_batch):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert train_call["moment_policy"] == "fixed_evaluation_bundle"


def test_train_with_standard_schedule_accepts_kwargs_evaluate_adapter() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()
    evaluate_calls = []

    class KwargsEvalModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(self, **kwargs):
            evaluate_calls.append(dict(kwargs))
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    train_with_standard_schedule(
        model=KwargsEvalModel(),
        stage_plan=[(1, 1.0e-3)],
        final_plan=[],
        eval_batches=fixed_batches,
        precision_target=None,
        label="pass1:block5",
        coupling_const=1.0,
    )

    assert len(evaluate_calls) >= 2
    for call in evaluate_calls:
        assert call["n_batches"] == len(fixed_batches)
        assert call["evaluation_batches"] == fixed_batches
        assert call["moment_policy"] == "fixed_evaluation_bundle"


def test_train_with_standard_schedule_fixed_eval_passes_integer_batch_count() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()

    class CastsBatchCountModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            assert int(n_batches) == len(fixed_batches)
            for actual_batch, expected_batch in zip(list(evaluation_batches), fixed_batches):
                for actual, expected in zip(actual_batch, expected_batch):
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
            assert moment_policy == "fixed_evaluation_bundle"
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    train_with_standard_schedule(
        model=CastsBatchCountModel(),
        stage_plan=[(1, 1.0e-3)],
        final_plan=[],
        eval_batches=fixed_batches,
        precision_target=None,
        label="pass1:block5",
        coupling_const=1.0,
    )


def test_train_with_standard_schedule_rejects_fixed_eval_without_bundle_api() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()

    class RestrictedEvalModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(self, const_value=None, n_batches=5):
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    try:
        train_with_standard_schedule(
            model=RestrictedEvalModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=fixed_batches,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except TypeError as exc:
        assert "fixed evaluation batches require model.evaluate" in str(exc)
        assert "evaluation_batches" in str(exc)
        assert "moment_policy" in str(exc)
    else:
        raise AssertionError("fixed eval bundles must fail fast when evaluate lacks bundle kwargs")


def test_train_with_standard_schedule_rejects_noninspectable_fixed_eval_api() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()

    class NoninspectableEvaluate:
        __signature__ = "invalid"

        def __call__(self, const_value=None, n_batches=5):
            raise AssertionError("fixed eval should fail before calling a noninspectable evaluate")

    class NoninspectableEvalModel:
        const = np.float32(1.0)
        evaluate = NoninspectableEvaluate()

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

    try:
        train_with_standard_schedule(
            model=NoninspectableEvalModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=fixed_batches,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except TypeError as exc:
        assert "fixed evaluation batches require model.evaluate" in str(exc)
    else:
        raise AssertionError("fixed eval bundles must reject noninspectable evaluate callables")


def test_train_with_standard_schedule_rejects_positional_only_fixed_eval_api() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = _schedule_fixed_eval_batches()

    class PositionalOnlyEvalModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(self, const_value, n_batches, evaluation_batches, moment_policy, /):
            raise AssertionError("fixed eval should fail before calling a positional-only evaluate")

    try:
        train_with_standard_schedule(
            model=PositionalOnlyEvalModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=fixed_batches,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except TypeError as exc:
        assert "fixed evaluation batches require model.evaluate" in str(exc)
    else:
        raise AssertionError("fixed eval bundles must reject positional-only evaluate parameters")


def test_train_with_standard_schedule_rejects_malformed_fixed_eval_batches() -> None:
    from .orchestration import train_with_standard_schedule

    t_batch = np.zeros((2, 2, 1), dtype=np.float32)
    W_batch = np.zeros((2, 2, 4), dtype=np.float32)
    Xi_batch = np.zeros((2, 4), dtype=np.float32)

    class ShouldNotEvaluateModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            raise AssertionError("malformed fixed eval batches should fail before evaluate")

    def assert_rejected(eval_batches, expected):
        try:
            train_with_standard_schedule(
                model=ShouldNotEvaluateModel(),
                stage_plan=[(1, 1.0e-3)],
                final_plan=[],
                eval_batches=eval_batches,
                precision_target=None,
                label="pass1:block5",
                coupling_const=1.0,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"fixed eval batches should reject malformed input: {expected}")

    assert_rejected([(t_batch, W_batch)], "triple")
    assert_rejected([(t_batch, W_batch[:, :1, :], Xi_batch)], "time dimension")
    assert_rejected([(t_batch, W_batch, Xi_batch[:, :3])], "state dimension")


def test_train_with_standard_schedule_rejects_model_incompatible_fixed_eval_horizon() -> None:
    from .orchestration import train_with_standard_schedule

    fixed_batches = [
        (
            np.linspace(0.0, 0.25, 4, dtype=np.float32).reshape(1, 4, 1).repeat(2, axis=0),
            np.zeros((2, 4, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )
    ]
    calls = {"train": 0, "evaluate": 0}

    class DimensionedModel:
        const = np.float32(1.0)
        M = 2
        N = 2
        D = 4

        def train(self, **kwargs):
            calls["train"] += 1
            raise AssertionError("fixed eval horizon mismatch should fail before training")

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            calls["evaluate"] += 1
            raise AssertionError("fixed eval horizon mismatch should fail before evaluation")

    try:
        train_with_standard_schedule(
            model=DimensionedModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=fixed_batches,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except ValueError as exc:
        assert "N+1=3" in str(exc)
    else:
        raise AssertionError("fixed eval batches should reject model-incompatible time horizons")
    assert calls == {"train": 0, "evaluate": 0}


def test_train_with_standard_schedule_rejects_zero_path_fixed_eval_batches() -> None:
    from .orchestration import train_with_standard_schedule

    class ShouldNotEvaluateModel:
        const = np.float32(1.0)

        def train(self, **kwargs):
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
            }

        def evaluate(
            self,
            const_value=None,
            n_batches=5,
            evaluation_batches=None,
            moment_policy="batch_current",
        ):
            raise AssertionError("zero-path fixed eval batches should fail before evaluate")

    try:
        train_with_standard_schedule(
            model=ShouldNotEvaluateModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=[
                (
                    np.zeros((0, 2, 1), dtype=np.float32),
                    np.zeros((0, 2, 4), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32),
                )
            ],
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except ValueError as exc:
        assert "at least one path" in str(exc)
    else:
        raise AssertionError("fixed eval batches should reject zero-path batches")


def test_train_with_standard_schedule_fixed_eval_bundle_with_real_fbsnn() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .orchestration import train_with_standard_schedule
    from .tf_backend import set_seed

    set_seed(33)
    params = _default_params()
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=4,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.0,
        t_end=0.25,
        T_total=0.25,
        normalize_time_input=True,
    )
    fixed_batches = [
        (
            np.linspace(0.0, 0.25, 3, dtype=np.float32).reshape(1, 3, 1).repeat(4, axis=0),
            np.zeros((4, 3, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )
    ]

    try:
        result = train_with_standard_schedule(
            model=model,
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=fixed_batches,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
        )
        eval_stats = result["eval_stats"]
        assert eval_stats["moment_policy"] == "fixed_evaluation_bundle"
        assert eval_stats["n_batches"] == 1
        assert eval_stats["evaluation_batches"] == 1
        assert np.isfinite(float(eval_stats["mean_loss"]))
    finally:
        model.close()


def test_train_with_standard_schedule_aggregates_gradient_diagnostic_maxima() -> None:
    from .orchestration import train_with_standard_schedule

    class GradientDiagnosticModel:
        const = np.float32(1.0)

        def __init__(self) -> None:
            self.train_calls = 0

        def train(self, **kwargs):
            self.train_calls += 1
            assert kwargs["return_gradient_diagnostics"] is True
            if self.train_calls == 1:
                grad = {
                    "train_grad_global_norm": 10.0,
                    "train_grad_global_norm_clipped": 4.0,
                    "train_grad_clip_ratio": 0.4,
                    "train_grad_global_norm_max": 10.0,
                    "train_grad_global_norm_clipped_max": 4.0,
                    "train_grad_clip_ratio_max": 0.4,
                }
            else:
                grad = {
                    "train_grad_global_norm": 2.0,
                    "train_grad_global_norm_clipped": 2.0,
                    "train_grad_clip_ratio": 1.0,
                    "train_grad_global_norm_max": 2.0,
                    "train_grad_global_norm_clipped_max": 2.0,
                    "train_grad_clip_ratio_max": 1.0,
                }
            return {
                "last_loss": 0.25,
                "best_score": 0.20,
                "best_iter": 0,
                "stopped_early": False,
                **grad,
            }

        def evaluate(self, **kwargs):
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    model = GradientDiagnosticModel()
    result = train_with_standard_schedule(
        model=model,
        stage_plan=[(1, 1.0e-3)],
        final_plan=[(1, 1.0e-4)],
        eval_batches=1,
        precision_target=None,
        label="pass1:block5",
        coupling_const=1.0,
        return_gradient_diagnostics=True,
    )

    assert model.train_calls == 2
    diagnostics = result["gradient_diagnostics"]
    assert diagnostics["train_grad_global_norm"] == 2.0
    assert diagnostics["train_grad_clip_ratio"] == 1.0
    assert diagnostics["train_grad_global_norm_max"] == 10.0
    assert diagnostics["train_grad_global_norm_clipped_max"] == 4.0
    assert diagnostics["train_grad_clip_ratio_max"] == 1.0
    assert result["stage_logs"][0]["train_grad_global_norm_max"] == 10.0
    assert result["stage_logs"][1]["train_grad_global_norm_max"] == 2.0


def test_train_with_standard_schedule_rejects_gradient_diagnostics_without_train_api() -> None:
    from .orchestration import train_with_standard_schedule

    class IncompatibleTrainModel:
        const = np.float32(1.0)

        def train(self, N_Iter, learning_rate, const_value):
            raise AssertionError("diagnostic API guard should fail before train")

        def evaluate(self, **kwargs):
            return {
                "mean_loss": 0.25,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.125,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }

    try:
        train_with_standard_schedule(
            model=IncompatibleTrainModel(),
            stage_plan=[(1, 1.0e-3)],
            final_plan=[],
            eval_batches=1,
            precision_target=None,
            label="pass1:block5",
            coupling_const=1.0,
            return_gradient_diagnostics=True,
        )
    except TypeError as exc:
        assert "return_gradient_diagnostics" in str(exc)
    else:
        raise AssertionError("gradient diagnostics should require explicit model.train API support")


def test_train_with_standard_schedule_rejects_nonfinite_stage() -> None:
    from .orchestration import train_with_standard_schedule

    class NonFiniteStageModel:
        def __init__(self) -> None:
            self.const = np.float32(1.0)
            self.train_calls = 0
            self.evaluate_calls = 0

        def train(self, **kwargs):
            self.train_calls += 1
            return {
                "last_loss": float("nan"),
                "best_iter": -1,
                "best_score": float("inf"),
                "stopped_early": False,
            }

        def evaluate(self, **kwargs):
            self.evaluate_calls += 1
            return {
                "mean_loss": float("nan"),
                "std_loss": float("nan"),
                "mean_loss_per_sample": float("nan"),
                "std_loss_per_sample": float("nan"),
                "mean_y0": float("nan"),
                "std_y0": float("nan"),
            }

    model = NonFiniteStageModel()
    try:
        train_with_standard_schedule(
            model,
            stage_plan=[(1, 1.0e-4), (1, 5.0e-5)],
            final_plan=[],
            eval_batches=1,
            label="pass1:block5",
            coupling_const=1.0,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        assert "pass1:block5" in message
        assert "non-finite" in message
    else:
        raise AssertionError("non-finite training stage should abort the schedule")
    assert model.train_calls == 1
    assert model.evaluate_calls == 1


def test_train_with_standard_schedule_rejects_nonfinite_secondary_eval_stats() -> None:
    from .orchestration import train_with_standard_schedule

    class SecondaryNonFiniteEvalModel:
        def __init__(self, bad_key: str) -> None:
            self.bad_key = bad_key
            self.const = np.float32(1.0)
            self.train_calls = 0
            self.evaluate_calls = 0

        def train(self, **kwargs):
            self.train_calls += 1
            return {
                "last_loss": 1.0,
                "best_iter": 0,
                "best_score": 1.0,
                "stopped_early": False,
            }

        def evaluate(self, **kwargs):
            self.evaluate_calls += 1
            stats = {
                "mean_loss": 1.0,
                "std_loss": 0.0,
                "mean_loss_per_sample": 0.25,
                "std_loss_per_sample": 0.0,
                "mean_y0": 0.0,
                "std_y0": 0.0,
            }
            stats[self.bad_key] = float("nan")
            return stats

    for bad_key in ("std_loss_per_sample", "std_y0"):
        model = SecondaryNonFiniteEvalModel(bad_key)
        try:
            train_with_standard_schedule(
                model,
                stage_plan=[(1, 1.0e-4), (1, 5.0e-5)],
                final_plan=[],
                eval_batches=1,
                label=f"pass1:block5:{bad_key}",
                coupling_const=1.0,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            assert "pass1:block5" in message
            assert "non-finite" in message
            assert bad_key in message
        else:
            raise AssertionError(f"non-finite {bad_key} should abort the schedule")
        assert model.train_calls == 1
        assert model.evaluate_calls == 1


def test_recursive_visual_bundle_acceptance() -> None:
    from .orchestration import print_recursive_pass
    from .plotting import _PLOTTING_AVAILABLE
    from .tf_backend import set_seed

    set_seed(29)
    params = _default_params()
    layers = [5, 8, 1]
    blob = _make_blob(layers)
    blocks = [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}]
    logs = [
        {
            "pass": 1,
            "block": 0,
            "t_start": 0.0,
            "t_end": 0.25,
            "T_block": 0.25,
            "eval_mean_loss": 1.0,
            "eval_std_loss": 0.0,
            "eval_mean_loss_per_sample": 0.25,
            "eval_std_loss_per_sample": 0.0,
            "eval_mean_y0": 0.0,
            "precision_target": None,
            "refine_rounds": 0,
        }
    ]
    exact = build_exact_solution_functions("quadratic_coupled", params, D=4)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary = print_recursive_pass(
            pass_entries=[{"pass_id": 1, "logs": logs, "blobs": [blob]}],
            blocks=blocks,
            rec_dir=str(tmp_path),
            params=params,
            N_per_block=2,
            D=4,
            layers=layers,
            T_total=0.25,
            exact_solution=exact,
            selection_metric="loss",
            eval_bundle_path=str(tmp_path / "evaluation_bundle.npz"),
            eval_seed=101,
            eval_min_paths=4,
            sample_paths=2,
            visual_sample_paths=2,
            visual_seed=202,
            print_compact_logs=False,
        )
        assert summary["visual_sample_paths"] == 2
        assert summary["visual_seed"] == 202
        assert summary["application_summary_by_pass"] == {}
        assert summary["selected_application_summary"] is None
        assert (tmp_path / "visual_stitched_predictions_pass00.npz").exists()
        assert (tmp_path / "visual_stitched_predictions_final.npz").exists()
        assert (tmp_path / "exact_metrics_final.json").exists()
        assert not (tmp_path / "application_metrics_pass00.json").exists()
        assert not (tmp_path / "application_metrics_final.json").exists()
        for filename in (
            "stitched_predictions_pass00.npz",
            "stitched_predictions_final.npz",
            "visual_stitched_predictions_pass00.npz",
            "visual_stitched_predictions_final.npz",
        ):
            with np.load(tmp_path / filename) as data:
                assert set(data.files) == {"t", "X", "Y", "Z"}
        if _PLOTTING_AVAILABLE:
            expected_plots = [
                "recursive_stitched_Y_exact_pass00.png",
                "recursive_stitched_Z_S_exact_pass00.png",
                "recursive_stitched_Z_V_exact_pass00.png",
                "recursive_stitched_abs_error_pass00.png",
                "recursive_stitched_Y_exact.png",
                "recursive_stitched_Z_S_exact.png",
            ]
            for name in expected_plots:
                path = tmp_path / "plots" / name
                assert path.exists(), name
                assert path.stat().st_size > 0, name


def test_same_xi_antithetic_minibatch() -> None:
    from .models import NN_Quadratic_Coupled_Recursive
    from .tf_backend import set_seed

    set_seed(23)
    params = _default_params()
    params["same_xi_antithetic_sampling"] = True
    model = NN_Quadratic_Coupled_Recursive(
        Xi_generator=Xi_generator_default,
        T=0.25,
        M=5,
        N=2,
        D=4,
        layers=[5, 8, 1],
        parameters=params,
        t_start=0.5,
        t_end=0.75,
        T_total=0.75,
        normalize_time_input=True,
    )
    t_batch, W_batch, Xi_batch = model.fetch_minibatch()
    dW = W_batch[:, 1:, :] - W_batch[:, :-1, :]
    np.testing.assert_allclose(dW[0], -dW[2], atol=1.0e-6)
    np.testing.assert_allclose(dW[1], -dW[3], atol=1.0e-6)
    np.testing.assert_allclose(Xi_batch[0], Xi_batch[2], atol=1.0e-6)
    np.testing.assert_allclose(Xi_batch[1], Xi_batch[3], atol=1.0e-6)
    assert np.isclose(t_batch[0, 0, 0], 0.5)
    assert np.isclose(t_batch[0, -1, 0], 0.75)
    model.close()


def test_z_selection_metrics() -> None:
    from .orchestration import resolve_pass_selection

    pass_scores = {1: 10.0, 2: 1.0}
    exact_summary = {
        1: {
            "mean_abs_error_y": 5.0,
            "rmse_y": 6.0,
            "abs_error_mean_y0": 7.0,
            "mean_abs_error_z": 0.30,
            "mean_abs_error_z_by_component": [0.90, 0.01, 0.20, 0.0],
        },
        2: {
            "mean_abs_error_y": 6.0,
            "rmse_y": 7.0,
            "abs_error_mean_y0": 8.0,
            "mean_abs_error_z": 0.20,
            "mean_abs_error_z_by_component": [0.70, 0.02, 0.25, 0.0],
        },
    }
    selected, label, score, _ = resolve_pass_selection(
        pass_scores,
        exact_summary,
        selection_metric="exact_mae_z",
    )
    assert selected == 2
    assert label == "exact.mean_abs_error_z"
    assert np.isclose(score, 0.20)

    selected, label, score, _ = resolve_pass_selection(
        pass_scores,
        exact_summary,
        selection_metric="exact_mae_z_s",
    )
    assert selected == 2
    assert label == "exact.mean_abs_error_z_component_Z_S"
    assert np.isclose(score, 0.70)


def test_v1_prediction_parity() -> None:
    from .tf_backend import require_tensorflow

    require_tensorflow()
    repo_root = Path(__file__).resolve().parents[2]
    legacy_path = repo_root / "code" / "Network" / "recursive1_GirsanovLike.py"
    if not legacy_path.exists():
        raise RuntimeError(f"legacy source not found: {legacy_path}")

    layers = [5, 8, 1]
    blob = _make_blob(layers)
    rng = np.random.RandomState(123)
    Xi = np.zeros((4, 4), dtype=np.float32)
    Xi[:, 0] = rng.normal(1.0, 1.0, 4)
    Xi[:, 1] = rng.normal(1.0, 1.0, 4)
    Xi[:, 2] = rng.normal(0.0, 1.0, 4)
    Xi[:, 3] = rng.uniform(3.0, 7.0, 4)
    rollout = build_stitched_rollout_inputs(
        [{"idx": 0, "t_start": 0.0, "t_end": 0.25, "T_block": 0.25}],
        M=4,
        N_per_block=2,
        D=4,
        seed=321,
    )
    t_batch, W_batch = rollout[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture = tmp_path / "fixture.npz"
        old_out = tmp_path / "old.npz"
        new_out = tmp_path / "new.npz"
        np.savez(fixture, Xi=Xi, t=t_batch, W=W_batch, **blob)

        common = f"""
import importlib.util
import numpy as np
from pathlib import Path
fixture = Path({str(fixture)!r})
out_path = Path({str(old_out)!r})
params = {{
    'mu1': np.float32(1.0), 'mu2': np.float32(1.0),
    'c1': np.float32(1.0), 'c2': np.float32(1.0),
    'c3': np.float32(10.0), 'c4': np.float32(10.0),
    'gamma': np.float32(1.0), 'd': np.float32(1.0),
    'x_max': np.float32(10.0), 'v_max': np.float32(2.0),
    'v_min': np.float32(-2.0), 's1': np.float32(0.5),
    's2': np.float32(0.5), 's3': np.float32(0.5),
    'const': np.float32(1.0),
}}
data = np.load(fixture, allow_pickle=False)
blob = {{k: data[k] for k in data.files if k not in ('Xi', 't', 'W')}}
"""
        old_script = common + f"""
spec = importlib.util.spec_from_file_location('legacy_recursive', {str(legacy_path)!r})
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
model = legacy.NN_Quadratic_Coupled_Recursive(
    Xi_generator=lambda M, D: data['Xi'].astype(np.float32),
    T=0.25, M=4, N=2, D=4, layers={layers!r}, parameters=params,
    t_start=0.0, t_end=0.25, T_total=0.25,
    terminal_blob=None, normalize_time_input=True,
    x_norm_mean=np.zeros((1, 4), dtype=np.float32),
    x_norm_std=np.ones((1, 4), dtype=np.float32),
)
model.import_parameter_blob(blob, strict=True)
X, Y, Z = model.predict(data['Xi'], data['t'], data['W'], const_value=1.0)
feed = {{
    model.Xi_tf: data['Xi'],
    model.t_tf: data['t'],
    model.W_tf: data['W'],
    model.learning_rate: np.float32(1.0e-3),
    model.const_tf: np.float32(1.0),
}}
model.sess.run(model.train_op, feed)
trained = model.export_parameter_blob()
post = {{"post_" + key: value for key, value in trained.items()}}
np.savez(out_path, X=X, Y=Y, Z=Z, **post)
model.sess.close()
"""
        new_script = common.replace(str(old_out), str(new_out)) + f"""
import tensorflow as tf
from final_recursive.models import NN_Quadratic_Coupled_Recursive
model = NN_Quadratic_Coupled_Recursive(
    Xi_generator=lambda M, D: data['Xi'].astype(np.float32),
    T=0.25, M=4, N=2, D=4, layers={layers!r}, parameters=params,
    t_start=0.0, t_end=0.25, T_total=0.25,
    terminal_blob=None, normalize_time_input=True,
    x_norm_mean=np.zeros((1, 4), dtype=np.float32),
    x_norm_std=np.ones((1, 4), dtype=np.float32),
)
model.import_parameter_blob(blob, strict=True)
X, Y, Z = model.predict(data['Xi'], data['t'], data['W'], const_value=1.0)
model._set_optimizer_learning_rate(1.0e-3)
model._train_step_tensor(
    tf.convert_to_tensor(data['t'], dtype=tf.float32),
    tf.convert_to_tensor(data['W'], dtype=tf.float32),
    tf.convert_to_tensor(data['Xi'], dtype=tf.float32),
    tf.constant(1.0, dtype=tf.float32),
)
trained = model.export_parameter_blob()
post = {{"post_" + key: value for key, value in trained.items()}}
np.savez(out_path, X=X, Y=Y, Z=Z, **post)
model.close()
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "code") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-c", old_script], check=True, env=env)
        subprocess.run([sys.executable, "-c", new_script], check=True, env=env)

        old = np.load(old_out)
        new = np.load(new_out)
        for key in ("X", "Y", "Z"):
            np.testing.assert_allclose(new[key], old[key], rtol=2.0e-5, atol=2.0e-5)
        for key in ("post_W_0", "post_b_0", "post_W_1", "post_b_1"):
            np.testing.assert_allclose(new[key], old[key], rtol=2.0e-5, atol=2.0e-5)


def run_tests(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run final_recursive tests")
    parser.add_argument(
        "--include-v1-parity",
        action="store_true",
        help="Also compare TF2 native predictions against the v1_compat source in a subprocess.",
    )
    args = parser.parse_args([] if argv is None else argv)

    cases: List[tuple[str, Callable[[], None]]] = [
        ("numpy_math_and_schedules", test_numpy_math_and_schedules),
        ("rollout_inputs_are_antithetic", test_rollout_inputs_are_antithetic),
        ("evaluation_bundle_rejects_block_metadata_mismatch", test_evaluation_bundle_rejects_block_metadata_mismatch),
        ("evaluation_bundle_rejects_wrong_bundle_kind", test_evaluation_bundle_rejects_wrong_bundle_kind),
        ("evaluation_bundle_loads_legacy_no_kind_as_evaluation_only", test_evaluation_bundle_loads_legacy_no_kind_as_evaluation_only),
        ("model_spec_contract", test_model_spec_contract),
        (
            "pascucci_model_spec_uses_paper_like_initial_distribution",
            test_pascucci_model_spec_uses_paper_like_initial_distribution,
        ),
        ("pascucci_tdd_contract_metadata", test_pascucci_tdd_contract_metadata),
        ("pascucci_model_layer_tdd_contract_metadata", test_pascucci_model_layer_tdd_contract_metadata),
        ("pascucci_oracle_fixture_tdd_contract_metadata", test_pascucci_oracle_fixture_tdd_contract_metadata),
        ("pascucci_equation_oracle_tdd_contract_metadata", test_pascucci_equation_oracle_tdd_contract_metadata),
        ("pascucci_prepare_H_hourly_mean_net_power_and_scale", test_pascucci_prepare_H_hourly_mean_net_power_and_scale),
        ("pascucci_prepare_H_missing_columns_raise", test_pascucci_prepare_H_missing_columns_raise),
        (
            "pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log",
            test_pascucci_prepare_S_xlsx_hourly_mean_comma_decimal_and_no_log,
        ),
        (
            "pascucci_prepare_S_missing_or_non_numeric_values_raise",
            test_pascucci_prepare_S_missing_or_non_numeric_values_raise,
        ),
        (
            "pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling",
            test_pascucci_calibrate_ou_variable_recovers_daynight_drift_dt_scaling,
        ),
        (
            "pascucci_calibrate_ou_variable_start_hour_controls_phase",
            test_pascucci_calibrate_ou_variable_start_hour_controls_phase,
        ),
        (
            "pascucci_calibrate_ou_variable_rejects_degenerate_inputs",
            test_pascucci_calibrate_ou_variable_rejects_degenerate_inputs,
        ),
        (
            "pascucci_calibration_output_contract_shapes",
            test_pascucci_calibration_output_contract_shapes,
        ),
        (
            "pascucci_calibrate_inputs_log_price_guard_and_parity",
            test_pascucci_calibrate_inputs_log_price_guard_and_parity,
        ),
        (
            "quadratic_spec_unaffected_by_pascucci_calibration_import",
            test_quadratic_spec_unaffected_by_pascucci_calibration_import,
        ),
        (
            "pascucci_ou_params_json_safe_after_serialization",
            test_pascucci_ou_params_json_safe_after_serialization,
        ),
        (
            "pascucci_day_night_boundary_semantics_are_explicit",
            test_pascucci_day_night_boundary_semantics_are_explicit,
        ),
        (
            "pascucci_log_price_false_calibrates_linear_prices",
            test_pascucci_log_price_false_calibrates_linear_prices,
        ),
        (
            "pascucci_calibration_config_records_units_log_price_dt_and_sources",
            test_pascucci_calibration_config_records_units_log_price_dt_and_sources,
        ),
        (
            "pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags",
            test_pascucci_build_run_config_params_injects_calibrated_ou_without_losing_solver_flags,
        ),
        (
            "pascucci_minimal_fixture_pipeline_builds_json_run_params",
            test_pascucci_minimal_fixture_pipeline_builds_json_run_params,
        ),
        (
            "pascucci_cli_calibration_paths_inject_run_config_params",
            test_cli_pascucci_calibration_paths_inject_run_config_params,
        ),
        (
            "pascucci_cli_require_calibration_fails_before_training_without_paths",
            test_cli_pascucci_require_calibration_fails_before_training_without_paths,
        ),
        ("model_spec_params_overlay_preserves_solver_flags", test_model_spec_params_overlay_preserves_solver_flags),
        ("exact_path_plot_outputs", test_exact_path_plot_outputs),
        (
            "pascucci_paper_plot_bundle_from_artifacts_smoke",
            test_pascucci_paper_plot_bundle_from_artifacts_smoke,
        ),
        (
            "pascucci_paper_plot_bundle_legacy_cost_trace_fallback",
            test_pascucci_paper_plot_bundle_legacy_cost_trace_fallback,
        ),
        (
            "pascucci_paper_plot_bundle_uses_to_ema_calibration_sources",
            test_pascucci_paper_plot_bundle_uses_to_ema_calibration_sources,
        ),
        (
            "pascucci_paper_plot_ou_calibration_sources_are_pass_invariant",
            test_pascucci_paper_plot_ou_calibration_sources_are_pass_invariant,
        ),
        (
            "pascucci_to_ema_ou_reference_rejects_basename_fallback_sha_mismatch",
            test_pascucci_to_ema_ou_reference_rejects_basename_fallback_sha_mismatch,
        ),
        (
            "pascucci_to_ema_ou_reference_matches_legacy_formula",
            test_pascucci_to_ema_ou_reference_matches_legacy_formula,
        ),
        (
            "pascucci_to_ema_ou_reference_honors_harmonics_and_start_hour",
            test_pascucci_to_ema_ou_reference_honors_harmonics_and_start_hour,
        ),
        (
            "pascucci_to_ema_uncontrolled_reference_matches_legacy_formula",
            test_pascucci_to_ema_uncontrolled_reference_matches_legacy_formula,
        ),
        (
            "pascucci_paper_plot_bundle_handles_skewed_total_costs",
            test_pascucci_paper_plot_bundle_handles_skewed_total_costs,
        ),
        (
            "cli_plot_pascucci_paper_from_artifacts_does_not_train",
            test_cli_plot_pascucci_paper_from_artifacts_does_not_train,
        ),
        (
            "cli_plot_pascucci_paper_resolves_single_nested_run_dir",
            test_cli_plot_pascucci_paper_resolves_single_nested_run_dir,
        ),
        (
            "pascucci_paper_plot_bundle_loads_blocks_from_recursive_results",
            test_pascucci_paper_plot_bundle_loads_blocks_from_recursive_results,
        ),
        (
            "pascucci_paper_plot_bundle_rejects_non_pascucci_run_config",
            test_pascucci_paper_plot_bundle_rejects_non_pascucci_run_config,
        ),
        (
            "pascucci_paper_plot_bundle_rejects_incompatible_application_schema",
            test_pascucci_paper_plot_bundle_rejects_incompatible_application_schema,
        ),
        (
            "pascucci_paper_plot_bundle_rejects_legacy_artifacts_without_cumulative_cost",
            test_pascucci_paper_plot_bundle_rejects_legacy_artifacts_without_cumulative_cost,
        ),
    ]

    ok = True
    for name, fn in cases:
        ok = _run_case(name, fn) and ok
    ok = _run_subprocess_case(
        "tf2_model_smoke_and_blob_roundtrip",
        "test_tf2_model_smoke_and_blob_roundtrip",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equation_fixtures",
        "test_pascucci_equation_fixtures",
    ) and ok
    ok = _run_subprocess_case(
        "model_spec_mean_field_moment_names",
        "test_model_spec_mean_field_moment_names",
    ) and ok
    ok = _run_subprocess_case(
        "model_spec_application_metric_names",
        "test_model_spec_application_metric_names",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_functional_decomposes_to_f_plus_g",
        "test_pascucci_application_cost_functional_decomposes_to_f_plus_g",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_trajectory_matches_plotmaker_nonuniform_grid",
        "test_pascucci_application_cost_trajectory_matches_plotmaker_nonuniform_grid",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_trajectory_uses_per_time_mean_field_oracle",
        "test_pascucci_application_cost_trajectory_uses_per_time_mean_field_oracle",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_summary_has_quantiles_and_metadata",
        "test_pascucci_application_cost_summary_has_quantiles_and_metadata",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_summary_handles_zero_step_cumulative",
        "test_pascucci_application_cost_summary_handles_zero_step_cumulative",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_functional_restores_const_state",
        "test_pascucci_application_cost_functional_restores_const_state",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_cost_from_path_rejects_invalid_baseline_mode",
        "test_pascucci_application_cost_from_path_rejects_invalid_baseline_mode",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_uncontrolled_baseline_is_paired_and_alpha_zero",
        "test_pascucci_uncontrolled_baseline_is_paired_and_alpha_zero",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_physical_tail_and_stitching_diagnostics_contract",
        "test_pascucci_physical_tail_and_stitching_diagnostics_contract",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_controlled_uncontrolled_comparison_is_paired_and_sign_safe",
        "test_pascucci_controlled_uncontrolled_comparison_is_paired_and_sign_safe",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_alpha_summary_is_controlled_only_and_plot_ready",
        "test_pascucci_application_alpha_summary_is_controlled_only_and_plot_ready",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_stitched_diagnostics_report_component_boundary_drift",
        "test_pascucci_stitched_diagnostics_report_component_boundary_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_application_pass_stability_uses_same_grid_pathwise_deltas",
        "test_pascucci_application_pass_stability_uses_same_grid_pathwise_deltas",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_cost_profile_default_is_exp_and_json_safe",
        "test_pascucci_cost_profile_default_is_exp_and_json_safe",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_cost_profile_exp_minus_offset_changes_only_running_cost",
        "test_pascucci_cost_profile_exp_minus_offset_changes_only_running_cost",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_cost_profile_rejects_unknown_profile",
        "test_pascucci_cost_profile_rejects_unknown_profile",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_recursive_cost_profile_matches_standard_formula",
        "test_pascucci_recursive_cost_profile_matches_standard_formula",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_physical_constraint_diagnostics_q_v_are_model_owned",
        "test_pascucci_physical_constraint_diagnostics_q_v_are_model_owned",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_q_v_barrier_drift_pushes_toward_physical_domain",
        "test_pascucci_q_v_barrier_drift_pushes_toward_physical_domain",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_q_v_barrier_sweep_with_nonzero_z_v",
        "test_pascucci_q_v_barrier_sweep_with_nonzero_z_v",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_fixture_generation_contract",
        "test_pascucci_oracle_fixture_generation_contract",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_fixture_reproducible_and_seed_sensitive",
        "test_pascucci_oracle_fixture_reproducible_and_seed_sensitive",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_fixture_save_load_roundtrip",
        "test_pascucci_oracle_fixture_save_load_roundtrip",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_provenance_metadata_roundtrip",
        "test_pascucci_oracle_provenance_metadata_roundtrip",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_fixture_missing_historical_reference_fails_fast",
        "test_pascucci_oracle_fixture_missing_historical_reference_fails_fast",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_oracle_fixture_cost_profile_variants_are_explicit",
        "test_pascucci_oracle_fixture_cost_profile_variants_are_explicit",
    ) and ok
    ok = _run_subprocess_case(
        "quadratic_spec_unaffected_by_pascucci_oracle_fixture_import",
        "test_quadratic_spec_unaffected_by_pascucci_oracle_fixture_import",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equation_oracle_final_model3_matches_tf2_fixture",
        "test_pascucci_equation_oracle_final_model3_matches_tf2_fixture",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equation_oracle_exp_minus_offset_variant",
        "test_pascucci_equation_oracle_exp_minus_offset_variant",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equation_oracle_uses_explicit_fixture_moments",
        "test_pascucci_equation_oracle_uses_explicit_fixture_moments",
    ) and ok
    ok = _run_subprocess_case(
        "quadratic_spec_unaffected_by_pascucci_equation_oracle_import",
        "test_quadratic_spec_unaffected_by_pascucci_equation_oracle_import",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_mean_field_moments_contract",
        "test_pascucci_mean_field_moments_contract",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equations_accept_explicit_mean_field_moments",
        "test_pascucci_equations_accept_explicit_mean_field_moments",
    ) and ok
    ok = _run_subprocess_case(
        "quadratic_loss_context_hook_is_noop",
        "test_quadratic_loss_context_hook_is_noop",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_loss_function_forwards_model_owned_context",
        "test_pascucci_loss_function_forwards_model_owned_context",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_runtime_moment_diagnostics_follow_loss_context",
        "test_pascucci_runtime_moment_diagnostics_follow_loss_context",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_runtime_physical_q_v_diagnostics_follow_loss_context",
        "test_pascucci_runtime_physical_q_v_diagnostics_follow_loss_context",
    ) and ok
    ok = _run_subprocess_case(
        "quadratic_runtime_moment_diagnostics_are_noop",
        "test_quadratic_runtime_moment_diagnostics_are_noop",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_evaluate_reports_block_end_moment_scalars",
        "test_pascucci_evaluate_reports_block_end_moment_scalars",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_fixed_eval_bundle_recomputes_moments_deterministically",
        "test_pascucci_fixed_eval_bundle_recomputes_moments_deterministically",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_fixed_eval_does_not_reuse_prior_live_batch_context",
        "test_pascucci_fixed_eval_does_not_reuse_prior_live_batch_context",
    ) and ok
    ok = _run_subprocess_case(
        "quadratic_fixed_eval_bundle_keeps_moment_policy_noop",
        "test_quadratic_fixed_eval_bundle_keeps_moment_policy_noop",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_f_mu_select_z_v_from_full_z",
        "test_pascucci_f_mu_select_z_v_from_full_z",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_equation_shapes_dtypes_finite",
        "test_pascucci_equation_shapes_dtypes_finite",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_recursive_loss_shapes_dtypes_finite",
        "test_pascucci_recursive_loss_shapes_dtypes_finite",
    ) and ok
    ok = _run_subprocess_case(
        "model_spec_recursive_factory_matches_direct_constructor",
        "test_model_spec_recursive_factory_matches_direct_constructor",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_recursive_factory_matches_direct_constructor",
        "test_pascucci_recursive_factory_matches_direct_constructor",
    ) and ok
    ok = _run_subprocess_case(
        "predict_recursive_stitched_two_block_model_spec",
        "test_predict_recursive_stitched_two_block_model_spec",
    ) and ok
    ok = _run_subprocess_case(
        "prefixed_eval_diagnostics_forwards_scalar_model_moments_only",
        "test_prefixed_eval_diagnostics_forwards_scalar_model_moments_only",
    ) and ok
    ok = _run_subprocess_case(
        "predict_recursive_stitched_pascucci_moment_traces",
        "test_predict_recursive_stitched_pascucci_moment_traces",
    ) and ok
    ok = _run_subprocess_case(
        "predict_recursive_stitched_carries_model_owned_diagnostics_without_recomputing",
        "test_predict_recursive_stitched_carries_model_owned_diagnostics_without_recomputing",
    ) and ok
    ok = _run_subprocess_case(
        "predict_recursive_stitched_schema_none_diagnostics_do_not_change_boundary_keyset",
        "test_predict_recursive_stitched_schema_none_diagnostics_do_not_change_boundary_keyset",
    ) and ok
    ok = _run_subprocess_case(
        "predict_recursive_stitched_accepts_legacy_three_value_predict",
        "test_predict_recursive_stitched_accepts_legacy_three_value_predict",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_saves_stitched_moment_traces",
        "test_print_recursive_pass_saves_stitched_moment_traces",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_application_metrics_emit_comparison_and_stability",
        "test_print_recursive_pass_application_metrics_emit_comparison_and_stability",
    ) and ok
    ok = _run_subprocess_case(
        "rollout_boundaries_accepts_fixed_initial_and_rollout_inputs",
        "test_rollout_boundaries_accepts_fixed_initial_and_rollout_inputs",
    ) and ok
    ok = _run_subprocess_case(
        "rollout_boundaries_rejects_fixed_input_shape_mismatch",
        "test_rollout_boundaries_rejects_fixed_input_shape_mismatch",
    ) and ok
    ok = _run_subprocess_case(
        "rollout_boundaries_rejects_malformed_fixed_rollout_inputs",
        "test_rollout_boundaries_rejects_malformed_fixed_rollout_inputs",
    ) and ok
    ok = _run_subprocess_case(
        "rollout_boundaries_rejects_nonmonotonic_rollout_times",
        "test_rollout_boundaries_rejects_nonmonotonic_rollout_times",
    ) and ok
    ok = _run_subprocess_case(
        "rollout_boundaries_legacy_path_still_uses_generator_and_minibatch",
        "test_rollout_boundaries_legacy_path_still_uses_generator_and_minibatch",
    ) and ok
    ok = _run_subprocess_case(
        "load_evaluation_bundle_rejects_nonfinite_payload",
        "test_load_evaluation_bundle_rejects_nonfinite_payload",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_fails_fast_on_nonfinite_eval_bundle_before_predict",
        "test_print_recursive_pass_fails_fast_on_nonfinite_eval_bundle_before_predict",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_wires_fixed_eval_and_gradient_diagnostics_to_logs",
        "test_run_recursive_training_wires_fixed_eval_and_gradient_diagnostics_to_logs",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_warm_start_norm_policy_blob_uses_blob_normalization",
        "test_run_recursive_training_warm_start_norm_policy_blob_uses_blob_normalization",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_blob_norm_policy_rejects_blob_without_norm_stats",
        "test_run_recursive_training_blob_norm_policy_rejects_blob_without_norm_stats",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_blob_norm_policy_rejects_invalid_norm_stats",
        "test_run_recursive_training_blob_norm_policy_rejects_invalid_norm_stats",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_rejects_unknown_warm_start_norm_policy",
        "test_run_recursive_training_rejects_unknown_warm_start_norm_policy",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_pass_boundary_rollouts_use_fixed_rollout_inputs",
        "test_run_recursive_training_pass_boundary_rollouts_use_fixed_rollout_inputs",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_negative_fixed_eval_seed_keeps_legacy_batches",
        "test_run_recursive_training_negative_fixed_eval_seed_keeps_legacy_batches",
    ) and ok
    ok = _run_subprocess_case(
        "run_recursive_training_copies_external_fixed_eval_batches_by_block",
        "test_run_recursive_training_copies_external_fixed_eval_batches_by_block",
    ) and ok
    ok = _run_subprocess_case(
        "score_pass_logs_rejects_nonfinite_rows",
        "test_score_pass_logs_rejects_nonfinite_rows",
    ) and ok
    ok = _run_subprocess_case(
        "resolve_pass_selection_rejects_no_finite_loss_scores",
        "test_resolve_pass_selection_rejects_no_finite_loss_scores",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_rejects_invalid_candidate_without_rewriting_final_artifacts",
        "test_print_recursive_pass_rejects_invalid_candidate_without_rewriting_final_artifacts",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_can_skip_intermediate_final_promotion",
        "test_print_recursive_pass_can_skip_intermediate_final_promotion",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_t12_gate_validator_requires_results_manifest_without_nameerror",
        "test_pascucci_t12_gate_validator_requires_results_manifest_without_nameerror",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_t12_gate_validator_green_contract_and_numeric_scalars",
        "test_pascucci_t12_gate_validator_green_contract_and_numeric_scalars",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_t12_gate_validator_rejects_science_red_selected_pass",
        "test_pascucci_t12_gate_validator_rejects_science_red_selected_pass",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_t12_gate_validator_reports_malformed_json_without_exception",
        "test_pascucci_t12_gate_validator_reports_malformed_json_without_exception",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_green_contract_and_wrapper_resolution",
        "test_pascucci_paper_plot_parity_validator_green_contract_and_wrapper_resolution",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_t24_legacy_semantics",
        "test_pascucci_paper_plot_parity_validator_rejects_t24_legacy_semantics",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_non_plotmaker_pascucci_params",
        "test_pascucci_paper_plot_parity_validator_rejects_non_plotmaker_pascucci_params",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_require_calibration_guard",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_require_calibration_guard",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_xi_initial_quantile_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_xi_initial_quantile_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_science_red_results",
        "test_pascucci_paper_plot_parity_validator_rejects_science_red_results",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_cost_accounting_mismatch",
        "test_pascucci_paper_plot_parity_validator_rejects_cost_accounting_mismatch",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_alpha_zero_as_plot40_paper_benchmark",
        "test_pascucci_paper_plot_parity_validator_rejects_alpha_zero_as_plot40_paper_benchmark",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_plot_style_contract",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_plot_style_contract",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_plot_dimension_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_plot_dimension_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_unsupported_pixel_exact_claim",
        "test_pascucci_paper_plot_parity_validator_rejects_unsupported_pixel_exact_claim",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_accepts_pixel_hash_golden_contract",
        "test_pascucci_paper_plot_parity_validator_accepts_pixel_hash_golden_contract",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_pixel_hash_golden_mismatch",
        "test_pascucci_paper_plot_parity_validator_rejects_pixel_hash_golden_mismatch",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_nonpaper_cost_profile",
        "test_pascucci_paper_plot_parity_validator_rejects_nonpaper_cost_profile",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_plot_data_npz",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_plot_data_npz",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_yz",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_yz",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_plot_data",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_plotmaker_native_plot_data",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_wrong_plotmaker_dataset",
        "test_pascucci_paper_plot_parity_validator_rejects_wrong_plotmaker_dataset",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_plot_data_stat_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_plot_data_stat_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_calibration_source_hash_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_calibration_source_hash_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_recomputed_calibration_param_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_recomputed_calibration_param_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_missing_mc_confirmation",
        "test_pascucci_paper_plot_parity_validator_rejects_missing_mc_confirmation",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_mc_confirmation_schema_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_schema_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_mc_confirmation_not_independent",
        "test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_not_independent",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_mc_confirmation_sample_count_drift",
        "test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_sample_count_drift",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_rejects_mc_confirmation_science_red",
        "test_pascucci_paper_plot_parity_validator_rejects_mc_confirmation_science_red",
    ) and ok
    ok = _run_subprocess_case(
        "pascucci_paper_plot_parity_validator_mc_violation_rate_uses_positive_indicator",
        "test_pascucci_paper_plot_parity_validator_mc_violation_rate_uses_positive_indicator",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_rejects_mismatched_eval_bundle_metadata",
        "test_print_recursive_pass_rejects_mismatched_eval_bundle_metadata",
    ) and ok
    ok = _run_subprocess_case(
        "recursive_coarse_prepass_model_spec_argument",
        "test_recursive_coarse_prepass_model_spec_argument",
    ) and ok
    ok = _run_subprocess_case(
        "cli_coarse_boundary_only_and_blob_norm_flags_wire_to_recursive",
        "test_cli_coarse_boundary_only_and_blob_norm_flags_wire_to_recursive",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_coarse_boundary_only_with_pass1_warm_start_from_next",
        "test_cli_rejects_coarse_boundary_only_with_pass1_warm_start_from_next",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_mc_confirmation_for_quadratic_model",
        "test_cli_rejects_mc_confirmation_for_quadratic_model",
    ) and ok
    ok = _run_subprocess_case(
        "cli_tiny_standard_and_both_model_spec_outputs",
        "test_cli_tiny_standard_and_both_model_spec_outputs",
    ) and ok
    ok = _run_subprocess_case(
        "cli_wires_recursive_diagnostic_flags_and_separate_rollout_bundle",
        "test_cli_wires_recursive_diagnostic_flags_and_separate_rollout_bundle",
    ) and ok
    ok = _run_subprocess_case(
        "cli_recursive_two_pass_callbacks_finalize_selection_only_at_last_pass",
        "test_cli_recursive_two_pass_callbacks_finalize_selection_only_at_last_pass",
    ) and ok
    ok = _run_subprocess_case(
        "cli_records_pascucci_cost_profile_params_in_run_config",
        "test_cli_records_pascucci_cost_profile_params_in_run_config",
    ) and ok
    ok = _run_subprocess_case(
        "cli_records_application_metric_manifest_in_run_config",
        "test_cli_records_application_metric_manifest_in_run_config",
    ) and ok
    ok = _run_subprocess_case(
        "run_config_sha256_excludes_timestamp_metadata",
        "test_run_config_sha256_excludes_timestamp_metadata",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_pascucci_cost_profile_for_quadratic_model",
        "test_cli_rejects_pascucci_cost_profile_for_quadratic_model",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_pascucci_calibration_paths_for_quadratic_model",
        "test_cli_rejects_pascucci_calibration_paths_for_quadratic_model",
    ) and ok
    ok = _run_subprocess_case(
        "cli_records_pascucci_cost_profile_params_in_recursive_run_config",
        "test_cli_records_pascucci_cost_profile_params_in_recursive_run_config",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_unsupported_model_argument",
        "test_cli_rejects_unsupported_model_argument",
    ) and ok
    ok = _run_subprocess_case(
        "cli_rejects_pascucci_exact_profile",
        "test_cli_rejects_pascucci_exact_profile",
    ) and ok
    ok = _run_subprocess_case(
        "cli_uses_resolved_model_spec_for_runtime_wiring",
        "test_cli_uses_resolved_model_spec_for_runtime_wiring",
    ) and ok
    ok = _run_subprocess_case(
        "loss_improvement_flags_are_finite",
        "test_loss_improvement_flags_are_finite",
    ) and ok
    ok = _run_subprocess_case(
        "terminal_blob_constants_survive_train_graph",
        "test_terminal_blob_constants_survive_train_graph",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_uses_fixed_evaluation_batches",
        "test_train_with_standard_schedule_uses_fixed_evaluation_batches",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_accepts_kwargs_evaluate_adapter",
        "test_train_with_standard_schedule_accepts_kwargs_evaluate_adapter",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_fixed_eval_passes_integer_batch_count",
        "test_train_with_standard_schedule_fixed_eval_passes_integer_batch_count",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_fixed_eval_without_bundle_api",
        "test_train_with_standard_schedule_rejects_fixed_eval_without_bundle_api",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_noninspectable_fixed_eval_api",
        "test_train_with_standard_schedule_rejects_noninspectable_fixed_eval_api",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_positional_only_fixed_eval_api",
        "test_train_with_standard_schedule_rejects_positional_only_fixed_eval_api",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_malformed_fixed_eval_batches",
        "test_train_with_standard_schedule_rejects_malformed_fixed_eval_batches",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_model_incompatible_fixed_eval_horizon",
        "test_train_with_standard_schedule_rejects_model_incompatible_fixed_eval_horizon",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_zero_path_fixed_eval_batches",
        "test_train_with_standard_schedule_rejects_zero_path_fixed_eval_batches",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_fixed_eval_bundle_with_real_fbsnn",
        "test_train_with_standard_schedule_fixed_eval_bundle_with_real_fbsnn",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_aggregates_gradient_diagnostic_maxima",
        "test_train_with_standard_schedule_aggregates_gradient_diagnostic_maxima",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_gradient_diagnostics_without_train_api",
        "test_train_with_standard_schedule_rejects_gradient_diagnostics_without_train_api",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_reports_gradient_norm_diagnostics",
        "test_model_train_reports_gradient_norm_diagnostics",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_gradient_diagnostics_marks_disabled_clipping",
        "test_model_train_gradient_diagnostics_marks_disabled_clipping",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_uses_fixed_evaluation_batches_for_best_snapshot",
        "test_model_train_uses_fixed_evaluation_batches_for_best_snapshot",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_rejects_malformed_fixed_evaluation_batches",
        "test_model_train_rejects_malformed_fixed_evaluation_batches",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_rejects_wrong_time_horizon_fixed_evaluation_batches",
        "test_model_train_rejects_wrong_time_horizon_fixed_evaluation_batches",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_rejects_wrong_time_horizon_before_update",
        "test_model_train_rejects_wrong_time_horizon_before_update",
    ) and ok
    ok = _run_subprocess_case(
        "model_train_rejects_nonfinite_loss",
        "test_model_train_rejects_nonfinite_loss",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_nonfinite_stage",
        "test_train_with_standard_schedule_rejects_nonfinite_stage",
    ) and ok
    ok = _run_subprocess_case(
        "train_with_standard_schedule_rejects_nonfinite_secondary_eval_stats",
        "test_train_with_standard_schedule_rejects_nonfinite_secondary_eval_stats",
    ) and ok
    ok = _run_subprocess_case(
        "recursive_visual_bundle_acceptance",
        "test_recursive_visual_bundle_acceptance",
    ) and ok
    ok = _run_subprocess_case(
        "same_xi_antithetic_minibatch",
        "test_same_xi_antithetic_minibatch",
    ) and ok
    ok = _run_subprocess_case(
        "z_selection_metrics",
        "test_z_selection_metrics",
    ) and ok
    if args.include_v1_parity:
        ok = _run_subprocess_case("v1_prediction_parity", "test_v1_prediction_parity") and ok
    return 0 if ok else 1
