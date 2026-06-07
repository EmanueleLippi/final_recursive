"""In-package test runner used by `python -m final_recursive test`."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List
from xml.sax.saxutils import escape

import numpy as np

from .exact import (
    build_exact_solution_functions,
    quadratic_coupled_exact_z_np,
    quadratic_coupled_mu_np,
)
from .sampling import build_blocks, build_stitched_rollout_inputs, Xi_generator_default
from .schedules import parse_float_sequence_arg, resolve_coarse_curriculum_schedule


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
        "l_a": np.float32(0.01),
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
        "c_h": np.float32(0.0001),
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


def test_model_spec_contract() -> None:
    from .model_specs import get_model_spec

    spec = get_model_spec()
    assert spec.name == "quadratic_coupled"
    assert spec.state_dim == 4
    assert spec.state_labels == ("S", "H", "V", "X_state")
    assert spec.z_labels == ("Z_S", "Z_H", "Z_V", "Z_X")
    assert spec.build_layers(4) == [5, 256, 256, 256, 256, 1]

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

    pascucci_params = pascucci_spec.build_default_params(const=0.55)
    assert pascucci_params["const"] == np.float32(0.55)
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
}


def _assert_pascucci_tdd_contract(name: str) -> None:
    contract = PASCUCCI_CALIBRATION_TDD_CONTRACTS[name]
    for key in ("type", "target", "purpose", "expected", "failure"):
        value = str(contract.get(key, "")).strip()
        assert value, f"{name} missing TDD contract field {key}"


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
    }
    assert set(PASCUCCI_CALIBRATION_TDD_CONTRACTS) == expected_names
    for name in expected_names:
        _assert_pascucci_tdd_contract(name)


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
        recovered_sigmas[dt] = (float(noisy_params["sigma_day"]), float(noisy_params["sigma_night"]))
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
        ("too few regression rows", np.linspace(0.0, 1.0, 8), {"K": 1, "dt": 3.0, "start_hour": 4.0}),
        ("missing night regime", np.linspace(0.0, 1.0, 8), {"K": 0, "dt": 1.0, "start_hour": 7.0}),
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
        ),
    ]
    for label, series, kwargs in invalid_cases:
        try:
            calibrate_OU_variable(series, **kwargs)
        except ValueError as exc:
            assert str(exc).strip(), label
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
    for key, expected_trace in expected.items():
        assert key in stitched, f"{key} missing from stitched keys: {sorted(stitched)}"
        assert stitched[key].shape == (len(blocks) * N_per_block + 1, 1)
        assert stitched[key].dtype == np.float32
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
            X = np.zeros((M, steps, self.D), dtype=np.float32)
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
    assert np.all(stitched["X"] == 0.0)


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
    assert set(stitched) == {"t", "X", "Y", "Z"}
    assert stitched["t"].shape == (M, 5, 1)
    assert stitched["X"].shape == (M, 5, D)
    assert stitched["Y"].shape == (M, 5, 1)
    assert stitched["Z"].shape == (M, 5, D)


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
                    for key in ("mean_v", "mean_q", "mean_h_plus_v"):
                        assert key in keys, f"{key} missing from {filename}: {sorted(keys)}"
                        assert data[key].shape == (3, 1)
                    assert "block_params" not in keys
                    assert not any(key.startswith(("params_", "weights_")) for key in keys)
    finally:
        orchestration.predict_recursive_stitched = original_predict
        orchestration.plot_recursive_pass_logs_multi = original_plot_logs
        orchestration.plot_recursive_stitched_predictions = original_plot_stitched
        orchestration.plot_recursive_stitched_y_convergence = original_plot_convergence


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


def test_cli_tiny_standard_and_both_model_spec_outputs() -> None:
    import json

    from . import cli, orchestration
    from .sampling import build_blocks
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
        assert len(print_calls) == 3
    finally:
        orchestration.run_standard_reference = original_run_standard_reference
        orchestration.run_recursive_training = original_run_recursive_training
        orchestration.print_recursive_pass = original_print_recursive_pass
        cli.export_standard_parameter_blob = original_export_standard_parameter_blob
        cli.plot_stage_logs = original_plot_stage_logs


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
        assert (tmp_path / "visual_stitched_predictions_pass00.npz").exists()
        assert (tmp_path / "visual_stitched_predictions_final.npz").exists()
        assert (tmp_path / "exact_metrics_final.json").exists()
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
        ("model_spec_contract", test_model_spec_contract),
        ("pascucci_tdd_contract_metadata", test_pascucci_tdd_contract_metadata),
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
        ("model_spec_params_overlay_preserves_solver_flags", test_model_spec_params_overlay_preserves_solver_flags),
        ("exact_path_plot_outputs", test_exact_path_plot_outputs),
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
        "predict_recursive_stitched_accepts_legacy_three_value_predict",
        "test_predict_recursive_stitched_accepts_legacy_three_value_predict",
    ) and ok
    ok = _run_subprocess_case(
        "print_recursive_pass_saves_stitched_moment_traces",
        "test_print_recursive_pass_saves_stitched_moment_traces",
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
        "cli_tiny_standard_and_both_model_spec_outputs",
        "test_cli_tiny_standard_and_both_model_spec_outputs",
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
