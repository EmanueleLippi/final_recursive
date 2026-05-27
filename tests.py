"""In-package test runner used by `python -m final_recursive test`."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List

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
