"""High-level standard and recursive training orchestration."""

from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .application_metrics import (
    summarize_application_alpha,
    summarize_application_pass_stability,
    summarize_controlled_uncontrolled_comparison,
    summarize_pascucci_stitched_diagnostics,
)
from .exact import compute_stitched_exact_bundle, save_exact_error_timeseries_csv
from .io_utils import _as_blob_dict, save_blob_npz, save_json, save_rows_csv
from .models import FBSNN
from .naming import _pass_index, _pass_label, _pass_tag
from .plotting import (
    plot_recursive_exact_comparison,
    plot_recursive_pass_logs_multi,
    plot_recursive_stitched_predictions,
    plot_recursive_stitched_y_convergence,
    score_pass_logs,
)
from .sampling import (
    build_blocks,
    build_stitched_rollout_inputs,
    estimate_generator_stats,
    load_evaluation_bundle,
    make_empirical_generator,
    save_evaluation_bundle,
    summarize_boundary_samples,
    validate_boundary_samples,
)
from .schedules import (
    _const_stage_tag,
    resolve_coarse_curriculum_schedule,
    resolve_training_plan_for_block,
    scale_schedule,
    scale_training_plan_rules,
)
from .tf_backend import reset_backend_state

from .model_specs import ModelSpec, get_model_spec

_BASE_EVAL_LOG_KEYS = {
    "mean_loss",
    "std_loss",
    "mean_loss_per_sample",
    "std_loss_per_sample",
    "mean_y0",
    "std_y0",
}

def _resolve_model_spec(model_spec: Optional[ModelSpec]) -> ModelSpec:
    return get_model_spec() if model_spec is None else model_spec

def _prefixed_eval_diagnostics(eval_stats: Dict[str, Any]) -> Dict[str, float]:
    diagnostics = {}
    for key, value in eval_stats.items():
        key_str = str(key)
        if key_str in _BASE_EVAL_LOG_KEYS:
            continue
        if not (
            key_str.startswith("mean_loss_")
            or key_str.startswith("std_loss_")
            or key_str.startswith("mean_block_end_")
            or key_str.startswith("std_block_end_")
        ):
            continue
        try:
            value_array = np.asarray(value)
            if value_array.shape != ():
                continue
            scalar = float(value_array)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            diagnostics[f"eval_{key_str}"] = scalar
    return diagnostics

def _require_finite_schedule_scalar(value: Any, *, label: str, phase: str, key: str) -> float:
    scalar = float(np.asarray(value))
    if not np.isfinite(scalar):
        raise RuntimeError(f"{label} {phase} produced non-finite {key}: {scalar}")
    return scalar

def _require_finite_schedule_stats(
    stats: Dict[str, Any],
    *,
    label: str,
    phase: str,
    keys: Tuple[str, ...],
) -> None:
    for key in keys:
        if key in stats and stats[key] is not None:
            _require_finite_schedule_scalar(stats[key], label=label, phase=phase, key=key)

def _callable_accepts_keywords(callable_obj: Any, keyword_names: Tuple[str, ...]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    keyword_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    return all(name in parameters and parameters[name].kind in keyword_kinds for name in keyword_names)

def _optional_int_attr(obj: Any, name: str) -> Optional[int]:
    if obj is None or not hasattr(obj, name):
        return None
    try:
        return int(getattr(obj, name))
    except (TypeError, ValueError):
        return None

def _validate_fixed_evaluation_batches(
    eval_batches: Any,
    model: Optional[Any] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fixed_batches = list(eval_batches)
    if not fixed_batches:
        raise ValueError("fixed evaluation batches must contain at least one (t, W, Xi) batch")
    expected_M = _optional_int_attr(model, "M")
    expected_N = _optional_int_attr(model, "N")
    expected_D = _optional_int_attr(model, "D")
    validated = []
    for idx, batch in enumerate(fixed_batches):
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(f"fixed evaluation batches[{idx}] must be a (t, W, Xi) triple")
        t_batch, W_batch, Xi_batch = batch
        t_arr = np.asarray(t_batch, dtype=np.float32)
        W_arr = np.asarray(W_batch, dtype=np.float32)
        Xi_arr = np.asarray(Xi_batch, dtype=np.float32)
        if t_arr.ndim != 3 or t_arr.shape[2] != 1:
            raise ValueError(f"fixed evaluation batches[{idx}][0] must have shape (M, N+1, 1)")
        if W_arr.ndim != 3:
            raise ValueError(f"fixed evaluation batches[{idx}][1] must have shape (M, N+1, D)")
        if Xi_arr.ndim != 2:
            raise ValueError(f"fixed evaluation batches[{idx}][2] must have shape (M, D)")
        if t_arr.shape[0] == 0:
            raise ValueError(f"fixed evaluation batches[{idx}] must contain at least one path")
        if t_arr.shape[1] <= 1:
            raise ValueError(f"fixed evaluation batches[{idx}] must contain at least one time step")
        if expected_M is not None and t_arr.shape[0] != expected_M:
            raise ValueError(
                f"fixed evaluation batches[{idx}] batch size must match model M={expected_M}, got {t_arr.shape[0]}"
            )
        if expected_N is not None and t_arr.shape[1] != expected_N + 1:
            raise ValueError(
                f"fixed evaluation batches[{idx}] time dimension must match model N+1={expected_N + 1}, "
                f"got {t_arr.shape[1]}"
            )
        if t_arr.shape[0] != W_arr.shape[0] or t_arr.shape[0] != Xi_arr.shape[0]:
            raise ValueError(f"fixed evaluation batches[{idx}] batch dimension mismatch")
        if t_arr.shape[1] != W_arr.shape[1]:
            raise ValueError(f"fixed evaluation batches[{idx}] time dimension mismatch")
        if W_arr.shape[2] != Xi_arr.shape[1]:
            raise ValueError(f"fixed evaluation batches[{idx}] state dimension mismatch")
        if expected_D is not None and (W_arr.shape[2] != expected_D or Xi_arr.shape[1] != expected_D):
            raise ValueError(
                f"fixed evaluation batches[{idx}] state dimension must match model D={expected_D}, "
                f"got W D={W_arr.shape[2]} and Xi D={Xi_arr.shape[1]}"
            )
        if not np.isfinite(t_arr).all() or not np.isfinite(W_arr).all() or not np.isfinite(Xi_arr).all():
            raise ValueError(f"fixed evaluation batches[{idx}] contains non-finite values")
        validated.append((t_arr, W_arr, Xi_arr))
    return validated

def _evaluate_with_schedule_batches(model: FBSNN, *, const_value: float, eval_batches: Any) -> Dict[str, Any]:
    eval_batches = _normalize_schedule_eval_batches(eval_batches, model=model)
    if isinstance(eval_batches, (int, np.integer)):
        return model.evaluate(const_value=const_value, n_batches=int(eval_batches))
    fixed_batches = eval_batches
    required_kwargs = ("const_value", "n_batches", "evaluation_batches", "moment_policy")
    if not _callable_accepts_keywords(model.evaluate, required_kwargs):
        raise TypeError(
            "fixed evaluation batches require model.evaluate to accept "
            "const_value, n_batches, evaluation_batches, and moment_policy"
        )
    return model.evaluate(
        const_value=const_value,
        n_batches=len(fixed_batches),
        evaluation_batches=fixed_batches,
        moment_policy="fixed_evaluation_bundle",
    )

def _normalize_schedule_eval_batches(eval_batches: Any, model: Optional[Any] = None) -> Any:
    if isinstance(eval_batches, (int, np.integer)):
        return int(eval_batches)
    return _validate_fixed_evaluation_batches(eval_batches, model=model)

def _train_evaluation_kwargs(eval_batches: Any) -> Dict[str, Any]:
    if isinstance(eval_batches, (int, np.integer)):
        return {}
    return {
        "evaluation_batches": eval_batches,
        "moment_policy": "fixed_evaluation_bundle",
    }

def _nonfinite_stitched_reason(stitched: Dict[str, np.ndarray], *, pass_label: str) -> Optional[str]:
    required = ("t", "X", "Y", "Z")
    for key in required:
        if key not in stitched:
            return f"{pass_label} stitched predictions missing required key '{key}'"
    for key, value in stitched.items():
        arr = np.asarray(value)
        if arr.dtype.kind not in "fciu":
            continue
        if arr.size == 0:
            return f"{pass_label} stitched predictions key '{key}' is empty"
        if not np.isfinite(arr).all():
            return f"{pass_label} stitched predictions key '{key}' contains non-finite values"
    return None


def _stitch_rollout_inputs(
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    t_segments = []
    W_segments = []
    previous_W_terminal = None
    for idx, (t_block, W_block) in enumerate(rollout_inputs):
        t_np = np.asarray(t_block, dtype=np.float32)
        W_np = np.asarray(W_block, dtype=np.float32)
        if previous_W_terminal is not None:
            W_np = W_np - W_np[:, :1, :] + previous_W_terminal[:, None, :]
        start_idx = 0 if idx == 0 else 1
        t_segments.append(t_np[:, start_idx:, :])
        W_segments.append(W_np[:, start_idx:, :])
        previous_W_terminal = W_np[:, -1, :].astype(np.float32)
    return (
        np.concatenate(t_segments, axis=1).astype(np.float32),
        np.concatenate(W_segments, axis=1).astype(np.float32),
    )


def _summarize_application_cost_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "schema": result["schema"],
        "metadata": dict(result.get("metadata", {})),
        "summary": dict(result.get("summary", {})),
    }
    pathwise = result.get("pathwise", {})
    if "alpha" in pathwise:
        payload["alpha_summary"] = summarize_application_alpha(
            pathwise["alpha"],
            baseline_mode=payload["metadata"].get("baseline_mode", ""),
        )
    return payload


def _application_pathwise_npz(result: Dict[str, Any], prefix: str) -> Dict[str, np.ndarray]:
    arrays = {}
    for key, value in result.get("pathwise", {}).items():
        arrays[f"{prefix}_{key}"] = np.asarray(value, dtype=np.float32)
    return arrays


def _application_z_v_index(model_spec: ModelSpec) -> int:
    z_labels = tuple(str(label) for label in getattr(model_spec, "z_labels", ()))
    if "Z_V" not in z_labels:
        raise ValueError("application pass stability requires model_spec.z_labels to include 'Z_V'")
    return int(z_labels.index("Z_V"))


def _compute_application_metric_artifact(
    *,
    stitched: Dict[str, np.ndarray],
    blocks: List[Dict[str, float]],
    rollout_inputs: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    params: Dict[str, np.ndarray],
    D: int,
    layers: List[int],
    T_total: float,
    coupling_const: float,
    model_spec: ModelSpec,
) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    schema = str(getattr(model_spec, "application_metric_schema", "none"))
    if schema == "none":
        return None

    t = np.asarray(stitched["t"], dtype=np.float32)
    X = np.asarray(stitched["X"], dtype=np.float32)
    Y = np.asarray(stitched["Y"], dtype=np.float32)
    Z = np.asarray(stitched["Z"], dtype=np.float32)
    if t.shape[1] < 2:
        raise ValueError("application metrics require at least two stitched time points")

    reset_backend_state()
    model = model_spec.build_recursive_model(
        Xi_generator=make_empirical_generator(X[:, 0, :], jitter_scale=0.0),
        T=float(T_total),
        M=int(X.shape[0]),
        N=int(t.shape[1] - 1),
        D=int(D),
        layers=layers,
        parameters=params,
        t_start=float(t[0, 0, 0]),
        t_end=float(t[0, -1, 0]),
        T_total=float(T_total),
        terminal_blob=None,
    )
    try:
        if not hasattr(model, "application_cost_from_path"):
            return None

        controlled = model.application_cost_from_path(
            t,
            X,
            Y,
            Z,
            const_value=float(coupling_const),
            baseline_mode="controlled",
            control_law="alpha_tf",
            paired_inputs="stitched_XYZ",
        )
        pathwise_npz = _application_pathwise_npz(controlled, "controlled")
        payload = {
            "schema": schema,
            "model_name": model_spec.name,
            "horizon": {
                "t_start": float(t[0, 0, 0]),
                "t_end": float(t[0, -1, 0]),
                "T_total": float(T_total),
                "n_time_points": int(t.shape[1]),
                "n_steps": int(t.shape[1] - 1),
                "sample_paths": int(X.shape[0]),
            },
            "controlled": _summarize_application_cost_result(controlled),
            "diagnostics": summarize_pascucci_stitched_diagnostics(
                stitched=stitched,
                blocks=blocks,
                params=params,
                state_labels=getattr(model_spec, "state_labels", None),
            ),
        }

        if rollout_inputs is not None and hasattr(model, "application_cost_functional"):
            t_rollout, W_rollout = _stitch_rollout_inputs(rollout_inputs)
            if t_rollout.shape != t.shape or not np.allclose(t_rollout, t, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError("application metrics require rollout inputs on the stitched time grid")
            uncontrolled = model.application_cost_functional(
                t_rollout,
                W_rollout,
                X[:, 0, :],
                const_value=float(coupling_const),
                baseline_mode="uncontrolled",
            )
            pathwise_npz.update(_application_pathwise_npz(uncontrolled, "uncontrolled"))
            payload["uncontrolled"] = _summarize_application_cost_result(uncontrolled)
            payload["comparison"] = summarize_controlled_uncontrolled_comparison(
                controlled=controlled,
                uncontrolled=uncontrolled,
            )

        payload["pathwise_npz_keys"] = sorted(pathwise_npz)
        return payload, pathwise_npz
    finally:
        model.close()


def print_recursive_pass(
    pass_entries: List[Dict[str, Any]],
    blocks: List[Dict[str, float]],
    rec_dir: str,
    params: Dict[str, np.ndarray],
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    exact_solution: Optional[Dict[str, Any]],
    selection_metric: str = "auto",
    exact_regression_tolerance: float = 0.20,
    exact_regression_action: str = "warn",
    eval_bundle_path: str = "",
    eval_seed: int = 1234,
    eval_min_paths: int = 64,
    sample_paths: int = 8,
    visual_sample_paths: Optional[int] = None,
    visual_seed: Optional[int] = None,
    enforce_exact_regression_guardrail: bool = True,
    print_compact_logs: bool = True,
    promote_final_artifacts: bool = True,
    exclude_pass_ids_from_selection: Optional[List[int]] = None,
    coupling_const: float = 1.0,
    model_spec: Optional[ModelSpec] = None,
) -> Dict[str, Any]:
    if pass_entries is None or len(pass_entries) == 0:
        raise RuntimeError("print_recursive_pass called with empty pass_entries")
    spec = _resolve_model_spec(model_spec)
    spec.validate_state_dim(D)

    pass_entries = sorted(pass_entries, key=lambda x: int(x["pass_id"]))
    os.makedirs(rec_dir, exist_ok=True)
    plots_dir = os.path.join(rec_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pass_logs_by_pass = {}
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_idx = _pass_index(pass_id)
        logs = p.get("logs", [])
        pass_logs_by_pass[pass_id] = logs

        if print_compact_logs:
            print(f"\n=== Recursive Log {_pass_label(pass_id)} (compact) ===")
            for row in logs:
                norm_msg = ""
                if "eval_mean_loss_per_sample" in row:
                    norm_msg = f", eval_loss/M={row['eval_mean_loss_per_sample']:.3e}"
                freeze_msg = ""
                if bool(row.get("was_frozen", False)):
                    freeze_reason = str(row.get("freeze_reason", "") or "").strip()
                    freeze_msg = ", frozen"
                    if freeze_reason != "":
                        freeze_msg += f"({freeze_reason})"
                print(
                    f"block={row['block']}, t=[{row['t_start']:.1f},{row['t_end']:.1f}], "
                    f"eval_loss={row['eval_mean_loss']:.3e}{norm_msg}, eval_y0={row['eval_mean_y0']:.3f}, "
                    f"target={row['precision_target']}, refine={row['refine_rounds']}{freeze_msg}"
                )

        save_rows_csv(logs, os.path.join(rec_dir, f"pass_{pass_idx:02d}_logs.csv"))
        if pass_idx == 0:
            save_rows_csv(logs, os.path.join(rec_dir, "pass0_logs.csv"))
        if pass_idx == 1:
            save_rows_csv(logs, os.path.join(rec_dir, "pass1_logs.csv"))

    plot_recursive_pass_logs_multi(pass_logs_by_pass, plots_dir)

    score_key = "eval_mean_loss_per_sample"
    all_rows = [row for rows in pass_logs_by_pass.values() for row in rows]
    if not all(score_key in row for row in all_rows):
        score_key = "eval_mean_loss"
    pass_scores_loss = {
        int(pass_id): score_pass_logs(rows, loss_key=score_key)
        for pass_id, rows in pass_logs_by_pass.items()
        if len(rows) > 0
    }
    if len(pass_scores_loss) == 0:
        raise RuntimeError("No pass logs available for pass selection")
    pass_invalid_reasons = {
        int(pass_id): f"{_pass_label(int(pass_id))} has non-finite or missing {score_key}"
        for pass_id, score in pass_scores_loss.items()
        if not np.isfinite(float(score))
    }

    eval_bundle_path = str(eval_bundle_path or "").strip()
    if eval_bundle_path == "":
        eval_bundle_path = os.path.join(rec_dir, "evaluation_bundle.npz")
    eval_bundle_path = os.path.abspath(os.path.expanduser(eval_bundle_path))

    if os.path.isfile(eval_bundle_path):
        Xi_stitched, rollout_inputs = load_evaluation_bundle(
            path=eval_bundle_path,
            n_blocks_expected=len(blocks),
            N_per_block_expected=N_per_block,
            D_expected=D,
            blocks_expected=blocks,
            T_total_expected=T_total,
        )
        print(
            f"[EvalBundle] loaded path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"blocks={len(rollout_inputs)}"
        )
    else:
        Xi_stitched = spec.deterministic_xi(
            max(1, int(eval_min_paths)),
            D,
            seed=int(eval_seed),
        )
        rollout_inputs = build_stitched_rollout_inputs(
            blocks=blocks,
            M=Xi_stitched.shape[0],
            N_per_block=N_per_block,
            D=D,
            seed=int(eval_seed),
        )
        save_evaluation_bundle(
            path=eval_bundle_path,
            Xi_initial=Xi_stitched,
            rollout_inputs=rollout_inputs,
            blocks=blocks,
        )
        print(
            f"[EvalBundle] created path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"seed={int(eval_seed)}"
        )

    visual_path_count = int(sample_paths if visual_sample_paths is None else visual_sample_paths)
    visual_seed_effective = int(eval_seed) + 7919 if visual_seed is None else int(visual_seed)
    Xi_visual = None
    visual_rollout_inputs = None
    if exact_solution is not None and visual_path_count > 0:
        Xi_visual = spec.deterministic_xi(
            visual_path_count,
            D,
            seed=visual_seed_effective,
        )
        visual_rollout_inputs = build_stitched_rollout_inputs(
            blocks=blocks,
            M=Xi_visual.shape[0],
            N_per_block=N_per_block,
            D=D,
            seed=visual_seed_effective,
        )
        print(
            f"[VisualBundle] created M={Xi_visual.shape[0]}, seed={visual_seed_effective}"
        )

    stitched_by_pass = {}
    exact_summary_by_pass = {}
    exact_bundle_by_pass = {}
    visual_stitched_by_pass = {}
    visual_exact_bundle_by_pass = {}
    application_summary_by_pass = {}
    application_pathwise_by_pass = {}
    application_stability_by_pass = {}
    previous_application_pass_id = None
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_tag = _pass_tag(pass_id)
        if pass_id in pass_invalid_reasons:
            print(f"[PassValidity] {_pass_label(pass_id)} invalid: {pass_invalid_reasons[pass_id]}")
            continue
        stitched_pred = predict_recursive_stitched(
            block_blobs=p["blobs"],
            blocks=blocks,
            Xi_initial=Xi_stitched,
            params=params,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
            rollout_inputs=rollout_inputs,
            coupling_const=float(coupling_const),
            model_spec=spec,
        )
        stitched_by_pass[pass_id] = stitched_pred

        save_blob_npz(
            stitched_pred,
            os.path.join(rec_dir, f"stitched_predictions_{pass_tag}.npz"),
        )
        stitched_invalid_reason = _nonfinite_stitched_reason(
            stitched_pred,
            pass_label=_pass_label(pass_id),
        )
        if stitched_invalid_reason is not None:
            pass_scores_loss[pass_id] = float("inf")
            pass_invalid_reasons[pass_id] = stitched_invalid_reason
            print(f"[PassValidity] {_pass_label(pass_id)} invalid: {stitched_invalid_reason}")
            continue
        plot_recursive_stitched_predictions(
            stitched=stitched_pred,
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix=f"_{pass_tag}",
        )

        application_artifact = _compute_application_metric_artifact(
            stitched=stitched_pred,
            blocks=blocks,
            rollout_inputs=rollout_inputs,
            params=params,
            D=D,
            layers=layers,
            T_total=T_total,
            coupling_const=float(coupling_const),
            model_spec=spec,
        )
        if application_artifact is not None:
            application_summary, application_pathwise = application_artifact
            application_json_path = os.path.join(rec_dir, f"application_metrics_{pass_tag}.json")
            application_npz_path = os.path.join(rec_dir, f"application_metrics_{pass_tag}.npz")
            application_summary["output_path"] = application_json_path
            application_summary["pathwise_npz_path"] = application_npz_path
            if previous_application_pass_id is not None:
                stability = summarize_application_pass_stability(
                    previous_stitched=stitched_by_pass[previous_application_pass_id],
                    current_stitched=stitched_pred,
                    previous_pathwise=application_pathwise_by_pass[previous_application_pass_id],
                    current_pathwise=application_pathwise,
                    z_v_index=_application_z_v_index(spec),
                )
                application_summary["stability_vs_previous_pass"] = stability
                application_stability_by_pass[pass_id] = stability
            save_blob_npz(application_pathwise, application_npz_path)
            save_json(application_summary, application_json_path)
            application_summary_by_pass[pass_id] = application_summary
            application_pathwise_by_pass[pass_id] = application_pathwise
            previous_application_pass_id = pass_id

        if exact_solution is not None:
            exact_bundle = compute_stitched_exact_bundle(
                stitched=stitched_pred,
                exact_solution=exact_solution,
            )
            exact_summary = exact_bundle["summary"]
            exact_summary_by_pass[pass_id] = exact_summary
            exact_bundle_by_pass[pass_id] = exact_bundle
            print(
                f"[Exact] {_pass_label(pass_id)} "
                f"mean_pred_Y0={exact_summary['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_summary['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_summary['abs_error_mean_y0']:.6e}, "
                f"mean_abs_err_Y={exact_summary['mean_abs_error_y']:.6e}, "
                f"mean_abs_err_Z={exact_summary['mean_abs_error_z']:.6e}"
            )

            save_json(
                {
                    "summary": exact_summary,
                    "timeseries": exact_bundle["timeseries"],
                },
                os.path.join(rec_dir, f"exact_metrics_{pass_tag}.json"),
            )
            save_exact_error_timeseries_csv(
                exact_bundle["timeseries"],
                os.path.join(rec_dir, f"exact_errors_{pass_tag}.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_pred,
                Y_exact=exact_bundle["Y_exact"],
                Z_exact=exact_bundle["Z_exact"],
                blocks=blocks,
                out_dir=plots_dir,
                sample_paths=sample_paths,
                file_suffix=f"_{pass_tag}",
                include_path_plots=False,
                include_error_plots=True,
            )

            if Xi_visual is not None and visual_rollout_inputs is not None:
                visual_stitched = predict_recursive_stitched(
                    block_blobs=p["blobs"],
                    blocks=blocks,
                    Xi_initial=Xi_visual,
                    params=params,
                    N_per_block=N_per_block,
                    D=D,
                    layers=layers,
                    T_total=T_total,
                    rollout_inputs=visual_rollout_inputs,
                    coupling_const=float(coupling_const),
                    model_spec=spec,
                )
                visual_exact_bundle = compute_stitched_exact_bundle(
                    stitched=visual_stitched,
                    exact_solution=exact_solution,
                )
                visual_stitched_by_pass[pass_id] = visual_stitched
                visual_exact_bundle_by_pass[pass_id] = visual_exact_bundle
                save_blob_npz(
                    visual_stitched,
                    os.path.join(rec_dir, f"visual_stitched_predictions_{pass_tag}.npz"),
                )
                plot_recursive_exact_comparison(
                    stitched=visual_stitched,
                    Y_exact=visual_exact_bundle["Y_exact"],
                    Z_exact=visual_exact_bundle["Z_exact"],
                    blocks=blocks,
                    out_dir=plots_dir,
                    sample_paths=visual_path_count,
                    file_suffix=f"_{pass_tag}",
                    include_path_plots=True,
                    include_error_plots=False,
                )

    if (
        enforce_exact_regression_guardrail
        and exact_solution is not None
        and len(exact_summary_by_pass) >= 2
        and str(exact_regression_action) != "ignore"
    ):
        tol = float(exact_regression_tolerance)
        if tol > 0.0:
            sorted_pass_ids = sorted(exact_summary_by_pass.keys())
            prev_id = sorted_pass_ids[0]
            prev_val = float(exact_summary_by_pass[prev_id]["mean_abs_error_y"])
            for pass_id in sorted_pass_ids[1:]:
                curr_val = float(exact_summary_by_pass[pass_id]["mean_abs_error_y"])
                if prev_val > 0.0 and curr_val > prev_val * (1.0 + tol):
                    msg = (
                        "[ExactGuardrail] Regression detected on mean_abs_error_y: "
                        f"{_pass_label(prev_id)}={prev_val:.6e} -> {_pass_label(pass_id)}={curr_val:.6e} "
                        f"(+{(curr_val / prev_val - 1.0) * 100.0:.2f}%, tol={tol * 100.0:.2f}%)"
                    )
                    if str(exact_regression_action) == "error":
                        raise RuntimeError(msg)
                    print(msg)
                prev_id = pass_id
                prev_val = curr_val

    if not bool(promote_final_artifacts):
        plot_recursive_stitched_y_convergence(
            stitched_by_pass=stitched_by_pass,
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
        )
        return {
            "processed_pass_ids": sorted(pass_logs_by_pass.keys()),
            "processed_pass_indices": sorted(_pass_index(pid) for pid in pass_logs_by_pass.keys()),
            "score_key": score_key,
            "pass_scores_loss": pass_scores_loss,
            "pass_scores_loss_by_index": {
                str(_pass_index(k)): float(v) for k, v in pass_scores_loss.items()
            },
            "pass_invalid_reasons": pass_invalid_reasons,
            "pass_invalid_reasons_by_index": {
                str(_pass_index(k)): v for k, v in pass_invalid_reasons.items()
            },
            "selection_finalized": False,
            "promoted_final_artifacts": False,
            "excluded_pass_ids_from_selection": [],
            "excluded_pass_indices_from_selection": [],
            "selected_pass_id": None,
            "selected_pass_index": None,
            "selected_score_metric": None,
            "selected_score": None,
            "selected_scores_by_pass": {},
            "selected_scores_by_pass_index": {},
            "exact_summary_by_pass": exact_summary_by_pass,
            "exact_summary_by_pass_index": {
                str(_pass_index(k)): v for k, v in exact_summary_by_pass.items()
            },
            "application_summary_by_pass": application_summary_by_pass,
            "application_summary_by_pass_index": {
                str(_pass_index(k)): v for k, v in application_summary_by_pass.items()
            },
            "application_stability_by_pass": application_stability_by_pass,
            "application_stability_by_pass_index": {
                str(_pass_index(k)): v for k, v in application_stability_by_pass.items()
            },
            "selected_application_summary": None,
            "eval_bundle_path": eval_bundle_path,
            "evaluation_bundle_M": int(Xi_stitched.shape[0]),
            "visual_sample_paths": int(max(0, visual_path_count)),
            "visual_seed": int(visual_seed_effective),
        }

    finite_pass_scores_loss = {
        int(pass_id): float(score)
        for pass_id, score in pass_scores_loss.items()
        if np.isfinite(float(score))
    }
    if len(finite_pass_scores_loss) > 0:
        best_pass_by_loss = int(min(finite_pass_scores_loss, key=finite_pass_scores_loss.get))
        print(
            f"[Selection:loss] metric={score_key}, best={_pass_label(best_pass_by_loss)}, "
            f"score={finite_pass_scores_loss[best_pass_by_loss]:.6e}"
        )
    else:
        print(f"[Selection:loss] metric={score_key}, no finite pass scores")

    excluded_pass_ids_effective = sorted(
        int(pid)
        for pid in {int(x) for x in (exclude_pass_ids_from_selection or [])}
        if bool(promote_final_artifacts) and int(pid) in pass_scores_loss
    )
    pass_scores_loss_for_selection = {
        int(pass_id): float(score)
        for pass_id, score in finite_pass_scores_loss.items()
        if int(pass_id) not in excluded_pass_ids_effective
    }
    if len(excluded_pass_ids_effective) > 0:
        print(
            "[Selection] excluding passes from final choice: "
            + ", ".join(_pass_label(pid) for pid in excluded_pass_ids_effective)
        )

    exact_summary_by_pass_for_selection = {
        int(pass_id): summary
        for pass_id, summary in exact_summary_by_pass.items()
        if int(pass_id) in pass_scores_loss_for_selection
    }
    selected_pass_id, selected_score_metric, selected_score, selected_score_by_pass = resolve_pass_selection(
        pass_scores_by_loss=pass_scores_loss_for_selection,
        exact_summary_by_pass=exact_summary_by_pass_for_selection,
        selection_metric=str(selection_metric),
        loss_metric_label=score_key,
    )
    print(
        f"[Selection:final] metric={selected_score_metric}, best={_pass_label(selected_pass_id)}, "
        f"score={selected_score:.6e}"
    )

    selected_stitched = stitched_by_pass[selected_pass_id]
    selected_exact_bundle = exact_bundle_by_pass.get(selected_pass_id, None)
    selected_visual_stitched = visual_stitched_by_pass.get(selected_pass_id, None)
    selected_visual_exact_bundle = visual_exact_bundle_by_pass.get(selected_pass_id, None)
    selected_application_summary = application_summary_by_pass.get(selected_pass_id, None)
    if promote_final_artifacts:
        save_blob_npz(
            selected_stitched,
            os.path.join(rec_dir, "stitched_predictions_final.npz"),
        )
        plot_recursive_stitched_predictions(
            stitched=selected_stitched,
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix="",
        )

        if exact_solution is not None and selected_exact_bundle is not None:
            save_json(
                {
                    "summary": selected_exact_bundle["summary"],
                    "timeseries": selected_exact_bundle["timeseries"],
                },
                os.path.join(rec_dir, "exact_metrics_final.json"),
            )
            save_exact_error_timeseries_csv(
                selected_exact_bundle["timeseries"],
                os.path.join(rec_dir, "exact_errors_final.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=selected_stitched,
                Y_exact=selected_exact_bundle["Y_exact"],
                Z_exact=selected_exact_bundle["Z_exact"],
                blocks=blocks,
                out_dir=plots_dir,
                sample_paths=sample_paths,
                file_suffix="",
                include_path_plots=False,
                include_error_plots=True,
            )
            if selected_visual_stitched is not None and selected_visual_exact_bundle is not None:
                save_blob_npz(
                    selected_visual_stitched,
                    os.path.join(rec_dir, "visual_stitched_predictions_final.npz"),
                )
                plot_recursive_exact_comparison(
                    stitched=selected_visual_stitched,
                    Y_exact=selected_visual_exact_bundle["Y_exact"],
                    Z_exact=selected_visual_exact_bundle["Z_exact"],
                    blocks=blocks,
                    out_dir=plots_dir,
                    sample_paths=visual_path_count,
                    file_suffix="",
                    include_path_plots=True,
                    include_error_plots=False,
                )

        if selected_application_summary is not None:
            application_final_json_path = os.path.join(rec_dir, "application_metrics_final.json")
            application_final_npz_path = os.path.join(rec_dir, "application_metrics_final.npz")
            application_final_summary = dict(selected_application_summary)
            application_final_summary["selected_from_pass_id"] = int(selected_pass_id)
            application_final_summary["selected_from_pass_index"] = int(_pass_index(selected_pass_id))
            application_final_summary["output_path"] = application_final_json_path
            application_final_summary["pathwise_npz_path"] = application_final_npz_path
            save_blob_npz(application_pathwise_by_pass[selected_pass_id], application_final_npz_path)
            save_json(application_final_summary, application_final_json_path)
            selected_application_summary = application_final_summary

    plot_recursive_stitched_y_convergence(
        stitched_by_pass=stitched_by_pass,
        blocks=blocks,
        out_dir=plots_dir,
        sample_paths=sample_paths,
    )

    return {
        "processed_pass_ids": sorted(pass_logs_by_pass.keys()),
        "processed_pass_indices": sorted(_pass_index(pid) for pid in pass_logs_by_pass.keys()),
        "score_key": score_key,
        "pass_scores_loss": pass_scores_loss,
        "pass_scores_loss_by_index": {
            str(_pass_index(k)): float(v) for k, v in pass_scores_loss.items()
        },
        "pass_invalid_reasons": pass_invalid_reasons,
        "pass_invalid_reasons_by_index": {
            str(_pass_index(k)): v for k, v in pass_invalid_reasons.items()
        },
        "selection_finalized": True,
        "promoted_final_artifacts": bool(promote_final_artifacts),
        "excluded_pass_ids_from_selection": excluded_pass_ids_effective,
        "excluded_pass_indices_from_selection": [
            int(_pass_index(pid)) for pid in excluded_pass_ids_effective
        ],
        "selected_pass_id": int(selected_pass_id),
        "selected_pass_index": int(_pass_index(selected_pass_id)),
        "selected_score_metric": selected_score_metric,
        "selected_score": float(selected_score),
        "selected_scores_by_pass": selected_score_by_pass,
        "selected_scores_by_pass_index": {
            str(_pass_index(int(k))): float(v)
            for k, v in selected_score_by_pass.items()
        },
        "exact_summary_by_pass": exact_summary_by_pass,
        "exact_summary_by_pass_index": {
            str(_pass_index(k)): v for k, v in exact_summary_by_pass.items()
        },
        "application_summary_by_pass": application_summary_by_pass,
        "application_summary_by_pass_index": {
            str(_pass_index(k)): v for k, v in application_summary_by_pass.items()
        },
        "application_stability_by_pass": application_stability_by_pass,
        "application_stability_by_pass_index": {
            str(_pass_index(k)): v for k, v in application_stability_by_pass.items()
        },
        "selected_application_summary": selected_application_summary,
        "eval_bundle_path": eval_bundle_path,
        "evaluation_bundle_M": int(Xi_stitched.shape[0]),
        "visual_sample_paths": int(max(0, visual_path_count)),
        "visual_seed": int(visual_seed_effective),
    }

def resolve_pass_selection(
    pass_scores_by_loss: Dict[int, float],
    exact_summary_by_pass: Dict[int, Dict[str, Any]],
    selection_metric: str,
    loss_metric_label: str = "eval_mean_loss_per_sample",
) -> Tuple[int, str, float, Dict[str, float]]:
    if len(pass_scores_by_loss) == 0:
        raise RuntimeError("No finite pass scores available for pass selection")
    finite_loss_scores = {
        int(pass_id): float(score)
        for pass_id, score in pass_scores_by_loss.items()
        if np.isfinite(float(score))
    }
    if len(finite_loss_scores) == 0:
        raise RuntimeError("No finite pass scores available for pass selection")

    metric = str(selection_metric or "auto").strip().lower()
    selected_by_loss = int(min(finite_loss_scores, key=finite_loss_scores.get))

    if metric in ("", "auto"):
        if len(exact_summary_by_pass) > 0:
            metric = "exact_mae_y"
        else:
            metric = "loss"

    if metric == "loss":
        return (
            selected_by_loss,
            f"{loss_metric_label}+0.35*worst_block",
            float(finite_loss_scores[selected_by_loss]),
            {str(k): float(v) for k, v in finite_loss_scores.items()},
        )
    if metric == "last":
        selected_last = int(max(finite_loss_scores))
        return (
            selected_last,
            "last_pass",
            float(finite_loss_scores[selected_last]),
            {str(k): float(v) for k, v in finite_loss_scores.items()},
        )

    metric_extractors = {
        "exact_mae_y": ("exact.mean_abs_error_y", lambda s: float(s["mean_abs_error_y"])),
        "exact_rmse_y": ("exact.rmse_y", lambda s: float(s["rmse_y"])),
        "exact_abs_y0": ("exact.abs_error_mean_y0", lambda s: float(s["abs_error_mean_y0"])),
        "exact_mae_z": ("exact.mean_abs_error_z", lambda s: float(s["mean_abs_error_z"])),
        "exact_mae_z_s": (
            "exact.mean_abs_error_z_component_Z_S",
            lambda s: float(s["mean_abs_error_z_by_component"][0]),
        ),
    }
    if metric not in metric_extractors:
        raise ValueError(
            f"Unsupported selection_metric='{selection_metric}'. "
            "Supported: auto, loss, last, exact_mae_y, exact_rmse_y, exact_abs_y0, "
            "exact_mae_z, exact_mae_z_s"
        )
    if len(exact_summary_by_pass) == 0:
        raise RuntimeError(
            f"selection_metric='{metric}' requires exact_solution metrics, but none are available"
        )

    label, extractor = metric_extractors[metric]
    scores = {}
    for pass_id, summary in exact_summary_by_pass.items():
        score = float(extractor(summary))
        if np.isfinite(score):
            scores[int(pass_id)] = score
    if len(scores) == 0:
        raise RuntimeError(f"No finite exact scores available for selection_metric='{metric}'")
    selected_pass = int(min(scores, key=scores.get))
    return (
        selected_pass,
        label,
        float(scores[selected_pass]),
        {str(k): float(v) for k, v in scores.items()},
    )

def _predict_accepts_runtime_diagnostics(model: FBSNN) -> bool:
    try:
        signature = inspect.signature(model.predict)
    except (TypeError, ValueError):
        return True
    parameters = list(signature.parameters.values())
    return (
        "return_runtime_diagnostics" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
    )

def _predict_with_optional_runtime_diagnostics(
    model: FBSNN,
    Xi_star: np.ndarray,
    t_star: np.ndarray,
    W_star: np.ndarray,
    coupling_const: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    predict_kwargs = {"const_value": float(coupling_const)}
    if _predict_accepts_runtime_diagnostics(model):
        predict_kwargs["return_runtime_diagnostics"] = True

    result = model.predict(Xi_star, t_star, W_star, **predict_kwargs)
    if not isinstance(result, (tuple, list)):
        raise ValueError("model.predict must return a tuple/list with X, Y, Z")
    if len(result) == 4:
        X, Y, Z, runtime_diagnostics = result
        if runtime_diagnostics is None:
            runtime_diagnostics = {}
        else:
            runtime_diagnostics = dict(runtime_diagnostics)
        return X, Y, Z, runtime_diagnostics
    if len(result) == 3:
        X, Y, Z = result
        return X, Y, Z, {}
    raise ValueError(
        "model.predict must return either (X, Y, Z) or (X, Y, Z, runtime_diagnostics)"
    )

def predict_recursive_stitched(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_initial: np.ndarray,
    params: Dict[str, np.ndarray],
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    rollout_inputs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    coupling_const: float = 1.0,
    model_spec: Optional[ModelSpec] = None,
) -> Dict[str, np.ndarray]:
    if len(blocks) == 0:
        raise ValueError("blocks must contain at least one block")
    spec = _resolve_model_spec(model_spec)
    spec.validate_state_dim(D)
    if Xi_initial.ndim != 2 or Xi_initial.shape[1] != D:
        raise ValueError(f"Xi_initial must have shape [M, {D}]")
    if rollout_inputs is not None and len(rollout_inputs) != len(blocks):
        raise ValueError("rollout_inputs must have one (t, W) pair per block")

    Xi_curr = Xi_initial.astype(np.float32)
    t_segments = []
    X_segments = []
    Y_segments = []
    Z_segments = []
    diagnostic_segments = {}
    boundary_signed_jumps = []
    boundary_abs_jumps = []
    boundary_y_signed_jumps = []
    boundary_y_abs_jumps = []
    boundary_z_signed_jumps = []
    boundary_z_abs_jumps = []
    previous_terminal_X = None
    previous_terminal_Y = None
    previous_terminal_Z = None

    for b, block in enumerate(blocks):
        blob = block_blobs[b]
        reset_backend_state()

        model = spec.build_recursive_model(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"],
            M=Xi_curr.shape[0],
            N=N_per_block,
            D=D,
            layers=layers,
            parameters=params,
            t_start=block["t_start"],
            t_end=block["t_end"],
            T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(blob.get("normalize_time_input", 1))),
            x_norm_mean=blob.get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=blob.get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )

        try:
            model.import_parameter_blob(blob, strict=True)
            if rollout_inputs is None:
                t_b, W_b, _ = model.fetch_minibatch()
            else:
                t_b, W_b = rollout_inputs[b]
            X_b, Y_b, Z_b, runtime_diagnostics_b = _predict_with_optional_runtime_diagnostics(
                model,
                Xi_curr,
                t_b,
                W_b,
                float(coupling_const),
            )

            if previous_terminal_X is not None:
                x_jump = np.asarray(X_b[:, 0, :], dtype=np.float32) - previous_terminal_X
                y_jump = np.asarray(Y_b[:, 0, :], dtype=np.float32) - previous_terminal_Y
                z_jump = np.asarray(Z_b[:, 0, :], dtype=np.float32) - previous_terminal_Z
                boundary_signed_jumps.append(x_jump)
                boundary_abs_jumps.append(np.abs(x_jump))
                boundary_y_signed_jumps.append(y_jump)
                boundary_y_abs_jumps.append(np.abs(y_jump))
                boundary_z_signed_jumps.append(z_jump)
                boundary_z_abs_jumps.append(np.abs(z_jump))

            start_idx = 0 if b == 0 else 1
            t_segments.append(t_b[:, start_idx:, :].astype(np.float32))
            X_segments.append(X_b[:, start_idx:, :].astype(np.float32))
            Y_segments.append(Y_b[:, start_idx:, :].astype(np.float32))
            Z_segments.append(Z_b[:, start_idx:, :].astype(np.float32))
            for key, value in runtime_diagnostics_b.items():
                value_np = np.asarray(value, dtype=np.float32)
                if value_np.ndim == 0:
                    value_np = value_np.reshape(1, 1)
                elif value_np.ndim == 1:
                    value_np = value_np.reshape(-1, 1)
                if value_np.shape[0] != t_b.shape[1]:
                    raise ValueError(
                        f"runtime diagnostic '{key}' must have {t_b.shape[1]} time steps, "
                        f"got {value_np.shape[0]}"
                    )
                diagnostic_segments.setdefault(str(key), []).append(
                    value_np[start_idx:, ...].astype(np.float32)
                )

            previous_terminal_X = X_b[:, -1, :].astype(np.float32)
            previous_terminal_Y = Y_b[:, -1, :].astype(np.float32)
            previous_terminal_Z = Z_b[:, -1, :].astype(np.float32)
            Xi_curr = previous_terminal_X
        finally:
            model.sess.close()

    stitched = {
        "t": np.concatenate(t_segments, axis=1),
        "X": np.concatenate(X_segments, axis=1),
        "Y": np.concatenate(Y_segments, axis=1),
        "Z": np.concatenate(Z_segments, axis=1),
    }
    for key, segments in diagnostic_segments.items():
        stitched[key] = np.concatenate(segments, axis=0).astype(np.float32)
    application_metric_schema = str(getattr(spec, "application_metric_schema", "none"))
    emit_boundary_jumps = bool(boundary_abs_jumps) and application_metric_schema != "none"
    emit_signed_boundary_jumps = bool(boundary_signed_jumps) and application_metric_schema in {
        "pascucci_application_metrics_v1",
        "pascucci_application_metrics_v2",
    }
    if emit_boundary_jumps:
        if emit_signed_boundary_jumps:
            stitched["stitch_X_boundary_signed_jump"] = np.stack(boundary_signed_jumps, axis=0).astype(np.float32)
        stitched["stitch_X_boundary_abs_jump"] = np.stack(boundary_abs_jumps, axis=0).astype(np.float32)
        if emit_signed_boundary_jumps:
            stitched["stitch_Y_boundary_signed_jump"] = np.stack(boundary_y_signed_jumps, axis=0).astype(np.float32)
        stitched["stitch_Y_boundary_abs_jump"] = np.stack(boundary_y_abs_jumps, axis=0).astype(np.float32)
        if emit_signed_boundary_jumps:
            stitched["stitch_Z_boundary_signed_jump"] = np.stack(boundary_z_signed_jumps, axis=0).astype(np.float32)
        stitched["stitch_Z_boundary_abs_jump"] = np.stack(boundary_z_abs_jumps, axis=0).astype(np.float32)
    return stitched

def train_with_standard_schedule(
    model: FBSNN,
    stage_plan: List[Tuple[int, float]],
    final_plan: List[Tuple[int, float]],
    eval_batches=5,
    precision_target: Optional[float] = None,
    max_refine_rounds: int = 3,
    refine_plan: Optional[List[Tuple[int, float]]] = None,
    label: str = "",
    coupling_const: float = 1.0,
):
    stage_logs = []
    schedule_eval_batches = _normalize_schedule_eval_batches(eval_batches, model=model)
    train_eval_kwargs = _train_evaluation_kwargs(schedule_eval_batches)

    coupling_levels = np.asarray([np.float32(coupling_const)], dtype=np.float32)

    for level in coupling_levels:
        model.const = np.float32(level)
        print(f"=== [{label}] Coupling stage: const={float(level):.1f} ===")
        for n_iter, lr in stage_plan:
            t0 = time.time()
            train_stats = model.train(
                N_Iter=n_iter,
                learning_rate=lr,
                const_value=level,
                **train_eval_kwargs,
            )
            eval_stats = _evaluate_with_schedule_batches(
                model,
                const_value=float(level),
                eval_batches=schedule_eval_batches,
            )
            _require_finite_schedule_stats(
                train_stats,
                label=label,
                phase=f"train lr={lr:.1e}",
                keys=("last_loss", "best_score"),
            )
            _require_finite_schedule_stats(
                eval_stats,
                label=label,
                phase=f"eval lr={lr:.1e}",
                keys=(
                    "mean_loss",
                    "std_loss",
                    "mean_loss_per_sample",
                    "std_loss_per_sample",
                    "mean_y0",
                    "std_y0",
                ),
            )
            elapsed = time.time() - t0
            stage_logs.append(
                {
                    "phase": "curriculum",
                    "const": float(level),
                    "lr": float(lr),
                    "n_iter": int(n_iter),
                    "train_last_loss": train_stats["last_loss"],
                    "eval_mean_loss": eval_stats["mean_loss"],
                    "eval_std_loss": eval_stats["std_loss"],
                    "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
                    "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
                    "eval_mean_y0": eval_stats["mean_y0"],
                    "eval_std_y0": eval_stats["std_y0"],
                    "elapsed_sec": float(elapsed),
                }
                | _prefixed_eval_diagnostics(eval_stats)
            )
            print(
                f"[StageSummary] {label} const={level:.1f}, lr={lr:.1e}, iters={n_iter}, "
                f"eval_loss={eval_stats['mean_loss']:.3e}+/-{eval_stats['std_loss']:.2e}, "
                f"eval_Y0={eval_stats['mean_y0']:.3f}+/-{eval_stats['std_y0']:.3f}, time={elapsed:.1f}s"
            )

    model.const = np.float32(coupling_const)
    print(f"=== [{label}] Final fine-tuning at const={float(coupling_const):.1f} ===")
    for n_iter, lr in final_plan:
        t0 = time.time()
        train_stats = model.train(
            N_Iter=n_iter,
            learning_rate=lr,
            const_value=float(coupling_const),
            eval_every=25,
            val_batches=8,
            early_stopping_metric="loss",
            patience=150,
            min_delta=1e-3,
            restore_best=True,
            **train_eval_kwargs,
        )
        eval_stats = _evaluate_with_schedule_batches(
            model,
            const_value=float(coupling_const),
            eval_batches=schedule_eval_batches,
        )
        _require_finite_schedule_stats(
            train_stats,
            label=label,
            phase=f"train lr={lr:.1e}",
            keys=("last_loss", "best_score"),
        )
        _require_finite_schedule_stats(
            eval_stats,
            label=label,
            phase=f"eval lr={lr:.1e}",
            keys=(
                "mean_loss",
                "std_loss",
                "mean_loss_per_sample",
                "std_loss_per_sample",
                "mean_y0",
                "std_y0",
            ),
        )
        elapsed = time.time() - t0
        stage_logs.append(
            {
                "phase": "final_finetune",
                "const": float(coupling_const),
                "lr": float(lr),
                "n_iter": int(n_iter),
                "train_last_loss": train_stats["last_loss"],
                "best_iter": train_stats["best_iter"],
                "best_score": train_stats["best_score"],
                "stopped_early": train_stats["stopped_early"],
                "eval_mean_loss": eval_stats["mean_loss"],
                "eval_std_loss": eval_stats["std_loss"],
                "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
                "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
                "eval_mean_y0": eval_stats["mean_y0"],
                "eval_std_y0": eval_stats["std_y0"],
                "elapsed_sec": float(elapsed),
            }
            | _prefixed_eval_diagnostics(eval_stats)
        )
        print(
            f"[FinalSummary] {label} const={float(coupling_const):.1f}, lr={lr:.1e}, iters={n_iter}, "
            f"best_it={train_stats['best_iter']}, best_score={train_stats['best_score']:.3e}, "
            f"eval_loss={eval_stats['mean_loss']:.3e}+/-{eval_stats['std_loss']:.2e}, "
            f"eval_Y0={eval_stats['mean_y0']:.3f}+/-{eval_stats['std_y0']:.3f}, time={elapsed:.1f}s"
        )

    eval_stats = _evaluate_with_schedule_batches(
        model,
        const_value=float(coupling_const),
        eval_batches=schedule_eval_batches,
    )
    refine_rounds = 0
    local_refine_plan = refine_plan if refine_plan is not None else [(50, 1e-5), (50, 5e-6)]

    while (
        precision_target is not None
        and eval_stats["mean_loss"] > precision_target
        and refine_rounds < max_refine_rounds
    ):
        refine_rounds += 1
        print(
            f"[Refine] {label} round={refine_rounds}, "
            f"loss={eval_stats['mean_loss']:.3e} > target={precision_target:.3e}"
        )
        for n_iter, lr in local_refine_plan:
            train_stats = model.train(
                N_Iter=n_iter,
                learning_rate=lr,
                const_value=float(coupling_const),
                eval_every=25,
                val_batches=8,
                early_stopping_metric="loss",
                patience=100,
                min_delta=1e-3,
                restore_best=True,
                **train_eval_kwargs,
            )
            _require_finite_schedule_stats(
                train_stats,
                label=label,
                phase=f"refine train lr={lr:.1e}",
                keys=("last_loss", "best_score"),
            )
        eval_stats = _evaluate_with_schedule_batches(
            model,
            const_value=float(coupling_const),
            eval_batches=schedule_eval_batches,
        )
        _require_finite_schedule_stats(
            eval_stats,
            label=label,
            phase=f"refine eval round={refine_rounds}",
            keys=(
                "mean_loss",
                "std_loss",
                "mean_loss_per_sample",
                "std_loss_per_sample",
                "mean_y0",
                "std_y0",
            ),
        )

    return {
        "stage_logs": stage_logs,
        "eval_stats": eval_stats,
        "refine_rounds": int(refine_rounds),
        "precision_target": None if precision_target is None else float(precision_target),
    }

def run_standard_reference(
    Xi_generator,
    params,
    M,
    N,
    D,
    T,
    layers,
    stage_plan,
    final_plan,
    coupling_const=1.0,
    model_spec: Optional[ModelSpec] = None,
):
    reset_backend_state()
    spec = _resolve_model_spec(model_spec)
    model = spec.build_standard_model(
        Xi_generator=Xi_generator,
        T=T,
        M=M,
        N=N,
        D=D,
        layers=layers,
        params=params,
    )
    logs = train_with_standard_schedule(
        model=model,
        stage_plan=stage_plan,
        final_plan=final_plan,
        eval_batches=5,
        precision_target=None,
        label="standard",
        coupling_const=float(coupling_const),
    )
    return model, logs

def _validate_fixed_rollout_initial(Xi_initial: np.ndarray, M_rollout: int, D: int) -> np.ndarray:
    Xi = np.asarray(Xi_initial, dtype=np.float32)
    expected = (int(M_rollout), int(D))
    if Xi.shape != expected:
        raise ValueError(f"Xi_initial shape must be {expected}, got {Xi.shape}")
    if not np.isfinite(Xi).all():
        raise ValueError("Xi_initial contains non-finite values")
    return Xi

def _validate_fixed_rollout_inputs(
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    *,
    blocks: List[Dict[str, float]],
    M_rollout: int,
    N_per_block: int,
    D: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs = list(rollout_inputs)
    if len(pairs) != len(blocks):
        raise ValueError(
            f"rollout_inputs must contain one (t, W) pair per block, got {len(pairs)} for {len(blocks)} blocks"
        )
    validated = []
    t_shape = (int(M_rollout), int(N_per_block) + 1, 1)
    w_shape = (int(M_rollout), int(N_per_block) + 1, int(D))
    for idx, ((t_b, W_b), block) in enumerate(zip(pairs, blocks)):
        t_arr = np.asarray(t_b, dtype=np.float32)
        W_arr = np.asarray(W_b, dtype=np.float32)
        if t_arr.shape != t_shape:
            raise ValueError(f"rollout_inputs[{idx}][0] shape must be {t_shape}, got {t_arr.shape}")
        if W_arr.shape != w_shape:
            raise ValueError(f"rollout_inputs[{idx}][1] shape must be {w_shape}, got {W_arr.shape}")
        if not np.isfinite(t_arr).all() or not np.isfinite(W_arr).all():
            raise ValueError(f"rollout_inputs[{idx}] contains non-finite values")
        if not np.allclose(t_arr[:, 0, 0], float(block["t_start"]), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"rollout_inputs[{idx}] t_start does not match block metadata")
        if not np.allclose(t_arr[:, -1, 0], float(block["t_end"]), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"rollout_inputs[{idx}] t_end does not match block metadata")
        if not np.all(np.diff(t_arr[:, :, 0], axis=1) > 0.0):
            raise ValueError(f"rollout_inputs[{idx}] time grid must be strictly increasing")
        if not np.allclose(W_arr[:, 0, :], 0.0, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"rollout_inputs[{idx}] W_start must be zero for a block-local Brownian path")
        validated.append((t_arr, W_arr))
    return validated

def rollout_boundaries(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_generator,
    params,
    M_rollout,
    N_per_block,
    D,
    layers,
    T_total,
    coupling_const: float = 1.0,
    model_spec: Optional[ModelSpec] = None,
    Xi_initial: Optional[np.ndarray] = None,
    rollout_inputs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
):
    spec = _resolve_model_spec(model_spec)
    spec.validate_state_dim(D)
    boundary_samples = []
    Xi_curr = (
        _validate_fixed_rollout_initial(Xi_initial, M_rollout, D)
        if Xi_initial is not None
        else Xi_generator(M_rollout, D).astype(np.float32)
    )
    fixed_rollout_inputs = (
        _validate_fixed_rollout_inputs(
            rollout_inputs,
            blocks=blocks,
            M_rollout=M_rollout,
            N_per_block=N_per_block,
            D=D,
        )
        if rollout_inputs is not None
        else None
    )
    boundary_samples.append(Xi_curr.copy())

    for b, block in enumerate(blocks):
        reset_backend_state()
        model = spec.build_recursive_model(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"],
            M=M_rollout,
            N=N_per_block,
            D=D,
            layers=layers,
            parameters=params,
            t_start=block["t_start"],
            t_end=block["t_end"],
            T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(block_blobs[b].get("normalize_time_input", 1))),
            x_norm_mean=block_blobs[b].get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=block_blobs[b].get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )
        model.import_parameter_blob(block_blobs[b], strict=True)
        if fixed_rollout_inputs is None:
            t_b, W_b, _ = model.fetch_minibatch()
        else:
            t_b, W_b = fixed_rollout_inputs[b]
        X_pred, _, _ = model.predict(
            Xi_curr, t_b, W_b, const_value=float(coupling_const)
        )
        Xi_curr = X_pred[:, -1, :].astype(np.float32)
        boundary_samples.append(Xi_curr.copy())
        model.sess.close()

    return boundary_samples

def resolve_active_set_from_prev_pass(
    n_blocks: int,
    pass_id: int,
    prev_pass_logs_by_block: Optional[Dict[int, Dict[str, Any]]],
    freeze_stable_blocks_after_pass: int = 0,
    freeze_loss_threshold: float = 0.0,
    freeze_neighbor_radius: int = 1,
) -> Dict[str, Any]:
    freeze_after = int(freeze_stable_blocks_after_pass)
    threshold = float(freeze_loss_threshold)
    neighbor_radius = max(int(freeze_neighbor_radius), 0)

    active_blocks = list(range(int(n_blocks)))
    summary = {
        "enabled": False,
        "pass_id": int(pass_id),
        "freeze_stable_blocks_after_pass": freeze_after,
        "freeze_loss_threshold": threshold,
        "freeze_neighbor_radius": neighbor_radius,
        "active_blocks": active_blocks,
        "frozen_blocks": [],
        "unstable_blocks": [],
        "active_reasons_by_block": {
            str(b): ["freeze_disabled"]
            for b in active_blocks
        },
        "disabled_reason": "",
    }

    if freeze_after <= 0 or threshold <= 0.0:
        summary["disabled_reason"] = "freeze_flags_disabled"
        return summary
    if int(pass_id) <= freeze_after:
        summary["disabled_reason"] = "pass_not_after_freeze_start"
        return summary
    if prev_pass_logs_by_block is None or len(prev_pass_logs_by_block) == 0:
        summary["disabled_reason"] = "previous_pass_logs_unavailable"
        return summary

    unstable_blocks = []
    active_reasons: Dict[int, List[str]] = {0: ["always_keep_block0"]}

    for block_idx in range(int(n_blocks)):
        row = prev_pass_logs_by_block.get(int(block_idx))
        if row is None:
            continue
        loss_value = row.get("eval_mean_loss_per_sample", row.get("eval_mean_loss", np.nan))
        try:
            loss_value = float(loss_value)
        except Exception:
            loss_value = float("nan")
        if np.isfinite(loss_value) and loss_value > threshold:
            unstable_blocks.append(int(block_idx))
            active_reasons.setdefault(int(block_idx), []).append("loss_above_threshold")
            left = max(0, int(block_idx) - neighbor_radius)
            right = min(int(n_blocks) - 1, int(block_idx) + neighbor_radius)
            for neighbor in range(left, right + 1):
                if neighbor == int(block_idx):
                    continue
                active_reasons.setdefault(int(neighbor), []).append(
                    f"neighbor_of_unstable_block_{int(block_idx)}"
                )

    active_blocks = sorted(int(b) for b in active_reasons.keys())
    frozen_blocks = [int(b) for b in range(int(n_blocks)) if int(b) not in set(active_blocks)]
    summary.update(
        {
            "enabled": True,
            "active_blocks": active_blocks,
            "frozen_blocks": frozen_blocks,
            "unstable_blocks": sorted(set(int(b) for b in unstable_blocks)),
            "active_reasons_by_block": {
                str(int(b)): reasons for b, reasons in sorted(active_reasons.items())
            },
            "disabled_reason": "",
        }
    )
    return summary

def run_recursive_training(
    Xi_generator,
    params,
    M,
    N_per_block,
    D,
    T_total,
    block_size,
    layers,
    stage_plan,
    final_plan,
    output_dir,
    precision_margin=0.10,
    max_refine_rounds=3,
    rollout_M=2000,
    save_tf_checkpoints=True,
    training_plan_rules: Optional[List[Dict]] = None,
    pass1_warm_start_from_next=False,
    cross_pass_warm_start: bool = True,
    n_passes: int = 2,
    empirical_jitter_scale: float = 0.02,
    pass1_init_mode: str = "base",
    initial_boundary_samples: Optional[List[np.ndarray]] = None,
    initial_warm_start_blobs: Optional[List[Dict[str, np.ndarray]]] = None,
    freeze_stable_blocks_after_pass: int = 0,
    freeze_loss_threshold: float = 0.0,
    freeze_neighbor_radius: int = 1,
    coupling_const: float = 1.0,
    on_pass_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    model_spec: Optional[ModelSpec] = None,
):
    if int(n_passes) < 1:
        raise ValueError("n_passes must be >= 1")
    spec = _resolve_model_spec(model_spec)
    spec.validate_state_dim(D)

    blocks = build_blocks(T_total=T_total, block_size=block_size)
    validate_boundary_samples(
        boundary_samples=initial_boundary_samples,
        blocks=blocks,
        D=D,
        label="initial_boundary_samples",
    )
    if initial_warm_start_blobs is not None and len(initial_warm_start_blobs) != len(blocks):
        raise ValueError(
            "initial_warm_start_blobs must contain one blob per block when provided"
        )
    print(
        f"[Recursive] blocks={len(blocks)} -> {[ (b['t_start'], b['t_end']) for b in blocks ]}, "
        f"n_passes={int(n_passes)}"
    )

    def _run_pass(
        pass_id,
        generators_per_block,
        warm_start_blobs=None,
        carry_over_blobs=None,
        warm_start_from_next=False,
        prev_pass_loss_by_block=None,
        prev_pass_logs_by_block=None,
    ):
        pass_dir = os.path.join(output_dir, f"pass_{pass_id}")
        os.makedirs(pass_dir, exist_ok=True)

        next_blob = None
        block_blobs = [None] * len(blocks)
        logs = []
        reference_loss = None
        active_set_summary = resolve_active_set_from_prev_pass(
            n_blocks=len(blocks),
            pass_id=pass_id,
            prev_pass_logs_by_block=prev_pass_logs_by_block,
            freeze_stable_blocks_after_pass=freeze_stable_blocks_after_pass,
            freeze_loss_threshold=freeze_loss_threshold,
            freeze_neighbor_radius=freeze_neighbor_radius,
        )
        active_blocks_set = {int(b) for b in active_set_summary.get("active_blocks", [])}
        if bool(active_set_summary.get("enabled", False)):
            print(
                f"[ActiveSet] {_pass_label(pass_id)} active={sorted(active_blocks_set)} "
                f"frozen={active_set_summary.get('frozen_blocks', [])} "
                f"unstable_prev={active_set_summary.get('unstable_blocks', [])} "
                f"threshold={float(active_set_summary.get('freeze_loss_threshold', 0.0)):.3e}"
            )

        for b in range(len(blocks) - 1, -1, -1):
            block = blocks[b]
            label = f"{_pass_label(pass_id)}:block{b}"
            print(
                f"\n[RecursiveBlock] {label} t=[{block['t_start']:.2f},{block['t_end']:.2f}] "
                f"T_block={block['T_block']:.2f}"
            )

            if int(b) not in active_blocks_set and carry_over_blobs is not None and carry_over_blobs[b] is not None:
                blob = _as_blob_dict(carry_over_blobs[b])
                reset_backend_state()
                model = spec.build_recursive_model(
                    Xi_generator=generators_per_block[b],
                    T=block["T_block"],
                    M=M,
                    N=N_per_block,
                    D=D,
                    layers=layers,
                    parameters=params,
                    t_start=block["t_start"],
                    t_end=block["t_end"],
                    T_total=T_total,
                    terminal_blob=next_blob,
                    normalize_time_input=bool(int(blob.get("normalize_time_input", 1))),
                    x_norm_mean=blob.get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
                    x_norm_std=blob.get("x_norm_std", np.ones((1, D), dtype=np.float32)),
                )
                model.import_parameter_blob(blob, strict=True)
                eval_stats = model.evaluate(
                    const_value=float(coupling_const), n_batches=5
                )
                blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
                save_blob_npz(blob, blob_path)
                if reference_loss is None:
                    reference_loss = float(eval_stats["mean_loss"])
                log_row = {
                    "pass": int(pass_id),
                    "block": int(b),
                    "t_start": float(block["t_start"]),
                    "t_end": float(block["t_end"]),
                    "T_block": float(block["T_block"]),
                    "eval_mean_loss": float(eval_stats["mean_loss"]),
                    "eval_std_loss": float(eval_stats["std_loss"]),
                    "eval_mean_loss_per_sample": float(eval_stats["mean_loss_per_sample"]),
                    "eval_std_loss_per_sample": float(eval_stats["std_loss_per_sample"]),
                    "eval_mean_y0": float(eval_stats["mean_y0"]),
                    "precision_target": None,
                    "refine_rounds": 0,
                    "stage_plan_used": [],
                    "final_plan_used": [],
                    "refine_plan_used": [],
                    "blob_path": blob_path,
                    "ckpt_path": None,
                    "was_frozen": True,
                    "freeze_reason": "stable_prev_pass",
                    "freeze_source_pass": int(pass_id) - 1,
                    "freeze_threshold": float(freeze_loss_threshold),
                    "freeze_neighbor_radius": int(freeze_neighbor_radius),
                    "active_set_enabled": True,
                } | _prefixed_eval_diagnostics(eval_stats)
                logs.append(log_row)
                block_blobs[b] = blob
                next_blob = blob
                model.sess.close()
                print(f"[ActiveSet] {label} frozen -> reusing {_pass_label(int(pass_id) - 1)} block{b}")
                continue

            x_mean, x_std = estimate_generator_stats(generators_per_block[b], D=D, n_samples=max(4096, M))

            reset_backend_state()
            model = spec.build_recursive_model(
                Xi_generator=generators_per_block[b],
                T=block["T_block"],
                M=M,
                N=N_per_block,
                D=D,
                layers=layers,
                parameters=params,
                t_start=block["t_start"],
                t_end=block["t_end"],
                T_total=T_total,
                terminal_blob=next_blob,
                normalize_time_input=True,
                x_norm_mean=x_mean,
                x_norm_std=x_std,
            )

            # Opzione: nella passata 1 inizializza il blocco i coi pesi del blocco i+1.
            if warm_start_from_next and next_blob is not None:
                model.import_parameter_blob(next_blob, strict=False)

            if warm_start_blobs is not None and warm_start_blobs[b] is not None:
                model.import_parameter_blob(warm_start_blobs[b], strict=False)

            precision_target = None
            if prev_pass_loss_by_block is not None and b in prev_pass_loss_by_block:
                precision_target = float(prev_pass_loss_by_block[b]) * (1.0 + precision_margin)
            elif reference_loss is not None:
                precision_target = reference_loss * (1.0 + precision_margin)

            default_refine_plan = [(50, 1e-5), (50, 5e-6)]
            resolved_plan = resolve_training_plan_for_block(
                rules=training_plan_rules or [],
                pass_id=pass_id,
                block_idx=b,
                n_blocks=len(blocks),
                default_stage=stage_plan,
                default_final=final_plan,
                default_refine=default_refine_plan,
            )

            block_stats = train_with_standard_schedule(
                model=model,
                stage_plan=resolved_plan["stage_plan"],
                final_plan=resolved_plan["final_plan"],
                eval_batches=5,
                precision_target=precision_target,
                max_refine_rounds=max_refine_rounds,
                refine_plan=resolved_plan["refine_plan"],
                label=label,
                coupling_const=float(coupling_const),
            )

            eval_loss = block_stats["eval_stats"]["mean_loss"]
            if reference_loss is None:
                reference_loss = eval_loss
                print(f"[Recursive] reference_loss set from terminal block: {reference_loss:.6e}")

            blob = model.export_parameter_blob()
            blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
            save_blob_npz(blob, blob_path)
            ckpt_path = None
            if save_tf_checkpoints:
                ckpt_path = os.path.join(pass_dir, f"block_{b:02d}.ckpt")
                model.save_model(ckpt_path)

            log_row = {
                "pass": int(pass_id),
                "block": int(b),
                "t_start": float(block["t_start"]),
                "t_end": float(block["t_end"]),
                "T_block": float(block["T_block"]),
                "eval_mean_loss": float(block_stats["eval_stats"]["mean_loss"]),
                "eval_std_loss": float(block_stats["eval_stats"]["std_loss"]),
                "eval_mean_loss_per_sample": float(block_stats["eval_stats"]["mean_loss_per_sample"]),
                "eval_std_loss_per_sample": float(block_stats["eval_stats"]["std_loss_per_sample"]),
                "eval_mean_y0": float(block_stats["eval_stats"]["mean_y0"]),
                "precision_target": None
                if block_stats["precision_target"] is None
                else float(block_stats["precision_target"]),
                "refine_rounds": int(block_stats["refine_rounds"]),
                "stage_plan_used": resolved_plan["stage_plan"],
                "final_plan_used": resolved_plan["final_plan"],
                "refine_plan_used": resolved_plan["refine_plan"],
                "blob_path": blob_path,
                "ckpt_path": ckpt_path,
                "was_frozen": False,
                "freeze_reason": "",
                "freeze_source_pass": None,
                "freeze_threshold": float(freeze_loss_threshold),
                "freeze_neighbor_radius": int(freeze_neighbor_radius),
                "active_set_enabled": bool(active_set_summary.get("enabled", False)),
            } | _prefixed_eval_diagnostics(block_stats["eval_stats"])
            logs.append(log_row)

            block_blobs[b] = blob
            next_blob = blob

            model.sess.close()

        logs = sorted(logs, key=lambda x: x["block"])
        return block_blobs, logs, float(reference_loss), pass_dir, active_set_summary

    pass_results = []
    prev_blobs = None
    prev_boundary_samples = None
    prev_pass_loss_by_block = None
    prev_pass_logs_by_block = None
    pass1_init_mode = str(pass1_init_mode or "base").strip().lower()

    for pass_id in range(1, int(n_passes) + 1):
        pass_init_mode_current = "recursive_empirical"
        boundary_source_current = "previous_pass_rollout"
        is_bootstrap_pass_current = False
        if pass_id == 1:
            if initial_boundary_samples is not None:
                pass1_jitter_scale = float(empirical_jitter_scale) if pass1_init_mode == "coarse" else 0.0
                generators = [
                    make_empirical_generator(
                        np.asarray(initial_boundary_samples[b], dtype=np.float32),
                        jitter_scale=pass1_jitter_scale,
                    )
                    for b in range(len(blocks))
                ]
                warm_start = initial_warm_start_blobs
                pass_init_mode_current = pass1_init_mode
                boundary_source_current = (
                    "coarse_prepass" if pass1_init_mode == "coarse" else "exact_diagnostic"
                )
                is_bootstrap_pass_current = False
            else:
                generators = [Xi_generator for _ in blocks]
                warm_start = None
                pass_init_mode_current = "base"
                boundary_source_current = "base_xi"
                is_bootstrap_pass_current = True
            warm_from_next = bool(pass1_warm_start_from_next)
        else:
            if prev_boundary_samples is None:
                if prev_blobs is None:
                    raise RuntimeError("Internal error: missing previous blobs for pass>=2")
                prev_boundary_samples = rollout_boundaries(
                    block_blobs=prev_blobs,
                    blocks=blocks,
                    Xi_generator=Xi_generator,
                    params=params,
                    M_rollout=rollout_M,
                    N_per_block=N_per_block,
                    D=D,
                    layers=layers,
                    T_total=T_total,
                    coupling_const=float(coupling_const),
                    model_spec=spec,
                )
            generators = [
                make_empirical_generator(prev_boundary_samples[b], jitter_scale=empirical_jitter_scale)
                for b in range(len(blocks))
            ]
            warm_start = prev_blobs if bool(cross_pass_warm_start) else None
            warm_from_next = False

        blobs_i, logs_i, ref_loss_i, pass_dir_i, active_set_summary_i = _run_pass(
            pass_id=pass_id,
            generators_per_block=generators,
            warm_start_blobs=warm_start,
            carry_over_blobs=prev_blobs,
            warm_start_from_next=warm_from_next,
            prev_pass_loss_by_block=prev_pass_loss_by_block,
            prev_pass_logs_by_block=prev_pass_logs_by_block,
        )

        prev_blobs = blobs_i
        prev_pass_loss_by_block = {
            int(row["block"]): float(row["eval_mean_loss"])
            for row in logs_i
            if "eval_mean_loss" in row
        }
        prev_pass_logs_by_block = {int(row["block"]): dict(row) for row in logs_i}
        prev_boundary_samples = rollout_boundaries(
            block_blobs=blobs_i,
            blocks=blocks,
            Xi_generator=Xi_generator,
            params=params,
            M_rollout=rollout_M,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
            coupling_const=float(coupling_const),
            model_spec=spec,
        )

        pass_results.append(
            {
                "pass_id": int(pass_id),
                "reference_loss": float(ref_loss_i),
                "logs": logs_i,
                "blobs": blobs_i,
                "models_dir": pass_dir_i,
                "pass_init_mode": pass_init_mode_current,
                "boundary_source": boundary_source_current,
                "is_bootstrap_pass": bool(is_bootstrap_pass_current),
                "active_set_summary": active_set_summary_i,
            }
        )

        if on_pass_end is not None:
            on_pass_end(
                {
                    "pass_id": int(pass_id),
                    "passes": list(pass_results),
                    "blocks": blocks,
                    "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
                }
            )

    result = {
        "blocks": blocks,
        "passes": pass_results,
        "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
    }

    for item in pass_results:
        if item["pass_id"] == 1:
            result["pass1"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }
        if item["pass_id"] == 2:
            result["pass2"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }

    return result

def run_recursive_coarse_prepass(
    Xi_generator,
    params,
    M,
    N_per_block,
    D,
    T_total,
    block_size,
    layers,
    stage_plan,
    final_plan,
    output_dir,
    precision_margin=0.10,
    training_plan_rules: Optional[List[Dict[str, Any]]] = None,
    pass1_warm_start_from_next: bool = False,
    empirical_jitter_scale: float = 0.0,
    iter_scale: float = 0.15,
    prepass_M: int = 0,
    prepass_N: int = 0,
    rollout_M: int = 0,
    curriculum_consts: Optional[List[float]] = None,
    curriculum_stage_scales: Optional[List[float]] = None,
    curriculum_jitter_scale: Optional[float] = None,
    coupling_const: float = 1.0,
    model_spec: Optional[ModelSpec] = None,
) -> Dict[str, Any]:
    spec = _resolve_model_spec(model_spec)
    spec.validate_state_dim(D)

    coarse_M = int(prepass_M)
    if coarse_M <= 0:
        coarse_M = max(64, min(int(M), max(256, int(round(float(M) * 0.25)))))

    coarse_N = int(prepass_N)
    if coarse_N <= 0:
        coarse_N = max(8, min(int(N_per_block), int(round(max(8.0, float(N_per_block) * 0.5)))))

    coarse_rollout_M = int(rollout_M)
    if coarse_rollout_M <= 0:
        coarse_rollout_M = max(int(coarse_M), 512)

    resolved_curriculum_consts, resolved_curriculum_scales = resolve_coarse_curriculum_schedule(
        curriculum_consts=[] if curriculum_consts is None else curriculum_consts,
        curriculum_stage_scales=[] if curriculum_stage_scales is None else curriculum_stage_scales,
        terminal_const=float(coupling_const),
    )
    resolved_curriculum_jitter_scale = (
        float(empirical_jitter_scale)
        if curriculum_jitter_scale is None
        else float(curriculum_jitter_scale)
    )
    if (not np.isfinite(resolved_curriculum_jitter_scale)) or resolved_curriculum_jitter_scale < 0.0:
        raise ValueError(
            "coarse curriculum jitter scale must be finite and >= 0, "
            f"got {resolved_curriculum_jitter_scale}"
        )

    print(
        "[CoarseCurriculum] "
        + " -> ".join(
            f"const={const_value:.2f} (x{stage_scale:.3f})"
            for const_value, stage_scale in zip(resolved_curriculum_consts, resolved_curriculum_scales)
        )
        + f", boundary_jitter={resolved_curriculum_jitter_scale:.4f}"
    )

    stage_input_boundary_samples = None
    stage_input_blobs = None
    prepass_result = None
    stage_summaries = []
    for stage_idx, (stage_const, stage_scale_multiplier) in enumerate(
        zip(resolved_curriculum_consts, resolved_curriculum_scales),
        start=1,
    ):
        effective_stage_iter_scale = float(iter_scale) * float(stage_scale_multiplier)
        stage_stage_plan = scale_schedule(
            stage_plan,
            iter_scale=effective_stage_iter_scale,
            min_iter=50,
        )
        stage_final_plan = scale_schedule(
            final_plan,
            iter_scale=effective_stage_iter_scale,
            min_iter=50,
        )
        stage_rules = scale_training_plan_rules(
            training_plan_rules or [],
            iter_scale=effective_stage_iter_scale,
            min_iter=50,
        )
        stage_tag = f"stage_{stage_idx:02d}_const_{_const_stage_tag(stage_const)}"
        stage_output_dir = os.path.join(output_dir, stage_tag)
        stage_init_mode = "base" if stage_input_boundary_samples is None else "coarse"
        stage_empirical_jitter = (
            0.0
            if stage_input_boundary_samples is None
            else float(resolved_curriculum_jitter_scale)
        )
        stage_warm_from_next = bool(pass1_warm_start_from_next) if stage_input_blobs is None else False

        print(
            f"[CoarseCurriculum] stage={stage_idx}/{len(resolved_curriculum_consts)}, "
            f"const={float(stage_const):.2f}, iter_scale={effective_stage_iter_scale:.3f}, "
            f"init={stage_init_mode}, jitter={stage_empirical_jitter:.4f}"
        )

        prepass_result = run_recursive_training(
            Xi_generator=Xi_generator,
            params=params,
            M=int(coarse_M),
            N_per_block=int(coarse_N),
            D=D,
            T_total=T_total,
            block_size=block_size,
            layers=layers,
            stage_plan=stage_stage_plan,
            final_plan=stage_final_plan,
            output_dir=stage_output_dir,
            precision_margin=precision_margin,
            max_refine_rounds=1,
            rollout_M=int(coarse_rollout_M),
            save_tf_checkpoints=False,
            training_plan_rules=stage_rules,
            pass1_warm_start_from_next=stage_warm_from_next,
            cross_pass_warm_start=False,
            n_passes=1,
            empirical_jitter_scale=stage_empirical_jitter,
            pass1_init_mode=stage_init_mode,
            initial_boundary_samples=stage_input_boundary_samples,
            initial_warm_start_blobs=stage_input_blobs,
            freeze_stable_blocks_after_pass=0,
            freeze_loss_threshold=0.0,
            freeze_neighbor_radius=0,
            coupling_const=float(stage_const),
            on_pass_end=None,
            model_spec=spec,
        )

        stage_input_boundary_samples = prepass_result.get("boundary_samples", [])
        stage_input_blobs = prepass_result.get("pass1", {}).get("blobs", None)
        pass1_logs = prepass_result.get("pass1", {}).get("logs", [])
        stage_summaries.append(
            {
                "stage_index": int(stage_idx),
                "const": float(stage_const),
                "relative_iter_scale": float(stage_scale_multiplier),
                "effective_iter_scale": float(effective_stage_iter_scale),
                "init_mode": stage_init_mode,
                "empirical_jitter_scale": float(stage_empirical_jitter),
                "stage_plan": stage_stage_plan,
                "final_plan": stage_final_plan,
                "training_plan_rules_count": len(stage_rules),
                "reference_loss": float(prepass_result.get("pass1", {}).get("reference_loss", 0.0)),
                "mean_eval_loss_per_sample": None
                if len(pass1_logs) == 0
                else float(np.mean([row["eval_mean_loss_per_sample"] for row in pass1_logs])),
                "boundary_stats": summarize_boundary_samples(stage_input_boundary_samples),
                "models_dir": stage_output_dir,
            }
        )

    if prepass_result is None:
        raise RuntimeError("coarse prepass curriculum produced no stages")

    return {
        "boundary_samples": prepass_result.get("boundary_samples", []),
        "pass1_blobs": prepass_result.get("pass1", {}).get("blobs", None),
        "summary": {
            "M": int(coarse_M),
            "N": int(coarse_N),
            "rollout_M": int(coarse_rollout_M),
            "iter_scale": float(iter_scale),
            "curriculum_consts": resolved_curriculum_consts,
            "curriculum_stage_scales": resolved_curriculum_scales,
            "curriculum_jitter_scale": float(resolved_curriculum_jitter_scale),
            "n_curriculum_stages": len(resolved_curriculum_consts),
            "stage_summaries": stage_summaries,
            "boundary_stats": summarize_boundary_samples(prepass_result.get("boundary_samples", [])),
        },
    }
