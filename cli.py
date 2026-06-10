"""Single command-line entry point for program execution and tests."""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import hashlib
import json

from .application_metrics import summarize_application_alpha, summarize_controlled_uncontrolled_comparison
from .exact import compute_stitched_exact_bundle, save_exact_error_timeseries_csv
from .io_utils import export_standard_parameter_blob, save_blob_npz, save_json, save_rows_csv, _to_serializable
from .naming import _pass_index, _pass_label
from .pascucci_plotting import plot_pascucci_paper_bundle_from_artifacts
from .plotting import plot_recursive_exact_comparison, plot_stage_logs, _PLOTTING_AVAILABLE
from .sampling import build_blocks, build_stitched_rollout_inputs, summarize_boundary_samples
from .schedules import load_training_plan_csv, parse_float_sequence_arg, resolve_coarse_curriculum_schedule
from .tf_backend import set_tf_seed
from .tests import run_tests
from .model_specs import get_model_spec


def _parse_optional_component_weights(value: str, arg_name: str, D: int):
    value = str(value or "").strip()
    if value == "":
        return None
    weights = parse_float_sequence_arg(value, arg_name=arg_name)
    if len(weights) not in (1, int(D)):
        raise ValueError(f"{arg_name} must contain 1 or {int(D)} values, got {len(weights)}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{arg_name} must contain non-negative values")
    return [float(weight) for weight in weights]

def _run_config_sha256(config: dict) -> str:
    payload = {k: v for k, v in config.items() if k not in {"run_config_sha256", "timestamp"}}
    encoded = json.dumps(_to_serializable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summarize_application_cost_result(result: dict) -> dict:
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


def _application_pathwise_npz(result: dict, prefix: str) -> dict:
    return {
        f"{prefix}_{key}": np.asarray(value, dtype=np.float32)
        for key, value in result.get("pathwise", {}).items()
    }


def _save_application_cost_artifacts(
    *,
    output_dir: str,
    stem: str,
    model_name: str,
    horizon: dict,
    controlled: dict,
    uncontrolled: Optional[dict] = None,
) -> dict:
    pathwise = _application_pathwise_npz(controlled, "controlled")
    payload = {
        "schema": controlled["schema"],
        "model_name": model_name,
        "horizon": horizon,
        "controlled": _summarize_application_cost_result(controlled),
    }
    if uncontrolled is not None:
        pathwise.update(_application_pathwise_npz(uncontrolled, "uncontrolled"))
        payload["uncontrolled"] = _summarize_application_cost_result(uncontrolled)
        payload["comparison"] = summarize_controlled_uncontrolled_comparison(
            controlled=controlled,
            uncontrolled=uncontrolled,
        )

    json_path = os.path.join(output_dir, f"{stem}.json")
    npz_path = os.path.join(output_dir, f"{stem}.npz")
    payload["output_path"] = json_path
    payload["pathwise_npz_path"] = npz_path
    payload["pathwise_npz_keys"] = sorted(pathwise)
    save_blob_npz(pathwise, npz_path)
    save_json(payload, json_path)
    return payload


def run_program(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Recursive time-stitching experiment (TF2 native)")
    parser.add_argument("--mode", type=str, default="recursive", choices=["standard", "recursive", "both"])
    parser.add_argument(
        "--model",
        type=str,
        default="quadratic_coupled",
        help=(
            "Modello da usare via ModelSpec/factory. "
            "Default: quadratic_coupled. Valori supportati ora: quadratic_coupled, pascucci."
        ),
    )
    parser.add_argument(
        "--pascucci_cost_profile",
        type=str,
        default="exp",
        choices=["exp", "exp_minus_offset"],
        help=(
            "Profilo di costo di Pascucci. "
            "exp: costo proporzionale a exp(S) (come in Pascucci); "
            "exp_minus_offset: costo proporzionale a exp(S) - offset, con offset >= 0."
        ),
    )
    parser.add_argument(
        "--pascucci_cost_offset",
        type=float,
        default=0.0,
        help="Offset per il profilo di costo di Pascucci.",
    )
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--N", type=int, default=100, help="N steps per block")
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--T_standard", type=float, default=12.0)
    parser.add_argument("--T_total", type=float, default=48.0)
    parser.add_argument("--block_size", type=float, default=12.0)
    parser.add_argument("--output_dir", type=str, default="recursive1_outputs")
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Numero totale di pass ricorsive da eseguire (>=1).",
    )
    parser.add_argument(
        "--empirical_jitter_scale",
        type=float,
        default=0.02,
        help="Rumore relativo usato nel generatore empirico per pass >= 2.",
    )
    parser.add_argument(
        "--training_plan_csv",
        type=str,
        default="",
        help=(
            "CSV opzionale con piano training per blocco/pass. "
            "Colonne richieste: pass_scope,block_scope,phase,n_iter,lr "
            "(opzionali: order,enabled)."
        ),
    )
    parser.add_argument(
        "--pass1_warm_start_from_next",
        action="store_true",
        help=(
            "Se attivo, in pass1 il blocco i viene inizializzato coi pesi del blocco i+1 "
            "(quando disponibile). Le passate successive possono usare warm-start dal pass "
            "precedente (default attivo, disattivabile con --disable_cross_pass_warm_start)."
        ),
    )
    parser.add_argument(
        "--disable_cross_pass_warm_start",
        action="store_true",
        help=(
            "Se attivo, disabilita il warm-start dalle passate precedenti "
            "(warm_start=prev_blobs) per pass>=2."
        ),
    )
    parser.add_argument(
        "--freeze_stable_blocks_after_pass",
        type=int,
        default=0,
        help=(
            "Attiva il freezing dei blocchi stabili per pass_id > valore. "
            "0 disabilita la logica active-set/freezing."
        ),
    )
    parser.add_argument(
        "--freeze_loss_threshold",
        type=float,
        default=0.0,
        help=(
            "Soglia su eval_mean_loss_per_sample della passata precedente: "
            "i blocchi sopra soglia restano attivi, quelli sotto possono essere congelati."
        ),
    )
    parser.add_argument(
        "--freeze_neighbor_radius",
        type=int,
        default=1,
        help=(
            "Numero di blocchi vicini a ciascun blocco instabile da mantenere attivi "
            "quando il freezing e' abilitato."
        ),
    )
    parser.add_argument(
        "--exact_solution",
        type=str,
        default="none",
        help=(
            "Profilo opzionale per confronto con soluzione esatta. "
            "Valori supportati: none, quadratic_coupled"
        ),
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="auto",
        choices=[
            "auto",
            "loss",
            "last",
            "exact_mae_y",
            "exact_rmse_y",
            "exact_abs_y0",
            "exact_mae_z",
            "exact_mae_z_s",
        ],
        help=(
            "Metrica di selezione della pass finale: "
            "auto usa exact_mae_y se exact_solution e' attiva, altrimenti loss; "
            "last forza la selezione dell'ultima passata completata; "
            "exact_mae_z ed exact_mae_z_s selezionano rispettivamente su tutto Z e su Z_S."
        ),
    )
    parser.add_argument(
        "--exact_regression_tolerance",
        type=float,
        default=0.20,
        help=(
            "Tolleranza regressione relativa tra pass consecutive su mean_abs_error_y "
            "(es. 0.20 = +20%%). <=0 disabilita il guardrail."
        ),
    )
    parser.add_argument(
        "--exact_regression_action",
        type=str,
        default="warn",
        choices=["warn", "error", "ignore"],
        help="Azione quando il guardrail exact rileva regressione oltre soglia.",
    )
    parser.add_argument(
        "--eval_bundle_path",
        type=str,
        default="",
        help=(
            "Percorso opzionale a evaluation_bundle.npz da riusare per confronto path-by-path "
            "tra pass/run."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=1234,
        help="Seed usato per costruire un evaluation bundle nuovo quando non viene caricato.",
    )
    parser.add_argument(
        "--visual_sample_paths",
        type=int,
        default=8,
        help=(
            "Numero di path random deterministici usati solo per figure Y/Z pred-vs-exact. "
            "Le metriche exact restano calcolate sull'evaluation bundle grande."
        ),
    )
    parser.add_argument(
        "--visual_seed",
        type=int,
        default=-1,
        help=(
            "Seed per i path visuali. Usa -1 per derivarlo automaticamente da --eval_seed."
        ),
    )
    parser.add_argument(
        "--pass1_init",
        type=str,
        default="base",
        choices=["base", "coarse", "exact"],
        help=(
            "Strategia di inizializzazione della passata 1. "
            "base=bootstrap puro; coarse=prepass economica; exact=boundary oracle per benchmark."
        ),
    )
    parser.add_argument(
        "--coarse_prepass_M",
        type=int,
        default=0,
        help="Batch size della prepass coarse. 0=auto.",
    )
    parser.add_argument(
        "--coarse_prepass_N",
        type=int,
        default=0,
        help="Numero di step temporali per blocco nella prepass coarse. 0=auto.",
    )
    parser.add_argument(
        "--coarse_prepass_iter_scale",
        type=float,
        default=0.15,
        help="Fattore globale di scala delle iterazioni del training plan nella prepass coarse.",
    )
    parser.add_argument(
        "--coarse_curriculum_consts",
        type=str,
        default="0.0,0.5,0.75,1.0",
        help=(
            "Sequenza di const per il curriculum della coarse prepass. "
            "L'ultimo stage viene sempre riallineato al const del training ricorsivo."
        ),
    )
    parser.add_argument(
        "--coarse_curriculum_stage_scales",
        type=str,
        default="1.0,0.5,0.35,0.25",
        help=(
            "Moltiplicatori relativi delle iterazioni per ciascuno stage coarse. "
            "Si applicano sopra coarse_prepass_iter_scale. Un solo valore viene broadcastato."
        ),
    )
    parser.add_argument(
        "--coarse_curriculum_jitter_scale",
        type=float,
        default=None,
        help=(
            "Jitter relativo sui boundary samples empirici tra stage del curriculum coarse. "
            "Default: eredita --empirical_jitter_scale; usa 0.0 per ripristinare il vecchio "
            "comportamento senza jitter tra stage."
        ),
    )
    parser.add_argument(
        "--coarse_prepass_seed",
        type=int,
        default=4321,
        help="Seed logico da salvare per la prepass coarse.",
    )
    parser.add_argument(
        "--exact_init_seed",
        type=int,
        default=4321,
        help="Seed usato per i boundary samples exact nella diagnostica di pass1.",
    )
    parser.add_argument(
        "--const_override",
        type=float,
        default=None,
        help="Override del parametro const per Girsanov"
    )
    parser.add_argument(
        "--dynamic_loss_dt_normalization",
        action="store_true",
        help=(
            "Normalizza il residuo dinamico per dt nella composizione di loss migliorata. "
            "Default disattivo per preservare la compatibilita' legacy."
        ),
    )
    parser.add_argument(
        "--dynamic_loss_weight",
        type=float,
        default=1.0,
        help="Peso del residuo dinamico nella composizione di loss migliorata.",
    )
    parser.add_argument(
        "--same_xi_antithetic_sampling",
        action="store_true",
        help=(
            "Nelle coppie antitetiche usa la stessa Xi iniziale oltre a Brownian increments opposti. "
            "Default disattivo per preservare il campionamento legacy."
        ),
    )
    parser.add_argument(
        "--terminal_y_loss_weight",
        type=float,
        default=1.0,
        help="Peso del vincolo terminale su Y nella composizione di loss migliorata.",
    )
    parser.add_argument(
        "--terminal_z_loss_weight",
        type=float,
        default=1.0,
        help="Peso globale del vincolo terminale su Z nella composizione di loss migliorata.",
    )
    parser.add_argument(
        "--terminal_z_component_weights",
        type=str,
        default="",
        help=(
            "Pesi opzionali per le componenti terminali di Z, separati da virgola. "
            "Usa un solo valore per broadcast oppure D valori, ad esempio '3,1,2,0'."
        ),
    )
    parser.add_argument(
        "--structural_z_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Peso opzionale per penalizzare componenti strutturalmente nulle di Z. "
            "Resta inattivo a 0."
        ),
    )
    parser.add_argument(
        "--structural_z_component_weights",
        type=str,
        default="",
        help=(
            "Pesi per la penalita strutturale sulle componenti di Z, separati da virgola. "
            "Esempio per penalizzare Z_H: '0,1,0,0'."
        ),
    )
    args = parser.parse_args(argv)
    if int(args.visual_sample_paths) < 1:
        raise ValueError("--visual_sample_paths must be >= 1")

    from .orchestration import (
        print_recursive_pass,
        run_recursive_coarse_prepass,
        run_recursive_training,
        run_standard_reference,
    )

    np.random.seed(1234)
    set_tf_seed(1234)

    M = args.M
    N = args.N
    D = args.D
    model_spec = get_model_spec(args.model)
    model_spec.validate_state_dim(D)
    effective_const = 1.0 if args.const_override is None else float(args.const_override)
    terminal_z_component_weights = _parse_optional_component_weights(
        args.terminal_z_component_weights,
        arg_name="--terminal_z_component_weights",
        D=D,
    )
    structural_z_component_weights = _parse_optional_component_weights(
        args.structural_z_component_weights,
        arg_name="--structural_z_component_weights",
        D=D,
    )
    coarse_curriculum_consts = parse_float_sequence_arg(
        args.coarse_curriculum_consts,
        arg_name="--coarse_curriculum_consts",
    )
    coarse_curriculum_stage_scales = parse_float_sequence_arg(
        args.coarse_curriculum_stage_scales,
        arg_name="--coarse_curriculum_stage_scales",
    )
    resolved_coarse_curriculum_consts, resolved_coarse_curriculum_stage_scales = (
        resolve_coarse_curriculum_schedule(
            curriculum_consts=coarse_curriculum_consts,
            curriculum_stage_scales=coarse_curriculum_stage_scales,
            terminal_const=float(effective_const),
        )
    )
    effective_coarse_curriculum_jitter_scale = (
        float(args.empirical_jitter_scale)
        if args.coarse_curriculum_jitter_scale is None
        else float(args.coarse_curriculum_jitter_scale)
    )
    if (
        not np.isfinite(effective_coarse_curriculum_jitter_scale)
        or effective_coarse_curriculum_jitter_scale < 0.0
    ):
        raise ValueError(
            "--coarse_curriculum_jitter_scale must be finite and >= 0 "
            f"(got {effective_coarse_curriculum_jitter_scale})"
        )

    params = model_spec.build_default_params(const=effective_const)
    pascucci_cost_profile = str(args.pascucci_cost_profile).strip().lower()
    pascucci_cost_offset = float(args.pascucci_cost_offset)
    if model_spec.name == "pascucci":
        params["pascucci_cost_profile"] = pascucci_cost_profile
        params["pascucci_cost_offset"] = np.float32(pascucci_cost_offset)
    elif pascucci_cost_profile != "exp" or abs(pascucci_cost_offset) > 1e-8:
        raise ValueError("--pascucci_cost_profile/--pascucci_cost_offset are supported only with --model pascucci")
    params.update(
        {
            "same_xi_antithetic_sampling": bool(args.same_xi_antithetic_sampling),
            "dynamic_loss_dt_normalization": bool(args.dynamic_loss_dt_normalization),
            "dynamic_loss_weight": np.float32(args.dynamic_loss_weight),
            "terminal_y_loss_weight": np.float32(args.terminal_y_loss_weight),
            "terminal_z_loss_weight": np.float32(args.terminal_z_loss_weight),
            "terminal_z_component_weights": terminal_z_component_weights,
            "structural_z_loss_weight": np.float32(args.structural_z_loss_weight),
            "structural_z_component_weights": structural_z_component_weights,
        }
    )
    layers = model_spec.build_layers(D)
    stage_plan = [(5000, 1e-3), (5000, 5e-4), (5000, 1e-4), (5000, 5e-5)]
    final_plan = [(5000, 1e-5), (5000, 5e-6)]
    training_plan_rules = load_training_plan_csv(args.training_plan_csv)
    training_plan_effective_source = str(args.training_plan_csv or "").strip()

    if len(training_plan_rules) > 0:
        print(
            f"[TrainingPlan] loaded {len(training_plan_rules)} rules from {training_plan_effective_source}"
        )

    exact_solution = model_spec.build_exact_solution(args.exact_solution, params, D)
    requested_selection_metric = str(args.selection_metric)
    requested_exact_regression_action = str(args.exact_regression_action)
    effective_selection_metric = requested_selection_metric
    effective_exact_regression_action = requested_exact_regression_action
    const_override_active = abs(float(effective_const) - 1.0) > 1.0e-8
    if exact_solution is None:
        print("[ExactSolution] disabled")
    else:
        print(f"[ExactSolution] enabled profile='{exact_solution['name']}'")
    if args.mode in ("recursive", "both") and const_override_active:
        if str(args.pass1_init or "base").strip().lower() == "exact":
            raise ValueError(
                "pass1_init='exact' non e' supportato con const_override != 1.0: "
                "l'inizializzazione exact usa ancora il drift del problema base "
                "e non il sistema Girsanov-like modificato."
            )

    visual_seed_requested = None if int(args.visual_seed) < 0 else int(args.visual_seed)
    visual_seed_effective = int(args.eval_seed) + 7919 if visual_seed_requested is None else int(visual_seed_requested)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)

    run_config = {
        "timestamp": run_id,
        "mode": args.mode,
        "M": M,
        "N": N,
        "D": D,
        "T_standard": args.T_standard,
        "T_total": args.T_total,
        "block_size": args.block_size,
        "passes": int(args.passes),
        "empirical_jitter_scale": float(args.empirical_jitter_scale),
        "layers": layers,
        "stage_plan": stage_plan,
        "final_plan": final_plan,
        "training_plan_csv": args.training_plan_csv,
        "training_plan_effective_source": training_plan_effective_source,
        "training_plan_rules_count": len(training_plan_rules),
        "training_plan_rules": training_plan_rules,
        "pass1_warm_start_from_next": bool(args.pass1_warm_start_from_next),
        "cross_pass_warm_start": not bool(args.disable_cross_pass_warm_start),
        "freeze_stable_blocks_after_pass": int(args.freeze_stable_blocks_after_pass),
        "freeze_loss_threshold": float(args.freeze_loss_threshold),
        "freeze_neighbor_radius": int(args.freeze_neighbor_radius),
        "selection_metric_requested": requested_selection_metric,
        "selection_metric": effective_selection_metric,
        "exact_regression_tolerance": float(args.exact_regression_tolerance),
        "exact_regression_action_requested": requested_exact_regression_action,
        "exact_regression_action": effective_exact_regression_action,
        "eval_bundle_path": str(args.eval_bundle_path),
        "eval_seed": int(args.eval_seed),
        "visual_sample_paths": int(args.visual_sample_paths),
        "visual_seed": visual_seed_requested,
        "visual_seed_effective": int(visual_seed_effective),
        "pass1_init": str(args.pass1_init),
        "coarse_prepass_M": int(args.coarse_prepass_M),
        "coarse_prepass_N": int(args.coarse_prepass_N),
        "coarse_prepass_iter_scale": float(args.coarse_prepass_iter_scale),
        "coarse_curriculum_consts_requested": coarse_curriculum_consts,
        "coarse_curriculum_stage_scales_requested": coarse_curriculum_stage_scales,
        "coarse_curriculum_consts": resolved_coarse_curriculum_consts,
        "coarse_curriculum_stage_scales": resolved_coarse_curriculum_stage_scales,
        "coarse_curriculum_jitter_scale_requested": args.coarse_curriculum_jitter_scale,
        "coarse_curriculum_jitter_scale": float(effective_coarse_curriculum_jitter_scale),
        "coarse_prepass_seed": int(args.coarse_prepass_seed),
        "exact_init_seed": int(args.exact_init_seed),
        "exact_solution": "none" if exact_solution is None else exact_solution["name"],
        "params": params,
        "const_override": None if args.const_override is None else float(args.const_override),
        "effective_const": float(effective_const),
        "same_xi_antithetic_sampling": bool(args.same_xi_antithetic_sampling),
        "dynamic_loss_dt_normalization": bool(args.dynamic_loss_dt_normalization),
        "dynamic_loss_weight": float(args.dynamic_loss_weight),
        "terminal_y_loss_weight": float(args.terminal_y_loss_weight),
        "terminal_z_loss_weight": float(args.terminal_z_loss_weight),
        "terminal_z_component_weights": terminal_z_component_weights,
        "structural_z_loss_weight": float(args.structural_z_loss_weight),
        "structural_z_component_weights": structural_z_component_weights,
        "plotting_available": _PLOTTING_AVAILABLE,
        "model_requested": str(args.model),
        "model_name": model_spec.name,
        "state_labels": model_spec.state_labels,
        "z_labels": model_spec.z_labels,
        "application_metric_schema": model_spec.application_metric_schema,
        "application_metric_names": list(model_spec.application_metric_names),
        "application_metric_aggregation": model_spec.application_metric_aggregation,
        "seed_manifest": {
            "global_seed": 1234,
            "eval_seed": int(args.eval_seed),
            "visual_seed": visual_seed_requested,
            "visual_seed_effective": int(visual_seed_effective),
            "coarse_prepass_seed": int(args.coarse_prepass_seed),
            "exact_init_seed": int(args.exact_init_seed),
        },
    }
    run_config["run_config_sha256"] = _run_config_sha256(run_config)
    save_json(run_config, os.path.join(run_root, "run_config.json"))
    print(f"[Artifacts] run directory: {run_root}")

    if args.mode in ("standard", "both"):
        print("\n==================== STANDARD ====================")
        std_dir = os.path.join(run_root, "standard")
        os.makedirs(std_dir, exist_ok=True)
        standard_const = float(effective_const)
        model_std, logs_std = run_standard_reference(
            Xi_generator=model_spec.xi_generator,
            params=params,
            M=M,
            N=N,
            D=D,
            T=args.T_standard,
            layers=layers,
            stage_plan=stage_plan,
            final_plan=final_plan,
            coupling_const=standard_const,
            model_spec=model_spec,
        )

        std_ckpt_path = os.path.join(std_dir, "model.ckpt")
        model_std.save_model(std_ckpt_path)

        std_blob = export_standard_parameter_blob(model_std)
        std_blob_path = os.path.join(std_dir, "model_weights.npz")
        save_blob_npz(std_blob, std_blob_path)

        save_rows_csv(logs_std.get("stage_logs", []), os.path.join(std_dir, "stage_logs.csv"))
        plot_stage_logs(
            logs_std.get("stage_logs", []),
            out_prefix=os.path.join(std_dir, "standard"),
            title="Standard",
        )

        std_summary = {
            "final_eval": logs_std.get("eval_stats", {}),
            "refine_rounds": logs_std.get("refine_rounds", 0),
            "checkpoint_path": std_ckpt_path,
            "weights_npz_path": std_blob_path,
        }
        if (
            model_spec.application_metric_schema != "none"
            and hasattr(model_std, "application_cost_functional")
        ):
            metric_blocks = [
                {
                    "idx": 0,
                    "t_start": 0.0,
                    "t_end": float(args.T_standard),
                    "T_block": float(args.T_standard),
                }
            ]
            Xi_metric = model_spec.deterministic_xi(M, D, seed=int(args.eval_seed))
            metric_rollout = build_stitched_rollout_inputs(
                blocks=metric_blocks,
                M=Xi_metric.shape[0],
                N_per_block=N,
                D=D,
                seed=int(args.eval_seed),
            )
            t_metric, W_metric = metric_rollout[0]
            controlled_metric = model_std.application_cost_functional(
                t_metric,
                W_metric,
                Xi_metric,
                const_value=float(standard_const),
                baseline_mode="controlled",
            )
            uncontrolled_metric = model_std.application_cost_functional(
                t_metric,
                W_metric,
                Xi_metric,
                const_value=float(standard_const),
                baseline_mode="uncontrolled",
            )
            std_summary["application_metrics"] = _save_application_cost_artifacts(
                output_dir=std_dir,
                stem="application_metrics",
                model_name=model_spec.name,
                horizon={
                    "t_start": 0.0,
                    "t_end": float(args.T_standard),
                    "T_total": float(args.T_standard),
                    "n_time_points": int(t_metric.shape[1]),
                    "n_steps": int(t_metric.shape[1] - 1),
                    "sample_paths": int(Xi_metric.shape[0]),
                    "eval_seed": int(args.eval_seed),
                },
                controlled=controlled_metric,
                uncontrolled=uncontrolled_metric,
            )
        if exact_solution is not None:
            t_test, W_test, Xi_test = model_std.fetch_minibatch()
            X_pred, Y_pred, Z_pred = model_std.predict(Xi_test, t_test, W_test, const_value=float(standard_const))
            stitched_std = {
                "t": t_test.astype(np.float32),
                "X": X_pred.astype(np.float32),
                "Y": Y_pred.astype(np.float32),
                "Z": Z_pred.astype(np.float32),
            }
            exact_std = compute_stitched_exact_bundle(
                stitched=stitched_std,
                exact_solution=exact_solution,
            )
            print(
                "[Exact][Standard] "
                f"mean_pred_Y0={exact_std['summary']['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_std['summary']['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_std['summary']['abs_error_mean_y0']:.6e}"
            )

            save_json(
                {
                    "summary": exact_std["summary"],
                    "timeseries": exact_std["timeseries"],
                },
                os.path.join(std_dir, "exact_metrics.json"),
            )
            save_exact_error_timeseries_csv(
                exact_std["timeseries"],
                os.path.join(std_dir, "exact_errors.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_std,
                Y_exact=exact_std["Y_exact"],
                Z_exact=exact_std["Z_exact"],
                blocks=[{"t_start": 0.0, "t_end": float(args.T_standard), "T_block": float(args.T_standard)}],
                out_dir=os.path.join(std_dir, "plots"),
                sample_paths=8,
                file_suffix="",
            )
            std_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "summary": exact_std["summary"],
            }
        else:
            std_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        save_json(std_summary, os.path.join(std_dir, "results.json"))

        print(f"[STANDARD] final eval: {logs_std['eval_stats']}")
        model_std.sess.close()

    if args.mode in ("recursive", "both"):
        print("\n==================== RECURSIVE ====================")
        rec_dir = os.path.join(run_root, "recursive")
        os.makedirs(rec_dir, exist_ok=True)
        recursive_const = float(effective_const)
        pass1_init_mode = str(args.pass1_init or "base").strip().lower()
        initial_boundary_samples = None
        initial_warm_start_blobs = None
        initialization_summary = {
            "pass1_init": pass1_init_mode,
            "coarse_prepass": None,
            "exact_initialization": None,
        }

        explicit_eval_bundle = str(args.eval_bundle_path or "").strip()
        if explicit_eval_bundle != "":
            eval_bundle_path = os.path.abspath(os.path.expanduser(explicit_eval_bundle))
        else:
            eval_bundle_path = os.path.abspath(os.path.join(rec_dir, "evaluation_bundle.npz"))

        rollout_M_recursive = max(2000, M)
        if pass1_init_mode == "coarse":
            np_state_before_prepass = np.random.get_state()
            np.random.seed(int(args.coarse_prepass_seed))
            set_tf_seed(int(args.coarse_prepass_seed))
            coarse_prepass_dir = os.path.join(run_root, "coarse_prepass", "models")
            try:
                coarse_prepass = run_recursive_coarse_prepass(
                    Xi_generator=model_spec.xi_generator,
                    params=params,
                    M=M,
                    N_per_block=N,
                    D=D,
                    T_total=args.T_total,
                    block_size=args.block_size,
                    layers=layers,
                    stage_plan=stage_plan,
                    final_plan=final_plan,
                    output_dir=coarse_prepass_dir,
                    precision_margin=0.10,
                    training_plan_rules=training_plan_rules,
                    pass1_warm_start_from_next=bool(args.pass1_warm_start_from_next),
                    empirical_jitter_scale=float(args.empirical_jitter_scale),
                    iter_scale=float(args.coarse_prepass_iter_scale),
                    prepass_M=int(args.coarse_prepass_M),
                    prepass_N=int(args.coarse_prepass_N),
                    rollout_M=rollout_M_recursive,
                    curriculum_consts=resolved_coarse_curriculum_consts,
                    curriculum_stage_scales=resolved_coarse_curriculum_stage_scales,
                    curriculum_jitter_scale=float(effective_coarse_curriculum_jitter_scale),
                    coupling_const=float(recursive_const),
                    model_spec=model_spec,
                )
            finally:
                np.random.set_state(np_state_before_prepass)
                set_tf_seed(1234)
            initial_boundary_samples = coarse_prepass["boundary_samples"]
            initial_warm_start_blobs = coarse_prepass["pass1_blobs"]
            initialization_summary["coarse_prepass"] = coarse_prepass["summary"]
            initialization_summary["coarse_prepass"]["seed"] = int(args.coarse_prepass_seed)
            save_json(
                initialization_summary["coarse_prepass"],
                os.path.join(run_root, "coarse_prepass", "summary.json"),
            )
            print(
                "[Pass1Init] coarse prepass ready: "
                f"M={initialization_summary['coarse_prepass']['M']}, "
                f"N={initialization_summary['coarse_prepass']['N']}, "
                f"iter_scale={initialization_summary['coarse_prepass']['iter_scale']:.3f}, "
                "curriculum="
                + " -> ".join(
                    f"{value:.2f}"
                    for value in initialization_summary["coarse_prepass"]["curriculum_consts"]
                )
            )
        elif pass1_init_mode == "exact":
            np_state_before_exact = np.random.get_state()
            np.random.seed(int(args.exact_init_seed))
            try:
                if model_spec.build_exact_initial_boundary_samples is None:
                    raise ValueError(
                        f"pass1_init='exact' is not supported for model '{model_spec.name}'"
                    )
                initial_boundary_samples = model_spec.build_exact_initial_boundary_samples(
                    Xi_generator=model_spec.xi_generator,
                    exact_solution=exact_solution,
                    params=params,
                    blocks=build_blocks(T_total=args.T_total, block_size=args.block_size),
                    M_rollout=rollout_M_recursive,
                    N_per_block=N,
                    D=D,
                    seed=int(args.exact_init_seed),
                )
            finally:
                np.random.set_state(np_state_before_exact)
            initialization_summary["exact_initialization"] = {
                "seed": int(args.exact_init_seed),
                "boundary_stats": summarize_boundary_samples(initial_boundary_samples),
            }
            save_json(
                initialization_summary["exact_initialization"],
                os.path.join(run_root, "exact_init_summary.json"),
            )
            print(
                "[Pass1Init] exact diagnostic ready: "
                f"M={rollout_M_recursive}, seed={int(args.exact_init_seed)}"
            )

        excluded_pass_ids_from_selection = (
            [1]
            if (pass1_init_mode == "base" and int(args.passes) > 1)
            else []
        )

        pass_plot_summary_holder = {"summary": None}

        def _on_recursive_pass_end(progress: Dict[str, Any]) -> None:
            passes_so_far = sorted(progress.get("passes", []), key=lambda x: int(x["pass_id"]))
            if len(passes_so_far) == 0:
                return
            pass_id = int(progress.get("pass_id", passes_so_far[-1]["pass_id"]))
            is_last_requested_pass = pass_id >= int(args.passes)
            print(
                f"\n[RecursivePlot] completed {_pass_label(pass_id)}: "
                f"updating cumulative plots up to {_pass_label(pass_id)}"
            )
            pass_plot_summary_holder["summary"] = print_recursive_pass(
                pass_entries=passes_so_far,
                blocks=progress.get("blocks", []),
                rec_dir=rec_dir,
                params=params,
                N_per_block=N,
                D=D,
                layers=layers,
                T_total=args.T_total,
                exact_solution=exact_solution,
                selection_metric=effective_selection_metric,
                exact_regression_tolerance=float(args.exact_regression_tolerance),
                exact_regression_action=effective_exact_regression_action,
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(args.eval_seed),
                eval_min_paths=max(64, M),
                sample_paths=int(args.visual_sample_paths),
                visual_sample_paths=int(args.visual_sample_paths),
                visual_seed=visual_seed_requested,
                enforce_exact_regression_guardrail=is_last_requested_pass,
                print_compact_logs=is_last_requested_pass,
                exclude_pass_ids_from_selection=excluded_pass_ids_from_selection,
                coupling_const=float(recursive_const),
                model_spec=model_spec,
            )

        rec = run_recursive_training(
            Xi_generator=model_spec.xi_generator,
            params=params,
            M=M,
            N_per_block=N,
            D=D,
            T_total=args.T_total,
            block_size=args.block_size,
            layers=layers,
            stage_plan=stage_plan,
            final_plan=final_plan,
            output_dir=os.path.join(rec_dir, "models"),
            precision_margin=0.10,
            max_refine_rounds=3,
            rollout_M=rollout_M_recursive,
            save_tf_checkpoints=True,
            training_plan_rules=training_plan_rules,
            pass1_warm_start_from_next=bool(args.pass1_warm_start_from_next),
            cross_pass_warm_start=not bool(args.disable_cross_pass_warm_start),
            n_passes=int(args.passes),
            empirical_jitter_scale=float(args.empirical_jitter_scale),
            pass1_init_mode=pass1_init_mode,
            initial_boundary_samples=initial_boundary_samples,
            initial_warm_start_blobs=initial_warm_start_blobs,
            freeze_stable_blocks_after_pass=int(args.freeze_stable_blocks_after_pass),
            freeze_loss_threshold=float(args.freeze_loss_threshold),
            freeze_neighbor_radius=int(args.freeze_neighbor_radius),
            coupling_const=float(recursive_const),
            on_pass_end=_on_recursive_pass_end,
            model_spec=model_spec,
        )

        pass_entries = sorted(rec.get("passes", []), key=lambda x: int(x["pass_id"]))
        if len(pass_entries) == 0:
            raise RuntimeError("No pass results available after recursive training")

        expected_pass_ids = sorted(int(p["pass_id"]) for p in pass_entries)
        plot_summary = pass_plot_summary_holder.get("summary", None)
        if plot_summary is None or plot_summary.get("processed_pass_ids", []) != expected_pass_ids:
            plot_summary = print_recursive_pass(
                pass_entries=pass_entries,
                blocks=rec["blocks"],
                rec_dir=rec_dir,
                params=params,
                N_per_block=N,
                D=D,
                layers=layers,
                T_total=args.T_total,
                exact_solution=exact_solution,
                selection_metric=effective_selection_metric,
                exact_regression_tolerance=float(args.exact_regression_tolerance),
                exact_regression_action=effective_exact_regression_action,
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(args.eval_seed),
                eval_min_paths=max(64, M),
                sample_paths=int(args.visual_sample_paths),
                visual_sample_paths=int(args.visual_sample_paths),
                visual_seed=visual_seed_requested,
                enforce_exact_regression_guardrail=True,
                print_compact_logs=True,
                exclude_pass_ids_from_selection=excluded_pass_ids_from_selection,
                coupling_const=float(recursive_const),
                model_spec=model_spec,
            )

        exact_summary_by_pass = plot_summary["exact_summary_by_pass"]
        exact_summary_by_pass_index = plot_summary.get("exact_summary_by_pass_index", {})

        boundary_stats = summarize_boundary_samples(rec.get("boundary_samples", []))

        passes_summary = []
        for p in pass_entries:
            pass_id = int(p["pass_id"])
            passes_summary.append(
                {
                    "pass_id": pass_id,
                    "pass_index": _pass_index(pass_id),
                    "reference_loss": float(p["reference_loss"]),
                    "logs": p.get("logs", []),
                    "models_dir": p.get("models_dir", None),
                    "pass_init_mode": p.get("pass_init_mode", "unknown"),
                    "boundary_source": p.get("boundary_source", "unknown"),
                    "is_bootstrap_pass": bool(p.get("is_bootstrap_pass", False)),
                    "active_set_summary": p.get("active_set_summary", None),
                    "frozen_blocks": p.get("active_set_summary", {}).get("frozen_blocks", []),
                    "active_blocks": p.get("active_set_summary", {}).get("active_blocks", []),
                }
            )
        pass_summary_by_index = {int(p["pass_index"]): p for p in passes_summary}

        rec_summary = {
            "blocks": rec["blocks"],
            "passes": passes_summary,
            "boundary_stats": boundary_stats,
            "models_dir": os.path.join(rec_dir, "models"),
            "evaluation_bundle_path": plot_summary["eval_bundle_path"],
            "evaluation_bundle_M": int(plot_summary["evaluation_bundle_M"]),
            "visual_seed_effective": int(plot_summary.get("visual_seed", visual_seed_effective)),
            "initialization_summary": initialization_summary,
            "active_set_freezing": {
                "enabled": bool(
                    int(args.freeze_stable_blocks_after_pass) > 0
                    and float(args.freeze_loss_threshold) > 0.0
                ),
                "freeze_stable_blocks_after_pass": int(args.freeze_stable_blocks_after_pass),
                "freeze_loss_threshold": float(args.freeze_loss_threshold),
                "freeze_neighbor_radius": int(args.freeze_neighbor_radius),
            },
            "selection_excluded_pass_ids": plot_summary.get("excluded_pass_ids_from_selection", []),
            "selection_excluded_pass_indices": plot_summary.get("excluded_pass_indices_from_selection", []),
            "selected_pass_id": int(plot_summary["selected_pass_id"]),
            "selected_pass_index": int(plot_summary["selected_pass_index"]),
            "selected_score_metric": plot_summary["selected_score_metric"],
            "selected_score": float(plot_summary["selected_score"]),
            "selected_scores_by_pass": plot_summary["selected_scores_by_pass"],
            "selected_scores_by_pass_index": plot_summary["selected_scores_by_pass_index"],
            "loss_score_metric": plot_summary["score_key"],
            "loss_pass_scores": {str(k): float(v) for k, v in plot_summary["pass_scores_loss"].items()},
            "loss_pass_scores_by_index": {
                str(k): float(v) for k, v in plot_summary["pass_scores_loss_by_index"].items()
            },
        }
        if plot_summary.get("application_summary_by_pass", {}):
            rec_summary["application_metrics"] = {
                "schema": model_spec.application_metric_schema,
                "by_pass": {
                    str(k): v for k, v in plot_summary.get("application_summary_by_pass", {}).items()
                },
                "by_pass_index": plot_summary.get("application_summary_by_pass_index", {}),
                "stability_by_pass": {
                    str(k): v for k, v in plot_summary.get("application_stability_by_pass", {}).items()
                },
                "stability_by_pass_index": plot_summary.get("application_stability_by_pass_index", {}),
                "selected_pass_summary": plot_summary.get("selected_application_summary", None),
            }
        if exact_solution is None:
            rec_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        else:
            rec_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "by_pass": {str(k): v for k, v in exact_summary_by_pass.items()},
                "by_pass_index": exact_summary_by_pass_index,
                "selected_pass_summary": exact_summary_by_pass.get(
                    int(plot_summary["selected_pass_id"]),
                    None,
                ),
            }
        if 0 in pass_summary_by_index:
            rec_summary["pass0"] = {
                "reference_loss": pass_summary_by_index[0]["reference_loss"],
                "logs": pass_summary_by_index[0]["logs"],
            }
        if 1 in pass_summary_by_index:
            rec_summary["pass1"] = {
                "reference_loss": pass_summary_by_index[1]["reference_loss"],
                "logs": pass_summary_by_index[1]["logs"],
            }
        if 2 in pass_summary_by_index:
            rec_summary["pass2"] = {
                "reference_loss": pass_summary_by_index[2]["reference_loss"],
                "logs": pass_summary_by_index[2]["logs"],
            }
        save_json(rec_summary, os.path.join(rec_dir, "results.json"))


def plot_program(argv: Optional[List[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description="Generate Pascucci paper plots from saved artifacts")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing run_config.json")
    parser.add_argument(
        "--recursive_dir",
        type=str,
        default="",
        help="Optional recursive artifact directory. Defaults to RUN_DIR/recursive.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Defaults to RECURSIVE_DIR/plots/pascucci_paper.",
    )
    parser.add_argument(
        "--source_label",
        type=str,
        default="plot_command",
        help="Reader-facing source label recorded in the plot manifest.",
    )
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    recursive_dir = (
        os.path.abspath(os.path.expanduser(args.recursive_dir))
        if str(args.recursive_dir or "").strip()
        else os.path.join(run_dir, "recursive")
    )
    out_dir = (
        os.path.abspath(os.path.expanduser(args.out_dir))
        if str(args.out_dir or "").strip()
        else os.path.join(recursive_dir, "plots", "pascucci_paper")
    )
    manifest = plot_pascucci_paper_bundle_from_artifacts(
        stitched_npz_path=os.path.join(recursive_dir, "stitched_predictions_final.npz"),
        application_npz_path=os.path.join(recursive_dir, "application_metrics_final.npz"),
        run_config_path=os.path.join(run_dir, "run_config.json"),
        out_dir=out_dir,
        source_label=str(args.source_label),
    )
    print(f"[Plot] Pascucci paper plots written to {out_dir}")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Single access point: `run` executes the solver, `plot` post-processes, `test` runs checks."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) > 0 and argv[0] == "test":
        return run_tests(argv[1:])
    if len(argv) > 0 and argv[0] == "plot":
        plot_program(argv[1:])
        return 0
    if len(argv) > 0 and argv[0] == "run":
        argv = argv[1:]
    run_program(argv)
    return 0
