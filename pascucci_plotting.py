"""Artifact-driven Pascucci paper plot helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .io_utils import _as_blob_dict, save_json
from .plotting import _PLOTTING_AVAILABLE, plt


PAPER_PLOT_SCHEMA = "pascucci_paper_plots_v1"
PAPER_PLOT_MANIFEST = "pascucci_paper_plots_manifest.json"
APPLICATION_METRIC_SCHEMA = "pascucci_application_metrics_v2"


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


def _plot_state_with_ou(
    *,
    time: np.ndarray,
    values: np.ndarray,
    ou_params: Dict[str, Any],
    ylabel: str,
    title: str,
    path: str,
) -> None:
    empirical = _band(values)
    ou = _ou_mean_band(time, ou_params)
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


def _plot_accumulated_cost(
    *,
    time: np.ndarray,
    controlled: np.ndarray,
    uncontrolled: np.ndarray,
    path: str,
) -> None:
    controlled_band = _band(np.asarray(controlled, dtype=np.float32)[:, :, 0])
    uncontrolled_band = _band(np.asarray(uncontrolled, dtype=np.float32)[:, :, 0])
    plt.figure(figsize=(10, 5))
    plt.fill_between(time, controlled_band["q05"], controlled_band["q95"], color="tab:blue", alpha=0.18)
    plt.plot(time, controlled_band["mean"], color="tab:blue", linewidth=1.8, label="controlled mean")
    plt.fill_between(time, uncontrolled_band["q05"], uncontrolled_band["q95"], color="tab:red", alpha=0.14)
    plt.plot(time, uncontrolled_band["mean"], color="tab:red", linewidth=1.6, linestyle="--", label="uncontrolled mean")
    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    plt.title("Pascucci accumulated running cost")
    plt.xlabel("Time")
    plt.ylabel("Running cost cumulative")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_alpha(*, time: np.ndarray, controlled_alpha: np.ndarray, uncontrolled_alpha: np.ndarray, path: str) -> None:
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


def _plot_state_bands(*, time: np.ndarray, X: np.ndarray, path: str) -> None:
    components = ((0, "S", "tab:blue"), (2, "V", "tab:green"), (3, "Q", "tab:purple"))
    plt.figure(figsize=(10, 5))
    for idx, label, color in components:
        stats = _band(X[:, :, idx])
        plt.fill_between(time, stats["q05"], stats["q95"], color=color, alpha=0.12)
        plt.plot(time, stats["q50"], color=color, linewidth=1.6, label=f"{label} median")
    plt.title("Pascucci state percentile bands")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_controlled_uncontrolled(
    *,
    controlled_total: np.ndarray,
    uncontrolled_total: np.ndarray,
    path: str,
) -> None:
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
    plt.xticks(x, ["controlled", "uncontrolled"])
    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    plt.title("Pascucci controlled vs uncontrolled total cost distribution")
    plt.ylabel("J total median with q05-q95; diamonds show mean")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _story_entry(filename: str, title: str, inputs: Iterable[str]) -> Dict[str, Any]:
    return {
        "filename": filename,
        "path": filename,
        "path_relative_to": "manifest_dir",
        "title": title,
        "inputs": list(inputs),
    }


def _story_path(entry: Dict[str, Any], out_dir: str) -> str:
    return os.path.join(out_dir, str(entry["path"]))


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
    params_S = dict(params.get("params_S", {}))
    params_H = dict(params.get("params_H", {}))
    if not params_S or not params_H:
        raise ValueError("Pascucci paper plots require params_S and params_H in run config params")

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
    steps = time.shape[0] - 1
    for name, arr in (
        ("controlled_alpha", controlled_alpha),
        ("uncontrolled_alpha", uncontrolled_alpha),
        ("controlled_cost_J_running_cumulative", controlled_cumulative),
        ("uncontrolled_cost_J_running_cumulative", uncontrolled_cumulative),
    ):
        if arr.shape[0] != X.shape[0] or arr.shape[1] != steps:
            raise ValueError(f"{name} must have shape (M, n_steps, 1), got {arr.shape}")
    if controlled_total.shape[0] != X.shape[0] or uncontrolled_total.shape[0] != X.shape[0]:
        raise ValueError("controlled/uncontrolled total costs must match stitched path count")

    os.makedirs(out_dir, exist_ok=True)
    plot_specs = {
        "#35": _story_entry(
            "pascucci_paper_35_S_ou_band.png",
            "S with OU envelope",
            ("stitched.t", "stitched.X[:, :, S]", "run_config.params.params_S"),
        ),
        "#36": _story_entry(
            "pascucci_paper_36_H_ou_band.png",
            "H with OU envelope",
            ("stitched.t", "stitched.X[:, :, H]", "run_config.params.params_H"),
        ),
        "#37": _story_entry(
            "pascucci_paper_37_accumulated_cost.png",
            "Accumulated running cost",
            (
                "application_metrics.controlled_cost_J_running_cumulative",
                "application_metrics.uncontrolled_cost_J_running_cumulative",
            ),
        ),
        "#38": _story_entry(
            "pascucci_paper_38_alpha.png",
            "Control alpha",
            ("application_metrics.controlled_alpha", "application_metrics.uncontrolled_alpha"),
        ),
        "#39": _story_entry(
            "pascucci_paper_39_state_bands_S_V_Q.png",
            "State bands S,V,Q",
            ("stitched.t", "stitched.X[:, :, S,V,Q]"),
        ),
        "#40": _story_entry(
            "pascucci_paper_40_controlled_uncontrolled.png",
            "Controlled vs uncontrolled total cost",
            ("application_metrics.controlled_cost_J_total", "application_metrics.uncontrolled_cost_J_total"),
        ),
    }

    _plot_state_with_ou(
        time=time,
        values=X[:, :, 0],
        ou_params=params_S,
        ylabel="S",
        title="Pascucci S with calibrated OU envelope",
        path=_story_path(plot_specs["#35"], out_dir),
    )
    _plot_state_with_ou(
        time=time,
        values=X[:, :, 1],
        ou_params=params_H,
        ylabel="H",
        title="Pascucci H with calibrated OU envelope",
        path=_story_path(plot_specs["#36"], out_dir),
    )
    _plot_accumulated_cost(
        time=time[1:],
        controlled=controlled_cumulative,
        uncontrolled=uncontrolled_cumulative,
        path=_story_path(plot_specs["#37"], out_dir),
    )
    _plot_alpha(
        time=time[:-1],
        controlled_alpha=controlled_alpha,
        uncontrolled_alpha=uncontrolled_alpha,
        path=_story_path(plot_specs["#38"], out_dir),
    )
    _plot_state_bands(time=time, X=X, path=_story_path(plot_specs["#39"], out_dir))
    _plot_controlled_uncontrolled(
        controlled_total=controlled_total,
        uncontrolled_total=uncontrolled_total,
        path=_story_path(plot_specs["#40"], out_dir),
    )

    source = dict(source_metadata or {})
    manifest = {
        "schema": PAPER_PLOT_SCHEMA,
        "model_name": str(source.get("model_name", "pascucci")),
        "plots": plot_specs,
        "plot_count": int(len(plot_specs)),
        "source": source,
        "horizon": {
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "n_time_points": int(time.shape[0]),
            "n_steps": int(steps),
            "sample_paths": int(X.shape[0]),
        },
        "state_column_map": {"S": 0, "H": 1, "V": 2, "Q": 3},
        "cost_trace_source": "cost_J_running_cumulative",
        "controlled_uncontrolled_available": True,
        "plot_path_policy": "relative_to_manifest_dir",
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
