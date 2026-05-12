"""Plotting helpers for standard and recursive diagnostics."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .io_utils import save_rows_csv
from .naming import _pass_label, _z_component_labels

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    _PLOTTING_AVAILABLE = False

def plot_stage_logs(stage_logs: List[Dict], out_prefix: str, title: str) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_stage_logs")
        return
    if stage_logs is None or len(stage_logs) == 0:
        return

    x = np.arange(len(stage_logs))
    loss = np.array([row.get("eval_mean_loss", np.nan) for row in stage_logs], dtype=np.float64)
    y0 = np.array([row.get("eval_mean_y0", np.nan) for row in stage_logs], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.plot(x, loss, "b-o", markersize=3, linewidth=1.2, label="eval mean loss")
    plt.yscale("log")
    plt.title(f"{title} - Eval Mean Loss")
    plt.xlabel("Stage index")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_eval_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y0, "g-o", markersize=3, linewidth=1.2, label="eval mean y0")
    plt.title(f"{title} - Eval Mean Y0")
    plt.xlabel("Stage index")
    plt.ylabel("Mean Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_eval_y0.png", dpi=160)
    plt.close()

def plot_recursive_pass_logs_multi(pass_logs_by_pass: Dict[int, List[Dict]], out_dir: str) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_pass_logs_multi")
        return
    if pass_logs_by_pass is None or len(pass_logs_by_pass) == 0:
        return

    normalized = {}
    for pass_id, rows in pass_logs_by_pass.items():
        rows_sorted = sorted(rows or [], key=lambda r: r["block"])
        if len(rows_sorted) > 0:
            normalized[int(pass_id)] = rows_sorted
    if len(normalized) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    pass_ids = sorted(normalized.keys())
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(pass_ids), 2)))
    use_per_sample_loss = all(
        len(rows) > 0 and ("eval_mean_loss_per_sample" in rows[0]) for rows in normalized.values()
    )
    loss_key = "eval_mean_loss_per_sample" if use_per_sample_loss else "eval_mean_loss"

    plt.figure(figsize=(10, 6))
    for i, pass_id in enumerate(pass_ids):
        rows = normalized[pass_id]
        b = np.array([r["block"] for r in rows], dtype=np.int32)
        l = np.array([r[loss_key] for r in rows], dtype=np.float64)
        plt.plot(
            b,
            l,
            marker="o",
            linewidth=1.5,
            color=colors[i],
            label=f"{_pass_label(pass_id)} loss",
        )
    plt.yscale("log")
    if use_per_sample_loss:
        plt.title("Recursive blocks - Eval Mean Loss per Sample")
        plt.ylabel("Loss / M (log scale)")
    else:
        plt.title("Recursive blocks - Eval Mean Loss")
        plt.ylabel("Loss (log scale)")
    plt.xlabel("Block index")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, pass_id in enumerate(pass_ids):
        rows = normalized[pass_id]
        b = np.array([r["block"] for r in rows], dtype=np.int32)
        y = np.array([r["eval_mean_y0"] for r in rows], dtype=np.float64)
        plt.plot(
            b,
            y,
            marker="o",
            linewidth=1.5,
            color=colors[i],
            label=f"{_pass_label(pass_id)} y0",
        )
    plt.title("Recursive blocks - Eval Mean Y0")
    plt.xlabel("Block index")
    plt.ylabel("Mean Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_y0.png"), dpi=160)
    plt.close()

def plot_recursive_pass_logs(pass1_logs: List[Dict], pass2_logs: List[Dict], out_dir: str) -> None:
    plot_recursive_pass_logs_multi(
        pass_logs_by_pass={
            1: pass1_logs or [],
            2: pass2_logs or [],
        },
        out_dir=out_dir,
    )

def score_pass_logs(
    rows: List[Dict],
    loss_key: str = "eval_mean_loss_per_sample",
    worst_block_weight: float = 0.35,
) -> float:
    losses = np.array([float(r.get(loss_key, np.nan)) for r in (rows or [])], dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return float("inf")
    return float(np.mean(losses) + worst_block_weight * np.max(losses))

def plot_recursive_exact_comparison(
    stitched: Dict[str, np.ndarray],
    Y_exact: np.ndarray,
    Z_exact: np.ndarray,
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 5,
    file_suffix: str = "",
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_exact_comparison")
        return
    if stitched is None:
        return
    if "t" not in stitched or "Y" not in stitched or "Z" not in stitched:
        return

    t_all = stitched["t"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]
    if t_all.size == 0 or Y_pred.size == 0 or Z_pred.size == 0:
        return
    if Y_exact.shape != Y_pred.shape or Z_exact.shape != Z_pred.shape:
        return

    os.makedirs(out_dir, exist_ok=True)
    n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))
    z_labels = _z_component_labels(int(Z_pred.shape[2]))
    z_colors = ["b", "g", "r", "m", "c", "y", "k"]

    plt.figure(figsize=(12, 6))
    for i in range(n_paths):
        alpha = 0.95 if i == 0 else 0.28
        width = 1.8 if i == 0 else 0.9
        pred_label = "Y pred" if i == 0 else None
        exact_label = "Y exact" if i == 0 else None
        plt.plot(
            t_all[i, :, 0],
            Y_pred[i, :, 0],
            color="tab:blue",
            alpha=alpha,
            linewidth=width,
            label=pred_label,
        )
        plt.plot(
            t_all[i, :, 0],
            Y_exact[i, :, 0],
            color="tab:red",
            alpha=alpha,
            linewidth=width,
            linestyle="--",
            label=exact_label,
        )
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.title("Recursive stitched prediction - Y predicted vs exact")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_Y_exact{file_suffix}.png"), dpi=160)
    plt.close()

    abs_err_Z_full = np.abs(Z_pred - Z_exact)
    valid_mask = np.abs(Z_exact) > 1.0e-8
    rel_err_Z = np.zeros_like(abs_err_Z_full, dtype=np.float32)
    np.divide(
        abs_err_Z_full,
        np.abs(Z_exact) + 1.0e-8,
        out=rel_err_Z,
        where=valid_mask,
    )
    valid_count_t = np.maximum(np.sum(valid_mask, axis=0), 1.0).astype(np.float32)
    mean_rel_err_Z = np.sum(rel_err_Z, axis=0) / valid_count_t

    plt.figure(figsize=(12, 6))
    for d in range(Z_pred.shape[2]):
        color = z_colors[d % len(z_colors)]
        label = z_labels[d] if d < len(z_labels) else f"Z[{d}]"
        curve = np.maximum(mean_rel_err_Z[:, d], 1.0e-14)
        plt.plot(t_all[0, :, 0], curve, color=color, linewidth=1.5, label=f"Mean rel err {label}")
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.yscale("log")
    plt.title("Recursive stitched prediction - Mean relative error on Z vs exact")
    plt.xlabel("Time")
    plt.ylabel("Relative error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_Z_rel_error{file_suffix}.png"), dpi=160)
    plt.close()

    abs_err_Z = np.mean(abs_err_Z_full, axis=0)
    abs_err_Y = np.mean(np.abs(Y_pred - Y_exact), axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(
        t_all[0, :, 0],
        np.maximum(abs_err_Y[:, 0], 1.0e-14),
        color="tab:orange",
        linewidth=1.8,
        label="Mean abs err Y",
    )
    for d in range(Z_pred.shape[2]):
        color = z_colors[d % len(z_colors)]
        label = z_labels[d] if d < len(z_labels) else f"Z[{d}]"
        plt.plot(
            t_all[0, :, 0],
            np.maximum(abs_err_Z[:, d], 1.0e-14),
            color=color,
            linewidth=1.4,
            label=f"Mean abs err {label}",
        )
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.yscale("log")
    plt.title("Recursive stitched prediction - Mean absolute error on Y and Z vs exact")
    plt.xlabel("Time")
    plt.ylabel("Absolute error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_abs_error{file_suffix}.png"), dpi=160)
    plt.close()

def plot_recursive_stitched_predictions(
    stitched: Dict[str, np.ndarray],
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 5,
    file_suffix: str = "",
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_stitched_predictions")
        return
    if stitched is None:
        return
    if "t" not in stitched or "X" not in stitched or "Y" not in stitched:
        return

    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_all = stitched["Y"]
    if t_all.size == 0 or X_all.size == 0 or Y_all.size == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))

    component_labels = ["S", "H", "V", "X"]
    component_colors = ["b", "r", "y", "g"]

    plt.figure(figsize=(12, 6))
    for d in range(X_all.shape[2]):
        label = component_labels[d] if d < len(component_labels) else f"X[{d}]"
        color = component_colors[d % len(component_colors)]
        plt.plot(t_all[0, :, 0], X_all[0, :, d], color=color, linewidth=1.5, label=label)
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    plt.title("Recursive stitched prediction - State path (single continuous horizon)")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"recursive_stitched_state_path{file_suffix}.png"),
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    for i in range(n_paths):
        alpha = 0.95 if i == 0 else 0.35
        width = 1.8 if i == 0 else 1.0
        label = "Y pred (path 0)" if i == 0 else None
        plt.plot(t_all[i, :, 0], Y_all[i, :, 0], color="tab:blue", alpha=alpha, linewidth=width, label=label)
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    plt.title("Recursive stitched prediction - Y over full horizon")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    if n_paths > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"recursive_stitched_Y_pred{file_suffix}.png"),
        dpi=160,
    )
    plt.close()

def plot_recursive_stitched_y_convergence(
    stitched_by_pass: Dict[int, Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 8,
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_stitched_y_convergence")
        return
    if stitched_by_pass is None or len(stitched_by_pass) == 0:
        return

    pass_ids = sorted(stitched_by_pass.keys())
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(pass_ids)))

    plt.figure(figsize=(12, 6))
    for i, pass_id in enumerate(pass_ids):
        stitched = stitched_by_pass[pass_id]
        t_all = stitched.get("t", None)
        Y_all = stitched.get("Y", None)
        if t_all is None or Y_all is None or t_all.size == 0 or Y_all.size == 0:
            continue
        n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))
        t_flat = t_all[:n_paths, :, 0].reshape(-1)
        y_flat = Y_all[:n_paths, :, 0].reshape(-1)
        plt.scatter(t_flat, y_flat, s=2, color=colors[i], alpha=0.06)
        y_mean = np.mean(Y_all[:n_paths, :, 0], axis=0)
        plt.plot(
            t_all[0, :, 0],
            y_mean,
            color=colors[i],
            linewidth=1.8,
            label=f"{_pass_label(pass_id)} mean Y",
        )

    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)

    plt.title("Recursive stitched prediction - Y convergence across passes")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_stitched_Y_convergence.png"), dpi=160)
    plt.close()
