"""Sampling, boundary, and rollout input utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def build_stitched_rollout_inputs(
    blocks: List[Dict[str, float]],
    M: int,
    N_per_block: int,
    D: int,
    seed: int = 1234,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    rollout_inputs = []
    for block in blocks:
        dt = float(block["T_block"]) / float(N_per_block)
        Dt = np.zeros((M, N_per_block + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N_per_block + 1, D), dtype=np.float32)
        Dt[:, 1:, :] = dt

        if M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * rng.normal(size=(half_M, N_per_block, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * rng.normal(size=(N_per_block, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * rng.normal(size=(M, N_per_block, D))

        t_abs = float(block["t_start"]) + np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        rollout_inputs.append((t_abs.astype(np.float32), W.astype(np.float32)))
    return rollout_inputs

def make_deterministic_xi_default(M: int, D: int, seed: int = 1234) -> np.ndarray:
    if int(D) != 4:
        raise ValueError(f"make_deterministic_xi_default currently supports D=4, got D={int(D)}")
    rng = np.random.RandomState(int(seed))
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    Xi[:, 0] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 1] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 2] = rng.normal(0.0, 1.0, int(M))
    Xi[:, 3] = rng.uniform(3.0, 7.0, int(M))
    return Xi.astype(np.float32)

def save_evaluation_bundle(
    path: str,
    Xi_initial: np.ndarray,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    blocks: List[Dict[str, float]],
) -> None:
    dir_name = os.path.dirname(path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    t_stack = np.stack([pair[0] for pair in rollout_inputs], axis=0).astype(np.float32)
    w_stack = np.stack([pair[1] for pair in rollout_inputs], axis=0).astype(np.float32)
    t_start = np.array([float(b["t_start"]) for b in blocks], dtype=np.float32)
    t_end = np.array([float(b["t_end"]) for b in blocks], dtype=np.float32)
    np.savez(
        path,
        Xi_initial=np.asarray(Xi_initial, dtype=np.float32),
        t_bundle=t_stack,
        W_bundle=w_stack,
        block_t_start=t_start,
        block_t_end=t_end,
    )

def load_evaluation_bundle(
    path: str,
    n_blocks_expected: int,
    N_per_block_expected: int,
    D_expected: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as data:
        Xi = np.asarray(data["Xi_initial"], dtype=np.float32)
        t_bundle = np.asarray(data["t_bundle"], dtype=np.float32)
        W_bundle = np.asarray(data["W_bundle"], dtype=np.float32)

    if Xi.ndim != 2 or Xi.shape[1] != int(D_expected):
        raise ValueError(
            f"Invalid Xi_initial shape in evaluation bundle: {Xi.shape}, expected [M, {int(D_expected)}]"
        )
    if t_bundle.ndim != 4 or W_bundle.ndim != 4:
        raise ValueError(
            f"Invalid rollout bundle rank: t={t_bundle.shape}, W={W_bundle.shape}; expected rank-4"
        )
    if t_bundle.shape[0] != int(n_blocks_expected) or W_bundle.shape[0] != int(n_blocks_expected):
        raise ValueError(
            f"Evaluation bundle blocks mismatch: got {t_bundle.shape[0]}, expected {int(n_blocks_expected)}"
        )
    if t_bundle.shape[2] != int(N_per_block_expected) + 1:
        raise ValueError(
            "Evaluation bundle N_per_block mismatch: "
            f"got {t_bundle.shape[2]-1}, expected {int(N_per_block_expected)}"
        )
    if W_bundle.shape[3] != int(D_expected):
        raise ValueError(
            f"Evaluation bundle D mismatch in W: got {W_bundle.shape[3]}, expected {int(D_expected)}"
        )
    if t_bundle.shape[1] != Xi.shape[0] or W_bundle.shape[1] != Xi.shape[0]:
        raise ValueError(
            "Evaluation bundle M mismatch between Xi and rollout tensors: "
            f"Xi={Xi.shape[0]}, t_bundle={t_bundle.shape[1]}, W_bundle={W_bundle.shape[1]}"
        )

    rollout_inputs = []
    for i in range(int(n_blocks_expected)):
        rollout_inputs.append((t_bundle[i], W_bundle[i]))
    return Xi, rollout_inputs

def Xi_generator_default(M, D):
    assert D == 4
    Xi = np.zeros((M, 4), dtype=np.float32)
    Xi[:, 0] = np.random.normal(1.0, 1.0, M)
    Xi[:, 1] = np.random.normal(1.0, 1.0, M)
    Xi[:, 2] = np.random.normal(0.0, 1.0, M)
    Xi[:, 3] = np.random.uniform(3.0, 7.0, M)
    return Xi.astype(np.float32)

def make_empirical_generator(samples: np.ndarray, jitter_scale: float = 0.0):
    samples = np.asarray(samples, dtype=np.float32)
    std = np.std(samples, axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-3)

    def _gen(M, D):
        idx = np.random.randint(0, samples.shape[0], size=M)
        Xi = samples[idx].copy()
        if jitter_scale > 0.0:
            Xi += jitter_scale * std * np.random.normal(size=Xi.shape).astype(np.float32)
        return Xi.astype(np.float32)

    return _gen

def estimate_generator_stats(generator_fn, D, n_samples=4096):
    x = generator_fn(n_samples, D).astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(np.std(x, axis=0, keepdims=True), 1.0e-3).astype(np.float32)
    return mean, std

def build_blocks(T_total: float, block_size: float) -> List[Dict[str, float]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    n_blocks = int(np.ceil(T_total / block_size))
    edges = [0.0]
    for i in range(1, n_blocks + 1):
        edges.append(min(float(i * block_size), float(T_total)))
    blocks = []
    for i in range(n_blocks):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        blocks.append({"idx": i, "t_start": t0, "t_end": t1, "T_block": (t1 - t0)})
    return blocks

def validate_boundary_samples(
    boundary_samples: Optional[List[np.ndarray]],
    blocks: List[Dict[str, float]],
    D: int,
    label: str = "boundary_samples",
) -> None:
    if boundary_samples is None:
        return
    if len(boundary_samples) != len(blocks) + 1:
        raise ValueError(
            f"{label} must contain len(blocks)+1 arrays, got {len(boundary_samples)} vs expected {len(blocks) + 1}"
        )
    for idx, arr in enumerate(boundary_samples):
        arr_np = np.asarray(arr)
        if arr_np.ndim != 2 or arr_np.shape[1] != int(D):
            raise ValueError(
                f"{label}[{idx}] must have shape [M, {int(D)}], got {list(arr_np.shape)}"
            )

def summarize_boundary_samples(boundary_samples: List[np.ndarray]) -> List[Dict[str, Any]]:
    summary = []
    for i, arr in enumerate(boundary_samples or []):
        arr_np = np.asarray(arr, dtype=np.float32)
        summary.append(
            {
                "boundary_idx": int(i),
                "n_samples": int(arr_np.shape[0]),
                "mean": np.mean(arr_np, axis=0),
                "std": np.std(arr_np, axis=0),
                "min": np.min(arr_np, axis=0),
                "max": np.max(arr_np, axis=0),
            }
        )
    return summary
