"""Sampling, boundary, and rollout input utilities."""

from __future__ import annotations

import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_EVALUATION_BUNDLE_SCHEMA = "evaluation_bundle_v2"
_VALID_BUNDLE_KINDS = {"evaluation", "boundary_rollout", "mc_confirmation"}

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

def make_deterministic_xi_pascucci_paper(M: int, D: int, seed: int = 1234) -> np.ndarray:
    if int(D) != 4:
        raise ValueError(f"make_deterministic_xi_pascucci_paper currently supports D=4, got D={int(D)}")
    rng = np.random.RandomState(int(seed))
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    Xi[:, 0] = rng.normal(-2.3, 0.2, int(M))
    Xi[:, 1] = rng.normal(0.4, 0.5, int(M))
    Xi[:, 2] = rng.normal(0.0, 1.0, int(M))
    Xi[:, 3] = rng.uniform(1.0, 9.0, int(M))
    return Xi.astype(np.float32)

def save_evaluation_bundle(
    path: str,
    Xi_initial: np.ndarray,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    blocks: List[Dict[str, float]],
    bundle_kind: str = "evaluation",
) -> None:
    bundle_kind = str(bundle_kind)
    if bundle_kind not in _VALID_BUNDLE_KINDS:
        raise ValueError(
            f"Unsupported evaluation bundle kind: {bundle_kind!r}; "
            f"expected one of {sorted(_VALID_BUNDLE_KINDS)}"
        )
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
        bundle_schema=np.asarray(_EVALUATION_BUNDLE_SCHEMA),
        bundle_kind=np.asarray(bundle_kind),
    )

def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_evaluation_bundle(
    path: str,
    n_blocks_expected: int,
    N_per_block_expected: int,
    D_expected: int,
    blocks_expected: Optional[List[Dict[str, float]]] = None,
    T_total_expected: Optional[float] = None,
    bundle_kind_expected: Optional[str] = None,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        Xi = np.asarray(data["Xi_initial"], dtype=np.float32)
        t_bundle = np.asarray(data["t_bundle"], dtype=np.float32)
        W_bundle = np.asarray(data["W_bundle"], dtype=np.float32)
        saved_t_start = (
            np.asarray(data["block_t_start"], dtype=np.float32)
            if "block_t_start" in files
            else None
        )
        saved_t_end = (
            np.asarray(data["block_t_end"], dtype=np.float32)
            if "block_t_end" in files
            else None
        )
        saved_kind = str(np.asarray(data["bundle_kind"]).item()) if "bundle_kind" in files else None
        saved_schema = (
            str(np.asarray(data["bundle_schema"]).item())
            if "bundle_schema" in files
            else None
        )

    if bundle_kind_expected is not None:
        expected_kind = str(bundle_kind_expected)
        if expected_kind not in _VALID_BUNDLE_KINDS:
            raise ValueError(
                f"Unsupported expected evaluation bundle kind: {expected_kind!r}; "
                f"expected one of {sorted(_VALID_BUNDLE_KINDS)}"
            )
        if saved_kind is None:
            if expected_kind == "evaluation":
                saved_kind = "evaluation"
            else:
                raise ValueError("Evaluation bundle metadata mismatch: missing bundle_kind")
        if saved_kind != expected_kind:
            raise ValueError(
                "Evaluation bundle metadata mismatch: "
                f"bundle_kind={saved_kind!r}, expected {expected_kind!r}"
            )
    if saved_schema is not None and saved_schema != _EVALUATION_BUNDLE_SCHEMA:
        raise ValueError(
            "Evaluation bundle metadata mismatch: "
            f"bundle_schema={saved_schema!r}, expected {_EVALUATION_BUNDLE_SCHEMA!r}"
        )

    if Xi.ndim != 2 or Xi.shape[1] != int(D_expected):
        raise ValueError(
            f"Invalid Xi_initial shape in evaluation bundle: {Xi.shape}, expected [M, {int(D_expected)}]"
        )
    if t_bundle.ndim != 4 or W_bundle.ndim != 4:
        raise ValueError(
            f"Invalid rollout bundle rank: t={t_bundle.shape}, W={W_bundle.shape}; expected rank-4"
        )
    if t_bundle.shape[3] != 1:
        raise ValueError(
            f"Evaluation bundle time tensor must have final dimension 1, got {t_bundle.shape}"
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
    if W_bundle.shape[2] != int(N_per_block_expected) + 1:
        raise ValueError(
            "Evaluation bundle N_per_block mismatch in W: "
            f"got {W_bundle.shape[2]-1}, expected {int(N_per_block_expected)}"
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
    if not np.isfinite(Xi).all() or not np.isfinite(t_bundle).all() or not np.isfinite(W_bundle).all():
        raise ValueError("Evaluation bundle contains non-finite values")

    if blocks_expected is not None:
        expected_start = np.asarray(
            [float(b["t_start"]) for b in blocks_expected],
            dtype=np.float32,
        )
        expected_end = np.asarray(
            [float(b["t_end"]) for b in blocks_expected],
            dtype=np.float32,
        )
        if saved_t_start is None or saved_t_end is None:
            raise ValueError("Evaluation bundle metadata mismatch: missing block_t_start/block_t_end")
        if saved_t_start.shape != expected_start.shape or not np.allclose(
            saved_t_start, expected_start, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("Evaluation bundle metadata mismatch: block_t_start differs")
        if saved_t_end.shape != expected_end.shape or not np.allclose(
            saved_t_end, expected_end, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("Evaluation bundle metadata mismatch: block_t_end differs")

    if T_total_expected is not None and saved_t_end is not None and saved_t_end.size > 0:
        if not np.isclose(float(saved_t_end[-1]), float(T_total_expected), rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "Evaluation bundle metadata mismatch: "
                f"T_total differs, got {float(saved_t_end[-1])}, expected {float(T_total_expected)}"
            )

    rollout_inputs = []
    for i in range(int(n_blocks_expected)):
        t_i = t_bundle[i]
        W_i = W_bundle[i]
        if not np.all(np.diff(t_i[:, :, 0], axis=1) > 0.0):
            raise ValueError(f"Evaluation bundle block {i} time grid must be strictly increasing")
        if not np.allclose(W_i[:, 0, :], 0.0, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"Evaluation bundle block {i} W_start must be zero")
        expected_start_value = None
        expected_end_value = None
        if blocks_expected is not None:
            expected_start_value = float(blocks_expected[i]["t_start"])
            expected_end_value = float(blocks_expected[i]["t_end"])
        elif saved_t_start is not None and saved_t_end is not None:
            expected_start_value = float(saved_t_start[i])
            expected_end_value = float(saved_t_end[i])
        if expected_start_value is not None:
            if not np.allclose(t_i[:, 0, 0], expected_start_value, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError(f"Evaluation bundle block {i} t_start does not match metadata")
            if not np.allclose(t_i[:, -1, 0], expected_end_value, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError(f"Evaluation bundle block {i} t_end does not match metadata")
        rollout_inputs.append((t_i, W_i))
    return Xi, rollout_inputs


def Xi_generator_default(M, D):
    assert D == 4
    Xi = np.zeros((M, 4), dtype=np.float32)
    Xi[:, 0] = np.random.normal(1.0, 1.0, M)
    Xi[:, 1] = np.random.normal(1.0, 1.0, M)
    Xi[:, 2] = np.random.normal(0.0, 1.0, M)
    Xi[:, 3] = np.random.uniform(3.0, 7.0, M)
    return Xi.astype(np.float32)

def Xi_generator_pascucci_paper(M, D):
    assert D == 4
    Xi = np.zeros((M, 4), dtype=np.float32)
    Xi[:, 0] = np.random.normal(-2.3, 0.2, M)
    Xi[:, 1] = np.random.normal(0.4, 0.5, M)
    Xi[:, 2] = np.random.normal(0.0, 1.0, M)
    Xi[:, 3] = np.random.uniform(1.0, 9.0, M)
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
