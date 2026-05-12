"""Training plan parsing and schedule resolution."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

def _normalize_training_plan_rule(raw: Dict[str, Any], source_row: int = 0) -> Optional[Dict]:
    if raw is None:
        return None

    pass_scope = str(raw.get("pass_scope", "")).strip()
    block_scope = str(raw.get("block_scope", "")).strip().lower()
    phase = str(raw.get("phase", "")).strip().lower()
    if pass_scope == "" or block_scope == "" or phase == "":
        return None
    if phase not in ("stage", "final", "refine"):
        raise ValueError(
            f"Invalid phase '{phase}' in training plan rule (allowed: stage, final, refine)"
        )

    enabled_raw = str(raw.get("enabled", "1")).strip().lower()
    enabled = enabled_raw not in ("0", "false", "no", "off", "")
    if not enabled:
        return None

    order_raw = str(raw.get("order", "0")).strip()
    order = int(order_raw) if order_raw != "" else 0
    n_iter = int(str(raw.get("n_iter", "")).strip())
    lr = float(str(raw.get("lr", "")).strip())
    if n_iter <= 0:
        raise ValueError(f"n_iter must be > 0 in training plan rule (source_row={source_row})")
    if lr <= 0:
        raise ValueError(f"lr must be > 0 in training plan rule (source_row={source_row})")

    return {
        "pass_scope": pass_scope,
        "block_scope": block_scope,
        "phase": phase,
        "order": int(order),
        "n_iter": int(n_iter),
        "lr": float(lr),
        "source_row": int(source_row),
    }

def load_training_plan_csv(csv_path: Optional[str]) -> List[Dict]:
    if csv_path is None or str(csv_path).strip() == "":
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training plan CSV not found: {csv_path}")

    rules = []
    required = {"pass_scope", "block_scope", "phase", "n_iter", "lr"}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Training plan CSV is empty or has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Training plan CSV missing required columns: {sorted(missing)}")

        for i_row, row in enumerate(reader, start=2):
            normalized = _normalize_training_plan_rule(row, source_row=i_row)
            if normalized is not None:
                rules.append(normalized)

    return rules

def _pass_scope_priority(pass_scope: str, pass_id: int) -> int:
    ps = str(pass_scope).strip().lower()
    if ps in ("*", "all"):
        return 1

    if ps.endswith("+"):
        base = ps[:-1].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.startswith(">="):
        base = ps[2:].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.isdigit() and pass_id == int(ps):
        return 3

    return -1

def _block_scope_priority(block_scope: str, block_idx: int, n_blocks: int) -> int:
    bs = str(block_scope).strip().lower()
    is_terminal = block_idx == (n_blocks - 1)

    if bs in ("*", "all"):
        return 1
    if bs == "terminal" and is_terminal:
        return 2
    if bs == "other" and (not is_terminal):
        return 2

    if bs.startswith("block:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3
    if bs.startswith("idx:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3

    if bs.isdigit() and block_idx == int(bs):
        return 3

    return -1

def _resolve_phase_plan(
    rules: List[Dict],
    phase: str,
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_plan: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    matched = []
    for r in rules:
        if r["phase"] != phase:
            continue
        p_prio = _pass_scope_priority(r["pass_scope"], pass_id)
        if p_prio < 0:
            continue
        b_prio = _block_scope_priority(r["block_scope"], block_idx, n_blocks)
        if b_prio < 0:
            continue
        matched.append((p_prio, b_prio, r["order"], r))

    if len(matched) == 0:
        return list(default_plan)

    best_scope = max((x[0], x[1]) for x in matched)
    selected = [x for x in matched if (x[0], x[1]) == best_scope]
    selected.sort(key=lambda x: x[2])
    return [(int(x[3]["n_iter"]), float(x[3]["lr"])) for x in selected]

def resolve_training_plan_for_block(
    rules: List[Dict],
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_stage: List[Tuple[int, float]],
    default_final: List[Tuple[int, float]],
    default_refine: List[Tuple[int, float]],
) -> Dict[str, List[Tuple[int, float]]]:
    if rules is None:
        rules = []
    return {
        "stage_plan": _resolve_phase_plan(
            rules=rules,
            phase="stage",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_stage,
        ),
        "final_plan": _resolve_phase_plan(
            rules=rules,
            phase="final",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_final,
        ),
        "refine_plan": _resolve_phase_plan(
            rules=rules,
            phase="refine",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_refine,
        ),
    }

def scale_schedule(
    plan: List[Tuple[int, float]],
    iter_scale: float,
    min_iter: int = 50,
) -> List[Tuple[int, float]]:
    scaled = []
    for n_iter, lr in plan:
        scaled_iters = max(int(min_iter), int(round(float(n_iter) * float(iter_scale))))
        scaled.append((scaled_iters, float(lr)))
    return scaled

def scale_training_plan_rules(
    rules: List[Dict[str, Any]],
    iter_scale: float,
    min_iter: int = 50,
) -> List[Dict[str, Any]]:
    scaled_rules = []
    for rule in rules or []:
        scaled_rule = dict(rule)
        scaled_rule["n_iter"] = max(
            int(min_iter),
            int(round(float(rule["n_iter"]) * float(iter_scale))),
        )
        scaled_rules.append(scaled_rule)
    return scaled_rules

def parse_float_sequence_arg(raw_value: Union[str, List[float], Tuple[float, ...], None], arg_name: str) -> List[float]:
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple)):
        raw_items = list(raw_value)
    else:
        text = str(raw_value).strip()
        if text == "":
            return []
        raw_items = [item.strip() for item in text.split(",")]

    values = []
    for idx, item in enumerate(raw_items):
        item_str = str(item).strip()
        if item_str == "":
            raise ValueError(f"{arg_name} contains an empty value at position {idx}")
        value = float(item_str)
        if not np.isfinite(value):
            raise ValueError(f"{arg_name} contains a non-finite value at position {idx}: {item_str}")
        values.append(float(value))
    return values

def resolve_coarse_curriculum_schedule(
    curriculum_consts: List[float],
    curriculum_stage_scales: List[float],
    terminal_const: float,
) -> Tuple[List[float], List[float]]:
    resolved_consts = [float(x) for x in curriculum_consts]
    if len(resolved_consts) == 0:
        resolved_consts = [float(terminal_const)]
    else:
        # L'ultimo stage deve produrre artefatti coerenti col training ricorsivo vero.
        resolved_consts[-1] = float(terminal_const)

    resolved_scales = [float(x) for x in curriculum_stage_scales]
    if len(resolved_scales) == 0:
        resolved_scales = [1.0] * len(resolved_consts)
    elif len(resolved_scales) == 1 and len(resolved_consts) > 1:
        resolved_scales = [float(resolved_scales[0])] * len(resolved_consts)
    elif len(resolved_scales) != len(resolved_consts):
        raise ValueError(
            "coarse curriculum stage scales must have length 1 or the same length as curriculum consts"
        )

    for idx, scale in enumerate(resolved_scales):
        if (not np.isfinite(scale)) or scale <= 0.0:
            raise ValueError(
                f"coarse curriculum stage scale at position {idx} must be finite and > 0, got {scale}"
            )

    return resolved_consts, resolved_scales

def _const_stage_tag(value: float) -> str:
    return str(f"{float(value):.3f}").replace("-", "m").replace(".", "p")
