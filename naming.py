"""Small naming helpers shared by plotting, logs, and selection code."""

from __future__ import annotations

from typing import List


def _pass_index(pass_id: int) -> int:
    pid = int(pass_id)
    return pid - 1 if pid >= 1 else pid


def _pass_label(pass_id: int) -> str:
    return f"pass{_pass_index(pass_id)}"


def _pass_tag(pass_id: int, width: int = 2) -> str:
    return f"pass{_pass_index(pass_id):0{int(width)}d}"


def _z_component_labels(n_components: int) -> List[str]:
    base = ["Z_S", "Z_H", "Z_V", "Z_X"]
    labels = []
    for i in range(int(n_components)):
        labels.append(base[i] if i < len(base) else f"Z_{i}")
    return labels

