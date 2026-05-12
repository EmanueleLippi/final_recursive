"""TensorFlow 2 backend helpers for the modular recursive solver."""

from __future__ import annotations

import os
import importlib
from typing import Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_TF_MODULE = None
_TF_IMPORT_ERROR = None


MIN_TF_VERSION: Tuple[int, int] = (2, 21)


def _parse_major_minor(version: str) -> Tuple[int, int]:
    parts = str(version).split(".")
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def require_tensorflow():
    """Return the TensorFlow module, or raise a precise installation error."""
    global _TF_MODULE, _TF_IMPORT_ERROR
    if _TF_MODULE is not None:
        return _TF_MODULE
    try:
        _TF_MODULE = importlib.import_module("tensorflow")
    except Exception as exc:  # pragma: no cover - exercised only with broken envs.
        _TF_IMPORT_ERROR = exc
        raise RuntimeError(
            "TensorFlow could not be imported in this Python environment. "
            "Install the latest supported release with: "
            "python -m pip install -r code/final_recursive/requirements.txt"
        ) from exc
    return _TF_MODULE


class _TensorFlowProxy:
    def __getattr__(self, name):
        return getattr(require_tensorflow(), name)


tf = _TensorFlowProxy()


def assert_modern_tensorflow() -> None:
    """Fail fast if a TF1/compat execution mode has leaked into this process."""
    module = require_tensorflow()
    if _parse_major_minor(module.__version__) < MIN_TF_VERSION:
        raise RuntimeError(
            f"This package targets TensorFlow >= {MIN_TF_VERSION[0]}.{MIN_TF_VERSION[1]}; "
            f"found {module.__version__}."
        )
    if not module.executing_eagerly():
        raise RuntimeError(
            "TensorFlow eager execution is disabled. Run the TF1/v1_compat code in a "
            "separate process before importing final_recursive."
        )
    module.keras.backend.set_floatx("float32")


def set_seed(seed: int) -> None:
    """Set NumPy and TensorFlow RNG seeds in one place."""
    np.random.seed(int(seed))
    require_tensorflow().random.set_seed(int(seed))


def set_tf_seed(seed: int) -> None:
    """Set only TensorFlow's RNG seed."""
    require_tensorflow().random.set_seed(int(seed))


def reset_backend_state() -> None:
    """Clear Keras global state between independent model builds."""
    require_tensorflow().keras.backend.clear_session()
