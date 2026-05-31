"""Model specifications and thin factories for supported benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .exact import build_exact_initial_boundary_samples, build_exact_solution_functions
from .sampling import Xi_generator_default, make_deterministic_xi_default


def _require_quadratic_dim(D: int) -> None:
    if int(D) != 4:
        raise ValueError(f"quadratic_coupled requires D=4, found D={int(D)}")


def _build_quadratic_params(const: float = 1.0) -> dict:
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


def _build_quadratic_layers(D: int) -> list[int]:
    _require_quadratic_dim(D)
    return [int(D) + 1] + 4 * [256] + [1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    state_dim: int
    state_labels: tuple[str, ...]
    z_labels: tuple[str, ...]
    build_default_params: Callable[..., dict]
    build_layers: Callable[[int], list[int]]
    xi_generator: Callable[[int, int], np.ndarray]
    deterministic_xi: Callable[[int, int, int], np.ndarray]
    standard_model_factory: Callable[..., Any]
    recursive_model_factory: Callable[..., Any]
    build_exact_solution: Callable[..., Optional[dict]]
    build_exact_initial_boundary_samples: Optional[Callable[..., list[np.ndarray]]] = None

    def validate_state_dim(self, D: int) -> None:
        if int(D) != int(self.state_dim):
            raise ValueError(f"{self.name} requires D={self.state_dim}, found D={int(D)}")

    def build_standard_model(self, **kwargs: Any):
        self.validate_state_dim(kwargs["D"])
        return self.standard_model_factory(**kwargs)

    def build_recursive_model(self, **kwargs: Any):
        self.validate_state_dim(kwargs["D"])
        return self.recursive_model_factory(**kwargs)


def _build_quadratic_standard_model(**kwargs: Any):
    from .models import NN_Quadratic_Coupled

    return NN_Quadratic_Coupled(
        kwargs["Xi_generator"],
        kwargs["T"],
        kwargs["M"],
        kwargs["N"],
        kwargs["D"],
        kwargs["layers"],
        kwargs["params"],
    )


def _build_quadratic_recursive_model(**kwargs: Any):
    from .models import NN_Quadratic_Coupled_Recursive

    return NN_Quadratic_Coupled_Recursive(**kwargs)


def get_model_spec(name: Optional[str] = None) -> ModelSpec:
    requested = "quadratic_coupled" if name in (None, "") else str(name).strip().lower()
    if requested != "quadratic_coupled":
        raise ValueError(f"Unknown model '{name}'. Supported: quadratic_coupled")

    return ModelSpec(
        name="quadratic_coupled",
        state_dim=4,
        state_labels=("S", "H", "V", "X_state"),
        z_labels=("Z_S", "Z_H", "Z_V", "Z_X"),
        build_default_params=_build_quadratic_params,
        build_layers=_build_quadratic_layers,
        xi_generator=Xi_generator_default,
        deterministic_xi=make_deterministic_xi_default,
        standard_model_factory=_build_quadratic_standard_model,
        recursive_model_factory=_build_quadratic_recursive_model,
        build_exact_solution=build_exact_solution_functions,
        build_exact_initial_boundary_samples=build_exact_initial_boundary_samples,
    )
