"""Model specifications and thin factories for supported benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .exact import build_exact_initial_boundary_samples, build_exact_solution_functions
from .sampling import Xi_generator_default, make_deterministic_xi_default


def _require_state_dim_4(D: int) -> None:
    if int(D) != 4:
        raise ValueError(f"ModelSpec-backed models in this repo require D=4, found D={int(D)}")


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
    _require_state_dim_4(D)
    return [int(D) + 1] + 4 * [256] + [1]


def _build_default_ou_params() -> dict:
    return {
        "kappa_day": np.float32(0.40),
        "kappa_night": np.float32(0.40),
        "a0_day": np.float32(-1.0),
        "a0_night": np.float32(-1.0),
        "sigma_day": np.float32(0.10),
        "sigma_night": np.float32(0.15),
        "alpha_day": np.asarray([np.float32(0.0)], dtype=np.float32),
        "alpha_night": np.asarray([np.float32(0.0)], dtype=np.float32),
        "beta_day": np.asarray([np.float32(0.0)], dtype=np.float32),
        "beta_night": np.asarray([np.float32(0.0)], dtype=np.float32),
    }


def _build_pascucci_params(const: float = 1.0) -> dict:
    return {
        "l_v": np.float32(0.01),
        "l_a": np.float32(0.01),
        "c3": np.float32(10.0),
        "c4": np.float32(10.0),
        "gamma": np.float32(1.0),
        "d": np.float32(1.0),
        "x_max": np.float32(10.0),
        "v_max": np.float32(2.0),
        "v_min": np.float32(-2.0),
        "s3": np.float32(0.01),
        "s3h": np.float32(0.001),
        "s3v": np.float32(0.001),
        "s3k": np.float32(0.001),
        "omega": np.float32(0.01),
        "c_h": np.float32(0.0001),
        "c_con": np.float32(0.01),
        "const": np.float32(const),
        "pascucci_cost_profile": "exp",
        "pascucci_cost_offset": np.float32(0.0),
        "params_S": _build_default_ou_params(),
        "params_H": _build_default_ou_params(),
    }


def _build_pascucci_layers(D: int) -> list[int]:
    _require_state_dim_4(D)
    return [int(D) + 1] + 4 * [256] + [1]


def _build_pascucci_exact_solution(exact_solution: str, params: dict, D: int):
    requested = str(exact_solution or "none").strip().lower()
    if requested in ("", "none"):
        return None
    raise ValueError(
        "pascucci does not provide an exact solution profile yet; "
        "use --exact_solution none"
    )


def _build_pascucci_standard_model(**kwargs: Any):
    from .models import NN_Pascucci

    return NN_Pascucci(
        kwargs["Xi_generator"],
        kwargs["T"],
        kwargs["M"],
        kwargs["N"],
        kwargs["D"],
        kwargs["layers"],
        kwargs["params"],
    )


def _build_pascucci_recursive_model(**kwargs: Any):
    from .models import NN_Pascucci_Recursive

    return NN_Pascucci_Recursive(**kwargs)


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
    moment_names: tuple[str, ...] = ()

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
    if requested == "quadratic_coupled":
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

    if requested == "pascucci":
        return ModelSpec(
            name="pascucci",
            state_dim=4,
            state_labels=("S", "H", "V", "X_state"),
            z_labels=("Z_S", "Z_H", "Z_V", "Z_X"),
            build_default_params=_build_pascucci_params,
            build_layers=_build_pascucci_layers,
            xi_generator=Xi_generator_default,
            deterministic_xi=make_deterministic_xi_default,
            standard_model_factory=_build_pascucci_standard_model,
            recursive_model_factory=_build_pascucci_recursive_model,
            build_exact_solution=_build_pascucci_exact_solution,
            build_exact_initial_boundary_samples=None,
            moment_names=("mean_v", "mean_q", "mean_h_plus_v"),
        )

    raise ValueError(
        f"Unknown model '{name}'. Supported: quadratic_coupled, pascucci"
    )
