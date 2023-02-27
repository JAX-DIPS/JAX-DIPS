from typing import Callable, TypeVar
from jax_dips._jaxmd_modules import dataclasses, util

T = TypeVar("T")
Array = util.Array
i32 = util.i32
f32 = util.f32
f64 = util.f64


@dataclasses.dataclass
class PoissonSimStateFn:
    u_0_fn: Callable[..., Array]
    dir_bc_fn: Callable[..., Array]
    phi_fn: Callable[..., Array]
    mu_m_fn: Callable[..., Array]
    mu_p_fn: Callable[..., Array]
    k_m_fn: Callable[..., Array]
    k_p_fn: Callable[..., Array]
    f_m_fn: Callable[..., Array]
    f_p_fn: Callable[..., Array]
    alpha_fn: Callable[..., Array]
    beta_fn: Callable[..., Array]
    nonlinear_op_m: Callable[..., T]
    nonlinear_op_p: Callable[..., T]


@dataclasses.dataclass
class PoissonAdvectionSimStateFn:
    u_0_fn: Callable[..., Array]
    dir_bc_fn: Callable[..., Array]
    phi_fn: Callable[..., Array]
    mu_m_fn: Callable[..., Array]
    mu_p_fn: Callable[..., Array]
    k_m_fn: Callable[..., Array]
    k_p_fn: Callable[..., Array]
    f_m_fn: Callable[..., Array]
    f_p_fn: Callable[..., Array]
    alpha_fn: Callable[..., Array]
    beta_fn: Callable[..., Array]
    vel_fn: Callable[..., Array]


@dataclasses.dataclass
class AdvectionSimState:
    """A struct containing the state of the semi-lagrangian advection simulation.

    This tuple stores the state of a simulation.

    Attributes:
    u: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
    """

    phi: Array
    velocity_nm1: Array


@dataclasses.dataclass
class PoissonSimState:
    """A struct containing the state of the simulation.

    This tuple stores the state of a simulation.

    Attributes:
    u: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
    """

    phi: Array
    solution: Array
    dirichlet_bc: Array
    mu_m: Array
    mu_p: Array
    k_m: Array
    k_p: Array
    f_m: Array
    f_p: Array
    alpha: Array
    beta: Array
    grad_solution: Array
    grad_normal_solution: Array


@dataclasses.dataclass
class PoissonAdvectionSimState:
    """A struct containing the state of the simulation.

    This tuple stores the state of a simulation.

    Attributes:
    u: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
    """

    phi: Array
    solution: Array
    dirichlet_bc: Array
    mu_m: Array
    mu_p: Array
    k_m: Array
    k_p: Array
    f_m: Array
    f_p: Array
    alpha: Array
    beta: Array
    grad_solution: Array
    grad_normal_solution: Array
    velocity_nm1: Array
    dt: f32
