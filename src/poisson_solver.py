from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from src import (poisson_kernels_cellular, util, space, dataclasses, interpolate)
from jax import (vmap, numpy as jnp)


Array = util.Array

T = TypeVar('T')
InitFn = Callable[..., T]
SolveFn = Callable[[T,T], T]
Simulator = Tuple[InitFn, SolveFn]


i32 = util.i32
f32 = util.f32
f64 = util.f64



@dataclasses.dataclass
class SState:
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



def setup(initial_value_fn :  Callable[..., Array], 
          dirichlet_bc_fn  :  Callable[..., Array],
          lvl_set_fn       :  Callable[..., Array], 
          mu_m_fn_         :  Callable[..., Array], 
          mu_p_fn_         :  Callable[..., Array], 
          k_m_fn_          :  Callable[..., Array], 
          k_p_fn_          :  Callable[..., Array],
          f_m_fn_          :  Callable[..., Array],
          f_p_fn_          :  Callable[..., Array],
          alpha_fn_        :  Callable[..., Array],
          beta_fn_         :  Callable[..., Array]
          ) -> Simulator:

    u_0_fn   = vmap(initial_value_fn)
    dir_bc_fn= vmap(dirichlet_bc_fn)
    phi_fn   = vmap(lvl_set_fn)
    mu_m_fn  = vmap(mu_m_fn_)
    mu_p_fn  = vmap(mu_p_fn_)
    k_m_fn   = vmap(k_m_fn_)
    k_p_fn   = vmap(k_p_fn_)
    f_m_fn   = vmap(f_m_fn_)
    f_p_fn   = vmap(f_p_fn_)
    alpha_fn = vmap(alpha_fn_)
    beta_fn  = vmap(beta_fn_)
    
    def init_fn(R):
        PHI   = phi_fn(R)
        DIRBC = dir_bc_fn(R)
        U     = u_0_fn(R)
        MU_M  = mu_m_fn(R)
        MU_P  = mu_p_fn(R)
        K_M   = k_m_fn(R)
        K_P   = k_p_fn(R)
        F_M   = f_m_fn(R)
        F_P   = f_p_fn(R)
        ALPHA = alpha_fn(R)
        BETA  = beta_fn(R)
        return SState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None) 

    def solve_fn(gstate, sim_state):
        U_sol, grad_u_mp, grad_u_mp_normal_to_interface = poisson_kernels_cellular.poisson_solver(gstate, sim_state)
        return dataclasses.replace(sim_state, solution=U_sol, grad_solution=grad_u_mp, grad_normal_solution=grad_u_mp_normal_to_interface)
    
    return init_fn, solve_fn