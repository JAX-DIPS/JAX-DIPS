from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from src import poisson_kernels_cellular_core_algorithms_scalable
from jax import (vmap, numpy as jnp)

from src.jaxmd_modules import dataclasses, util


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






@dataclasses.dataclass
class SStateFn:
    u_0_fn           :  Callable[..., Array]
    dir_bc_fn        :  Callable[..., Array]
    phi_fn           :  Callable[..., Array] 
    mu_m_fn          :  Callable[..., Array] 
    mu_p_fn          :  Callable[..., Array] 
    k_m_fn           :  Callable[..., Array] 
    k_p_fn           :  Callable[..., Array]
    f_m_fn           :  Callable[..., Array]
    f_p_fn           :  Callable[..., Array]
    alpha_fn         :  Callable[..., Array]
    beta_fn          :  Callable[..., Array]





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
    sim_state_fn = SStateFn(u_0_fn, dir_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)

    def init_fn(R):
        PHI   = sim_state_fn.phi_fn(R)
        DIRBC = None #sim_state_fn.dir_bc_fn(R)
        U     = None #sim_state_fn.u_0_fn(R)
        MU_M  = None #sim_state_fn.mu_m_fn(R)
        MU_P  = None #sim_state_fn.mu_p_fn(R)
        K_M   = None #sim_state_fn.k_m_fn(R)
        K_P   = None #sim_state_fn.k_p_fn(R)
        F_M   = None #sim_state_fn.f_m_fn(R)
        F_P   = None #sim_state_fn.f_p_fn(R)
        ALPHA = None #sim_state_fn.alpha_fn(R)
        BETA  = None #sim_state_fn.beta_fn(R)
        sim_state = SState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None) 
        return sim_state

    def solve_fn(gstate, sim_state, algorithm=0, switching_interval=3):
        U_sol, grad_u_mp, grad_u_mp_normal_to_interface, epoch_store, loss_epochs = poisson_kernels_cellular_core_algorithms_scalable.poisson_solver(gstate, sim_state, sim_state_fn, algorithm, switching_interval=switching_interval)

        return dataclasses.replace(sim_state, solution=U_sol, grad_solution=grad_u_mp, grad_normal_solution=grad_u_mp_normal_to_interface), epoch_store, loss_epochs
    
    return init_fn, solve_fn