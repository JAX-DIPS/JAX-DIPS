from typing import Callable, TypeVar, Union, Tuple, Dict, Optional
from src import (util, space, dataclasses, interpolate)
from jax import (vmap, numpy as jnp)

Array = util.Array
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
    solution: Array
    mu_m: Array
    mu_p: Array
    k_m: Array
    k_p: Array



def setup(lvl_set_fn:  Callable[..., Array], 
          mu_m_fn_  : Callable[..., Array], 
          mu_p_fn_  : Callable[..., Array], 
          k_m_fn_   :  Callable[..., Array], 
          k_p_fn_   :  Callable[..., Array]
          ):

    phi_fn  = vmap(lvl_set_fn)
    mu_m_fn = vmap(mu_m_fn_)
    mu_p_fn = vmap(mu_p_fn_)
    k_m_fn  = vmap(k_m_fn_)
    k_p_fn  = vmap(k_p_fn_)
    
    def init_fn(R):
        U = phi_fn(R)
        MU_M = mu_m_fn(R)
        MU_P = mu_p_fn(R)
        K_M = k_m_fn(R)
        K_P = k_p_fn(R)
        return SState(U, MU_M, MU_P, K_M, K_P) 

    def apply_fn():
        return
    
    return init_fn, apply_fn