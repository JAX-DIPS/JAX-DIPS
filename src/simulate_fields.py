from collections import namedtuple

from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from jax import grad
from jax import ops
from jax import random
from jax._src.dtypes import dtype
import jax.numpy as jnp
from jax import lax, vmap

from src import util, space, dataclasses, interpolate, quantity
import pdb


static_cast = util.static_cast


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T,T], T]
Simulator = Tuple[InitFn, ApplyFn]





def advect_one_step(velocity_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    sstate: T,
                    gstate: T,
                    time: float,
                    **kwargs) -> T:
    """Apply a single step of semi-Lagrangian integration to a state."""

    dt = f32(dt)
    dt_2 = f32(dt / 2.0)

    R, U_n, V_nm1= gstate.R, sstate.solution, sstate.velocity_nm1
    V_n = velocity_fn(R, time)
   
    # Get interpolation functions
    Vn_interp_fn = interpolate.vec_multilinear_interpolation(V_n, gstate)
    Vnm1_interp_fn = interpolate.vec_multilinear_interpolation(V_nm1, gstate)

    # Un_interp_fn = interpolate.multilinear_interpolation(U_n, gstate)
    Un_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(U_n, gstate)
    
    # Find Departure Point
    R_star = shift_fn(R, -dt_2 * V_n)                              
    V_n_star = Vn_interp_fn(R_star)
    V_nm1_star = Vnm1_interp_fn(R_star)
    V_mid = f32(1.5) * V_n_star - f32(0.5) * V_nm1_star 
    R_d =  R - dt * V_mid
    # substitute solution from departure point to arrival points (=grid points)
    U_np1 = Un_interp_fn(R_d).flatten()
    return dataclasses.replace(sstate,
                               solution=U_np1,
                               velocity_nm1=V_n)


@dataclasses.dataclass
class SIMState:
    """A struct containing the state of the simulation.

    This tuple stores the state of a simulation.

    Attributes:
    u: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
    """
    solution: Array
    velocity_nm1: Array


def level_set(velocity_or_energy_fn: Callable[..., Array],
              level_set_fn: Callable[..., Array],
              shift_fn: ShiftFn,
              dt: float) -> Simulator:
    """
    Simulates a system.

    Args:
    velocity_fn: A function that produces the velocity field on
        a set of grid points specified as an ndarray of shape
        [n, spatial_dimension]. 
        velocity_fn = -grad(Energy_fn)

    shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
        simulation.

    Returns:
    See above.
    """
    # velocity_fn = quantity.canonicalize_force(velocity_or_energy_fn)
    velocity_fn = vmap(velocity_or_energy_fn, (0,None))
    phi_fn = vmap(level_set_fn)

    def init_fn(R, **kwargs):
        # V = jnp.zeros(R.shape, dtype=R.dtype)
        # U = jnp.zeros(R.shape[0], dtype=R.dtype)
        # U = U + space.square_distance(R) - f32(0.25)
        V = velocity_fn(R, 0.0) #,**kwargs)
        U = phi_fn(R)
        return SIMState(U, V)  

    def apply_fn(sim_state, grid_state, time, **kwargs):
        return advect_one_step(velocity_fn, shift_fn, dt, sim_state, grid_state, time, **kwargs)

    return init_fn, apply_fn



def reinitialize_level_set():
    pass