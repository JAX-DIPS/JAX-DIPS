from collections import namedtuple

from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from jax import grad
from jax import ops
from jax import random
from jax._src.dtypes import dtype
import jax.numpy as jnp
from jax import lax

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
                    **kwargs) -> T:
    """Apply a single step of semi-Lagrangian integration to a state."""

    dt = f32(dt)
    dt_2 = f32(dt / 2)

    R, U_n, V_nm1= gstate.R, sstate.solution, sstate.velocity_nm1
    V_n = velocity_fn(R, **kwargs)
   
    # Get interpolation functions

    def vec_interp_fn(Vec, gstate):
        vx = Vec[:,0]; vy = Vec[:,1]; vz = Vec[:,2]

        def interp_fn(R_):
            vx_interp_fn = interpolate.multilinear_interpolation(vx, gstate)
            vy_interp_fn = interpolate.multilinear_interpolation(vy, gstate)
            vz_interp_fn = interpolate.multilinear_interpolation(vz, gstate)
            xvals = vx_interp_fn(R_)
            yvals = vy_interp_fn(R_)
            zvals = vz_interp_fn(R_)
            return jnp.vstack((xvals, yvals, zvals))
        
        return interp_fn


    Vn_interp_fn = vec_interp_fn(V_n, gstate)
    Vnm1_interp_fn = vec_interp_fn(V_nm1, gstate)
   
    # FIX THIS:
    Un_interp_fn = interpolate.multilinear_interpolation(U_n, gstate)
    # Un_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(U_n, gstate)
    
    # Find Departure Point
    R_star = R - dt_2 * V_n
    V_n_star = Vn_interp_fn(R_star)
    V_nm1__star = Vnm1_interp_fn(R_star)
    V_mid = f32(1.5) * V_n_star - f32(0.5) * V_nm1__star 
    R_d = R - dt * V_mid
    # substitute solution from departure point to arrival points (=grid points)
    U_np1 = Un_interp_fn(R_d)
    
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
              shift_fn: ShiftFn,
              dt: float) -> Simulator:
    """Simulates a system.

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
    velocity_fn = quantity.canonicalize_force(velocity_or_energy_fn)

    def init_fn(R, **kwargs):
        V = jnp.zeros(R.shape, dtype=R.dtype)
        U = jnp.zeros(R.shape[0], dtype=R.dtype)
        U = U + space.square_distance(R) - f32(0.25)
        return SIMState(U, V)  

    def step_fn(sim_state, grid_state, **kwargs):
        return advect_one_step(velocity_fn, shift_fn, dt, sim_state, grid_state, **kwargs)

    return init_fn, step_fn
