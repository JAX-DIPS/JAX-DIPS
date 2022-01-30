from collections import namedtuple

from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from jax import grad, vmap
from jax import ops
from jax import random
from jax._src.dtypes import dtype
import jax.numpy as jnp
from jax import lax, vmap
from numpy import int32

from src import util, space, dataclasses, interpolate, quantity
import pdb


static_cast = util.static_cast


Array = util.Array
i32 = util.i32
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T,T], T]
ReinitializeFn = Callable[[T,T], T]
Simulator = Tuple[InitFn, ApplyFn, ReinitializeFn]

def which_cell_index(cond): 
    """A USEFULE utility function to find the index of True in a list, 
    for example cond = 0.1 < gstate.x provides a list of True's and False's
    and thie function returns the first time True appears
    """
    cond = jnp.asarray(cond)
    return (jnp.argwhere(~cond, size=1) - 1).flatten()

def find_lower_left_cell_idx(point, gstate):
    """
    find cell index (i,j,k) containing point
    """
    x = gstate.x; y = gstate.y; z = gstate.z
    x_p, y_p, z_p = point
    i = which_cell_index(jnp.asarray(x_p >= x))
    j = which_cell_index(jnp.asarray(y_p >= y))
    k = which_cell_index(jnp.asarray(z_p >= z))
    return i, j, k


def advect_level_set(gstate: T,
                     sstate: T,
                     velocity_fn: Callable[..., Array], 
                     time: float):
    
    R, U_n, V_nm1= gstate.R, sstate.solution, sstate.velocity_nm1
    V_n = vmap(velocity_fn, (0, None))(R, time)

    # Get interpolation functions
    Vn_interp_fn = interpolate.vec_multilinear_interpolation(V_n, gstate)
    Vnm1_interp_fn = interpolate.vec_multilinear_interpolation(V_nm1, gstate)
    Un_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(U_n, gstate)

    def advect_one_step_at_node(point: Array,
                                phi_point: float,                            
                                dt: float) -> T:
        """Apply a single step of semi-Lagrangian integration to a state."""

        dt = f32(dt)
        dt_2 = f32(dt / 2.0)

        vel_nm1_point = velocity_fn(point, time - dt)
        vel_n_point = velocity_fn(point, time)
        u_n_point = phi_point
        
        
        
        # Find Departure Point
        r_star = point - dt_2 * vel_n_point   
        r_star = r_star.reshape(1,-1)      


        vel_n_star_point = Vn_interp_fn(r_star)
        vel_nm1_star_point = Vnm1_interp_fn(r_star)
        v_mid = f32(1.5) * vel_n_star_point - f32(0.5) * vel_nm1_star_point 

        point_d =  point - dt * v_mid
        # substitute solution from departure point to arrival points (=grid points)
        u_np1_point = Un_interp_fn(point_d) #.flatten()
        return u_np1_point.reshape()
    
    return advect_one_step_at_node


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
    R_star = R - dt_2 * V_n                             
    V_n_star = Vn_interp_fn(R_star)
    V_nm1_star = Vnm1_interp_fn(R_star)
    V_mid = f32(1.5) * V_n_star - f32(0.5) * V_nm1_star 
    R_d =  R - dt * V_mid
    # substitute solution from departure point to arrival points (=grid points)
    U_np1 = Un_interp_fn(R_d).flatten()
    return dataclasses.replace(sstate,
                               solution=U_np1,
                               velocity_nm1=V_n)



def reinitialize_level_set(sstate: T,
                           gstate: T,
                           **kwargs) -> T:
    """
    Sussman's reinitialization of the level set function
    to retain its signed distance nature. 
    
    $ \partial_\tau \phi + sgn(\phi^0)(|\nabla \phi| - 1) = 0 $
    where $\tau$ represents a fictitious time.

    This function should be called every few iterations to maintain 
    the level set function.
    """
    # x = gstate.x; y = gstate.y; z = gstate.z
    # dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]

    phi_n = sstate.solution
    sgn_0 = jnp.sign(phi_n)
    
    def step_phi_fn(i, sgn_phi_n):
        sgn_0, phi_n_ = sgn_phi_n
        hg_n = interpolate.godunov_hamiltonian(phi_n_, sgn_0, gstate)              # this function pre-multiplies by proper dt, subtracts -1, multiplies by sign of phi_ijk
        phi_t_np1 = phi_n_ - hg_n * f32(0.01)                                          # jnp.multiply(sgn_0, hg_n)  
        hg_np1 = interpolate.godunov_hamiltonian(phi_t_np1, sgn_0, gstate)
        phi_t_np2 = phi_t_np1 - hg_np1  * f32(0.01)                                     # jnp.multiply(sgn_0, hg_np1)
        phi_n_ = f32(0.5) * (phi_n_ + phi_t_np2)
        return sgn_0, phi_n_

    (sgn_0, phi_n) = lax.fori_loop(i32(0), i32(10), step_phi_fn, (sgn_0, phi_n))

    return dataclasses.replace(sstate, solution=phi_n)



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

    def reinitialize_fn(sim_state, grid_state, **kwargs):
        return reinitialize_level_set(sim_state, grid_state, **kwargs)

    return init_fn, apply_fn, reinitialize_fn



