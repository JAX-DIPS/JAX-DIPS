from collections import namedtuple

from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from jax import grad
from jax import ops
from jax import random
import jax.numpy as jnp
from jax import lax

from src import util, space, dataclasses, partition, smap, quantity

static_cast = util.static_cast


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]



def advect_one_step(force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    state: T,
                    **kwargs) -> T:
  """Apply a single step of velocity verlet integration to a state."""
  
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  dt2_2 = f32(dt ** 2 / 2)

  R, V, F, M = state.position, state.velocity, state.force, state.mass

  Minv = 1 / M

  R = shift_fn(R, V * dt + F * dt2_2 * Minv, **kwargs)

  F_new = force_fn(R, **kwargs)
  V += (F + F_new) * dt_2 * Minv
  return dataclasses.replace(state,
                             position=R,
                             velocity=V,
                             force=F_new)


@dataclasses.dataclass
class SIMState:
    """A struct containing the state of the simulation.

    This tuple stores the state of a simulation.
    
    Attributes:
    position: An ndarray of shape [n, spatial_dimension] storing the position
        of grid points.
    velocity: An ndarray of shape [n, spatial_dimension] storing the velocity
        of particles.
    force: An ndarray of shape [n, spatial_dimension] storing the force acting
        on particles from the previous step.
    mass: A float or an ndarray of shape [n] containing the masses of the
        particles.
    """
    position: Array
    velocity: Array
    force: Array
    mass: Array



def conservative(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float) -> Simulator:
    """Simulates a system.
    
    Args:
    energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
        simulation.
    quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
    See above.
    """
    force_fn = quantity.canonicalize_force(energy_or_force_fn)

    def init_fn(key, R, mass=f32(1.0), **kwargs):
        mass = quantity.canonicalize_mass(mass)
        V = random.normal(key, R.shape, dtype=R.dtype)
        V = V - jnp.mean(V * mass, axis=0, keepdims=True) / mass
        return SIMState(R, V, force_fn(R, **kwargs), mass)  # pytype: disable=wrong-arg-count

    def step_fn(state, **kwargs):
        return advect_one_step(force_fn, shift_fn, dt, state, **kwargs)

    return init_fn, step_fn
