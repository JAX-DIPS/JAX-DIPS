"""Describes different physical quantities."""


from functools import partial
from typing import Callable, Tuple, TypeVar, Union

import jax.numpy as jnp
from jax import eval_shape, grad, vmap

from jax_dips._jaxmd_modules import dataclasses, partition, space, util

# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

DisplacementFn = space.DisplacementFn
MetricFn = space.MetricFn
Box = space.Box

EnergyFn = Callable[..., Array]
ForceFn = Callable[..., Array]

T = TypeVar("T")
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


def force(energy_fn: EnergyFn) -> ForceFn:
    """Computes the force as the negative gradient of an energy."""
    return grad(lambda R, *args, **kwargs: -energy_fn(R, *args, **kwargs))


def clipped_force(energy_fn: EnergyFn, max_force: float) -> ForceFn:
    force_fn = force(energy_fn)

    def wrapped_force_fn(R, *args, **kwargs):
        force = force_fn(R, *args, **kwargs)
        force_norm = jnp.linalg.norm(force, axis=-1, keepdims=True)
        return jnp.where(force_norm > max_force, force / force_norm * max_force, force)

    return wrapped_force_fn


def canonicalize_force(energy_or_force_fn: Union[EnergyFn, ForceFn]) -> ForceFn:
    _force_fn = None

    def force_fn(R, **kwargs):
        nonlocal _force_fn
        if _force_fn is None:
            out_shape = eval_shape(energy_or_force_fn, R, **kwargs).shape
            if out_shape == ():
                _force_fn = force(energy_or_force_fn)
            else:
                _force_fn = energy_or_force_fn
        return _force_fn(R, **kwargs)

    return force_fn


def volume(dimension: int, box: Box) -> float:
    if jnp.isscalar(box) or not box.ndim:
        return box**dimension
    elif box.ndim == 1:
        return jnp.prod(box)
    elif box.ndim == 2:
        return jnp.linalg.det(box)
    raise ValueError(("Box must be either: a scalar, a vector, or a matrix. " f"Found {box}."))


def canonicalize_mass(mass: Union[float, Array]) -> Union[float, Array]:
    if isinstance(mass, float):
        return mass
    elif isinstance(mass, jnp.ndarray):
        if len(mass.shape) == 2 and mass.shape[1] == 1:
            return mass
        elif len(mass.shape) == 1:
            return jnp.reshape(mass, (mass.shape[0], 1))
        elif len(mass.shape) == 0:
            return mass
    elif isinstance(mass, f32) or isinstance(mass, f64):
        return mass
    msg = "Expected mass to be either a floating point number or a one-dimensional" "ndarray. Found {}.".format(mass)
    raise ValueError(msg)
