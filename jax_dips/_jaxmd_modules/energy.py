from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

import jax
import jax.numpy as np
from jax import vmap
from jax.scipy.special import erfc  # error function
from jax.tree_util import tree_map

from jax_dips._jaxmd_modules import partition, quantity, smap, space, util

maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList


def gravity(dr: Array, a: Array = 0, b: Array = 1, c: Array = 0, **unused_kwargs) -> Array:
    """Expanding wave motion with conservative potential for velocity
    Args:
      dr: An ndarray of shape [n, m] of pairwise distances between grid points and a reference (e.g., center of grid).
      a: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      b: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      c: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
    Returns:
      Matrix of energies of shape [n, m].
    """
    dr_inv = 1.0 / (dr + 1.0e-7)
    return np.nan_to_num(-b * dr_inv)


def oscillate(dr: Array, a: Array = 0, b: Array = 1, c: Array = 0, **unused_kwargs) -> Array:
    """Expanding wave motion with conservative potential for velocity
    Args:
      dr: An ndarray of shape [n, m] of pairwise distances between grid points and a reference (e.g., center of grid).
      a: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      b: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      c: Should either be a floating point scalar or an ndarray whose shape is [n, m].
      unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
    Returns:
      Matrix of energies of shape [n, m].
    """
    dr2 = dr * dr
    dr4 = dr2 * dr2

    return np.nan_to_num(f32(0.5) * a * dr4 + f32(0.5) * b * dr2 - c * dr)


def energy(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    r_onset: float = 2.0,
    r_cutoff: float = 2.5,
    dr_threshold: float = 0.5,
    per_particle: bool = False,
    fractional_coordinates: bool = False,
) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box_size,
        r_cutoff,
        dr_threshold,
        fractional_coordinates=fractional_coordinates,
    )
    energy_fn = smap.pair_neighbor_list(
        multiplicative_isotropic_cutoff(oscillate, r_onset, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        a=0,
        b=1,
        c=0,
        reduce_axis=(1,) if per_particle else None,
    )

    return neighbor_fn, energy_fn


def multiplicative_isotropic_cutoff(fn: Callable[..., Array], r_onset: float, r_cutoff: float) -> Callable[..., Array]:
    """Takes an isotropic function and constructs a truncated function.
    Given a function f:R -> R, we construct a new function f':R -> R such that
    f'(r) = f(r) for r < r_onset, f'(r) = 0 for r > r_cutoff, and f(r) is C^1
    everywhere. To do this, we follow the approach outlined in HOOMD Blue [1]
    (thanks to Carl Goodrich for the pointer). We construct a function S(r) such
    that S(r) = 1 for r < r_onset, S(r) = 0 for r > r_cutoff, and S(r) is C^1.
    Then f'(r) = S(r)f(r).
    Args:
      fn: A function that takes an ndarray of distances of shape [n, m] as well
        as varargs.
      r_onset: A float specifying the distance marking the onset of deformation.
      r_cutoff: A float specifying the cutoff distance.
    Returns:
      A new function with the same signature as fn, with the properties outlined
      above.
    [1] HOOMD Blue documentation. Accessed on 05/31/2019.
        https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
    """

    r_c = r_cutoff ** f32(2)
    r_o = r_onset ** f32(2)

    def smooth_fn(dr):
        r = dr ** f32(2)

        inner = np.where(
            dr < r_cutoff,
            (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o) ** 3,
            0,
        )

        return np.where(dr < r_onset, 1, inner)

    @wraps(fn)
    def cutoff_fn(dr, *args, **kwargs):
        return smooth_fn(dr) * fn(dr, *args, **kwargs)

    return cutoff_fn
