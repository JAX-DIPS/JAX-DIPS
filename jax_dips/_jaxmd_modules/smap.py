import math
from collections import namedtuple
from functools import partial, reduce
from operator import mul
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as onp
from jax import eval_shape, lax, ops, vmap
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe

from jax_dips._jaxmd_modules import quantity, space, util

high_precision_sum = util.high_precision_sum

# Typing

Array = util.Array
f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

DisplacementOrMetricFn = space.DisplacementOrMetricFn


def pair_neighbor_list(
    fn: Callable[..., Array],
    displacement_or_metric: DisplacementOrMetricFn,
    species: Union[Array, int] = None,
    reduce_axis: Optional[Tuple[int, ...]] = None,
    keepdims: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Array]:
    """Promotes a function acting on pairs of particles to use neighbor lists.
    Args:
      fn: A function that takes an ndarray of pairwise distances or displacements
        of shape [n, m] or [n, m, d_in] respectively as well as kwargs specifying
        parameters for the function. fn returns an ndarray of evaluations of
        shape [n, m, d_out].
      metric: A function that takes two ndarray of positions of shape
        [spatial_dimension] and [spatial_dimension] respectively and returns
        an ndarray of distances or displacements of shape [] or [d_in]
        respectively. The metric can optionally take a floating point time as a
        third argument.
      species: Species information for the different particles. This should either
        be None (in which case it is assumed that all the particles have the same
        species), an integer ndarray of shape [n] with species data, or an integer
        in which case the species data will be specified dynamically in the
        mapped function with at most `species` types of particles. Note: that
        dynamic species specification is less efficient, because we cannot
        specialize shape information.
      reduce_axis: A list of axes to reduce over. This is supplied to jnp.sum and
        so the same convention is used.
      keepdims: A boolean specifying whether the empty dimensions should be kept
        upon reduction. This is supplied to jnp.sum and so the same convention is
        used.
      ignore_unused_parameters: A boolean that denotes whether dynamically
        specified keyword arguments passed to the mapped function get ignored
        if they were not first specified as keyword arguments when calling
        `smap.pair_neighbor_list(...)`.
      kwargs: Arguments providing parameters to the mapped function. In cases
        where no species information is provided these should be either 1) a
        scalar, 2) an ndarray of shape [n], 3) an ndarray of shape [n, n],
        3) a binary function that determines how per-particle parameters are to
        be combined. If unspecified then this is taken to be the average of the
        two per-particle parameters. If species information is provided then the
        parameters should be specified as either 1) a scalar or 2) an ndarray of
        shape [max_species, max_species].
    Returns:
      A function fn_mapped that takes an ndarray of floats of shape [N, d_in] of
      positions and and ndarray of integers of shape [N, max_neighbors]
      specifying neighbors.
    """

    kwargs, param_combinators = _split_params_and_combinators(kwargs)
    merge_dicts = partial(util.merge_dicts, ignore_unused_parameters=ignore_unused_parameters)

    def fn_mapped(R, neighbor, **dynamic_kwargs):
        d = partial(displacement_or_metric, **dynamic_kwargs)
        d = vmap(vmap(d, (None, 0)))
        mask = neighbor.idx != R.shape[0]
        R_neigh = R[neighbor.idx]
        dR = d(R, R_neigh)
        merged_kwargs = merge_dicts(kwargs, dynamic_kwargs)
        merged_kwargs = _neighborhood_kwargs_to_params(neighbor.idx, species, merged_kwargs, param_combinators)
        out = fn(dR, **merged_kwargs)
        if out.ndim > mask.ndim:
            ddim = out.ndim - mask.ndim
            mask = jnp.reshape(mask, mask.shape + (1,) * ddim)
        out = jnp.where(mask, out, 0.0)
        return high_precision_sum(out, reduce_axis, keepdims) / 2.0

    return fn_mapped


def _split_params_and_combinators(kwargs):
    combinators = {}
    params = {}

    for k, v in kwargs.items():
        if isinstance(v, Callable):
            combinators[k] = v
        elif isinstance(v, tuple) and isinstance(v[0], Callable):
            assert len(v) == 2
            combinators[k] = v[0]
            params[k] = v[1]
        else:
            params[k] = v
    return params, combinators


def _neighborhood_kwargs_to_params(
    idx: Array,
    species: Array,
    kwargs: Dict[str, Array],
    combinators: Dict[str, Callable],
) -> Dict[str, Array]:
    out_dict = {}
    for k in kwargs:
        if species is None or (util.is_array(kwargs[k]) and kwargs[k].ndim == 1):
            combinator = combinators.get(k, lambda x, y: 0.5 * (x + y))
            out_dict[k] = _get_neighborhood_matrix_params(idx, kwargs[k], combinator)
        else:
            if k in combinators:
                raise ValueError()
            out_dict[k] = _get_neighborhood_species_params(idx, species, kwargs[k])
    return out_dict


# Mapping pairwise functional forms to systems using neighbor lists.


def _get_neighborhood_matrix_params(idx: Array, params: Array, combinator: Callable) -> Array:
    if util.is_array(params):
        if len(params.shape) == 1:
            return combinator(jnp.reshape(params, params.shape + (1,)), params[idx])
        elif len(params.shape) == 2:

            def query(id_a, id_b):
                return params[id_a, id_b]

            query = vmap(vmap(query, (None, 0)))
            return query(jnp.arange(idx.shape[0], dtype=jnp.int32), idx)
        elif len(params.shape) == 0:
            return params
        else:
            raise NotImplementedError()
    elif (
        isinstance(params, int)
        or isinstance(params, float)
        or jnp.issubdtype(params, jnp.integer)
        or jnp.issubdtype(params, jnp.floating)
    ):
        return params
    else:
        raise NotImplementedError


def _get_neighborhood_species_params(idx: Array, species: Array, params: Array) -> Array:
    """Get parameters for interactions between species pairs."""

    # TODO(schsam): We should do better error checking here.
    def lookup(species_a, species_b, params):
        return params[species_a, species_b]

    lookup = vmap(vmap(lookup, (None, 0, None)), (0, 0, None))

    neighbor_species = jnp.reshape(species[idx], idx.shape)
    if util.is_array(params):
        if len(params.shape) == 2:
            return lookup(species, neighbor_species, params)
        elif len(params.shape) == 0:
            return params
        else:
            raise ValueError("Params must be a scalar or a 2d array if using a species lookup.")
    return params
