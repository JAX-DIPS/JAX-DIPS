# This file is copied from
# https://github.com/google/jax-md/blob/main/jax_md/util.py

from functools import partial
from typing import Any, Iterable, Optional, Union

import jax.numpy as jnp
import numpy as onp
from jax import jit
from jax.lib import xla_bridge
from jax.tree_util import register_pytree_node

Array = jnp.ndarray

i16 = jnp.int16
i32 = jnp.int32
i64 = jnp.int64

f32 = jnp.float32
f64 = jnp.float64


def static_cast(*xs):
    """Function to cast a value to the lowest dtype that can express it."""
    # NOTE(schsam): static_cast is so named because it cannot be jit.
    if xla_bridge.get_backend().platform == "tpu":
        return (jnp.array(x, jnp.float32) for x in xs)
    else:
        return (jnp.array(x, dtype=onp.min_scalar_type(x)) for x in xs)


def register_pytree_namedtuple(cls):
    register_pytree_node(cls, lambda xs: (tuple(xs), None), lambda _, xs: cls(*xs))


def merge_dicts(a, b, ignore_unused_parameters=False):
    if not ignore_unused_parameters:
        return {**a, **b}

    merged = dict(a)
    for key in merged.keys():
        b_val = b.get(key)
        if b_val is not None:
            merged[key] = b_val
    return merged


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def high_precision_sum(X: Array, axis: Optional[Union[Iterable[int], int]] = None, keepdims: bool = False):
    """Sums over axes at 64-bit precision then casts back to original dtype."""
    return jnp.array(jnp.sum(X, axis=axis, dtype=f64, keepdims=keepdims), dtype=X.dtype)


def maybe_downcast(x):
    if isinstance(x, jnp.ndarray) and x.dtype is jnp.dtype("float64"):
        return x
    return jnp.array(x, f32)


def is_array(x: Any) -> bool:
    return isinstance(x, (jnp.ndarray, onp.ndarray))
