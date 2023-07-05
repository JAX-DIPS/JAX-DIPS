# adapted from https://github.com/google/jax/issues/11319
from functools import partial

import jax
from jax import numpy as jnp


def get_device_memory_in_GB(device_id: int) -> float:
    import torch

    total = torch.cuda.get_device_properties(device_id).total_memory
    reserved = torch.cuda.memory_reserved(device_id)
    allocated = torch.cuda.memory_allocated(device_id)
    free = reserved - allocated  # free inside reserved
    return total / (1024**3)


def estimate_chunk_size(array: jnp.ndarray) -> int:
    if jnp.isclose(get_device_memory_in_GB(0), 48, atol=5):
        return jnp.ceil(array.shape[0] // 2**23)


def pad_along_axis(array: jnp.ndarray, axis_length: int, axis: int = 0, *args, **kwargs) -> jnp.ndarray:
    target_size = axis_length - jnp.shape(array)[axis]

    padding = [(0, 0)] * jnp.ndim(array)
    padding[axis] = (0, target_size)

    return jnp.pad(array, padding, *args, **kwargs)


def chunked_vmap(fun, num_chunks: int = 1, in_axes=0, out_axes=0, axis_name=None, axis_size=None):
    # TODO: Compatibility on flattened in_axes. Implementation for out_axes, axis_name, axis_size.

    # Note, num_chunks == 1 is equivalent to just using `vmap_fun`.
    vmap_fun = jax.vmap(fun, in_axes, out_axes, axis_name, axis_size)

    # Leaf structure of input splitting: ([chunk_a, chunk_b, ...], [pad_a, pad_b, ...])
    splitted_treedef = jax.tree_util.tree_structure(([1] * num_chunks,) * 2)

    def split_fun(arg, ax):
        # Operates on pytree leaves.

        if ax is None:
            return [arg] * num_chunks, [0] * num_chunks

        chunks = jnp.array_split(arg, num_chunks, axis=ax)

        leading_size = jnp.shape(chunks[0])[ax]
        batch_dims = jax.tree_map(lambda a: jnp.shape(a)[ax], chunks)

        padded_chunks = jax.tree_map(partial(pad_along_axis, axis_length=leading_size, axis=ax), chunks)

        return padded_chunks, batch_dims

    def vmap_f(*args, **kwargs):  # TODO: Incorporate kwargs?
        splitted = jax.tree_map(split_fun, args, in_axes)

        input_chunks, canonical_sizes = jax.tree_util.tree_transpose(
            jax.tree_util.tree_structure(args), splitted_treedef, splitted
        )
        out_sizes = [max(jax.tree_util.tree_leaves(s)) for s in canonical_sizes]

        # TODO: use jax.lax.scan? Note the dynamic shapes of jax.lax.slice and that in_axes is not yet supported.
        results = [jax.lax.slice(vmap_fun(*c), (0,), (s,)) for c, s in zip(input_chunks, out_sizes)]

        # TODO: collect all outputs immediately, or use a generator with `yield`?
        out = jax.tree_map(lambda *a: jnp.concatenate(a), *results)
        return out

    return vmap_f
