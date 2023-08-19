import functools
from typing import Callable, List, Tuple

import flax.linen as nn
import jax
from jax.nn.initializers import Initializer
import jax.numpy as jnp


class Preconditioner(nn.Module):
    "Preconditioner model P, so that ||P(Ax-b)|| is minimized when Ax=b"

    # hidden layer widths
    Ds: List[int]
    out_dim: int = 1
    scaling_coeff: float = 1.0

    # as described in the paper
    kernel_init: Initializer = nn.initializers.glorot_uniform()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i, d in enumerate(self.Ds):
            x = nn.Dense(
                d,
                use_bias=True,
                kernel_init=self.kernel_init,
            )(x)
            x = jnp.tanh(x)
        x = nn.Dense(
            self.out_dim,
            use_bias=True,
            kernel_init=self.kernel_init,
        )(x)
        return 0.5 + self.scaling_coeff * nn.sigmoid(x)
