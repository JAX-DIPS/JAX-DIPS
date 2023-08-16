"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2022 Pouria Mistani and Samira Pakravan. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: mistani

"""
from typing import Callable, Tuple, List
import functools

import flax.linen as nn
import haiku as hk
import jax
from jax import config
from jax import numpy as jnp

config.update("jax_debug_nans", False)

from jax_dips._jaxmd_modules.util import f32
from jax_dips.nn.hash_encoding.blocks.encoders import (
    Encoder,
    FrequencyEncoder,
    HashGridEncoder,
)
from jax_dips.nn.hash_encoding.blocks.common import (
    ActivationType,
    PositionalEncodingType,
    empty_impl,
    mkValueError,
)
from jax_dips.nn.hash_encoding.blocks.nerfs import (
    make_activation,
    CoordinateBasedMLP,
    linear_act,
)


@empty_impl
class HashMLP(nn.Module):
    bound: float

    position_encoder: Encoder

    sol_mlp: nn.Module

    sol_activation: Callable

    @nn.compact
    def __call__(
        self,
        xyz: jax.Array,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        """
        Inputs:
            xyz `[..., 3]`: coordinates in $\R^3$

        Returns:
            density `[..., 1]`: density (ray terminating probability) of each query points
            rgb `[..., 3]`: predicted color for each query point
        """
        original_aux_shapes = xyz.shape[:-1]
        n_samples = functools.reduce(int.__mul__, original_aux_shapes)
        xyz = xyz.reshape(n_samples, 3)

        # [n_samples, D_pos], `float32`
        pos_enc, tv = self.position_encoder(xyz, self.bound)

        x = self.sol_mlp(pos_enc)
        # [n_samples, 1], [n_samples, density_MLP_out-1]
        sol, _ = jnp.split(x, [1], axis=-1)

        sol = self.sol_activation(sol)

        return sol


def make_hash_network(
    bound: float,
    # encodings
    pos_enc: PositionalEncodingType = "hashgrid",
    # total variation
    tv_scale: float = 0.0,
    # encoding levels
    pos_levels: int = 16,
    # layer widths, these are relu activated
    layer_widths: List[int] = [64],
    # skip connections
    sol_skip_in_layers: List[int] = [],
    # activations
    sol_act: ActivationType = "linear",
) -> HashMLP:
    if pos_enc == "identity":
        position_encoder = linear_act
    elif pos_enc == "frequency":
        position_encoder = FrequencyEncoder(L=10)
    elif "hashgrid" in pos_enc:
        HGEncoder = HashGridEncoder
        position_encoder = HGEncoder(
            L=pos_levels,
            T=2**19,
            F=2,
            N_min=2**4,
            N_max=int(2**11 * bound),
            tv_scale=tv_scale,
            param_dtype=jnp.float32,
        )
    else:
        raise mkValueError(
            desc="positional encoding",
            value=pos_enc,
            type=PositionalEncodingType,
        )
    sol_mlp = CoordinateBasedMLP(Ds=layer_widths, out_dim=1, skip_in_layers=sol_skip_in_layers)

    sol_activation = make_activation(sol_act)

    model = HashMLP(
        bound=bound,
        position_encoder=position_encoder,
        sol_mlp=sol_mlp,
        sol_activation=sol_activation,
    )

    return model


class HashNetwork(hk.Module):
    def __init__(
        self,
        name=None,
        model_type: str = "multiresolution_hash_network",
        hashnet: dict = {
            "Nx": 32,
            "Ny": 32,
            "Nz": 32,
            "xmin": -1.0,
            "xmax": 1.0,
            "ymin": -1.0,
            "ymax": 1.0,
            "zmin": -1.0,
            "zmax": 1.0,
        },
        **kwargs,
    ):
        super().__init__(name=name)
        bound = hashnet["xmax"] - hashnet["xmin"]
        self.model_m = make_hash_network(bound)
        self.model_p = make_hash_network(bound)

        import jax.random as jran

        KEY = jran.PRNGKey(0)
        KEY, key = jran.split(KEY, 2)
        xyz = jnp.ones((100, 3))
        params = self.model_m.init(key, xyz)
        print(self.model_m.tabulate(key, xyz))

        sol = self.model_m.apply(
            params,
            jnp.asarray([[0, 0, 0], [1, 1, 1], [1.1, 0, 0], [0.6, 0.9, -0.5], [0.99, 0.99, 0.99]]),
        )
        print(sol)

    def __call__(self, r, phi_r):
        """
        Driver function for evaluating neural networks in appropriate regions
        based on the value of the level set function at the point.
        """
        return jnp.where(phi_r >= 0, self.model_p(r), self.model_m(r))

    @staticmethod
    def __version__():
        return "0.3.0"
