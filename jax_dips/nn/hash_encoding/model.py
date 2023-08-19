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


import flax.linen as nn

import jax
from jax import config
from jax import numpy as jnp

config.update("jax_debug_nans", False)

from jax_dips._jaxmd_modules.util import f32
from jax_dips.nn.hash_encoding.blocks.common import (
    ActivationType,
    PositionalEncodingType,
    empty_impl,
    mkValueError,
)
from jax_dips.nn.hash_encoding.blocks.encoders_flax import (
    Encoder,
    FrequencyEncoder,
    HashGridEncoder,
    SphericalHarmonicsEncoder,
)
from jax_dips.nn.hash_encoding.blocks.nerfs_flax import (
    make_activation,
    CoordinateBasedMLP,
    linear_act,
)


@empty_impl
class HashMLP(nn.Module):
    bound: float

    position_encoder: Encoder

    sol_m: nn.Module

    sol_p: nn.Module

    sol_activation: Callable

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        phi_x: f32,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        r"""
        Inputs:
            x `[..., 3]`: coordinate in $\R^3$
            phi_x `[..., 1]`: level-set value

        Returns:
            solution `[..., 1]`: solution field of each query points
        """
        # original_aux_shapes = x.shape[:-1]
        # n_samples = functools.reduce(int.__mul__, original_aux_shapes)
        x = x[jnp.newaxis]

        # [n_samples, D_pos], `float32`
        pos_enc, tv = self.position_encoder(x, self.bound)

        # sol = self.sol_mlp(pos_enc)
        # sol_m, sol_p = jnp.split(sol, [1], axis=-1)

        sol_m = self.sol_m(pos_enc)
        sol_p = self.sol_p(pos_enc)
        sol_m = self.sol_activation(sol_m)
        sol_p = self.sol_activation(sol_p)

        return jnp.where(phi_x >= 0, sol_p.squeeze(), sol_m.squeeze())


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
    # highest spherical harmonic encoding in hash grid featurizer
    highest_sh_order: int = 8,
    # Coarsest resolution (16)
    N_min: int = 2**4,
    # Finest resolution (512 to 524288).
    N_max: int = 2**11,
    # Number of feature dimensions per entry (2).
    F: int = 2,
    # Maximum entries per level (hash table size) (2**14 to 2**24).
    T: int = 2**19,
) -> HashMLP:
    if pos_enc == "identity":
        position_encoder = lambda x, y: (x, y)
    elif pos_enc == "frequency":
        position_encoder = FrequencyEncoder(L=10)
    elif "hashgrid" in pos_enc:
        HGEncoder = HashGridEncoder
        position_encoder = HGEncoder(
            L=pos_levels,
            T=T,
            F=F,
            N_min=N_min,
            N_max=int(N_max * bound),
            tv_scale=tv_scale,
            param_dtype=jnp.float32,
        )
    elif pos_enc == "spherical_harmonics":
        position_encoder = SphericalHarmonicsEncoder(L=highest_sh_order)
    else:
        raise mkValueError(
            desc="positional encoding",
            value=pos_enc,
            type=PositionalEncodingType,
        )
    # sol_mlp = CoordinateBasedMLP(Ds=layer_widths, out_dim=2, skip_in_layers=sol_skip_in_layers)

    # sol_activation = make_activation(sol_act)

    # model = HashMLP(
    #     bound=bound,
    #     position_encoder=position_encoder,
    #     sol_mlp=sol_mlp,
    #     sol_activation=sol_activation,
    # )

    sol_m = CoordinateBasedMLP(Ds=layer_widths, out_dim=1, skip_in_layers=sol_skip_in_layers)
    sol_p = CoordinateBasedMLP(Ds=layer_widths, out_dim=1, skip_in_layers=sol_skip_in_layers)

    sol_activation = make_activation(sol_act)

    model = HashMLP(
        bound=bound,
        position_encoder=position_encoder,
        sol_m=sol_m,
        sol_p=sol_p,
        sol_activation=sol_activation,
    )

    return model
