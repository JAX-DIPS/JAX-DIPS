import math

import chex
import flax.linen as nn
from flax.linen.dtypes import Dtype
import jax
import jax.numpy as jnp
import jax.random as jran

from jax_dips.nn.hash_encoding.blocks.common import (
    empty_impl,
    vmap_jaxfn_with,
)


cell_vert_offsets = {
    2: jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    ),
    3: jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    ),
}

adjacent_offsets = {
    2: jnp.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    ),
    3: jnp.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
}


class Encoder(nn.Module):
    ...


# TODO: enforce types used in arrays
@empty_impl
class HashGridEncoder(Encoder):
    # Let's use the same notations as in the paper

    # Number of levels (16).
    L: int
    # Maximum entries per level (hash table size) (2**14 to 2**24).
    T: int
    # Number of feature dimensions per entry (2).
    F: int
    # Coarsest resolution (16).
    N_min: int
    # Finest resolution (512 to 524288).
    N_max: int

    tv_scale: float

    param_dtype: Dtype = jnp.float32

    @property
    def b(self) -> float:
        # Equation(3)
        # Essentially, it is $(n_max / n_min) ** (1/(L - 1))$
        return math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))

    @nn.compact
    def __call__(self, pos: jax.Array, bound: float) -> jax.Array:
        dim = pos.shape[-1]

        # CAVEAT: hashgrid encoder is defined only in the unit cube [0, 1)^3
        pos = (pos + bound) / (2 * bound)

        scales, resolutions, first_hash_level, offsets = [], [], 0, [0]
        for i in range(self.L):
            scale = self.N_min * (self.b**i) - 1
            scales.append(scale)
            res = math.ceil(scale) + 1
            resolutions.append(res)

            n_entries = res**dim

            if n_entries <= self.T:
                first_hash_level += 1
            else:
                n_entries = self.T

            offsets.append(offsets[-1] + n_entries)

        latents = self.param(
            "latent codes stored on grid vertices",
            # paper:
            #   We initialize the hash table entries using the uniform distribution U(−10^{−4}, 10^{−4})
            #   to provide a small amount of randomness while encouraging initial predictions close
            #   to zero.
            lambda key, shape, dtype: jran.uniform(key, shape, dtype, -1e-4, 1e-4),
            (offsets[-1], self.F),
            self.param_dtype,
        )

        @jax.vmap
        @jax.vmap
        def make_vert_pos(pos_scaled: jax.Array):
            # [dim]
            pos_floored = jnp.floor(pos_scaled)
            # [2**dim, dim]
            vert_pos = pos_floored[None, :] + cell_vert_offsets[dim]
            return vert_pos.astype(jnp.uint32)

        @jax.vmap
        @jax.vmap
        def make_adjacent_pos(pos_scaled: jax.Array):
            # [dim]
            pos_floored = jnp.floor(pos_scaled)
            # [dim * 2, dim]
            adjacent_pos = pos_floored[None, :] + adjacent_offsets[dim]
            return adjacent_pos.astype(jnp.uint32)

        @vmap_jaxfn_with(in_axes=(0, 0))
        @vmap_jaxfn_with(in_axes=(None, 0))
        def make_tiled_indices(res, vert_pos):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                res `uint32` `[L]`: each hierarchy's resolution
                vert_pos `uint32` `[L, n_points, B, dim]`: integer positions of grid cell's
                                                           vertices, of each level

            Returns:
                indices `uint32` `[L, n_points, B]`: grid cell indices of the vertices
            """
            # [dim]
            if dim == 2:
                strides = jnp.stack([jnp.ones_like(res), res]).T
            elif dim == 3:
                strides = jnp.stack([jnp.ones_like(res), res, res**2]).T
            else:
                raise NotImplementedError("{} is only implemented for 2D and 3D data".format(__class__.__name__))
            # [2**dim]
            indices = jnp.sum(strides[None, :] * vert_pos, axis=-1)
            return indices

        @jax.vmap
        @jax.vmap
        def make_hash_indices(vert_pos):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                vert_pos `uint32` `[L, n_points, B, dim]`: integer positions of grid cell's
                                                           vertices, of each level

            Returns:
                indices `uint32` `[L, n_points, B]`: grid cell indices of the vertices
            """
            # use primes as reported in the paper
            primes = jnp.asarray([1, 2_654_435_761, 805_459_861], dtype=jnp.uint32)
            # [2**dim]
            if dim == 2:
                indices = vert_pos[:, 0] ^ (vert_pos[:, 1] * primes[1])
            elif dim == 3:
                indices = vert_pos[:, 0] ^ (vert_pos[:, 1] * primes[1]) ^ (vert_pos[:, 2] * primes[2])
            else:
                raise NotImplementedError("{} is only implemented for 2D and 3D data".format(__class__.__name__))
            return indices

        def make_indices(vert_pos, resolutions, first_hash_level):
            if first_hash_level > 0:
                resolutions = jnp.asarray(resolutions, dtype=jnp.uint32)
                indices = make_tiled_indices(resolutions[:first_hash_level], vert_pos[:first_hash_level, ...])
            else:
                indices = jnp.empty(0, dtype=jnp.uint32)
            if first_hash_level < self.L:
                indices = jnp.concatenate([indices, make_hash_indices(vert_pos[first_hash_level:, ...])], axis=0)
            indices = jnp.mod(indices, self.T)
            indices += jnp.asarray(offsets[:-1], dtype=jnp.uint32)[:, None, None]
            return indices

        @jax.vmap
        @jax.vmap
        def lerp_weights(pos_scaled: jax.Array):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                pos_scaled `float` `[L, n_points, dim]`: coordinates of query points, scaled to the
                                                         hierarchy in question

            Returns:
                weights `float` `[L, n_points, 2**dim]`: linear interpolation weights for each cell
                                                         vertex
            """
            # [dim]
            pos_offset, _ = jnp.modf(pos_scaled)
            # [2**dim, dim]
            widths = jnp.clip(
                # cell_vert_offsets: [2**dim, dim]
                (1 - cell_vert_offsets[dim]) + (2 * cell_vert_offsets[dim] - 1) * pos_offset[None, :],
                0,
                1,
            )
            # [2**dim]
            return jnp.prod(widths, axis=-1)

        # [L]
        scales = jnp.asarray(scales, dtype=jnp.float32)
        # [L, n_points, dim]
        pos_scaled = pos[None, :, :] * scales[:, None, None] + 0.5
        # [L, n_points, 2**dim, dim]
        vert_pos = make_vert_pos(pos_scaled)

        # [L, n_points, 2**dim]
        indices = make_indices(vert_pos, resolutions, first_hash_level)

        # [L, n_points, 2**dim, F]
        vert_latents = latents[indices]
        # [L, n_points, 2**dim]
        vert_weights = lerp_weights(pos_scaled)

        # [L, n_points, F]
        encodings = (vert_latents * vert_weights[..., None]).sum(axis=-2)
        # [n_points, L*F]
        encodings = encodings.transpose(1, 0, 2).reshape(-1, self.L * self.F)

        ## Total variation
        if self.tv_scale > 0:
            # [L, n_points, dim * 2, dim]
            adjacent_pos = make_adjacent_pos(pos_scaled)

            # [L, n_points, dim * 2]
            adjacent_indices = make_indices(adjacent_pos, resolutions, first_hash_level)

            # [L, n_points, dim * 2, F]
            adjacent_latents = latents[adjacent_indices]

            # [L, n_points, dim * 2, F]
            tv = self.tv_scale * jnp.square(adjacent_latents - vert_latents[:, :, :1, :])

            # [L, n_points]
            tv = tv.sum(axis=(-2, -1))

            tv = tv.mean()
        else:
            tv = 0

        return encodings, tv


@empty_impl
class FrequencyEncoder(Encoder):
    """
    Frequency encoding from Equation(4) of the NeRF paper, except the encoded frequency orders are
    different.
    """

    # number of frequencies
    L: int

    # NOTE:
    #   adding @nn.compact makes this not directly callable (CallCompactUnboundModuleError)
    #   See: <https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.CallCompactUnboundModuleError>
    #
    # @nn.compact
    # TODO:
    #   using a function for this (vmap, then jit the vmapped function) seems to be faster (~47ms vs
    #   ~57ms)
    def __call__(self, pos: jax.Array, bound: float) -> jax.Array:
        """
        Inuts:
            pos [..., dim]: `dim`-d coordinates to be frequency-encoded

        Returns:
            encodings [..., 2*dim*L]: frequency-encoded coordinates
        """
        dim = pos.shape[-1]
        pos = (pos + bound) / (2 * bound)
        # [..., dim, L]: 2^{l} * pi * p
        A = jnp.exp2(jnp.arange(self.L)) * jnp.pi * pos[..., None]
        # [..., dim, L], [..., dim, L]
        senc, cenc = jnp.sin(A), jnp.cos(A)
        # [..., dim*L], [..., dim*L]
        senc, cenc = senc.reshape(*A.shape[:-2], dim * self.L), cenc.reshape(*A.shape[:-2], dim * self.L)
        # [..., 2*dim*L]
        encodings = jnp.stack([senc, cenc], axis=-1).reshape(*A.shape[:-2], 2 * dim * self.L)
        return (
            encodings,
            0.0,  # placeholder for the total variation loss
        )


@empty_impl
class SphericalHarmonicsEncoder(Encoder):
    # highest degree
    L: int

    def __call__(self, dirs: jax.Array) -> jax.Array:
        """
        Adapted from <https://github.com/NVlabs/tiny-cuda-nn/blob/39df2387a684e4fe0cfa33542aebf5eab237716b/include/tiny-cuda-nn/encodings/spherical_harmonics.h#L52-L123>

        Inputs:
            dirs [..., 3]: **unit** vectors, representing directions.

        Returns:
            encodings [..., L**2]: real parts of the spherical harmonics up to the L-th degree.
        """
        chex.assert_axis_dimension(dirs, -1, 3)
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        xy, xz, yz = x * y, x * z, y * z
        x2, y2, z2 = x * x, y * y, z * z
        x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
        x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2

        encodings = jnp.empty((*dirs.shape[:-1], self.L**2))

        encodings = encodings.at[..., 0].set(0.28209479177387814)  # 1/(2*sqrt(pi))
        if self.L <= 1:
            return encodings

        encodings = encodings.at[..., 1].set(-0.48860251190291987 * y)  # -sqrt(3)*y/(2*sqrt(pi))
        encodings = encodings.at[..., 2].set(0.48860251190291987 * z)  # sqrt(3)*z/(2*sqrt(pi))
        encodings = encodings.at[..., 3].set(-0.48860251190291987 * x)  # -sqrt(3)*x/(2*sqrt(pi))
        if self.L <= 2:
            return encodings

        encodings = encodings.at[..., 4].set(1.0925484305920792 * xy)  # sqrt(15)*xy/(2*sqrt(pi))
        encodings = encodings.at[..., 5].set(-1.0925484305920792 * yz)  # -sqrt(15)*yz/(2*sqrt(pi))
        encodings = encodings.at[..., 6].set(
            0.94617469575755997 * z2 - 0.31539156525251999
        )  # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
        encodings = encodings.at[..., 7].set(-1.0925484305920792 * xz)  # -sqrt(15)*xz/(2*sqrt(pi))
        encodings = encodings.at[..., 8].set(
            0.54627421529603959 * x2 - 0.54627421529603959 * y2
        )  # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
        if self.L <= 3:
            return encodings

        encodings = encodings.at[..., 9].set(
            0.59004358992664352 * y * (-3.0 * x2 + y2)
        )  # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
        encodings = encodings.at[..., 10].set(2.8906114426405538 * xy * z)  # sqrt(105)*xy*z/(2*sqrt(pi))
        encodings = encodings.at[..., 11].set(
            0.45704579946446572 * y * (1.0 - 5.0 * z2)
        )  # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
        encodings = encodings.at[..., 12].set(
            0.3731763325901154 * z * (5.0 * z2 - 3.0)
        )  # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
        encodings = encodings.at[..., 13].set(
            0.45704579946446572 * x * (1.0 - 5.0 * z2)
        )  # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
        encodings = encodings.at[..., 14].set(1.4453057213202769 * z * (x2 - y2))  # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
        encodings = encodings.at[..., 15].set(
            0.59004358992664352 * x * (-x2 + 3.0 * y2)
        )  # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
        if self.L <= 4:
            return encodings

        encodings = encodings.at[..., 16].set(
            2.5033429417967046 * xy * (x2 - y2)
        )  # 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
        encodings = encodings.at[..., 17].set(
            1.7701307697799304 * yz * (-3.0 * x2 + y2)
        )  # 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
        encodings = encodings.at[..., 18].set(
            0.94617469575756008 * xy * (7.0 * z2 - 1.0)
        )  # 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
        encodings = encodings.at[..., 19].set(
            0.66904654355728921 * yz * (3.0 - 7.0 * z2)
        )  # 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
        encodings = encodings.at[..., 20].set(
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293
        )  # 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
        encodings = encodings.at[..., 21].set(
            0.66904654355728921 * xz * (3.0 - 7.0 * z2)
        )  # 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
        encodings = encodings.at[..., 22].set(
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0)
        )  # 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
        encodings = encodings.at[..., 23].set(
            1.7701307697799304 * xz * (-x2 + 3.0 * y2)
        )  # 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
        encodings = encodings.at[..., 24].set(
            -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4
        )  # 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
        if self.L <= 5:
            return encodings

        encodings = encodings.at[..., 25].set(
            0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        )  # 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 26].set(
            8.3026492595241645 * xy * z * (x2 - y2)
        )  # 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
        encodings = encodings.at[..., 27].set(
            -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0)
        )  # -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
        encodings = encodings.at[..., 28].set(
            4.7935367849733241 * xy * z * (3.0 * z2 - 1.0)
        )  # sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
        encodings = encodings.at[..., 29].set(
            0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0)
        )  # sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
        encodings = encodings.at[..., 30].set(
            0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0)
        )  # sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
        encodings = encodings.at[..., 31].set(
            0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0)
        )  # sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
        encodings = encodings.at[..., 32].set(
            2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0)
        )  # sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
        encodings = encodings.at[..., 33].set(
            -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0)
        )  # -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
        encodings = encodings.at[..., 34].set(
            2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
        encodings = encodings.at[..., 35].set(
            0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        )  # 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
        if self.L <= 6:
            return encodings

        encodings = encodings.at[..., 36].set(
            1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        )  # sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 37].set(
            2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        )  # 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 38].set(
            2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0)
        )  # 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
        encodings = encodings.at[..., 39].set(
            -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)
        )  # -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
        encodings = encodings.at[..., 40].set(
            0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0)
        )  # sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
        encodings = encodings.at[..., 41].set(
            0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0)
        )  # sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
        encodings = encodings.at[..., 42].set(
            6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211
        )  # sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
        encodings = encodings.at[..., 43].set(
            0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0)
        )  # sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
        encodings = encodings.at[..., 44].set(
            0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0)
        )  # sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
        encodings = encodings.at[..., 45].set(
            -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)
        )  # -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
        encodings = encodings.at[..., 46].set(
            0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 47].set(
            2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        )  # 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 48].set(
            10.247761577878714 * x2 * y4
            - 10.247761577878714 * x4 * y2
            + 0.6831841051919143 * x6
            - 0.6831841051919143 * y6
        )  # sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
        if self.L <= 7:
            return encodings

        encodings = encodings.at[..., 49].set(
            0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6)
        )  # 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
        encodings = encodings.at[..., 50].set(
            5.2919213236038001 * xy * z * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        )  # 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 51].set(
            -0.51891557872026028 * y * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + 5.0 * x4 + y4)
        )  # -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
        encodings = encodings.at[..., 52].set(
            4.1513246297620823 * xy * z * (x2 - y2) * (13.0 * z2 - 3.0)
        )  # 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
        encodings = encodings.at[..., 53].set(
            -0.15645893386229404 * y * (3.0 * x2 - y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )  # -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
        encodings = encodings.at[..., 54].set(
            0.44253269244498261 * xy * z * (-110.0 * z2 + 143.0 * z4 + 15.0)
        )  # 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
        encodings = encodings.at[..., 55].set(
            0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )  # sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
        encodings = encodings.at[..., 56].set(
            0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0)
        )  # sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
        encodings = encodings.at[..., 57].set(
            0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )  # sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
        encodings = encodings.at[..., 58].set(
            0.07375544874083044 * z * (x2 - y2) * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0)
        )  # sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
        encodings = encodings.at[..., 59].set(
            -0.15645893386229404 * x * (x2 - 3.0 * y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )  # -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
        encodings = encodings.at[..., 60].set(
            1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
        encodings = encodings.at[..., 61].set(
            -0.51891557872026028 * x * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + x4 + 5.0 * y4)
        )  # -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
        encodings = encodings.at[..., 62].set(
            2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6)
        )  # 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
        encodings = encodings.at[..., 63].set(
            0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6)
        )  # 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
        if self.L <= 8:
            return encodings

        raise NotImplementedError("Largest supported degree of spherical harmonics is 8, got {}".format(self.L))


def bench_sh():
    import time

    L = 8

    sh = SphericalHarmonicsEncoder(L=L)

    @jax.jit
    def shjax_jitted(x):
        return sh(x)

    d = jnp.asarray([[0.1, 0.5, -0.7]])
    d /= jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-15

    K = jran.PRNGKey(0xDEADBEEF)
    for i in range(100):
        K, key = jran.split(K, 2)
        n = 800 * 800
        d = jran.normal(key, (n, 3))
        d /= jnp.linalg.norm(d) + 1e-15

        print("{:03d}-th check ({} coordinates, degree={}): ".format(i + 1, n, L), end="")

        stime = time.time()
        print("|jax...", end="")
        result = shjax_jitted(d).block_until_ready()
        etime = time.time()
        durms = 1000 * (etime - stime)
        print("{:.2f}ms|".format(durms), end="")
        print()


def bench_hg():
    import time

    dim = 3
    L = 16
    F = 2
    T = 2**19
    N_min = 16
    per_level_scale = 2.0
    N_max = int(N_min * per_level_scale ** (L - 1))
    tv_scale = 1.0

    hg = HashGridEncoder(
        # dim=dim,
        L=L,
        T=T,
        F=F,
        N_min=N_min,
        N_max=N_max,
        tv_scale=tv_scale,
    )
    K = jran.PRNGKey(0xABCDEF)
    bound = 1.0
    variables = hg.init(K, jnp.zeros([5, 3]), bound)
    (params_array,) = (variables["params"]["latent codes stored on grid vertices"],)

    print(params_array.shape)

    @jax.jit
    def hgjax_jitted(d):
        return hg.apply(variables, d, bound)

    resolutions, offsets = [], [0]
    for i in range(L):
        res = int(N_min * (per_level_scale**i))
        resolutions.append(res)

        n_entries = (res + 1) ** dim

        if n_entries <= T:
            pass
        else:
            n_entries = T

        offsets.append(offsets[-1] + n_entries)

    # assert offsets[-1] == params_array.shape[0]

    print("starting!")

    for i in range(100):
        K, key = jran.split(K, 2)
        n = 256_000
        d = jran.uniform(key, (n, 3), minval=0, maxval=1.0)
        d_T = jran.uniform(key, (3, n), minval=0, maxval=1.0)

        print("{:03d}-th check ({} coordinates, degree={}): ".format(i + 1, n, L), end="")

        stime = time.time()
        print("|jax...", end="")
        result = hgjax_jitted(d)[0].block_until_ready()
        etime = time.time()
        durms = 1000 * (etime - stime)
        print("{:.2f}ms".format(durms), end="")

        print()


if __name__ == "__main__":
    print("bench_hg")
    bench_hg()

    print()

    print("bench_sh:")
    bench_sh()
