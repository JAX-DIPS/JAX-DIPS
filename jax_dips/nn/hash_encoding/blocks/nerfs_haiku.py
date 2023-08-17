import functools
from typing import Callable, List, Tuple

import haiku as hk
from typing import Any as Dtype

import jax
from jax.nn.initializers import Initializer
from jax import nn
import jax.numpy as jnp

from jax_dips.nn.hash_encoding.blocks.encoders_haiku import (
    Encoder,
    FrequencyEncoder,
    HashGridEncoder,
    SphericalHarmonicsEncoder,
)
from jax_dips.nn.hash_encoding.blocks.common import (
    ActivationType,
    DirectionalEncodingType,
    PositionalEncodingType,
    mkValueError,
)

import jax.random as jran

KEY = jran.PRNGKey(0)
KEY, key = jran.split(KEY, 2)


class HashEncMLP(hk.Module):
    def __init__(
        self,
        bound: float,
        position_encoder: Encoder,
        sol_mlp: hk.Module,
        sol_activation: Callable,
    ):
        super().__init__()
        self.bound = bound
        self.position_encoder = position_encoder
        self.sol_mlp = sol_mlp
        self.sol_activation = sol_activation

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


class NeRF(hk.Module):
    def __init__(
        self,
        bound: float,
        position_encoder: Encoder,
        direction_encoder: Encoder,
        density_mlp: hk.Module,
        rgb_mlp: hk.Module,
        density_activation: Callable,
        rgb_activation: Callable,
    ):
        super().__init__()
        self.bound = bound

        self.position_encoder = position_encoder
        self.direction_encoder = direction_encoder

        self.density_mlp = density_mlp
        self.rgb_mlp = rgb_mlp

        self.density_activation = density_activation
        self.rgb_activation = rgb_activation

    def __call__(
        self,
        xyz: jax.Array,
        dir: jax.Array | None,
        appearance_embeddings: jax.Array | None = None,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        """
        Inputs:
            xyz `[..., 3]`: coordinates in $\R^3$
            dir `[..., 3]`: **unit** vectors, representing viewing directions.  If `None`, only
                            return densities
            appearance_embeddings `[..., n_extra_learnable_dims]` or `[n_extra_learnable_dims]`:
                per-image latent code to model illumination, if it's a 1D vector of length
                `n_extra_learnable_dims`, all sampled points will use this embedding.

        Returns:
            density `[..., 1]`: density (ray terminating probability) of each query points
            rgb `[..., 3]`: predicted color for each query point
        """
        original_aux_shapes = xyz.shape[:-1]
        n_samples = functools.reduce(int.__mul__, original_aux_shapes)
        xyz = xyz.reshape(n_samples, 3)

        # [n_samples, D_pos], `float32`
        pos_enc, tv = self.position_encoder(xyz, self.bound)

        x = self.density_mlp(pos_enc)
        # [n_samples, 1], [n_samples, density_MLP_out-1]
        density, _ = jnp.split(x, [1], axis=-1)

        if dir is None:
            return density.reshape(*original_aux_shapes, 1), tv
        dir = dir.reshape(n_samples, 3)

        # [n_samples, D_dir]
        dir_enc = self.direction_encoder(dir)

        # [n_samples, 3]
        if appearance_embeddings is None:
            rgb = self.rgb_mlp(
                jnp.concatenate(
                    [
                        x,
                        dir_enc,
                    ],
                    axis=-1,
                )
            )
        else:
            rgb = self.rgb_mlp(
                jnp.concatenate(
                    [
                        x,
                        dir_enc,
                        jnp.broadcast_to(appearance_embeddings, (n_samples, appearance_embeddings.shape[-1])),
                    ],
                    axis=-1,
                )
            )

        density, rgb = self.density_activation(density), self.rgb_activation(rgb)

        return jnp.concatenate([density, rgb], axis=-1).reshape(*original_aux_shapes, 4), tv


class CoordinateBasedMLP(hk.Module):
    "Coordinate-based MLP"

    def __init__(
        self,  # hidden layer widths
        Ds: List[int],
        out_dim: int,
        skip_in_layers: List[int],
        # as described in the paper
        kernel_init: Initializer = nn.initializers.glorot_uniform(),
    ):
        super().__init__()
        # hidden layer widths
        self.Ds = Ds
        self.out_dim = out_dim
        self.skip_in_layers = skip_in_layers
        # as described in the paper
        self.kernel_init = kernel_init

    def __call__(self, x: jax.Array) -> jax.Array:
        in_x = x
        for i, d in enumerate(self.Ds):
            if i in self.skip_in_layers:
                x = jnp.concatenate([in_x, x], axis=-1)
            x = hk.Linear(
                d,
                with_bias=False,
                w_init=self.kernel_init,
            )(x)
            x = nn.relu(x)
        x = hk.Linear(
            self.out_dim,
            with_bias=False,
            w_init=self.kernel_init,
        )(x)
        return x


class BackgroundModel(hk.Module):
    ...


class SkySphereBg(BackgroundModel):
    """
    A sphere that centers at the origin and encloses a bounded scene and provides all the background
    color, this is an over-simplified model.

    When a ray intersects with the sphere from inside, it calculates the intersection point's
    coordinate and predicts a color based on the intersection point and the viewing direction.
    """

    def __init__(
        self,
        r: float,  # radius
        position_encoder: Encoder,  # encoder for position
        direction_encoder: Encoder,  # encoder for viewing direction
        rgb_mlp: CoordinateBasedMLP,  # color predictor
        activation: Callable,
    ):
        super().__init__()
        # radius
        self.r = r
        # encoder for position
        self.position_encoder = position_encoder
        # encoder for viewing direction
        self.direction_encoder = direction_encoder
        # color predictor
        self.rgb_mlp = rgb_mlp
        self.activation = activation

    def __call__(
        self,
        rays_o: jax.Array,
        rays_d: jax.Array,
        appearance_embeddings: jax.Array,
    ) -> jax.Array:
        # the distance of a point (o+td) on the ray to the origin is given by:
        #
        #   dist(t) = (dx^2 + dy^2 + dz^2)t^2 + 2(dx*ox + dy*oy + dz*oz)t + ox^2 + oy^2 + oz^2
        #
        # the minimal distance is achieved when
        #
        #   2(dx^2 + dy^2 + dz^2)t + 2(dx*ox + dy*oy + dz*oz) = 0,
        #       ==> t = -(dx*ox + dy*oy + dz*oz) / (dx^2 + dy^2 + dz^2)
        a = (rays_d * rays_d).sum(axis=-1)
        b = 2 * (rays_o * rays_d).sum(axis=-1)
        c = (rays_o * rays_o).sum(axis=-1) - self.r**2

        # if min_dist < self.r, there are at most two intersections, given by:
        #
        #   dist(t) = r^2
        #
        # want the farther intersection point
        t = jnp.maximum(
            (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a),
            (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a),
        )
        t = t.reshape(-1, 1)

        finite_mask = jnp.isfinite(t)

        pos = rays_o + t * rays_d
        pos_dirs = jnp.where(
            finite_mask,
            pos / (jnp.linalg.norm(pos) + 1e-15),
            0.0,
        )

        n_rays = functools.reduce(int.__mul__, rays_d.shape[:-1])
        # we then encode the positions/directions, and predict a view-dependent color for each ray
        pos_enc = self.position_encoder(pos_dirs)
        dir_enc = self.direction_encoder(rays_d)
        appearance_embeddings = jnp.broadcast_to(
            appearance_embeddings,
            shape=(n_rays, appearance_embeddings.shape[-1]),
        )

        colors = self.rgb_mlp(jnp.concatenate([pos_enc, dir_enc, appearance_embeddings], axis=-1))
        colors = self.activation(colors)

        return jnp.where(
            finite_mask,
            colors,
            0.0,
        )


@jax.jit
def linear_act(x: jax.Array) -> jax.Array:
    return x


def make_activation(act: ActivationType):
    if act == "sigmoid":
        return nn.sigmoid
    elif act == "linear":
        return linear_act
    elif act == "exponential":
        return jnp.exp
    elif act == "truncated_exponential":

        @jax.custom_vjp
        def trunc_exp(x):
            "Exponential function, except its gradient calculation uses a truncated input value"
            return jnp.exp(x)

        def __fwd_trunc_exp(x):
            y = trunc_exp(x)
            aux = x  # aux contains additional information that is useful in the backward pass
            return y, aux

        def __bwd_trunc_exp(aux, grad_y):
            # REF: <https://github.com/NVlabs/instant-ngp/blob/d0d35d215c7c63c382a128676f905ecb676fa2b8/src/testbed_nerf.cu#L303>
            grad_x = jnp.exp(jnp.clip(aux, -15, 15)) * grad_y
            return (grad_x,)

        trunc_exp.defvjp(
            fwd=__fwd_trunc_exp,
            bwd=__bwd_trunc_exp,
        )
        return trunc_exp

    elif act == "thresholded_exponential":

        def thresh_exp(x, thresh):
            """
            Exponential function translated along -y direction by 1e-2, and thresholded to have
            non-negative values.
            """
            # paper:
            #   the occupancy grids ... is updated every 16 steps ... corresponds to thresholding
            #   the opacity of a minimal ray marching step by 1 − exp(−0.01) ≈ 0.01
            return nn.relu(jnp.exp(x) - thresh)

        return functools.partial(thresh_exp, thresh=1e-2)

    elif act == "truncated_thresholded_exponential":

        @jax.custom_vjp
        def trunc_thresh_exp(x, thresh):
            """
            Exponential, but value is translated along -y axis by value `thresh`, negative values
            are removed, and gradient is truncated.
            """
            return nn.relu(jnp.exp(x) - thresh)

        def __fwd_trunc_threash_exp(x, thresh):
            y = trunc_thresh_exp(x, thresh=thresh)
            aux = x, thresh  # aux contains additional information that is useful in the backward pass
            return y, aux

        def __bwd_trunc_threash_exp(aux, grad_y):
            x, thresh = aux
            grad_x = jnp.exp(jnp.clip(x, -15, 15)) * grad_y
            # clip gradient for values that has been thresholded by relu during forward pass
            grad_x = jnp.signbit(jnp.log(thresh) - x) * grad_x
            # first tuple element is gradient for input, second tuple element is gradient for the
            # `threshold` value.
            return (grad_x, 0)

        trunc_thresh_exp.defvjp(
            fwd=__fwd_trunc_threash_exp,
            bwd=__bwd_trunc_threash_exp,
        )
        return functools.partial(trunc_thresh_exp, thresh=1e-2)
    elif act == "relu":
        return nn.relu
    else:
        raise mkValueError(
            desc="activation",
            value=act,
            type=ActivationType,
        )


def make_pos_enc_dir(
    bound: float,
    # encodings
    pos_enc: PositionalEncodingType,
    dir_enc: DirectionalEncodingType,
    # total variation
    tv_scale: float,
    # encoding levels
    pos_levels: int,
    dir_levels: int,
):
    if pos_enc == "identity":
        position_encoder = lambda x: x
    elif pos_enc == "frequency":
        raise NotImplementedError("Frequency encoding for NeRF is not tuned")
        position_encoder = FrequencyEncoder(L=10)
    elif "hashgrid" in pos_enc:
        # @hk.transform
        # def position_encoder(x, y):
        #     HGEncoder = HashGridEncoder(
        #         L=pos_levels,
        #         T=2**19,
        #         F=2,
        #         N_min=2**4,
        #         N_max=int(2**11 * bound),
        #         tv_scale=tv_scale,
        #         param_dtype=jnp.float32,
        #     )
        #     return HGEncoder(x, y)

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

    if dir_enc == "identity":
        direction_encoder = lambda x: x
    elif dir_enc == "sh":
        direction_encoder = SphericalHarmonicsEncoder(L=dir_levels)
        # @hk.transform
        # def direction_encoder(x, y):
        #     direction_encoder_ = SphericalHarmonicsEncoder(L=dir_levels)
        #     return direction_encoder_(x, y)

    else:
        raise mkValueError(
            desc="directional encoding",
            value=dir_enc,
            type=DirectionalEncodingType,
        )

    return position_encoder, direction_encoder


def make_nerf(
    bound: float,
    # encodings
    pos_enc: PositionalEncodingType,
    dir_enc: DirectionalEncodingType,
    # total variation
    tv_scale: float,
    # encoding levels
    pos_levels: int,
    dir_levels: int,
    # layer widths
    density_Ds: List[int],
    rgb_Ds: List[int],
    # output dimensions
    density_out_dim: int,
    rgb_out_dim: int,
    # skip connections
    density_skip_in_layers: List[int],
    rgb_skip_in_layers: List[int],
    # activations
    density_act: ActivationType,
    rgb_act: ActivationType,
) -> NeRF:
    position_encoder, direction_encoder = make_pos_enc_dir(
        bound=bound, pos_enc=pos_enc, dir_enc=dir_enc, tv_scale=tv_scale, pos_levels=pos_levels, dir_levels=dir_levels
    )
    density_mlp = CoordinateBasedMLP(Ds=density_Ds, out_dim=density_out_dim, skip_in_layers=density_skip_in_layers)
    rgb_mlp = CoordinateBasedMLP(Ds=rgb_Ds, out_dim=rgb_out_dim, skip_in_layers=rgb_skip_in_layers)

    density_activation = make_activation(density_act)
    rgb_activation = make_activation(rgb_act)

    model = NeRF(
        bound=bound,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        density_mlp=density_mlp,
        rgb_mlp=rgb_mlp,
        density_activation=density_activation,
        rgb_activation=rgb_activation,
    )

    return model


def make_skysphere_background_model(
    radius: float,
    pos_levels: int,
    dir_levels: int,
    Ds: List[int],
    skip_in_layers: List[int],
    act: ActivationType,
) -> SkySphereBg:
    position_encoder = SphericalHarmonicsEncoder(L=pos_levels)
    direction_encoder = SphericalHarmonicsEncoder(L=dir_levels)
    rgb_mlp = CoordinateBasedMLP(
        Ds=Ds,
        out_dim=3,
        skip_in_layers=skip_in_layers,
    )
    activation = make_activation(act)
    return SkySphereBg(
        r=radius,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        rgb_mlp=rgb_mlp,
        activation=activation,
    )


def make_skysphere_background_model_ngp(bound: float) -> SkySphereBg:
    return make_skysphere_background_model(
        radius=bound * 4,
        pos_levels=2,
        dir_levels=4,
        Ds=[32, 32],
        skip_in_layers=[],
        act="sigmoid",
    )


def make_nerf_ngp(
    bound: float,
    tv_scale: float = 0.0,
) -> NeRF:
    return make_nerf(
        bound=bound,
        pos_enc="hashgrid",
        dir_enc="sh",
        tv_scale=tv_scale,
        pos_levels=16,
        dir_levels=4,
        density_Ds=[64],
        density_out_dim=16,
        density_skip_in_layers=[],
        density_act="truncated_exponential",
        rgb_Ds=[64, 64],
        rgb_out_dim=3,
        rgb_skip_in_layers=[],
        rgb_act="sigmoid",
    )


def make_debug_nerf(bound: float) -> NeRF:
    return NeRF(
        bound=bound,
        position_encoder=lambda x: x,
        direction_encoder=lambda x: x,
        density_mlp=CoordinateBasedMLP(
            Ds=[64],
            out_dim=16,
            skip_in_layers=[],
        ),
        rgb_mlp=CoordinateBasedMLP(
            Ds=[64, 64],
            out_dim=3,
            skip_in_layers=[],
        ),
        density_activation=lambda x: x,
        rgb_activation=lambda x: x,
    )


def make_test_cube(
    width: int,
    bound: float,
    density: float = 32,
    pos_enc: PositionalEncodingType = "hashgrid",
    dir_enc: DirectionalEncodingType = "sh",
    tv_scale: float = 0.0,
    pos_levels: int = 16,
    dir_levels: int = 4,
) -> NeRF:
    @jax.jit
    @jax.vmap
    def cube_density_fn(x: jax.Array) -> jax.Array:
        # x is pre-normalized unit cube, we map it back to specified aabb.
        x = (x + bound) / (2 * bound)
        mask = (abs(x).max(axis=-1, keepdims=True) <= width / 2).astype(float)
        # concatenate input xyz for use in later rgb querying
        return jnp.concatenate([density * mask, x], axis=-1)

    @jax.jit
    @jax.vmap
    def cube_rgb_fn(density_xyz_dir: jax.Array) -> jax.Array:
        # xyz(3) + dir(3), only take xyz to infer color
        x = density_xyz_dir[:3]
        x = jnp.clip(x, -width / 2, width / 2)
        return x / width + 0.5

    position_encoder, direction_encoder = make_pos_enc_dir(
        bound=bound, pos_enc=pos_enc, dir_enc=dir_enc, tv_scale=tv_scale, pos_levels=pos_levels, dir_levels=dir_levels
    )

    # @hk.transform
    # def nerf_forward(x, y):
    #     nerf = NeRF(
    #         bound=bound,
    #         position_encoder=position_encoder,
    #         direction_encoder=direction_encoder,
    #         density_mlp=cube_density_fn,
    #         rgb_mlp=cube_rgb_fn,
    #         density_activation=lambda x: x,
    #         rgb_activation=lambda x: x,
    #     )
    #     return nerf(x, y)

    # return nerf_forward
    nerf = NeRF(
        bound=bound,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        density_mlp=cube_density_fn,
        rgb_mlp=cube_rgb_fn,
        density_activation=lambda x: x,
        rgb_activation=lambda x: x,
    )
    return nerf


def make_hash_encoding_network(
    bound: float,
    # encodings
    pos_enc: PositionalEncodingType,
    # total variation
    tv_scale: float,
    # encoding levels
    pos_levels: int,
    # layer widths
    layer_widths: List[int],
    # output dimensions
    sol_out_dim: int,
    # skip connections
    sol_skip_in_layers: List[int],
    # activations
    sol_act: ActivationType,
) -> HashEncMLP:
    if pos_enc == "identity":
        position_encoder = linear_act
    elif pos_enc == "frequency":
        raise NotImplementedError("Frequency encoding for NeRF is not tuned")
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
    sol_mlp = CoordinateBasedMLP(Ds=layer_widths, out_dim=sol_out_dim, skip_in_layers=sol_skip_in_layers)

    sol_activation = make_activation(sol_act)

    model = HashEncMLP(
        bound=bound,
        position_encoder=position_encoder,
        sol_mlp=sol_mlp,
        sol_activation=sol_activation,
    )

    return model


def main():
    import jax.numpy as jnp
    import jax.random as jran

    bound = 1.0
    KEY = jran.PRNGKey(0)
    KEY, key = jran.split(KEY, 2)
    xyz = jnp.ones((100, 3))
    dir = jnp.ones((100, 3))

    if False:
        m = make_nerf_ngp(bound=bound)

    else:

        @hk.transform
        def nerf_forward(x, y):
            m = make_test_cube(
                width=2,
                bound=1.0,
                density=32,
            )
            return m(x, y)

        m = nerf_forward

    params = m.init(key, xyz, dir)
    print(m.tabulate(key, xyz, dir))

    density, rgb = m.apply(
        params,
        jnp.asarray([[0, 0, 0], [1, 1, 1], [1.1, 0, 0], [0.6, 0.9, -0.5], [0.99, 0.99, 0.99]]),
        jnp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    )
    print(density)
    print(rgb)


if __name__ == "__main__":
    main()
