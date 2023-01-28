from jax import random, numpy as jnp, jit
import jax
from src.jaxmd_modules.util import f32


def get_initial_level_set_fn():
    # -- Initialize the STARS in the BOX
    num_stars_x = num_stars_y = num_stars_z = 2  # Ensure you are solving your system
    scale = 0.35  # This is for proper separation between stars

    r0 = 0.483 * scale
    ri = 0.151 * scale
    re = 0.911 * scale
    n_1 = 3.0
    beta_1 = 0.1 * scale
    theta_1 = 0.5
    n_2 = 4.0
    beta_2 = -0.1 * scale
    theta_2 = 1.8
    n_3 = 7.0
    beta_3 = 0.15 * scale
    theta_3 = 0.0

    key = random.PRNGKey(0)
    cov = jnp.eye(3)
    mean = jnp.zeros(3)
    angles = random.multivariate_normal(
        key, mean, cov, shape=(num_stars_x * num_stars_y * num_stars_z,)
    )
    xc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_x, dtype=f32)
    yc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_y, dtype=f32)
    zc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_z, dtype=f32)
    Xce, Yce, Zce = jnp.meshgrid(xc, yc, zc)
    positions = jnp.column_stack((Xce.reshape(-1), Yce.reshape(-1), Zce.reshape(-1)))

    stars = jnp.concatenate((positions, angles), axis=1)

    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        def initialize(carry, xyz):
            (phi_,) = carry
            # xc, yc, zc = xyz
            xc, yc, zc, theta_1, theta_2, theta_3 = xyz
            theta_1 *= jnp.pi
            theta_2 *= jnp.pi
            theta_3 *= jnp.pi
            core = beta_1 * jnp.cos(n_1 * (jnp.arctan2(y - yc, x - xc) - theta_1))
            core += beta_2 * jnp.cos(n_2 * (jnp.arctan2(y - yc, x - xc) - theta_2))
            core += beta_3 * jnp.cos(n_3 * (jnp.arctan2(y - yc, x - xc) - theta_3))
            phi_ = jnp.min(
                jnp.array(
                    [
                        phi_,
                        jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
                        - 1.0
                        * r0
                        * (
                            1.0
                            + (
                                ((x - xc) ** 2 + (y - yc) ** 2)
                                / ((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
                            )
                            ** 2
                            * core
                        ),
                    ]
                )
            )
            phi_ = jnp.nan_to_num(phi_, -r0 * core)
            return (phi_,), None

        phi_ = 1e9
        (phi_,), _ = jax.lax.scan(initialize, (phi_,), stars)

        return phi_

    return unperturbed_phi_fn
