from jax import jit, numpy as jnp
from functools import partial


COMPILE_BACKEND = "gpu"
custom_jit = partial(jit, backend=COMPILE_BACKEND)


@custom_jit
def dirichlet_bc_fn(r):
    return 0.0


@custom_jit
def mu_m_fn(r):
    """
    Diffusion coefficient function in $\Omega^-$
    """
    x = r[0]
    y = r[1]
    z = r[2]
    return 1.0


@custom_jit
def mu_p_fn(r):
    """
    Diffusion coefficient function in $\Omega^+$
    """
    x = r[0]
    y = r[1]
    z = r[2]
    return 80.0


@custom_jit
def alpha_fn(r):
    """
    Jump in solution at interface
    """
    return 0.0


@custom_jit
def beta_fn(r):
    """
    Jump in flux at interface
    """
    return 0.0


@custom_jit
def k_m_fn(r):
    """
    Linear term function in $\Omega^-$
    """
    return 0.0


@custom_jit
def k_p_fn(r):
    """
    Linear term function in $\Omega^+$
    """
    return 0.0


@custom_jit
def initial_value_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return 0.0


@custom_jit
def f_m_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    fm = (
        -1.0 * mu_m_fn(r) * (-7.0 * jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z))
        + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * x) * 2 * jnp.cos(2 * x) * jnp.cos(2 * y) * jnp.exp(z)
        + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * y) * (-2) * jnp.sin(2 * x) * jnp.sin(2 * y) * jnp.exp(z)
        + 2
        * jnp.cos(2 * jnp.pi * (x + y))
        * jnp.sin(2 * jnp.pi * (x - y))
        * jnp.sin(z)
        * jnp.sin(2 * x)
        * jnp.cos(2 * y)
        * jnp.exp(z)
    )

    return fm


@custom_jit
def f_p_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return 0.0
