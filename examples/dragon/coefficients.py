from jax import jit
from jax import numpy as jnp


@jit
def dirichlet_bc_fn(r):
    return 0.2  # 0.01 / (1e-3 + abs(phi_fn(r)))


@jit
def mu_m_fn(r):
    """
    Diffusion coefficient function in $\Omega^-$
    """
    x = r[0]
    y = r[1]
    z = r[2]
    return 1.0


@jit
def mu_p_fn(r):
    """
    Diffusion coefficient function in $\Omega^+$
    """
    x = r[0]
    y = r[1]
    z = r[2]
    return 2.0


@jit
def alpha_fn(r):
    """
    Jump in solution at interface
    """
    return 0.10


@jit
def beta_fn(r):
    """
    Jump in flux at interface
    """
    return 1.0


@jit
def k_m_fn(r):
    """
    Linear term function in $\Omega^-$
    """
    return 0.0


@jit
def k_p_fn(r):
    """
    Linear term function in $\Omega^+$
    """
    return 0.0


@jit
def nonlinear_operator_m(u):
    return 0.0


@jit
def nonlinear_operator_p(u):
    return 0.0


@jit
def initial_value_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return 0.0


@jit
def f_m_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return jnp.sin(40 * jnp.pi * x) * jnp.cos(40 * jnp.pi * y) * jnp.sin(40 * jnp.pi * z)


@jit
def f_p_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return 0.0
