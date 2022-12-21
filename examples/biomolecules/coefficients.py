from jax import jit, numpy as jnp, lax
from functools import partial
from examples.biomolecules.units import *


COMPILE_BACKEND = 'gpu'
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
    return eps_m


@custom_jit
def mu_p_fn(r):
    """
    Diffusion coefficient function in $\Omega^+$
    """
    x = r[0]
    y = r[1]
    z = r[2]
    return eps_s


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



def get_f_m_fn(atom_xyz_rad_chg):
    @custom_jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        def initialize(carry, xyzsc):
            rho, = carry
            xc, yc, zc, sigma, chg = xyzsc
            rho += chg * jnp.exp( -((x-xc)**2 + (y-yc)**2 + (z-zc)**2) / (2*sigma*sigma) ) / ( (2*jnp.pi)**1.5 * sigma*sigma*sigma)
            rho  = jnp.nan_to_num(rho)
            return (rho,), None
        fm = 0.0
        (fm,), _ = lax.scan(initialize, (fm,), atom_xyz_rad_chg)
        return fm
    return f_m_fn


@custom_jit
def f_p_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return 0.0


@custom_jit
def nonlinear_operator_m(u):
    return 0.0 


@custom_jit
def nonlinear_operator_p(u):
    return 0.01 * jnp.sinh(u)