from functools import partial

from jax import jit, lax
from jax import numpy as jnp
from jax import vmap

from examples.benchmark_LPBE.units import (
    PI,
    alpha_1,
    alpha_2,
    beta_1,
    beta_2,
    omega,
    kappa,
)

COMPILE_BACKEND = "gpu"
custom_jit = partial(jit, backend=COMPILE_BACKEND)


g_r_coeff = omega / (4.0 * PI * alpha_1)
u_bc_coeff = omega / (4.0 * PI * alpha_2)


# -------------------------------------------------------
# Setting far-field to 0. Start from uniform 0 guess.
# -------------------------------------------------------


@custom_jit
def initial_value_fn(r):
    return 0.0


def get_dirichlet_bc_fn(atom_xyz_rad_chg):
    def dirichlet_bc_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]

        def g_pairwise_kernel(carry, xyzsc):
            (psi,) = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
            # dst = jnp.clip(dst, a_min=sigma * 0.01)
            tmp_psi = chg * jnp.exp((sigma - dst) * kappa) / (dst * (1 + kappa * sigma))
            psi += jnp.nan_to_num(tmp_psi)
            return (psi,), None

        psi = 0.0
        (psi,), _ = lax.scan(g_pairwise_kernel, (psi,), atom_xyz_rad_chg)
        return u_bc_coeff * psi

    # def single_atom_dirichlet_bc_fn(r):
    #     xc, yc, zc, sigma, chg = jnp.squeeze(atom_xyz_rad_chg)
    #     x = r[0]
    #     y = r[1]
    #     z = r[2]
    #     coeff = omega / (4.0 * PI * alpha_2)
    #     dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
    #     dst = jnp.clip(dst, a_min=0.01 * sigma)
    #     val = jnp.exp(kappa * (sigma - dst)) / (dst * (1.0 + kappa * sigma))
    #     return coeff * val

    return jit(dirichlet_bc_fn)


# -------------------------------------------------------
#  Electric permittivities in nondimensionalized PBE
# -------------------------------------------------------


@custom_jit
def mu_m_fn(r):
    r"""
    Diffusion coefficient function in $\Omega^-$
    """
    return alpha_1


@custom_jit
def mu_p_fn(r):
    r"""
    Diffusion coefficient function in $\Omega^+$
    """
    return alpha_2


# -------------------------------------------------------
#  Solvent Effects
# -------------------------------------------------------


@custom_jit
def k_m_fn(r):
    r"""
    Linear term function in $\Omega^-$
    """
    return beta_1


@custom_jit
def k_p_fn(r):
    r"""
    Linear term function in $\Omega^+$
    """
    return beta_2


@custom_jit
def nonlinear_operator_m(u):
    return 0.0


@custom_jit
def nonlinear_operator_p(u):
    return 0.0


# -------------------------------------------------------
#  Jump conditions, and the effect of atomic charges
# -------------------------------------------------------


def get_g_dg_fns(atom_xyz_rad_chg, EPS=1e-6):
    r"""Function for calculating g_star which is the Coloumbic potential generated
    by singular charges; according to Guo-Wei's method.

    Args:
        atom_xyz_rad_chg (_type_): an array with each row having [x, y, z, sigma, charge]
        for each atom in the molecule.
            charges: electron units
            positions: l_tilde units

    Returns:
        g_fn: a function that outputs psi at given coordinates.
    """

    def g_at_r(r):
        x = r[0]
        y = r[1]
        z = r[2]

        def g_pairwise_kernel(carry, xyzsc):
            (psi,) = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
            # dst = jnp.clip(dst, a_min=sigma * 0.01)
            tmp_psi = chg / dst
            psi += jnp.nan_to_num(tmp_psi)
            return (psi,), None

        psi = 0.0
        (psi,), _ = lax.scan(g_pairwise_kernel, (psi,), atom_xyz_rad_chg)
        return g_r_coeff * psi

    g_vec_fn = vmap(g_at_r)

    @custom_jit
    def grad_g_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]

        def grad_g_pairwise_kernel(carry, xyzsc):
            (grad_psi,) = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
            # dst = jnp.clip(dst, a_min=0.01 * sigma)
            tmp_psi = chg * jnp.array([xc - x, yc - y, zc - z]) / dst**3
            grad_psi += jnp.nan_to_num(tmp_psi)
            return (grad_psi,), None

        grad_psi = jnp.array([0.0, 0.0, 0.0])
        (grad_psi,), _ = lax.scan(grad_g_pairwise_kernel, (grad_psi,), atom_xyz_rad_chg)
        return g_r_coeff * grad_psi

    grad_vec_g_fn = vmap(grad_g_fn)
    return jit(g_at_r), jit(g_vec_fn), jit(grad_g_fn), jit(grad_vec_g_fn)


def get_jump_conditions(atom_xyz_rad_chg, g_fn_uns, phi_fn_uns, dx, dy, dz):
    """_summary_

    Args:
        atom_xyz_rad_chg (_type_): atomic coordinates and properties
        g_fn_uns (_type_): kernel for the jump conditions given in LPBE
        phi_fn_uns (_type_): level set function
        dx (_type_): for computing normal derivatives of the level set
        dy (_type_): for computing normal derivatives of the level set
        dz (_type_): for computing normal derivatives of the level set
    """

    def g_fn(r):
        return g_fn_uns(jnp.squeeze(r))

    def phi_fn(r):
        return phi_fn_uns(jnp.squeeze(r))

    @custom_jit
    def normal_fn(point):
        r"""
        Evaluate normal vector at a given point based on interpolated values
        of the level set function at the face-centers of a 3D cell centered at the
        point with each side length given by dx, dy, dz.
        """
        point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
        phi_x = (phi_fn(point_ip1_j_k) - phi_fn(point_im1_j_k)) / (2 * dx)

        point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
        phi_y = (phi_fn(point_i_jp1_k) - phi_fn(point_i_jm1_k)) / (2 * dy)

        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
        phi_z = (phi_fn(point_i_j_kp1) - phi_fn(point_i_j_km1)) / (2 * dz)

        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm])

    @custom_jit
    def grad_g_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]

        def grad_g_pairwise_kernel(carry, xyzsc):
            (grad_psi,) = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
            # dst = jnp.clip(dst, a_min=0.01 * sigma)
            tmp_psi = -chg * jnp.array([xc - x, yc - y, zc - z]) / dst**3
            grad_psi += jnp.nan_to_num(tmp_psi)
            return (grad_psi,), None

        grad_psi = jnp.array([0.0, 0.0, 0.0])
        (grad_psi,), _ = lax.scan(grad_g_pairwise_kernel, (grad_psi,), atom_xyz_rad_chg)
        return g_r_coeff * grad_psi

    @custom_jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return g_fn(r)

    @custom_jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        return alpha_1 * jnp.dot(grad_g_fn(r), normal_fn(r))

    return alpha_fn, beta_fn


# -------------------------------------------------------
#  Source terms in the nondimensionalized PBE with Guo-Wei's
#  treatment of singular charges
# -------------------------------------------------------

""" Note: we use Guo-Wei's strategy, point charges are treated separately through g_star """


def get_rho_fn(atom_xyz_rad_chg):
    r"""Charge density function; in units of eC/Angstrom^3"""

    @custom_jit
    def rho_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]

        def initialize(carry, xyzsc):
            (rho,) = carry
            xc, yc, zc, sigma, chg = xyzsc
            ch_sigma = 0.1 * sigma
            rho += (
                chg
                * jnp.exp(-((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2) / (2 * ch_sigma * ch_sigma))
                / ((2 * PI) ** 1.5 * ch_sigma * ch_sigma * ch_sigma)
            )
            rho = jnp.nan_to_num(rho)
            return (rho,), None

        fm = 0.0
        (fm,), _ = lax.scan(initialize, (fm,), atom_xyz_rad_chg)
        return fm

    return rho_fn


@custom_jit
def f_m_fn(r):
    return 0.0


@custom_jit
def f_p_fn(r):
    return 0.0


def get_exact_sol_fns(single_atom_xyz_rad_chg):
    xc, yc, zc, sigma, chg = jnp.squeeze(single_atom_xyz_rad_chg)

    @custom_jit
    def exact_sol_m_fn(r):
        coeff = omega / (4.0 * PI * sigma)
        val = 1.0 / (alpha_2 * (1 + kappa * sigma)) - 1.0 / alpha_1
        return coeff * val

    @custom_jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        coeff = omega / (4.0 * PI * alpha_2)
        dst = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
        # dst = jnp.clip(dst, a_min=0.01 * sigma)
        val = jnp.exp(kappa * (sigma - dst)) / (dst * (1.0 + kappa * sigma))
        return coeff * val

    return exact_sol_m_fn, exact_sol_p_fn
