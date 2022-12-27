from jax import jit, numpy as jnp, lax, vmap, grad
from functools import partial
from examples.biomolecules.units import *
import pdb

COMPILE_BACKEND = 'gpu'
custom_jit = partial(jit, backend=COMPILE_BACKEND)


##-------------------------------------------------------
## Setting far-field to 0. Start from uniform 0 guess.
##-------------------------------------------------------

@custom_jit
def initial_value_fn(r):
    return 0.0


@custom_jit
def dirichlet_bc_fn(r):
    return 0.0



##-------------------------------------------------------
## Electric permittivities in nondimensionalized PBE
##-------------------------------------------------------

@custom_jit
def mu_m_fn(r):
    """
    Diffusion coefficient function in $\Omega^-$
    """
    return eps_m_r


@custom_jit
def mu_p_fn(r):
    """
    Diffusion coefficient function in $\Omega^+$
    """
    return eps_s_r



##-------------------------------------------------------
## Solvent Effects
##-------------------------------------------------------

if LINEAR_PB:
    @custom_jit
    def k_m_fn(r):
        """
        Linear term function in $\Omega^-$
        """
        return kappa_m_sq

    @custom_jit
    def k_p_fn(r):
        """
        Linear term function in $\Omega^+$
        """
        return kappa_p_sq

    @custom_jit
    def nonlinear_operator_m(u):
        return 0.0

    @custom_jit
    def nonlinear_operator_p(u):
        return 0.0

else:
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
    def nonlinear_operator_m(u):
        return kappa_m_sq * jnp.sinh(u) 

    @custom_jit
    def nonlinear_operator_p(u):
        return kappa_p_sq * jnp.sinh(u)
    




##-------------------------------------------------------
## Jump conditions, and the effect of atomic charges
##-------------------------------------------------------

def get_psi_star(atom_xyz_rad_chg, EPS=1e-6):
    """Function for calculating psi_star which is the Coloumbic potential generated 
    by singular charges; according to Guo-Wei's method.

    Args:
        atom_xyz_rad_chg (_type_): an array with each row having [x, y, z, sigma, charge] 
        for each atom in the molecule. 
            charges: electron units
            positions: l_tilde units

    Returns:
        psi_fn: a function that outputs psi at given coordinates.
    """
    psi_star_coeff = e_tilde * solvent_chg_tilde / (4 * 3.141592653589793 * eps_m * K_B * T * l_tilde)
    def psi_at_r(r):
        x = r[0]
        y = r[1]
        z = r[2]
        def psi_pairwise_kernel(carry, xyzsc):
            psi, = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2)  
            # dst = jnp.clip(dst, a_min=sigma*0.1)
            tmp_psi = chg / dst
            psi += jnp.nan_to_num(tmp_psi)
            return (psi,), None
        psi = 0.0
        (psi,), _ = lax.scan(psi_pairwise_kernel, (psi,), atom_xyz_rad_chg)
        return psi_star_coeff * psi
    psi_vec_fn = vmap(psi_at_r)
    return jit(psi_at_r), jit(psi_vec_fn)



def get_jump_conditions(atom_xyz_rad_chg, psi_fn_uns, phi_fn_uns, dx, dy, dz):
    
    def psi_fn(r):
        return psi_fn_uns(jnp.squeeze(r))
    
    def phi_fn(r):
        return phi_fn_uns(jnp.squeeze(r))
    
    @custom_jit
    def normal_fn(point):
        """
            Evaluate normal vector at a given point based on interpolated values
            of the level set function at the face-centers of a 3D cell centered at the
            point with each side length given by dx, dy, dz.
        """
        point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
        phi_x = (phi_fn(point_ip1_j_k) - phi_fn(point_im1_j_k) ) / (2 * dx)

        point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
        phi_y = (phi_fn(point_i_jp1_k) - phi_fn(point_i_jm1_k) ) / (2 * dy)

        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
        phi_z = (phi_fn(point_i_j_kp1) - phi_fn(point_i_j_km1) ) / (2 * dz)

        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm])
    
    
    psi_star_coeff = e_tilde * solvent_chg_tilde / (4 * 3.141592653589793 * eps_m * K_B * T * l_tilde)
    @custom_jit
    def grad_psi_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        def grad_psi_pairwise_kernel(carry, xyzsc):
            grad_psi, = carry
            xc, yc, zc, sigma, chg = xyzsc
            dst = jnp.sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2)  
            # dst = jnp.clip(dst, a_min=0.1*sigma)
            tmp_psi = chg * jnp.array([xc - x, yc - y, zc - z]) / dst**3
            grad_psi += jnp.nan_to_num(tmp_psi)
            return (grad_psi,), None
        grad_psi = jnp.array([0., 0., 0.])
        (grad_psi,), _ = lax.scan(grad_psi_pairwise_kernel, (grad_psi,), atom_xyz_rad_chg)
        return psi_star_coeff * grad_psi
    
    @custom_jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return psi_fn(r)


    @custom_jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        return eps_m_r * jnp.dot(grad_psi_fn(r), normal_fn(r))  
    
    return alpha_fn, beta_fn 

##-------------------------------------------------------
## Source terms in the nondimensionalized PBE with Guo-Wei's 
## treatment of singular charges
##-------------------------------------------------------

""" Note: we use Guo-Wei's strategy, point charges are treated separately through psi_star """
def get_rho_fn(atom_xyz_rad_chg):
    """ Charge density function """
    @custom_jit
    def rho_fn(r):
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
    return rho_fn


@custom_jit
def f_m_fn(r):
    return 0.0


@custom_jit
def f_p_fn(r):
    return 0.0







