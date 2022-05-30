
from jax.config import config
from src import io, poisson_solver, mesh, level_set
from src.util import f32, i32
from jax import (jit, numpy as jnp, vmap, grad)
import jax
import jax.profiler
import pdb
import time
import os
import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def test_poisson_solver_without_jump():

    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    Nx = i32(16)
    Ny = i32(16)
    Nz = i32(16)

    # --------- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y)*jnp.cos(x)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y)*jnp.cos(x)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + y**2 + z**2) + 0.5
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

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
        return 1.0

    @jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r)*grad_u_p_fn(r)
        vec_2 = mu_m_fn(r)*grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec)

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
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return y
        # return exact_sol_p_fn(r)   # PAM: testing

    @jit
    def f_m_fn(r):
        """
        Source function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0 #2.0 * jnp.sin(y) * jnp.cos(x)

    @jit
    def f_p_fn(r):
        """
        Source function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2.0 * jnp.sin(y) * jnp.cos(x)

    exact_sol = vmap(evaluate_exact_solution_fn)(R)

    init_fn, solve_fn = poisson_solver.setup(
        initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)

    t1 = time.time()

    sim_state = solve_fn(gstate, sim_state)
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver.prof")

    log = {
        'phi': sim_state.phi,
        'U': sim_state.solution,
        'U_exact': exact_sol,
        'U-U_exact': sim_state.solution - exact_sol,
        'alpha': sim_state.alpha,
        'beta': sim_state.beta,
        'mu_m': sim_state.mu_m,
        'mu_p': sim_state.mu_p,
        'f_m': sim_state.f_m,
        'f_p': sim_state.f_p
    }
    io.write_vtk_manual(gstate, log)

    L_inf_err = abs(sim_state.solution - exact_sol).max()
    print(f"L_inf error = {L_inf_err}")

    pdb.set_trace()


if __name__ == "__main__":
    test_poisson_solver_without_jump()
