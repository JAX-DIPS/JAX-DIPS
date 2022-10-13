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
from jax.config import config
from src import io, poisson_solver, mesh, level_set
from src.jaxmd_modules.util import f32, i32
from jax import (jit, numpy as jnp, vmap, grad, lax)
import jax
import jax.profiler
import time
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def test_poisson_solver_with_jump():

    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    Nx = i32(8)
    Ny = i32(8)
    Nz = i32(8)

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
        return jnp.exp(z)

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
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.5
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
        return y*y*jnp.log(x+2.0)+4.0

    @jit
    def mu_p_fn(r):
        """
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.exp(-1.0*z)

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
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)

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
        return 0.0 #evaluate_exact_solution_fn(r)



    @jit
    def f_m_fn_(r):
        """
        Source function in $\Omega^-$
        """
        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p)*grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)
            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i]**2 + tangent[i]
            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)
        return laplacian_m_fn(r) * (-1.0)

    @jit
    def f_p_fn_(r):
        """
        Source function in $\Omega^+$
        """
        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p)*grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)
            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i]**2 + tangent[i]
            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)
        return laplacian_p_fn(r) * (-1.0)


    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return -1.0 * jnp.exp(z) * (y*y*jnp.log(x+2)+4)

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 2.0 * jnp.exp(-1.0 * z) * jnp.cos(x) * jnp.sin(y)


    exact_sol = vmap(evaluate_exact_solution_fn)(R)

    init_fn, solve_fn = poisson_solver.setup(initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)



    SWITCHING_INTERVAL = 2

    t1 = time.time()

    sim_state, epoch_store, loss_epochs = solve_fn(gstate, sim_state, algorithm=0, switching_interval=SWITCHING_INTERVAL)

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
        'f_p': sim_state.f_p,
        'grad_um_x': sim_state.grad_solution[0][:,0],
        'grad_um_y': sim_state.grad_solution[0][:,1],
        'grad_um_z': sim_state.grad_solution[0][:,2],
        'grad_up_x': sim_state.grad_solution[1][:,0],
        'grad_up_y': sim_state.grad_solution[1][:,1],
        'grad_up_z': sim_state.grad_solution[1][:,2],
        'grad_um_n': sim_state.grad_normal_solution[0],
        'grad_up_n': sim_state.grad_normal_solution[1]
    }
    io.write_vtk_manual(gstate, log)

    L_inf_err = abs(sim_state.solution - exact_sol).max()
    rms_err = jnp.square(sim_state.solution - exact_sol).mean()**0.5
    print("\n SOLUTION ERROR\n")

    print(f"L_inf error on solution everywhere in the domain is = {L_inf_err} and root-mean-squared error = {rms_err} ")

    assert L_inf_err<0.3


if __name__ == "__main__":
    test_poisson_solver_with_jump()
