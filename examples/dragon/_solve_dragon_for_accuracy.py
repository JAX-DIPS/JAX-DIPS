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

import os
import sys
import time
from functools import partial

import jax
import jax.profiler
import numpy as onp
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random, vmap
from jax.config import config

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.solvers.poisson.deprecated import poisson_solver_scalable
from jax_dips.domain import interpolate, mesh
from jax_dips.geometry import level_set
from jax_dips.utils import io

COMPILE_BACKEND = "gpu"
custom_jit = partial(jit, backend=COMPILE_BACKEND)

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def poisson_solver_with_jump_complex():
    ALGORITHM = 0  # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    Nx_tr = Ny_tr = Nz_tr = 64  # 1024
    multi_gpu = False  # True
    num_epochs = 1000
    batch_size = min(64 * 64 * 32, Nx_tr * Ny_tr * Nz_tr)

    dim = i32(3)
    init_mesh_fn, coord_at = mesh.construct(dim)

    # --------- Grid nodes for level set
    """ load the dragon """
    dragon_host = onp.loadtxt(currDir + "/dragonian_full.csv", delimiter=",", skiprows=1)
    # file_x = onp.array(xmin + dragon_host[:,0] * gstate.dx)
    # file_y = onp.array(ymin + dragon_host[:,1] * gstate.dy)
    # file_z = onp.array(zmin + dragon_host[:,2] * gstate.dz)
    # file_R = onp.column_stack((file_x, file_y, file_z))

    xmin = 0.118
    xmax = 0.353
    ymin = 0.088
    ymax = 0.263
    zmin = 0.0615
    zmax = 0.1835
    Nx = 236
    Ny = 176
    Nz = 123

    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    gstate = init_mesh_fn(xc, yc, zc)

    """ find the mapping between stored file and solver """
    file_order_index = onp.array(dragon_host[:, :3], dtype=int).astype(str)
    file_order_dragon_phi = onp.array(dragon_host[:, 3])
    mapping = {}
    for idx, val in zip(file_order_index, file_order_dragon_phi):
        mapping["_".join(list(idx))] = val
    I = jnp.arange(Nx)
    J = jnp.arange(Ny)
    K = jnp.arange(Nz)
    II, JJ, KK = jnp.meshgrid(I, J, K, indexing="ij")
    IJK = onp.array(
        jnp.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1)), axis=1),
        dtype=str,
    )
    dragon_phi = []
    for idx in IJK:
        dragon_phi.append(mapping["_".join(list(idx))])
    dragon_phi = onp.array(dragon_phi)
    # io.write_vtk_manual(gstate, {'phi': dragon_phi.reshape((Nx,Ny,Nz))}, filename=currDir + '/results/dragon_initial')

    """ Scaling the system up, rewrite gstate """
    scalefac = 10.0
    xmin = scalefac * xmin - 2.0
    xmax = scalefac * xmax - 2.0
    ymin = scalefac * ymin - 2.0
    ymax = scalefac * ymax - 2.0
    zmin = scalefac * zmin - 2.0
    zmax = scalefac * zmax - 2.0
    dragon = jnp.array(dragon_phi) * scalefac

    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    """ Create interpolant on gstate """
    phi_fn = interpolate.nonoscillatory_quadratic_interpolation_per_point(dragon, gstate)

    """ Evaluation Mesh for Visualization  """
    Nx_eval = 2 * Nx
    Ny_eval = 2 * Ny
    Nz_eval = 2 * Nz
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    """ Define functions """

    @custom_jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z)

    @custom_jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        yx3 = (y - x) / 3.0
        return (16.0 * yx3**5 - 20.0 * yx3**3 + 5.0 * yx3) * jnp.log(x + y + 3) * jnp.cos(z)

    @custom_jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @custom_jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @custom_jit
    def mu_m_fn(r):
        """
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 10.0 * (1 + 0.2 * jnp.cos(2 * jnp.pi * (x + y)) * jnp.sin(2 * jnp.pi * (x - y)) * jnp.cos(z))

    @custom_jit
    def mu_p_fn(r):
        """
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @custom_jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @custom_jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.nan_to_num(jnp.dot(vec_1 - vec_2, n_vec) * (-1.0))

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
        return 0.0  # evaluate_exact_solution_fn(r)

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
        f_p = -1.0 * (
            (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * (y - x) / 3)
            * (-2)
            * jnp.cos(z)
            / (x + y + 3) ** 2
            + 2
            * (16 * 5 * 4 * (1.0 / 9.0) * ((y - x) / 3) ** 3 - 20 * 3 * 2 * (1.0 / 9.0) * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
            + -1
            * (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
        )
        return f_p

    init_fn, solve_fn = poisson_solver_scalable.setup(
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
    )
    sim_state = init_fn(R)

    t1 = time.time()

    sim_state, epoch_store, loss_epochs = solve_fn(
        gstate,
        eval_gstate,
        sim_state,
        algorithm=ALGORITHM,
        switching_interval=SWITCHING_INTERVAL,
        Nx_tr=Nx_tr,
        Ny_tr=Ny_tr,
        Nz_tr=Nz_tr,
        num_epochs=num_epochs,
        multi_gpu=multi_gpu,
        batch_size=batch_size,
    )
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_dragon_poisson_solver_scalable.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    exact_sol = vmap(evaluate_exact_solution_fn)(eval_gstate.R)
    error = sim_state.solution - exact_sol
    log = {
        "phi": eval_phi.reshape((Nx_eval, Ny_eval, Nz_eval)),
        "U": sim_state.solution.reshape((Nx_eval, Ny_eval, Nz_eval)),
        "U_exact": exact_sol.reshape((Nx_eval, Ny_eval, Nz_eval)),
        "U-U_exact": error.reshape((Nx_eval, Ny_eval, Nz_eval)),
    }
    io.write_vtk_manual(eval_gstate, log, filename=currDir + f"/results/dragon_visual{Nx_tr}")

    L_inf_err = abs(sim_state.solution - exact_sol).max()
    rms_err = jnp.square(sim_state.solution - exact_sol).mean() ** 0.5

    print("\n SOLUTION ERROR\n")

    print(
        f"L_inf error on solution everywhere in the domain is = {L_inf_err} and root-mean-squared error = {rms_err} "
    )


if __name__ == "__main__":
    poisson_solver_with_jump_complex()
