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
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random, vmap
from jax.config import config

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import interpolate, mesh
from jax_dips.geometry import level_set
from jax_dips.solvers.poisson.deprecated import poisson_solver_scalable
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
    Nx_tr = Ny_tr = Nz_tr = 128
    multi_gpu = False
    num_epochs = 50

    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    init_mesh_fn, coord_at = mesh.construct(dim)

    # --------- Grid nodes for level set
    Nx = Ny = Nz = i32(128)
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    # ----------  Evaluation Mesh for Visualization
    Nx_eval = Ny_eval = Nz_eval = i32(256)
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    # -- Initialize the STARS in the BOX
    num_stars_x = num_stars_y = num_stars_z = 4  # Ensure you are solving your system
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
    angles = random.multivariate_normal(key, mean, cov, shape=(num_stars_x * num_stars_y * num_stars_z,))
    xc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_x, dtype=f32)
    yc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_y, dtype=f32)
    zc = jnp.linspace(-1 + 1.15 * re, 1 - 1.15 * re, num_stars_z, dtype=f32)
    Xce, Yce, Zce = jnp.meshgrid(xc, yc, zc)
    positions = jnp.column_stack((Xce.reshape(-1), Yce.reshape(-1), Zce.reshape(-1)))

    stars = jnp.concatenate((positions, angles), axis=1)

    @custom_jit
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
                            + (((x - xc) ** 2 + (y - yc) ** 2) / ((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)) ** 2
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

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

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
    )
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver_scalable.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    log = {"phi": eval_phi, "U": sim_state.solution}
    io.write_vtk_manual(eval_gstate, log, filename=currDir + "results/starbox")


if __name__ == "__main__":
    poisson_solver_with_jump_complex()
