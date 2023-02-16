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
from jax_dips import io, poisson_solver_scalable, mesh, level_set, interpolate
from jax_dips.jaxmd_modules.util import f32, i32
from jax import jit, numpy as jnp, vmap, grad, lax, random
import jax
import jax.profiler
import time
import os
import sys
import numpy as onp
from functools import partial

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
    gpu_cnt = jax.local_device_count()

    if gpu_cnt == 0:
        print("No GPUs found.")
        exit(1)
    elif gpu_cnt == 1:
        multi_gpu = False
    else:
        multi_gpu = True

    ALGORITHM = 0  # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    Nx_tr = Ny_tr = Nz_tr = 64
    num_epochs = 20
    batch_size = min(64 * 64 * 32, Nx_tr * Ny_tr * Nz_tr)
    # batch_size = min( 32*32*16, Nx_tr*Ny_tr*Nz_tr)

    checkpoint_dir = f"./checkpoints_{Nx_tr}_{gpu_cnt}"

    print(f"Starting dragon example with resolution {Nx_tr} # of GPU(s) {gpu_cnt} and batchSize {batch_size}")

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
    R = gstate.R

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
    Nx_eval = Nx
    Ny_eval = Ny
    Nz_eval = Nz
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    """ Define functions """

    @custom_jit
    def dirichlet_bc_fn(r):
        return 0.2  # 0.01 / (1e-3 + abs(phi_fn(r)))

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
        return 2.0

    @custom_jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return 0.10

    @custom_jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        return 1.0

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
        return jnp.sin(40 * jnp.pi * x) * jnp.cos(40 * jnp.pi * y) * jnp.sin(40 * jnp.pi * z)

    @custom_jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0

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

    eval_phi = vmap(phi_fn)(eval_gstate.R)

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
        checkpoint_dir=checkpoint_dir,
    )
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver_scalable.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    log = {
        "phi": eval_phi.reshape((Nx_eval, Ny_eval, Nz_eval)),
        "U": sim_state.solution.reshape((Nx_eval, Ny_eval, Nz_eval)),
    }
    io.write_vtk_manual(eval_gstate, log, filename=currDir + f"/results/dragon_final{Nx_tr}_{gpu_cnt}")


if __name__ == "__main__":
    poisson_solver_with_jump_complex()
