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

"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import time
from functools import partial
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import jax
import jax.profiler
import numpy as onp
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random, vmap
from jax.config import config

config.update("jax_enable_x64", False)

from examples.dragon.coefficients import (
    f_m_fn,
    f_p_fn,
    dirichlet_bc_fn,
    alpha_fn,
    beta_fn,
    initial_value_fn,
    k_m_fn,
    k_p_fn,
    mu_m_fn,
    mu_p_fn,
    nonlinear_operator_m,
    nonlinear_operator_p,
)
from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.solvers.poisson import trainer
from jax_dips.solvers.poisson.deprecated import poisson_solver_scalable
from jax_dips.solvers.optimizers import get_optimizer
from jax_dips.domain import interpolate, mesh
from jax_dips.geometry import level_set
from jax_dips.utils import io


@hydra.main(config_path="conf", config_name="dragon", version_base="1.1")
def solve_poisson_over_dragon(cfg: DictConfig):
    logger.info("Starting the dragon training")
    logger.info(OmegaConf.to_yaml(cfg))

    gpu_cnt = jax.local_device_count()

    if gpu_cnt == 0:
        logger.warning("No GPUs found.")
        exit(1)
    elif gpu_cnt == 1:
        multi_gpu = False
    else:
        multi_gpu = True

    dragon_name = cfg.experiment.dragon.name
    ALGORITHM = cfg.solver.algorithm  # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = cfg.solver.switching_interval  # 0: no switching, 1: 10
    multi_gpu = cfg.solver.multi_gpu
    num_epochs = cfg.solver.num_epochs
    checkpoint_interval = cfg.experiment.logging.checkpoint_interval
    log_dir = cfg.experiment.logging.log_dir

    Nx_tr = cfg.solver.Nx_tr
    Ny_tr = cfg.solver.Ny_tr
    Nz_tr = cfg.solver.Nz_tr

    batch_size = min(64 * 64 * 32, Nx_tr * Ny_tr * Nz_tr)

    optimizer = get_optimizer(
        optimizer_name=cfg.solver.optim.optimizer_name,
        scheduler_name=cfg.solver.sched.scheduler_name,
        learning_rate=cfg.solver.optim.learning_rate,
        decay_rate=cfg.solver.sched.decay_rate,
    )

    checkpoint_name = f"checkpoints_{Nx_tr}_{gpu_cnt}"
    checkpoint_dir = os.path.join(log_dir, checkpoint_name)

    logger.info(f"Starting dragon example with resolution {Nx_tr} # of GPU(s) {gpu_cnt} and batchSize {batch_size}")

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

    if cfg.solver.version == 0:
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
        t0 = time.time()
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
        t1 = time.time()

    elif cfg.solver.version == 2:
        init_fn = trainer.setup(
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
            nonlinear_operator_m,
            nonlinear_operator_p,
        )
        sim_state, solve_fn = init_fn(
            gstate=gstate,
            eval_gstate=eval_gstate,
            algorithm=ALGORITHM,
            switching_interval=SWITCHING_INTERVAL,
            Nx_tr=Nx_tr,
            Ny_tr=Ny_tr,
            Nz_tr=Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            checkpoint_interval=checkpoint_interval,
            results_dir=log_dir,
            loss_plot_name=dragon_name,
            optimizer=optimizer,
            restart=cfg.solver.restart_from_checkpoint,
            print_rate=cfg.solver.print_rate,
        )
        t0 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
        t1 = time.time()

    logger.info(f"solve took {(t1 - t0)} seconds")
    # jax.profiler.save_device_memory_profile("solve_poisson_over_dragon.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    log = {
        "phi": eval_phi.reshape((Nx_eval, Ny_eval, Nz_eval)),
        "U": sim_state.solution.reshape((Nx_eval, Ny_eval, Nz_eval)),
    }
    save_name = log_dir + "/" + f"{dragon_name}_{Nx_tr}_{gpu_cnt}"
    io.write_vtk_manual(eval_gstate, log, filename=save_name)


if __name__ == "__main__":
    solve_poisson_over_dragon()
