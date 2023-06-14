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
import logging
import os
import sys
import time
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import jax
import jax.profiler
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import vmap
from jax.config import config

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import mesh
from jax_dips.geometry import level_set
from jax_dips.solvers.optimizers import get_optimizer
from jax_dips.solvers.poisson import trainer
from jax_dips.utils import io
from tests.confs.experiment_configs import no_jump, sphere, star

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


@hydra.main(config_path="confs", config_name="poisson", version_base="1.1")
def test_poisson(cfg: DictConfig):
    logger.info(f"Starting {__file__}")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.experiment.sphere:
        logger.info("Performing sphere experiment...\n")
        poisson_solve(
            cfg,
            test_name="sphere",
            exp_fn=sphere,
        )
    if cfg.experiment.star:
        logger.info("Performing star experiment...\n")
        poisson_solve(
            cfg,
            test_name="star",
            exp_fn=star,
        )
    if cfg.experiment.no_jump:
        logger.info("Performing bulk/no jump experiment...\n")
        poisson_solve(
            cfg,
            test_name="no_jump",
            exp_fn=no_jump,
        )


def create_dirs(
    results_path: str,
    test_name: str,
):
    results_path = os.path.join(results_path, test_name)
    os.path.exists(results_path) or os.makedirs(results_path)
    return results_path


def poisson_solve(
    cfg: DictConfig,
    test_name: str,
    exp_fn: object,
):
    results_path = create_dirs(results_path=cfg.experiment.results_path, test_name=test_name)
    checkpoint_dir = os.path.join(results_path, "checkpoints")
    checkpoint_interval = cfg.experiment.logging.checkpoint_interval

    algorithm = cfg.solver.algorithm
    switching_interval = cfg.solver.switching_interval
    multi_gpu = cfg.solver.multi_gpu
    num_epochs = cfg.solver.num_epochs

    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    init_mesh_fn, coord_at = mesh.construct(dim)

    Nx_tr = cfg.solver.Nx_tr
    Ny_tr = cfg.solver.Nx_tr
    Nz_tr = cfg.solver.Nx_tr
    batch_size = min(64 * 64 * 32, Nx_tr * Ny_tr * Nz_tr)

    # --------- Grid nodes for level set
    Nx = cfg.gridstates.Nx
    Ny = cfg.gridstates.Ny
    Nz = cfg.gridstates.Nz
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    # ----------  Evaluation Mesh for Visualization
    Nx_eval = cfg.gridstates.Nx_eval
    Ny_eval = cfg.gridstates.Ny_eval
    Nz_eval = cfg.gridstates.Nz_eval
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    # ----------  Set up current experiment
    (
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
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    ) = exp_fn()

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
    )
    sim_state, solve_fn = init_fn(
        gstate=gstate,
        eval_gstate=eval_gstate,
        algorithm=algorithm,
        switching_interval=switching_interval,
        Nx_tr=Nx_tr,
        Ny_tr=Ny_tr,
        Nz_tr=Nz_tr,
        num_epochs=num_epochs,
        multi_gpu=multi_gpu,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_path,
        loss_plot_name=test_name,
        optimizer=get_optimizer(),
        restart=cfg.solver.restart_from_checkpoint,
        print_rate=cfg.solver.print_rate,
    )
    t1 = time.time()
    sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
    t2 = time.time()

    logger.info(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile(f"{results_path}/memory_poisson_test_{test_name}.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    exact_sol = vmap(evaluate_exact_solution_fn)(eval_gstate.R)
    error = sim_state.solution - exact_sol
    log = {
        "phi": eval_phi,
        "U": sim_state.solution,
        "U_exact": exact_sol,
        "U-U_exact": error,
    }
    io.write_vtk_manual(
        eval_gstate,
        log,
        filename=os.path.join(results_path, test_name),
    )

    # log = {
    #     'phi': sim_state.phi,
    #     'U': sim_state.solution,
    #     'U_exact': exact_sol,
    #     'U-U_exact': sim_state.solution - exact_sol,
    #     'alpha': sim_state.alpha,
    #     'beta': sim_state.beta,
    #     'mu_m': sim_state.mu_m,
    #     'mu_p': sim_state.mu_p,
    #     'f_m': sim_state.f_m,
    #     'f_p': sim_state.f_p,
    #     'grad_um_x': sim_state.grad_solution[0][:,0],
    #     'grad_um_y': sim_state.grad_solution[0][:,1],
    #     'grad_um_z': sim_state.grad_solution[0][:,2],
    #     'grad_up_x': sim_state.grad_solution[1][:,0],
    #     'grad_up_y': sim_state.grad_solution[1][:,1],
    #     'grad_up_z': sim_state.grad_solution[1][:,2],
    #     'grad_um_n': sim_state.grad_normal_solution[0],
    #     'grad_up_n': sim_state.grad_normal_solution[1]
    # }
    # io.write_vtk_manual(gstate, log)

    L_inf_err = abs(sim_state.solution - exact_sol).max()
    rms_err = jnp.square(sim_state.solution - exact_sol).mean() ** 0.5

    logger.info("Solution error computed everywhere in the domain:\n")
    logger.info(f"\t L_inf error \t=\t {L_inf_err}")
    logger.info(f"\t RMSD error \t=\t {rms_err}\n")
    logger.info(f"Experiment {test_name} completed! \n")

    """
    MASK the solution over sphere only
    """
    """
    logger.info("\n GRADIENT ERROR\n")

    grad_um = sim_state.grad_solution[0].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up = sim_state.grad_solution[1].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_exact = vmap(grad(exact_sol_m_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up_exact = vmap(grad(exact_sol_p_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    mask_m = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 #-0.5*dx
    err_x_m = abs(grad_um[mask_m][:,0] - grad_um_exact[mask_m][:,0]).max()
    err_y_m = abs(grad_um[mask_m][:,1] - grad_um_exact[mask_m][:,1]).max()
    err_z_m = abs(grad_um[mask_m][:,2] - grad_um_exact[mask_m][:,2]).max()

    mask_p = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 #0.5*dx
    err_x_p = abs(grad_up[mask_p][:,0] - grad_up_exact[mask_p][:,0]).max()
    err_y_p = abs(grad_up[mask_p][:,1] - grad_up_exact[mask_p][:,1]).max()
    err_z_p = abs(grad_up[mask_p][:,2] - grad_up_exact[mask_p][:,2]).max()

    logger.info(f"L_inf errors in grad u in Omega_minus x: {err_x_m}, \t y: {err_y_m}, \t z: {err_z_m}")
    logger.info(f"L_inf errors in grad u in Omega_plus  x: {err_x_p}, \t y: {err_y_p}, \t z: {err_z_p}")



    #--- normal gradients over interface
    normal_fn = grad(phi_fn)
    normal_vec = vmap(normal_fn)(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_n = sim_state.grad_normal_solution[0].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]
    grad_up_n = sim_state.grad_normal_solution[1].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]

    mask_i_m = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 )
    mask_i_p = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 )

    grad_um_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_um_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))
    grad_up_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_up_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))

    err_um_n = abs(grad_um_n - grad_um_n_exact)[mask_i_m].max()
    err_up_n = abs(grad_up_n - grad_up_n_exact)[mask_i_p].max()


    logger.info(f"L_inf error in normal grad u on interface minus: {err_um_n} \t plus: {err_up_n}")

    #----
    assert L_inf_err<0.2

    """


if __name__ == "__main__":
    test_poisson()
