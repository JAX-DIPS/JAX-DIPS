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

This example is based on https://www.sciencedirect.com/science/article/pii/S0021999117306861

"""
import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:
    sys.path.append(rootDir)

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
import glob
import multiprocessing as mp
import pdb
import time

import numpy as onp
from jax.config import config

config.update("jax_enable_x64", False)
config.update("jax_debug_nans", False)
from jax import numpy as jnp
from jax import profiler, vmap

from examples.benchmark_LPBE.coefficients import (
    f_m_fn,
    f_p_fn,
    get_dirichlet_bc_fn,
    get_jump_conditions,
    get_g_dg_fns,
    get_rho_fn,
    initial_value_fn,
    k_m_fn,
    k_p_fn,
    mu_m_fn,
    mu_p_fn,
    nonlinear_operator_m,
    nonlinear_operator_p,
    get_exact_sol_fns,
)
from examples.benchmark_LPBE.geometry import get_initial_level_set_fn
from examples.benchmark_LPBE.load_pqr import base
from examples.benchmark_LPBE.inn.inn_data_sampler import INNSphereData
from jax_dips._jaxmd_modules.util import f32
from jax_dips.domain import mesh
from jax_dips.geometry import level_set
from jax_dips.solvers.optimizers import get_optimizer
from jax_dips.solvers.poisson import trainer
from jax_dips.solvers.poisson.deprecated import poisson_solver_scalable, trainer_poisson
from jax_dips.utils import io


def biomolecule_solvation_energy(
    cfg: DictConfig, file_name: str = "pdb:1ajj.pqr", molecule_pqr_address: str = "pqr_input_mols", gpu_id: str = ""
) -> None:
    if gpu_id == "":
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    ###########################################################

    num_epochs = cfg.solver.num_epochs
    batch_size = cfg.solver.batch_size

    Nx_tr = cfg.solver.Nx_tr  # grid for training
    Ny_tr = cfg.solver.Ny_tr  # grid for training
    Nz_tr = cfg.solver.Nz_tr  # grid for training
    Nx_lvl = cfg.gridstates.Nx_lvl  # grid for level-set
    Ny_lvl = cfg.gridstates.Ny_lvl  # grid for level-set
    Nz_lvl = cfg.gridstates.Nz_lvl  # grid for level-set
    Nx_eval = cfg.gridstates.Nx_eval  # grid for evaluation/visualization
    Ny_eval = cfg.gridstates.Ny_eval  # grid for evaluation/visualization
    Nz_eval = cfg.gridstates.Nz_eval  # grid for evaluation/visualization

    optimizer = get_optimizer(
        optimizer_name=cfg.solver.optim.optimizer_name,
        scheduler_name=cfg.solver.sched.scheduler_name,
        learning_rate=cfg.solver.optim.learning_rate,
        decay_rate=cfg.solver.sched.decay_rate,
    )

    ALGORITHM = cfg.solver.algorithm  # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = cfg.solver.switching_interval  # 0: no switching, 1: 10
    multi_gpu = cfg.solver.multi_gpu
    checkpoint_interval = cfg.experiment.logging.checkpoint_interval
    log_dir = cfg.experiment.logging.log_dir

    molecule_name = file_name.split(".pqr")[0]

    ###########################################################

    address = os.path.join(currDir, molecule_pqr_address)
    mol_base = base(address, file_name)
    num_atoms = len(mol_base.atoms["x"])
    logger.info(f"number of atoms = {num_atoms}")

    atom_locations = onp.stack(
        [
            onp.array(mol_base.atoms["x"]),
            onp.array(mol_base.atoms["y"]),
            onp.array(mol_base.atoms["z"]),
        ],
        axis=-1,
    )  # in Angstroms
    atom_locations -= atom_locations.mean(axis=0)  # recenter in the box
    sigma_i = onp.array(mol_base.atoms["R"])
    sigma_s = 0.0
    atom_sigmas = sigma_i + sigma_s

    atom_charges = jnp.array(mol_base.atoms["q"])  # partial charges, in units of electron charge e
    atom_xyz_rad_chg = jnp.concatenate(
        (atom_locations, atom_sigmas[..., jnp.newaxis], atom_charges[..., jnp.newaxis]),
        axis=1,
    )

    ###########################################################

    xmin = ymin = zmin = -2.5  # min((atom_locations.min(), (-atom_sigmas).min())) * 3
    xmax = ymax = zmax = 2.5  # max((atom_locations.max(), atom_sigmas.max())) * 3
    init_mesh_fn, coord_at = mesh.construct(3)

    # --------- GSTATE for level set
    xc = jnp.linspace(xmin, xmax, Nx_lvl, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny_lvl, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz_lvl, dtype=f32)
    gstate_lvl = init_mesh_fn(xc, yc, zc)

    # --------- GSTATE for trainer (discretization), using INN
    inn = INNSphereData(
        sigma=atom_sigmas[0],  # sphere radius
        L=atom_locations[0],  # sphere center coord
        box=[xmin, xmax, ymin, ymax, zmin, zmax],  # box dimensions
        device="cuda",
        train_out=1000,  # points outside/positive domain; originally was 2000
        train_inner=1000,  # points inside/negative domain; originally was 100
        train_boundary=1000,  # points on the boundary; originally was 1000
        train_gamma=1000,  # points on interface; originally was 200
    )
    xc = inn.x
    yc = inn.y
    zc = inn.z
    dx = (xmax - xmin) / Nx_tr
    dy = (ymax - ymin) / Ny_tr
    dz = (zmax - zmin) / Nz_tr
    gstate_tr = mesh.init_gstate_3d_manually(
        xc,
        yc,
        zc,
        dx,
        dy,
        dz,
        inn.R_xmin,
        inn.R_xmax,
        inn.R_ymin,
        inn.R_ymax,
        inn.R_zmin,
        inn.R_zmax,
    )

    # ----------  GSTATE Evaluation & Visualization
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    ###########################################################
    dirichlet_bc_fn = get_dirichlet_bc_fn(atom_xyz_rad_chg)

    unperturbed_phi_fn = get_initial_level_set_fn(atom_xyz_rad_chg)
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    g_fn, g_vec_fn, grad_g_fn, grad_g_vec_fn = get_g_dg_fns(atom_xyz_rad_chg)

    alpha_fn, beta_fn = get_jump_conditions(
        atom_xyz_rad_chg, g_fn, phi_fn, gstate_lvl.dx, gstate_lvl.dy, gstate_lvl.dz
    )

    ###########################################################
    if False:
        """Testing u_star, without solvent, only singular point charges"""
        psi_star = g_vec_fn(eval_gstate.R)
        eval_phi = vmap(phi_fn)(eval_gstate.R)
        chg_density = vmap(f_m_fn)(eval_gstate.R)
        log = {"phi": eval_phi, "Ustar": psi_star, "rho": chg_density}
        io.write_vtk_manual(eval_gstate, log, filename=log_dir + "/benchmark_lpbe")
        pdb.set_trace()

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
        lvl_gstate=gstate_lvl,
        tr_gstate=gstate_tr,
        eval_gstate=eval_gstate,
        algorithm=ALGORITHM,
        switching_interval=SWITCHING_INTERVAL,
        num_epochs=num_epochs,
        batch_size=batch_size,
        multi_gpu=multi_gpu,
        checkpoint_interval=checkpoint_interval,
        results_dir=log_dir,
        loss_plot_name=molecule_name,
        optimizer=optimizer,
        restart=cfg.solver.restart_from_checkpoint,
        print_rate=cfg.solver.print_rate,
    )
    t0 = time.time()
    sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
    t1 = time.time()

    elapsed_time = t1 - t0

    # get learned solution
    sim_sol = sim_state.solution

    # get exact solution for comparison
    exact_sol_m_fn, exact_sol_p_fn = get_exact_sol_fns(atom_xyz_rad_chg)

    def compose_exact_sol_fn(r):
        phi_at_r = phi_fn(r)
        return jnp.where(phi_at_r >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    exact_sol = vmap(compose_exact_sol_fn, (0,))(eval_gstate.R)

    # postprocessing
    error = exact_sol - sim_sol
    L_inf = jnp.max(jnp.abs(error))
    L_2 = jnp.mean(jnp.square(error))

    # L_infty
    L_inf = jnp.max(jnp.abs(sim_sol - exact_sol))
    # rela L_2
    L2_rel_loss = jnp.sqrt(((sim_sol - exact_sol) ** 2).sum() / (exact_sol**2).sum())
    logger.info(f"Accuracy measurements = L_inf : {L_inf} \t Rel. L_2 : {L2_rel_loss}")

    # visualization

    rho_fn = get_rho_fn(atom_xyz_rad_chg)
    chg_density = vmap(rho_fn)(eval_gstate.R)
    psi_star = g_vec_fn(eval_gstate.R)

    eval_phi = vmap(phi_fn)(eval_gstate.R)

    grad_u_jump = vmap(beta_fn)(eval_gstate.R)
    u_jump = vmap(alpha_fn)(eval_gstate.R)
    log = {
        "phi": eval_phi,
        "rho": chg_density,
        "sol": sim_sol,
        "exact_sol": exact_sol,
        "error": error,
        "g_r": psi_star,
        "jump": u_jump,
        "grad_jump": grad_u_jump,
    }
    save_name = log_dir + "/" + molecule_name
    io.write_vtk_manual(eval_gstate, log, filename=save_name)

    train_grid_size = (xmax - xmin) / Nx_tr
    logger.info(f"Molecule = {file_name} \t training grid spacing (h_g) = {train_grid_size}")
    logger.info(f"Solver init() + solve() elapsed time = {elapsed_time} (sec)")
    logger.info("done!")


@hydra.main(config_path="conf", config_name="lpbe", version_base="1.1")
def main(cfg: DictConfig):
    logger.info("Starting the linear Poisson-Boltzmann Equation training")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.experiment.sphere.enable:
        logger.info("sphere experiment enabled")
        biomolecule_solvation_energy(
            cfg=cfg,
            file_name=cfg.experiment.sphere.protein_name,
            molecule_pqr_address=cfg.experiment.sphere.protein_path,
        )

    if cfg.experiment.double_sphere.enable:
        logger.info("double_sphere experiment enabled")
        biomolecule_solvation_energy(
            cfg=cfg,
            file_name=cfg.experiment.double_sphere.protein_name,
            molecule_pqr_address=cfg.experiment.double_sphere.protein_path,
        )

    if cfg.experiment.molecule.enable:
        logger.info("molecule experiment enabled")

        biomolecule_solvation_energy(
            cfg=cfg,
            file_name=cfg.experiment.molecule.protein_name,
            molecule_pqr_address=cfg.experiment.molecule.protein_path,
        )


if __name__ == "__main__":
    main()
