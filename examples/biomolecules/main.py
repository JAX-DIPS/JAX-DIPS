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

from examples.biomolecules.coefficients import (
    f_m_fn,
    f_p_fn,
    get_dirichlet_bc_fn,
    get_jump_conditions,
    get_psi_star,
    get_rho_fn,
    initial_value_fn,
    k_m_fn,
    k_p_fn,
    mu_m_fn,
    mu_p_fn,
    nonlinear_operator_m,
    nonlinear_operator_p,
)
from examples.biomolecules.free_energy import get_free_energy
from examples.biomolecules.geometry import get_initial_level_set_fn
from examples.biomolecules.load_pqr import base
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

    optim_dict = dict(cfg.solver.optim)
    model_dict = dict(cfg.model)

    ALGORITHM = cfg.solver.algorithm  # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = cfg.solver.switching_interval  # 0: no switching, 1: 10
    mgrad_over_pgrad_scalefactor = cfg.solver.mgrad_over_pgrad_scalefactor  # 0: no switching, 1: 10
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

    xmin = ymin = zmin = min((atom_locations.min(), (-atom_sigmas).min())) * 3  # length unit is l_tilde units
    xmax = ymax = zmax = max((atom_locations.max(), atom_sigmas.max())) * 3
    init_mesh_fn, coord_at = mesh.construct(3)
    # --------- Grid nodes for trainer
    dx = (xmax - xmin) / Nx_tr
    dy = (ymax - ymin) / Ny_tr
    dz = (zmax - zmin) / Nz_tr
    xc = jnp.linspace(xmin, xmax, Nx_tr, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny_tr, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz_tr, dtype=f32)
    gstate_tr = init_mesh_fn(xc, yc, zc)

    # --------- Grid nodes for level set
    xc = jnp.linspace(xmin, xmax, Nx_lvl, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny_lvl, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz_lvl, dtype=f32)
    gstate_lvl = init_mesh_fn(xc, yc, zc)

    # ----------  Evaluation Mesh for Visualization
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    ###########################################################
    dirichlet_bc_fn = get_dirichlet_bc_fn(atom_xyz_rad_chg)

    unperturbed_phi_fn = get_initial_level_set_fn(atom_xyz_rad_chg)
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    psi_star_fn, psi_star_vec_fn, grad_psi_star_fn, grad_psi_star_vec_fn = get_psi_star(atom_xyz_rad_chg)

    alpha_fn, beta_fn = get_jump_conditions(
        atom_xyz_rad_chg, psi_star_fn, phi_fn, gstate_tr.dx, gstate_tr.dy, gstate_tr.dz
    )

    ###########################################################
    if False:
        """Testing u_star, without solvent, only singular point charges"""
        psi_star = psi_star_vec_fn(eval_gstate.R)
        eval_phi = vmap(phi_fn)(eval_gstate.R)
        chg_density = vmap(f_m_fn)(eval_gstate.R)
        log = {"phi": eval_phi, "Ustar": psi_star, "rho": chg_density}
        io.write_vtk_manual(eval_gstate, log, filename=log_dir + "/biomolecules")
        pdb.set_trace()

    t0 = t1 = 0.0
    if cfg.solver.version == 0:
        logger.warning("this solver was deprecated")
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
        sim_state = init_fn(gstate_lvl.R)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        t0 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(
            gstate_lvl,
            eval_gstate,
            sim_state,
            algorithm=ALGORITHM,
            switching_interval=SWITCHING_INTERVAL,
            Nx_tr=Nx_tr,
            Ny_tr=Ny_tr,
            Nz_tr=Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )
        t1 = time.time()

    elif cfg.solver.version == 1:
        logger.warning("this solver was deprecated")
        optimizer = get_optimizer(
            optimizer_name=cfg.solver.optim.optimizer_name,
            scheduler_name=cfg.solver.optim.sched.scheduler_name,
            learning_rate=cfg.solver.optim.learning_rate,
            decay_rate=cfg.solver.optim.sched.decay_rate,
        )
        init_fn = trainer_poisson.setup(
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
            gstate=gstate_lvl,
            eval_gstate=eval_gstate,
            algorithm=ALGORITHM,
            switching_interval=SWITCHING_INTERVAL,
            Nx_tr=Nx_tr,
            Ny_tr=Ny_tr,
            Nz_tr=Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            checkpoint_interval=checkpoint_interval,
            currDir=log_dir,
            loss_plot_name=molecule_name,
            optimizer=optimizer,
        )
        t0 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
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
            lvl_gstate=gstate_lvl,
            tr_gstate=gstate_tr,
            eval_gstate=eval_gstate,
            algorithm=ALGORITHM,
            mgrad_over_pgrad_scalefactor=mgrad_over_pgrad_scalefactor,
            num_epochs=num_epochs,
            batch_size=batch_size,
            multi_gpu=multi_gpu,
            checkpoint_interval=checkpoint_interval,
            results_dir=log_dir,
            loss_plot_name=molecule_name,
            optimizer_dict=optim_dict,
            model_dict=model_dict,
            restart=cfg.solver.restart_from_checkpoint,
            restart_checkpoint_dir=cfg.solver.restart_checkpoint_dir,
            print_rate=cfg.solver.print_rate,
        )
        t0 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
        t1 = time.time()

    elapsed_time = t1 - t0

    # profiler.save_device_memory_profile("memory_biomolecule.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)

    rho_fn = get_rho_fn(atom_xyz_rad_chg)
    chg_density = vmap(rho_fn)(eval_gstate.R)

    psi_star = psi_star_vec_fn(eval_gstate.R)
    psi_hat = sim_state.solution

    def compose_psi_fn(r, psi_hat_r, psi_star_r):
        phi_at_r = phi_fn(r)
        return jnp.where(phi_at_r > 0, psi_hat_r, psi_hat_r + psi_star_r)

    psi_solution = vmap(compose_psi_fn, (0, 0, 0))(eval_gstate.R, psi_hat, psi_star)

    grad_u_jump = vmap(beta_fn)(eval_gstate.R)
    u_jump = vmap(alpha_fn)(eval_gstate.R)
    log = {
        "phi": eval_phi,
        "rho": chg_density,
        "U": psi_solution,
        "Uhat": psi_hat,
        "Ustar": psi_star,
        "jump": u_jump,
        "grad_jump": grad_u_jump,
    }
    save_name = log_dir + "/" + molecule_name
    io.write_vtk_manual(eval_gstate, log, filename=save_name)

    grad_psi_hat = sim_state.grad_solution
    grad_psi_star = grad_psi_star_vec_fn(eval_gstate.R)

    def get_epsilon_E_sq_field(r, g_hat_r, g_star_r):
        phi_at_r = phi_fn(r)
        return jnp.where(
            phi_at_r > 0,
            mu_p_fn(r) * jnp.dot(g_hat_r, g_hat_r),
            mu_m_fn(r) * jnp.dot(g_hat_r + g_star_r, g_hat_r + g_star_r),
        )

    epsilon_grad_psi_sq = vmap(get_epsilon_E_sq_field, (0, 0, 0))(eval_gstate.R, grad_psi_hat, grad_psi_star)

    def get_epsilon_E_coul_sq_field(r, g_star_r):
        phi_at_r = phi_fn(r)
        return jnp.where(
            phi_at_r > 0,
            mu_p_fn(r) * jnp.dot(g_star_r, g_star_r),
            mu_m_fn(r) * jnp.dot(g_star_r, g_star_r),
        )

    epsilon_grad_psi_star_sq = vmap(get_epsilon_E_coul_sq_field, (0, 0))(eval_gstate.R, grad_psi_star)
    epsilon_grad_psi_hat_sq = vmap(get_epsilon_E_coul_sq_field, (0, 0))(eval_gstate.R, grad_psi_hat)

    SFE, SFE_z = get_free_energy(
        eval_gstate,
        eval_phi,
        psi_hat,
        atom_xyz_rad_chg,
        epsilon_grad_psi_sq,
        psi_solution,
        epsilon_grad_psi_star_sq,
        epsilon_grad_psi_hat_sq,
    )
    train_grid_size = (xmax - xmin) / Nx_tr
    logger.info(f"Molecule = {file_name} \t training grid spacing (h_g) = {train_grid_size} (Angstrom)")
    logger.info(f"SFE_polar = {SFE} (kcal/mol) \t SFE_ionic = {SFE_z} (kcal/mol) \t => SFE = {SFE + SFE_z} (kcal/mol)")
    logger.info(f"Solver init() + solve() elapsed time = {elapsed_time} (sec)")

    data_file = log_dir + "/data_logs.txt"
    with open(data_file, "a") as f:
        result = f"{molecule_name} \t {train_grid_size} \t {SFE} \t {SFE_z} \t {SFE + SFE_z} \t {elapsed_time} \n"
        f.write(result)


@hydra.main(config_path="conf", config_name="biomolecule", version_base="1.1")
def main(cfg: DictConfig):
    logger.info("Starting the biomolecule training")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.experiment.kirkwood.enable:
        logger.info("Kirkwood experiment enabled")
        biomolecule_solvation_energy(
            cfg=cfg,
            file_name=cfg.experiment.kirkwood.protein_name,
            molecule_pqr_address=cfg.experiment.kirkwood.protein_path,
        )

    else:
        if cfg.experiment.single_protein.enable:
            logger.info("Single protein experiment enabled")
            biomolecule_solvation_energy(
                cfg=cfg,
                file_name=cfg.experiment.single_protein.protein_name,
                molecule_pqr_address=cfg.experiment.single_protein.protein_path,
            )

        elif cfg.experiment.multi_proteins.enable:
            logger.info("Multi protein experiment enabled")

            molecule_pqr_address = cfg.experiment.multi_proteins.proteins_dir
            tmp = glob.glob(currDir + "/" + molecule_pqr_address + "/*.pqr")
            molecules = [mol.split("/")[-1] for mol in tmp]

            gpu_count = cfg.experiment.multi_proteins.num_gpus_batching

            logger.info(f"Using {gpu_count} devices")
            gpu_ids = [str(i) for i in range(gpu_count)]

            process_pool = [
                mp.Process(
                    target=biomolecule_solvation_energy,
                    args=(cfg, molecules[i], molecule_pqr_address, gpu_ids[i % gpu_count]),
                )
                for i in range(len(molecules))
            ]

            mol_count = 0
            while mol_count < len(molecules):
                for i, gpu in enumerate(gpu_ids):
                    try:
                        process_pool[i + mol_count].start()
                    except IndexError:
                        pass

                for i, gpu in enumerate(gpu_ids):
                    try:
                        process_pool[i + mol_count].join()
                    except IndexError:
                        pass
                mol_count += gpu_count


if __name__ == "__main__":
    main()
