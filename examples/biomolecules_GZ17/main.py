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


This example is based on https://www.sciencedirect.com/science/article/pii/S0021999117306861

"""





from jax.config import config
config.update("jax_enable_x64", False)
import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import sys
currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  
    sys.path.append(rootDir)

from jax import (numpy as jnp, vmap)
import jax
import jax.profiler
import pdb
import numpy as onp

from src import io, trainer_poisson, mesh, level_set, poisson_solver_scalable
from src.jaxmd_modules.util import f32
from examples.biomolecules_GZ17.coefficients import *
from examples.biomolecules_GZ17.geometry import get_initial_level_set_fn
from examples.biomolecules_GZ17.load_pqr import base
from examples.biomolecules_GZ17.free_energy import get_free_energy



def biomolecule_solvation_energy(file_name = 'pdb:1ajj.pqr', molecule_pqr_address = 'pqr_input_mols', gpu_id=None):   
    
    if gpu_id==None:
      pass
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    ###########################################################
    
    num_epochs = 1000
    
    Nx_tr = Ny_tr = Nz_tr = 64                  # grid for training
    Nx = Ny = Nz = 256                           # grid for level-set
    Nx_eval = Ny_eval = Nz_eval = 256            # grid for visualization
    
    ALGORITHM = 0                                # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    multi_gpu = False
    checkpoint_interval = 500
    
    molecule_name = file_name.split('.pqr')[0]
    
    ###########################################################
    
    address = os.path.join(currDir, molecule_pqr_address)
    mol_base = base(address, file_name)
    num_atoms = len(mol_base.atoms['x'])
    print(f'\n number of atoms = {num_atoms} \n ')
    
    atom_locations = onp.stack([onp.array(mol_base.atoms['x']), 
                                onp.array(mol_base.atoms['y']), 
                                onp.array(mol_base.atoms['z'])
                                ], axis=-1)                                  # in Angstroms   
    atom_locations -= atom_locations.mean(axis=0)                            # recenter in the box
    sigma_i = onp.array(mol_base.atoms['R'])        
    sigma_s = 1.4                                  
    atom_sigmas = sigma_i + sigma_s                                          
    
    atom_charges = jnp.array(mol_base.atoms['q'])                            # partial charges, in units of electron charge e
    atom_xyz_rad_chg = jnp.concatenate((atom_locations, 
                                        atom_sigmas[..., jnp.newaxis], 
                                        atom_charges[..., jnp.newaxis]
                                        ), axis=1)
    
    ###########################################################
    
    xmin = ymin = zmin = atom_locations.min() * 3                             # length unit is l_tilde units
    xmax = ymax = zmax = atom_locations.max() * 3
    init_mesh_fn, coord_at = mesh.construct(3)

    # --------- Grid nodes for level set
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    gstate = init_mesh_fn(xc, yc, zc)
    
    #----------  Evaluation Mesh for Visualization
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    ###########################################################
    dirichlet_bc_fn = get_dirichlet_bc_fn(atom_xyz_rad_chg)
    
    unperturbed_phi_fn = get_initial_level_set_fn(atom_xyz_rad_chg)  
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)
    
    psi_star_fn, psi_star_vec_fn, grad_psi_star_fn, grad_psi_star_vec_fn = get_psi_star(atom_xyz_rad_chg)
    
    alpha_fn, beta_fn = get_jump_conditions(atom_xyz_rad_chg, psi_star_fn, phi_fn, eval_gstate.dx*0.01, eval_gstate.dy*0.01, eval_gstate.dz*0.01)
    
    ###########################################################
    
    if False:
      """ Testing u_star, without solvent, only singular point charges """
      psi_star = psi_star_vec_fn(eval_gstate.R)
      eval_phi = vmap(phi_fn)(eval_gstate.R)
      chg_density = vmap(f_m_fn)(eval_gstate.R)
      log = {'phi': eval_phi, 'Ustar': psi_star, 'rho': chg_density}
      io.write_vtk_manual(eval_gstate, log, filename=currDir + '/results/biomolecules')
      pdb.set_trace()
    
    
    if False:
      #-- v1 old code
      init_fn, solve_fn = poisson_solver_scalable.setup(initial_value_fn, 
                                                        dirichlet_bc_fn, 
                                                        phi_fn, 
                                                        mu_m_fn, 
                                                        mu_p_fn, 
                                                        k_m_fn, 
                                                        k_p_fn, 
                                                        f_m_fn, 
                                                        f_p_fn, 
                                                        alpha_fn, 
                                                        beta_fn)
      sim_state = init_fn(gstate.R)
      checkpoint_dir = os.path.join(currDir, 'checkpoints')
      sim_state, epoch_store, loss_epochs = solve_fn(gstate, eval_gstate, sim_state, algorithm=ALGORITHM, switching_interval=SWITCHING_INTERVAL,
                                                    Nx_tr=Nx_tr, Ny_tr=Ny_tr, Nz_tr=Nz_tr, num_epochs=num_epochs, multi_gpu=multi_gpu,
                                                    checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval)
      
    
    else:    
      #-- v2 new code
      init_fn = trainer_poisson.setup(initial_value_fn, 
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
                                      nonlinear_operator_p)
      
      sim_state, solve_fn = init_fn(gstate=gstate, 
                                    eval_gstate=eval_gstate, 
                                    algorithm=ALGORITHM, 
                                    switching_interval=SWITCHING_INTERVAL, 
                                    Nx_tr=Nx_tr, 
                                    Ny_tr=Ny_tr, 
                                    Nz_tr=Nz_tr, 
                                    num_epochs=num_epochs, 
                                    multi_gpu=multi_gpu, 
                                    checkpoint_interval=checkpoint_interval,
                                    currDir=currDir + '/results/',
                                    loss_plot_name=molecule_name)
      
      sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
 
    
    # jax.profiler.save_device_memory_profile("memory_biomolecule.prof")


    eval_phi = vmap(phi_fn)(eval_gstate.R)
    
    
    rho_fn = get_rho_fn(atom_xyz_rad_chg)
    chg_density = vmap(rho_fn)(eval_gstate.R)
    
    
    
    psi_star = psi_star_vec_fn(eval_gstate.R)
    psi_hat = sim_state.solution
    def compose_psi_fn(r, psi_hat_r, psi_star_r):
      phi_at_r = phi_fn(r)
      return jnp.where(phi_at_r>0, psi_hat_r, psi_hat_r + psi_star_r)
    psi_solution = vmap(compose_psi_fn, (0, 0, 0))(eval_gstate.R, psi_hat, psi_star)
    
    
    grad_u_jump = vmap(beta_fn)(eval_gstate.R)
    u_jump = vmap(alpha_fn)(eval_gstate.R)
    log = {'phi'  : eval_phi,
           'rho'  : chg_density, 
           'U'    : psi_solution,
           'Uhat' : psi_hat, 
           'Ustar': psi_star,
           'jump' : u_jump,
           'grad_jump': grad_u_jump
           }
    save_name = currDir + '/results/' + molecule_name
    io.write_vtk_manual(eval_gstate, log, filename=save_name)
    
    
    grad_psi_hat = sim_state.grad_solution
    grad_psi_star = grad_psi_star_vec_fn(eval_gstate.R)
    def get_epsilon_E_sq_field(r, g_hat_r, g_star_r):
      phi_at_r = phi_fn(r)
      return jnp.where(phi_at_r>0, mu_p_fn(r) * jnp.dot(g_hat_r, g_hat_r), mu_m_fn(r) * jnp.dot(g_hat_r + g_star_r, g_hat_r + g_star_r) )
    epsilon_grad_psi_sq = vmap(get_epsilon_E_sq_field, (0, 0, 0))(eval_gstate.R, grad_psi_hat, grad_psi_star)
    
    def get_epsilon_E_coul_sq_field(r, g_star_r):
      phi_at_r = phi_fn(r)
      return jnp.where(phi_at_r>0, mu_p_fn(r) * jnp.dot(g_star_r, g_star_r), mu_m_fn(r) * jnp.dot(g_star_r, g_star_r) )
    epsilon_grad_psi_star_sq = vmap(get_epsilon_E_coul_sq_field, (0, 0))(eval_gstate.R, grad_psi_star)
    epsilon_grad_psi_hat_sq = vmap(get_epsilon_E_coul_sq_field, (0, 0))(eval_gstate.R, grad_psi_hat)
    
    SFE = get_free_energy(eval_gstate, eval_phi, psi_hat, atom_xyz_rad_chg, epsilon_grad_psi_sq, psi_solution, epsilon_grad_psi_star_sq, epsilon_grad_psi_hat_sq)
    train_grid_size = (xmax - xmin)/Nx_tr
    print(f"Molecule = {file_name} \t grid spacing (h_g) = {train_grid_size} (Angstrom) \t Solvation Free Energy = {SFE} (kcal/mol/e_C)")
    
    data_file = currDir + '/results/data_logs.txt'
    with open(data_file, "a") as f:
      result = f"{molecule_name} \t {train_grid_size} \t {SFE} \n"
      f.write(result)
   

    










if __name__ == "__main__":
  
    if True:
        """ For testing only run 1 molecule """
        biomolecule_solvation_energy()
        
    else:
        """ For final evaluation, run over all molecules in parallel """
        import multiprocessing as mp
        import glob
        
        molecule_pqr_address = 'pqr_input_mols'
        tmp = glob.glob(currDir + '/' + molecule_pqr_address + '/*.pqr')
        molecules = [mol.split('/')[-1] for mol in tmp]
      
        gpu_count = 3 
        
        print(f"Using {gpu_count} devices!")
        gpu_ids = [str(i) for i in range(gpu_count)] 
        
        process_pool = [mp.Process(target=biomolecule_solvation_energy, args=(molecules[i], molecule_pqr_address, gpu_ids[ i % gpu_count ],)) for i in range(len(molecules)) ]
        
        
        mol_count = 0
        while mol_count < len(molecules):
            for i, gpu in enumerate(gpu_ids):
                try:
                    process_pool[i + mol_count].start()
                except:
                    pass
              
            for i, gpu in enumerate(gpu_ids):
                try:
                    process_pool[i + mol_count].join()
                except:
                    pass
            mol_count += gpu_count
    
    
 
