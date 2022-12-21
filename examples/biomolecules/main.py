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

from src import io, trainer_poisson, mesh, level_set
from src.jaxmd_modules.util import f32, i32
from examples.biomolecules.coefficients import *
from examples.biomolecules.geometry import get_initial_level_set_fn
from examples.biomolecules.load_pqr import base




def biomolecule_solvation_energy():
    
    ###########################################################
    
    num_epochs = 50
    
    Nx_tr = Ny_tr = Nz_tr = 8                    # grid for training
    Nx = Ny = Nz = 512                           # grid for level-set
    Nx_eval = Ny_eval = Nz_eval = 128            # grid for visualization
    
    ALGORITHM = 0                                # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    multi_gpu = False
    checkpoint_interval = 1000
    
    
    file_name = 'pdb:1ajj.pqr'                                                      # change the name of the molecule
    
    ###########################################################
    
    address = os.path.join(currDir, 'pqr_input_mols')
    mol_base = base(address, file_name)
    num_atoms = len(mol_base.atoms['x'])
    print(f'\n number of atoms = {num_atoms} \n ')
    
    atom_locations = onp.stack([onp.array(mol_base.atoms['x']), 
                                onp.array(mol_base.atoms['y']), 
                                onp.array(mol_base.atoms['z'])
                                ], axis=-1) * Angstroms_to_nm_coeff                 # was in Angstroms, converted to nm   
    
    sigma_i = onp.array(mol_base.atoms['R']) * Angstroms_to_nm_coeff                # was Angstroms, converted to nm
    sigma_s = 0.65 * Angstroms_to_nm_coeff                                          # was Angstroms, converted to nm
    atom_sigmas = sigma_i + sigma_s                                                 # is in nm
    atom_charges = jnp.array(mol_base.atoms['q'])                                   # partial charges, in units of electron charge e
    atom_xyz_rad_chg = jnp.concatenate((atom_locations, 
                                        atom_sigmas[..., jnp.newaxis], 
                                        atom_charges[..., jnp.newaxis]
                                        ), axis=1)
    
    unperturbed_phi_fn = get_initial_level_set_fn(atom_xyz_rad_chg)  
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)
    
    f_m_fn = get_f_m_fn(atom_xyz_rad_chg)
    
    ###########################################################
    
    xmin = ymin = zmin = atom_locations.min() * 3                                   # length unit is nm
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
                                  currDir=currDir)
    
    sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
 
    
    jax.profiler.save_device_memory_profile("memory_biomolecule.prof")


    eval_phi = vmap(phi_fn)(eval_gstate.R)
    chg_density = vmap(f_m_fn)(eval_gstate.R)
   
    log = {'phi'  : eval_phi, 
           'U'    : sim_state.solution, 
           'dU_dx': sim_state.grad_solution[:,0],
           'dU_dy': sim_state.grad_solution[:,1],
           'dU_dz': sim_state.grad_solution[:,2],
           'dU_dn': sim_state.grad_normal_solution,
           'rho'  : chg_density
           }
    io.write_vtk_manual(eval_gstate, log, filename=currDir + '/results/biomolecules')
    
    pdb.set_trace()





if __name__ == "__main__":
    biomolecule_solvation_energy()
