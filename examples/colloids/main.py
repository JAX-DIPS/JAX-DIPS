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
    
from jax import (jit, numpy as jnp)
import jax
import jax.profiler
from functools import partial
import pdb
import time

from src import io, solver_advection, trainer_poisson_advection, mesh, level_set, interpolate
from src.jaxmd_modules.util import f32, i32
from src.jaxmd_modules import dataclasses
from examples.colloids.coefficients import *
from examples.colloids.geometry import get_initial_level_set_fn






def poisson_advection_simulation():
    
    ###########################################################
    
    num_epochs = 50
    simulation_steps = 5
    tf = f32(2 * jnp.pi)
    Nx_tr = Ny_tr = Nz_tr = 8                    # grid for training
    Nx_eval = Ny_eval = Nz_eval = 128            # grid for visualization
    Nx = Ny = Nz = 128                           # grid for level-set
    
    ALGORITHM = 0                                # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    multi_gpu = False
    checkpoint_interval = 1000
        
    ###########################################################
    
    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    init_mesh_fn, coord_at = mesh.construct(dim)

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

    R = gstate.R
    dx = xc[1] - xc[0]
    dt = dx * f32(0.9)
    # simulation_steps = i32(tf / dt) + 1
    
    unperturbed_phi_fn = get_initial_level_set_fn()
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    ###########################################################

    """ Solve Poisson """
    init_fn, solve_fn = trainer_poisson_advection.setup(initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)
    solve_poisson_fn = partial(solve_fn, gstate=gstate, eval_gstate=eval_gstate, algorithm=ALGORITHM, switching_interval=SWITCHING_INTERVAL, Nx_tr=Nx_tr, Ny_tr=Ny_tr, Nz_tr=Nz_tr, num_epochs=num_epochs, multi_gpu=multi_gpu, checkpoint_interval=checkpoint_interval)
    
    """ Advect Level Set """
    adv_init_fn, adv_apply_fn, adv_reinitialize_fn, adv_reinitialized_advect_fn = solver_advection.level_set(phi_fn, dt)
     
    
    def get_velocity_fn(vels):
        interp_vel = interpolate.vec_multilinear_interpolation(vels, eval_gstate)
        velocity_fn = lambda R, t: interp_vel(R)
        return velocity_fn
    
    dummy_vels = jnp.zeros_like(eval_gstate.R)
    velocity_fn = get_velocity_fn(dummy_vels) 
    adv_sim_state = adv_init_fn(velocity_fn, R)
    
    

    @jit
    def step_levelset(sim_state, adv_sim_state, gstate, time_):
        velocity_fn = get_velocity_fn(sim_state.grad_solution)
        adv_sim_state = adv_reinitialize_fn(adv_sim_state, gstate)
        adv_sim_state = adv_apply_fn(velocity_fn, adv_sim_state, gstate, time_)
        return dataclasses.replace(sim_state, phi=adv_sim_state.phi)
        
        
        
        
    t1 = time.time()
    time_ = 0.0

    for step in range(simulation_steps):
        print(f"solving at {step}")
        sim_state, epoch_store, loss_epochs = solve_poisson_fn(sim_state=sim_state)
        print(f"advecting at {step}")
        sim_state = step_levelset(sim_state, adv_sim_state, gstate, time_)
        time_ += dt
    
    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
 
    
    jax.profiler.save_device_memory_profile("memory_trainer_poisson_advection.prof")


    eval_phi = adv_sim_state.phi #vmap(phi_fn)(eval_gstate.R)
   
    log = {'phi': eval_phi, 
           'U': sim_state.solution, 
           'dU_dx': sim_state.grad_solution[:,0],
           'dU_dy': sim_state.grad_solution[:,1],
           'dU_dz': sim_state.grad_solution[:,2],
           'dU_dn': sim_state.grad_normal_solution
           }
    io.write_vtk_manual(eval_gstate, log, filename=currDir + '/results/colloids')
    
    pdb.set_trace()





if __name__ == "__main__":
    poisson_advection_simulation()
