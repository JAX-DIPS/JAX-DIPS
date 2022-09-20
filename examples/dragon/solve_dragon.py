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
from src import io, poisson_solver_scalable, mesh, level_set, interpolate
from src.jaxmd_modules.util import f32, i32
from jax import (jit, numpy as jnp, vmap, grad, lax, random)
import jax
import jax.profiler
import pdb
import time
import os
import sys
import numpy as onp
from functools import partial

COMPILE_BACKEND = 'gpu'
custom_jit = partial(jit, backend=COMPILE_BACKEND)

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def poisson_solver_with_jump_complex():
    ALGORITHM = 0                          # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3
    Nx_tr = Ny_tr = Nz_tr = 128
    multi_gpu = False
    num_epochs = 50


    dim = i32(3)
    init_mesh_fn, coord_at = mesh.construct(dim)
    
    
    # --------- Grid nodes for level set
    xmin = 0.118; xmax = 0.353
    ymin = 0.088; ymax = 0.263
    zmin = 0.0615; zmax = 0.1835
    Nx = 236; Ny = 176; Nz = 123
    
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    dragon_host = onp.loadtxt(currDir + '/dragonian.csv')
    dragon = jnp.array(dragon_host)
    # log = {'phi': dragon}
    # io.write_vtk_manual(gstate, log, filename='results/dragon')
    
    

    #----------  Evaluation Mesh for Visualization
    Nx_eval = Nx
    Ny_eval = Ny
    Nz_eval = Nz
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)    
    
    
    # -- Initialize the Dragons in the BOX
    num_stars_x = num_stars_y = num_stars_z = 4      # Ensure you are solving your system
    scale = 0.35                                     # This is for proper separation between stars

    phi_fn = interpolate.nonoscillatory_quadratic_interpolation_per_point(dragon, gstate)
    
    
    
    
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
        fm   = -1.0 * mu_m_fn(r) * (-7.0 * jnp.sin(2.0*x) * jnp.cos(2.0*y) * jnp.exp(z)) +\
               -4*jnp.pi*jnp.cos(z)*jnp.cos(4*jnp.pi*x) * 2*jnp.cos(2*x)*jnp.cos(2*y)*jnp.exp(z)   +\
               -4*jnp.pi*jnp.cos(z)*jnp.cos(4*jnp.pi*y) * (-2)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.exp(z) +\
                2*jnp.cos(2*jnp.pi*(x+y))*jnp.sin(2*jnp.pi*(x-y))*jnp.sin(z) * jnp.sin(2*x)*jnp.cos(2*y)*jnp.exp(z)
        
        return fm

    @custom_jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        f_p = -1.0 * ( 
            ( 16*((y-x)/3)**5 - 20*((y-x)/3)**3 + 5*(y-x)/3 ) * (-2)*jnp.cos(z) / (x+y+3)**2 +\
             2*( 16*5*4*(1.0/9.0)*((y-x)/3)**3 - 20*3*2*(1.0/9.0)*((y-x)/3) ) * jnp.log(x+y+3)*jnp.cos(z) +\
            -1*( 16*((y-x)/3)**5 - 20*((y-x)/3)**3 + 5*((y-x)/3) ) * jnp.log(x+y+3)*jnp.cos(z)
        )
        return f_p


    

    init_fn, solve_fn = poisson_solver_scalable.setup(initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)
   
    t1 = time.time()

    
    sim_state, epoch_store, loss_epochs = solve_fn(gstate, eval_gstate, sim_state, algorithm=ALGORITHM, switching_interval=SWITCHING_INTERVAL, Nx_tr=Nx_tr, Ny_tr=Ny_tr, Nz_tr=Nz_tr, num_epochs=num_epochs, multi_gpu=multi_gpu)
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver_scalable.prof")

    
    eval_phi = vmap(phi_fn)(eval_gstate.R)
    log = {'phi': eval_phi, 'U': sim_state.solution}
    io.write_vtk_manual(eval_gstate, log, filename=currDir + 'results/dragon')
    




if __name__ == "__main__":
    poisson_solver_with_jump_complex()
