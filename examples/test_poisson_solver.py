import os, sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path: # add parent dir to paths
    sys.path.append(rootDir)
import time
import pdb
from functools import partial

import jax
import jax.profiler
from jax import (jit, random, lax, numpy as jnp, vmap)
from src.util import f32, i32
from src import io, poisson_solver, mesh, level_set, simulate_fields, interpolate, space
from jax.config import config
config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'

key = random.PRNGKey(0)
dim = i32(3)
xmin = ymin = zmin = f32(-2.0)
xmax = ymax = zmax = f32(2.0)
box_size = xmax - xmin
Nx = i32(128)
Ny = i32(128)
Nz = i32(128)
dimension = i32(3)


#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
dx = xc[1] - xc[0]


#---------------
# Create helper functions to define a periodic box of some size.
init_mesh_fn, coord_at = mesh.construct(dim)
gstate = init_mesh_fn(xc, yc, zc)
R = gstate.R
sample_pnt = coord_at(gstate, [1,1,1])
displacement_fn, shift_fn = space.periodic(box_size)

#-- define velocity field as gradient of a scalar field
@jit
def velocity_fn(r, time_=0.0):
    """
    Velocity field to advect the level-set function
    """
    x = r[0]; y = r[1]; z = r[2]
    return jnp.array([0.0, 0.0, 0.0], dtype=f32)
velocity_fn = vmap(velocity_fn, (0,None))


def phi_fn(r):
    """
    Level-set function for the interface
    """
    x = r[0]; y = r[1]; z = r[2]
    return jnp.sqrt(x**2 + (y)**2 + z**2) - 0.5


def mu_m_fn(r):
    """
    Diffusion coefficient function in $\Omega^-$
    """
    return 1.0

def mu_p_fn(r):
    """
    Diffusion coefficient function in $\Omega^+$
    """
    return 2.0

def k_m_fn(r):
    """
    Linear term function in $\Omega^-$
    """
    return 0.0

def k_p_fn(r):
    """
    Linear term function in $\Omega^+$
    """
    return 1.0

def initial_value_fn(r):
    x = r[0]; y = r[1]; z = r[2]
    return 1.0

def f_m_fn(r):
    """
    Source function in $\Omega^-$
    """
    return 2.0

def f_p_fn(r):
    """
    Source function in $\Omega^+$
    """
    return 1.0

def alpha_fn(r):
    """
    Jump in solution at interface
    """
    return 0.0

def beta_fn(r):
    """
    Jump in flux at interface
    """
    return 0.0

init_fn, solve_fn = poisson_solver.setup(initial_value_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
sim_state = init_fn(R)

Ax_fn = solve_fn(gstate, sim_state)

pdb.set_trace()

# get normal vector and mean curvature
normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature_4th_order) 


def add_null_argument(func):
    def func_(x, y):
        return func(x)
    return func_

simulation_steps = int(1)
log = {
        'U' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'kappaM' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
      }

t1 = time.time()

normal, curve = normal_curve_fn(sim_state.solution, gstate)




sim_state.solution.block_until_ready()
t2 = time.time()

print(f"time per timestep is {(t2 - t1)/simulation_steps}")
jax.profiler.save_device_memory_profile("memory.prof")
io.write_vtk_log(gstate, log)

pdb.set_trace()