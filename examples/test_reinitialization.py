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
from src import space
from src import simulate_fields, interpolate
from src import io
from src import mesh, level_set
from jax.config import config
config.update("jax_enable_x64", True)

os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'

key = random.PRNGKey(0)
#key, split = random.split(key)
dim = i32(3)
xmin = ymin = zmin = f32(-2.0)
xmax = ymax = zmax = f32(2.0)
box_size = xmax - xmin
Nx = i32(128)
Ny = i32(128)
Nz = i32(128)
dimension = i32(3)
tf = f32( 2 * jnp.pi) 

#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
dx = xc[1] - xc[0]
dy = yc[1] - yc[0]
dz = zc[1] - zc[0]

dt = dx * f32(0.95)
simulation_steps = i32(tf / dt) 

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
    x = r[0]; y = r[1]; z = r[2]
    return jnp.array([0.0, 0.0, 0.0], dtype=f32)
velocity_fn = vmap(velocity_fn, (0,None))

def phi_fn(r):
    x = r[0]; y = r[1]; z = r[2]
    return lax.cond(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] > 0.25, lambda p: f32(1.0), lambda p: f32(-1.0), r)
    
init_fn, apply_fn, reinitialize_fn, reinitialized_advect_fn = simulate_fields.level_set(phi_fn, shift_fn, dt)
sim_state = init_fn(velocity_fn, R)

# get normal vector and mean curvature
normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature) 


def add_null_argument(func):
    def func_(x, y):
        return func(x)
    return func_


log = {
        'U' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'kappaM' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'nx' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'ny' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'nz' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        't' : jnp.zeros((simulation_steps,), dtype=f32)
      }
@jit
def step_func(i, state_and_nbrs):
    state, log, dt = state_and_nbrs
    time_ = i * dt

    normal, curve = normal_curve_fn(state.solution, gstate)

    log['t'] = log['t'].at[i].set(time_)
    log['U'] = log['U'].at[i].set(state.solution)
    log['kappaM'] = log['kappaM'].at[i].set(curve)
    log['nx'] = log['nx'].at[i].set(normal[:,0])
    log['ny'] = log['ny'].at[i].set(normal[:,1])
    log['nz'] = log['nz'].at[i].set(normal[:,2])
    
    state = reinitialize_fn(state, gstate)
    return state, log, dt

t1 = time.time()
sim_state, log, dt = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (sim_state, log, dt))
sim_state.solution.block_until_ready()
t2 = time.time()
print(f"time per timestep is {(t2 - t1)/simulation_steps}")

jax.profiler.save_device_memory_profile("memory.prof")


io.write_vtk_log(gstate, log)

pdb.set_trace()