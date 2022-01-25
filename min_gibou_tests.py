import jax
import jax.profiler
from jax import jit, random, lax, ops, grad
from jax._src.api import vmap
import jax.numpy as jnp
import numpy as onp
from jax.config import config
from src import quantity
from src.quantity import EnergyFn
config.update("jax_enable_x64", True)
from src.util import f32, i32
from src import partition, space
from src import simulate_fields
from src import energy, simulate_particles
from src import io
from src import mesh
from src import util, interpolate 
import pdb
import os
import numpy as onp


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


# Use JAX's random number generator to generate random initial positions.
key = random.PRNGKey(0)
#key, split = random.split(key)

dim = i32(3)
xmin = ymin = zmin = f32(-2.0)
xmax = ymax = zmax = f32(2.0)
box_size = xmax - xmin
Nx = i32(128)
Ny = i32(128)
Nz = i32(64)
dimension = i32(3)

tf = f32(2 * jnp.pi)




#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)

dx = xc[1] - xc[0]
dt = f32(0.02) #f32(6.0)*dx
simulation_steps = i32(tf / dt)
#---------------
# Create helper functions to define a periodic box of some size.


init_mesh_fn, coord_at = mesh.construct(dim)
gstate = init_mesh_fn(xc, yc, zc)
R = gstate.R
sample_pnt = coord_at(gstate, [1,1,1])

displacement_fn, shift_fn = space.periodic(box_size)

#-- define velocity field as gradient of a scalar field
# def energy_fn(r):
#     x = r[0]; y = r[1]; z = r[2]
#     engy =  0.5 * space.square_distance(r)
#     return engy
# velocity_fn = grad(jit(energy_fn))
@jit
def velocity_fn(r, time=0.0):
    x = r[0]; y = r[1]; z = r[2]
    # return lax.cond(time < 0.5, lambda p : jnp.array([-p[1], p[0], 0.0], dtype=f32), lambda p : jnp.array([p[1], -p[0], 0.0], dtype=f32), (x,y,z))
    # return lax.cond(time < 0.5, lambda p : jnp.array([1.0, 1.0, 0.0], dtype=f32), lambda p : jnp.array([-1.0, -1.0, 0.0], dtype=f32), (x,y,z))
    return jnp.array([-y, x, 0.0], dtype=f32)
    # return jnp.array([1.0, 1.0, -1.0], dtype=f32)

def phi_fn(r):
    x = r[0]; y = r[1]; z = r[2]
    return (x**2 + (y-1.0)**2 + z**2 - 0.5**2)


init_fn, apply_fn, reinitialize_fn = simulate_fields.level_set(velocity_fn, phi_fn, shift_fn, dt)

sim_state = init_fn(R)


@jit
def step_func(i, state_and_nbrs):
    state, log = state_and_nbrs
    time = i * dt
    log['t'] = ops.index_update(log['t'], i, time)
    sol = state.solution
    log['U'] = ops.index_update(log['U'], i, sol)
    vel = state.velocity_nm1
    log['V'] = ops.index_update(log['V'], i, vel)
    # state = reinitialize_fn(state, gstate)
    # state = lax.cond(i//10==0, lambda p: reinitialize_fn(p[0], p[1]), lambda p : p[0], (state, gstate))
    return apply_fn(state, gstate, time), log

log = {
        'U' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        'V' : jnp.zeros((simulation_steps,) + sim_state.velocity_nm1.shape, dtype=f32),
        't' : jnp.zeros((simulation_steps,), dtype=f32)
      }

import time
t1 = time.time()
sim_state, log = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (sim_state, log))
t2 = time.time()

print(f"time per timestep is {(t2 - t1)/simulation_steps}")
sim_state.solution.block_until_ready()

jax.profiler.save_device_memory_profile("memory.prof")

io.write_vtk(gstate, log)




#--- TEST INTERPOLATION
# u1 = log['U'][0]
# interp_fn = interpolate.multilinear_interpolation(u1, gstate)
# u2 = interp_fn(R).flatten().reshape(gstate.shape())
# u1 = u1.reshape(Nx,Ny,Nz)
# diff = (u2 - u1)
# gstate.x[interpolate.which_cell_index(jnp.asarray(1>=gstate.x))]

# u3 = interp_fn(R+0.1).flatten().reshape(gstate.shape())
# import matplotlib.pyplot as plt
# plt.contour(u3[:,:,gstate.shape()[2]//2]); plt.show()


# X, Y, Z = jnp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')
# import matplotlib.pyplot as plt
# vxyz = log['V'][0].reshape(Nx, Ny,Nz, 3)
# plt.figure(figsize=(7,7))
# plt.pcolor(X[:,:,0], Y[:,:,0],vxyz[:,:,0,1])
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()

# for i in range(len(log['U'])): 
#     u_ = log['U'][i].reshape(Nx, Ny,Nz)
#     plt.figure(figsize=(7,7))
#     plt.pcolor(X[:,:,Nz//2], Y[:,:,Nz//2],u_[:,:,Nz//2])
#     plt.ylabel('y')
#     plt.xlabel('x')
#     plt.show()

pdb.set_trace()