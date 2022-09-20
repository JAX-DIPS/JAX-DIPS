import jax
import jax.profiler
from jax import jit, random, lax, ops, grad
from jax._src.api import vmap
import jax.numpy as jnp
import numpy as onp
from jax.config import config
from src.jaxmd_modules import quantity, space, util
from src.jaxmd_modules import energy, partition
from src.jaxmd_modules.quantity import EnergyFn
config.update("jax_enable_x64", True)
from src.jaxmd_modules.util import f32, i32
from src import compositions
from src import simulate_fields
from src import simulate_particles
from src import io
from src import mesh
from src import interpolate 
import pdb
import os
import numpy as onp
from functools import partial

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
Nx = i32(256)
Ny = i32(256)
Nz = i32(256)
dimension = i32(3)

tf = f32(2 * jnp.pi / 30.0) 




#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)

dx = xc[1] - xc[0]
# dt = f32(0.02) 
dt = dx * f32(0.8)
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
    return jnp.array([-y, x, 0.0*z], dtype=f32)
    # return jnp.array([0.0*x, 0.0*y, 0.0*z], dtype=f32)

def phi_fn(r):
    x = r[0]; y = r[1]; z = r[2]
    return jnp.sqrt(x**2 + (y-1.0)**2 + z**2) - 0.5
    
    # return x**2 + (y-1.0)**2 + z**2 - 0.25

init_fn, apply_fn, reinitialize_fn, reinitialized_advect_fn = simulate_fields.level_set(phi_fn, shift_fn, dt)
sim_state = init_fn(R)

grad_fn = jax.vmap(jax.grad(phi_fn))
grad_phi = grad_fn(gstate.R)


"""
# This section is autodiff for the spatial gradients rather than discretizations
#--------------------------------------

curve_phi_fn = compositions.vec_curvature_fn(phi_fn) 
curvature_phi_n =  curve_phi_fn(gstate.R)
io.write_vtk_manual(gstate, {"phi" : sim_state.solution, "curvature phi" : curvature_phi_n}, 'results/manual_dump_n')

#--------------------------------------

advect_fn = compositions.vec_advect_one_step_autodiff(phi_fn, velocity_fn)
phi_np1 = advect_fn(dt, gstate.R)

#--------------------------------------

compose_fn = compositions.node_advect_step_autodiff

phi_np1_node_fn = jit(partial(compose_fn(phi_fn, velocity_fn), dt))
phi_np2_node_fn = jit(partial(compose_fn(phi_np1_node_fn, velocity_fn), dt))
phi_np3_node_fn = jit(partial(compose_fn(phi_np2_node_fn, velocity_fn), dt))
phi_np4_node_fn = jit(partial(compose_fn(phi_np3_node_fn, velocity_fn), dt))
# phi_np5_node_fn = jit(partial(compose_fn(phi_np4_node_fn, velocity_fn), dt))
# phi_np6_node_fn = jit(partial(compose_fn(phi_np5_node_fn, velocity_fn), dt))
# phi_np7_node_fn = jit(partial(compose_fn(phi_np6_node_fn, velocity_fn), dt))

print(phi_fn(gstate.R[0]))
print(phi_np1_node_fn(gstate.R[0]))
print(phi_np2_node_fn(gstate.R[0]))
print(phi_np3_node_fn(gstate.R[0]))
print(phi_np4_node_fn(gstate.R[0]))
# print(phi_np5_node_fn(gstate.R[0]))
# print(phi_np6_node_fn(gstate.R[0]))
# print(phi_np7_node_fn(gstate.R[0]))

phi_np2 = vmap(phi_np2_node_fn)(gstate.R)
curve_phi_np2_fn = compositions.vec_curvature_fn(phi_np2_node_fn) 
curvature_phi_np2 =  curve_phi_np2_fn(gstate.R)

# io.write_vtk_manual(gstate, {"phi" : phi_np2, "laplacian phi" : curvature_phi_np2}, 'results/manual_dump_np2')

pdb.set_trace()
"""


#---
# time = 0
# advect_phi_fn, grad_advect_phi_fn, reinitialized_fn, grad_reinitialized_fn = simulate_fields.advect_level_set(gstate, sim_state.velocity_nm1, velocity_fn, time)

# phi_n = sim_state.solution

# phi_np1 = advect_phi_fn(R, phi_n, dt)
# grad_phi_np1 = grad_advect_phi_fn(R, phi_n, dt)
# norm_grad_phi_np1 = jnp.linalg.norm(grad_phi_np1, axis=1)


# out = reinitialized_fn(R, phi_n, dt)
# grad_out = grad_reinitialized_fn(R, phi_n, dt)
# norm_grad_out = jnp.linalg.norm(grad_out, axis=1)


# err1 = jnp.linalg.norm( norm_grad_phi_np1 - 1.0 )
# err2 = jnp.linalg.norm( norm_grad_out - 1.0 )
# print(f"2-norm error was {err1} and is now {err2}")
# print(f"infinity-norm error was {abs(norm_grad_phi_np1 - 1.0).max()} and is now {abs(norm_grad_out - 1.0).max()}" )

# pdb.set_trace()

#---
# phi_0 = vmap(phi_fn)(gstate.R)
# g_phi_0 = vmap(grad(phi_fn))(gstate.R)



log = {
        'U' : jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        # 'V' : jnp.zeros((simulation_steps,) + sim_state.velocity_nm1.shape, dtype=f32),
        't' : jnp.zeros((simulation_steps,), dtype=f32)
      }

# log = {
#     'U' : jnp.zeros((2,) + sim_state.solution.shape, dtype=f32),
#     't' : jnp.zeros((2,), dtype=f32)
# }

# log['U'] = log['U'].at[0].set(sim_state.solution)
# log['t'] = log['t'].at[0].set(0.0)

@jit
def step_func(i, state_and_nbrs):
    state, log, dt = state_and_nbrs
    
    time_ = i * dt
    log['t'] = log['t'].at[i].set(time_)
    log['U'] = log['U'].at[i].set(state.solution)

    # vel = state.velocity_nm1
    # log['V'] = ops.index_update(log['V'], i, vel)
    
    state = reinitialize_fn(state, gstate)
    # state = lax.cond(i//10==0, lambda p: reinitialize_fn(p[0], p[1]), lambda p : p[0], (state, gstate))
    return apply_fn(state, gstate, time_), log, dt
    # return reinitialized_advect_fn(state, gstate, time_), log, dt


import time
t1 = time.time()
sim_state, log, dt = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (sim_state, log, dt))
sim_state.solution.block_until_ready()
t2 = time.time()
print(f"time per timestep is {(t2 - t1)/simulation_steps}")

# pdb.set_trace()
# log['U'] = log['U'].at[1].set(sim_state.solution)
# log['t'] = log['t'].at[1].set(tf - dt)


jax.profiler.save_device_memory_profile("memory.prof")

# io.write_vtk(gstate, log)
io.write_vtk_solution(gstate, log)



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