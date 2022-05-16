from jax.config import config
from src import mesh
from src import io
from src import simulate_fields
from src import space
from src.util import f32, i32
from jax import (jit, lax, numpy as jnp)
import jax.profiler
import jax
import time
import os
import sys
import pdb

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'





def test_spinning_sphere():
    dim = i32(3)
    xmin = ymin = zmin = f32(-2.0)
    xmax = ymax = zmax = f32(2.0)
    box_size = xmax - xmin
    Nx = i32(128)
    Ny = i32(128)
    Nz = i32(128)
    tf = f32(2 * jnp.pi)

    # --------- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    dt = dx * f32(0.9)
    simulation_steps = i32(tf / dt) + 1

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R
    # sample_pnt = coord_at(gstate, [1, 1, 1])

    displacement_fn, shift_fn = space.periodic(box_size)

    # -- define velocity field as gradient of a scalar field


    @jit
    def velocity_node_fn(r, time_=0.0):
        x = r[0]
        y = r[1]
        z = r[2]
        # return lax.cond(time_ < 0.5, lambda p : jnp.array([-p[1], p[0], 0.0], dtype=f32), lambda p : jnp.array([p[1], -p[0], 0.0], dtype=f32), (x,y,z))
        return jnp.array([-y, x, 0.0*z], dtype=f32)


    velocity_fn = jax.vmap(velocity_node_fn, (0, None))


    def phi_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + (y-1.0)**2 + z**2) - 0.5


    init_fn, apply_fn, reinitialize_fn, reinitialized_advect_fn = simulate_fields.level_set(
        phi_fn, shift_fn, dt)
    sim_state = init_fn(velocity_fn, R)
    # grad_fn = jax.vmap(jax.grad(phi_fn))
    # grad_phi = grad_fn(gstate.R)


    log = {
        'U': jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
        't': jnp.zeros((simulation_steps,), dtype=f32)
    }


    @jit
    def step_func(i, state_and_nbrs):
        state, log, dt = state_and_nbrs
        time_ = jnp.where(i==simulation_steps, tf, i * dt)
        log['t'] = log['t'].at[i].set(time_)
        log['U'] = log['U'].at[i].set(state.solution)
        state = reinitialize_fn(state, gstate)
        # state = lax.cond(i//10==0, lambda p: reinitialize_fn(p[0], p[1]), lambda p : p[0], (state, gstate))
        return apply_fn(velocity_fn, state, gstate, time_), log, dt



    t1 = time.time()
    sim_state, log, dt = lax.fori_loop(i32(0), i32(
        simulation_steps-1), step_func, (sim_state, log, dt))
    sim_state.solution.block_until_ready()
    t2 = time.time()
    print(f"time per timestep is {(t2 - t1)/simulation_steps}")
    

    dt_last = tf - (simulation_steps-1) * dt
    sim_state, log, dt = step_func(i32(simulation_steps), (sim_state, log, dt_last))
    log['t'] = log['t'].at[simulation_steps-1].set(tf)
    log['U'] = log['U'].at[simulation_steps-1].set(sim_state.solution)


    difference_l2 = jnp.mean(jnp.square(log['U'][-1] - log['U'][0]))
    
    jax.profiler.save_device_memory_profile("memory_advecting_sphere.prof")
    
    print(f"L2 error in 2\pi advected sphere of radius 0.5 is equal to {difference_l2} \t should ideally be \t 0.0")
    assert jnp.isclose(difference_l2, 0.0, atol=1e-4)
    #--- to save snapshots uncomment below line
    # io.write_vtk_solution(gstate, log, 'results/')



if __name__=="__main__":
    test_spinning_sphere()
    