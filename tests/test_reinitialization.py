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
from src.dips.utils.data import StateData
from src import mesh, level_set
from src import io
from src import simulate_fields, interpolate
from src.jaxmd_modules import space
from src.jaxmd_modules.util import f32, i32
from jax.experimental import host_callback
from jax import (jit, lax, numpy as jnp, vmap)
import jax.profiler
import jax
from functools import partial
import time
import os
import sys
import queue

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)


config.update("jax_enable_x64", True)

jax.profiler.start_trace("./tensorboard")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def test_reinitialization():
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
    dy = yc[1] - yc[0]
    dz = zc[1] - zc[0]

    dt = dx * f32(0.95)
    simulation_steps = i32(tf / dt)

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)
    displacement_fn, shift_fn = space.periodic(box_size)

    # -- define velocity field as gradient of a scalar field
    @jit
    def velocity_fn(r, time_=0.0):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.array([0.0, 0.0, 0.0], dtype=f32)
    velocity_fn = vmap(velocity_fn, (0, None))

    def phi_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        # return lax.cond(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] > 0.25, lambda p: f32(1.0), lambda p: f32(-1.0), r)
        return jnp.where(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] > 0.25, f32(1.0), f32(-1.0))

    init_fn, apply_fn, reinitialize_fn, reinitialized_advect_fn = simulate_fields.level_set(
        phi_fn, shift_fn, dt)

    # get normal vector and mean curvature
    normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature)

    def add_null_argument(func):
        def func_(x, y):
            return func(x)
        return func_

    @partial(jit, static_argnums=0)
    def step_func(state_cb_fn, i, state_and_nbrs):
        state, gstate, dt = state_and_nbrs
        time_ = i * dt

        normal, curve = normal_curve_fn(state.solution, gstate)

        host_callback.call(state_cb_fn,
                           ({'t': time_,
                             'U': state.solution,
                             'kappaM': curve,
                             'nx': normal[:, 0],
                             'ny': normal[:, 1],
                             'nz': normal[:, 2]
                             }))
        state = reinitialize_fn(state, gstate)
        return state, gstate, dt

    # --- Actual simulation

    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R
    sim_state = init_fn(velocity_fn, R)

    q = queue.Queue()
    state_data = StateData(q, content_dir='./database_tmp', cols=['t', 'U', 'kappaM', 'nx', 'ny', 'nz'])
    state_data.start()

    partial_step_func = partial(step_func, state_data.queue_state)

    t1 = time.time()
    sim_state, gstate, dt = lax.fori_loop(i32(0), i32(
        simulation_steps), partial_step_func, (sim_state, gstate, dt))
    sim_state.solution.block_until_ready()
    t2 = time.time()
    print(
        f"time per timestep is {(t2 - t1)/simulation_steps}, Total steps  {simulation_steps}")

    state_data.stop()

    jax.profiler.save_device_memory_profile("memory.prof")
    jax.profiler.stop_trace()

    print(f"minimum distance to the sphere is {sim_state.solution.min()} \t should be \t -0.5")
    print(f"maximum distance to the sphere is {sim_state.solution.max()} \t should be \t 3.0")
    assert jnp.isclose(sim_state.solution.min(), -0.5, atol=2*dx)
    assert jnp.isclose(sim_state.solution.max(), 3.0, atol=2*dx)

    # --- if you want to visualize simulation uncomment below line
    # io.write_vtk_log(gstate, state_data)


if __name__ == "__main__":
    test_reinitialization()
