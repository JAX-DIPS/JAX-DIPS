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
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import queue
import time
from functools import partial

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import vmap
from jax.config import config
from jax.experimental import host_callback

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.data.data_stream import StateData
from jax_dips.domain import interpolate, mesh
from jax_dips.geometry import level_set
from jax_dips.solvers.advection import solver_advection
from jax_dips.utils import io

config.update("jax_enable_x64", False)


@hydra.main(config_path="confs", config_name="reinitialization", version_base="1.1")
def test_reinitialization(cfg: DictConfig):
    logger.info(f"Starting {__file__}")
    logger.info(OmegaConf.to_yaml(cfg))
    if cfg.io.jax_profiler:
        jax.profiler.start_trace("./tensorboard")

    dim = i32(3)
    xmin = ymin = zmin = f32(cfg.gridstates.Lmin)
    xmax = ymax = zmax = f32(cfg.gridstates.Lmax)
    box_size = xmax - xmin
    Nx = i32(cfg.gridstates.Nx)
    Ny = i32(cfg.gridstates.Ny)
    Nz = i32(cfg.gridstates.Nz)

    # --------- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    dz = zc[1] - zc[0]

    tf = f32(cfg.advect.tf)
    dt = dx * f32(cfg.advect.cfl)
    simulation_steps = i32(tf / dt)

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)

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
        return jnp.where(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] > 0.25, f32(1.0), f32(-1.0))

    (
        init_fn,
        apply_fn,
        reinitialize_fn,
        reinitialized_advect_fn,
    ) = solver_advection.level_set(phi_fn, dt)

    # get normal vector and mean curvature
    normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature)

    def add_null_argument(func):
        def func_(x, y):
            return func(x)

        return func_

    @partial(jit, static_argnums=0)
    def step_func_w_callback(state_cb_fn, i, state_and_nbrs):
        state, gstate, dt = state_and_nbrs
        time_ = i * dt

        normal, curve = normal_curve_fn(state.phi, gstate)

        host_callback.call(
            state_cb_fn,
            (
                {
                    "t": time_,
                    "U": state.phi,
                    "kappaM": curve,
                    "nx": normal[:, 0],
                    "ny": normal[:, 1],
                    "nz": normal[:, 2],
                }
            ),
        )
        state = reinitialize_fn(state, gstate)
        return state, gstate, dt

    @jit
    def step_func(i, state_and_nbrs):
        state, gstate, dt = state_and_nbrs
        time_ = i * dt
        normal, curve = normal_curve_fn(state.phi, gstate)
        state = reinitialize_fn(state, gstate)
        return state, gstate, dt

    # --- Actual simulation

    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R
    sim_state = init_fn(velocity_fn, R)

    if cfg.io.listen:
        q = queue.Queue()
        state_data = StateData(
            q,
            content_dir=cfg.io.listen_db_path,
            cols=["t", "U", "kappaM", "nx", "ny", "nz"],
        )
        state_data.start()
        _step_func = partial(step_func_w_callback, state_data.queue_state)
    else:
        _step_func = step_func

    t1 = time.time()
    sim_state, gstate, dt = lax.fori_loop(i32(0), i32(simulation_steps), _step_func, (sim_state, gstate, dt))
    sim_state.phi.block_until_ready()
    t2 = time.time()
    logger.info(f"time per timestep is {(t2 - t1)/simulation_steps}, Total steps  {simulation_steps}")

    if cfg.io.listen:
        state_data.stop()

    if cfg.io.jax_profiler:
        jax.profiler.save_device_memory_profile(f"memory_{__name__}.prof")
        jax.profiler.stop_trace()

    logger.info(f"minimum distance to the sphere is {sim_state.phi.min()} \t should be \t -0.5")
    logger.info(f"maximum distance to the sphere is {sim_state.phi.max()} \t should be \t 3.0")
    assert jnp.isclose(sim_state.phi.min(), -0.5, atol=2 * dx)
    assert jnp.isclose(sim_state.phi.max(), 3.0, atol=2 * dx)

    # --- if you want to visualize simulation uncomment below line
    if cfg.io.save_vtk:
        io.write_vtk_log(
            gstate,
            state_data,
            address=cfg.io.save_vtk_path,
        )

    logger.info("test passed successfully.")


if __name__ == "__main__":
    test_reinitialization()
