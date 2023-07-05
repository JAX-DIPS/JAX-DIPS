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
"""
import os
import time

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import jax
import jax.profiler as profiler
from jax import jit, lax
from jax import numpy as jnp
from jax.config import config

config.update("jax_enable_x64", False)

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import mesh
from jax_dips.geometry import geometric_integrations
from jax_dips.solvers.advection import solver_advection


@hydra.main(config_path="confs", config_name="advection", version_base="1.1")
def test_spinning_sphere(cfg: DictConfig):
    logger.info(f"Starting {__file__}")
    logger.info(OmegaConf.to_yaml(cfg))

    dim = i32(3)
    xmin = ymin = zmin = f32(cfg.gridstates.Lmin)
    xmax = ymax = zmax = f32(cfg.gridstates.Lmax)
    box_size = xmax - xmin
    Nx = i32(cfg.gridstates.Nx)
    Ny = i32(cfg.gridstates.Ny)
    Nz = i32(cfg.gridstates.Nz)
    tf = f32(cfg.advect.tf)

    # --------- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    dt = dx * f32(cfg.advect.cfl)
    simulation_steps = i32(tf / dt) + 1

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R
    # sample_pnt = coord_at(gstate, [1, 1, 1])

    # -- define velocity field as gradient of a scalar field

    @jit
    def velocity_node_fn(r, time_=0.0):
        x = r[0]
        y = r[1]
        z = r[2]
        # return lax.cond(time_ < 0.5, lambda p : jnp.array([-p[1], p[0], 0.0], dtype=f32), lambda p : jnp.array([p[1], -p[0], 0.0], dtype=f32), (x,y,z))
        return jnp.array([-y, x, 0.0 * z], dtype=f32)

    velocity_fn = jax.vmap(velocity_node_fn, (0, None))

    def phi_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + (y - 1.0) ** 2 + z**2) - 0.5

    (
        init_fn,
        apply_fn,
        reinitialize_fn,
        reinitialized_advect_fn,
    ) = solver_advection.level_set(phi_fn, dt)
    sim_state = init_fn(velocity_fn, R)

    # grad_fn = jax.vmap(jax.grad(phi_fn))
    # grad_phi = grad_fn(gstate.R)

    log = {
        "U": jnp.zeros((simulation_steps,) + sim_state.phi.shape, dtype=f32),
        "t": jnp.zeros((simulation_steps,), dtype=f32),
    }

    @jit
    def step_func(i, state_and_nbrs):
        state, log, dt = state_and_nbrs
        time_ = jnp.where(i == simulation_steps, tf, i * dt)
        log["t"] = log["t"].at[i].set(time_)
        log["U"] = log["U"].at[i].set(state.phi)
        state = reinitialize_fn(state, gstate)
        # state = lax.cond(i//10==0, lambda p: reinitialize_fn(p[0], p[1]), lambda p : p[0], (state, gstate))
        return apply_fn(velocity_fn, state, gstate, time_), log, dt

    t1 = time.time()
    sim_state, log, dt = lax.fori_loop(i32(0), i32(simulation_steps - 1), step_func, (sim_state, log, dt))
    sim_state.phi.block_until_ready()
    t2 = time.time()
    logger.info(f"time per timestep is {(t2 - t1)/simulation_steps}")

    dt_last = tf - (simulation_steps - 1) * dt
    sim_state, log, dt = step_func(i32(simulation_steps), (sim_state, log, dt_last))
    log["t"] = log["t"].at[simulation_steps - 1].set(tf)
    log["U"] = log["U"].at[simulation_steps - 1].set(sim_state.phi)

    difference_l2 = jnp.mean(jnp.square(log["U"][-1] - log["U"][0]))

    if cfg.io.jax_profiler:
        profiler.save_device_memory_profile("memory_advecting_sphere.prof")

    logger.info(
        f"L2 error in 2*pi advected sphere of radius 0.5 is equal to {difference_l2} \t should ideally be \t 0.0"
    )
    assert jnp.isclose(difference_l2, 0.0, atol=1e-4)

    def one_fn(r):
        return 1.0

    ii = jnp.arange(2, Nx + 2)
    jj = jnp.arange(2, Ny + 2)
    kk = jnp.arange(2, Nz + 2)
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing="ij")
    nodes = jnp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1)))
    vone_fn = jax.vmap(one_fn)
    (
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
    ) = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(gstate, sim_state)
    (
        integrate_over_interface_at_node,
        integrate_in_negative_domain,
    ) = geometric_integrations.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
        vone_fn,
    )
    sphere_volume = jax.vmap(integrate_in_negative_domain)(nodes).sum()
    sphere_area = jax.vmap(integrate_over_interface_at_node)(nodes).sum()
    logger.info(f"Final volume of the sphere is {sphere_volume}; it should be {4.0*3.141592653589793*0.5**3 / 3.0}")
    logger.info(f"Final surface area of the sphere is {sphere_area}; it should be {4.0*3.141592653589793*0.5**2}")

    # --- to save snapshots uncomment below line
    if cfg.io.save_vtk:
        io.write_vtk_solution(gstate, log, address=cfg.io.save_vtk_path)

    logger.info("done.")


if __name__ == "__main__":
    test_spinning_sphere()
