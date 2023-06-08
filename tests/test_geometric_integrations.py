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
import os
import sys

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from jax import numpy as jnp
from jax import random, vmap
from jax.config import config

config.update("jax_enable_x64", True)
from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.domain import interpolate, mesh
from jax_dips.geometry import geometric_integrations, geometric_integrations_per_point
from jax_dips.utils import io

Array = util.Array
f32 = util.f32
i32 = util.i32


@hydra.main(config_path="confs", config_name="geometric_integrations", version_base="1.1")
def test_geometric_integrations(cfg: DictConfig):
    logger.info(f"Starting {__file__}")
    logger.info(OmegaConf.to_yaml(cfg))

    # --- Parameters of test
    key = random.PRNGKey(0)
    dim = i32(3)
    xmin = ymin = zmin = f32(cfg.gridstates.Lmin)
    xmax = ymax = zmax = f32(cfg.gridstates.Lmax)
    box_size = xmax - xmin
    Nx = cfg.gridstates.Nx
    Ny = cfg.gridstates.Ny
    Nz = cfg.gridstates.Nz

    # --- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    init_mesh_fn, coord_at = mesh.construct(dim)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R
    ii = jnp.arange(2, Nx + 2)
    jj = jnp.arange(2, Ny + 2)
    kk = jnp.arange(2, Nz + 2)
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing="ij")
    nodes = jnp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1)))

    # --- Data structures and functions for testing

    @dataclasses.dataclass
    class CaseClass:
        """A struct containing the state of the mesh for this test case.

        This tuple stores the state of a simulation.

        Attributes:
        solution: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
        """

        phi: Array
        solution: Array
        jump: Array
        mu_m: Array
        mu_p: Array

    def lvl_set_node_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x * x + y * y + z * z) - 0.5

    def u_node_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    def alpha_node_fn(r):
        return 1.0

    def mu_m_node_fn(r):
        return 0.5

    def mu_p_node_fn(r):
        return 2.0

    phi_fn = vmap(lvl_set_node_fn)
    u_fn = vmap(u_node_fn)
    mu_m_fn = vmap(mu_m_node_fn)
    mu_p_fn = vmap(mu_p_node_fn)
    alpha_fn = vmap(alpha_node_fn)

    PHI = phi_fn(R)
    U = u_fn(R)
    ALPHA = alpha_fn(R)
    MU_M = mu_m_fn(R)
    MU_P = mu_p_fn(R)

    test_state = CaseClass(PHI, U, ALPHA, MU_M, MU_P)

    # --- Get interpolants
    phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(test_state.phi, gstate)
    u_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(test_state.solution, gstate)
    mu_m_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(test_state.mu_m, gstate)
    mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(test_state.mu_p, gstate)
    jump_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(test_state.jump, gstate)

    ########################################################################
    # --- Get functions for pointwise integration
    ########################################################################
    (
        get_vertices_of_cell_intersection_with_interface_at_point,
        is_cell_crossed_by_interface_pointwise,
    ) = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(phi_interp_fn)
    (
        integrate_over_interface_at_point,
        integrate_in_negative_domain_at_point,
    ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_point,
        is_cell_crossed_by_interface_pointwise,
        u_interp_fn,
    )
    (
        alpha_integrate_over_interface_at_point,
        _,
    ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_point,
        is_cell_crossed_by_interface_pointwise,
        jump_interp_fn,
    )
    compute_face_centroids_values_plus_minus_at_point = (
        geometric_integrations_per_point.compute_cell_faces_areas_values(
            get_vertices_of_cell_intersection_with_interface_at_point,
            is_cell_crossed_by_interface_pointwise,
            mu_m_interp_fn,
            mu_p_interp_fn,
        )
    )

    def test_surface_area_and_volume_pointwise():
        u_dGammas = vmap(integrate_over_interface_at_point, (0, None, None, None))(
            gstate.R, gstate.dx, gstate.dy, gstate.dz
        )
        logger.info(f"Surface area is computed to be {u_dGammas.sum()} ~~ must be ~~ {jnp.pi}")
        u_dOmegas = vmap(integrate_in_negative_domain_at_point, (0, None, None, None))(
            gstate.R, gstate.dx, gstate.dy, gstate.dz
        )
        logger.info(f"Volume is computed to be {u_dOmegas.sum()} ~~ must be ~~ {4.0 * jnp.pi * 0.5**3 / 3.0}")
        assert jnp.isclose(u_dGammas.sum(), jnp.pi, atol=0.02)
        assert jnp.isclose(u_dOmegas.sum(), 4.0 * jnp.pi * 0.5**3 / 3.0, atol=0.02)

    ########################################################################
    # --- Get functions for gridwise integration
    ########################################################################
    (
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
    ) = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(gstate, test_state)
    (
        integrate_over_interface_at_node,
        integrate_in_negative_domain_at_node,
    ) = geometric_integrations.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
        u_interp_fn,
    )
    (
        alpha_integrate_over_interface_at_node,
        _,
    ) = geometric_integrations.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
        jump_interp_fn,
    )
    compute_face_centroids_values_plus_minus_at_node = geometric_integrations.compute_cell_faces_areas_values(
        gstate,
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_cell_crossed_by_interface,
        mu_m_interp_fn,
        mu_p_interp_fn,
    )

    def test_surface_area_and_volume_gridwise():
        u_dGammas = vmap(integrate_over_interface_at_node)(nodes)
        logger.info(f"Surface area is computed to be {u_dGammas.sum()} ~~ must be ~~ {jnp.pi}")
        u_dOmegas = vmap(integrate_in_negative_domain_at_node)(nodes)
        logger.info(f"Volume is computed to be {u_dOmegas.sum()} ~~ must be ~~ {4.0 * jnp.pi * 0.5**3 / 3.0}")
        assert jnp.isclose(u_dGammas.sum(), jnp.pi, atol=0.02)
        assert jnp.isclose(u_dOmegas.sum(), 4.0 * jnp.pi * 0.5**3 / 3.0, atol=0.02)

    logger.info("performing pointwise integration...")
    test_surface_area_and_volume_pointwise()
    logger.info("performing gridwise integration...")
    test_surface_area_and_volume_gridwise()
    logger.info("done.")


if __name__ == "__main__":
    test_geometric_integrations()
