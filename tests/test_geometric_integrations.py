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
from unittest import TextTestResult
from jax.config import config
from src import io, mesh, interpolate, geometric_integrations
from src.jaxmd_modules import dataclasses, util
from src.jaxmd_modules.util import f32, i32
from jax import (random, numpy as jnp, vmap)

import os
import sys
Array = util.Array


currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'

# --- Parameters of test
key = random.PRNGKey(0)
dim = i32(3)
xmin = ymin = zmin = f32(-2.0)
xmax = ymax = zmax = f32(2.0)
box_size = xmax - xmin
Nx = i32(128)
Ny = i32(128)
Nz = i32(128)
dimension = i32(3)


# --- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
dx = xc[1] - xc[0]
init_mesh_fn, coord_at = mesh.construct(dim)
gstate = init_mesh_fn(xc, yc, zc)
R = gstate.R
ii = jnp.arange(2, Nx+2)
jj = jnp.arange(2, Ny+2)
kk = jnp.arange(2, Nz+2)
I, J, K = jnp.meshgrid(ii, jj, kk, indexing='ij')
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
    return jnp.sqrt(x*x + y*y + z*z) - 0.5


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
phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
    test_state.phi, gstate)
u_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
    test_state.solution, gstate)
mu_m_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
    test_state.mu_m, gstate)
mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
    test_state.mu_p, gstate)
jump_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
    test_state.jump, gstate)


# --- Get functions
get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(
    gstate, test_state)
integrate_over_interface_at_node, integrate_in_negative_domain_at_node = geometric_integrations.integrate_over_gamma_and_omega_m(
    get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, u_interp_fn)
alpha_integrate_over_interface_at_node, _ = geometric_integrations.integrate_over_gamma_and_omega_m(
    get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, jump_interp_fn)
compute_face_centroids_values_plus_minus_at_node = geometric_integrations.compute_cell_faces_areas_values(
    gstate, get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, mu_m_interp_fn, mu_p_interp_fn)


# --- Measurements: PyTest will pick this up
def test_surface_area_and_volume():
    # poisson_scheme_coeffs = compute_face_centroids_values_plus_minus_at_node(
    #     nodes[794302])

    u_dGammas = vmap(integrate_over_interface_at_node)(nodes)
    print("\n\n\n")
    print(
        f"Surface area is computed to be {u_dGammas.sum()} ~~ must be ~~ {jnp.pi}")
    print("\n\n\n")

    u_dOmegas = vmap(integrate_in_negative_domain_at_node)(nodes)
    print(
        f"Volume is computed to be {u_dOmegas.sum()} ~~ must be ~~ {4.0 * jnp.pi * 0.5**3 / 3.0}")
    print("\n\n\n")

    assert jnp.isclose(u_dGammas.sum(), jnp.pi, atol=0.02)
    assert jnp.isclose(u_dOmegas.sum(), 4.0 * jnp.pi * 0.5**3 / 3.0, atol=0.02)

if __name__=="__main__":
    test_surface_area_and_volume()
    
# log = {
#         'U' : sim_state.solution,
#         'phi': sim_state.phi
#       }
# io.write_vtk_manual(gstate, log)
