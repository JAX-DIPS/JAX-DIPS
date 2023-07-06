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


import functools

from jax import config, jit, lax
from jax import numpy as jnp
from jax import vmap

from jax_dips.domain import interpolate

config.update("jax_debug_nans", True)
from jax_dips._jaxmd_modules import util

f32 = util.f32
i32 = util.i32


@jit
def sign_pm_fn(a):
    sgn = jnp.sign(a)
    return jnp.sign(sgn - 0.5)


@jit
def sign_p_fn(a):
    # returns 1 only if a>0, otherwise is 0
    sgn = jnp.sign(a)
    return jnp.floor(0.5 * sgn + 0.75)


@jit
def sign_m_fn(a):
    # returns 1 only if a<0, otherwise is 0
    sgn = jnp.sign(a)
    return jnp.ceil(0.5 * sgn - 0.75) * (-1.0)


@jit
def get_vertices_S_intersect_Gamma(S, phi_S, eta_S):
    """
    This function returns the vertices splitted by the level set.
    The intersection of the mesh-cell simplex S crossed by the level-set function.
    """
    zeros_gamma = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=f32,
    )

    def eta_1_fn(arg):
        S, phi_S, zeros_gamma = arg
        S_sorted = S[jnp.argsort(phi_S)]
        phi_S_sorted = jnp.sort(phi_S)
        Q_0 = (phi_S_sorted[0] * S_sorted[1] - phi_S_sorted[1] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[1])
        Q_1 = (phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[2])
        Q_2 = (phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[3])
        return jnp.array(
            [[Q_0, Q_1, Q_2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=f32,
        )

    def eta_3_fn(arg):
        S, phi_S, zeros_gamma = arg
        return eta_1_fn((S, -1.0 * phi_S, zeros_gamma))

    def eta_2_fn(arg):
        S, phi_S, zeros_gamma = arg
        S_sorted = S[jnp.argsort(phi_S)]
        phi_S_sorted = jnp.sort(phi_S)
        Q_0 = (phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[2])
        Q_1 = (phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[3])
        Q_2 = (phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1]) / (phi_S_sorted[1] - phi_S_sorted[3])
        Q_5 = (phi_S_sorted[1] * S_sorted[2] - phi_S_sorted[2] * S_sorted[1]) / (phi_S_sorted[1] - phi_S_sorted[2])
        return jnp.array([[Q_0, Q_1, Q_2], [Q_0, Q_5, Q_2]], dtype=f32)

    def eta_2_3_fn(arg):
        return jnp.where(eta_S == 2, eta_2_fn(arg), eta_3_fn(arg))

    def eta_1_2_3_fn(arg):
        return jnp.where(eta_S == 1, eta_1_fn(arg), eta_2_3_fn(arg))

    def eta_0_4_fn(arg):
        S, phi_S, zeros_gamma = arg
        return zeros_gamma

    shared_vertices = jnp.where(
        eta_S * (eta_S - 4) == 0,
        eta_0_4_fn((S, phi_S, zeros_gamma)),
        eta_1_2_3_fn((S, phi_S, zeros_gamma)),
    )
    return shared_vertices


@jit
def get_vertices_S_intersect_Omega_m(S, phi_S, eta_S: int):
    """
    This function returns the tetrahedra vertices splitted by the level set.
    The intersection of the mesh-cell volume crossed by the level-set function.
    """
    zeros_ = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=f32,
    )

    ones_ = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=f32,
    )

    def eta_1_fn(arg):
        S, phi_S, zeros_, ones_ = arg
        S_sorted = S[jnp.argsort(phi_S)]
        phi_S_sorted = jnp.sort(phi_S)
        Q_0 = S_sorted[0]
        Q_1 = (phi_S_sorted[0] * S_sorted[1] - phi_S_sorted[1] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[1])
        Q_2 = (phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[2])
        Q_3 = (phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[3])
        return jnp.array(
            [
                [Q_0, Q_1, Q_2, Q_3],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=f32,
        )

    def eta_2_fn(arg):
        S, phi_S, zeros_, ones_ = arg
        S_sorted = S[jnp.argsort(phi_S)]
        phi_S_sorted = jnp.sort(phi_S)
        Q_0 = S_sorted[0]
        Q_1 = S_sorted[1]
        Q_2 = (phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[2])
        Q_3 = (phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1]) / (phi_S_sorted[1] - phi_S_sorted[3])
        Q_4 = (phi_S_sorted[1] * S_sorted[2] - phi_S_sorted[2] * S_sorted[1]) / (phi_S_sorted[1] - phi_S_sorted[2])
        Q_5 = (phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[3])
        return jnp.array(
            [[Q_0, Q_1, Q_2, Q_3], [Q_4, Q_1, Q_2, Q_3], [Q_0, Q_5, Q_2, Q_3]],
            dtype=f32,
        )

    def eta_3_fn(arg):
        S, phi_S, zeros_, ones_ = arg
        S_sorted = S[jnp.argsort(phi_S)]
        phi_S_sorted = jnp.sort(phi_S)
        Q_0 = S_sorted[0]
        Q_1 = S_sorted[1]
        Q_2 = S_sorted[2]
        Q_3 = (phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1]) / (phi_S_sorted[1] - phi_S_sorted[3])
        Q_4 = (phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0]) / (phi_S_sorted[0] - phi_S_sorted[3])
        Q_5 = (phi_S_sorted[2] * S_sorted[3] - phi_S_sorted[3] * S_sorted[2]) / (phi_S_sorted[2] - phi_S_sorted[3])
        return jnp.array(
            [[Q_0, Q_1, Q_2, Q_3], [Q_0, Q_4, Q_2, Q_3], [Q_5, Q_4, Q_2, Q_3]],
            dtype=f32,
        )

    def eta_2_3_fn(arg):
        return jnp.where(eta_S == 2, eta_2_fn(arg), eta_3_fn(arg))

    def eta_1_2_3_fn(arg):
        return jnp.where(eta_S == 1, eta_1_fn(arg), eta_2_3_fn(arg))

    def eta_0_4_fn(arg):
        S, phi_S, zeros_, ones_ = arg
        ones_ = ones_.at[0, :].set(S)  # ops.index_update(ones_, jnp.index_exp[0,:], S)
        return jnp.where(eta_S == 0, zeros_, ones_)

    shared_vertices = jnp.where(
        eta_S * (eta_S - 4) == 0,
        eta_0_4_fn((S, phi_S, zeros_, ones_)),
        eta_1_2_3_fn((S, phi_S, zeros_, ones_)),
    )

    return shared_vertices


def get_vertices_of_cell_intersection_with_interface_at_node(gstate, sim_state):
    xo = gstate.x
    yo = gstate.y
    zo = gstate.z

    phi_n = sim_state.phi
    phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(phi_n, gstate)
    phi_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, phi_cube_)
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    @jit
    def is_node_crossed_by_interface(node):
        """
        If the control volume around node is crossed, it returns 0 (crossed), otherwise this returns +-1 (not crossed).

        Returns:
            +1: in Omega^+
             0: on interface (crossed)
            -1: in Omega^-
        """
        i, j, k = node
        # Get corners of the control volume
        dXcorners = jnp.array(
            [
                [-dx, -dy, -dz],
                [dx, -dy, -dz],
                [dx, -dy, dz],
                [-dx, -dy, dz],
                [-dx, dy, -dz],
                [dx, dy, -dz],
                [-dx, dy, dz],
                [dx, dy, dz],
            ],
            dtype=f32,
        ) * (0.5)

        R_cell_corners = dXcorners + jnp.array([x[i], y[j], z[k]])
        phi_cell_corners = phi_interp_fn(R_cell_corners)

        P_000 = R_cell_corners[0]
        P_100 = R_cell_corners[1]
        P_101 = R_cell_corners[2]
        P_001 = R_cell_corners[3]
        P_010 = R_cell_corners[4]
        P_110 = R_cell_corners[5]
        P_011 = R_cell_corners[6]
        P_111 = R_cell_corners[7]

        phi_P_000 = phi_cell_corners[0]
        phi_P_100 = phi_cell_corners[1]
        phi_P_101 = phi_cell_corners[2]
        phi_P_001 = phi_cell_corners[3]
        phi_P_010 = phi_cell_corners[4]
        phi_P_110 = phi_cell_corners[5]
        phi_P_011 = phi_cell_corners[6]
        phi_P_111 = phi_cell_corners[7]

        phis = jnp.array(
            [
                phi_P_000,
                phi_P_100,
                phi_P_101,
                phi_P_001,
                phi_P_010,
                phi_P_110,
                phi_P_011,
                phi_P_111,
            ]
        )
        eta_S = (sign_m_fn(phis).sum()).astype(int)

        return jnp.where(eta_S * (eta_S - 8) == 0, jnp.sign(phi_P_000), 0)

    @jit
    def get_vertices_of_cell_intersection_with_interface_at_node(node):
        """
        Based on Min & Gibou 2007: Geometric integration over irregular domains
        with application to level-set methods

        Returns a tuple with: ( S1 \cap \Gamma, ..., S5 \cap \Gamma, S1 \cap \Omega^-, ..., S5 \cap \Omega^-, S1, ..., S5 )
        """
        i, j, k = node
        # Get corners of the control volume
        dXcorners = 0.5 * jnp.array(
            [
                [-dx, -dy, -dz],
                [dx, -dy, -dz],
                [dx, -dy, dz],
                [-dx, -dy, dz],
                [-dx, dy, -dz],
                [dx, dy, -dz],
                [-dx, dy, dz],
                [dx, dy, dz],
            ],
            dtype=f32,
        )

        R_cell_corners = dXcorners + jnp.array([x[i], y[j], z[k]])
        phi_cell_corners = phi_interp_fn(R_cell_corners)

        P_000 = R_cell_corners[0]
        P_100 = R_cell_corners[1]
        P_101 = R_cell_corners[2]
        P_001 = R_cell_corners[3]
        P_010 = R_cell_corners[4]
        P_110 = R_cell_corners[5]
        P_011 = R_cell_corners[6]
        P_111 = R_cell_corners[7]

        phi_P_000 = phi_cell_corners[0]
        phi_P_100 = phi_cell_corners[1]
        phi_P_101 = phi_cell_corners[2]
        phi_P_001 = phi_cell_corners[3]
        phi_P_010 = phi_cell_corners[4]
        phi_P_110 = phi_cell_corners[5]
        phi_P_011 = phi_cell_corners[6]
        phi_P_111 = phi_cell_corners[7]

        # there are 5 simplices
        S_1 = jnp.array([P_000, P_100, P_010, P_001], dtype=f32)
        S_2 = jnp.array([P_110, P_100, P_010, P_111], dtype=f32)
        S_3 = jnp.array([P_101, P_100, P_111, P_001], dtype=f32)
        S_4 = jnp.array([P_011, P_111, P_010, P_001], dtype=f32)
        S_5 = jnp.array([P_111, P_100, P_010, P_001], dtype=f32)

        phi_S_1 = jnp.array([phi_P_000, phi_P_100, phi_P_010, phi_P_001], dtype=f32)
        phi_S_2 = jnp.array([phi_P_110, phi_P_100, phi_P_010, phi_P_111], dtype=f32)
        phi_S_3 = jnp.array([phi_P_101, phi_P_100, phi_P_111, phi_P_001], dtype=f32)
        phi_S_4 = jnp.array([phi_P_011, phi_P_111, phi_P_010, phi_P_001], dtype=f32)
        phi_S_5 = jnp.array([phi_P_111, phi_P_100, phi_P_010, phi_P_001], dtype=f32)

        eta_S_1 = (sign_m_fn(phi_S_1).sum()).astype(int)
        eta_S_2 = (sign_m_fn(phi_S_2).sum()).astype(int)
        eta_S_3 = (sign_m_fn(phi_S_3).sum()).astype(int)
        eta_S_4 = (sign_m_fn(phi_S_4).sum()).astype(int)
        eta_S_5 = (sign_m_fn(phi_S_5).sum()).astype(int)

        sv_gamma_s1 = get_vertices_S_intersect_Gamma(S_1, phi_S_1, eta_S_1)
        sv_omega_s1 = get_vertices_S_intersect_Omega_m(S_1, phi_S_1, eta_S_1)

        sv_gamma_s2 = get_vertices_S_intersect_Gamma(S_2, phi_S_2, eta_S_2)
        sv_omega_s2 = get_vertices_S_intersect_Omega_m(S_2, phi_S_2, eta_S_2)

        sv_gamma_s3 = get_vertices_S_intersect_Gamma(S_3, phi_S_3, eta_S_3)
        sv_omega_s3 = get_vertices_S_intersect_Omega_m(S_3, phi_S_3, eta_S_3)

        sv_gamma_s4 = get_vertices_S_intersect_Gamma(S_4, phi_S_4, eta_S_4)
        sv_omega_s4 = get_vertices_S_intersect_Omega_m(S_4, phi_S_4, eta_S_4)

        sv_gamma_s5 = get_vertices_S_intersect_Gamma(S_5, phi_S_5, eta_S_5)
        sv_omega_s5 = get_vertices_S_intersect_Omega_m(S_5, phi_S_5, eta_S_5)

        # ( S1 \cap \Gamma, ..., S5 \cap \Gamma, S1 \cap \Omega^-, ..., S5 \cap \Omega^-, S1, ..., S5 )
        return (
            sv_gamma_s1,
            sv_gamma_s2,
            sv_gamma_s3,
            sv_gamma_s4,
            sv_gamma_s5,
            sv_omega_s1,
            sv_omega_s2,
            sv_omega_s3,
            sv_omega_s4,
            sv_omega_s5,
            S_1,
            S_2,
            S_3,
            S_4,
            S_5,
        )

    # pieces = get_vertices_of_cell_intersection_with_interface_at_node(nodes[0])
    # pieces = vmap(get_vertices_of_cell_intersection_with_interface_at_node)(nodes)

    return (
        get_vertices_of_cell_intersection_with_interface_at_node,
        is_node_crossed_by_interface,
    )


# def integrate_over_gamma_and_omega(gstate, sim_state):
#     get_vertices_fn = get_vertices_of_cell_intersection_with_interface_at_node(gstate, sim_state)


def integrate_over_gamma_and_omega_m(get_vertices_fn, is_node_crossed_by_interface, u_interp_fn):
    @jit
    def vol_fn(A):
        tmp = jnp.linalg.det((A[1:] - A[0]) @ (A[1:] - A[0]).T)
        vol_tmp = (1.0 / 6.0) * jnp.sqrt(jnp.abs(jnp.nan_to_num(tmp)))
        return vol_tmp

    @jit
    def area_fn(A):
        tmp = jnp.linalg.det(
            (A[1:] - A[0]) @ (A[1:] - A[0]).T
        )  # jnp.where(a_shape==(3,3), jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) , 0.0 )
        return 0.5 * jnp.sqrt(jnp.abs(jnp.nan_to_num(tmp)))

    @jit
    def compute_interface_integral(node):
        pieces = get_vertices_fn(node)

        # ( S1 \cap \Gamma, ..., S5 \cap \Gamma, S1 \cap \Omega^-, ..., S5 \cap \Omega^-, S1, ..., S5 )
        S1_Gamma = pieces[0]
        S2_Gamma = pieces[1]
        S3_Gamma = pieces[2]
        S4_Gamma = pieces[3]
        S5_Gamma = pieces[4]

        # vol_fn = lambda A: 0.5 * jnp.sqrt(jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) )

        integral = area_fn(S1_Gamma[0]) * u_interp_fn(S1_Gamma[0]).mean()
        integral += area_fn(S1_Gamma[1]) * u_interp_fn(S1_Gamma[1]).mean()

        integral += area_fn(S2_Gamma[0]) * u_interp_fn(S2_Gamma[0]).mean()
        integral += area_fn(S2_Gamma[1]) * u_interp_fn(S2_Gamma[1]).mean()

        integral += area_fn(S3_Gamma[0]) * u_interp_fn(S3_Gamma[0]).mean()
        integral += area_fn(S3_Gamma[1]) * u_interp_fn(S3_Gamma[1]).mean()

        integral += area_fn(S4_Gamma[0]) * u_interp_fn(S4_Gamma[0]).mean()
        integral += area_fn(S4_Gamma[1]) * u_interp_fn(S4_Gamma[1]).mean()

        integral += area_fn(S5_Gamma[0]) * u_interp_fn(S5_Gamma[0]).mean()
        integral += area_fn(S5_Gamma[1]) * u_interp_fn(S5_Gamma[1]).mean()

        return integral

    @jit
    def integrate_over_interface_at_node(node):
        """
        node: cube indices in the range [2:Nx+2, 2:Ny+2, 2:Nz+2], excluding ghost layers
        """
        is_interface = is_node_crossed_by_interface(node)
        return jnp.where(is_interface == 0, compute_interface_integral(node), 0.0)

    @jit
    def compute_negative_bulk_integral(node):
        pieces = get_vertices_fn(node)

        # ( S1 \cap \Gamma, ..., S5 \cap \Gamma, S1 \cap \Omega^-, ..., S5 \cap \Omega^-, S1, ..., S5 )
        S1_Omega_m = pieces[5]
        S2_Omega_m = pieces[6]
        S3_Omega_m = pieces[7]
        S4_Omega_m = pieces[8]
        S5_Omega_m = pieces[9]

        # vol_fn = lambda A: (1.0 / 6.0) * jnp.sqrt(jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) )

        integral = vol_fn(S1_Omega_m[0]) * u_interp_fn(S1_Omega_m[0]).mean()
        integral += vol_fn(S1_Omega_m[1]) * u_interp_fn(S1_Omega_m[1]).mean()
        integral += vol_fn(S1_Omega_m[2]) * u_interp_fn(S1_Omega_m[2]).mean()

        integral += vol_fn(S2_Omega_m[0]) * u_interp_fn(S2_Omega_m[0]).mean()
        integral += vol_fn(S2_Omega_m[1]) * u_interp_fn(S2_Omega_m[1]).mean()
        integral += vol_fn(S2_Omega_m[2]) * u_interp_fn(S2_Omega_m[2]).mean()

        integral += vol_fn(S3_Omega_m[0]) * u_interp_fn(S3_Omega_m[0]).mean()
        integral += vol_fn(S3_Omega_m[1]) * u_interp_fn(S3_Omega_m[1]).mean()
        integral += vol_fn(S3_Omega_m[2]) * u_interp_fn(S3_Omega_m[2]).mean()

        integral += vol_fn(S4_Omega_m[0]) * u_interp_fn(S4_Omega_m[0]).mean()
        integral += vol_fn(S4_Omega_m[1]) * u_interp_fn(S4_Omega_m[1]).mean()
        integral += vol_fn(S4_Omega_m[2]) * u_interp_fn(S4_Omega_m[2]).mean()

        integral += vol_fn(S5_Omega_m[0]) * u_interp_fn(S5_Omega_m[0]).mean()
        integral += vol_fn(S5_Omega_m[1]) * u_interp_fn(S5_Omega_m[1]).mean()
        integral += vol_fn(S5_Omega_m[2]) * u_interp_fn(S5_Omega_m[2]).mean()

        return integral

    @jit
    def integrate_in_negative_domain_at_node(node):
        return compute_negative_bulk_integral(node)

    return integrate_over_interface_at_node, integrate_in_negative_domain_at_node


def compute_cell_faces_areas_values(
    gstate,
    get_vertices_fn,
    is_node_crossed_by_interface,
    mu_m_interp_fn,
    mu_p_interp_fn,
):
    """
    This function identifies centroids of each face in the positive and negative domain and on the interface,
    and evaluates values of some coefficient (diffusion coefficient, etc) on those centroids.

    This is done by the middle-cut triangulation (cf. Min & Gibou 2007), where the grid cells crossed by the
    interface are decomposed to five tetrahedra given by:
      number;      vertices                              ; faces exposed to external
        S1: conv(P_{000} ; P_{100} ; P_{010} ; P_{001}) -> z = 0 plane, x = 0 plane, y = 0 plane
        S2: conv(P_{110} ; P_{100} ; P_{010} ; P_{111}) -> z = 0 plane, x = 1 plane, y = 1 plane
        S3: conv(P_{101} ; P_{100} ; P_{111} ; P_{001}) -> z = 1 plane, x = 1 plane, y = 0 plane
        S4: conv(P_{011} ; P_{111} ; P_{010} ; P_{001}) -> z = 1 plane, x = 0 plane, y = 1 plane
        S5: conv(P_{111} ; P_{100} ; P_{010} ; P_{001}) -> no face exposure

    Args:
        get_vertices_fn: returns the triangulation of the grid cell
        is_node_crossed_by_interface: checks whether or not queried cell centered on the node is crossed by interface.
        mu_interp_fn: an interpolant to report function values on the centroids of the positive/negative segments on each face.

    Returns:
        An array with all cell specific information for the Poisson solver:
                [mu_A_dx_imh_m, mu_A_dx_imh_p, mu_A_dx_iph_m, mu_A_dx_iph_p, \
                 mu_A_dy_jmh_m, mu_A_dy_jmh_p, mu_A_dy_jph_m, mu_A_dy_jph_p,\
                 mu_A_dz_kmh_m, mu_A_dz_kmh_p, mu_A_dz_kph_m, mu_A_dz_kph_p,\
                 vol_m, vol_p]

    """
    xo = gstate.x
    yo = gstate.y
    zo = gstate.z
    dx = xo[2] - xo[1]
    dy = yo[2] - yo[1]
    dz = zo[2] - zo[1]

    vol = dx * dy * dz
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy

    # vol_fn = lambda A: jnp.nan_to_num( (1.0 / 6.0) * jnp.sqrt( jnp.nan_to_num( jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) ) ) )
    # area_fn = lambda A: jnp.nan_to_num( 0.5 * jnp.sqrt( jnp.nan_to_num( jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) ) ) )
    @jit
    def vol_fn(A):
        tmp = jnp.linalg.det((A[1:] - A[0]) @ (A[1:] - A[0]).T)
        vol_tmp = (1.0 / 6.0) * jnp.sqrt(jnp.abs(jnp.nan_to_num(tmp)))
        return vol_tmp

    @jit
    def area_fn(A):
        tmp = jnp.linalg.det(
            (A[1:] - A[0]) @ (A[1:] - A[0]).T
        )  # jnp.where(a_shape==(3,3), jnp.linalg.det( (A[1:] - A[0]) @ (A[1:] - A[0]).T ) , 0.0 )
        return 0.5 * jnp.sqrt(jnp.abs(jnp.nan_to_num(tmp)))

    @jit
    def positive_or_zero(num):
        return jnp.where(num < 0, 0.0, num)

    @jit
    def compute_interface_faces(node):
        pieces = get_vertices_fn(node)

        # ( S1 \cap \Gamma, ..., S5 \cap \Gamma, S1 \cap \Omega^-, ..., S5 \cap \Omega^-, S1, ..., S5 )
        # vertices of decomposed tetrahedra enclosed in \Omega^m
        S1_Omega_m = pieces[5]
        S2_Omega_m = pieces[6]
        S3_Omega_m = pieces[7]
        S4_Omega_m = pieces[8]
        S5_Omega_m = pieces[9]

        i, j, k = node - 2
        # Get face centers of the control volume centered on node
        dXfaces = 0.5 * jnp.array(
            [
                [-dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [0.0, -dy, 0.0],
                [0.0, dy, 0.0],
                [0.0, 0.0, -dz],
                [0.0, 0.0, dz],
            ],
            dtype=f32,
        )

        R_cell_faces = dXfaces + jnp.array([xo[i], yo[j], zo[k]])

        mu_m_faces = mu_m_interp_fn(R_cell_faces)
        mu_p_faces = mu_p_interp_fn(R_cell_faces)

        # ---- compute volumes of minus/plus of the cell
        vol_m = vol_fn(S1_Omega_m[0])
        vol_m += vol_fn(S1_Omega_m[1])
        vol_m += vol_fn(S1_Omega_m[2])
        vol_m += vol_fn(S2_Omega_m[0])
        vol_m += vol_fn(S2_Omega_m[1])
        vol_m += vol_fn(S2_Omega_m[2])
        vol_m += vol_fn(S3_Omega_m[0])
        vol_m += vol_fn(S3_Omega_m[1])
        vol_m += vol_fn(S3_Omega_m[2])
        vol_m += vol_fn(S4_Omega_m[0])
        vol_m += vol_fn(S4_Omega_m[1])
        vol_m += vol_fn(S4_Omega_m[2])
        vol_m += vol_fn(S5_Omega_m[0])
        vol_m += vol_fn(S5_Omega_m[1])
        vol_m += vol_fn(S5_Omega_m[2])
        vol_p = positive_or_zero(vol - vol_m)
        # vol_p = jnp.where(vol_p_ < 0, 0.0, vol_p_)
        # ----

        # ---- compute areas of minus/plus on each face
        x_m_face = xo[i] - 0.5 * dx
        x_p_face = xo[i] + 0.5 * dx

        y_m_face = yo[j] - 0.5 * dy
        y_p_face = yo[j] + 0.5 * dy

        z_m_face = zo[k] - 0.5 * dz
        z_p_face = zo[k] + 0.5 * dz

        # ---- this number is used to keep track of garbage points in area_from_partitions function below
        fiducial_point = jnp.rint((xo[-1] - xo[0]) * 100)

        def compute_area_from_partitions(
            on_face_partition_1,
            s_omega_m_partition_1,
            on_face_partition_2,
            s_omega_m_partition_2,
        ):
            def comp(partition):
                """
                Figure 4 of Min & Gibou 2007: 3 different cases are possible. Among them the ones where
                exactly 3 vertices lie on the corresponding face contribute an area element. We add those areas.
                """
                simplex_0 = partition[0]
                simplex_1 = partition[1]
                simplex_2 = partition[2]

                num_face_points_0 = jnp.count_nonzero(simplex_0[:, 0] - fiducial_point)
                num_face_points_1 = jnp.count_nonzero(simplex_1[:, 0] - fiducial_point)
                num_face_points_2 = jnp.count_nonzero(simplex_2[:, 0] - fiducial_point)

                sort_indices_0 = jnp.argsort(simplex_0[:, 0])
                sort_indices_1 = jnp.argsort(simplex_1[:, 0])
                sort_indices_2 = jnp.argsort(simplex_2[:, 0])

                area_0 = lax.cond(
                    num_face_points_0 == 3,
                    lambda p: area_fn(p),
                    lambda p: f32(0.0),
                    simplex_0[sort_indices_0][:3],
                )
                area_1 = lax.cond(
                    num_face_points_1 == 3,
                    lambda p: area_fn(p),
                    lambda p: f32(0.0),
                    simplex_1[sort_indices_1][:3],
                )
                area_2 = lax.cond(
                    num_face_points_2 == 3,
                    lambda p: area_fn(p),
                    lambda p: f32(0.0),
                    simplex_2[sort_indices_2][:3],
                )

                return area_0 + area_1 + area_2

            partition_1 = jnp.zeros_like(s_omega_m_partition_1)

            partition_1 = partition_1.at[0, 0].set(
                jnp.where(
                    on_face_partition_1[0, 0],
                    s_omega_m_partition_1[0, 0],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[0, 1].set(
                jnp.where(
                    on_face_partition_1[0, 1],
                    s_omega_m_partition_1[0, 1],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[0, 2].set(
                jnp.where(
                    on_face_partition_1[0, 2],
                    s_omega_m_partition_1[0, 2],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[0, 3].set(
                jnp.where(
                    on_face_partition_1[0, 3],
                    s_omega_m_partition_1[0, 3],
                    fiducial_point,
                )
            )

            partition_1 = partition_1.at[1, 0].set(
                jnp.where(
                    on_face_partition_1[1, 0],
                    s_omega_m_partition_1[1, 0],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[1, 1].set(
                jnp.where(
                    on_face_partition_1[1, 1],
                    s_omega_m_partition_1[1, 1],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[1, 2].set(
                jnp.where(
                    on_face_partition_1[1, 2],
                    s_omega_m_partition_1[1, 2],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[1, 3].set(
                jnp.where(
                    on_face_partition_1[1, 3],
                    s_omega_m_partition_1[1, 3],
                    fiducial_point,
                )
            )

            partition_1 = partition_1.at[2, 0].set(
                jnp.where(
                    on_face_partition_1[2, 0],
                    s_omega_m_partition_1[2, 0],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[2, 1].set(
                jnp.where(
                    on_face_partition_1[2, 1],
                    s_omega_m_partition_1[2, 1],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[2, 2].set(
                jnp.where(
                    on_face_partition_1[2, 2],
                    s_omega_m_partition_1[2, 2],
                    fiducial_point,
                )
            )
            partition_1 = partition_1.at[2, 3].set(
                jnp.where(
                    on_face_partition_1[2, 3],
                    s_omega_m_partition_1[2, 3],
                    fiducial_point,
                )
            )

            partition_2 = jnp.zeros_like(s_omega_m_partition_2)

            partition_2 = partition_2.at[0, 0].set(
                jnp.where(
                    on_face_partition_2[0, 0],
                    s_omega_m_partition_2[0, 0],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[0, 1].set(
                jnp.where(
                    on_face_partition_2[0, 1],
                    s_omega_m_partition_2[0, 1],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[0, 2].set(
                jnp.where(
                    on_face_partition_2[0, 2],
                    s_omega_m_partition_2[0, 2],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[0, 3].set(
                jnp.where(
                    on_face_partition_2[0, 3],
                    s_omega_m_partition_2[0, 3],
                    fiducial_point,
                )
            )

            partition_2 = partition_2.at[1, 0].set(
                jnp.where(
                    on_face_partition_2[1, 0],
                    s_omega_m_partition_2[1, 0],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[1, 1].set(
                jnp.where(
                    on_face_partition_2[1, 1],
                    s_omega_m_partition_2[1, 1],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[1, 2].set(
                jnp.where(
                    on_face_partition_2[1, 2],
                    s_omega_m_partition_2[1, 2],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[1, 3].set(
                jnp.where(
                    on_face_partition_2[1, 3],
                    s_omega_m_partition_2[1, 3],
                    fiducial_point,
                )
            )

            partition_2 = partition_2.at[2, 0].set(
                jnp.where(
                    on_face_partition_2[2, 0],
                    s_omega_m_partition_2[2, 0],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[2, 1].set(
                jnp.where(
                    on_face_partition_2[2, 1],
                    s_omega_m_partition_2[2, 1],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[2, 2].set(
                jnp.where(
                    on_face_partition_2[2, 2],
                    s_omega_m_partition_2[2, 2],
                    fiducial_point,
                )
            )
            partition_2 = partition_2.at[2, 3].set(
                jnp.where(
                    on_face_partition_2[2, 3],
                    s_omega_m_partition_2[2, 3],
                    fiducial_point,
                )
            )

            area_partition_1 = comp(partition_1)
            area_partition_2 = comp(partition_2)

            return area_partition_1 + area_partition_2

        def extract_area_minus_x_face(s_omega_m_partition_1, s_omega_m_partition_2, x_face):
            on_face_partition_1 = jnp.isclose(s_omega_m_partition_1[:, :, 0], x_face, atol=1e-10 * dx)
            on_face_partition_2 = jnp.isclose(s_omega_m_partition_2[:, :, 0], x_face, atol=1e-10 * dx)
            return compute_area_from_partitions(
                on_face_partition_1,
                s_omega_m_partition_1,
                on_face_partition_2,
                s_omega_m_partition_2,
            )

        def extract_area_minus_y_face(s_omega_m_partition_1, s_omega_m_partition_2, y_face):
            on_face_partition_1 = jnp.isclose(s_omega_m_partition_1[:, :, 1], y_face, atol=1e-10 * dy)
            on_face_partition_2 = jnp.isclose(s_omega_m_partition_2[:, :, 1], y_face, atol=1e-10 * dy)
            return compute_area_from_partitions(
                on_face_partition_1,
                s_omega_m_partition_1,
                on_face_partition_2,
                s_omega_m_partition_2,
            )

        def extract_area_minus_z_face(s_omega_m_partition_1, s_omega_m_partition_2, z_face):
            on_face_partition_1 = jnp.isclose(s_omega_m_partition_1[:, :, 2], z_face, atol=1e-10 * dz)
            on_face_partition_2 = jnp.isclose(s_omega_m_partition_2[:, :, 2], z_face, atol=1e-10 * dz)
            return compute_area_from_partitions(
                on_face_partition_1,
                s_omega_m_partition_1,
                on_face_partition_2,
                s_omega_m_partition_2,
            )

        area_imh_m = extract_area_minus_x_face(S1_Omega_m, S4_Omega_m, x_m_face)
        area_iph_m = extract_area_minus_x_face(S2_Omega_m, S3_Omega_m, x_p_face)
        area_imh_p = positive_or_zero(area_x - area_imh_m)
        area_iph_p = positive_or_zero(area_x - area_iph_m)

        area_jmh_m = extract_area_minus_y_face(S1_Omega_m, S3_Omega_m, y_m_face)
        area_jph_m = extract_area_minus_y_face(S2_Omega_m, S4_Omega_m, y_p_face)
        area_jmh_p = positive_or_zero(area_y - area_jmh_m)
        area_jph_p = positive_or_zero(area_y - area_jph_m)

        area_kmh_m = extract_area_minus_z_face(S1_Omega_m, S2_Omega_m, z_m_face)
        area_kph_m = extract_area_minus_z_face(S3_Omega_m, S4_Omega_m, z_p_face)
        area_kmh_p = positive_or_zero(area_z - area_kmh_m)
        area_kph_p = positive_or_zero(area_z - area_kph_m)

        # -----

        mu_A_dx_imh_m = area_imh_m * mu_m_faces[0] / dx
        mu_A_dx_iph_m = area_iph_m * mu_m_faces[1] / dx
        mu_A_dx_imh_p = area_imh_p * mu_p_faces[0] / dx
        mu_A_dx_iph_p = area_iph_p * mu_p_faces[1] / dx

        mu_A_dy_jmh_m = area_jmh_m * mu_m_faces[2] / dy
        mu_A_dy_jph_m = area_jph_m * mu_m_faces[3] / dy
        mu_A_dy_jmh_p = area_jmh_p * mu_p_faces[2] / dy
        mu_A_dy_jph_p = area_jph_p * mu_p_faces[3] / dy

        mu_A_dz_kmh_m = area_kmh_m * mu_m_faces[4] / dz
        mu_A_dz_kph_m = area_kph_m * mu_m_faces[5] / dz
        mu_A_dz_kmh_p = area_kmh_p * mu_p_faces[4] / dz
        mu_A_dz_kph_p = area_kph_p * mu_p_faces[5] / dz

        # node_coeff_times_area_divided_size_and_volumes = jnp.array([mu_A_dx_imh_m, mu_A_dx_imh_p, mu_A_dx_iph_m, mu_A_dx_iph_p, \
        #                                                             mu_A_dy_jmh_m, mu_A_dy_jmh_p, mu_A_dy_jph_m, mu_A_dy_jph_p,\
        #                                                             mu_A_dz_kmh_m, mu_A_dz_kmh_p, mu_A_dz_kph_m, mu_A_dz_kph_p,\
        #                                                             vol_m, vol_p], dtype=f32)
        # return node_coeff_times_area_divided_size_and_volumes

        return jnp.array(
            [
                mu_A_dx_imh_m,
                mu_A_dx_imh_p,
                mu_A_dx_iph_m,
                mu_A_dx_iph_p,
                mu_A_dy_jmh_m,
                mu_A_dy_jmh_p,
                mu_A_dy_jph_m,
                mu_A_dy_jph_p,
                mu_A_dz_kmh_m,
                mu_A_dz_kmh_p,
                mu_A_dz_kph_m,
                mu_A_dz_kph_p,
                vol_m,
                vol_p,
                area_imh_m,
                area_imh_p,
                area_iph_m,
                area_iph_p,
                area_jmh_m,
                area_jmh_p,
                area_jph_m,
                area_jph_p,
                area_kmh_m,
                area_kmh_p,
                area_kph_m,
                area_kph_p,
            ],
            dtype=f32,
        )

    @jit
    def compute_domain_faces(node, is_interface):
        """
        A domain face is not crossed by the interface, therefore plus/minus centroids overlap
        and either plus (in \Omega^+) or minus (in \Omega^-) surface areas are nonzero
        depending on sign of the level-set function at the node.
        """
        i, j, k = node - 2
        # Get face centers of the control volume centered on node
        dXfaces = 0.5 * jnp.array(
            [
                [-dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [0.0, -dy, 0.0],
                [0.0, dy, 0.0],
                [0.0, 0.0, -dz],
                [0.0, 0.0, dz],
            ],
            dtype=f32,
        )

        R_cell_faces = dXfaces + jnp.array([xo[i], yo[j], zo[k]])

        mu_m_faces = mu_m_interp_fn(R_cell_faces) * sign_m_fn(is_interface)
        mu_p_faces = mu_p_interp_fn(R_cell_faces) * sign_p_fn(is_interface)

        vol_m = vol * sign_m_fn(is_interface)
        vol_p = vol * sign_p_fn(is_interface)

        # ---------

        mu_A_dx_imh_m = area_x * mu_m_faces[0] / dx
        mu_A_dx_iph_m = area_x * mu_m_faces[1] / dx
        mu_A_dx_imh_p = area_x * mu_p_faces[0] / dx
        mu_A_dx_iph_p = area_x * mu_p_faces[1] / dx

        mu_A_dy_jmh_m = area_y * mu_m_faces[2] / dy
        mu_A_dy_jph_m = area_y * mu_m_faces[3] / dy
        mu_A_dy_jmh_p = area_y * mu_p_faces[2] / dy
        mu_A_dy_jph_p = area_y * mu_p_faces[3] / dy

        mu_A_dz_kmh_m = area_z * mu_m_faces[4] / dz
        mu_A_dz_kph_m = area_z * mu_m_faces[5] / dz
        mu_A_dz_kmh_p = area_z * mu_p_faces[4] / dz
        mu_A_dz_kph_p = area_z * mu_p_faces[5] / dz

        # node_coeff_times_area_divided_size_and_volumes = jnp.array([mu_A_dx_imh_m, mu_A_dx_imh_p, mu_A_dx_iph_m, mu_A_dx_iph_p, \
        #                                                             mu_A_dy_jmh_m, mu_A_dy_jmh_p, mu_A_dy_jph_m, mu_A_dy_jph_p,\
        #                                                             mu_A_dz_kmh_m, mu_A_dz_kmh_p, mu_A_dz_kph_m, mu_A_dz_kph_p,\
        #                                                             vol_m, vol_p], dtype=f32)
        # return node_coeff_times_area_divided_size_and_volumes

        p_mask = sign_p_fn(is_interface)
        m_mask = sign_m_fn(is_interface)

        return jnp.array(
            [
                mu_A_dx_imh_m,
                mu_A_dx_imh_p,
                mu_A_dx_iph_m,
                mu_A_dx_iph_p,
                mu_A_dy_jmh_m,
                mu_A_dy_jmh_p,
                mu_A_dy_jph_m,
                mu_A_dy_jph_p,
                mu_A_dz_kmh_m,
                mu_A_dz_kmh_p,
                mu_A_dz_kph_m,
                mu_A_dz_kph_p,
                vol_m,
                vol_p,
                area_x * m_mask,
                area_x * p_mask,
                area_x * m_mask,
                area_x * p_mask,
                area_y * m_mask,
                area_y * p_mask,
                area_y * m_mask,
                area_y * p_mask,
                area_z * m_mask,
                area_z * p_mask,
                area_z * m_mask,
                area_z * p_mask,
            ],
            dtype=f32,
        )

    def compute_face_centroids_values_plus_minus_at_node(node):
        """
        Main driver, differentiating between domain cells and interface cells.
        """
        is_interface = is_node_crossed_by_interface(node)
        return jnp.where(
            is_interface == 0,
            compute_interface_faces(node),
            compute_domain_faces(node, is_interface),
        )

    return compute_face_centroids_values_plus_minus_at_node
