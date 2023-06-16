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
import pickle

import jax
from jax import config, grad, jit
from jax import numpy as jnp
from jax import random, value_and_grad, vmap

from jax_dips.geometry.level_set import geometric_integrations_per_point

config.update("jax_debug_nans", False)
from functools import partial

import haiku as hk
import optax

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.geometry import interpolate
from jax_dips.nn.nn_solution_model import DoubleMLP


class PoissonAdvectionTrainer:
    """
    This is a completely local point-based Poisson solver.
    """

    def __init__(self, gstate, sim_state, sim_state_fn, optimizer, algorithm=0, precondition=1):
        """
        algorithm = 0: use regression to evaluate u^\pm
        algorithm = 1: use neural network to evaluate u^\pm
        """
        self.optimizer = optimizer
        self.gstate = gstate
        self.sim_state_fn = sim_state_fn
        self.sim_state = sim_state
        self.algorithm = algorithm

        """ Grid Info """
        # self.bandwidth_squared = (2.0 * self.dx)*(2.0 * self.dx)
        self.xmin = gstate.xmin()
        self.xmax = gstate.xmax()
        self.ymin = gstate.ymin()
        self.ymax = gstate.ymax()
        self.zmin = gstate.zmin()
        self.zmax = gstate.zmax()

        """ functions for the method """
        self.dir_bc_fn = self.sim_state_fn.dir_bc_fn
        self.f_m_interp_fn = self.sim_state_fn.f_m_fn
        self.f_p_interp_fn = self.sim_state_fn.f_p_fn
        self.k_m_interp_fn = self.sim_state_fn.k_m_fn
        self.k_p_interp_fn = self.sim_state_fn.k_p_fn
        self.mu_m_interp_fn = self.sim_state_fn.mu_m_fn
        self.mu_p_interp_fn = self.sim_state_fn.mu_p_fn
        self.alpha_interp_fn = self.sim_state_fn.alpha_fn
        self.beta_interp_fn = self.sim_state_fn.beta_fn

        self.mu_m_over_mu_p_interp_fn = lambda r: self.mu_m_interp_fn(r) / self.mu_p_interp_fn(r)
        self.beta_over_mu_m_interp_fn = lambda r: self.beta_interp_fn(r) / self.mu_m_interp_fn(r)
        self.beta_over_mu_p_interp_fn = lambda r: self.beta_interp_fn(r) / self.mu_p_interp_fn(r)

        """ The level set function or its interpolant (if is free boundary) """
        # self.phi_cube_ = sim_state.phi.reshape(self.grid_shape)
        # x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, self.phi_cube_)
        # x, y, z, self.phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)
        # self.phi_flat = self.phi_cube_.reshape(-1)
        # self.phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(self.sim_state.phi, self.gstate)
        self.phi_interp_fn = self.sim_state_fn.phi_fn

        """ Geometric operations per point """
        (
            self.get_vertices_of_cell_intersection_with_interface_at_point,
            self.is_cell_crossed_by_interface,
        ) = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(self.phi_interp_fn)
        (
            self.beta_integrate_over_interface_at_point,
            self.beta_integrate_in_negative_domain,
        ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
            self.get_vertices_of_cell_intersection_with_interface_at_point,
            self.is_cell_crossed_by_interface,
            self.beta_interp_fn,
        )
        self.compute_face_centroids_values_plus_minus_at_point = (
            geometric_integrations_per_point.compute_cell_faces_areas_values(
                self.get_vertices_of_cell_intersection_with_interface_at_point,
                self.is_cell_crossed_by_interface,
                self.mu_m_interp_fn,
                self.mu_p_interp_fn,
            )
        )

        # self.ngbs = jnp.array([ [-1, -1, -1],
        #                         [0, -1, -1],
        #                         [1, -1, -1],
        #                         [-1,  0, -1],
        #                         [0,  0, -1],
        #                         [1,  0, -1],
        #                         [-1,  1, -1],
        #                         [0,  1, -1],
        #                         [1,  1, -1],
        #                         [-1, -1,  0],
        #                         [0, -1,  0],
        #                         [1, -1,  0],
        #                         [-1,  0,  0],
        #                         [0,  0,  0],
        #                         [1,  0,  0],
        #                         [-1,  1,  0],
        #                         [0,  1,  0],
        #                         [1,  1,  0],
        #                         [-1, -1,  1],
        #                         [0, -1,  1],
        #                         [1, -1,  1],
        #                         [-1,  0,  1],
        #                         [0,  0,  1],
        #                         [1,  0,  1],
        #                         [-1,  1,  1],
        #                         [0,  1,  1],
        #                         [1,  1,  1]], dtype=i32)

        """ initialize configurated solver """
        if self.algorithm == 0:
            self.u_mp_fn = self.get_u_mp_by_regression_at_point_fn

        elif self.algorithm == 1:
            self.initialize_neural_based_algorithm()
            self.u_mp_fn = NotImplemented  # self.get_u_mp_by_neural_network_at_node_fn

        self.compute_normal_gradient_solution_mp_on_interface = (
            self.compute_normal_gradient_solution_mp_on_interface_neural_network
        )
        self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_neural_network
        self.compute_normal_gradient_solution_on_interface = (
            self.compute_normal_gradient_solution_on_interface_neural_network
        )
        self.compute_gradient_solution = self.compute_gradient_solution_neural_network

        if precondition == 1:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_preconditioned_fn
        elif precondition == 0:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_vanilla_fn

    def fetch_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            return None
        else:
            checkpoints = [p for p in os.listdir(checkpoint_dir) if "checkpoint_" in p]
            if checkpoints == []:
                return None
            checkpoint = os.path.join(checkpoint_dir, max(checkpoints))
            print(f"Loading checkpoint {checkpoint}")
            with open(checkpoint, "rb") as f:
                state = pickle.load(f)
            return state

    def save_checkpoint(self, checkpoint_dir, state):
        if checkpoint_dir is None:
            print("No checkpoint dir. specified. Skipping checkpoint.")
            return

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = os.path.join(checkpoint_dir, "checkpoint_" + str(state["epoch"]))
        print(f"Saving checkpoint {checkpoint}")
        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)
        return checkpoint

    def get_Xijk(self, cell_dx, cell_dy, cell_dz):
        Xijk = jnp.array(
            [
                [-cell_dx, -cell_dy, -cell_dz],
                [0.0, -cell_dy, -cell_dz],
                [cell_dx, -cell_dy, -cell_dz],
                [-cell_dx, 0.0, -cell_dz],
                [0.0, 0.0, -cell_dz],
                [cell_dx, 0.0, -cell_dz],
                [-cell_dx, cell_dy, -cell_dz],
                [0.0, cell_dy, -cell_dz],
                [cell_dx, cell_dy, -cell_dz],
                [-cell_dx, -cell_dy, 0.0],
                [0.0, -cell_dy, 0.0],
                [cell_dx, -cell_dy, 0.0],
                [-cell_dx, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [cell_dx, 0.0, 0.0],
                [-cell_dx, cell_dy, 0.0],
                [0.0, cell_dy, 0.0],
                [cell_dx, cell_dy, 0.0],
                [-cell_dx, -cell_dy, cell_dz],
                [0.0, -cell_dy, cell_dz],
                [cell_dx, -cell_dy, cell_dz],
                [-cell_dx, 0.0, cell_dz],
                [0.0, 0.0, cell_dz],
                [cell_dx, 0.0, cell_dz],
                [-cell_dx, cell_dy, cell_dz],
                [0.0, cell_dy, cell_dz],
                [cell_dx, cell_dy, cell_dz],
            ],
            dtype=f32,
        )
        return Xijk

    def normal_point_fn(self, point, dx, dy, dz):
        """
        Evaluate normal vector at a given point based on interpolated values
        of the level set function at the face-centers of a 3D cell centered at the
        point with each side length given by dx, dy, dz.
        """
        point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
        phi_x = (self.phi_interp_fn(point_ip1_j_k) - self.phi_interp_fn(point_im1_j_k)) / (2 * dx)

        point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
        phi_y = (self.phi_interp_fn(point_i_jp1_k) - self.phi_interp_fn(point_i_jm1_k)) / (2 * dy)

        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
        phi_z = (self.phi_interp_fn(point_i_j_kp1) - self.phi_interp_fn(point_i_j_km1)) / (2 * dz)

        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    def initialize_neural_based_algorithm(self):
        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        def sign_m_fn(a):
            # returns 1 only if a<0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.ceil(0.5 * sgn - 0.75) * (-1.0)

        self.mask_region_m = sign_m_fn(self.phi_flat)
        self.mask_region_p = sign_p_fn(self.phi_flat)

        self.mask_interface_bandwidth = sign_m_fn(self.phi_flat**2 - self.bandwidth_squared)
        self.mask_non_interface_bandwidth = sign_p_fn(self.phi_flat**2 - self.bandwidth_squared)

    def get_regression_coeffs_at_point(self, point, dx, dy, dz):
        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        def sign_m_fn(a):
            # returns 1 only if a<0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.ceil(0.5 * sgn - 0.75) * (-1.0)

        x, y, z = point
        Xijk = self.get_Xijk(dx, dy, dz)
        curr_vertices = jnp.add(jnp.array([x, y, z]), Xijk)
        phi_vertices = self.phi_interp_fn(curr_vertices)

        Wijk_p = jnp.diag(vmap(sign_p_fn)(phi_vertices))
        Wijk_m = jnp.diag(vmap(sign_m_fn)(phi_vertices))

        Dp = jnp.linalg.pinv(Xijk.T @ Wijk_p @ Xijk) @ (Wijk_p @ Xijk).T
        Dm = jnp.linalg.pinv(Xijk.T @ Wijk_m @ Xijk) @ (Wijk_m @ Xijk).T
        D_m_mat = jnp.nan_to_num(Dm)
        D_p_mat = jnp.nan_to_num(Dp)

        normal_vec = self.normal_point_fn(point, dx, dy, dz).T
        phi_point = self.phi_interp_fn(point[jnp.newaxis])

        Cm_ijk_pqm = normal_vec @ D_m_mat
        Cp_ijk_pqm = normal_vec @ D_p_mat

        zeta_p_ijk_pqm = (
            (self.mu_p_interp_fn(point[jnp.newaxis]) - self.mu_m_interp_fn(point[jnp.newaxis]))
            / self.mu_m_interp_fn(point[jnp.newaxis])
        ) * phi_point
        zeta_p_ijk_pqm = zeta_p_ijk_pqm[..., jnp.newaxis] * Cp_ijk_pqm
        zeta_m_ijk_pqm = (
            (self.mu_p_interp_fn(point[jnp.newaxis]) - self.mu_m_interp_fn(point[jnp.newaxis]))
            / self.mu_p_interp_fn(point[jnp.newaxis])
        ) * phi_point
        zeta_m_ijk_pqm = zeta_m_ijk_pqm[..., jnp.newaxis] * Cm_ijk_pqm
        zeta_p_ijk = (zeta_p_ijk_pqm.sum(axis=1) - zeta_p_ijk_pqm[:, 13]) * f32(-1.0)
        zeta_m_ijk = (zeta_m_ijk_pqm.sum(axis=1) - zeta_m_ijk_pqm[:, 13]) * f32(-1.0)

        gamma_p_ijk_pqm = zeta_p_ijk_pqm / (1.0 + zeta_p_ijk[:, jnp.newaxis])
        gamma_m_ijk_pqm = zeta_m_ijk_pqm / (1.0 - zeta_m_ijk[:, jnp.newaxis])
        gamma_p_ijk = (gamma_p_ijk_pqm.sum(axis=1) - gamma_p_ijk_pqm[:, 13]) * f32(-1.0)
        gamma_m_ijk = (gamma_m_ijk_pqm.sum(axis=1) - gamma_m_ijk_pqm[:, 13]) * f32(-1.0)

        return (
            normal_vec,
            gamma_m_ijk,
            gamma_m_ijk_pqm,
            gamma_p_ijk,
            gamma_p_ijk_pqm,
            zeta_m_ijk,
            zeta_m_ijk_pqm,
            zeta_p_ijk,
            zeta_p_ijk_pqm,
        )

    @staticmethod
    @hk.transform
    def forward(x, phi_x):
        """
        Forward pass of the neural network.

        Args:
            x: input data

        Returns:
            output of the neural network
        """
        model = DoubleMLP()
        return model(x, phi_x)

    @partial(jit, static_argnums=0)
    def init(self, seed=42):
        rng = random.PRNGKey(seed)
        params = self.forward.init(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=0.1)
        opt_state = self.optimizer.init(params)
        return opt_state, params

    @partial(jit, static_argnums=(0))
    def evaluate_solution_fn(self, params, R_flat):
        phi_flat = self.phi_interp_fn(R_flat)
        sol_fn = partial(self.forward.apply, params, None)
        pred_sol = vmap(sol_fn, (0, 0))(R_flat, phi_flat)
        return pred_sol

    def solution_at_point_fn(self, params, r_point, phi_point):
        sol_fn = partial(self.forward.apply, params, None)
        return sol_fn(r_point, phi_point).reshape()

    def get_sol_grad_sol_fn(self, params):
        u_at_point_fn = partial(self.solution_at_point_fn, params)
        grad_u_at_point_fn = grad(u_at_point_fn)
        return u_at_point_fn, grad_u_at_point_fn

    def get_mask_plus(self, points):
        """
        For a set of points, returns 1 if in external region
        returns 0 if inside the geometry.
        """
        phi_points = self.phi_interp_fn(points)

        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        mask_p = sign_p_fn(phi_points)
        return mask_p

    @partial(jit, static_argnums=(0))
    def loss(self, params, points, dx, dy, dz):
        """
        Loss function of the neural network
        """
        lhs_rhs = vmap(self.compute_Ax_and_b_fn, (None, 0, None, None, None))(params, points, dx, dy, dz)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        tot_loss = jnp.mean(optax.l2_loss(lhs, rhs))

        # du_xmax = (self.evaluate_solution_fn(params, self.gstate.R_xmax_boundary) - self.dir_bc_fn(self.gstate.R_xmax_boundary)[...,jnp.newaxis])
        # du_xmin = (self.evaluate_solution_fn(params, self.gstate.R_xmin_boundary) - self.dir_bc_fn(self.gstate.R_xmin_boundary)[...,jnp.newaxis])

        # du_ymax = (self.evaluate_solution_fn(params, self.gstate.R_ymax_boundary) - self.dir_bc_fn(self.gstate.R_ymax_boundary)[...,jnp.newaxis])
        # du_ymin = (self.evaluate_solution_fn(params, self.gstate.R_ymin_boundary) - self.dir_bc_fn(self.gstate.R_ymin_boundary)[...,jnp.newaxis])

        # du_zmax = (self.evaluate_solution_fn(params, self.gstate.R_zmax_boundary) - self.dir_bc_fn(self.gstate.R_zmax_boundary)[...,jnp.newaxis])
        # du_zmin = (self.evaluate_solution_fn(params, self.gstate.R_zmin_boundary) - self.dir_bc_fn(self.gstate.R_zmin_boundary)[...,jnp.newaxis])

        # tot_loss += 0.01 * (jnp.mean(jnp.square(du_xmax)) + jnp.mean(jnp.square(du_xmin)) + jnp.mean(jnp.square(du_ymax)) + jnp.mean(jnp.square(du_ymin)) + jnp.mean(jnp.square(du_zmax)) + jnp.mean(jnp.square(du_zmin)))
        return tot_loss

    @partial(jit, static_argnums=(0))
    def update(self, opt_state, params, points, dx, dy, dz):
        loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def update_multi_gpu(self, opt_state, params, points, dx, dy, dz):
        loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)
        """ Muli-GPU """
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def compute_Ax_and_b_preconditioned_fn(self, params, point, dx, dy, dz):
        """
        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field.

        * PROCEDURE:
            first compute u = B:u + r for each node
            then use the actual cell geometries (face areas and mu coeffs) to
            compute the rhs of the linear system given currently passed-in u vector
            for solution estimate.

        """

        u_mp_at_point = partial(self.u_mp_fn, params, dx, dy, dz)

        def is_box_boundary_point(point):
            """
            Check if current node is on the boundary of box
            """
            x, y, z = point
            boundary = jnp.where(abs(x - self.xmin) < 1e-6 * dx, 0, 1) * jnp.where(
                abs(x - self.xmax) < 1e-6 * dx, 0, 1
            )
            boundary *= jnp.where(abs(y - self.ymin) < 1e-6 * dy, 0, 1) * jnp.where(
                abs(y - self.ymax) < 1e-6 * dy, 0, 1
            )
            boundary *= jnp.where(abs(z - self.zmin) < 1e-6 * dz, 0, 1) * jnp.where(
                abs(z - self.zmax) < 1e-6 * dz, 0, 1
            )
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz):
            # --- LHS
            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(point, dx, dy, dz)
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]
            Vol_cell_nominal = dx * dy * dz

            def get_lhs_at_interior_point(point):
                point_ijk = point
                point_imjk = jnp.array([point[0] - dx, point[1], point[2]], dtype=f32)
                point_ipjk = jnp.array([point[0] + dx, point[1], point[2]], dtype=f32)
                point_ijmk = jnp.array([point[0], point[1] - dy, point[2]], dtype=f32)
                point_ijpk = jnp.array([point[0], point[1] + dy, point[2]], dtype=f32)
                point_ijkm = jnp.array([point[0], point[1], point[2] - dz], dtype=f32)
                point_ijkp = jnp.array([point[0], point[1], point[2] + dz], dtype=f32)

                k_m_ijk = self.k_m_interp_fn(point[jnp.newaxis])
                k_p_ijk = self.k_p_interp_fn(point[jnp.newaxis])

                u_m_ijk, u_p_ijk = u_mp_at_point(point_ijk)
                u_m_imjk, u_p_imjk = u_mp_at_point(point_imjk)
                u_m_ipjk, u_p_ipjk = u_mp_at_point(point_ipjk)
                u_m_ijmk, u_p_ijmk = u_mp_at_point(point_ijmk)
                u_m_ijpk, u_p_ijpk = u_mp_at_point(point_ijpk)
                u_m_ijkm, u_p_ijkm = u_mp_at_point(point_ijkm)
                u_m_ijkp, u_p_ijkp = u_mp_at_point(point_ijkp)

                lhs = k_m_ijk * V_m_ijk * u_m_ijk
                lhs += k_p_ijk * V_p_ijk * u_p_ijk
                lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + (
                    coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]
                ) * u_p_ijk
                lhs += -1.0 * coeffs[0] * u_m_imjk - coeffs[1] * u_p_imjk
                lhs += -1.0 * coeffs[2] * u_m_ipjk - coeffs[3] * u_p_ipjk
                lhs += -1.0 * coeffs[4] * u_m_ijmk - coeffs[5] * u_p_ijmk
                lhs += -1.0 * coeffs[6] * u_m_ijpk - coeffs[7] * u_p_ijpk
                lhs += -1.0 * coeffs[8] * u_m_ijkm - coeffs[9] * u_p_ijkm
                lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

                diag_coeff = (
                    k_p_ijk * V_p_ijk
                    + k_m_ijk * V_m_ijk
                    + (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10])
                    + (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11])
                )
                return jnp.array([lhs.reshape(), diag_coeff.reshape()])

            def get_lhs_on_box_boundary(point):
                phi_boundary = self.phi_interp_fn(point[jnp.newaxis])
                u_boundary = self.solution_at_point_fn(params, point, phi_boundary)
                lhs = u_boundary * Vol_cell_nominal
                return jnp.array([lhs, Vol_cell_nominal])

            lhs_diagcoeff = jnp.where(
                is_box_boundary_point(point),
                get_lhs_on_box_boundary(point),
                get_lhs_at_interior_point(point),
            )
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

            # --- RHS
            def get_rhs_at_interior_point(point):
                rhs = (
                    self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk + self.f_p_interp_fn(point[jnp.newaxis]) * V_p_ijk
                )
                rhs += self.beta_integrate_over_interface_at_point(point, dx, dy, dz)
                return rhs

            def get_rhs_on_box_boundary(point):
                return self.dir_bc_fn(point[jnp.newaxis]).reshape() * Vol_cell_nominal

            rhs = jnp.where(
                is_box_boundary_point(point),
                get_rhs_on_box_boundary(point),
                get_rhs_at_interior_point(point),
            )
            lhs_over_diag = jnp.nan_to_num(lhs / diagcoeff)
            rhs_over_diag = jnp.nan_to_num(rhs / diagcoeff)
            return jnp.array([lhs_over_diag, rhs_over_diag])

        lhs_rhs = evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz)
        return lhs_rhs

    @partial(jit, static_argnums=(0))
    def get_u_mp_by_regression_at_point_fn(self, params, dx, dy, dz, point):
        """
        This function evaluates pairs of u^+ and u^- at each grid point
        in the domain, given the neural network models.

        BIAS SLOW:
            This function evaluates
                u_m = B_m : u + r_m
            and
                u_p = B_p : u + r_p
        """
        delta_ijk = self.phi_interp_fn(point[jnp.newaxis])
        u_ijk = self.solution_at_point_fn(params, point, delta_ijk)
        Xijk = self.get_Xijk(dx, dy, dz)

        curr_vertices = jnp.add(point, Xijk)
        u_cube_ijk = self.evaluate_solution_fn(params, curr_vertices)

        (
            normal_ijk,
            gamma_m_ijk,
            gamma_m_ijk_pqm,
            gamma_p_ijk,
            gamma_p_ijk_pqm,
            zeta_m_ijk,
            zeta_m_ijk_pqm,
            zeta_p_ijk,
            zeta_p_ijk_pqm,
        ) = self.get_regression_coeffs_at_point(point, dx, dy, dz)

        def bulk_point(is_interface_, u_ijk_):
            return jnp.array(
                [
                    jnp.where(is_interface_ == -1, u_ijk_, 0.0),
                    jnp.where(is_interface_ == 1, u_ijk_, 0.0),
                ]
            )

        def interface_point(point):
            def mu_minus_bigger_fn(point):
                def extrapolate_u_m_from_negative_domain(r_ijk):
                    r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_m = -1.0 * jnp.dot(gamma_m_ijk_pqm, u_cube_ijk)
                    u_m += (1.0 - gamma_m_ijk + gamma_m_ijk_pqm[:, 13]) * u_ijk
                    u_m += (
                        -1.0
                        * (1.0 - gamma_m_ijk)
                        * (self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_p_interp_fn(r_m_proj))
                    )
                    return u_m.reshape()

                def extrapolate_u_p_from_positive_domain(r_ijk):
                    r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk[0]
                    u_p = -1.0 * jnp.dot(zeta_m_ijk_pqm, u_cube_ijk)
                    u_p += (1.0 - zeta_m_ijk + zeta_m_ijk_pqm[:, 13]) * u_ijk
                    u_p += self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_over_mu_p_interp_fn(r_p_proj)
                    return u_p.reshape()

                u_m = jnp.where(delta_ijk > 0, extrapolate_u_m_from_negative_domain(point), u_ijk)[0]
                u_p = jnp.where(delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain(point))[0]
                return jnp.array([u_m, u_p])

            def mu_plus_bigger_fn(point):
                def extrapolate_u_m_from_negative_domain_(r_ijk):
                    r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_m = -1.0 * jnp.dot(zeta_p_ijk_pqm, u_cube_ijk)
                    u_m += (1.0 - zeta_p_ijk + zeta_p_ijk_pqm[:, 13]) * u_ijk
                    u_m += (-1.0) * (
                        self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_m_proj)
                    )
                    return u_m.reshape()

                def extrapolate_u_p_from_positive_domain_(r_ijk):
                    r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_p = -1.0 * jnp.dot(gamma_p_ijk_pqm, u_cube_ijk)
                    u_p += (1.0 - gamma_p_ijk + gamma_p_ijk_pqm[:, 13]) * u_ijk
                    u_p += (1.0 - gamma_p_ijk) * (
                        self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_p_proj)
                    )
                    return u_p.reshape()

                u_m = jnp.where(delta_ijk > 0, extrapolate_u_m_from_negative_domain_(point), u_ijk)[0]
                u_p = jnp.where(delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain_(point))[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = self.mu_m_interp_fn(point[jnp.newaxis])
            mu_p_ijk = self.mu_p_interp_fn(point[jnp.newaxis])
            return jnp.where(mu_m_ijk > mu_p_ijk, mu_minus_bigger_fn(point), mu_plus_bigger_fn(point))

        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface(point, dx, dy, dz)
        # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
        u_mp = jnp.where(is_interface == 0, interface_point(point), bulk_point(is_interface, u_ijk))
        return u_mp

    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, params, points, dx, dy, dz):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
        normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
        grad_n_u_m = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_m)
        grad_n_u_p = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_p)
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp_neural_network(self, params, points):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
        return grad_u_m, grad_u_p

    def compute_normal_gradient_solution_on_interface_neural_network(self, params, points, dx, dy, dz):
        phi_flat = self.phi_interp_fn(points)
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
        normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
        grad_n_u = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u)
        return grad_n_u

    def compute_gradient_solution_neural_network(self, params, points):
        phi_flat = self.phi_interp_fn(points)
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
        return grad_u
