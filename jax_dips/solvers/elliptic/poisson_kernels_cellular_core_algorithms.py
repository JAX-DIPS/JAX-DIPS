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


import jax
from jax import numpy as jnp, vmap, jit, grad, random, nn as jnn, value_and_grad, config
from jax_dips.geometry import geometric_integrations

config.update("jax_debug_nans", False)
import optax
import haiku as hk

import numpy as onp

from jax_dips.domain import interpolate
from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.nn.nn_solution_model import DoubleMLP
from jax_dips.utils.inspect import print_architecture

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functools import partial
import time


class PDETrainer:
    def __init__(self, gstate, sim_state, optimizer, algorithm=0, precondition=1):
        self.optimizer = optimizer
        self.gstate = gstate
        self.sim_state = sim_state

        self.algorithm = algorithm
        """
        algorithm = 0: use regression to evaluate u^\pm
        algorithm = 1: use neural network to evaluate u^\pm
        """

        phi_n = sim_state.phi
        dirichlet_bc = sim_state.dirichlet_bc
        mu_m = sim_state.mu_m
        mu_p = sim_state.mu_p
        k_m = sim_state.k_m
        k_p = sim_state.k_p
        f_m = sim_state.f_m
        f_p = sim_state.f_p
        alpha = sim_state.alpha
        beta = sim_state.beta

        xo = gstate.x
        yo = gstate.y
        zo = gstate.z
        dx = xo[2] - xo[1]
        dy = yo[2] - yo[1]
        dz = zo[2] - zo[1]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        Nx = xo.shape[0]
        Ny = yo.shape[0]
        Nz = zo.shape[0]
        grid_shape = (Nx, Ny, Nz)
        ii = onp.arange(2, Nx + 2)
        jj = onp.arange(2, Ny + 2)
        kk = onp.arange(2, Nz + 2)
        I, J, K = onp.meshgrid(ii, jj, kk, indexing="ij")
        self.nodes = jnp.array(onp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1))))

        self.bandwidth_squared = (2.0 * self.dx) * (2.0 * self.dx)

        self.phi_cube_ = phi_n.reshape(grid_shape)
        x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, self.phi_cube_)
        x, y, z, self.phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)

        self.mu_m_cube_internal = mu_m.reshape(grid_shape)
        self.mu_p_cube_internal = mu_p.reshape(grid_shape)

        self.mu_m_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(mu_m, gstate)
        self.mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(mu_p, gstate)
        self.alpha_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(alpha, gstate)
        self.beta_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(beta, gstate)

        mu_m_over_mu_p = mu_m / mu_p
        beta_over_mu_m = beta / mu_m
        beta_over_mu_p = beta / mu_p
        self.mu_m_over_mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(mu_m_over_mu_p, gstate)
        self.beta_over_mu_m_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(beta_over_mu_m, gstate)
        self.beta_over_mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(beta_over_mu_p, gstate)

        self.dirichlet_cube = dirichlet_bc.reshape(grid_shape)
        self.k_m_cube_internal = k_m.reshape(grid_shape)
        self.k_p_cube_internal = k_p.reshape(grid_shape)
        self.f_m_cube_internal = f_m.reshape(grid_shape)
        self.f_p_cube_internal = f_p.reshape(grid_shape)

        self.phi_flat = self.phi_cube_.reshape(-1)
        self.Vol_cell_nominal = dx * dy * dz
        self.grid_shape = grid_shape

        (
            get_vertices_of_cell_intersection_with_interface_at_node,
            self.is_cell_crossed_by_interface,
        ) = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(gstate, sim_state)
        (
            self.beta_integrate_over_interface_at_node,
            _,
        ) = geometric_integrations.integrate_over_gamma_and_omega_m(
            get_vertices_of_cell_intersection_with_interface_at_node,
            self.is_cell_crossed_by_interface,
            self.beta_interp_fn,
        )
        self.compute_face_centroids_values_plus_minus_at_node = geometric_integrations.compute_cell_faces_areas_values(
            gstate,
            get_vertices_of_cell_intersection_with_interface_at_node,
            self.is_cell_crossed_by_interface,
            self.mu_m_interp_fn,
            self.mu_p_interp_fn,
        )

        self.initialize_algorithms()
        if precondition == 1:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_preconditioned_fn
        elif precondition == 0:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_vanilla_fn

    def initialize_algorithms(self):
        self.normal_vec_fn = partial(
            self.normal_vector_fn,
            phi_cube=self.phi_cube,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
        )
        self.normal_vecs = vmap(self.normal_vec_fn)(self.nodes)

        if self.algorithm == 0:
            self.initialize_regression_based_algorithm()
            self.u_mp_fn = self.get_u_mp_by_regression_at_node_fn
            self.compute_normal_gradient_solution_mp_on_interface = (
                self.compute_normal_gradient_solution_mp_on_interface_regression
            )
            self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_regression

        elif self.algorithm == 1:
            self.initialize_neural_based_algorithm()
            self.u_mp_fn = self.get_u_mp_by_neural_network_at_node_fn
            self.compute_normal_gradient_solution_mp_on_interface = (
                self.compute_normal_gradient_solution_mp_on_interface_neural_network
            )
            self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_neural_network

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

    def initialize_regression_based_algorithm(self):
        dx = self.dx
        dy = self.dy
        dz = self.dz

        def get_X_ijk():
            Xijk = jnp.array(
                [
                    [-dx, -dy, -dz],
                    [0.0, -dy, -dz],
                    [dx, -dy, -dz],
                    [-dx, 0.0, -dz],
                    [0.0, 0.0, -dz],
                    [dx, 0.0, -dz],
                    [-dx, dy, -dz],
                    [0.0, dy, -dz],
                    [dx, dy, -dz],
                    [-dx, -dy, 0.0],
                    [0.0, -dy, 0.0],
                    [dx, -dy, 0.0],
                    [-dx, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [dx, 0.0, 0.0],
                    [-dx, dy, 0.0],
                    [0.0, dy, 0.0],
                    [dx, dy, 0.0],
                    [-dx, -dy, dz],
                    [0.0, -dy, dz],
                    [dx, -dy, dz],
                    [-dx, 0.0, dz],
                    [0.0, 0.0, dz],
                    [dx, 0.0, dz],
                    [-dx, dy, dz],
                    [0.0, dy, dz],
                    [dx, dy, dz],
                ],
                dtype=f32,
            )

            ngbs = jnp.array(
                [
                    [-1, -1, -1],
                    [0, -1, -1],
                    [1, -1, -1],
                    [-1, 0, -1],
                    [0, 0, -1],
                    [1, 0, -1],
                    [-1, 1, -1],
                    [0, 1, -1],
                    [1, 1, -1],
                    [-1, -1, 0],
                    [0, -1, 0],
                    [1, -1, 0],
                    [-1, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 1, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, -1, 1],
                    [0, -1, 1],
                    [1, -1, 1],
                    [-1, 0, 1],
                    [0, 0, 1],
                    [1, 0, 1],
                    [-1, 1, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=i32,
            )

            return Xijk, ngbs

        Xijk, self.ngbs = get_X_ijk()

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
        # plt.imshow(self.mask_interface_bandwidth.reshape((16, 16, 16))[8]);  plt.colorbar(); plt.savefig('test__.png')
        # plt.imshow((self.mask_region_m + self.mask_region_p).reshape((16,16,16))[8]); plt.colorbar(); plt.savefig('test__.png')

        def cube_at(cube, ind):
            return cube[ind[0], ind[1], ind[2]]

        self.cube_at_v = vmap(cube_at, (None, 0))

        def get_W_p_fn(cube, inds):
            return jnp.diag(vmap(sign_p_fn)(self.cube_at_v(cube, inds)))

        def get_W_m_fn(cube, inds):
            return jnp.diag(vmap(sign_m_fn)(self.cube_at_v(cube, inds)))

        def get_W_pm_matrices(node, phi_cube):
            i, j, k = node
            curr_ngbs = jnp.add(jnp.array([i, j, k]), self.ngbs)
            Wp = get_W_p_fn(phi_cube, curr_ngbs)
            Wm = get_W_m_fn(phi_cube, curr_ngbs)
            return Wm, Wp

        def D_mp_node_update(node, phi_cube):
            Wijk_m, Wijk_p = get_W_pm_matrices(node, phi_cube)
            Dp = jnp.linalg.pinv(Xijk.T @ Wijk_p @ Xijk) @ (Wijk_p @ Xijk).T
            Dm = jnp.linalg.pinv(Xijk.T @ Wijk_m @ Xijk) @ (Wijk_m @ Xijk).T
            return jnp.nan_to_num(Dm), jnp.nan_to_num(Dp)

        D_mp_fn = vmap(D_mp_node_update, (0, None))

        self.D_m_mat, self.D_p_mat = D_mp_fn(self.nodes, self.phi_cube)

        def get_c_ijk_pqm(normal_ijk, D_ijk):
            return normal_ijk @ D_ijk

        get_c_ijk_pqm_vec = vmap(get_c_ijk_pqm, (0, 0))

        self.Cm_ijk_pqm = get_c_ijk_pqm_vec(self.normal_vecs, self.D_m_mat)
        self.Cp_ijk_pqm = get_c_ijk_pqm_vec(self.normal_vecs, self.D_p_mat)

        self.zeta_p_ijk_pqm = (
            (self.mu_p_cube_internal - self.mu_m_cube_internal) / self.mu_m_cube_internal
        ) * self.phi_cube_
        self.zeta_p_ijk_pqm = self.zeta_p_ijk_pqm[..., jnp.newaxis] * self.Cp_ijk_pqm.reshape(
            self.phi_cube_.shape + (-1,)
        )

        self.zeta_m_ijk_pqm = (
            (self.mu_p_cube_internal - self.mu_m_cube_internal) / self.mu_p_cube_internal
        ) * self.phi_cube_
        self.zeta_m_ijk_pqm = self.zeta_m_ijk_pqm[..., jnp.newaxis] * self.Cm_ijk_pqm.reshape(
            self.phi_cube_.shape + (-1,)
        )

        """
        NOTE: zeta_m_ijk_pqm and zeta_p_ijk_pqm are the size of the original grid, not the ghost layers included!
        for example: zeta_m_ijk_pqm[4,4,4][13] is the p=q=m=0 index, and zeta_m_ijk_pqm.shape = (128, 128, 128, 27)
        """
        self.zeta_p_ijk = (self.zeta_p_ijk_pqm.sum(axis=3) - self.zeta_p_ijk_pqm[:, :, :, 13]) * f32(-1.0)
        self.zeta_m_ijk = (self.zeta_m_ijk_pqm.sum(axis=3) - self.zeta_m_ijk_pqm[:, :, :, 13]) * f32(-1.0)

        self.gamma_p_ijk_pqm = self.zeta_p_ijk_pqm / (1.0 + self.zeta_p_ijk[:, :, :, jnp.newaxis])
        self.gamma_m_ijk_pqm = self.zeta_m_ijk_pqm / (1.0 - self.zeta_m_ijk[:, :, :, jnp.newaxis])

        self.gamma_p_ijk = (self.gamma_p_ijk_pqm.sum(axis=3) - self.gamma_p_ijk_pqm[:, :, :, 13]) * f32(-1.0)
        self.gamma_m_ijk = (self.gamma_m_ijk_pqm.sum(axis=3) - self.gamma_m_ijk_pqm[:, :, :, 13]) * f32(-1.0)

    @staticmethod
    def normal_vector_fn(node, phi_cube, dx, dy, dz):
        i, j, k = node
        phi_x = (phi_cube[i + 1, j, k] - phi_cube[i - 1, j, k]) / (f32(2) * dx)
        phi_y = (phi_cube[i, j + 1, k] - phi_cube[i, j - 1, k]) / (f32(2) * dy)
        phi_z = (phi_cube[i, j, k + 1] - phi_cube[i, j, k - 1]) / (f32(2) * dz)
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

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

    # @partial(jit, static_argnums=0)
    def init(self, seed=42):
        rng = random.PRNGKey(seed)
        params = self.forward.init(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=0.1)
        opt_state = self.optimizer.init(params)
        return opt_state, params

    @partial(jit, static_argnums=(0))
    def evaluate_solution_fn(self, params, R_flat, phi_flat):
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

    @partial(jit, static_argnums=(0))
    def evaluate_loss_fn(self, lhs, rhs, sol_cube):
        """
        Weighted L2 loss with exp(-\phi^2) to emphasize error around boundaries
        """
        tot_loss = jnp.mean(optax.l2_loss(lhs, rhs))
        tot_loss += jnp.square(sol_cube[0, :, :] - self.dirichlet_cube[0, :, :]).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[-1, :, :] - self.dirichlet_cube[-1, :, :]).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[:, 0, :] - self.dirichlet_cube[:, 0, :]).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[:, -1, :] - self.dirichlet_cube[:, -1, :]).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[:, :, 0] - self.dirichlet_cube[:, :, 0]).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[:, :, -1] - self.dirichlet_cube[:, :, -1]).mean() * self.Vol_cell_nominal
        return tot_loss

    @partial(jit, static_argnums=(0))
    def evaluate_loss_m_region_fn(self, lhs_, rhs_):
        """
        in the minus region only
        """
        mask_region = self.mask_region_m[:, jnp.newaxis, jnp.newaxis]
        lhs = jnp.multiply(mask_region, lhs_)
        rhs = jnp.multiply(mask_region, rhs_)
        loss_m = jnp.mean(optax.l2_loss(lhs, rhs))
        return loss_m

    @partial(jit, static_argnums=(0))
    def evaluate_loss_p_region_fn(self, lhs_, rhs_, sol_cube):
        """
        in the plus region only
        """
        mask_region = self.mask_region_p[:, jnp.newaxis, jnp.newaxis]
        lhs = jnp.multiply(mask_region, lhs_)
        rhs = jnp.multiply(mask_region, rhs_)

        loss_p = jnp.mean(optax.l2_loss(lhs, rhs))
        loss_p += jnp.square(sol_cube[0, :, :] - self.dirichlet_cube[0, :, :]).mean() * self.Vol_cell_nominal
        loss_p += jnp.square(sol_cube[-1, :, :] - self.dirichlet_cube[-1, :, :]).mean() * self.Vol_cell_nominal
        loss_p += jnp.square(sol_cube[:, 0, :] - self.dirichlet_cube[:, 0, :]).mean() * self.Vol_cell_nominal
        loss_p += jnp.square(sol_cube[:, -1, :] - self.dirichlet_cube[:, -1, :]).mean() * self.Vol_cell_nominal
        loss_p += jnp.square(sol_cube[:, :, 0] - self.dirichlet_cube[:, :, 0]).mean() * self.Vol_cell_nominal
        loss_p += jnp.square(sol_cube[:, :, -1] - self.dirichlet_cube[:, :, -1]).mean() * self.Vol_cell_nominal
        return loss_p

    @partial(jit, static_argnums=(0))
    def evaluate_loss_region_fn(self, lhs_, rhs_, sol_cube, region):
        """
        region=0: everywhere
        region>0: in the plus sign region       / outside interface banded region
        region<0: in the negative sign region  / in the interface banded region
        """
        region_sgn = jnp.sign(region)
        half_region_sgn = 0.5 * region_sgn

        mask_region = (0.5 + half_region_sgn) * self.mask_region_p[:, jnp.newaxis, jnp.newaxis] + (
            0.5 - half_region_sgn
        ) * self.mask_region_m[:, jnp.newaxis, jnp.newaxis]
        # mask_region = (0.5 + half_region_sgn)*self.mask_non_interface_bandwidth[:, jnp.newaxis, jnp.newaxis] + (0.5 - half_region_sgn) * self.mask_interface_bandwidth[:, jnp.newaxis, jnp.newaxis]

        lhs = jnp.multiply(mask_region, lhs_)
        rhs = jnp.multiply(mask_region, rhs_)

        loss_p = jnp.mean(optax.l2_loss(lhs, rhs))
        loss_p += (
            jnp.square(sol_cube[0, :, :] - self.dirichlet_cube[0, :, :]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        loss_p += (
            jnp.square(sol_cube[-1, :, :] - self.dirichlet_cube[-1, :, :]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        loss_p += (
            jnp.square(sol_cube[:, 0, :] - self.dirichlet_cube[:, 0, :]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        loss_p += (
            jnp.square(sol_cube[:, -1, :] - self.dirichlet_cube[:, -1, :]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        loss_p += (
            jnp.square(sol_cube[:, :, 0] - self.dirichlet_cube[:, :, 0]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        loss_p += (
            jnp.square(sol_cube[:, :, -1] - self.dirichlet_cube[:, :, -1]).mean()
            * self.Vol_cell_nominal
            * (0.5 + half_region_sgn)
        )
        return loss_p

    def loss_region(self, params, region):
        """
        region=0: everywhere
        region>0: in the plus sign region
        region<0: in the negative sign region
        """
        lhs_rhs = self.compute_Ax_and_b_fn(params)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        pred_sol = self.evaluate_solution_fn(params, self.gstate.R, self.phi_flat)
        sol_cube = pred_sol.reshape(self.grid_shape)
        tot_loss = self.evaluate_loss_region_fn(lhs, rhs, sol_cube, region)
        return tot_loss

    @partial(jit, static_argnums=(0))
    def update_region(self, opt_state, params, region=0):
        """
        region=0: everywhere
        region>0: in the plus sign region
        region<0: in the negative sign region
        """
        loss, grads = value_and_grad(self.loss_region)(params, region)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    def loss_m(self, params):
        lhs_rhs = self.compute_Ax_and_b_fn(params)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        loss_m = self.evaluate_loss_m_region_fn(lhs, rhs)
        return loss_m

    def loss_p(self, params):
        lhs_rhs = self.compute_Ax_and_b_fn(params)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        pred_sol = self.evaluate_solution_fn(params, self.gstate.R, self.phi_flat)
        sol_cube = pred_sol.reshape(self.grid_shape)
        loss_p = self.evaluate_loss_p_region_fn(lhs, rhs, sol_cube)
        return loss_p

    def loss(self, params):
        """
        Loss function of the neural network
        """
        # jax.make_jaxpr(self.evaluate_solution_fn)(params, self.gstate.R, self.phi_flat).pretty_print()
        # jax.make_jaxpr(self.compute_Ax_and_b_fn)(params).pretty_print()
        lhs_rhs = self.compute_Ax_and_b_fn(params)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        pred_sol = self.evaluate_solution_fn(params, self.gstate.R, self.phi_flat)
        sol_cube = pred_sol.reshape(self.grid_shape)
        tot_loss = self.evaluate_loss_fn(lhs, rhs, sol_cube)
        return tot_loss

    @partial(jit, static_argnums=(0))
    def update_m(self, opt_state, params):
        loss, grads = value_and_grad(self.loss_m)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def update_p(self, opt_state, params):
        loss, grads = value_and_grad(self.loss_p)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def update(self, opt_state, params):
        loss, grads = value_and_grad(self.loss)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def compute_Ax_and_b_preconditioned_fn(self, params):
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

        x = self.gstate.x
        y = self.gstate.y
        z = self.gstate.z

        Nx, Ny, Nz = self.grid_shape

        u = self.evaluate_solution_fn(params, self.gstate.R, self.phi_flat)
        u_cube = u.reshape(self.grid_shape)

        # u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z, params)
        if self.algorithm == 0:
            u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z)
        elif self.algorithm == 1:
            u_mp_at_node = partial(self.u_mp_fn, params, x, y, z)

        def is_box_boundary_node(i, j, k):
            """
            Check if current node is on the boundary of box
            """
            boundary = (i - 2) * (i - Nx - 1) * (j - 2) * (j - Ny - 1) * (k - 2) * (k - Nz - 1)
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_node(node):
            # --- LHS
            i, j, k = node

            coeffs_ = self.compute_face_centroids_values_plus_minus_at_node(node)
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]

            def get_lhs_at_interior_node(node):
                i, j, k = node

                k_m_ijk = self.k_m_cube_internal[i - 2, j - 2, k - 2]
                k_p_ijk = self.k_p_cube_internal[i - 2, j - 2, k - 2]

                u_m_ijk, u_p_ijk = u_mp_at_node(i, j, k)
                u_m_imjk, u_p_imjk = u_mp_at_node(i - 1, j, k)
                u_m_ipjk, u_p_ipjk = u_mp_at_node(i + 1, j, k)
                u_m_ijmk, u_p_ijmk = u_mp_at_node(i, j - 1, k)
                u_m_ijpk, u_p_ijpk = u_mp_at_node(i, j + 1, k)
                u_m_ijkm, u_p_ijkm = u_mp_at_node(i, j, k - 1)
                u_m_ijkp, u_p_ijkp = u_mp_at_node(i, j, k + 1)

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
                return jnp.array([lhs, diag_coeff])

            def get_lhs_on_box_boundary(node):
                i, j, k = node
                lhs = u_cube[i - 2, j - 2, k - 2] * self.Vol_cell_nominal
                return jnp.array([lhs, self.Vol_cell_nominal])

            lhs_diagcoeff = jnp.where(
                is_box_boundary_node(i, j, k),
                get_lhs_on_box_boundary(node),
                get_lhs_at_interior_node(node),
            )
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

            # diagcoeff_ = jnp.sqrt(diagcoeff*diagcoeff)
            # --- RHS
            def get_rhs_at_interior_node(node):
                i, j, k = node
                rhs = (
                    self.f_m_cube_internal[i - 2, j - 2, k - 2] * V_m_ijk
                    + self.f_p_cube_internal[i - 2, j - 2, k - 2] * V_p_ijk
                )
                rhs += self.beta_integrate_over_interface_at_node(node)
                return rhs

            def get_rhs_on_box_boundary(node):
                i, j, k = node
                return self.dirichlet_cube[i - 2, j - 2, k - 2] * self.Vol_cell_nominal

            rhs = jnp.where(
                is_box_boundary_node(i, j, k),
                get_rhs_on_box_boundary(node),
                get_rhs_at_interior_node(node),
            )
            lhs_over_diag = jnp.nan_to_num(lhs / diagcoeff)
            rhs_over_diag = jnp.nan_to_num(rhs / diagcoeff)
            return jnp.array([lhs_over_diag, rhs_over_diag])
            # return jnp.array([lhs / (1e-13 + diagcoeff_), rhs / (1e-13 + diagcoeff_)])

        evaluate_on_nodes_fn = vmap(evaluate_discretization_lhs_rhs_at_node)

        lhs_rhs = evaluate_on_nodes_fn(self.nodes)
        return lhs_rhs

    @partial(jit, static_argnums=(0))
    def compute_Ax_and_b_vanilla_fn(self, params):
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

        x = self.gstate.x
        y = self.gstate.y
        z = self.gstate.z

        Nx, Ny, Nz = self.grid_shape

        u = self.evaluate_solution_fn(params, self.gstate.R, self.phi_flat)
        u_cube = u.reshape(self.grid_shape)

        # u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z, params)
        if self.algorithm == 0:
            u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z)
        elif self.algorithm == 1:
            u_mp_at_node = partial(self.u_mp_fn, params, x, y, z)

        def is_box_boundary_node(i, j, k):
            """
            Check if current node is on the boundary of box
            """
            boundary = (i - 2) * (i - Nx - 1) * (j - 2) * (j - Ny - 1) * (k - 2) * (k - Nz - 1)
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_node(node):
            # --- LHS
            i, j, k = node

            coeffs_ = self.compute_face_centroids_values_plus_minus_at_node(node)
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]

            def get_lhs_at_interior_node(node):
                i, j, k = node

                k_m_ijk = self.k_m_cube_internal[i - 2, j - 2, k - 2]
                k_p_ijk = self.k_p_cube_internal[i - 2, j - 2, k - 2]

                u_m_ijk, u_p_ijk = u_mp_at_node(i, j, k)
                u_m_imjk, u_p_imjk = u_mp_at_node(i - 1, j, k)
                u_m_ipjk, u_p_ipjk = u_mp_at_node(i + 1, j, k)
                u_m_ijmk, u_p_ijmk = u_mp_at_node(i, j - 1, k)
                u_m_ijpk, u_p_ijpk = u_mp_at_node(i, j + 1, k)
                u_m_ijkm, u_p_ijkm = u_mp_at_node(i, j, k - 1)
                u_m_ijkp, u_p_ijkp = u_mp_at_node(i, j, k + 1)

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

                return lhs

            def get_lhs_on_box_boundary(node):
                i, j, k = node
                lhs = u_cube[i - 2, j - 2, k - 2] * self.Vol_cell_nominal
                return lhs

            lhs = jnp.where(
                is_box_boundary_node(i, j, k),
                get_lhs_on_box_boundary(node),
                get_lhs_at_interior_node(node),
            )

            # --- RHS
            def get_rhs_at_interior_node(node):
                i, j, k = node
                rhs = (
                    self.f_m_cube_internal[i - 2, j - 2, k - 2] * V_m_ijk
                    + self.f_p_cube_internal[i - 2, j - 2, k - 2] * V_p_ijk
                )
                rhs += self.beta_integrate_over_interface_at_node(node)
                return rhs

            def get_rhs_on_box_boundary(node):
                i, j, k = node
                return self.dirichlet_cube[i - 2, j - 2, k - 2] * self.Vol_cell_nominal

            rhs = jnp.where(
                is_box_boundary_node(i, j, k),
                get_rhs_on_box_boundary(node),
                get_rhs_at_interior_node(node),
            )

            return jnp.array([lhs, rhs])

        evaluate_on_nodes_fn = vmap(evaluate_discretization_lhs_rhs_at_node)
        lhs_rhs = evaluate_on_nodes_fn(self.nodes)
        return lhs_rhs

    @partial(jit, static_argnums=(0))
    def get_u_mp_by_regression_at_node_fn(self, u_cube, x, y, z, i, j, k):
        """
        This function evaluates pairs of u^+ and u^- at each grid point
        in the domain, given the neural network models.

        BIAS SLOW:
            This function evaluates
                u_m = B_m : u + r_m
            and
                u_p = B_p : u + r_p
        """
        u_ijk = u_cube[i - 2, j - 2, k - 2]
        delta_ijk = self.phi_cube[i, j, k]

        def bulk_node(is_interface_, u_ijk_):
            return jnp.array(
                [
                    jnp.where(is_interface_ == -1, u_ijk_, 0.0),
                    jnp.where(is_interface_ == 1, u_ijk_, 0.0),
                ]
            )

        def interface_node(i, j, k):
            def mu_minus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain(i, j, k):
                    r_ijk = jnp.array([x[i - 2], y[j - 2], z[k - 2]], dtype=f32)
                    r_m_proj = r_ijk - delta_ijk * self.normal_vec_fn((i, j, k))
                    r_m_proj = r_m_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
                    u_m = -1.0 * jnp.dot(
                        self.gamma_m_ijk_pqm[i - 2, j - 2, k - 2],
                        self.cube_at_v(u_cube, curr_ngbs),
                    )
                    u_m += (
                        1.0 - self.gamma_m_ijk[i - 2, j - 2, k - 2] + self.gamma_m_ijk_pqm[i - 2, j - 2, k - 2, 13]
                    ) * u_cube[i - 2, j - 2, k - 2]
                    # u_m += -1.0 * (1.0 - self.gamma_m_ijk[i-2, j-2, k-2]) * (self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_interp_fn(r_m_proj) / self.mu_p_interp_fn(r_m_proj))
                    u_m += (
                        -1.0
                        * (1.0 - self.gamma_m_ijk[i - 2, j - 2, k - 2])
                        * (self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_p_interp_fn(r_m_proj))
                    )
                    return u_m

                def extrapolate_u_p_from_positive_domain(i, j, k):
                    r_ijk = jnp.array([x[i - 2], y[j - 2], z[k - 2]], dtype=f32)
                    r_p_proj = r_ijk - delta_ijk * self.normal_vec_fn((i, j, k))
                    r_p_proj = r_p_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
                    u_p = -1.0 * jnp.dot(
                        self.zeta_m_ijk_pqm[i - 2, j - 2, k - 2],
                        self.cube_at_v(u_cube, curr_ngbs),
                    )
                    u_p += (
                        1.0 - self.zeta_m_ijk[i - 2, j - 2, k - 2] + self.zeta_m_ijk_pqm[i - 2, j - 2, k - 2, 13]
                    ) * u_cube[i - 2, j - 2, k - 2]
                    # u_p += self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_interp_fn(r_p_proj) / self.mu_p_interp_fn(r_p_proj)
                    u_p += self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_over_mu_p_interp_fn(r_p_proj)
                    return u_p

                phi_ijk = self.phi_cube[i, j, k]
                u_m = jnp.where(
                    phi_ijk > 0,
                    extrapolate_u_m_from_negative_domain(i, j, k),
                    u_cube[i - 2, j - 2, k - 2],
                )[0]
                u_p = jnp.where(
                    phi_ijk > 0,
                    u_cube[i - 2, j - 2, k - 2],
                    extrapolate_u_p_from_positive_domain(i, j, k),
                )[0]
                return jnp.array([u_m, u_p])

            def mu_plus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain_(i, j, k):
                    r_ijk = jnp.array([x[i - 2], y[j - 2], z[k - 2]], dtype=f32)
                    r_m_proj = r_ijk - delta_ijk * self.normal_vec_fn((i, j, k))
                    r_m_proj = r_m_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
                    u_m = -1.0 * jnp.dot(
                        self.zeta_p_ijk_pqm[i - 2, j - 2, k - 2],
                        self.cube_at_v(u_cube, curr_ngbs),
                    )
                    u_m += (
                        1.0 - self.zeta_p_ijk[i - 2, j - 2, k - 2] + self.zeta_p_ijk_pqm[i - 2, j - 2, k - 2, 13]
                    ) * u_cube[i - 2, j - 2, k - 2]
                    # u_m += (-1.0)*(self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_interp_fn(r_m_proj) / self.mu_m_interp_fn(r_m_proj) )
                    u_m += (-1.0) * (
                        self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_m_proj)
                    )
                    return u_m

                def extrapolate_u_p_from_positive_domain_(i, j, k):
                    r_ijk = jnp.array([x[i - 2], y[j - 2], z[k - 2]], dtype=f32)
                    r_p_proj = r_ijk - delta_ijk * self.normal_vec_fn((i, j, k))
                    r_p_proj = r_p_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
                    u_p = -1.0 * jnp.dot(
                        self.gamma_p_ijk_pqm[i - 2, j - 2, k - 2],
                        self.cube_at_v(u_cube, curr_ngbs),
                    )
                    u_p += (
                        1.0 - self.gamma_p_ijk[i - 2, j - 2, k - 2] + self.gamma_p_ijk_pqm[i - 2, j - 2, k - 2, 13]
                    ) * u_cube[i - 2, j - 2, k - 2]
                    # u_p += (1.0 - self.gamma_p_ijk[i-2, j-2, k-2]) * (self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_interp_fn(r_p_proj) / self.mu_m_interp_fn(r_p_proj))
                    u_p += (1.0 - self.gamma_p_ijk[i - 2, j - 2, k - 2]) * (
                        self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_p_proj)
                    )
                    return u_p

                phi_ijk = self.phi_cube[i, j, k]
                u_m = jnp.where(
                    phi_ijk > 0,
                    extrapolate_u_m_from_negative_domain_(i, j, k),
                    u_cube[i - 2, j - 2, k - 2],
                )[0]
                u_p = jnp.where(
                    phi_ijk > 0,
                    u_cube[i - 2, j - 2, k - 2],
                    extrapolate_u_p_from_positive_domain_(i, j, k),
                )[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = self.mu_m_cube_internal[i - 2, j - 2, k - 2]
            mu_p_ijk = self.mu_p_cube_internal[i - 2, j - 2, k - 2]
            return jnp.where(
                mu_m_ijk > mu_p_ijk,
                mu_minus_bigger_fn(i, j, k),
                mu_plus_bigger_fn(i, j, k),
            )

        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface((i, j, k))
        # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
        u_mp = jnp.where(is_interface == 0, interface_node(i, j, k), bulk_node(is_interface, u_ijk))
        return u_mp

    # Compute normal gradients for error analysis
    def compute_normal_gradient_solution_mp_on_interface_regression(self, u, params=None):
        """
        Given the solution field u, this function computes gradient of u along normal direction
        of the level-set function on the interface itself; at r_proj.
        """
        u_cube = u.reshape(self.grid_shape)
        x = self.gstate.x
        y = self.gstate.y
        z = self.gstate.z
        u_mp = vmap(self.u_mp_fn, (None, None, None, None, 0, 0, 0))(
            u_cube, x, y, z, self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
        )

        def convolve_at_node(node):
            i, j, k = node
            curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
            u_mp_pqm = u_mp.reshape(self.phi_cube_.shape + (2,))[curr_ngbs[:, 0], curr_ngbs[:, 1], curr_ngbs[:, 2]]
            cm_pqm = self.Cm_ijk_pqm.reshape(self.phi_cube_.shape + (-1,))[i - 2, j - 2, k - 2]
            cp_pqm = self.Cp_ijk_pqm.reshape(self.phi_cube_.shape + (-1,))[i - 2, j - 2, k - 2]
            return jnp.sum(cm_pqm * u_mp_pqm[:, 0]), jnp.sum(cp_pqm * u_mp_pqm[:, 1])

        c_mp_u_mp_ngbs = vmap(convolve_at_node)(self.nodes)
        grad_n_u_m = -1.0 * self.Cm_ijk_pqm.sum(axis=1) * u_mp[:, 0] + c_mp_u_mp_ngbs[0]
        grad_n_u_p = -1.0 * self.Cp_ijk_pqm.sum(axis=1) * u_mp[:, 1] + c_mp_u_mp_ngbs[1]
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp_regression(self, u, params=None):
        """
        This function computes \nabla u^+ and \nabla u^- given a solution vector u.
        """
        u_cube = u.reshape(self.grid_shape)
        x = self.gstate.x
        y = self.gstate.y
        z = self.gstate.z

        def convolve_at_node(node, d_m_mat, d_p_mat):
            i, j, k = node
            curr_ngbs = jnp.add(jnp.array([i - 2, j - 2, k - 2]), self.ngbs)
            u_curr_ngbs = self.cube_at_v(u_cube, curr_ngbs)
            u_mp_node = self.u_mp_fn(u_cube, x, y, z, i, j, k)
            dU_mp = u_curr_ngbs[:, jnp.newaxis] - u_mp_node
            grad_m = d_m_mat @ dU_mp[:, 0]
            grad_p = d_p_mat @ dU_mp[:, 1]
            return grad_m, grad_p

        return vmap(convolve_at_node, (0, 0, 0))(self.nodes, self.D_m_mat, self.D_p_mat)

    @partial(jit, static_argnums=(0))
    def get_u_mp_by_neural_network_at_node_fn(self, params, x, y, z, i, j, k):
        """
        This function evaluates pairs of u^+ and u^- at each grid point
        in the domain, given the neural network models.
        """

        u_at_point_fn, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)

        r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
        delta_ijk = self.phi_cube[i, j, k]
        u_ijk = u_at_point_fn(r_ijk, delta_ijk)

        def bulk_node(is_interface_, u_ijk_):
            return jnp.array(
                [
                    jnp.where(is_interface_ == -1, u_ijk_, 0.0),
                    jnp.where(is_interface_ == 1, u_ijk_, 0.0),
                ]
            )

        def interface_node(i, j, k):
            def mu_minus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_m_proj = r_ijk - delta_ijk * normal_ijk
                    r_m_proj = r_m_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_m_proj, -1), normal_ijk)
                    u_m = u_at_point_fn(r_ijk, 1) - self.alpha_interp_fn(r_m_proj)
                    u_m -= delta_ijk * (
                        (self.mu_m_over_mu_p_interp_fn(r_m_proj) - 1.0) * du_dn
                        + self.beta_over_mu_p_interp_fn(r_m_proj)
                    )
                    # u_m -= delta_ijk * ( (self.mu_m_interp_fn(r_m_proj) / self.mu_p_interp_fn(r_m_proj) - 1.0) *  du_dn + self.beta_interp_fn(r_m_proj)/self.mu_p_interp_fn(r_m_proj))
                    return u_m

                def extrapolate_u_p_from_positive_domain(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_p_proj = r_ijk - delta_ijk * normal_ijk
                    r_p_proj = r_p_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_p_proj, -1), normal_ijk)
                    u_p = u_at_point_fn(r_ijk, -1) + self.alpha_interp_fn(r_p_proj)
                    u_p += delta_ijk * (
                        (self.mu_m_over_mu_p_interp_fn(r_p_proj) - 1.0) * du_dn
                        + self.beta_over_mu_p_interp_fn(r_p_proj)
                    )
                    # u_p += delta_ijk * ( (self.mu_m_interp_fn(r_p_proj)/self.mu_p_interp_fn(r_p_proj) - 1.0) * du_dn + self.beta_interp_fn(r_p_proj)/self.mu_p_interp_fn(r_p_proj) )
                    return u_p

                u_m = jnp.where(delta_ijk > 0, extrapolate_u_m_from_negative_domain(i, j, k), u_ijk)[0]
                u_p = jnp.where(delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain(i, j, k))[0]
                return jnp.array([u_m, u_p])

            def mu_plus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain_(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_m_proj = r_ijk - delta_ijk * normal_ijk
                    r_m_proj = r_m_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_m_proj, 1), normal_ijk)
                    u_m = u_at_point_fn(r_ijk, 1) - self.alpha_interp_fn(r_m_proj)
                    u_m -= delta_ijk * (
                        (1.0 - 1.0 / self.mu_m_over_mu_p_interp_fn(r_m_proj)) * du_dn
                        + self.beta_over_mu_m_interp_fn(r_m_proj)
                    )
                    # u_m -= delta_ijk * ( (1 - self.mu_p_interp_fn(r_m_proj)/self.mu_m_interp_fn(r_m_proj)) * du_dn + self.beta_interp_fn(r_m_proj) / self.mu_m_interp_fn(r_m_proj) )
                    return u_m

                def extrapolate_u_p_from_positive_domain_(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_p_proj = r_ijk - delta_ijk * normal_ijk
                    r_p_proj = r_p_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_p_proj, 1), normal_ijk)
                    u_p = u_at_point_fn(r_ijk, -1) + self.alpha_interp_fn(r_p_proj)
                    u_p += delta_ijk * (
                        (1.0 - 1.0 / self.mu_m_over_mu_p_interp_fn(r_p_proj)) * du_dn
                        + self.beta_over_mu_m_interp_fn(r_p_proj)
                    )
                    # u_p += delta_ijk * ((1 - self.mu_p_interp_fn(r_p_proj)/self.mu_m_interp_fn(r_p_proj)) * du_dn + self.beta_interp_fn(r_p_proj) / self.mu_m_interp_fn(r_p_proj))
                    return u_p

                u_m = jnp.where(delta_ijk > 0, extrapolate_u_m_from_negative_domain_(i, j, k), u_ijk)[0]
                u_p = jnp.where(delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain_(i, j, k))[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = self.mu_m_cube_internal[i - 2, j - 2, k - 2]
            mu_p_ijk = self.mu_p_cube_internal[i - 2, j - 2, k - 2]
            return jnp.where(
                mu_m_ijk > mu_p_ijk,
                mu_minus_bigger_fn(i, j, k),
                mu_plus_bigger_fn(i, j, k),
            )

        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface((i, j, k))
        # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
        u_mp = jnp.where(is_interface == 0, interface_node(i, j, k), bulk_node(is_interface, u_ijk))
        return u_mp

    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, u, params):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, -1)
        grad_n_u_m = vmap(jnp.dot, (0, 0))(self.normal_vecs, grad_u_m)
        grad_n_u_p = vmap(jnp.dot, (0, 0))(self.normal_vecs, grad_u_p)
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp_neural_network(self, u, params):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, -1)
        return grad_u_m, grad_u_p


def poisson_solver(gstate, sim_state, algorithm=0, switching_interval=3):
    # --- Defining Optimizer
    decay_rate_ = 0.975
    learning_rate = 1e-2
    scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=decay_rate_)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )
    # optimizer = optax.adam(learning_rate)
    # optimizer = optax.rmsprop(learning_rate)
    # ---------------------

    trainer = PDETrainer(gstate, sim_state, optimizer, algorithm)
    opt_state, params = trainer.init()
    print_architecture(params)

    num_epochs = 10000
    start_time = time.time()

    # loss_epochs = []
    # epoch_store = []
    # for epoch in range(num_epochs):
    #     opt_state, params, loss_epoch = trainer.update(opt_state, params)
    #     print(f"epoch # {epoch} loss is {loss_epoch}")
    #     loss_epochs.append(loss_epoch)
    #     epoch_store.append(epoch)

    """
    def learn_whole(carry, epoch):
        opt_state, params, loss_epochs = carry
        opt_state, params, loss_epoch = trainer.update(opt_state, params)
        loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
        return (opt_state, params, loss_epochs), None
    loss_epochs = jnp.zeros(num_epochs)
    epoch_store = jnp.arange(num_epochs)
    (opt_state, params, loss_epochs), _ = jax.lax.scan(learn_whole, (opt_state, params, loss_epochs), epoch_store)
    """
    lhs_rhs = trainer.compute_Ax_and_b_preconditioned_fn(params)
    uu = trainer.evaluate_solution_fn(params, trainer.gstate.R, trainer.phi_flat)
    u_cube = uu.reshape(trainer.grid_shape)
    u_mp = vmap(trainer.get_u_mp_by_regression_at_node_fn, (None, None, None, None, 0, 0, 0))(
        u_cube,
        trainer.gstate.x,
        trainer.gstate.y,
        trainer.gstate.z,
        trainer.nodes[:, 0],
        trainer.nodes[:, 1],
        trainer.nodes[:, 2],
    )

    def learn_interleaved(carry, epoch):
        # cur_region = 0:everywhere, <0: interface band/inside, >0: outside interface band/outside
        opt_state, params, loss_epochs = carry
        # cur_region = epoch % switching_interval - 1       # inside - outside - whole
        cur_region = i32(-1) * (epoch % switching_interval)  # whole - inside - inside
        opt_state, params, loss_epoch = trainer.update_region(opt_state, params, region=cur_region)
        # opt_state, params, loss_epoch = trainer.update(opt_state, params)
        loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
        return (opt_state, params, loss_epochs), None

    loss_epochs = jnp.zeros(num_epochs)
    epoch_store = jnp.arange(num_epochs)
    (opt_state, params, loss_epochs), _ = jax.lax.scan(
        learn_interleaved, (opt_state, params, loss_epochs), epoch_store
    )

    end_time = time.time()
    print(f"solve took {end_time - start_time} (sec)")

    fig, ax = plt.subplots(figsize=(8, 8))

    # plt.plot(epoch_store[epoch_store%switching_interval - 1 ==0], loss_epochs[epoch_store%switching_interval - 1 ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 <0], loss_epochs[epoch_store%switching_interval - 1 <0], color='b', label='negative domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 >0], loss_epochs[epoch_store%switching_interval - 1 >0], color='r', label='positive domain')

    plt.plot(
        epoch_store[epoch_store % switching_interval == 0],
        loss_epochs[epoch_store % switching_interval == 0],
        color="k",
        label="whole domain",
    )
    plt.plot(
        epoch_store[-1 * (epoch_store % switching_interval) < 0],
        loss_epochs[-1 * (epoch_store % switching_interval) < 0],
        color="b",
        label="negative domain",
    )

    # ax.plot(epoch_store, loss_epochs, color='k')

    ax.set_yscale("log")
    ax.set_xlabel(r"$\rm epoch$", fontsize=20)
    ax.set_ylabel(r"$\rm loss$", fontsize=20)
    plt.legend(fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)
    plt.tight_layout()
    plt.savefig("tests/poisson_solver_loss.png")
    plt.close()

    final_solution = trainer.evaluate_solution_fn(params, gstate.R, trainer.phi_flat).reshape(-1)

    # ------------- Gradients of discovered solutions are below:
    if algorithm == 0:
        grad_u_mp_normal_to_interface = trainer.compute_normal_gradient_solution_mp_on_interface(final_solution, None)
        grad_u_mp = trainer.compute_gradient_solution_mp(final_solution, None)
    elif algorithm == 1:
        grad_u_mp_normal_to_interface = trainer.compute_normal_gradient_solution_mp_on_interface(None, params)
        grad_u_mp = trainer.compute_gradient_solution_mp(None, params)

    return (
        final_solution,
        grad_u_mp,
        grad_u_mp_normal_to_interface,
        epoch_store,
        loss_epochs,
    )
