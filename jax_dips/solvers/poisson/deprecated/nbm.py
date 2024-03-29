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

from functools import partial
from typing import Callable

import haiku as hk
import jax

# import jaxopt
import optax
import jaxopt
from jax import grad, jit
from jax import numpy as jnp
from jax import random, value_and_grad, vmap

from jax_dips.domain.mesh import GridState

from jax_dips.solvers.poisson.discretization import Discretization
from jax_dips.solvers.simulation_states import PoissonSimState, PoissonSimStateFn
from jax_dips.utils.chunking import chunked_vmap


class Bootstrap(Discretization):
    """
    Neural bootstrap method for solving the Poisson equation with interfacial jump conditions.
    This class controls the neural network models and evaluates the loss residuals using the
    finite discretization class implemented in the residual.py

    This class has the API: update, update_multi_gpu,
    """

    def __init__(
        self,
        lvl_gstate: GridState,
        tr_gstate: GridState,
        sim_state: PoissonSimState,
        sim_state_fn: PoissonSimStateFn,
        optimizer: object = optax.adam(learning_rate=1e-3),
        algorithm: int = 0,
        mgrad_over_pgrad_scalefactor: float = 1.0,
        precondition: int = 1,
    ) -> None:
        r"""
        algorithm = 0: use regression to evaluate u^\pm; algorithm = 1: use neural network to evaluate u^\pm
        """
        super().__init__(
            lvl_gstate,
            sim_state,
            sim_state_fn,
            precondition,
            algorithm,
        )

        self.optimizer = optimizer

        self.tr_gstate = tr_gstate
        self.mgrad_over_pgrad_scalefactor = mgrad_over_pgrad_scalefactor

        """ initialize postprocessing methods """
        self.compute_normal_gradient_solution_mp_on_interface = (
            self.compute_normal_gradient_solution_mp_on_interface_neural_network
        )
        self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_neural_network
        self.compute_normal_gradient_solution_on_interface = (
            self.compute_normal_gradient_solution_on_interface_neural_network
        )
        self.compute_gradient_solution = self.compute_gradient_solution_neural_network

    # @staticmethod
    # def split_mp_networks_keys(params):
    #     """Split keys for negative and positive networks
    #     Expects the layer names include patterns _m_fn and _p_fn for -/+ networks respectively

    #     Args:
    #         params [dict]: parameters dictionary of the model with both positive/negative networks

    #     Returns:
    #         (mnet_keys, list[str]), (pnet_keys list[str]): pair of lists for key strings of -/+ networks
    #     """
    #     import re

    #     def find_keys_with_pattern(keys, pattern):
    #         matched_keys = []
    #         for key in keys:
    #             if re.search(pattern, key):
    #                 matched_keys.append(key)
    #         return matched_keys

    #     keys = params.keys()
    #     m_pattern = r"_m_fn"
    #     mnet_keys = find_keys_with_pattern(keys, m_pattern)
    #     p_pattern = r"_p_fn"
    #     pnet_keys = find_keys_with_pattern(keys, p_pattern)
    #     return mnet_keys, pnet_keys
    #     # mnet, pnet = hk.data_structures.partition(lambda m, n, p: m != "double_mlp/~resnet_m_fn/linear", params)

    # @partial(jit, static_argnums=0)
    # def init(self, seed=42):
    #     r"""This function initializes the neural network with random parameters."""
    #     rng = random.PRNGKey(seed)
    #     params = self.forward.init(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=0.1)
    #     opt_state = self.optimizer.init(params=params)
    #     self.mnet_keys, self.pnet_keys = self.split_mp_networks_keys(params)  # split keys for each domain
    #     return opt_state, params

    # @staticmethod
    # @hk.transform
    # def forward(x, phi_x):
    #     r"""Forward pass of the neural network.

    #     Args:
    #         x: input data

    #     Returns:
    #         output of the neural network
    #     """
    #     model = DoubleMLP()
    #     return model(x, phi_x)

    # @partial(jit, static_argnums=(0))
    # def evaluate_solution_fn(self, params, R_flat):
    #     phi_flat = self.phi_interp_fn(R_flat)
    #     sol_fn = partial(self.forward.apply, params, None)
    #     pred_sol = vmap(sol_fn, (0, 0))(R_flat, phi_flat)

    #     # pred_sol = chunked_vmap(
    #     #     sol_fn,
    #     #     num_chunks=2,
    #     #     in_axes=(0, 0),
    #     # )(R_flat, phi_flat)

    #     return pred_sol

    # def solution_at_point_fn(self, params, r_point, phi_point):
    #     sol_fn = partial(self.forward.apply, params, None)
    #     return sol_fn(r_point, phi_point).reshape()

    # def get_sol_grad_sol_fn(self, params):
    #     u_at_point_fn = partial(self.solution_at_point_fn, params)
    #     grad_u_at_point_fn = grad(u_at_point_fn)
    #     return u_at_point_fn, grad_u_at_point_fn

    # def get_mask_plus(self, points):
    #     """
    #     For a set of points, returns 1 if in external region
    #     returns 0 if inside the geometry.
    #     """
    #     phi_points = self.phi_interp_fn(points)

    #     def sign_p_fn(a):
    #         # returns 1 only if a>0, otherwise is 0
    #         sgn = jnp.sign(a)
    #         return jnp.floor(0.5 * sgn + 0.75)

    #     mask_p = sign_p_fn(phi_points)
    #     return mask_p

    # @partial(jit, static_argnums=(0))
    # def loss(self, params, points, dx, dy, dz):
    #     r"""Loss function of the neural network."""
    #     lhs_rhs = vmap(self.compute_Ax_and_b_fn, (None, 0, None, None, None))(params, points, dx, dy, dz)
    #     lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
    #     tot_loss = jnp.mean(optax.l2_loss(lhs, rhs))
    #     # du_xmax = (self.evaluate_solution_fn(params, self.tr_gstate.R_xmax_boundary) - self.dir_bc_fn(self.tr_gstate.R_xmax_boundary)[...,jnp.newaxis])
    #     # du_xmin = (self.evaluate_solution_fn(params, self.tr_gstate.R_xmin_boundary) - self.dir_bc_fn(self.tr_gstate.R_xmin_boundary)[...,jnp.newaxis])
    #     # du_ymax = (self.evaluate_solution_fn(params, self.tr_gstate.R_ymax_boundary) - self.dir_bc_fn(self.tr_gstate.R_ymax_boundary)[...,jnp.newaxis])
    #     # du_ymin = (self.evaluate_solution_fn(params, self.tr_gstate.R_ymin_boundary) - self.dir_bc_fn(self.tr_gstate.R_ymin_boundary)[...,jnp.newaxis])
    #     # du_zmax = (self.evaluate_solution_fn(params, self.tr_gstate.R_zmax_boundary) - self.dir_bc_fn(self.tr_gstate.R_zmax_boundary)[...,jnp.newaxis])
    #     # du_zmin = (self.evaluate_solution_fn(params, self.tr_gstate.R_zmin_boundary) - self.dir_bc_fn(self.tr_gstate.R_zmin_boundary)[...,jnp.newaxis])
    #     # tot_loss += 0.01 * (jnp.mean(jnp.square(du_xmax)) + jnp.mean(jnp.square(du_xmin)) + jnp.mean(jnp.square(du_ymax)) + jnp.mean(jnp.square(du_ymin)) + jnp.mean(jnp.square(du_zmax)) + jnp.mean(jnp.square(du_zmin)))
    #     return tot_loss

    # @partial(jit, static_argnums=(0))
    # def update(self, opt_state, params, points, dx, dy, dz):
    #     r"""One step of single-GPU optimization on the neural network model."""
    #     loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)
    #     updates, opt_state = self.optimizer.update(grads, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     return opt_state, params, loss

    # @partial(jit, static_argnums=(0))
    # def __update(self, opt_state, params, points, dx, dy, dz):
    #     r"""One step of single-GPU optimization on the neural network model.
    #     Update function must perform update rule on -/+ domains and networks independently
    #     """

    #     def _loss(params_trainable, params_untrainable, points, dx, dy, dz):
    #         params = hk.data_structures.merge(params_trainable, params_untrainable)
    #         loss = self.loss(params, points, dx, dy, dz)
    #         return loss

    #     # split parameters
    #     m_params, p_params = hk.data_structures.partition(lambda m, n, p: m not in self.pnet_keys, params)
    #     # -- first get gradients of negative network
    #     m_loss, m_grads = value_and_grad(_loss)(m_params, p_params, points, dx, dy, dz)
    #     # -- second get gradients of positive network
    #     p_loss, p_grads = value_and_grad(_loss)(p_params, m_params, points, dx, dy, dz)
    #     # --- Choose region to optimize by zeroing the other region
    #     m_grads = jax.tree_map(lambda x: x * self.mgrad_over_pgrad_scalefactor, m_grads)
    #     # -- append the grads into m_grads dictionary
    #     m_grads.update(p_grads)
    #     updates, opt_state = self.optimizer.update(m_grads, opt_state, params)  # -- get optimizer updates
    #     params = optax.apply_updates(params, updates)  # -- apply updates on params
    #     # -- collect losses
    #     loss = m_loss + p_loss
    #     return opt_state, params, loss

    # # def update_lbfgs(self, params, points, dx, dy, dz, maxiter=10):
    # #     solver = jaxopt.LBFGS(fun=self.loss, maxiter=maxiter)
    # #     params, opt_state = solver.run(params, points=points, dx=dx, dy=dy, dz=dz)
    # #     return opt_state, params

    # # @partial(jit, static_argnums=(0))
    # def update_multi_gpu(self, opt_state, params, points, dx, dy, dz):
    #     r"""One step of multi-GPU optimization on the neural network model."""
    #     loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)

    #     """ Muli-GPU """
    #     grads = jax.lax.psum(grads, axis_name="devices")
    #     loss = jax.lax.psum(loss, axis_name="devices")

    #     updates, opt_state = self.optimizer.update(grads, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     return opt_state, params, loss

    # def compute_normal_gradient_solution_mp_on_interface_neural_network(self, params, points, dx, dy, dz):
    #     _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
    #     grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
    #     grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
    #     normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
    #     grad_n_u_m = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_m)
    #     grad_n_u_p = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_p)
    #     return grad_n_u_m, grad_n_u_p

    # def compute_gradient_solution_mp_neural_network(self, params, points):
    #     _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
    #     grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
    #     grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
    #     return grad_u_m, grad_u_p

    # def compute_normal_gradient_solution_on_interface_neural_network(self, params, points, dx, dy, dz):
    #     phi_flat = self.phi_interp_fn(points)
    #     _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)

    #     grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
    #     normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
    #     grad_n_u = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u)
    #     # grad_u = chunked_vmap(grad_u_at_point_fn, num_chunks=2, in_axes=(0, 0))(
    #     #     points,
    #     #     phi_flat,
    #     # )
    #     # normal_vecs = chunked_vmap(self.normal_point_fn, num_chunks=2, in_axes=(0, None, None, None))(
    #     #     points,
    #     #     dx,
    #     #     dy,
    #     #     dz,
    #     # )
    #     # grad_n_u = chunked_vmap(jnp.dot, num_chunks=2, in_axes=(0, 0))(
    #     #     jnp.squeeze(normal_vecs),
    #     #     grad_u,
    #     # )

    #     return grad_n_u

    # def compute_gradient_solution_neural_network(self, params, points):
    #     phi_flat = self.phi_interp_fn(points)
    #     _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
    #     grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
    #     return grad_u
