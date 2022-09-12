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
from jax import (numpy as jnp, vmap, jit, grad, random, value_and_grad, config)
config.update("jax_debug_nans", True)
import optax
import haiku as hk

from src import (interpolate, geometric_integrations_per_point)
from src.jaxmd_modules.util import f32, i32
from src.nn_solution_model import DoubleMLP
from src.utils import print_architecture

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import partial
import time
import pdb


class PDETrainer:
    """
        This is a completely local point-based Poisson solver.
    """
    def __init__(self, gstate, eval_gstate, sim_state, sim_state_fn, optimizer, algorithm=0, precondition=1):
        """
            algorithm = 0: use regression to evaluate u^\pm
            algorithm = 1: use neural network to evaluate u^\pm
        """

        self.optimizer = optimizer
        self.gstate = gstate  
        self.eval_gstate = eval_gstate
        self.sim_state_fn = sim_state_fn
        self.sim_state = sim_state
        self.algorithm = algorithm
        
        
        """ Grid Info """
        self.dx = self.gstate.dx; self.dy = self.gstate.dy; self.dz = self.gstate.dz
        self.grid_shape = self.gstate.shape()
        self.Nx, self.Ny, self.Nz = self.grid_shape
        self.bandwidth_squared = (2.0 * self.dx)*(2.0 * self.dx)
        


        
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
        self.get_vertices_of_cell_intersection_with_interface_at_point, self.is_cell_crossed_by_interface = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(self.phi_interp_fn)
        self.beta_integrate_over_interface_at_point, self.beta_integrate_in_negative_domain = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(self.get_vertices_of_cell_intersection_with_interface_at_point, self.is_cell_crossed_by_interface, self.beta_interp_fn)
        self.compute_face_centroids_values_plus_minus_at_point = geometric_integrations_per_point.compute_cell_faces_areas_values(self.get_vertices_of_cell_intersection_with_interface_at_point, self.is_cell_crossed_by_interface, self.mu_m_interp_fn, self.mu_p_interp_fn)
        
        
        """ Evaluation/Training Grid Set """
        self.cell_dx = self.eval_gstate.dx
        self.cell_dy = self.eval_gstate.dy
        self.cell_dz = self.eval_gstate.dz
        self.Vol_cell_nominal =  self.cell_dx * self.cell_dy * self.cell_dz

        self.Xijk = jnp.array([ [-self.cell_dx, -self.cell_dy, -self.cell_dz],
                                [          0.0, -self.cell_dy, -self.cell_dz],
                                [ self.cell_dx, -self.cell_dy, -self.cell_dz],
                                [-self.cell_dx,           0.0, -self.cell_dz],
                                [0.0          ,           0.0, -self.cell_dz],
                                [ self.cell_dx,           0.0, -self.cell_dz],
                                [-self.cell_dx,  self.cell_dy, -self.cell_dz],
                                [          0.0,  self.cell_dy, -self.cell_dz],
                                [ self.cell_dx,  self.cell_dy, -self.cell_dz],
                                [-self.cell_dx, -self.cell_dy,           0.0],
                                [          0.0, -self.cell_dy,           0.0],
                                [ self.cell_dx, -self.cell_dy,           0.0],
                                [-self.cell_dx,           0.0,           0.0],
                                [          0.0,           0.0,           0.0],
                                [ self.cell_dx,           0.0,           0.0],
                                [-self.cell_dx,  self.cell_dy,           0.0],
                                [          0.0,  self.cell_dy,           0.0],
                                [ self.cell_dx,  self.cell_dy,           0.0],
                                [-self.cell_dx, -self.cell_dy,  self.cell_dz],
                                [          0.0, -self.cell_dy,  self.cell_dz],
                                [ self.cell_dx, -self.cell_dy,  self.cell_dz],
                                [-self.cell_dx,           0.0,  self.cell_dz],
                                [          0.0,           0.0,  self.cell_dz],
                                [ self.cell_dx,           0.0,  self.cell_dz],
                                [-self.cell_dx,  self.cell_dy,  self.cell_dz],
                                [          0.0,  self.cell_dy,  self.cell_dz],
                                [ self.cell_dx,  self.cell_dy,  self.cell_dz]], dtype=f32)

        self.ngbs = jnp.array([ [-1, -1, -1],
                                [0, -1, -1],
                                [1, -1, -1],
                                [-1,  0, -1],
                                [0,  0, -1],
                                [1,  0, -1],
                                [-1,  1, -1],
                                [0,  1, -1],
                                [1,  1, -1],
                                [-1, -1,  0],
                                [0, -1,  0],
                                [1, -1,  0],
                                [-1,  0,  0],
                                [0,  0,  0],
                                [1,  0,  0],
                                [-1,  1,  0],
                                [0,  1,  0],
                                [1,  1,  0],
                                [-1, -1,  1],
                                [0, -1,  1],
                                [1, -1,  1],
                                [-1,  0,  1],
                                [0,  0,  1],
                                [1,  0,  1],
                                [-1,  1,  1],
                                [0,  1,  1],
                                [1,  1,  1]], dtype=i32)
        


        

        """ initialize configurated solver """
        if self.algorithm==0:
            self.u_mp_fn = self.get_u_mp_by_regression_at_point_fn
            self.compute_normal_gradient_solution_mp_on_interface = self.compute_normal_gradient_solution_mp_on_interface_regression
            self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_regression

        elif self.algorithm==1:
            self.initialize_neural_based_algorithm()
            self.u_mp_fn = self.get_u_mp_by_neural_network_at_node_fn
            self.compute_normal_gradient_solution_mp_on_interface = self.compute_normal_gradient_solution_mp_on_interface_neural_network
            self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_neural_network


        if precondition==1:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_preconditioned_fn
        elif precondition==0:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_vanilla_fn



    def normal_point_fn(self, point, dx, dy, dz):
        """
            Evaluate normal vector at a given point based on interpolated values
            of the level set function at the face-centers of a 3D cell centered at the
            point with each side length given by dx, dy, dz.
        """
        point_ip1_j_k = jnp.array([[point[0] + f32(0.5)*dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - f32(0.5)*dx, point[1], point[2]]])
        phi_x = (self.phi_interp_fn(point_ip1_j_k) - self.phi_interp_fn(point_im1_j_k) ) / (dx) 
        
        point_i_jp1_k = jnp.array([[point[0], point[1] + f32(0.5)*dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - f32(0.5)*dy, point[2]]])
        phi_y = (self.phi_interp_fn(point_i_jp1_k) - self.phi_interp_fn(point_i_jm1_k) ) / (dy) 
        
        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + f32(0.5)*dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - f32(0.5)*dz]])
        phi_z = (self.phi_interp_fn(point_i_j_kp1) - self.phi_interp_fn(point_i_j_km1) ) / (dz) 
        
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
        curr_vertices = jnp.add(jnp.array([x, y, z]), self.Xijk)
        phi_vertices = self.phi_interp_fn(curr_vertices)

        Wijk_p = jnp.diag(vmap(sign_p_fn)(phi_vertices)) 
        Wijk_m = jnp.diag(vmap(sign_m_fn)(phi_vertices)) 

        Dp = jnp.linalg.pinv(self.Xijk.T @ Wijk_p @ self.Xijk) @ (Wijk_p @ self.Xijk).T
        Dm = jnp.linalg.pinv(self.Xijk.T @ Wijk_m @ self.Xijk) @ (Wijk_m @ self.Xijk).T
        D_m_mat = jnp.nan_to_num(Dm)
        D_p_mat = jnp.nan_to_num(Dp)
        
        
        normal_vec = self.normal_point_fn(point, dx, dy, dz).T
        phi_point = self.phi_interp_fn(point[jnp.newaxis])

        Cm_ijk_pqm = normal_vec @ D_m_mat
        Cp_ijk_pqm = normal_vec @ D_p_mat

        zeta_p_ijk_pqm = ((self.mu_p_interp_fn(point[jnp.newaxis]) - self.mu_m_interp_fn(point[jnp.newaxis])) / self.mu_m_interp_fn(point[jnp.newaxis])) * phi_point
        zeta_p_ijk_pqm = zeta_p_ijk_pqm[...,jnp.newaxis] * Cp_ijk_pqm
        zeta_m_ijk_pqm = ((self.mu_p_interp_fn(point[jnp.newaxis]) - self.mu_m_interp_fn(point[jnp.newaxis])) / self.mu_p_interp_fn(point[jnp.newaxis])) * phi_point
        zeta_m_ijk_pqm = zeta_m_ijk_pqm[...,jnp.newaxis] * Cm_ijk_pqm
        zeta_p_ijk = (zeta_p_ijk_pqm.sum(axis=1) - zeta_p_ijk_pqm[:, 13]) * f32(-1.0)
        zeta_m_ijk = (zeta_m_ijk_pqm.sum(axis=1) - zeta_m_ijk_pqm[:, 13]) * f32(-1.0)

        gamma_p_ijk_pqm = zeta_p_ijk_pqm / (1.0 + zeta_p_ijk[:, jnp.newaxis])
        gamma_m_ijk_pqm = zeta_m_ijk_pqm / (1.0 - zeta_m_ijk[:, jnp.newaxis])
        gamma_p_ijk = (gamma_p_ijk_pqm.sum(axis=1) - gamma_p_ijk_pqm[:, 13]) * f32(-1.0)
        gamma_m_ijk = (gamma_m_ijk_pqm.sum(axis=1) - gamma_m_ijk_pqm[:, 13]) * f32(-1.0)

        return normal_vec, gamma_m_ijk, gamma_m_ijk_pqm, gamma_p_ijk, gamma_p_ijk_pqm, \
                           zeta_m_ijk , zeta_m_ijk_pqm , zeta_p_ijk , zeta_p_ijk_pqm




    



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
    def evaluate_solution_fn(self, params, R_flat):
        phi_flat = self.phi_interp_fn(R_flat)
        sol_fn =  partial(self.forward.apply, params, None)
        pred_sol = vmap(sol_fn, (0,0))(R_flat, phi_flat)
        return pred_sol

    def solution_at_point_fn(self, params, r_point, phi_point):
        sol_fn =  partial(self.forward.apply, params, None)
        return sol_fn(r_point, phi_point).reshape()


    def get_sol_grad_sol_fn(self, params):
        u_at_point_fn = partial(self.solution_at_point_fn, params)
        grad_u_at_point_fn = grad(u_at_point_fn)
        return u_at_point_fn, grad_u_at_point_fn


    
    @partial(jit, static_argnums=(0))
    def evaluate_loss_fn(self, lhs, rhs, pred_sol_xmin_bc, pred_sol_xmax_bc, pred_sol_ymin_bc, pred_sol_ymax_bc, pred_sol_zmin_bc, pred_sol_zmax_bc):
        """
            Weighted L2 loss
        """
        tot_loss = jnp.mean(optax.l2_loss(lhs, rhs))
        tot_loss += jnp.square(pred_sol_xmin_bc - self.dir_bc_fn(self.gstate.R_xmin_boundary)).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(pred_sol_xmax_bc - self.dir_bc_fn(self.gstate.R_xmax_boundary)).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(pred_sol_ymin_bc - self.dir_bc_fn(self.gstate.R_ymin_boundary)).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(pred_sol_ymax_bc - self.dir_bc_fn(self.gstate.R_ymax_boundary)).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(pred_sol_zmin_bc - self.dir_bc_fn(self.gstate.R_zmin_boundary)).mean() * self.Vol_cell_nominal
        tot_loss += jnp.square(pred_sol_zmax_bc - self.dir_bc_fn(self.gstate.R_zmax_boundary)).mean() * self.Vol_cell_nominal
        return tot_loss
    
    

    
    def loss(self, params, points, pointset_gstate):
        """
            Loss function of the neural network
        """     
        lhs_rhs = vmap(self.compute_Ax_and_b_fn, (None, 0, None, None, None))(params, points, pointset_gstate.dx, pointset_gstate.dy, pointset_gstate.dz)   
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)

        pred_sol_xmin_bc = self.evaluate_solution_fn(params, pointset_gstate.R_xmin_boundary)
        pred_sol_xmax_bc = self.evaluate_solution_fn(params, pointset_gstate.R_xmax_boundary)
        pred_sol_ymin_bc = self.evaluate_solution_fn(params, pointset_gstate.R_ymin_boundary)
        pred_sol_ymax_bc = self.evaluate_solution_fn(params, pointset_gstate.R_ymax_boundary)
        pred_sol_zmin_bc = self.evaluate_solution_fn(params, pointset_gstate.R_zmin_boundary)
        pred_sol_zmax_bc = self.evaluate_solution_fn(params, pointset_gstate.R_zmax_boundary)

        tot_loss = self.evaluate_loss_fn( lhs, rhs, pred_sol_xmin_bc, pred_sol_xmax_bc, pred_sol_ymin_bc, pred_sol_ymax_bc, pred_sol_zmin_bc, pred_sol_zmax_bc)
        return tot_loss

   
    
    @partial(jit, static_argnums=(0))
    def update(self, opt_state, params, points, pointset_gstate):      
        loss, grads = value_and_grad(self.loss)(params, points, pointset_gstate)
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
            boundary = (x - self.gstate.xmin())*(x - self.gstate.xmax()) * (y - self.gstate.ymin())*(y-self.gstate.ymax()) * (z - self.gstate.zmin())*(z - self.gstate.zmax())
            return jnp.where(abs(boundary) < 1e-6*self.gstate.dx, True, False)



        def evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz):
            #--- LHS
            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(point, dx, dy, dz)
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]

            
            def get_lhs_at_interior_point(point):
                point_ijk = point
                point_imjk = jnp.array([point[0] - 0.5*dx, point[1], point[2]], dtype=f32)
                point_ipjk = jnp.array([point[0] + 0.5*dx, point[1], point[2]], dtype=f32)
                point_ijmk = jnp.array([point[0], point[1] - 0.5*dy, point[2]], dtype=f32)
                point_ijpk = jnp.array([point[0], point[1] + 0.5*dy, point[2]], dtype=f32)
                point_ijkm = jnp.array([point[0], point[1], point[2] - 0.5*dz], dtype=f32)
                point_ijkp = jnp.array([point[0], point[1], point[2] + 0.5*dz], dtype=f32)

                k_m_ijk = self.k_m_interp_fn(point[jnp.newaxis]) 
                k_p_ijk = self.k_p_interp_fn(point[jnp.newaxis])

                u_m_ijk , u_p_ijk  = u_mp_at_point(point_ijk)
                u_m_imjk, u_p_imjk = u_mp_at_point(point_imjk)
                u_m_ipjk, u_p_ipjk = u_mp_at_point(point_ipjk)
                u_m_ijmk, u_p_ijmk = u_mp_at_point(point_ijmk)
                u_m_ijpk, u_p_ijpk = u_mp_at_point(point_ijpk)
                u_m_ijkm, u_p_ijkm = u_mp_at_point(point_ijkm)
                u_m_ijkp, u_p_ijkp = u_mp_at_point(point_ijkp)

                lhs  = k_m_ijk * V_m_ijk * u_m_ijk
                lhs += k_p_ijk * V_p_ijk * u_p_ijk
                lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + \
                       (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]) * u_p_ijk
                lhs += -1.0 * coeffs[0 ] * u_m_imjk - coeffs[1 ] * u_p_imjk
                lhs += -1.0 * coeffs[2 ] * u_m_ipjk - coeffs[3 ] * u_p_ipjk
                lhs += -1.0 * coeffs[4 ] * u_m_ijmk - coeffs[5 ] * u_p_ijmk
                lhs += -1.0 * coeffs[6 ] * u_m_ijpk - coeffs[7 ] * u_p_ijpk
                lhs += -1.0 * coeffs[8 ] * u_m_ijkm - coeffs[9 ] * u_p_ijkm
                lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

                diag_coeff = k_p_ijk * V_p_ijk + k_m_ijk * V_m_ijk + (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) + (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11])
                return jnp.array([lhs.reshape(), diag_coeff.reshape()])
            

            def get_lhs_on_box_boundary(point):
                lhs = self.sim_state_fn.dir_bc_fn(point[jnp.newaxis]).reshape() * self.Vol_cell_nominal
                return jnp.array([lhs, self.Vol_cell_nominal])

            

            lhs_diagcoeff = jnp.where(is_box_boundary_point(point), get_lhs_on_box_boundary(point), get_lhs_at_interior_point(point))
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)
            diagcoeff_ = jnp.sqrt(diagcoeff*diagcoeff)
            #--- RHS  
            def get_rhs_at_interior_point(point):
                rhs = self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk  + self.f_p_interp_fn(point[jnp.newaxis]) * V_p_ijk
                rhs += self.beta_integrate_over_interface_at_point(point, dx, dy, dz)
                return rhs
            
            def get_rhs_on_box_boundary(point):
                return self.sim_state_fn.dir_bc_fn(point[jnp.newaxis]).reshape() * self.Vol_cell_nominal

            rhs = jnp.where(is_box_boundary_point(point), get_rhs_on_box_boundary(point), get_rhs_at_interior_point(point))

            return jnp.array([lhs / (1e-13 + diagcoeff_), rhs / (1e-13 + diagcoeff_)])
        
        lhs_rhs = evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz)
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
        
        u = self.evaluate_solution_fn(params, self.gstate.R)
        u_cube = u.reshape(self.grid_shape)

        # u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z, params)
        if self.algorithm==0:
            u_mp_at_node = partial(self.u_mp_fn, u_cube, x, y, z)
        elif self.algorithm==1:
            u_mp_at_node = partial(self.u_mp_fn, params, x, y, z)

        def is_box_boundary_node(i, j, k):
            """
            Check if current node is on the boundary of box
            """
            boundary = (i-2)*(i-Nx-1)*(j-2)*(j-Ny-1)*(k-2)*(k-Nz-1)
            return jnp.where(boundary == 0, True, False)



        def evaluate_discretization_lhs_rhs_at_node(node):
            #--- LHS
            i, j, k = node

            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(node)
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]
            
            def get_lhs_at_interior_node(node):
                i, j, k = node
                
                k_m_ijk = self.k_m_cube_internal[i-2, j-2, k-2]
                k_p_ijk = self.k_p_cube_internal[i-2, j-2, k-2]

                u_m_ijk , u_p_ijk  = u_mp_at_node(i  , j  , k  )
                u_m_imjk, u_p_imjk = u_mp_at_node(i-1, j  , k  )
                u_m_ipjk, u_p_ipjk = u_mp_at_node(i+1, j  , k  )
                u_m_ijmk, u_p_ijmk = u_mp_at_node(i  , j-1, k  )
                u_m_ijpk, u_p_ijpk = u_mp_at_node(i  , j+1, k  )
                u_m_ijkm, u_p_ijkm = u_mp_at_node(i  , j  , k-1)
                u_m_ijkp, u_p_ijkp = u_mp_at_node(i  , j  , k+1)

                lhs  = k_m_ijk * V_m_ijk * u_m_ijk
                lhs += k_p_ijk * V_p_ijk * u_p_ijk
                lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + \
                       (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]) * u_p_ijk
                lhs += -1.0 * coeffs[0 ] * u_m_imjk - coeffs[1 ] * u_p_imjk
                lhs += -1.0 * coeffs[2 ] * u_m_ipjk - coeffs[3 ] * u_p_ipjk
                lhs += -1.0 * coeffs[4 ] * u_m_ijmk - coeffs[5 ] * u_p_ijmk
                lhs += -1.0 * coeffs[6 ] * u_m_ijpk - coeffs[7 ] * u_p_ijpk
                lhs += -1.0 * coeffs[8 ] * u_m_ijkm - coeffs[9 ] * u_p_ijkm
                lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

                return lhs
            

            def get_lhs_on_box_boundary(node):
                i, j, k = node
                lhs = u_cube[i-2, j-2, k-2] * self.Vol_cell_nominal
                return lhs

            lhs = jnp.where(is_box_boundary_node(i, j, k), get_lhs_on_box_boundary(node), get_lhs_at_interior_node(node))


            #--- RHS  
            def get_rhs_at_interior_node(node):
                i, j, k = node
                rhs = self.f_m_cube_internal[i-2, j-2, k-2] * V_m_ijk + self.f_p_cube_internal[i-2, j-2, k-2] * V_p_ijk
                rhs += self.beta_integrate_over_interface_at_point(node)
                return rhs
            
            def get_rhs_on_box_boundary(node):
                i, j, k = node
                return self.dirichlet_cube[i-2, j-2, k-2] * self.Vol_cell_nominal

            rhs = jnp.where(is_box_boundary_node(i, j, k), get_rhs_on_box_boundary(node), get_rhs_at_interior_node(node))

            return jnp.array([lhs, rhs])

        evaluate_on_nodes_fn = vmap(evaluate_discretization_lhs_rhs_at_node)
        lhs_rhs = evaluate_on_nodes_fn(self.nodes)
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

        curr_vertices = jnp.add(point, self.Xijk)
        u_cube_ijk = self.evaluate_solution_fn(params, curr_vertices)
        
        normal_ijk, gamma_m_ijk, gamma_m_ijk_pqm, gamma_p_ijk, gamma_p_ijk_pqm, zeta_m_ijk , zeta_m_ijk_pqm , zeta_p_ijk , zeta_p_ijk_pqm = self.get_regression_coeffs_at_point(point, dx, dy, dz)
        

        def bulk_point(is_interface_, u_ijk_):
            return jnp.array([jnp.where(is_interface_ == -1, u_ijk_, 0.0), jnp.where(is_interface_ == 1, u_ijk_, 0.0)])

        def interface_point(point):
            def mu_minus_bigger_fn(point):
                def extrapolate_u_m_from_negative_domain(r_ijk):
                    r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk                 
                    u_m = -1.0 * jnp.dot(gamma_m_ijk_pqm, u_cube_ijk)
                    u_m += (1.0 - gamma_m_ijk + gamma_m_ijk_pqm[:, 13]) * u_ijk
                    u_m += -1.0 * (1.0 - gamma_m_ijk) * (self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_p_interp_fn(r_m_proj))
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
                    u_m += (-1.0)*(self.alpha_interp_fn(r_m_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_m_proj) )
                    return u_m.reshape()

                def extrapolate_u_p_from_positive_domain_(r_ijk):
                    r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_p = -1.0 * jnp.dot(gamma_p_ijk_pqm, u_cube_ijk)
                    u_p += (1.0 - gamma_p_ijk + gamma_p_ijk_pqm[:, 13]) * u_ijk
                    u_p += (1.0 - gamma_p_ijk) * (self.alpha_interp_fn(r_p_proj) + delta_ijk * self.beta_over_mu_m_interp_fn(r_p_proj))
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




    #Compute normal gradients for error analysis
    def compute_normal_gradient_solution_mp_on_interface_regression(self, u, params=None):
        """
        Given the solution field u, this function computes gradient of u along normal direction
        of the level-set function on the interface itself; at r_proj.
        """
        u_cube = u.reshape(self.grid_shape)
        x = self.gstate.x
        y = self.gstate.y
        z = self.gstate.z
        u_mp = vmap(self.u_mp_fn, (None, None, None, None, 0, 0, 0))(u_cube, x, y, z, self.nodes[:,0], self.nodes[:,1], self.nodes[:,2])
        def convolve_at_node(node):
            i,j,k = node
            curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), self.ngbs)
            u_mp_pqm = u_mp.reshape(self.phi_cube_.shape+(2,))[curr_ngbs[:,0], curr_ngbs[:,1], curr_ngbs[:,2]]
            cm_pqm = self.Cm_ijk_pqm.reshape(self.phi_cube_.shape+ (-1,))[i-2,j-2,k-2]
            cp_pqm = self.Cp_ijk_pqm.reshape(self.phi_cube_.shape+ (-1,))[i-2,j-2,k-2]
            return jnp.sum(cm_pqm * u_mp_pqm[:,0]), jnp.sum(cp_pqm * u_mp_pqm[:,1])

        c_mp_u_mp_ngbs = vmap(convolve_at_node)(self.nodes)      
        grad_n_u_m = -1.0 * self.Cm_ijk_pqm.sum(axis=1) * u_mp[:,0] + c_mp_u_mp_ngbs[0]
        grad_n_u_p = -1.0 * self.Cp_ijk_pqm.sum(axis=1) * u_mp[:,1] + c_mp_u_mp_ngbs[1]
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
            i,j,k = node
            curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), self.ngbs)
            u_curr_ngbs = self.cube_at_v(u_cube, curr_ngbs)
            u_mp_node = self.u_mp_fn(u_cube, x, y, z, i, j, k)
            dU_mp = u_curr_ngbs[:,jnp.newaxis] - u_mp_node
            grad_m = d_m_mat @ dU_mp[:,0]
            grad_p = d_p_mat @ dU_mp[:,1]
            return grad_m, grad_p
        return vmap(convolve_at_node, (0,0,0))(self.nodes, self.D_m_mat, self.D_p_mat)  
    





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
            return jnp.array([jnp.where(is_interface_ == -1, u_ijk_, 0.0), jnp.where(is_interface_ == 1, u_ijk_, 0.0)])

        def interface_node(i, j, k):
            def mu_minus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_m_proj = r_ijk - delta_ijk * normal_ijk
                    r_m_proj = r_m_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_m_proj, -1), normal_ijk)
                    u_m = u_at_point_fn(r_ijk, 1) - self.alpha_interp_fn(r_m_proj)
                    u_m -= delta_ijk * ( (self.mu_m_over_mu_p_interp_fn(r_m_proj) - 1.0) * du_dn + self.beta_over_mu_p_interp_fn(r_m_proj) )
                    # u_m -= delta_ijk * ( (self.mu_m_interp_fn(r_m_proj) / self.mu_p_interp_fn(r_m_proj) - 1.0) *  du_dn + self.beta_interp_fn(r_m_proj)/self.mu_p_interp_fn(r_m_proj))
                    return u_m

                def extrapolate_u_p_from_positive_domain(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_p_proj = r_ijk - delta_ijk * normal_ijk
                    r_p_proj = r_p_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_p_proj, -1), normal_ijk)
                    u_p = u_at_point_fn(r_ijk, -1) + self.alpha_interp_fn(r_p_proj)
                    u_p += delta_ijk * ( (self.mu_m_over_mu_p_interp_fn(r_p_proj) - 1.0) * du_dn + self.beta_over_mu_p_interp_fn(r_p_proj) )
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
                    u_m -= delta_ijk * ( (1.0 - 1.0 / self.mu_m_over_mu_p_interp_fn(r_m_proj)) * du_dn + self.beta_over_mu_m_interp_fn(r_m_proj) )
                    # u_m -= delta_ijk * ( (1 - self.mu_p_interp_fn(r_m_proj)/self.mu_m_interp_fn(r_m_proj)) * du_dn + self.beta_interp_fn(r_m_proj) / self.mu_m_interp_fn(r_m_proj) )
                    return u_m

                def extrapolate_u_p_from_positive_domain_(i, j, k):
                    normal_ijk = self.normal_vec_fn((i, j, k))
                    r_p_proj = r_ijk - delta_ijk * normal_ijk
                    r_p_proj = r_p_proj[jnp.newaxis]
                    du_dn = jnp.dot(grad_u_at_point_fn(r_p_proj, 1), normal_ijk)
                    u_p = u_at_point_fn(r_ijk, -1) + self.alpha_interp_fn(r_p_proj)
                    u_p += delta_ijk * ( (1.0 - 1.0/self.mu_m_over_mu_p_interp_fn(r_p_proj)) * du_dn + self.beta_over_mu_m_interp_fn(r_p_proj) )
                    # u_p += delta_ijk * ((1 - self.mu_p_interp_fn(r_p_proj)/self.mu_m_interp_fn(r_p_proj)) * du_dn + self.beta_interp_fn(r_p_proj) / self.mu_m_interp_fn(r_p_proj))
                    return u_p

                u_m = jnp.where(delta_ijk > 0, extrapolate_u_m_from_negative_domain_(i, j, k), u_ijk)[0]
                u_p = jnp.where(delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain_(i, j, k))[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = self.mu_m_cube_internal[i-2, j-2, k-2]
            mu_p_ijk = self.mu_p_cube_internal[i-2, j-2, k-2]
            return jnp.where(mu_m_ijk > mu_p_ijk, mu_minus_bigger_fn(i, j, k), mu_plus_bigger_fn(i, j, k))

        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface((i, j, k))
        # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
        u_mp = jnp.where(is_interface == 0, interface_node(i, j, k), bulk_node(is_interface, u_ijk))
        return u_mp


    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, u, params):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R,  1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, -1)
        grad_n_u_m = vmap(jnp.dot, (0,0))(self.normal_vecs, grad_u_m)
        grad_n_u_p = vmap(jnp.dot, (0,0))(self.normal_vecs, grad_u_p)
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp_neural_network(self, u, params):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R,  1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(self.gstate.R, -1)
        return grad_u_m, grad_u_p






class DatasetDict:
    def __init__(self,
                 x_dict,
                 batch_size=64):
        self.batch_size = batch_size
        self.x_dict = x_dict
        self._len = None
        self._counter = 0
        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_dict)
        return self
    
    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        data_x = self.x_dict[self._idx: min(self._len, self._idx + self.batch_size)]
        self._idx += self.batch_size
        self._counter += 1
        return data_x




def poisson_solver(gstate, eval_gstate, sim_state, sim_state_fn, algorithm=0, switching_interval=3):

    #--- Defining Optimizer
    decay_rate_ = 0.975
    learning_rate = 1e-2
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=100,
        decay_rate=decay_rate_)
    optimizer = optax.chain(                         
                            optax.clip_by_global_norm(1.0), 
                            optax.scale_by_adam(),  
                            optax.scale_by_schedule(scheduler), 
                            optax.scale(-1.0) 
    )
    # optimizer = optax.adam(learning_rate)
    # optimizer = optax.rmsprop(learning_rate) 
    #---------------------

    trainer = PDETrainer(gstate, eval_gstate, sim_state, sim_state_fn, optimizer, algorithm)
    opt_state, params = trainer.init(); print_architecture(params)    



    num_epochs=10
    start_time = time.time()
    
    
    loss_epochs = []
    epoch_store = []
    BATCHSIZE_A6000 = 200000
    for epoch in range(num_epochs):            
        ds_iter = DatasetDict(eval_gstate.R, batch_size=BATCHSIZE_A6000)
        for x in ds_iter:
            opt_state, params, loss_epoch = trainer.update(opt_state, params, x, eval_gstate)   
        print(f"epoch # {epoch} loss is {loss_epoch}")
        loss_epochs.append(loss_epoch)
        epoch_store.append(epoch)

    

    #----- HALF JITTED
    # BATCHSIZE_A6000 = 100000
    # NUMDATA = len(eval_gstate.R)
    # num_batches = jnp.ceil(NUMDATA // BATCHSIZE_A6000)
    # def learn_whole_batched(carry, batch_idx):
    #     opt_state, params, egstate = carry
    #     x = egstate.R[batch_idx: jnp.min(jnp.array([NUMDATA, batch_idx + BATCHSIZE_A6000]))]
    #     opt_state, params, loss_epoch = trainer.update(opt_state, params, x, egstate)
    #     return (opt_state, params, loss_epoch, egstate), None
    # epoch_store = []
    # loss_epochs = []
    # loss_epochs = jnp.zeros(num_epochs)
    # batch_store = jnp.arange(num_batches)
    # for epoch in range(num_epochs):
    #     (opt_state, params, loss_epoch, eval_gstate), _ = jax.lax.scan(learn_whole_batched, (opt_state, params, eval_gstate), batch_store)
    #     loss_epochs.append(loss_epoch)
    #     epoch_store.append(epoch)
    #     print(f"epoch # {epoch} loss is {loss_epoch}")
 

    
    # def learn_whole(carry, epoch):
    #     opt_state, params, loss_epochs = carry
    #     opt_state, params, loss_epoch = trainer.update(opt_state, params, eval_gstate)
    #     loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
    #     return (opt_state, params, loss_epochs), None
    # loss_epochs = jnp.zeros(num_epochs)
    # epoch_store = jnp.arange(num_epochs)
    # (opt_state, params, loss_epochs), _ = jax.lax.scan(learn_whole, (opt_state, params, loss_epochs), epoch_store)
    
    
   
    
    
    
    end_time = time.time()
    print(f"solve took {end_time - start_time} (sec)")



    fig, ax = plt.subplots(figsize=(8,8))

    # plt.plot(epoch_store[epoch_store%switching_interval - 1 ==0], loss_epochs[epoch_store%switching_interval - 1 ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 <0], loss_epochs[epoch_store%switching_interval - 1 <0], color='b', label='negative domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 >0], loss_epochs[epoch_store%switching_interval - 1 >0], color='r', label='positive domain')

    # plt.plot(epoch_store[epoch_store%switching_interval ==0], loss_epochs[epoch_store%switching_interval ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[-1*( epoch_store%switching_interval) <0], loss_epochs[-1*(epoch_store%switching_interval) <0], color='b', label='negative domain')
    
    ax.plot(epoch_store, loss_epochs, color='k')
    
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rm epoch$', fontsize=20)
    ax.set_ylabel(r'$\rm loss$', fontsize=20)
    plt.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.tight_layout()
    plt.savefig('tests/poisson_solver_loss.png')
    plt.close()
  
    final_solution = trainer.evaluate_solution_fn(params, eval_gstate.R).reshape(-1)

    #------------- Gradients of discovered solutions are below:    
    if algorithm==0:
        grad_u_mp_normal_to_interface = None #trainer.compute_normal_gradient_solution_mp_on_interface(final_solution, None)
        grad_u_mp = None #trainer.compute_gradient_solution_mp(final_solution, None)
    elif algorithm==1:
        grad_u_mp_normal_to_interface = trainer.compute_normal_gradient_solution_mp_on_interface(None, params)
        grad_u_mp = trainer.compute_gradient_solution_mp(None, params)

    return final_solution, grad_u_mp, grad_u_mp_normal_to_interface, epoch_store, loss_epochs