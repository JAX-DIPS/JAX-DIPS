from jax import (numpy as jnp, vmap, jit, grad, random, nn as jnn)
from functools import partial
import optax
import haiku as hk
from src import (interpolate, util, geometric_integrations)
from src.nn_solution_trainer import train
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

f32 = util.f32
i32 = util.i32

from jax import config
config.update("jax_debug_nans", False)


def poisson_solver(gstate, sim_state):
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

    mu_m_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
        mu_m, gstate)
    mu_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
        mu_p, gstate)
    alpha_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
        alpha, gstate)
    beta_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(
        beta, gstate)

    phi_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, phi_cube_)
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)

    dirichlet_cube = dirichlet_bc.reshape(
        (xo.shape[0], yo.shape[0], zo.shape[0]))

    k_m_cube_internal = k_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    k_p_cube_internal = k_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    f_m_cube_internal = f_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    f_p_cube_internal = f_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    Nx = gstate.x.shape[0]
    Ny = gstate.y.shape[0]
    Nz = gstate.z.shape[0]

    ii = jnp.arange(2, Nx+2)
    jj = jnp.arange(2, Ny+2)
    kk = jnp.arange(2, Nz+2)

    I, J, K = jnp.meshgrid(ii, jj, kk, indexing='ij')

    nodes = jnp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1)))

    @jit
    def get_X_ijk():
        Xijk = jnp.array([[-dx, -dy, -dz],
                          [0.0, -dy, -dz],
                          [dx, -dy, -dz],
                          [-dx, 0.0, -dz],
                          [0.0, 0.0, -dz],
                          [dx, 0.0, -dz],
                          [-dx,  dy, -dz],
                          [0.0,  dy, -dz],
                          [dx,  dy, -dz],
                          [-dx, -dy, 0.0],
                          [0.0, -dy, 0.0],
                          [dx, -dy, 0.0],
                          [-dx, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [dx, 0.0, 0.0],
                          [-dx,  dy, 0.0],
                          [0.0,  dy, 0.0],
                          [dx,  dy, 0.0],
                          [-dx, -dy,  dz],
                          [0.0, -dy,  dz],
                          [dx, -dy,  dz],
                          [-dx, 0.0,  dz],
                          [0.0, 0.0,  dz],
                          [dx, 0.0,  dz],
                          [-dx,  dy,  dz],
                          [0.0,  dy,  dz],
                          [dx,  dy,  dz]], dtype=f32)

        ngbs = jnp.array([[-1, -1, -1],
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

        return Xijk, ngbs

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

    Xijk, ngbs = get_X_ijk()

    def cube_at(cube, ind):
        return cube[ind[0], ind[1], ind[2]]
    cube_at_v = jit(vmap(cube_at, (None, 0)))

    @jit
    def get_W_p_fn(cube, inds):
        return jnp.diag(vmap(sign_p_fn)(cube_at_v(cube, inds)))

    @jit
    def get_W_m_fn(cube, inds):
        return jnp.diag(vmap(sign_m_fn)(cube_at_v(cube, inds)))

    @jit
    def get_W_pm_matrices(node, phi_cube):
        i, j, k = node
        curr_ngbs = jnp.add(jnp.array([i, j, k]), ngbs)
        Wp = get_W_p_fn(phi_cube, curr_ngbs)
        Wm = get_W_m_fn(phi_cube, curr_ngbs)
        return Wm, Wp

    @jit
    def D_mp_node_update(node):
        Wijk_m, Wijk_p = get_W_pm_matrices(node, phi_cube)
        Dp = jnp.linalg.pinv(Xijk.T @ Wijk_p @ Xijk) @ (Wijk_p @ Xijk).T
        Dm = jnp.linalg.pinv(Xijk.T @ Wijk_m @ Xijk) @ (Wijk_m @ Xijk).T
        return jnp.nan_to_num(Dm), jnp.nan_to_num(Dp)
    D_mp_fn = jit(vmap(D_mp_node_update))

    D_m_mat, D_p_mat = D_mp_fn(nodes)

    @jit
    def normal_vec_fn(node):
        i, j, k = node
        phi_x = (phi_cube[i+1, j, k] - phi_cube[i-1, j, k]) / (f32(2) * dx)
        phi_y = (phi_cube[i, j+1, k] - phi_cube[i, j-1, k]) / (f32(2) * dy)
        phi_z = (phi_cube[i, j, k+1] - phi_cube[i, j, k-1]) / (f32(2) * dz)
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    normal_vecs = vmap(normal_vec_fn)(nodes)

    def get_c_ijk_pqm(normal_ijk, D_ijk):
        return normal_ijk @ D_ijk
    get_c_ijk_pqm_vec = jit(vmap(get_c_ijk_pqm, (0, 0)))

    Cm_ijk_pqm = get_c_ijk_pqm_vec(normal_vecs, D_m_mat)
    Cp_ijk_pqm = get_c_ijk_pqm_vec(normal_vecs, D_p_mat)
    # Cp_ijk_pqm_ = jnp.einsum("ij,ijk->ik", normal_vecs, D_p_mat)

    mu_m_cube_internal = mu_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    mu_p_cube_internal = mu_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))

    zeta_p_ijk_pqm = ((mu_p_cube_internal - mu_m_cube_internal) / mu_m_cube_internal) * phi_cube_
    zeta_p_ijk_pqm = zeta_p_ijk_pqm[...,jnp.newaxis] * Cp_ijk_pqm.reshape(phi_cube_.shape + (-1,))

    zeta_m_ijk_pqm = ((mu_p_cube_internal - mu_m_cube_internal) / mu_p_cube_internal) * phi_cube_
    zeta_m_ijk_pqm = zeta_m_ijk_pqm[...,jnp.newaxis] * Cm_ijk_pqm.reshape(phi_cube_.shape + (-1,))

    """
    NOTE: zeta_m_ijk_pqm and zeta_p_ijk_pqm are the size of the original grid, not the ghost layers included!
    for example: zeta_m_ijk_pqm[4,4,4][13] is the p=q=m=0 index, and zeta_m_ijk_pqm.shape = (128, 128, 128, 27)
    """
    zeta_p_ijk = (zeta_p_ijk_pqm.sum(axis=3) - zeta_p_ijk_pqm[:, :, :, 13]) * f32(-1.0)
    zeta_m_ijk = (zeta_m_ijk_pqm.sum(axis=3) - zeta_m_ijk_pqm[:, :, :, 13]) * f32(-1.0)

    gamma_p_ijk_pqm = zeta_p_ijk_pqm / (1.0 + zeta_p_ijk[:, :, :, jnp.newaxis])
    gamma_m_ijk_pqm = zeta_m_ijk_pqm / (1.0 - zeta_m_ijk[:, :, :, jnp.newaxis])

    gamma_p_ijk = (gamma_p_ijk_pqm.sum(axis=3) - gamma_p_ijk_pqm[:, :, :, 13]) * f32(-1.0)
    gamma_m_ijk = (gamma_m_ijk_pqm.sum(axis=3) - gamma_m_ijk_pqm[:, :, :, 13]) * f32(-1.0)



    """
    BEGIN geometric integration functions intiation
    Getting simplices of the grid: intersection points 
    """
    get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(
        gstate, sim_state)
    # integrate_over_interface_at_node, integrate_in_negative_domain_at_node = geometric_integrations.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, u_interp_fn)
    # alpha_integrate_over_interface_at_node, _ = geometric_integrations.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, alpha_interp_fn)
    beta_integrate_over_interface_at_node, _ = geometric_integrations.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, beta_interp_fn)
    compute_face_centroids_values_plus_minus_at_node = geometric_integrations.compute_cell_faces_areas_values(
        gstate, get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, mu_m_interp_fn, mu_p_interp_fn)

    """
    END Geometric integration functions initiated
    """
    Vol_cell_nominal = dx*dy*dz


    @jit
    def is_box_boundary_node(i, j, k):
        """
        Check if current node is on the boundary of box
        """
        boundary = (i-2)*(i-Nx-1)*(j-2)*(j-Ny-1)*(k-2)*(k-Nz-1)
        return jnp.where(boundary == 0, True, False)


    @jit
    def get_u_mp_at_node_fn(u_cube, i, j, k):
        """
        This function evaluates pairs of u^+ and u^- at each grid point
        in the domain, given a current cube of u values.
      
        BIAS SLOW:
            This function evaluates 
                u_m = B_m : u + r_m 
            and 
                u_p = B_p : u + r_p
        """

        def bulk_node(is_interface_, u_ijk_):
            return jnp.array([jnp.where(is_interface_ == -1, u_ijk_, 0.0), jnp.where(is_interface_ == 1, u_ijk_, 0.0)])

        def interface_node(i, j, k):
            def mu_minus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain(i, j, k):
                    delta_ijk = phi_cube[i, j, k] 
                    r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                    r_m_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                    r_m_proj = r_m_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
                    u_m = -1.0 * jnp.dot(gamma_m_ijk_pqm[i-2, j-2, k-2], cube_at_v(u_cube, curr_ngbs))
                    u_m += (1.0 - gamma_m_ijk[i-2, j-2, k-2] + gamma_m_ijk_pqm[i-2, j-2, k-2, 13]) * u_cube[i-2, j-2, k-2]
                    u_m += -1.0 * (1.0 - gamma_m_ijk[i-2, j-2, k-2]) * (alpha_interp_fn(r_m_proj) + delta_ijk * beta_interp_fn(r_m_proj) / mu_p_interp_fn(r_m_proj))
                    return u_m

                def extrapolate_u_p_from_positive_domain(i, j, k):
                    delta_ijk = phi_cube[i, j, k]  
                    r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                    r_p_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                    r_p_proj = r_p_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
                    u_p = -1.0 * jnp.dot(zeta_m_ijk_pqm[i-2, j-2, k-2], cube_at_v(u_cube, curr_ngbs))
                    u_p += (1.0 - zeta_m_ijk[i-2, j-2, k-2] + zeta_m_ijk_pqm[i-2, j-2, k-2, 13]) * u_cube[i-2, j-2, k-2]
                    u_p += alpha_interp_fn(r_p_proj) + delta_ijk * beta_interp_fn(r_p_proj) / mu_p_interp_fn(r_p_proj)
                    return u_p
                phi_ijk = phi_cube[i, j, k]
                u_m = jnp.where(phi_ijk > 0, extrapolate_u_m_from_negative_domain(i, j, k), u_cube[i-2, j-2, k-2])[0]
                u_p = jnp.where(phi_ijk > 0, u_cube[i-2, j-2, k-2], extrapolate_u_p_from_positive_domain(i, j, k))[0]
                return jnp.array([u_m, u_p])

            def mu_plus_bigger_fn(i, j, k):
                def extrapolate_u_m_from_negative_domain_(i, j, k):
                    delta_ijk = phi_cube[i, j, k] 
                    r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                    r_m_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                    r_m_proj = r_m_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
                    u_m = -1.0 * jnp.dot(zeta_p_ijk_pqm[i-2, j-2, k-2], cube_at_v(u_cube, curr_ngbs))
                    u_m += (1.0 - zeta_p_ijk[i-2, j-2, k-2] + zeta_p_ijk_pqm[i-2, j-2, k-2, 13]) * u_cube[i-2, j-2, k-2]
                    u_m += (-1.0)*(alpha_interp_fn(r_m_proj) + delta_ijk * beta_interp_fn(r_m_proj) / mu_m_interp_fn(r_m_proj) )
                    return u_m

                def extrapolate_u_p_from_positive_domain_(i, j, k):
                    delta_ijk = phi_cube[i, j, k] 
                    r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                    r_p_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                    r_p_proj = r_p_proj[jnp.newaxis]
                    curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
                    u_p = -1.0 * jnp.dot(gamma_p_ijk_pqm[i-2, j-2, k-2], cube_at_v(u_cube, curr_ngbs))
                    u_p += (1.0 - gamma_p_ijk[i-2, j-2, k-2] + gamma_p_ijk_pqm[i-2, j-2, k-2, 13]) * u_cube[i-2, j-2, k-2]
                    u_p += (1.0 - gamma_p_ijk[i-2, j-2, k-2]) * (alpha_interp_fn(r_p_proj) + delta_ijk * beta_interp_fn(r_p_proj) / mu_m_interp_fn(r_p_proj))
                    return u_p
                phi_ijk = phi_cube[i, j, k]
                u_m = jnp.where(phi_ijk > 0, extrapolate_u_m_from_negative_domain_(i, j, k), u_cube[i-2, j-2, k-2])[0]
                u_p = jnp.where(phi_ijk > 0, u_cube[i-2, j-2, k-2], extrapolate_u_p_from_positive_domain_(i, j, k))[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = mu_m_cube_internal[i-2, j-2, k-2]
            mu_p_ijk = mu_p_cube_internal[i-2, j-2, k-2]
            return jnp.where(mu_m_ijk > mu_p_ijk, mu_minus_bigger_fn(i, j, k), mu_plus_bigger_fn(i, j, k))

        u_ijk = u_cube[i-2, j-2, k-2]
        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = is_cell_crossed_by_interface((i, j, k))
        u_mp = jnp.where(is_interface == 0, interface_node(i, j, k), bulk_node(is_interface, u_ijk))
        return u_mp
        
       


    @jit
    def compute_Ax_and_b_fn(u):
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
        
        u_cube = u.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))

        u_mp_at_node = partial(get_u_mp_at_node_fn, u_cube)

        def evaluate_discretization_lhs_rhs_at_node(node):
            #--- LHS
            i, j, k = node
            # poisson_scheme_coeffs = compute_face_centroids_values_plus_minus_at_node(node)
            # coeffs, vols = jnp.split(poisson_scheme_coeffs, [12], axis=0)

            coeffs_ = compute_face_centroids_values_plus_minus_at_node(node)
            
            coeffs = coeffs_[:12]
            vols = coeffs_[12:14]
            # areas = coeffs_[14:]

            V_m_ijk = vols[0]
            V_p_ijk = vols[1]

            def get_lhs_at_interior_node(node):
                i, j, k = node
                # k_cube's don't have ghost layers
                k_m_ijk = k_m_cube_internal[i-2, j-2, k-2]
                k_p_ijk = k_p_cube_internal[i-2, j-2, k-2]
                u_m_ijk, u_p_ijk = u_mp_at_node(i, j, k)
                lhs  = k_m_ijk * V_m_ijk * u_m_ijk
                lhs += k_p_ijk * V_p_ijk * u_p_ijk
                lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + \
                       (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]) * u_p_ijk
                u_m_imjk, u_p_imjk = u_mp_at_node(i-1, j  , k  )
                lhs += -1.0 * coeffs[0] * u_m_imjk - coeffs[1] * u_p_imjk
                u_m_ipjk, u_p_ipjk = u_mp_at_node(i+1, j  , k  )
                lhs += -1.0 * coeffs[2] * u_m_ipjk - coeffs[3] * u_p_ipjk
                u_m_ijmk, u_p_ijmk = u_mp_at_node(i  , j-1, k  )
                lhs += -1.0 * coeffs[4] * u_m_ijmk - coeffs[5] * u_p_ijmk
                u_m_ijpk, u_p_ijpk = u_mp_at_node(i  , j+1, k  )
                lhs += -1.0 * coeffs[6] * u_m_ijpk - coeffs[7] * u_p_ijpk
                u_m_ijkm, u_p_ijkm = u_mp_at_node(i  , j  , k-1)
                lhs += -1.0 * coeffs[8] * u_m_ijkm - coeffs[9] * u_p_ijkm
                u_m_ijkp, u_p_ijkp = u_mp_at_node(i  , j  , k+1)
                lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp
                return lhs

            def get_lhs_on_box_boundary(node):
                i, j, k = node
                lhs = u_cube[i-2, j-2, k-2] * Vol_cell_nominal
                return lhs
            lhs = jnp.where(is_box_boundary_node(i, j, k), get_lhs_on_box_boundary(node), get_lhs_at_interior_node(node))

            #--- RHS
            def get_rhs_at_interior_node(node):
                i, j, k = node
                rhs = f_m_cube_internal[i-2, j-2, k-2] * V_m_ijk + f_p_cube_internal[i-2, j-2, k-2] * V_p_ijk
                rhs += beta_integrate_over_interface_at_node(node)
                return rhs

            def get_rhs_on_box_boundary(node):
                """
                Imposing Dirichlet BCs on the RHS
                """
                i, j, k = node
                return dirichlet_cube[i-2, j-2, k-2] * Vol_cell_nominal
            rhs = jnp.where(is_box_boundary_node(i, j, k), get_rhs_on_box_boundary(node), get_rhs_at_interior_node(node))

            return jnp.array([lhs, rhs])

        evaluate_on_nodes_fn = vmap(evaluate_discretization_lhs_rhs_at_node)
        lhs_rhs = evaluate_on_nodes_fn(nodes)
        return lhs_rhs


    #------ Solvers are below:


    #--- Defining Optimizer
    
    # ------ Exponential decay of the learning rate.
    decay_rate_ = 0.975
    learning_rate = 1e-2
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=100,
        decay_rate=decay_rate_)
    # Combining gradient transforms using `optax.chain`.
    optimizer = optax.chain(                         
                            optax.clip_by_global_norm(1.0), # Clip the gradient by the global norm.
                            optax.scale_by_adam(),  # Use the updates from adam.
                            optax.scale_by_schedule(scheduler), # Use the learning rate from the scheduler.
                            optax.scale(-1.0) # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    )

    # optimizer = optax.adam(learning_rate)
    # optimizer = optax.rmsprop(learning_rate)
    shape_grid = (xo.shape[0], yo.shape[0], zo.shape[0])
    final_solution = train(optimizer, compute_Ax_and_b_fn, gstate.R, phi_cube_.reshape(-1), shape_grid, dirichlet_cube, Vol_cell_nominal, num_epochs=10000)



    #------- Explicitly solving for each node value is below:

    # @jit
    # def compute_residual(x):
    #     """
    #     Evaluates the residual given x in the l2 norm.
    #     """
    #     lhs_rhs = compute_Ax_and_b_fn(x)
    #     lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
    #     loss = optax.l2_loss(lhs, rhs).mean() 
    #     # regularizer on boundaries
    #     x_cube = x.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    #     loss += jnp.square(x_cube[ 0, :, :] - dirichlet_cube[ 0, :, :]).mean() * Vol_cell_nominal
    #     loss += jnp.square(x_cube[-1, :, :] - dirichlet_cube[-1, :, :]).mean() * Vol_cell_nominal
    #     loss += jnp.square(x_cube[ :, 0, :] - dirichlet_cube[ :, 0, :]).mean() * Vol_cell_nominal
    #     loss += jnp.square(x_cube[ :,-1, :] - dirichlet_cube[ :,-1, :]).mean() * Vol_cell_nominal
    #     loss += jnp.square(x_cube[ :, :, 0] - dirichlet_cube[ :, :, 0]).mean() * Vol_cell_nominal
    #     loss += jnp.square(x_cube[ :, :,-1] - dirichlet_cube[ :, :,-1]).mean() * Vol_cell_nominal
    #     return loss

    # --- initiate iterations from provided guess
    # x = sim_state.solution
    # # --- impose the dirichlet bc because bc's won't be updated.
    # x_cube = x.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    # x_cube = x_cube.at[ 0, :, :].set(dirichlet_cube[ 0, :, :])
    # x_cube = x_cube.at[-1, :, :].set(dirichlet_cube[-1, :, :])
    # x_cube = x_cube.at[ :, 0, :].set(dirichlet_cube[ :, 0, :])
    # x_cube = x_cube.at[ :,-1, :].set(dirichlet_cube[ :,-1, :])
    # x_cube = x_cube.at[ :, :, 0].set(dirichlet_cube[ :, :, 0])
    # x_cube = x_cube.at[ :, :,-1].set(dirichlet_cube[ :, :,-1])
    # x = x_cube.reshape(-1)

    
    # Optimization Problem is set up by defining: (1) params, (2) compute_loss(params) 
    # params = {'u': x}                        # parameters to be optimized
    # @jit
    # def compute_loss(params):                # loss function to minimize
    #     return compute_residual(params['u'])
    


    # # Generic Optimization Routine 
    # loss_store = []
    # grad_fn = jit(grad(compute_loss))
    # opt_state = optimizer.init(params)
    # for _ in range(20000):
    #     grads = grad_fn(params)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)

    #     loss_ = compute_loss(params)
    #     loss_store.append(loss_.tolist())
    #     print(f"iteration {_} loss = {loss_}")

    # plt.figure(figsize=(8, 8))
    # plt.plot(loss_store)
    # plt.yscale('log')
    # plt.xlabel('epoch', fontsize=20)
    # plt.ylabel('loss', fontsize=20)
    # plt.savefig('tests/poisson_solver_loss.png')
    # plt.close()
    
   

    #------------- Gradients of discovered solutions are below:

    #Compute normal gradients for error analysis
    def compute_normal_gradient_solution_mp_on_interface(u):
        """
        Given the solution field u, this function computes gradient of u along normal direction
        of the level-set function on the interface itself; at r_proj.
        """
        u_cube = u.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
        u_mp = vmap(get_u_mp_at_node_fn, (None, 0, 0, 0))(u_cube, nodes[:,0], nodes[:,1], nodes[:,2])
        def convolve_at_node(node):
            i,j,k = node
            curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
            u_mp_pqm = u_mp.reshape(phi_cube_.shape+(2,))[curr_ngbs[:,0], curr_ngbs[:,1], curr_ngbs[:,2]]
            cm_pqm = Cm_ijk_pqm.reshape(phi_cube_.shape+ (-1,))[i-2,j-2,k-2]
            cp_pqm = Cp_ijk_pqm.reshape(phi_cube_.shape+ (-1,))[i-2,j-2,k-2]
            return jnp.sum(cm_pqm * u_mp_pqm[:,0]), jnp.sum(cp_pqm * u_mp_pqm[:,1])

        c_mp_u_mp_ngbs = vmap(convolve_at_node)(nodes)      
        grad_n_u_m = -1.0 * Cm_ijk_pqm.sum(axis=1) * u_mp[:,0] + c_mp_u_mp_ngbs[0]
        grad_n_u_p = -1.0 * Cp_ijk_pqm.sum(axis=1) * u_mp[:,1] + c_mp_u_mp_ngbs[1]
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp(u):
        """
        This function computes \nabla u^+ and \nabla u^- given a solution vector u.
        """
        u_cube = u.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
        def convolve_at_node(node, d_m_mat, d_p_mat):
            i,j,k = node
            curr_ngbs = jnp.add(jnp.array([i-2, j-2, k-2]), ngbs)
            u_curr_ngbs = cube_at_v(u_cube, curr_ngbs)
            u_mp_node = get_u_mp_at_node_fn(u_cube, i, j, k)
            dU_mp = u_curr_ngbs[:,jnp.newaxis] - u_mp_node
            grad_m = d_m_mat @ dU_mp[:,0]
            grad_p = d_p_mat @ dU_mp[:,1]
            return grad_m, grad_p
        return vmap(convolve_at_node, (0,0,0))(nodes, D_m_mat, D_p_mat)  
    
    # grad_u_mp_normal_to_interface = compute_normal_gradient_solution_mp_on_interface(params['u'])
    # grad_u_mp = compute_gradient_solution_mp(params['u'])
    # return params['u'], grad_u_mp, grad_u_mp_normal_to_interface

    grad_u_mp_normal_to_interface = compute_normal_gradient_solution_mp_on_interface(final_solution)
    grad_u_mp = compute_gradient_solution_mp(final_solution)
    return final_solution, grad_u_mp, grad_u_mp_normal_to_interface