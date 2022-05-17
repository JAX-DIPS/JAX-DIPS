from functools import partial
from jax import (numpy as jnp, vmap, jit, grad, lax, ops)
from jax.scipy.sparse.linalg import gmres, bicgstab, cg
import optax
from src import (interpolate, util, geometric_integrations)
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

f32 = util.f32
i32 = util.i32

    


def poisson_solver(gstate, sim_state):
    phi_n = sim_state.phi
    mu_m = sim_state.mu_m
    mu_p = sim_state.mu_p
    k_m = sim_state.k_m
    k_p = sim_state.k_p
    f_m = sim_state.f_m
    f_p = sim_state.f_p

    alpha = sim_state.alpha
    beta = sim_state.beta

    xo = gstate.x; yo = gstate.y; zo = gstate.z

    mu_m_interp_fn  = interpolate.nonoscillatory_quadratic_interpolation(mu_m, gstate)
    mu_p_interp_fn  = interpolate.nonoscillatory_quadratic_interpolation(mu_p, gstate)
    alpha_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(alpha, gstate)
    beta_interp_fn  = interpolate.nonoscillatory_quadratic_interpolation(beta, gstate)
    
    phi_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, phi_cube_)
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)

    k_m_cube_internal = k_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    k_p_cube_internal = k_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    f_m_cube_internal = f_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    f_p_cube_internal = f_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))


    dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]

    Nx = gstate.x.shape[0]
    Ny = gstate.y.shape[0]
    Nz = gstate.z.shape[0]

    ii = jnp.arange(2, Nx+2)
    jj = jnp.arange(2, Ny+2)
    kk = jnp.arange(2, Nz+2)
    
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing='ij')
    
    nodes = jnp.column_stack( (I.reshape(-1), J.reshape(-1), K.reshape(-1) ))

    @jit
    def get_X_ijk():
        Xijk = jnp.array([[-dx, -dy, -dz],
                          [0.0, -dy, -dz],
                          [ dx, -dy, -dz],
                          [-dx, 0.0, -dz],
                          [0.0, 0.0, -dz],
                          [ dx, 0.0, -dz],
                          [-dx,  dy, -dz],
                          [0.0,  dy, -dz],
                          [ dx,  dy, -dz],
                          [-dx, -dy, 0.0],
                          [0.0, -dy, 0.0],
                          [ dx, -dy, 0.0],
                          [-dx, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [ dx, 0.0, 0.0],
                          [-dx,  dy, 0.0],
                          [0.0,  dy, 0.0],
                          [ dx,  dy, 0.0],
                          [-dx, -dy,  dz],
                          [0.0, -dy,  dz],
                          [ dx, -dy,  dz],
                          [-dx, 0.0,  dz],
                          [0.0, 0.0,  dz],
                          [ dx, 0.0,  dz],
                          [-dx,  dy,  dz],
                          [0.0,  dy,  dz],
                          [ dx,  dy,  dz] ], dtype=f32)

        ngbs = jnp.array([[-1, -1, -1],
                          [ 0, -1, -1],
                          [ 1, -1, -1],
                          [-1,  0, -1],
                          [ 0,  0, -1],
                          [ 1,  0, -1],
                          [-1,  1, -1],
                          [ 0,  1, -1],
                          [ 1,  1, -1],
                          [-1, -1,  0],
                          [ 0, -1,  0],
                          [ 1, -1,  0],
                          [-1,  0,  0],
                          [ 0,  0,  0],
                          [ 1,  0,  0],
                          [-1,  1,  0],
                          [ 0,  1,  0],
                          [ 1,  1,  0],
                          [-1, -1,  1],
                          [ 0, -1,  1],
                          [ 1, -1,  1],
                          [-1,  0,  1],
                          [ 0,  0,  1],
                          [ 1,  0,  1],
                          [-1,  1,  1],
                          [ 0,  1,  1],
                          [ 1,  1,  1] ], dtype=i32)
        
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
        return cube[ ind[0], ind[1], ind[2] ]
    cube_at_v = jit(vmap(cube_at, (None, 0)))   

    @jit
    def get_W_p_fn(cube, inds):
        return jnp.diag( vmap(sign_p_fn) (cube_at_v(cube, inds)) )

    @jit
    def get_W_m_fn(cube, inds):
        return jnp.diag( vmap(sign_m_fn) (cube_at_v(cube, inds)) ) 

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
        Dp = jnp.linalg.inv(Xijk.T @ Wijk_p @ Xijk) @ (Wijk_p @ Xijk).T
        Dm = jnp.linalg.inv(Xijk.T @ Wijk_m @ Xijk) @ (Wijk_m @ Xijk).T
        return jnp.nan_to_num(Dm), jnp.nan_to_num(Dp)
    D_mp_fn = jit(vmap(D_mp_node_update))

    D_m_mat, D_p_mat = D_mp_fn(nodes)
    
    
    @jit
    def normal_vec_fn(node):
        i, j, k = node
        phi_x = (phi_cube[i+1, j , k  ] - phi_cube[i-1,j  ,k  ]) / (f32(2) * dx)
        phi_y = (phi_cube[i, j+1 , k  ] - phi_cube[i  ,j-1,k  ]) / (f32(2) * dy)
        phi_z = (phi_cube[i, j   , k+1] - phi_cube[i  ,j  ,k-1]) / (f32(2) * dz) 
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)
    
    normal_vecs = vmap(normal_vec_fn)(nodes)


    def get_c_ijk_pqm(normal_ijk, D_ijk):
        return normal_ijk @ D_ijk
    get_c_ijk_pqm_vec = jit(vmap(get_c_ijk_pqm, (0, 0)))

    Cp_ijk_pqm = get_c_ijk_pqm_vec(normal_vecs, D_m_mat)
    Cm_ijk_pqm = get_c_ijk_pqm_vec(normal_vecs, D_p_mat)



    mu_m_cube_internal = mu_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    mu_p_cube_internal = mu_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    
    zeta_p_ijk_pqm = ((mu_p_cube_internal - mu_m_cube_internal) / mu_m_cube_internal) * phi_cube_ 
    zeta_p_ijk_pqm = zeta_p_ijk_pqm.reshape(zeta_p_ijk_pqm.shape + (-1,)) * Cp_ijk_pqm.reshape(phi_cube_.shape + (-1,))
    
    zeta_m_ijk_pqm = ((mu_p_cube_internal - mu_m_cube_internal) / mu_p_cube_internal) * phi_cube_ 
    zeta_m_ijk_pqm = zeta_m_ijk_pqm.reshape(zeta_m_ijk_pqm.shape + (-1,)) * Cm_ijk_pqm.reshape(phi_cube_.shape + (-1,))
    
    """
    NOTE: zeta_m_ijk_pqm and zeta_p_ijk_pqm are the size of the original grid, not the ghost layers included!
    for example: zeta_m_ijk_pqm[4,4,4][13] is the p=q=m=0 index, and zeta_m_ijk_pqm.shape = (128, 128, 128, 27)
    """
    zeta_p_ijk = ( zeta_p_ijk_pqm.sum(axis=3) - zeta_p_ijk_pqm[:,:,:,13] ) * f32(-1.0)
    zeta_m_ijk = ( zeta_m_ijk_pqm.sum(axis=3) - zeta_m_ijk_pqm[:,:,:,13] ) * f32(-1.0)
    
    
    gamma_p_ijk_pqm = zeta_p_ijk_pqm / (1.0 + zeta_p_ijk[:,:,:,jnp.newaxis])
    gamma_m_ijk_pqm = zeta_m_ijk_pqm / (1.0 - zeta_m_ijk[:,:,:,jnp.newaxis])

    gamma_p_ijk = (gamma_p_ijk_pqm.sum(axis=3) - gamma_p_ijk_pqm[:,:,:,3] ) * f32(-1.0)
    gamma_m_ijk = (gamma_m_ijk_pqm.sum(axis=3) - gamma_m_ijk_pqm[:,:,:,3] ) * f32(-1.0)


    
    """
    BEGIN geometric integration functions intiation
    Getting simplices of the grid: intersection points 
    """
    get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface = geometric_integrations.get_vertices_of_cell_intersection_with_interface_at_node(gstate, sim_state)
    # integrate_over_interface_at_node, integrate_in_negative_domain_at_node = geometric_integrations.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, u_interp_fn)
    # alpha_integrate_over_interface_at_node, _ = geometric_integrations.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, alpha_interp_fn)
    beta_integrate_over_interface_at_node, _ = geometric_integrations.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, beta_interp_fn)
    compute_face_centroids_values_plus_minus_at_node = geometric_integrations.compute_cell_faces_areas_values(gstate, get_vertices_of_cell_intersection_with_interface_at_node, is_cell_crossed_by_interface, mu_m_interp_fn, mu_p_interp_fn)
    
    """
    END Geometric integration functions initiated
    """
    @jit
    def compute_Ax_and_b_fn(u):
        """
        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2 

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field. 

        Note that this should return same shape and type as of u.
        This function is needed to be fed into jax sparse linalg solvers such as gmres:

        jax.scipy.sparse.linalg.gmres(A, b, x0=None, *, tol=1e-05, atol=0.0, restart=20, maxiter=None, M=None, solve_method='batched')
        
        A = this function!
        
        * PROCEDURE: 
            first compute u = B:u + r for each node
            then use the actual cell geometries (face areas and mu coeffs) to 
            compute the rhs of the linear system given currently passed-in u vector
            for solution estimate.
        
        """
        u_cube = u.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
        u_cube = u_cube.at[0,:,:].set(0.0)
        u_cube = u_cube.at[-1,:,:].set(0.0)
        u_cube = u_cube.at[:,0,:].set(0.0)
        u_cube = u_cube.at[:,-1,:].set(0.0)
        u_cube = u_cube.at[:,:,0].set(0.0)
        u_cube = u_cube.at[:,:,-1].set(0.0)
        """
        Impose boundary conditions:
            Dirichlet: update 
        """
        x_, y_, z_, u_cube = interpolate.add_ghost_layer_3d_Dirichlet_extension(xo, yo, zo, u_cube)
        _, _, _, u_cube = interpolate.add_ghost_layer_3d_Dirichlet_extension(x_, y_, z_, u_cube)


        @jit
        def is_box_boundary_node(i, j, k):
            """
            Check if current node is on the boundary of box
            """
            return jnp.where((i-2)*(i-Nx-2)*(j-2)*(j-Ny-2)*(k-2)*(k-Nz-2)==0, True, False)


        @jit 
        def u_mp_dirichlet_boundary_at_node(i, j, k):
            """
            Dirichlet boundary condition around the box
            """
            u_m = 0.0
            u_p = 0.0
            return jnp.array([u_m, u_p])


        @jit
        def u_mp_at_interior_node(i, j, k):
            """
            BIAS SLOW:
                This function evaluates 
                    u_m = B_m : u + r_m 
                and 
                    u_p = B_p : u + r_p
            """
            def bulk_node(is_interface_, u_ijk_):
                return jnp.array([ jnp.where(is_interface_==-1, u_ijk_, 0.0), jnp.where(is_interface_==1, u_ijk_, 0.0) ])

            def interface_node(i, j, k):
                def mu_minus_bigger_fn(i, j, k):
                    def extrapolate_u_m_from_negative_domain(i, j, k):
                        delta_ijk = phi_cube[i, j, k]
                        r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                        r_m_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                        r_m_proj = r_m_proj[jnp.newaxis]
                        curr_ngbs = jnp.add(jnp.array([i, j, k]), ngbs)
                        u_m  = -1.0 * jnp.dot(gamma_m_ijk_pqm[i-2,j-2,k-2], cube_at_v(u_cube, curr_ngbs) ) 
                        u_m += (1.0 - gamma_m_ijk[i-2, j-2, k-2] + gamma_m_ijk_pqm[i-2,j-2,k-1,13]) * u_cube[i,j,k] 
                        u_m += -1.0 * (1.0 - gamma_m_ijk[i-2, j-2, k-2]) * ( alpha_interp_fn(r_m_proj) + delta_ijk * beta_interp_fn(r_m_proj) / mu_p_interp_fn(r_m_proj) )
                        return u_m
                    def extrapolate_u_p_from_positive_domain(i, j, k):
                        delta_ijk = phi_cube[i, j, k]
                        r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                        r_p_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                        r_p_proj = r_p_proj[jnp.newaxis]
                        curr_ngbs = jnp.add(jnp.array([i, j, k]), ngbs)
                        u_p  = -1.0 * jnp.dot(zeta_m_ijk_pqm[i-2,j-2,k-2], cube_at_v(u_cube, curr_ngbs) )
                        u_p += (1.0 - zeta_m_ijk[i-2, j-2, k-2] + zeta_m_ijk_pqm[i-2,j-2,k-1,13]) * u_cube[i,j,k] 
                        u_p += alpha_interp_fn(r_p_proj) + delta_ijk * beta_interp_fn(r_p_proj) / mu_p_interp_fn(r_p_proj)
                        return u_p
                    phi_ijk = phi_cube[i, j, k]
                    u_m = jnp.where(phi_ijk>0, extrapolate_u_m_from_negative_domain(i, j, k), u_cube[i, j, k])[0]
                    u_p = jnp.where(phi_ijk>0, u_cube[i, j, k], extrapolate_u_p_from_positive_domain(i, j, k))[0]
                    return jnp.array([u_m, u_p])
                
                def mu_plus_bigger_fn(i, j, k):
                    def extrapolate_u_m_from_negative_domain_(i, j, k):
                        delta_ijk = phi_cube[i, j, k]
                        r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                        r_m_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                        r_m_proj = r_m_proj[jnp.newaxis]
                        curr_ngbs = jnp.add(jnp.array([i, j, k]), ngbs)
                        u_m  = -1.0 * jnp.dot(zeta_p_ijk_pqm[i-2,j-2,k-2], cube_at_v(u_cube, curr_ngbs) )
                        u_m += (1.0 - zeta_p_ijk[i-2, j-2, k-2] + zeta_p_ijk_pqm[i-2,j-2,k-1,13]) * u_cube[i,j,k] 
                        u_m += alpha_interp_fn(r_m_proj) + delta_ijk * beta_interp_fn(r_m_proj) / mu_m_interp_fn(r_m_proj)
                        return u_m
                    def extrapolate_u_p_from_positive_domain_(i, j, k):
                        delta_ijk = phi_cube[i, j, k]
                        r_ijk = jnp.array([x[i], y[j], z[k]], dtype=f32)
                        r_p_proj = r_ijk - delta_ijk * normal_vec_fn((i, j, k))
                        r_p_proj = r_p_proj[jnp.newaxis]
                        curr_ngbs = jnp.add(jnp.array([i, j, k]), ngbs)
                        u_p  = -1.0 * jnp.dot(gamma_p_ijk_pqm[i-2,j-2,k-2], cube_at_v(u_cube, curr_ngbs) )
                        u_p += (1.0 - gamma_p_ijk[i-2, j-2, k-2] + gamma_p_ijk_pqm[i-2,j-2,k-1,13]) * u_cube[i,j,k]
                        u_p += (1.0 - gamma_p_ijk[i-2, j-2, k-2]) * (alpha_interp_fn(r_p_proj) + delta_ijk * beta_interp_fn(r_p_proj) / mu_m_interp_fn(r_p_proj))
                        return u_p
                    phi_ijk = phi_cube[i, j, k]
                    u_m = jnp.where(phi_ijk>0, extrapolate_u_m_from_negative_domain_(i, j, k), u_cube[i, j, k])[0]
                    u_p = jnp.where(phi_ijk>0, u_cube[i, j, k], extrapolate_u_p_from_positive_domain_(i, j, k))[0]
                    return jnp.array([u_m, u_p])

                mu_m_ijk = mu_m_cube_internal[i-2, j-2, k-2]
                mu_p_ijk = mu_p_cube_internal[i-2, j-2, k-2]
                return jnp.where(mu_m_ijk > mu_p_ijk, mu_minus_bigger_fn(i, j, k), mu_plus_bigger_fn(i, j, k))
                
            u_ijk = u_cube[i, j, k]
            is_interface = is_cell_crossed_by_interface((i, j, k)) # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
            u_mp = jnp.where(is_interface==0, interface_node(i, j, k), bulk_node(is_interface, u_ijk))
            return u_mp
            

        @jit
        def u_mp_at_node(i, j, k):
            """
            Main u_minus/plus evaluator, takes care if node is on box boundary or is an interior node.
            """
            return jnp.where(is_box_boundary_node(i, j, k), u_mp_dirichlet_boundary_at_node(i, j, k), u_mp_at_interior_node(i, j, k))


        # @jit
        def evaluate_discretization_lhs_rhs_at_node(node):
            
            i, j, k = node
            
            poisson_scheme_coeffs = compute_face_centroids_values_plus_minus_at_node(node)
            coeffs, vols = jnp.split(poisson_scheme_coeffs, [12], axis=0)
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]
            
            #--- LHS
            k_m_ijk = k_m_cube_internal[i-2, j-2, k-2]  # k_cube's don't have ghost layers
            k_p_ijk = k_p_cube_internal[i-2, j-2, k-2]
            

            u_m_ijk, u_p_ijk = u_mp_at_node(i, j, k)
            lhs  = k_m_ijk * V_m_ijk * u_m_ijk
            lhs += k_p_ijk * V_p_ijk * u_p_ijk
            lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6]) * u_m_ijk + (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7]) * u_p_ijk

            u_m_imjk, u_p_imjk = u_mp_at_node(i-1, j, k)
            lhs += -1.0 * coeffs[0] * u_m_imjk - coeffs[1] * u_p_imjk

            u_m_ipjk, u_p_ipjk = u_mp_at_node(i+1, j, k)
            lhs += -1.0 * coeffs[2] * u_m_ipjk - coeffs[3] * u_p_ipjk

            u_m_ijmk, u_p_ijmk = u_mp_at_node(i, j-1, k)
            lhs += -1.0 * coeffs[4] * u_m_ijmk - coeffs[5] * u_p_ijmk

            u_m_ijpk, u_p_ijpk = u_mp_at_node(i, j+1, k)
            lhs += -1.0 * coeffs[6] * u_m_ijpk - coeffs[7] * u_p_ijpk

            u_m_ijkm, u_p_ijkm = u_mp_at_node(i, j, k-1)
            lhs += -1.0 * coeffs[8] * u_m_ijkm - coeffs[9] * u_p_ijkm

            u_m_ijkp, u_p_ijkp = u_mp_at_node(i, j, k+1)
            lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

            #--- RHS
            rhs = f_m_cube_internal[i-2, j-2, k-2] * V_m_ijk + f_p_cube_internal[i-2, j-2, k-2] * V_p_ijk
            rhs += beta_integrate_over_interface_at_node(node)

            return jnp.array([lhs, rhs])
        # lhs_rhs = evaluate_discretization_lhs_rhs_at_node(nodes[794302])
        # lhs_rhs = jit(vmap(evaluate_discretization_lhs_rhs_at_node))(nodes)
        evaluate_on_nodes_fn = vmap(evaluate_discretization_lhs_rhs_at_node)
        lhs_rhs = evaluate_on_nodes_fn(nodes)
        # lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        # def loss_fn(node):
        #     lhs, rhs = evaluate_discretization_lhs_rhs_at_node(node)
        #     jnp.linalg.norm()
        return lhs_rhs

    

    @jit
    def compute_Ax(x):
        lhs_rhs = compute_Ax_and_b_fn(x)
        lhs, _ = jnp.split(lhs_rhs, [1], axis=1)
        return lhs

    @jit
    def compute_b(x):
        lhs_rhs = compute_Ax_and_b_fn(x)
        _, rhs = jnp.split(lhs_rhs, [1], axis=1)
        return rhs


    x = jnp.ones(phi_n.shape[0], dtype=f32)


    
    @jit
    def compute_residual(x):
        lhs_rhs = compute_Ax_and_b_fn(x)
        # lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        # return lhs - rhs
        return jnp.square(lhs_rhs[:,0] - lhs_rhs[:,1]).mean()



    # # Exponential decay of the learning rate.
    # scheduler = optax.exponential_decay(
    #     init_value=1e-2, 
    #     transition_steps=1000,
    #     decay_rate=0.99)

    # # Combining gradient transforms using `optax.chain`.
    # gradient_transform = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
    #     optax.scale_by_adam(),  # Use the updates from adam.
    #     optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    #     # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    #     optax.scale(-1.0)
    # )
    # optimizer = gradient_transform
    
    learning_rate = 1e-1
    optimizer = optax.adam(learning_rate)
    params = {'u' : x}
    opt_state = optimizer.init(params)
    
    compute_loss = lambda params: jit(compute_residual)(params['u'])

    grad_fn = jit(grad(compute_loss))

    loss_store = []
    for _ in range(50):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        loss_ = compute_loss(params)
        loss_store.append(loss_.tolist())
        print(f"iteration {_} loss = {loss_}")
    
    pdb.set_trace()
    return params['u']
    
    
    

    # lhs_rhs = compute_Ax_and_b_fn(x)
    # Amat = lhs_rhs[:,0].reshape((xo.shape+yo.shape+zo.shape))
    # rhs = lhs_rhs[:,1].reshape((xo.shape+yo.shape+zo.shape))
    # pdb.set_trace()
    # sol = gmres(compute_Ax, rhs)

    
   

   
    

    
