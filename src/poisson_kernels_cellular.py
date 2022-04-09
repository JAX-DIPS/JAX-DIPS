from functools import partial
from jax import (numpy as jnp, vmap, jit, lax, ops)
from src import (interpolate, util)
import pdb

f32 = util.f32
i32 = util.i32

    


def poisson_solver(gstate, sim_state):
    phi_n = sim_state.phi
    u_n = sim_state.solution
    mu_m = sim_state.mu_m
    mu_p = sim_state.mu_p

    xo = gstate.x; yo = gstate.y; zo = gstate.z
    
    phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(phi_n, gstate)

    phi_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, phi_cube_)
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)
    

    u_cube = u_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x_, y_, z_, u_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, u_cube)
    _, _, _, u_cube = interpolate.add_ghost_layer_3d(x_, y_, z_, u_cube)


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



    mu_m_cube = mu_m.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    mu_p_cube = mu_p.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    
    zeta_p_ijk_pqm = ((mu_p_cube - mu_m_cube) / mu_m_cube) * phi_cube_ 
    zeta_p_ijk_pqm = zeta_p_ijk_pqm.reshape(zeta_p_ijk_pqm.shape + (-1,)) * Cp_ijk_pqm.reshape(phi_cube_.shape + (-1,))
    
    zeta_m_ijk_pqm = ((mu_p_cube - mu_m_cube) / mu_p_cube) * phi_cube_ 
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


    
    @jit
    def get_vertices_of_cell_intersection_with_interface_at_node(node):
        """
        Based on Min & Gibou 2007: Geometric integration over irregular domains
        with application to level-set methods
        """
        i, j, k = node
        # Get corners of the control volume    
        dXcorners = 0.5 * jnp.array([   [-dx, -dy, -dz],
                                        [ dx, -dy, -dz],
                                        [ dx, -dy,  dz],
                                        [-dx, -dy,  dz],
                                        [-dx,  dy, -dz],
                                        [ dx,  dy, -dz],
                                        [-dx,  dy,  dz],
                                        [ dx,  dy,  dz] ], dtype=f32)
        
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

        
        
        def get_vertices_S_intersect_Gamma(S, phi_S, eta_S):
            """
            This function returns the vertices splitted by the level set.
            The intersection of the mesh-cell simplex S crossed by the level-set function. 
            """
            zeros_gamma = jnp.array([ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                                  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]], dtype=f32)

            
            def eta_1_fn(arg):
                S, phi_S, zeros_gamma = arg
                S_sorted = S[jnp.argsort(phi_S)]
                phi_S_sorted = jnp.sort(phi_S)
                Q_0 =  ( phi_S_sorted[0] * S_sorted[1] - phi_S_sorted[1] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[1])
                Q_1 =  ( phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[2])
                Q_2 =  ( phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[3])
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,0], Q_0)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,1], Q_1)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,2], Q_2)
                # return zeros_gamma
                return jnp.array([[Q_0, Q_1, Q_2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ], dtype=f32)
            
            def eta_3_fn(arg):
                S, phi_S, zeros_gamma = arg
                return eta_1_fn((S, -1.0 * phi_S, zeros_gamma))
            
            def eta_2_fn(arg):
                S, phi_S, zeros_gamma = arg
                S_sorted = S[jnp.argsort(phi_S)]
                phi_S_sorted = jnp.sort(phi_S)
                Q_0 =  ( phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[2])
                Q_1 =  ( phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[3])
                Q_2 =  ( phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1] ) / (phi_S_sorted[1] - phi_S_sorted[3])
                Q_5 =  ( phi_S_sorted[1] * S_sorted[2] - phi_S_sorted[2] * S_sorted[1] ) / (phi_S_sorted[1] - phi_S_sorted[2])
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,0], Q_0)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,1], Q_1)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[0,2], Q_2)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[1,0], Q_0)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[1,1], Q_5)
                # zeros_gamma = ops.index_update(zeros_gamma, jnp.index_exp[1,2], Q_2)
                # return zeros_gamma
                return jnp.array([[Q_0, Q_1, Q_2], [Q_0, Q_5, Q_2]], dtype=f32)
                
            
            def eta_2_3_fn(arg):
                # return lax.cond(eta_S - 2, eta_2_fn, eta_3_fn, arg)
                return jnp.where(eta_S==2, eta_2_fn(arg), eta_3_fn(arg))
        
            def eta_1_2_3_fn(arg):
                # return lax.cond(eta_S - 1, eta_1_fn, eta_2_3_fn, arg)
                return jnp.where(eta_S==1, eta_1_fn(arg), eta_2_3_fn(arg))
            
            def eta_0_4_fn(arg):
                S, phi_S, zeros_gamma = arg
                return zeros_gamma

            # shared_vertices = lax.cond(eta_S*(eta_S - 4), eta_0_4_fn, eta_1_2_3_fn, (S, phi_S, zeros_gamma))
            shared_vertices = jnp.where(eta_S*(eta_S - 4)==0, eta_0_4_fn((S, phi_S, zeros_gamma)), eta_1_2_3_fn((S, phi_S, zeros_gamma)))
            return shared_vertices

        
        def get_vertices_S_intersect_Omega_m(S, phi_S, eta_S: int):
            """
            This function returns the tetrahedra vertices splitted by the level set.
            The intersection of the mesh-cell volume crossed by the level-set function. 
            """            
            zeros_ = jnp.array([ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                             [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                             [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]], dtype=f32)

            ones_ = jnp.array([ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]], dtype=f32)
            
            def eta_1_fn(arg):
                S, phi_S, zeros_, ones_ = arg
                S_sorted = S[jnp.argsort(phi_S)]
                phi_S_sorted = jnp.sort(phi_S)
                Q_0 = S_sorted[0]
                Q_1 =  ( phi_S_sorted[0] * S_sorted[1] - phi_S_sorted[1] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[1])
                Q_2 =  ( phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[2])
                Q_3 =  ( phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[3])
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,0], Q_0)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,1], Q_1)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,3], Q_3)
                # return zeros_
                return jnp.array([[Q_0, Q_1, Q_2, Q_3],\
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],\
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ], dtype=f32)
            
            def eta_2_fn(arg):
                S, phi_S, zeros_, ones_ = arg
                S_sorted = S[jnp.argsort(phi_S)]
                phi_S_sorted = jnp.sort(phi_S)
                Q_0 = S_sorted[0]
                Q_1 = S_sorted[1]
                Q_2 =  ( phi_S_sorted[0] * S_sorted[2] - phi_S_sorted[2] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[2])
                Q_3 =  ( phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1] ) / (phi_S_sorted[1] - phi_S_sorted[3])
                Q_4 =  ( phi_S_sorted[1] * S_sorted[2] - phi_S_sorted[2] * S_sorted[1] ) / (phi_S_sorted[1] - phi_S_sorted[2])
                Q_5 =  ( phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[3])
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,0], Q_0)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,1], Q_1)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,3], Q_3)
                
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,0], Q_4)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,1], Q_1)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,3], Q_3)

                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,0], Q_0)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,1], Q_5)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,3], Q_3)
                
                # return zeros_
                return jnp.array([[Q_0, Q_1, Q_2, Q_3],\
                                [Q_4, Q_1, Q_2, Q_3],\
                                [Q_0, Q_5, Q_2, Q_3]], dtype=f32)
            
            def eta_3_fn(arg):
                S, phi_S, zeros_, ones_ = arg
                S_sorted = S[jnp.argsort(phi_S)]
                phi_S_sorted = jnp.sort(phi_S)
                Q_0 = S_sorted[0]
                Q_1 = S_sorted[1]
                Q_2 = S_sorted[2]
                Q_3 =  ( phi_S_sorted[1] * S_sorted[3] - phi_S_sorted[3] * S_sorted[1] ) / (phi_S_sorted[1] - phi_S_sorted[3])
                Q_4 =  ( phi_S_sorted[0] * S_sorted[3] - phi_S_sorted[3] * S_sorted[0] ) / (phi_S_sorted[0] - phi_S_sorted[3])
                Q_5 =  ( phi_S_sorted[2] * S_sorted[3] - phi_S_sorted[3] * S_sorted[2] ) / (phi_S_sorted[2] - phi_S_sorted[3])

                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,0], Q_0)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,1], Q_1)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[0,3], Q_3)
                
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,0], Q_0)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,1], Q_4)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[1,3], Q_3)

                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,0], Q_5)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,1], Q_4)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,2], Q_2)
                # zeros_ = ops.index_update(zeros_, jnp.index_exp[2,3], Q_3)
                # return zeros_
                return jnp.array([[Q_0, Q_1, Q_2, Q_3],\
                                [Q_0, Q_4, Q_2, Q_3],\
                                [Q_5, Q_4, Q_2, Q_3]], dtype=f32)
            
            def eta_2_3_fn(arg):
                return jnp.where(eta_S == 2, eta_2_fn(arg), eta_3_fn(arg))
                # return lax.cond(eta_S - 2, eta_2_fn, eta_3_fn, arg)
            
            def eta_1_2_3_fn(arg):
                return jnp.where(eta_S==1, eta_1_fn(arg), eta_2_3_fn(arg))
                # return lax.cond(eta_S - 1, eta_1_fn, eta_2_3_fn, arg)

            def eta_0_4_fn(arg):
                S, phi_S, zeros_, ones_ = arg
                ones_ = ops.index_update(ones_, jnp.index_exp[0,:], S)
                return jnp.where(eta_S==0, zeros_, ones_)
                # return lax.cond(eta_S, lambda p: zeros_, lambda p: ones_, S)

            # shared_vertices = lax.cond(eta_S * (eta_S - 4), eta_0_4_fn, eta_1_2_3_fn, (S, phi_S, zeros_, ones_))
            shared_vertices = jnp.where(eta_S * (eta_S - 4)==0, eta_0_4_fn((S, phi_S, zeros_, ones_)), eta_1_2_3_fn((S, phi_S, zeros_, ones_)))
            
            return shared_vertices

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

        return sv_gamma_s1, sv_gamma_s2, sv_gamma_s3, sv_gamma_s4, sv_gamma_s5,\
               sv_omega_s1, sv_omega_s2, sv_omega_s3, sv_omega_s4, sv_omega_s5

    pieces = get_vertices_of_cell_intersection_with_interface_at_node(nodes[0])
    # pieces = vmap(get_vertices_of_cell_intersection_with_interface_at_node)(nodes)
    pdb.set_trace()

    def A_matmul_x_fn(u):
        """
        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || u^k - b ||^2 

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
        # index_to_ijk = (nodes - jnp.array([2,2,2]))
        nodes 
        ngbs
        
        gamma_m_ijk
        gamma_p_ijk
        
        gamma_p_ijk_pqm
        gamma_m_ijk_pqm

        zeta_m_ijk
        zeta_p_ijk

        zeta_m_ijk_pqm
        zeta_p_ijk_pqm

        def u_p_coeffs_residual(node):
            """
            BIAS SLOW

            For a given node in
            
            u_p = B_p : u + r_p 
            
            this function evaluates B_p and r_p
            """
            i, j, k = node
            pdb.set_trace()
            
            B = 0
            r = 0
            return B, r

        def u_m_coeffs_residual(node):
            """
            BIAS SLOW

            For a given node in
            
            u_m = B_m : u + r_m 
            
            this function evaluates B_m and r_m
            """
            i, j, k = node
            
            B = 0
            r = 0
            return B, r
        
        B_p, R_p = vmap(u_p_coeffs_residual)(nodes)
        pdb.set_trace()

        
        return

    x = jnp.ones(phi_n.shape[0], dtype=f32)
    out = A_matmul_x_fn(x)

    return jit(A_matmul_x_fn)
    

    
