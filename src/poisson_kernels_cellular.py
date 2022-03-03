from jax import (numpy as jnp, vmap, jit)
from src import (interpolate, util)
import pdb

f32 = util.f32
i32 = util.i32

    


def poisson_solver(gstate, sim_state):
    phi_n = sim_state.phi
    u_n = sim_state.solution

    xo = gstate.x; yo = gstate.y; zo = gstate.z

    phi_cube = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, phi_cube)
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
    
    pdb.set_trace()
