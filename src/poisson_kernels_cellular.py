from jax import (numpy as jnp, vmap, jit)
from src import (interpolate, util)
import pdb

f32 = util.f32


    


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

    def get_X_ijk():
        return jnp.array([[-dx, -dy, -dz],
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


    def get_W_neighborhood(node, phi_cube):
        i, j, k = node
        pdb.set_trace()
        return 0.0


    @jit
    def node_update(node):
        i, j, k = node
        Xijk = get_X_ijk()
        dd = get_W_neighborhood(node, phi_cube)
        res = 0
        return jnp.nan_to_num(res)

    return vmap(node_update)(nodes)