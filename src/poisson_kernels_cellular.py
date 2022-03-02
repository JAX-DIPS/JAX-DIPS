from jax import (numpy as jnp, vmap, jit)
from src import (interpolate, util)
import pdb

f32 = util.f32

def cell_geometrics(node):
    i, j, k = node
    


def poisson_solver(gstate, sim_state):
    phi_n = sim_state.phi
    xo = gstate.x; yo = gstate.y; zo = gstate.z

    c_cube = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, c_cube)
    x, y, z, c_cube = interpolate.add_ghost_layer_3d(x, y, z, c_cube)
    
    dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]

    Nx = gstate.x.shape[0]
    Ny = gstate.y.shape[0]
    Nz = gstate.z.shape[0]

    ii = jnp.arange(2, Nx+2)
    jj = jnp.arange(2, Ny+2)
    kk = jnp.arange(2, Nz+2)
    
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing='ij')
    
    nodes = jnp.column_stack( (I.reshape(-1), J.reshape(-1), K.reshape(-1) ))

    def find_cell_idx(node):
        """
        find cell index (i,j,k) containing point
        """
        i, j, k = node
        return i, j, k

    @jit
    def node_update(node):
        i,j,k = find_cell_idx(node)
        dd = cell_geometrics(i, j, k)
        res = 0
        return jnp.nan_to_num(res)

    return vmap(node_update)(nodes)