from jax import (lax, 
                 jit,
                 vmap, 
                 numpy as jnp)
from src.interpolate import add_ghost_layer_3d
from src import util
import pdb

# Typing

f32 = util.f32
i32 = util.i32






def get_normal_vec_mean_curvature(phi_n, gstate):
    EPS = f32(1e-6)
    xo = gstate.x; yo = gstate.y; zo = gstate.z
    c_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    x, y, z, c_cube = add_ghost_layer_3d(x, y, z, c_cube)
    
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
    def normal_vec_fn(node):
        i, j, k = node
        phi_x = (c_cube[i+1, j , k  ] - c_cube[i-1,j  ,k  ]) / (f32(2) * dx)
        phi_y = (c_cube[i, j+1 , k  ] - c_cube[i  ,j-1,k  ]) / (f32(2) * dy)
        phi_z = (c_cube[i, j   , k+1] - c_cube[i  ,j  ,k-1]) / (f32(2) * dz) 
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    @jit
    def mean_curvature_fn(node):
        i, j, k = node
        phi_xx = (c_cube[i+1, j  , k  ] - 2*c_cube[i,j,k] + c_cube[i-1,j  ,k  ]) / dx / dx
        phi_yy = (c_cube[i  , j+1, k  ] - 2*c_cube[i,j,k] + c_cube[i  ,j-1,k  ]) / dy / dy
        phi_zz = (c_cube[i  , j  , k+1] - 2*c_cube[i,j,k] + c_cube[i  ,j  ,k-1]) / dz / dz

        phi_x = (c_cube[i+1, j , k  ] - c_cube[i-1,j  ,k  ]) / (f32(2) * dx)
        phi_y = (c_cube[i, j+1 , k  ] - c_cube[i  ,j-1,k  ]) / (f32(2) * dy)
        phi_z = (c_cube[i, j   , k+1] - c_cube[i  ,j  ,k-1]) / (f32(2) * dz) 
        norm_squared = (phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)

        phi_xy = (c_cube[i+1, j+1, k] - c_cube[i+1, j-1, k] - c_cube[i-1, j+1, k] + c_cube[i-1, j-1, k]) / (f32(4.0) * dx * dy)
        phi_xz = (c_cube[i+1, j, k+1] - c_cube[i+1, j, k-1] - c_cube[i-1, j, k+1] + c_cube[i-1, j, k-1]) / (f32(4.0) * dx * dz)
        phi_yz = (c_cube[i, j+1, k+1] - c_cube[i, j-1, k+1] - c_cube[i, j+1, k-1] + c_cube[i, j-1, k-1]) / (f32(4.0) * dy * dz)

        kappa_Mean = ( (phi_yy + phi_zz)*phi_x*phi_x + \
                       (phi_xx + phi_zz)*phi_y*phi_y + \
                       (phi_xx + phi_yy)*phi_z*phi_z - \
                        f32(2.0) * phi_x * phi_y * phi_xy -\
                        f32(2.0) * phi_x * phi_z * phi_xz - \
                        f32(2.0) * phi_y * phi_z * phi_yz) / norm_squared**1.5

        return kappa_Mean

    normal_vecs = vmap(normal_vec_fn)(nodes)
    curvatures = vmap(mean_curvature_fn)(nodes)
    return normal_vecs, curvatures

    