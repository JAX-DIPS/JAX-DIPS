# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for constructing various interpolating functions.

This code was adapted from the way learning rate schedules are are built in JAX.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax._src.api import vmap
from jax._src.dtypes import dtype

import jax.numpy as np
from jax import ops, lax, jit
from numpy import where
from scipy.interpolate import splrep, PPoly
from scipy.interpolate import RegularGridInterpolator
from src import util
import pdb

# Typing

f32 = util.f32
f64 = util.f64
i32 = util.i32

#


def constant(f):
    def schedule(unused_t):
        return f
    return schedule


def canonicalize(scalar_or_schedule_fun):
    if callable(scalar_or_schedule_fun):
        return scalar_or_schedule_fun
    elif np.ndim(scalar_or_schedule_fun) == 0:
        return constant(scalar_or_schedule_fun)
    else:
        raise TypeError(type(scalar_or_schedule_fun))


def spline(y, dx, degree=3):
    """Spline fit a given scalar function.

    Args:
      y: The values of the scalar function evaluated on points starting at zero
      with the interval dx.
      dx: The interval at which the scalar function is evaluated.
      degree: Polynomial degree of the spline fit.

    Returns:
      A function that computes the spline function.
    """
    num_points = len(y)
    dx = f32(dx)
    x = np.arange(num_points, dtype=f32) * dx
    # Create a spline fit using the scipy function.
    # Turn off smoothing by setting s to zero.
    fn = splrep(x, y, s=0, k=degree)
    params = PPoly.from_spline(fn)
    # Store the coefficients of the spline fit to an array.
    coeffs = np.array(params.c)

    def spline_fn(x):
        """Evaluates the spline fit for values of x."""
        ind = np.array(x / dx, dtype=np.int64)
        # The spline is defined for x values between 0 and largest value of y. If x
        # is outside this domain, truncate its ind value to within the domain.
        truncated_ind = np.array(
            np.where(ind < num_points, ind, num_points - 1), np.int64)
        truncated_ind = np.array(
            np.where(truncated_ind >= 0, truncated_ind, 0), np.int64)
        result = np.array(0, x.dtype)
        dX = x - np.array(ind, np.float32) * dx
        # sum over the polynomial terms up to degree.
        for i in range(degree + 1):
            result = result + np.array(coeffs[degree - i, truncated_ind + 2],
                                       x.dtype) * dX ** np.array(i, x.dtype)
        # For x values that are outside the domain of the spline fit, return zeros.
        result = np.where(ind < num_points, result, np.array(0.0, x.dtype))
        return result
    return spline_fn



def godunov_hamiltonian(phi_n, gstate):
    """
    Godunov Hamiltonian given in equation 15 of Min & Gibou 2007
    """
    xo = gstate.x; yo = gstate.y; zo = gstate.z
    c_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]

    Dp_x = ( c_cube[2:  , 1:-1, 1:-1] - c_cube[1:-1, 1:-1, 1:-1] ) / dx
    Dm_x = ( c_cube[1:-1, 1:-1, 1:-1] - c_cube[ :-2, 1:-1, 1:-1] ) / dx

    Dp_y = ( c_cube[1:-1, 2:  , 1:-1] - c_cube[1:-1, 1:-1, 1:-1] ) / dy
    Dm_y = ( c_cube[1:-1, 1:-1, 1:-1] - c_cube[1:-1,  :-2, 1:-1] ) / dy

    Dp_z = ( c_cube[1:-1, 1:-1, 2:  ] - c_cube[1:-1, 1:-1, 1:-1] ) / dz
    Dm_z = ( c_cube[1:-1, 1:-1, 1:-1] - c_cube[1:-1, 1:-1,  :-2] ) / dz
    pdb.set_trace()





def nonoscillatory_quadratic_interpolation(c, gstate):
    """
    Under development for semi-Lagrangian method
    Min & Gibou 2007: eqns (12, 13)
    This should be used for solution interpolation
    """
    xo = gstate.x; yo = gstate.y; zo = gstate.z
    c_cube_ = c.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    cubex = np.zeros((8,3), dtype=i32)

    def find_lower_left_cell_idx(point):
        """
        find cell index (i,j,k) containing point
        """
        x_p, y_p, z_p = point
        i = which_cell_index(np.asarray(x_p >= x))
        j = which_cell_index(np.asarray(y_p >= y))
        k = which_cell_index(np.asarray(z_p >= z))
        return i, j, k

    @jit
    def second_order_deriv(i, j, k, dd):
        dx, dy, dz = dd
        dxx = (c_cube[i+1, j  , k  ] - 2*c_cube[i,j,k] + c_cube[i-1,j  ,k  ]) #/ dx / dx
        dyy = (c_cube[i  , j+1, k  ] - 2*c_cube[i,j,k] + c_cube[i  ,j-1,k  ]) #/ dy / dy
        dzz = (c_cube[i  , j  , k+1] - 2*c_cube[i,j,k] + c_cube[i  ,j  ,k-1]) #/ dz / dz
        return np.array([dxx, dyy, dzz])


    def single_cell_interp(point):
        """
        nonoscillatory quadratic interpolation
        """
        i,j,k = find_lower_left_cell_idx(point)
        
        c_111 = c_cube[i+1, j+1, k+1]
        c_110 = c_cube[i+1, j+1, k  ]
        c_011 = c_cube[i  , j+1, k+1]
        c_101 = c_cube[i+1, j  , k+1]
        c_001 = c_cube[i  , j  , k+1]
        c_010 = c_cube[i  , j+1, k  ]
        c_100 = c_cube[i+1, j  , k  ]
        c_000 = c_cube[i  , j  , k  ]

        x_p, y_p, z_p = point
        dx = x[i+1] - x[i]
        dy = y[j+1] - y[j]
        dz = z[k+1] - z[k]
        x_d = (x_p - x[i]) / dx
        y_d = (y_p - y[j]) / dy
        z_d = (z_p - z[k]) / dz

        c_00 = c_000 * (f32(1.0) - x_d) + c_100 * x_d
        c_01 = c_001 * (f32(1.0) - x_d) + c_101 * x_d
        c_10 = c_010 * (f32(1.0) - x_d) + c_110 * x_d
        c_11 = c_011 * (f32(1.0) - x_d) + c_111 * x_d

        c_0  = c_00  * (f32(1.0) - y_d) + c_10  * y_d
        c_1  = c_01  * (f32(1.0) - y_d) + c_11  * y_d

        c    = c_0   * (f32(1.0) - z_d) + c_1   * z_d

        # correcting for second derivatives:
        dd = (dx, dy, dz)
        
        d2x_000, d2y_000, d2z_000 = second_order_deriv(i  , j  , k  , dd)
        d2x_100, d2y_100, d2z_100 = second_order_deriv(i+1, j  , k  , dd)
        d2x_010, d2y_010, d2z_010 = second_order_deriv(i  , j+1, k  , dd)
        d2x_001, d2y_001, d2z_001 = second_order_deriv(i  , j  , k+1, dd)

        d2x_101, d2y_101, d2z_101 = second_order_deriv(i+1, j  , k+1, dd)
        d2x_011, d2y_011, d2z_011 = second_order_deriv(i  , j+1, k+1, dd)
        d2x_110, d2y_110, d2z_110 = second_order_deriv(i+1, j+1, k  , dd)
        d2x_111, d2y_111, d2z_111 = second_order_deriv(i+1, j+1, k+1, dd)
        
        d2c_dxx = np.min(np.array([abs(d2x_000), abs(d2x_100), abs(d2x_010), abs(d2x_001), abs(d2x_101), abs(d2x_011), abs(d2x_110), abs(d2x_111)]))
        d2c_dyy = np.min(np.array([abs(d2y_000), abs(d2y_100), abs(d2y_010), abs(d2y_001), abs(d2y_101), abs(d2y_011), abs(d2y_110), abs(d2y_111)]))
        d2c_dzz = np.min(np.array([abs(d2z_000), abs(d2z_100), abs(d2z_010), abs(d2z_001), abs(d2z_101), abs(d2z_011), abs(d2z_110), abs(d2z_111)]))

        c  = c - d2c_dxx * f32(0.5) * x_d * (f32(1.0) - x_d) - d2c_dyy * f32(0.5) * y_d * (f32(1.0) - y_d) - d2c_dzz * f32(0.5) * z_d * (f32(1.0) - z_d) 

        return c


    def interp_fn(R_star):
        """
        interpolate on all provided points
        """
        return vmap(jit(single_cell_interp))(R_star)

    return interp_fn







def which_cell_index(cond): 
    """A USEFULE utility function to find the index of True in a list, 
    for example cond = 0.1 < gstate.x provides a list of True's and False's
    and thie function returns the first time True appears
    """
    cond = np.asarray(cond)
    return (np.argwhere(~cond, size=1) - 1).flatten()






def add_ghost_layer_3d(x, y, z, c_cube):
    """
    add ghost layer around c_cube + extrapolate solutions linearly (u_m = 2*u_0 - u_p)
    """
    dx_l = x[1] - x[0]; dx_r = x[-1] - x[-2]
    x_layer_l = 2 * c_cube[ 0,:,:] - c_cube[ 1,:,:]; x_layer_l = np.expand_dims(x_layer_l, axis=0)
    x_layer_r = 2 * c_cube[-1,:,:] - c_cube[-2,:,:]; x_layer_r = np.expand_dims(x_layer_r, axis=0)
    c_cube_gh = np.concatenate((x_layer_l, c_cube, x_layer_r), axis=0)
    xx = np.concatenate((np.array([x[0] - dx_l]) , x, np.array([x[-1] + dx_r])))

    dy_b = y[1] - y[0]; dy_t = y[-1] - y[-2]
    y_layer_b = 2 * c_cube_gh[:, 0,:] - c_cube_gh[:, 1,:]; y_layer_b = np.expand_dims(y_layer_b, axis=1)
    y_layer_t = 2 * c_cube_gh[:,-1,:] - c_cube_gh[:,-2,:]; y_layer_t = np.expand_dims(y_layer_t, axis=1)
    c_cube_gh = np.concatenate((y_layer_b, c_cube_gh, y_layer_t), axis=1)
    yy = np.concatenate((np.array([y[0] - dy_b]) , y, np.array([y[-1] + dy_t])))

    dz_b = z[1] - z[0]; dz_t = z[-1] - z[-2]
    z_layer_b = 2 * c_cube_gh[:,:, 0] - c_cube_gh[:,:, 1]; z_layer_b = np.expand_dims(z_layer_b, axis=2)
    z_layer_t = 2 * c_cube_gh[:,:,-1] - c_cube_gh[:,:,-2]; z_layer_t = np.expand_dims(z_layer_t, axis=2)
    c_cube_gh = np.concatenate((z_layer_b, c_cube_gh, z_layer_t), axis=2)
    zz = np.concatenate((np.array([z[0] - dz_b]) , z, np.array([z[-1] + dz_t])))
    return xx, yy, zz, c_cube_gh





def update_ghost_layer_3d(c_cube):
    c_cube.at[ 0,:,:].set(2 * c_cube[ 1,:,:] - c_cube[ 2,:,:])
    c_cube.at[-1,:,:].set(2 * c_cube[-2,:,:] - c_cube[-3,:,:])

    c_cube.at[:, 0,:].set(2 * c_cube[:, 1,:] - c_cube[:, 2,:])
    c_cube.at[:,-1,:].set(2 * c_cube[:,-2,:] - c_cube[:,-3,:])

    c_cube.at[:,:, 0].set(2 * c_cube[:,:, 1] - c_cube[:,:, 2])
    c_cube.at[:,:,-1].set(2 * c_cube[:,:,-2] - c_cube[:,:,-3])






def multilinear_interpolation(c, gstate):
    """
    Under development for semi-Lagrangian method
    Min & Gibou 2007: eqns (11)
    This is to be used for velocity interpolation

    After reshape: c_cube[x-axis, y-axis, z-axis]
    This object is periodic by default!

    After adding ghost layer all points should be inside the ghosted cube.
    This is equivalent to: add ghost layer around c_cube + extrapolate solutions linearly (u_m = 2*u_0 - u_p)
    + shift indices of which_cell_index plus one
    """
    xo = gstate.x; yo = gstate.y; zo = gstate.z
    c_cube_ = c.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    # c_cube_ = np.swapaxes(c_cube_, 0, 1)
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)

    def find_lower_left_cell_idx(point):
        """
        find cell index (i,j,k) containing point
        """
        x_p, y_p, z_p = point
        i = which_cell_index(np.asarray(x_p >= x))
        j = which_cell_index(np.asarray(y_p >= y))
        k = which_cell_index(np.asarray(z_p >= z))
        return i, j, k

    def single_cell_interp(point):
        """
        Trilinear interpolation
        """
        i,j,k = find_lower_left_cell_idx(point)
        
        c_111 = c_cube[i+1, j+1, k+1]
        c_110 = c_cube[i+1, j+1, k  ]
        c_011 = c_cube[i  , j+1, k+1]
        c_101 = c_cube[i+1, j  , k+1]
        c_001 = c_cube[i  , j  , k+1]
        c_010 = c_cube[i  , j+1, k  ]
        c_100 = c_cube[i+1, j  , k  ]
        c_000 = c_cube[i  , j  , k  ]

        x_p, y_p, z_p = point
        dx = x[i+1] - x[i]
        dy = y[j+1] - y[j]
        dz = z[k+1] - z[k]
        x_d = (x_p - x[i]) / dx
        y_d = (y_p - y[j]) / dy
        z_d = (z_p - z[k]) / dz

        c_00 = c_000 * (f32(1.0) - x_d) + c_100 * x_d
        c_01 = c_001 * (f32(1.0) - x_d) + c_101 * x_d
        c_10 = c_010 * (f32(1.0) - x_d) + c_110 * x_d
        c_11 = c_011 * (f32(1.0) - x_d) + c_111 * x_d

        c_0  = c_00  * (f32(1.0) - y_d) + c_10  * y_d
        c_1  = c_01  * (f32(1.0) - y_d) + c_11  * y_d

        c    = c_0   * (f32(1.0) - z_d) + c_1   * z_d

        return c


    def interp_fn(R_star):
        """
        interpolate on all provided points
        """
        # update_ghost_layer_3d(c_cube)
        return vmap(jit(single_cell_interp))(R_star)

    return interp_fn



def vec_multilinear_interpolation(Vec, gstate):
    vx = Vec[:,0]; vy = Vec[:,1]; vz = Vec[:,2]

    def interp_fn(R_):
        vx_interp_fn = multilinear_interpolation(vx, gstate)
        vy_interp_fn = multilinear_interpolation(vy, gstate)
        vz_interp_fn = multilinear_interpolation(vz, gstate)
        xvals = vx_interp_fn(R_)
        yvals = vy_interp_fn(R_)
        zvals = vz_interp_fn(R_)
        return np.column_stack((xvals, yvals, zvals))
    
    return interp_fn