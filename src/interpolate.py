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


def nonoscillatory_quadratic_interpolation(U, R):
    """
    Under development for semi-Lagrangian method
    Min & Gibou 2007: eqns (12, 13)
    This should be used for solution interpolation
    """
    def interp_fn(R_star):
        result = 0
        return result

    return interp_fn



#  A USEFULE utility function to find the index of True in a list, 
# for example cond = 0.1 < gstate.x provides a list of True's and False's
# and thie function returns the first time True appears
def which_cell_index(cond): 
    cond = np.asarray(cond)
    return (np.argwhere(~cond, size=1) - 1).flatten()


def multilinear_interpolation(c, gstate):
    """
    Under development for semi-Lagrangian method
    Min & Gibou 2007: eqns (11)
    This is to be used for velocity interpolation
    """
    x = gstate.x; y = gstate.y; z = gstate.z
    c_cube = c.reshape((x.shape[0], y.shape[0], z.shape[0]))

    """
    After reshape: c_cube[x-axis, y-axis, z-axis]
    This object is periodic by default!
    """

    def find_lower_left_cell_idx(point):
        #find cell index (i,j,k) containing point
        x_p, y_p, z_p = point
        i = which_cell_index(np.asarray(x_p >= x))
        j = which_cell_index(np.asarray(y_p >= y))
        k = which_cell_index(np.asarray(z_p >= z))
        return i, j, k

    def single_cell_interp(point):
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
        return vmap(jit(single_cell_interp))(R_star)
        # num_interp = len(R_star)
        # def step(i, log):
        #     point = R_star[i]
        #     c_p = single_cell_interp(point)
        #     log['val'] = ops.index_update(log['val'], i, c_p)
        #     return log

        # log = {'val' : np.zeros(num_interp,)}
        # log = lax.fori_loop(i32(0), i32(num_interp), step, (log,))

        # return log['val']

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
        return np.vstack((xvals, yvals, zvals))
    
    return interp_fn