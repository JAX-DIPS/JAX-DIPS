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

from jax import jit, lax
from jax import numpy as jnp
from jax import vmap

from jax_dips._jaxmd_modules import util
from jax_dips.domain.interpolate import add_ghost_layer_3d

# Typing

f32 = util.f32
i32 = util.i32


def perturb_level_set_fn(phi_fn):
    EPS = 1.0e-10

    @jit
    def sign_pm_fn(a):
        sgn = jnp.sign(a)
        return jnp.sign(sgn - 0.5)

    @jit
    def perturbed_phi_fn(x):
        lvl = phi_fn(x)
        lvl_ = lvl + sign_pm_fn(lvl) * EPS
        return lvl_

    return perturbed_phi_fn


@jit
def smooth_3D_cube(c_cube):
    c_cube = c_cube[None, :, :, :, None]

    kernel = jnp.ones((3, 3, 3), dtype=f32)
    kernel /= kernel.sum()
    kernel = kernel[:, :, :, jnp.newaxis, jnp.newaxis]

    dn = lax.conv_dimension_numbers(c_cube.shape, kernel.shape, ("NHWDC", "HWDIO", "NHWDC"))
    smoothed_c = lax.conv_general_dilated(
        c_cube,  # lhs = image tensor
        kernel,  # rhs = conv kernel tensor
        (1, 1, 1),  # window strides
        "SAME",  # padding mode
        (1, 1, 1),  # lhs/image dilation
        (1, 1, 1),  # rhs/kernel dilation
        dn,
    )
    return smoothed_c[0, :, :, :, 0]


@jit
def smooth_phi_n(phi_n, gstate):
    xo = gstate.x
    yo = gstate.y
    zo = gstate.z
    c_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    x, y, z, c_cube = add_ghost_layer_3d(x, y, z, c_cube)
    c_cube = smooth_3D_cube(c_cube)
    return c_cube[2:-2, 2:-2, 2:-2].reshape(phi_n.shape)


def get_normal_vec_mean_curvature(phi_n, gstate):
    EPS = f32(1e-6)
    xo = gstate.x
    yo = gstate.y
    zo = gstate.z
    c_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    x, y, z, c_cube = add_ghost_layer_3d(x, y, z, c_cube)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    # c_cube = smooth_3D_cube(c_cube) # PAM

    Nx = gstate.x.shape[0]
    Ny = gstate.y.shape[0]
    Nz = gstate.z.shape[0]
    ii = jnp.arange(2, Nx + 2)
    jj = jnp.arange(2, Ny + 2)
    kk = jnp.arange(2, Nz + 2)
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing="ij")
    nodes = jnp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1)))

    @jit
    def normal_vec_fn(node):
        i, j, k = node
        phi_x = (c_cube[i + 1, j, k] - c_cube[i - 1, j, k]) / (f32(2) * dx)
        phi_y = (c_cube[i, j + 1, k] - c_cube[i, j - 1, k]) / (f32(2) * dy)
        phi_z = (c_cube[i, j, k + 1] - c_cube[i, j, k - 1]) / (f32(2) * dz)
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    @jit
    def mean_curvature_fn(node):
        i, j, k = node
        phi_xx = (c_cube[i + 1, j, k] - 2 * c_cube[i, j, k] + c_cube[i - 1, j, k]) / dx / dx
        phi_yy = (c_cube[i, j + 1, k] - 2 * c_cube[i, j, k] + c_cube[i, j - 1, k]) / dy / dy
        phi_zz = (c_cube[i, j, k + 1] - 2 * c_cube[i, j, k] + c_cube[i, j, k - 1]) / dz / dz

        phi_x = (c_cube[i + 1, j, k] - c_cube[i - 1, j, k]) / (f32(2) * dx)
        phi_y = (c_cube[i, j + 1, k] - c_cube[i, j - 1, k]) / (f32(2) * dy)
        phi_z = (c_cube[i, j, k + 1] - c_cube[i, j, k - 1]) / (f32(2) * dz)
        norm_squared = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z

        phi_xy = (
            c_cube[i + 1, j + 1, k] - c_cube[i + 1, j - 1, k] - c_cube[i - 1, j + 1, k] + c_cube[i - 1, j - 1, k]
        ) / (f32(4.0) * dx * dy)
        phi_xz = (
            c_cube[i + 1, j, k + 1] - c_cube[i + 1, j, k - 1] - c_cube[i - 1, j, k + 1] + c_cube[i - 1, j, k - 1]
        ) / (f32(4.0) * dx * dz)
        phi_yz = (
            c_cube[i, j + 1, k + 1] - c_cube[i, j - 1, k + 1] - c_cube[i, j + 1, k - 1] + c_cube[i, j - 1, k - 1]
        ) / (f32(4.0) * dy * dz)

        kappa_Mean = (
            (phi_yy + phi_zz) * phi_x * phi_x
            + (phi_xx + phi_zz) * phi_y * phi_y
            + (phi_xx + phi_yy) * phi_z * phi_z
            - f32(2.0) * phi_x * phi_y * phi_xy
            - f32(2.0) * phi_x * phi_z * phi_xz
            - f32(2.0) * phi_y * phi_z * phi_yz
        )
        kappa_Mean = kappa_Mean / norm_squared**1.5

        return kappa_Mean

    normal_vecs = vmap(normal_vec_fn)(nodes)
    curvatures = vmap(mean_curvature_fn)(nodes)
    return normal_vecs, curvatures


def get_normal_vec_mean_curvature_4th_order(phi_n, gstate):
    EPS = f32(1e-6)
    xo = gstate.x
    yo = gstate.y
    zo = gstate.z
    c_cube_ = phi_n.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    x, y, z, c_cube = add_ghost_layer_3d(x, y, z, c_cube)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    Nx = gstate.x.shape[0]
    Ny = gstate.y.shape[0]
    Nz = gstate.z.shape[0]
    ii = jnp.arange(2, Nx + 2)
    jj = jnp.arange(2, Ny + 2)
    kk = jnp.arange(2, Nz + 2)
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing="ij")
    nodes = jnp.column_stack((I.reshape(-1), J.reshape(-1), K.reshape(-1)))

    @jit
    def normal_vec_fn(node):
        i, j, k = node
        phi_x = (
            -c_cube[i + 2, j, k] + f32(8) * c_cube[i + 1, j, k] - f32(8) * c_cube[i - 1, j, k] + c_cube[i - 2, j, k]
        ) / (f32(12) * dx)
        phi_y = (
            -c_cube[i, j + 2, k] + f32(8) * c_cube[i, j + 1, k] - f32(8) * c_cube[i, j - 1, k] + c_cube[i, j - 2, k]
        ) / (f32(12) * dy)
        phi_z = (
            -c_cube[i, j, k + 2] + f32(8) * c_cube[i, j, k + 1] - f32(8) * c_cube[i, j, k - 1] + c_cube[i, j, k - 2]
        ) / (f32(12) * dz)
        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    @jit
    def mean_curvature_fn(node):
        i, j, k = node
        phi_xx = (
            -c_cube[i + 2, j, k]
            + f32(16) * c_cube[i + 1, j, k]
            - f32(30) * c_cube[i, j, k]
            + f32(16) * c_cube[i - 1, j, k]
            - c_cube[i - 2, j, k]
        ) / (f32(12) * dx * dx)
        phi_yy = (
            -c_cube[i, j + 2, k]
            + f32(16) * c_cube[i, j + 1, k]
            - f32(30) * c_cube[i, j, k]
            + f32(16) * c_cube[i, j - 1, k]
            - c_cube[i, j - 2, k]
        ) / (f32(12) * dy * dy)
        phi_zz = (
            -c_cube[i, j, k + 2]
            + f32(16) * c_cube[i, j, k + 1]
            - f32(30) * c_cube[i, j, k]
            + f32(16) * c_cube[i, j, k - 1]
            - c_cube[i, j, k - 2]
        ) / (f32(12) * dz * dz)

        phi_x = (
            -c_cube[i + 2, j, k] + f32(8) * c_cube[i + 1, j, k] - f32(8) * c_cube[i - 1, j, k] + c_cube[i - 2, j, k]
        ) / (f32(12) * dx)
        phi_y = (
            -c_cube[i, j + 2, k] + f32(8) * c_cube[i, j + 1, k] - f32(8) * c_cube[i, j - 1, k] + c_cube[i, j - 2, k]
        ) / (f32(12) * dy)
        phi_z = (
            -c_cube[i, j, k + 2] + f32(8) * c_cube[i, j, k + 1] - f32(8) * c_cube[i, j, k - 1] + c_cube[i, j, k - 2]
        ) / (f32(12) * dz)
        norm_squared = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z

        phi_xy = (
            -c_cube[i + 2, j + 2, k]
            + f32(16) * c_cube[i + 1, j + 1, k]
            + c_cube[i - 2, j + 2, k]
            - f32(16) * c_cube[i - 1, j + 1, k]
            + c_cube[i + 2, j - 2, k]
            - f32(16) * c_cube[i + 1, j - 1, k]
            - c_cube[i - 2, j - 2, k]
            + f32(16) * c_cube[i - 1, j - 1, k]
        ) / (f32(48.0) * dx * dy)

        phi_xz = (
            -c_cube[i + 2, j, k + 2]
            + f32(16) * c_cube[i + 1, j, k + 1]
            + c_cube[i - 2, j, k + 2]
            - f32(16) * c_cube[i - 1, j, k + 1]
            + c_cube[i + 2, j, k - 2]
            - f32(16) * c_cube[i + 1, j, k - 1]
            - c_cube[i - 2, j, k - 2]
            + f32(16) * c_cube[i - 1, j, k - 1]
        ) / (f32(48.0) * dx * dz)

        phi_yz = (
            -c_cube[i, j + 2, k + 2]
            + f32(16) * c_cube[i, j + 1, k + 1]
            + c_cube[i, j - 2, k + 2]
            - f32(16) * c_cube[i, j - 1, k + 1]
            + c_cube[i, j + 2, k - 2]
            - f32(16) * c_cube[i, j + 1, k - 1]
            - c_cube[i, j - 2, k - 2]
            + f32(16) * c_cube[i, j - 1, k - 1]
        ) / (f32(48.0) * dy * dz)

        kappa_Mean = (
            (phi_yy + phi_zz) * phi_x * phi_x
            + (phi_xx + phi_zz) * phi_y * phi_y
            + (phi_xx + phi_yy) * phi_z * phi_z
            - f32(2.0) * phi_x * phi_y * phi_xy
            - f32(2.0) * phi_x * phi_z * phi_xz
            - f32(2.0) * phi_y * phi_z * phi_yz
        )
        kappa_Mean = kappa_Mean / norm_squared**1.5

        return kappa_Mean

    normal_vecs = vmap(normal_vec_fn)(nodes)
    curvatures = vmap(mean_curvature_fn)(nodes)
    return normal_vecs, curvatures
