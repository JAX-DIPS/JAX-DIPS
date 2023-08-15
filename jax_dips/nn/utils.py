from jax import lax
from jax import numpy as jnp

from jax_dips.domain.interpolate import which_cell_index, add_ghost_layer_3d
from jax_dips._jaxmd_modules.util import f32, i32


def nonoscillatory_quadratic_interpolation_per_point(c, xo, yo, zo):
    """
    Under development for semi-Lagrangian method
    Min & Gibou 2007: eqns (12, 13)
    This should be used for solution interpolation
    """
    c_cube_ = c.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    def find_lower_left_cell_idx(point):
        """
        find cell index (i,j,k) containing point
        """
        x_p, y_p, z_p = point
        i = i32((x_p - x[0]) / dx)
        j = i32((y_p - y[0]) / dy)
        k = i32((z_p - z[0]) / dz)
        i = lax.cond(i >= x.shape[0] - 1, lambda p: i32(x.shape[0] - 2), lambda p: i32(p), i)
        j = lax.cond(j >= y.shape[0] - 1, lambda p: i32(y.shape[0] - 2), lambda p: i32(p), j)
        k = lax.cond(k >= z.shape[0] - 1, lambda p: i32(z.shape[0] - 2), lambda p: i32(p), k)
        i = lax.cond(i <= 1, lambda p: i32(2), lambda p: p, i)
        j = lax.cond(j <= 1, lambda p: i32(2), lambda p: p, j)
        k = lax.cond(k <= 1, lambda p: i32(2), lambda p: p, k)
        return i, j, k

    def second_order_deriv(i, j, k, dd):
        dx, dy, dz = dd
        dxx = c_cube[i + 1, j, k] - 2 * c_cube[i, j, k] + c_cube[i - 1, j, k]  # / dx / dx
        dyy = c_cube[i, j + 1, k] - 2 * c_cube[i, j, k] + c_cube[i, j - 1, k]  # / dy / dy
        dzz = c_cube[i, j, k + 1] - 2 * c_cube[i, j, k] + c_cube[i, j, k - 1]  # / dz / dz
        return jnp.array([dxx, dyy, dzz])

    def single_cell_interp(point):
        """
        nonoscillatory quadratic interpolation
        """
        i, j, k = find_lower_left_cell_idx(point)

        c_111 = c_cube[i + 1, j + 1, k + 1]
        c_110 = c_cube[i + 1, j + 1, k]
        c_011 = c_cube[i, j + 1, k + 1]
        c_101 = c_cube[i + 1, j, k + 1]
        c_001 = c_cube[i, j, k + 1]
        c_010 = c_cube[i, j + 1, k]
        c_100 = c_cube[i + 1, j, k]
        c_000 = c_cube[i, j, k]

        x_p, y_p, z_p = point
        dx = x[i + 1] - x[i]
        dy = y[j + 1] - y[j]
        dz = z[k + 1] - z[k]
        x_d = (x_p - x[i]) / dx
        y_d = (y_p - y[j]) / dy
        z_d = (z_p - z[k]) / dz

        c_00 = c_000 * (f32(1.0) - x_d) + c_100 * x_d
        c_01 = c_001 * (f32(1.0) - x_d) + c_101 * x_d
        c_10 = c_010 * (f32(1.0) - x_d) + c_110 * x_d
        c_11 = c_011 * (f32(1.0) - x_d) + c_111 * x_d

        c_0 = c_00 * (f32(1.0) - y_d) + c_10 * y_d
        c_1 = c_01 * (f32(1.0) - y_d) + c_11 * y_d

        c = c_0 * (f32(1.0) - z_d) + c_1 * z_d

        d2x_000 = c_cube[i + 1, j, k] - 2 * c_cube[i, j, k] + c_cube[i - 1, j, k]
        d2y_000 = c_cube[i, j + 1, k] - 2 * c_cube[i, j, k] + c_cube[i, j - 1, k]
        d2z_000 = c_cube[i, j, k + 1] - 2 * c_cube[i, j, k] + c_cube[i, j, k - 1]

        d2x_100 = c_cube[i + 2, j, k] - 2 * c_cube[i + 1, j, k] + c_cube[i, j, k]
        d2y_100 = c_cube[i + 1, j + 1, k] - 2 * c_cube[i + 1, j, k] + c_cube[i + 1, j - 1, k]
        d2z_100 = c_cube[i + 1, j, k + 1] - 2 * c_cube[i + 1, j, k] + c_cube[i + 1, j, k - 1]

        d2x_010 = c_cube[i + 1, j + 1, k] - 2 * c_cube[i, j + 1, k] + c_cube[i - 1, j + 1, k]
        d2y_010 = c_cube[i, j + 2, k] - 2 * c_cube[i, j + 1, k] + c_cube[i, j, k]
        d2z_010 = c_cube[i, j + 1, k + 1] - 2 * c_cube[i, j + 1, k] + c_cube[i, j + 1, k - 1]

        d2x_001 = c_cube[i + 1, j, k + 1] - 2 * c_cube[i, j, k + 1] + c_cube[i - 1, j, k + 1]
        d2y_001 = c_cube[i, j + 1, k + 1] - 2 * c_cube[i, j, k + 1] + c_cube[i, j - 1, k + 1]
        d2z_001 = c_cube[i, j, k + 2] - 2 * c_cube[i, j, k + 1] + c_cube[i, j, k]

        d2x_101 = c_cube[i + 2, j, k + 1] - 2 * c_cube[i + 1, j, k + 1] + c_cube[i, j, k + 1]
        d2y_101 = c_cube[i + 1, j + 1, k + 1] - 2 * c_cube[i + 1, j, k + 1] + c_cube[i + 1, j - 1, k + 1]
        d2z_101 = c_cube[i + 1, j, k + 2] - 2 * c_cube[i + 1, j, k + 1] + c_cube[i + 1, j, k]

        d2x_011 = c_cube[i + 1, j + 1, k + 1] - 2 * c_cube[i, j + 1, k + 1] + c_cube[i - 1, j + 1, k + 1]
        d2y_011 = c_cube[i, j + 2, k + 1] - 2 * c_cube[i, j + 1, k + 1] + c_cube[i, j, k + 1]
        d2z_011 = c_cube[i, j + 1, k + 2] - 2 * c_cube[i, j + 1, k + 1] + c_cube[i, j + 1, k]

        d2x_110 = c_cube[i + 2, j + 1, k] - 2 * c_cube[i + 1, j + 1, k] + c_cube[i, j + 1, k]
        d2y_110 = c_cube[i + 1, j + 2, k] - 2 * c_cube[i + 1, j + 1, k] + c_cube[i + 1, j, k]
        d2z_110 = c_cube[i + 1, j + 1, k + 1] - 2 * c_cube[i + 1, j + 1, k] + c_cube[i + 1, j + 1, k - 1]

        d2x_111 = c_cube[i + 2, j + 1, k + 1] - 2 * c_cube[i + 1, j + 1, k + 1] + c_cube[i, j + 1, k + 1]
        d2y_111 = c_cube[i + 1, j + 2, k + 1] - 2 * c_cube[i + 1, j + 1, k + 1] + c_cube[i + 1, j, k + 1]
        d2z_111 = c_cube[i + 1, j + 1, k + 2] - 2 * c_cube[i + 1, j + 1, k + 1] + c_cube[i + 1, j + 1, k]

        d2c_dxx = jnp.min(
            jnp.array(
                [
                    jnp.abs(d2x_000),
                    jnp.abs(d2x_100),
                    jnp.abs(d2x_010),
                    jnp.abs(d2x_001),
                    jnp.abs(d2x_101),
                    jnp.abs(d2x_011),
                    jnp.abs(d2x_110),
                    jnp.abs(d2x_111),
                ]
            )
        )
        d2c_dyy = jnp.min(
            jnp.array(
                [
                    jnp.abs(d2y_000),
                    jnp.abs(d2y_100),
                    jnp.abs(d2y_010),
                    jnp.abs(d2y_001),
                    jnp.abs(d2y_101),
                    jnp.abs(d2y_011),
                    jnp.abs(d2y_110),
                    jnp.abs(d2y_111),
                ]
            )
        )
        d2c_dzz = jnp.min(
            jnp.array(
                [
                    jnp.abs(d2z_000),
                    jnp.abs(d2z_100),
                    jnp.abs(d2z_010),
                    jnp.abs(d2z_001),
                    jnp.abs(d2z_101),
                    jnp.abs(d2z_011),
                    jnp.abs(d2z_110),
                    jnp.abs(d2z_111),
                ]
            )
        )

        c = (
            c
            - d2c_dxx * f32(0.5) * x_d * (f32(1.0) - x_d)
            - d2c_dyy * f32(0.5) * y_d * (f32(1.0) - y_d)
            - d2c_dzz * f32(0.5) * z_d * (f32(1.0) - z_d)
        )

        return c

    return single_cell_interp


def trilinear_interpolation_per_point(c, xo, yo, zo):
    """
    After reshape: c_cube[x-axis, y-axis, z-axis]
    This object is periodic by default!

    After adding ghost layer all points should be inside the ghosted cube.
    This is equivalent to: add ghost layer around c_cube + extrapolate solutions linearly (u_m = 2*u_0 - u_p)
    + shift indices of which_cell_index plus one
    """
    c_cube_ = c.reshape((xo.shape[0], yo.shape[0], zo.shape[0]))
    x, y, z, c_cube = add_ghost_layer_3d(xo, yo, zo, c_cube_)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    def find_lower_left_cell_idx(point):
        """
        find cell index (i,j,k) containing point
        """
        x_p, y_p, z_p = point
        i = i32((x_p - x[0]) / dx)
        j = i32((y_p - y[0]) / dy)
        k = i32((z_p - z[0]) / dz)

        i = lax.cond(i >= x.shape[0] - 1, lambda p: i32(x.shape[0] - 2), lambda p: i32(p), i)
        j = lax.cond(j >= y.shape[0] - 1, lambda p: i32(y.shape[0] - 2), lambda p: i32(p), j)
        k = lax.cond(k >= z.shape[0] - 1, lambda p: i32(z.shape[0] - 2), lambda p: i32(p), k)

        i = lax.cond(i <= 1, lambda p: i32(2), lambda p: p, i)
        j = lax.cond(j <= 1, lambda p: i32(2), lambda p: p, j)
        k = lax.cond(k <= 1, lambda p: i32(2), lambda p: p, k)

        return i, j, k

    def single_cell_interp(point):
        """
        Trilinear interpolation
        """
        i, j, k = find_lower_left_cell_idx(point)

        c_111 = c_cube[i + 1, j + 1, k + 1]
        c_110 = c_cube[i + 1, j + 1, k]
        c_011 = c_cube[i, j + 1, k + 1]
        c_101 = c_cube[i + 1, j, k + 1]
        c_001 = c_cube[i, j, k + 1]
        c_010 = c_cube[i, j + 1, k]
        c_100 = c_cube[i + 1, j, k]
        c_000 = c_cube[i, j, k]

        x_p, y_p, z_p = point
        dx = x[i + 1] - x[i]
        dy = y[j + 1] - y[j]
        dz = z[k + 1] - z[k]
        x_d = (x_p - x[i]) / dx
        y_d = (y_p - y[j]) / dy
        z_d = (z_p - z[k]) / dz

        c_00 = c_000 * (f32(1.0) - x_d) + c_100 * x_d
        c_01 = c_001 * (f32(1.0) - x_d) + c_101 * x_d
        c_10 = c_010 * (f32(1.0) - x_d) + c_110 * x_d
        c_11 = c_011 * (f32(1.0) - x_d) + c_111 * x_d

        c_0 = c_00 * (f32(1.0) - y_d) + c_10 * y_d
        c_1 = c_01 * (f32(1.0) - y_d) + c_11 * y_d

        c = c_0 * (f32(1.0) - z_d) + c_1 * z_d

        return c

    return single_cell_interp
