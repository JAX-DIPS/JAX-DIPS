from functools import partial

from jax import grad, jit, jvp, lax
from jax import numpy as jnp
from jax import vmap

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.geometry import level_set

COMPILE_BACKEND = "gpu"
custom_jit = partial(jit, backend=COMPILE_BACKEND)

dim = 3


#####################################################
#
#   Sphere Interface with Jump
#
#####################################################
def sphere():
    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.exp(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.5

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        """
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return y * y * jnp.log(x + 2.0) + 4.0

    @jit
    def mu_p_fn(r):
        """
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.exp(-1.0 * z)

    @jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)

    @jit
    def k_m_fn(r):
        """
        Linear term function in $\Omega^-$
        """
        return 0.0

    @jit
    def k_p_fn(r):
        """
        Linear term function in $\Omega^+$
        """
        return 0.0

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @jit
    def f_m_fn_(r):
        """
        Source function in $\Omega^-$
        """

        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p) * grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_m_fn(r) * (-1.0)

    @jit
    def f_p_fn_(r):
        """
        Source function in $\Omega^+$
        """

        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p) * grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_p_fn(r) * (-1.0)

    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return -1.0 * jnp.exp(z) * (y * y * jnp.log(x + 2) + 4)

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 2.0 * jnp.exp(-1.0 * z) * jnp.cos(x) * jnp.sin(y)

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )


#####################################################
#
#   Star Interface with Jump
#
#####################################################


def star():
    """Star interface with jump conditions"""

    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @custom_jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z)

    @custom_jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        yx3 = (y - x) / 3.0
        return (16.0 * yx3**5 - 20.0 * yx3**3 + 5.0 * yx3) * jnp.log(x + y + 3) * jnp.cos(z)

    @custom_jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @custom_jit
    def unperturbed_phi_fn(r):
        r"""
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        r0 = 0.483
        ri = 0.151
        re = 0.911
        n_1 = 3.0
        beta_1 = 0.1
        theta_1 = 0.5
        n_2 = 4.0
        beta_2 = -0.1
        theta_2 = 1.8
        n_3 = 7.0
        beta_3 = 0.15
        theta_3 = 0.0

        core = beta_1 * jnp.cos(n_1 * (jnp.arctan2(y, x) - theta_1))
        core += beta_2 * jnp.cos(n_2 * (jnp.arctan2(y, x) - theta_2))
        core += beta_3 * jnp.cos(n_3 * (jnp.arctan2(y, x) - theta_3))

        phi_ = jnp.sqrt(x**2 + y**2 + z**2)
        phi_ += -1.0 * r0 * (1.0 + ((x**2 + y**2) / (x**2 + y**2 + z**2)) ** 2 * core)

        return jnp.nan_to_num(phi_, -r0 * core)

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @custom_jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @custom_jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 10.0 * (1 + 0.2 * jnp.cos(2 * jnp.pi * (x + y)) * jnp.sin(2 * jnp.pi * (x - y)) * jnp.cos(z))

    @custom_jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @custom_jit
    def alpha_fn(r):
        r"""
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @custom_jit
    def beta_fn(r):
        r"""
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.nan_to_num(jnp.dot(vec_1 - vec_2, n_vec) * (-1.0))

    @custom_jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.0

    @custom_jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.0

    @custom_jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @custom_jit
    def f_m_fn_(r):
        r"""
        Source function in $\Omega^-$
        """

        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p) * grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_m_fn(r) * (-1.0)

    @custom_jit
    def f_p_fn_(r):
        r"""
        Source function in $\Omega^+$
        """

        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p) * grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_p_fn(r) * (-1.0)

    @custom_jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        fm = (
            -1.0 * mu_m_fn(r) * (-7.0 * jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z))
            + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * x) * 2 * jnp.cos(2 * x) * jnp.cos(2 * y) * jnp.exp(z)
            + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * y) * (-2) * jnp.sin(2 * x) * jnp.sin(2 * y) * jnp.exp(z)
            + 2
            * jnp.cos(2 * jnp.pi * (x + y))
            * jnp.sin(2 * jnp.pi * (x - y))
            * jnp.sin(z)
            * jnp.sin(2 * x)
            * jnp.cos(2 * y)
            * jnp.exp(z)
        )

        return fm

    @custom_jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        f_p = -1.0 * (
            (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * (y - x) / 3)
            * (-2)
            * jnp.cos(z)
            / (x + y + 3) ** 2
            + 2
            * (16 * 5 * 4 * (1.0 / 9.0) * ((y - x) / 3) ** 3 - 20 * 3 * 2 * (1.0 / 9.0) * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
            + -1
            * (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
        )
        return f_p

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )


#####################################################
#
#   No interface jump
#
#####################################################
def no_jump():
    """No interface jump"""

    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        r"""
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + y**2 + z**2) + 0.5

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @jit
    def alpha_fn(r):
        r"""
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @jit
    def beta_fn(r):
        r"""
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec)

    @jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.0

    @jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.0

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return y
        # return exact_sol_p_fn(r)   # PAM: testing

    @jit
    def f_m_fn(r):
        r"""
        Source function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # 2.0 * jnp.sin(y) * jnp.cos(x)

    @jit
    def f_p_fn(r):
        r"""
        Source function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2.0 * jnp.sin(y) * jnp.cos(x)

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )
