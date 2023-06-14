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


from typing import Callable, Dict, Optional, Tuple, TypeVar, Union

import jax.numpy as jnp
from jax import grad, jit, lax, vmap

from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.domain import interpolate
from jax_dips.geometry import level_set as ls
from jax_dips.solvers.simulation_states import AdvectionSimState

static_cast = util.static_cast

Array = util.Array
i32 = util.i32
f32 = util.f32

T = TypeVar("T")
InitFn = Callable[..., T]
ApplyFn = Callable[[T, T], T]
ReinitializeFn = Callable[[T, T], T]
ReinitializedAdvectFn = Callable[[T, T], T]
Simulator = Tuple[InitFn, ApplyFn, ReinitializeFn, ReinitializedAdvectFn]


def advect_level_set(gstate: T, V_nm1: Array, velocity_fn: Callable[..., Array], time: float):
    R = gstate.R
    V_n = vmap(velocity_fn, (0, None))(R, time)

    # Get interpolation functions
    Vn_interp_fn = interpolate.vec_multilinear_interpolation(V_n, gstate)
    Vnm1_interp_fn = interpolate.vec_multilinear_interpolation(V_nm1, gstate)

    def advect_one_step_at_node(point: Array, U_n: Array, dt: float) -> T:
        """Apply a single step of semi-Lagrangian integration to a state."""
        Un_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(U_n, gstate)
        dt = f32(dt)
        dt_2 = f32(dt / 2.0)

        # vel_nm1_point = velocity_fn(point, time - dt)
        vel_n_point = velocity_fn(point, time)

        # Find Departure Point
        r_star = point - dt_2 * vel_n_point
        r_star = r_star.reshape(1, -1)

        vel_n_star_point = Vn_interp_fn(r_star)
        vel_nm1_star_point = Vnm1_interp_fn(r_star)
        v_mid = f32(1.5) * vel_n_star_point - f32(0.5) * vel_nm1_star_point

        point_d = point - dt * v_mid
        # substitute solution from departure point to arrival points (=grid points)
        u_np1_point = Un_interp_fn(point_d)  # .flatten()
        return u_np1_point.reshape()

    advect_semi_lagrangian_one_step_grid_fn = jit(vmap(advect_one_step_at_node, (0, None, None)))

    grad_advect_level_set_one_step_at_node_fn = jit(grad(advect_one_step_at_node))
    grad_semi_lagrangian_one_step_grid_fn = jit(vmap(grad_advect_level_set_one_step_at_node_fn, (0, None, None)))

    def reinitialized_level_set_point_fn(point: Array, U_n: Array, dt: float) -> T:
        dtau = 0.001

        # --- one semi Lagrangian step.
        phi_np1 = advect_one_step_at_node(point, U_n, dt)
        grad_phi_np1 = grad_advect_level_set_one_step_at_node_fn(point, U_n, dt)
        sign_phi_0 = jnp.sign(phi_np1)
        # ---

        def phi_tilde_np1_point(
            point: Array,
            U_n: Array,
            phi_np1: Array,
            sign_phi_0: float,
            dt: float,
            dtau: float,
        ):
            # phi_np1 = advect_one_step_at_node(point, U_n, dt)
            grad_phi_np1 = grad_advect_level_set_one_step_at_node_fn(point, U_n, dt)
            norm_grad_phi_np1 = jnp.linalg.norm(grad_phi_np1)
            phi_t_np1 = phi_np1 - dtau * sign_phi_0 * (norm_grad_phi_np1 - 1.0)
            return phi_t_np1

        grad_phi_tilde_np1_point_fn = jit(grad(phi_tilde_np1_point))

        # phi_t_np1 = phi_tilde_np1_point(point, U_n, sign_phi_0, dt, dtau)
        # grad_phi_t_np1 = grad_phi_tilde_np1_point_fn(point, U_n, sign_phi_0, dt, dtau)
        # norm_grad_phi_t_np1 = jnp.linalg.norm(grad_phi_t_np1)

        # phi_t_np2 = phi_t_np1 - dtau * sign_phi_0 * (norm_grad_phi_t_np1 - 1.0)
        # phi_final = 0.5 * (phi_np1 + phi_t_np2)
        # ----
        phi_t_np1 = phi_tilde_np1_point(point, U_n, phi_np1, sign_phi_0, dt, dtau)
        grad_phi_t_np1 = grad_phi_tilde_np1_point_fn(point, U_n, phi_np1, sign_phi_0, dt, dtau)
        norm_grad_phi_t_np1 = jnp.linalg.norm(grad_phi_t_np1)

        phi_t_np2 = phi_t_np1 - dtau * sign_phi_0 * (norm_grad_phi_t_np1 - 1.0)
        phi_np1_ = 0.5 * (phi_np1 + phi_t_np2)

        return phi_np1_

    # def reinitialized_level_set_point_fn(point: Array, U_n: Array, dt: float) -> T:
    #     dtau = f32(0.5 * dt)
    #     dt = f32(dt)

    #     #--- one semi Lagrangian step.
    #     phi_np1 = advect_one_step_at_node(point, U_n, dt)
    #     grad_phi_np1 = grad_advect_level_set_one_step_at_node_fn(point, U_n, dt)
    #     sign_phi_0 = jnp.sign(phi_np1)
    #     #---
    #     def Sussman_RK2_step(params, i):
    #         phi_np1, grad_phi_np1, sign_phi_0, dtau = params

    #         def phi_tilde_np1_point_second_step(phi_np1: Array, grad_phi_np1: Array, sign_phi_0: float, dtau: float):
    #             def phi_tilde_np1_point_first_step( phi_np1: Array, grad_phi_np1: Array, sign_phi_0: float, dtau: float):
    #                 norm_grad_phi_np1 = jnp.linalg.norm(grad_phi_np1)
    #                 phi_t_np1 = phi_np1 - dtau * sign_phi_0 * (norm_grad_phi_np1 - f32(1.0))
    #                 return phi_t_np1
    #             grad_phi_tilde_np1_point_first_step_fn = jit(grad(phi_tilde_np1_point_first_step))
    #             phi_t_np1 = phi_tilde_np1_point_first_step(phi_np1, grad_phi_np1, sign_phi_0, dtau)
    #             grad_phi_t_np1 = grad_phi_tilde_np1_point_first_step_fn(phi_np1, grad_phi_np1, sign_phi_0, dtau)

    #             grad_phi_t_np1 *= grad_phi_np1

    #             norm_grad_phi_t_np1 = jnp.linalg.norm(grad_phi_t_np1)
    #             phi_t_np2 = phi_t_np1 - dtau * sign_phi_0 * (norm_grad_phi_t_np1 - f32(1.0))
    #             return f32(0.5) * (phi_np1 + phi_t_np2)

    #         grad_phi_tilde_np1_point_second_step_fn = jit(grad(phi_tilde_np1_point_second_step))
    #         phi_np1 = phi_tilde_np1_point_second_step(phi_np1, grad_phi_np1, sign_phi_0, dtau)
    #         grad_phi_np1_ = grad_phi_tilde_np1_point_second_step_fn(phi_np1, grad_phi_np1, sign_phi_0, dtau)

    #         grad_phi_np1 *= grad_phi_np1_

    #         return (phi_np1, grad_phi_np1, sign_phi_0, dtau), None

    #     # phi_np1, grad_phi_np1, sign_phi_0, dtau = Sussman_RK2_step(0, (phi_np1, grad_phi_np1, sign_phi_0, dtau))
    #     iters = jnp.arange(0, 1)
    #     (phi_np1, grad_phi_np1, sign_phi_0, dtau), _ = lax.scan(Sussman_RK2_step, (phi_np1, grad_phi_np1, sign_phi_0, dtau), iters)

    #     return phi_np1

    reinitialized_level_set_grid_fn = jit(vmap(reinitialized_level_set_point_fn, (0, None, None)))
    grad_reinitialized_level_set_grid_fn = jit(vmap(grad(reinitialized_level_set_point_fn), (0, None, None)))

    return (
        advect_semi_lagrangian_one_step_grid_fn,
        grad_semi_lagrangian_one_step_grid_fn,
        reinitialized_level_set_grid_fn,
        grad_reinitialized_level_set_grid_fn,
    )


def advect_one_step(velocity_fn: Callable[..., Array], dt: float, sstate: T, gstate: T, time: float, **kwargs) -> T:
    """Apply a single step of semi-Lagrangian integration to a state."""

    dt = f32(dt)
    dt_2 = f32(dt / 2.0)

    R, U_n, V_nm1 = gstate.R, sstate.phi, sstate.velocity_nm1
    V_n = velocity_fn(R, time)

    # Get interpolation functions
    Vn_interp_fn = interpolate.vec_multilinear_interpolation(V_n, gstate)
    Vnm1_interp_fn = interpolate.vec_multilinear_interpolation(V_nm1, gstate)

    # Un_interp_fn = interpolate.multilinear_interpolation(U_n, gstate)
    Un_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(U_n, gstate)

    # Find Departure Point
    R_star = R - dt_2 * V_n
    V_n_star = Vn_interp_fn(R_star)
    V_nm1_star = Vnm1_interp_fn(R_star)
    V_mid = f32(1.5) * V_n_star - f32(0.5) * V_nm1_star
    R_d = R - dt * V_mid
    # substitute solution from departure point to arrival points (=grid points)
    U_np1 = Un_interp_fn(R_d).flatten()

    # perturb
    # EPS = 1e-9
    # @vmap
    # def perturb(u):
    #     return lax.cond(jnp.sign(u)==0, lambda p: p + EPS, lambda p: p + EPS * jnp.sign(p), u)
    # U_np1 = perturb(U_np1)

    return dataclasses.replace(sstate, phi=U_np1, velocity_nm1=V_n)


def reinitialize_level_set(sstate: T, gstate: T, **kwargs) -> T:
    """
    Sussman's reinitialization of the level set function
    to retain its signed distance nature.

    $ \partial_\tau \phi + sgn(\phi^0)(|\nabla \phi| - 1) = 0 $
    where $\tau$ represents a fictitious time.

    This function should be called every few iterations to maintain
    the level set function.
    """
    # x = gstate.x; y = gstate.y; z = gstate.z
    # dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]

    phi_n = sstate.phi
    sgn_0 = jnp.sign(phi_n)

    def step_phi_fn(i, sgn_phi_n):
        sgn_0, phi_n_ = sgn_phi_n
        hg_n = interpolate.godunov_hamiltonian(
            phi_n_, sgn_0, gstate
        )  # this function pre-multiplies by proper dt, subtracts -1, multiplies by sign of phi_ijk
        phi_t_np1 = phi_n_ - hg_n  # jnp.multiply(sgn_0, hg_n)
        hg_np1 = interpolate.godunov_hamiltonian(phi_t_np1, sgn_0, gstate)
        phi_t_np2 = phi_t_np1 - hg_np1  # jnp.multiply(sgn_0, hg_np1)
        phi_n_ = f32(0.5) * (phi_n_ + phi_t_np2)
        return sgn_0, phi_n_

    (sgn_0, phi_n) = lax.fori_loop(i32(0), i32(10), step_phi_fn, (sgn_0, phi_n))
    ls.smooth_phi_n(phi_n, gstate)
    return dataclasses.replace(sstate, phi=phi_n)


# def level_set(velocity_or_energy_fn: Callable[..., Array],
def level_set(level_set_fn: Callable[..., Array], dt: float) -> Simulator:
    """
    Simulates a system.

    Args:
    velocity_fn: A function that produces the velocity field on
        a set of grid points specified as an ndarray of shape
        [n, spatial_dimension].
        velocity_fn = -grad(Energy_fn)


    dt: Floating point number specifying the timescale (step size) of the
        simulation.

    Returns:
    See above.
    """

    # velocity_fn = vmap(velocity_or_energy_fn, (0,None))
    phi_fn = vmap(level_set_fn)

    def init_fn(velocity_fn, R, **kwargs):
        # V = jnp.zeros(R.shape, dtype=R.dtype)
        # U = jnp.zeros(R.shape[0], dtype=R.dtype)
        # U = U + space.square_distance(R) - f32(0.25)

        V = velocity_fn(R, 0.0)  # ,**kwargs)
        U = phi_fn(R)
        return AdvectionSimState(U, V)

    def apply_fn(velocity_fn, sim_state, grid_state, time, **kwargs):
        return advect_one_step(velocity_fn, dt, sim_state, grid_state, time, **kwargs)

    def reinitialize_fn(sim_state, grid_state, **kwargs):
        return reinitialize_level_set(sim_state, grid_state, **kwargs)

    def reinitialized_advect_fn(velocity_fn, sim_state, grid_state, time, **kwargs):
        _, _, reinitialized_fn, _ = advect_level_set(grid_state, sim_state.velocity_nm1, velocity_fn, time)

        def wrapper(func):
            def wrapped(sim_state, grid_state, time):
                U_np1 = func(grid_state.R, sim_state.phi, dt)
                V_n = velocity_fn(grid_state.R, time)
                return dataclasses.replace(sim_state, phi=U_np1, velocity_nm1=V_n)

            return wrapped

        return wrapper(reinitialized_fn)(sim_state, grid_state, time)

    return init_fn, apply_fn, reinitialize_fn, reinitialized_advect_fn
