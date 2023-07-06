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
from functools import partial

import jax
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import ops, random, vmap

from jax_dips._jaxmd_modules.util import f32, i32


def node_normal_fn(f):
    def _normal_fn(x):
        grad_f = jax.grad(f)
        grad_f_closure = lambda y: grad_f(y) / jnp.linalg.norm(grad_f(y))
        return grad_f_closure(x)

    return _normal_fn


def vec_normal_fn(phi_n):
    return jax.jit(jax.vmap(node_normal_fn(phi_n)))


# ---------


def node_curvature_fn(f):
    def _lapl_over_f_v1(x):
        n = x.shape[0]
        eye = jnp.eye(n, dtype=f32)
        grad_f = jax.grad(f)
        grad_f_closure = lambda y: grad_f(y) / jnp.linalg.norm(grad_f(y))

        def _body_fun(i, val):
            primal_out, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + tangent[i]

        return lax.fori_loop(0, n, _body_fun, 0.0)

    def _lapl_over_f_v2(x):
        r_hessian_real = jax.hessian(lambda r_in: f(r_in), argnums=0, holomorphic=False)(x)
        return (jnp.diag(r_hessian_real)).sum()

    return _lapl_over_f_v1


def vec_curvature_fn(phi_fn):
    return jax.jit(jax.vmap(node_curvature_fn(phi_fn)))


# ---------


def node_laplacian_fn(f):
    def _lapl_over_f(x):
        n = x.shape[0]
        eye = jnp.eye(n, dtype=f32)
        grad_f = jax.grad(f)
        grad_f_closure = lambda y: grad_f(y)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[i] ** 2 + tangent[i]

        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


def vec_laplacian_fn(phi_fn):
    return jax.jit(jax.vmap(node_laplacian_fn(phi_fn)))


# --------
def advect_one_step_autodiff(f, vec_vel_fn):
    def advect_one_fn(dt, x):
        return vmap(f)(x) - dt * vmap(jnp.dot, (0, 0))(vmap(grad(f))(x), vec_vel_fn(x))

    return advect_one_fn


def node_advect_one_step_autodiff(f, vel_fn):
    def advect_one_fn(dt, x):
        return f(x) - dt * jnp.dot(grad(f)(x), vel_fn(x))

    return advect_one_fn


def vec_advect_one_step_autodiff(f, vel_fn):
    def advect_one_fn(dt, x):
        return f(x) - dt * jnp.dot(grad(f)(x), vel_fn(x))

    return vmap(advect_one_fn, (None, 0))
