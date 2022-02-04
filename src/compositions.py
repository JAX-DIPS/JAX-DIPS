import jax
from jax import (jit, random, lax, ops, vmap, grad, numpy as jnp)
from src.util import f32, i32
from functools import partial



def node_curvature_fn(f):
    def _lapl_over_f(x):
        n = x.shape[0]
        eye = jnp.eye(n, dtype=f32)
        grad_f = jax.grad(f)
        grad_f_closure = lambda y: grad_f(y) / jnp.linalg.norm(grad_f(y))
        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[i]**2 + tangent[i]
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)
    return _lapl_over_f


def vec_curvature_fn(phi_fn):
    return jax.jit(jax.vmap(node_curvature_fn(phi_fn)))
#---------

def node_laplacian_fn(f):
    def _lapl_over_f(x):
        n = x.shape[0]
        eye = jnp.eye(n, dtype=f32)
        grad_f = jax.grad(f)
        grad_f_closure = lambda y: grad_f(y)
        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[i]**2 + tangent[i]
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)
    return _lapl_over_f


def vec_laplacian_fn(phi_fn):
    return jax.jit(jax.vmap(node_laplacian_fn(phi_fn)))


#--------

def node_advect_step_autodiff(f, vel_fn):
    def advect_one_fn(dt, x):
        return f(x) - dt * jnp.dot(grad(f)(x), vel_fn(x))
    return advect_one_fn

def vec_advect_one_step_autodiff(f, vel_fn):
    def advect_one_fn(dt, x):
        return f(x) - dt * jnp.dot(grad(f)(x), vel_fn(x))
    return vmap(advect_one_fn, (None, 0))

