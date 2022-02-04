import jax
from jax import (jit, random, lax, ops, grad, numpy as jnp)
from src.util import f32, i32

def get_laplacian_fn(f):
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
    return jax.jit(jax.vmap(get_laplacian_fn(phi_fn)))
     