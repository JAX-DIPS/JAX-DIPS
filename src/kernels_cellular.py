from jax import (numpy as jnp, grad, vmap)
import pdb



def cell_geometrics(node):
    i, j, k = node
    