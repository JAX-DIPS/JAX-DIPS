
from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

from jax.abstract_arrays import ShapedArray

from jax import eval_shape
from jax import vmap
from jax import custom_jvp

import jax

import jax.numpy as jnp

from src.util import Array
from src.util import f32
from src.util import f64
from src.util import safe_mask
from src import dataclasses

# Types

T = TypeVar('T')
InitFn = Callable[..., T]
PointFn = Callable[[T, Array], Array]
Mesher = Tuple[InitFn, PointFn]

# Primitive Spatial Transforms

@dataclasses.dataclass
class GridState:
    """A struct containing the state of the grid.
    
    Attributes:
    R: An ndarray of shape [n, spatial_dimension] storing the position
        of grid points.
    x,y,z: 1D arrays of shape (n,) storing the linspace positions along each dimension
    PHI: initial values of the level set function

    """
    x: Array
    y: Array
    z: Array
    R: Array



def construct(dimension : int) -> Mesher:

    def init_fn_2d(x, y):
        X, Y, Z = jnp.meshgrid(x, y)
        X = X.flatten(); Y = Y.flatten()
        R = jnp.column_stack((X, Y))
        return GridState(x, y, None, R)

    def init_fn_3d(x, y, z):
        X, Y, Z = jnp.meshgrid(x, y, z)
        X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()
        R = jnp.column_stack((X, Y, Z))
        return GridState(x, y, z, R)

    def point3d_at(gstate, idx):
        i = idx[0]; j = idx[1]; k = idx[2]
        point = [gstate.x[i], gstate.y[j], gstate.z[k]] 
        return point
    
    def point2d_at(gstate, idx):
        i = idx[0]; j = idx[1]
        point = [gstate.x[i], gstate.y[j]] 
        return point

    if dimension == 3:
        init_fn = init_fn_3d
        point_fn = point3d_at
    elif dimension == 2:
        init_fn = init_fn_2d
        point_fn = point2d_at

    return init_fn, point_fn



