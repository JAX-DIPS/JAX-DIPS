
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
class GridState(object):
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

    def shape(self):
        return self.x.shape + self.y.shape + self.z.shape



def construct(dimension : int) -> Mesher:

    def init_fn_2d(x, y):
        X, Y = jnp.meshgrid(x, y)
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

    """
    Ghost layers are populated with values of last layer (=constant extrapolation)
    
    def add_ghost_layer_3d(gstate, sstate):
        x = gstate.x; y = gstate.y; z = gstate.z; c= sstate.solution
        c_cube = c.reshape((x.shape[0], y.shape[0], z.shape[0]))

        x_layer_l = c_cube[0,:,:]; x_layer_l = jnp.expand_dims(x_layer_l, axis=0)
        x_layer_r = c_cube[-1,:,:]; x_layer_r = jnp.expand_dims(x_layer_r, axis=0)
        c_cube_gh = jnp.concatenate((x_layer_l, c_cube, x_layer_r), axis=0)
        xx = jnp.concatenate((jnp.array([x[0]]) , x, jnp.array([x[-1]])))

        y_layer_b = c_cube_gh[:,0,:]; y_layer_b = jnp.expand_dims(y_layer_b, axis=1)
        y_layer_t = c_cube_gh[:,-1,:]; y_layer_t = jnp.expand_dims(y_layer_t, axis=1)
        c_cube_gh = jnp.concatenate((y_layer_b, c_cube_gh, y_layer_t), axis=1)
        yy = jnp.concatenate((jnp.array([y[0]]) , y, jnp.array([y[-1]])))

        z_layer_b = c_cube_gh[:,:,0]; z_layer_b = jnp.expand_dims(z_layer_b, axis=2)
        z_layer_t = c_cube_gh[:,:,-1]; z_layer_t = jnp.expand_dims(z_layer_t, axis=2)
        c_cube_gh = jnp.concatenate((z_layer_b, c_cube_gh, z_layer_t), axis=2)
        zz = jnp.concatenate((jnp.array([z[0]]) , z, jnp.array([z[-1]])))

        gstate = init_fn_3d(xx, yy, zz)
 
        cc = c_cube_gh.reshape((-1,1))
        sstate = dataclasses.replace(sstate, solution=cc)
        return gstate, sstate
 


    def add_ghost_layer_2d(gstate, sstate):
        x = gstate.x; y = gstate.y; c= sstate.solution
        c_cube = c.reshape((x.shape[0], y.shape[0]))

        x_layer_l = c_cube[0,:]; x_layer_l = jnp.expand_dims(x_layer_l, axis=0)
        x_layer_r = c_cube[-1,:]; x_layer_r = jnp.expand_dims(x_layer_r, axis=0)
        c_cube_gh = jnp.concatenate((x_layer_l, c_cube, x_layer_r), axis=0)
        xx = jnp.concatenate((jnp.array([x[0]]) , x, jnp.array([x[-1]])))

        y_layer_b = c_cube_gh[:,0]; y_layer_b = jnp.expand_dims(y_layer_b, axis=1)
        y_layer_t = c_cube_gh[:,-1]; y_layer_t = jnp.expand_dims(y_layer_t, axis=1)
        c_cube_gh = jnp.concatenate((y_layer_b, c_cube_gh, y_layer_t), axis=1)
        yy = jnp.concatenate((jnp.array([y[0]]) , y, jnp.array([y[-1]])))
        
        gstate = init_fn_2d(xx, yy)

        cc = c_cube_gh.reshape((-1,1))
        dataclasses.replace(sstate, solution=cc)
        return gstate, sstate

    """

    if dimension == 3:
        init_fn = init_fn_3d
        point_fn = point3d_at
        # add_ghost_layer_fn = add_ghost_layer_3d
    elif dimension == 2:
        init_fn = init_fn_2d
        point_fn = point2d_at
        # add_ghost_layer_fn = add_ghost_layer_2d

    return init_fn, point_fn #, add_ghost_layer_fn



