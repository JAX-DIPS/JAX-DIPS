
from typing import Callable, TypeVar, Union, Tuple, Dict, Optional

import numpy as np
# import jax.numpy as np

from src.jaxmd_modules.util import Array
from src.jaxmd_modules import dataclasses

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
    dx: float
    dy: float
    dz: float
    R_xmin_boundary: Array
    R_xmax_boundary: Array 
    R_ymin_boundary: Array 
    R_ymax_boundary: Array 
    R_zmin_boundary: Array 
    R_zmax_boundary: Array

    def shape(self):
        return self.x.shape + self.y.shape + self.z.shape

    def xmin(self): return self.x[0]
    def xmax(self): return self.x[-1]
    def ymin(self): return self.y[0]
    def ymax(self): return self.y[-1]
    def zmin(self): return self.z[0]
    def zmax(self): return self.z[-1]

    



def construct(dimension : int) -> Mesher:

    def init_fn_2d(x, y):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X_, Y_ = np.meshgrid(x, y, indexing='ij')
        X = X_.flatten(); Y = Y_.flatten()
        R = np.column_stack((X, Y))
        R_xmin_boundary = np.column_stack((X_[0,:].flatten(), Y_[0,:].flatten()))
        R_xmax_boundary = np.column_stack((X_[-1,:].flatten(), Y_[-1,:].flatten()))

        R_ymin_boundary = np.column_stack((X_[:,0].flatten(), Y_[:,0].flatten()))
        R_ymax_boundary = np.column_stack((X_[:,-1].flatten(), Y_[:,-1].flatten()))
        return GridState(x, y, None, R, dx, dy, None, R_xmin_boundary, R_xmax_boundary, R_ymin_boundary, R_ymax_boundary, None, None)

    def init_fn_3d(x, y, z):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        X_, Y_, Z_ = np.meshgrid(x, y, z, indexing='ij')
        X = X_.flatten(); Y = Y_.flatten(); Z = Z_.flatten()
        R = np.column_stack((X, Y, Z))
        R_xmin_boundary = np.column_stack((X_[0,:,:].flatten(), Y_[0,:,:].flatten(), Z_[0,:,:].flatten()))
        R_xmax_boundary = np.column_stack((X_[-1,:,:].flatten(), Y_[-1,:,:].flatten(), Z_[-1,:,:].flatten()))

        R_ymin_boundary = np.column_stack((X_[:,0,:].flatten(), Y_[:,0,:].flatten(), Z_[:,0,:].flatten()))
        R_ymax_boundary = np.column_stack((X_[:,-1,:].flatten(), Y_[:,-1,:].flatten(), Z_[:,-1,:].flatten()))

        R_zmin_boundary = np.column_stack((X_[:,:,0].flatten(), Y_[:,:,0].flatten(), Z_[:,:,0].flatten()))
        R_zmax_boundary = np.column_stack((X_[:,:,-1].flatten(), Y_[:,:,-1].flatten(), Z_[:,:,-1].flatten()))

        return GridState(x, y, z, R, dx, dy, dz, R_xmin_boundary, R_xmax_boundary, R_ymin_boundary, R_ymax_boundary, R_zmin_boundary, R_zmax_boundary)

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



