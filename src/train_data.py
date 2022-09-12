
from jax import (numpy as jnp, random, jit)
from functools import partial
import numpy as onp

from src.jaxmd_modules.util import f32, i32
from src import mesh
import pdb




class DatasetDict:
    def __init__(self,
                 x_dict,
                 batch_size=64):
        self.batch_size = batch_size
        self.x_dict = x_dict
        self._len = None
        self._counter = 0
        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_dict)
        return self
    
    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        data_x = self.x_dict[self._idx: min(self._len, self._idx + self.batch_size)]
        self._idx += self.batch_size
        self._counter += 1
        return data_x



class TrainData:
        def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, Nx=64, Ny=64, Nz=64) -> None:
            """ Training GSTATE """
            xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
            yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
            zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
            init_mesh_fn, coord_at = mesh.construct(3)
            self.gstate = init_mesh_fn(xc, yc, zc)
            
            self.key = random.PRNGKey(0)
            Lx = (self.gstate.xmax() - self.gstate.xmin())
            Ly = (self.gstate.ymax() - self.gstate.ymin())
            Lz = (self.gstate.zmax() - self.gstate.zmin())
            self.LL = jnp.array([[Lx, Ly, Lz]])
            
            self.alt_res = False
            self.base_level = int(onp.log2(Nx))
            
        @partial(jit, static_argnums=(0))
        def move_train_points(self, points, dx, dy, dz):
            cov = jnp.array([[dx, 0.0, 0.0], [0.0, dy, 0.0], [0.0, 0.0, dz]])*0.5
            mean = jnp.array([0.0,0.0, 0.0])
            Rnew = points + random.multivariate_normal(self.key, mean, cov, shape=(len(points),))    
            new_points = Rnew - jnp.floor(Rnew / self.LL ) * self.LL - 0.5*self.LL
            return new_points
        
        @staticmethod
        def batch(points, batch_size):
            ds_iter = DatasetDict(points, batch_size=batch_size)
            return ds_iter        
        
        @staticmethod
        def alternate_res(self, epoch, train_dx, train_dy, train_dz):
            self.alt_res = True
            if epoch % 4==0:
                train_dx = self.gstate.dx 
                train_dy = self.gstate.dy 
                train_dz = self.gstate.dz 
            else:
                train_dx *= 0.50
                train_dy *= 0.50
                train_dz *= 0.50
                
            return train_dx, train_dy, train_dz
            