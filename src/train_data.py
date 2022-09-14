
from jax import (numpy as jnp, random, jit)
from functools import partial
import numpy as onp

from src.jaxmd_modules.util import f32, i32
from src import mesh
import pdb



class DatasetDictMGPU:
    def __init__(self,
                 x_data,
                 batch_size=64,
                 drop_remainder: bool = False,
                 shuffle: bool = False):
        self.batch_size = batch_size
        self.x_data = x_data
        self._len = None
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        self._order = onp.arange(self._len)
        if self.shuffle:
            onp.random.shuffle(self._order)
        return self
    
    def __next__(self):
        max_idx = self._len
        if self.drop_remainder:
            max_idx -= self.batch_size

        if self._idx >= max_idx:
            raise StopIteration
      
        data_x = {}
        batch_idx = self._order[self._idx: min(self._len, self._idx + self.batch_size)]
        
        data_x = self.x_data[batch_idx]
        self._idx += self.batch_size
        return data_x



class DatasetDict_:
    def __init__(self,
                 x_data,
                 batch_size):
        
        self.x_data = x_data
        self._len = len(x_data)        
        self.batch_size = batch_size 
        self.num_batches = self._len // self.batch_size
        
        self.gpu_data = x_data.reshape((self.num_batches, self.batch_size,-1 ))
        self._batch_counter = 0

    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        return self
    
    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration        
        self._idx += self.batch_size
        self._batch_counter += 1
        return self.gpu_data[self._batch_counter - 1]
    
    

def generate_random_points_like(x_data, num_missing_points):
    key = random.PRNGKey(0)
    cov = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mean = jnp.array([0.0,0.0, 0.0])
    dR = random.multivariate_normal(key, mean, cov, shape=(num_missing_points,))
    Rnew = (x_data.min() + (x_data.max() - x_data.min()) * (dR - dR.min()) / (dR.max() - dR.min()))
    return Rnew


class DatasetDict:
    def __init__(self,
                 x_data,
                 batch_size,
                 num_gpus = 1):
        self.num_gpus = num_gpus
        self.x_data = x_data
        self._len = len(x_data)        
        self.batch_size = batch_size 
        
        if self.batch_size > self._len:
            self.batch_size = self._len
            
        self.extra_batch = int(jnp.heaviside(self._len % self.batch_size, 0))
        self.num_batches = self._len // self.batch_size 
        self._batch_counter = 0
        
        self.gpu_padded_batches()
        if self.num_gpus > 1 : self.split_over_devices()
        
        
    def gpu_padded_batches(self):
        """
            Pads the data in case it cannot be just folded exactly.
            The padded values are randomly scattered points inside the domain.
            dx, dy, dz are used to ensure
        """
        self.last_batch_size = self._len % self.batch_size
        self.batched_data = jnp.zeros((self.num_batches + self.extra_batch , self.batch_size, 3))
        for batch_id in range(self.num_batches):
            self.batched_data = self.batched_data.at[batch_id].set(self.x_data[batch_id*self.batch_size: min(self._len, (batch_id+1)*self.batch_size)])        
        
        if self.extra_batch==1:
            self.batched_data = self.batched_data.at[self.num_batches, 0:self.last_batch_size,:].set(self.x_data[self.num_batches*self.batch_size: self.num_batches*self.batch_size + self.last_batch_size])
            num_missing_points = self.batch_size - self.last_batch_size
            Rnew = generate_random_points_like(self.x_data, num_missing_points)
            self.batched_data = self.batched_data.at[self.num_batches, -num_missing_points:,:].set(Rnew)       
        
   
    def get_batched_data(self):
        return self.batched_data
    
    
    def split_over_devices(self):
        import jax
        def split(arr):
            """
                Splits the second axis of `self.batched_data` evenly across the number of devices.
                first axis is the mini-batch dimension, second is the gpu device, third & fourth are the points.
            """
            return arr.reshape( arr.shape[0] // self.num_gpus, self.num_gpus, *arr.shape[1:])
        
        self.batched_data = jax.tree_map(split, self.batched_data)
        
        
        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        return self
    
    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        self._idx += self.batch_size
        self._batch_counter += 1
        return self.batched_data[self._batch_counter - 1]
    



class TrainData:
        def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, Nx=64, Ny=64, Nz=64) -> None:
            """ Training GSTATE """
            self.Nx = Nx; self.Ny = Ny; self.Nz = Nz
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
            
            self.boundary_points = jnp.concatenate(( self.gstate.R_xmin_boundary, self.gstate.R_xmax_boundary, self.gstate.R_ymin_boundary, self.gstate.R_ymax_boundary, self.gstate.R_zmin_boundary, self.gstate.R_zmax_boundary ))
            
        @partial(jit, static_argnums=(0))
        def move_train_points(self, points, dx, dy, dz):
            cov = jnp.array([[dx, 0.0, 0.0], [0.0, dy, 0.0], [0.0, 0.0, dz]])*0.5
            mean = jnp.array([0.0,0.0, 0.0])
            Rnew = points + random.multivariate_normal(self.key, mean, cov, shape=(len(points),))    
            new_points = Rnew - jnp.floor(Rnew / self.LL ) * self.LL - 0.5*self.LL
            return new_points
        
        
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
            