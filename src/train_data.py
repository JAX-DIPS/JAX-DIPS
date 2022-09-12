
from jax import (numpy as jnp, random, jit)
from functools import partial
import numpy as onp

from src.jaxmd_modules.util import f32, i32
from src import mesh
import pdb



class DatasetDictMGPU:
    def __init__(self,
                 x_dict,
                 batch_size=64,
                 drop_remainder: bool = False,
                 shuffle: bool = False):
        self.batch_size = batch_size
        self.x_dict = x_dict
        self._len = None
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_dict)
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
        
        data_x = self.x_dict[batch_idx]
        self._idx += self.batch_size
        return data_x



class DatasetDict:
    def __init__(self,
                 x_dict,
                 batch_size):
        
        self.x_dict = x_dict
        self._len = len(x_dict)
        
        common_divisor = int(onp.round(self._len**(1/3)))
        block_size = self._len // common_divisor
        
        self.batch_size = batch_size #block_size
        
        self.num_batches = self._len // self.batch_size
        self.gpu_data = x_dict.reshape((self.num_batches, self.batch_size,-1 ))
        self._batch_counter = 0
        
        # self.gpu_padded_batches()
        
        
    def gpu_padded_batches(self):
        """
            Pads the data in case it cannot be just folded exactly.
        """
        _length = self.x_dict.shape[0]
        self.num_batches = int(onp.ceil(_length / self.batch_size))
        self.last_batch_size = _length % self.batch_size
        self.batched_data = jnp.zeros((self.num_batches, self.batch_size, 3))
        for batch_id in range(self.num_batches-1):
            self.batched_data = self.batched_data.at[batch_id].set(self.x_dict[batch_id: min(_length, batch_id + self.batch_size)])        
        self.batched_data = self.batched_data.at[self.num_batches-1, 0:self.last_batch_size,:].set(self.x_dict[self.num_batches-1: self.num_batches-1 + self.last_batch_size])
        return self.batched_data, self.last_batch_size

        
    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_dict)
        return self
    
    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        # data_x = self.x_dict[self._idx: min(self._len, self._idx + self.batch_size)]
        
        self._idx += self.batch_size
        self._batch_counter += 1
        # if self._batch_counter == self.num_batches:
        #     data_x = self.batched_data[self._batch_counter - 1]
        # else:
        #     data_x = self.batched_data[self._batch_counter - 1][:self.last_batch_size]
        return self.gpu_data[self._batch_counter - 1]



class TrainData:
        def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, Nx=64, Ny=64, Nz=64, num_devices=1) -> None:
            """ Training GSTATE """
            self.num_devices = num_devices
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
            