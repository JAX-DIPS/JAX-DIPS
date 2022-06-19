import haiku as hk
import jax
from jax import (numpy as jnp, vmap, grad, jit, random, nn as jnn)
from functools import partial
import optax
import pdb



class DoubleMLP(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)
    
        self.num_hidden_layers = 2
        self.hidden_dim = 10
        self.activation_fn = jnn.silu
        self.tr_normal_init = hk.initializers.TruncatedNormal(stddev=1.0, mean=0.0)


    def __call__(self, r, phi_r):
        '''
        Driver function for evaluating neural networks in appropriate regions
        based on the value of the level set function at the point.
        '''
        return jnp.where(phi_r >=0, self.nn_up_fn(r), self.nn_um_fn(r))


    def nn_up_fn(self, h):
        '''
        neural network function for solution in Omega plus
        input: 
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_p
        '''
        for _ in range(self.num_hidden_layers):
            h = hk.Linear(output_size=self.hidden_dim, with_bias=True, w_init=self.tr_normal_init)(h)
            h = self.activation_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h

    def nn_um_fn(self, h):
        '''
        neural network function for solution in Omega minus
        input: 
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_m
        '''
        for _ in range(self.num_hidden_layers):
            h = hk.Linear(output_size=self.hidden_dim, with_bias=True, w_init=self.tr_normal_init)(h)
            h = self.activation_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h


    @staticmethod
    def __version__():
        return '0.0.1'

    
    





