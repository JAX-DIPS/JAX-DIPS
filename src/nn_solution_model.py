"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2022 Pouria Mistani and Samira Pakravan. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: mistani

"""

import haiku as hk
from jax import (numpy as jnp, nn as jnn)
from jax import config
config.update("jax_debug_nans", False)
import pdb

class DoubleMLP(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)
    
        self.num_hidden_layers = 1
        self.hidden_dim = 100
        self.activation_fn = jnp.sin 
        self.tr_normal_init = hk.initializers.TruncatedNormal(stddev=0.05, mean=0.0)
        
        self.L = 3
        self.args = 2**jnp.arange(self.L) * jnp.pi
        self.encoding_m = jnp.zeros(6*self.L)
        self.encoding_p = jnp.zeros(6*self.L)
        
      


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
        # h = self.positional_encoding_p(h)
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
        # h = self.positional_encoding_m(h)
        for _ in range(self.num_hidden_layers):
            h = hk.Linear(output_size=self.hidden_dim, with_bias=True, w_init=self.tr_normal_init)(h)
            h = self.activation_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h


    def positional_encoding_p(self, h):
        '''
        positional encoding function 
        input: 
            h: vector of coordinates for one point (x,y,z)
        output:
            L dimensional encoding
        '''
        x_sin = jnp.sin(self.args * h[0]); x_cos = jnp.cos(self.args * h[0])
        y_sin = jnp.sin(self.args * h[1]); y_cos = jnp.cos(self.args * h[1])
        z_sin = jnp.sin(self.args * h[2]); z_cos = jnp.cos(self.args * h[2])

        self.encoding_p = self.encoding_p.at[:self.L].set(x_sin)
        self.encoding_p = self.encoding_p.at[self.L:2*self.L].set(x_cos)

        self.encoding_p = self.encoding_p.at[2*self.L:3*self.L].set(y_sin)
        self.encoding_p = self.encoding_p.at[3*self.L:4*self.L].set(y_cos)

        self.encoding_p = self.encoding_p.at[4*self.L:5*self.L].set(z_sin)
        self.encoding_p = self.encoding_p.at[5*self.L:        ].set(z_cos)
        return self.encoding_p

    def positional_encoding_m(self, h):
        '''
        positional encoding function 
        input: 
            h: vector of coordinates for one point (x,y,z)
        output:
            L dimensional encoding
        '''
        x_sin = jnp.sin(self.args * h[0]); x_cos = jnp.cos(self.args * h[0])
        y_sin = jnp.sin(self.args * h[1]); y_cos = jnp.cos(self.args * h[1])
        z_sin = jnp.sin(self.args * h[2]); z_cos = jnp.cos(self.args * h[2])

        self.encoding_m = self.encoding_m.at[:self.L].set(x_sin)
        self.encoding_m = self.encoding_m.at[self.L:2*self.L].set(x_cos)

        self.encoding_m = self.encoding_m.at[2*self.L:3*self.L].set(y_sin)
        self.encoding_m = self.encoding_m.at[3*self.L:4*self.L].set(y_cos)

        self.encoding_m = self.encoding_m.at[4*self.L:5*self.L].set(z_sin)
        self.encoding_m = self.encoding_m.at[5*self.L:        ].set(z_cos)
        return self.encoding_m



    @staticmethod
    def __version__():
        return '0.0.1'

    
    





