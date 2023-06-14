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
from jax import config
from jax import numpy as jnp
from jax import random

config.update("jax_debug_nans", False)

import pdb
from typing import Optional


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class DoubleMLP(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.num_hidden_layers = 1
        self.hidden_dim = 64
        self.activation_fn = jnp.sin
        self.tr_normal_init = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)

        d1 = 3  # dimension of input space; e.g., 3D space
        self.d2 = 16  # dimension of lifted space
        key = random.PRNGKey(0)
        cov = jnp.eye(d1)
        mean = jnp.zeros(d1)

        self.Bmat = random.multivariate_normal(key, mean, cov, shape=(self.d2,))
        self.twoPi = 2.0 * jnp.pi

        self.encoding_m = jnp.zeros(2 * self.d2)
        self.encoding_p = jnp.zeros(2 * self.d2)

    def __call__(self, r, phi_r):
        """
        Driver function for evaluating neural networks in appropriate regions
        based on the value of the level set function at the point.
        """
        return jnp.where(phi_r >= 0, self.nn_up_fn(r), self.nn_um_fn(r))

    def nn_up_fn(self, h):
        """
        neural network function for solution in Omega plus
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_p
        """
        # h = self.positional_encoding_p(h)
        for _ in range(self.num_hidden_layers):
            h = hk.Linear(output_size=self.hidden_dim, with_bias=True, w_init=self.tr_normal_init)(h)
            h = layer_norm(h)
            h = self.activation_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h

    def nn_um_fn(self, h):
        """
        neural network function for solution in Omega minus
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_m
        """
        # h = self.positional_encoding_m(h)
        for _ in range(self.num_hidden_layers):
            h = hk.Linear(output_size=self.hidden_dim, with_bias=True, w_init=self.tr_normal_init)(h)
            h = layer_norm(h)
            h = self.activation_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h

    def positional_encoding_p(self, h):
        """
        positional encoding function
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            2*d2 dimensional encoding
        """
        arg = self.twoPi * (self.Bmat @ h)
        self.encoding_p = self.encoding_p.at[: self.d2].set(jnp.sin(arg))
        self.encoding_p = self.encoding_p.at[self.d2 :].set(jnp.cos(arg))
        return self.encoding_p

    def positional_encoding_m(self, h):
        """
        positional encoding function
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            2*d2 dimensional encoding
        """
        arg = self.twoPi * (self.Bmat @ h)
        self.encoding_m = self.encoding_m.at[: self.d2].set(jnp.sin(arg))
        self.encoding_m = self.encoding_m.at[self.d2 :].set(jnp.cos(arg))
        return self.encoding_m

    @staticmethod
    def __version__():
        return "0.0.1"
