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
from jax import nn
from jax import config
from jax import numpy as jnp
from jax import random
from omegaconf import DictConfig, OmegaConf

config.update("jax_debug_nans", False)

import pdb
from typing import Optional


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


# def compose_model(cfg_model: DictConfig):
class DoubleMLP(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        # mlp
        self.num_hidden_layers_m = 1  # for mlp only
        self.hidden_dim_m = 1  # number of neurons per layer
        self.activation_m_fn = nn.tanh  # nn.celu, jnp.sin, jnp.tanh, nn.swish, ...
        self.tr_normal_init_m = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)

        self.num_hidden_layers_p = 2  # for mlp only
        self.hidden_dim_p = 10  # number of neurons per layer
        self.activation_p_fn = nn.celu  # nn.celu, jnp.sin, jnp.tanh, nn.swish, ...
        self.tr_normal_init_p = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)

        # resnet
        self.num_res_blocks = 2  # for resnet only

        # Positional Encoding Constants
        d1 = 3  # dimension of input space; e.g., 3D space
        self.d2 = self.hidden_dim_p // 2  # dimension of lifted space
        key = random.PRNGKey(0)
        stdev = 1.0  # standard deviation scale
        cov = jnp.eye(d1) * stdev**2
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
        return jnp.where(phi_r >= 0, self.mlp_p_fn(r), self.mlp_m_fn(r))
        # return jnp.where(phi_r >= 0, self.resnet_p_fn(r), self.resnet_m_fn(r))

    def mlp_p_fn(self, h):
        """
        neural network function for solution in Omega plus
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_p
        """
        # h = self.positional_encoding_p(h)

        # h = jnp.linalg.norm(h)[jnp.newaxis]
        for _ in range(self.num_hidden_layers_p):
            h = hk.Linear(output_size=self.hidden_dim_p, w_init=self.tr_normal_init_p)(h)
            # h = layer_norm(h)
            h = self.activation_p_fn(h)
        h = hk.Linear(output_size=1)(h)
        return h

    def mlp_m_fn(self, h):
        """
        neural network function for solution in Omega minus
        input:
            h: vector of coordinates for one point (x,y,z)
        output:
            one scalar value representing the solution u_m
        """
        # h = self.positional_encoding_m(h)
        # h = jnp.linalg.norm(h)[jnp.newaxis]
        for _ in range(self.num_hidden_layers_m):
            h = hk.Linear(output_size=self.hidden_dim_m, w_init=self.tr_normal_init_m)(h)
            # h = layer_norm(h)
            h = self.activation_m_fn(h)

        # bias_init = hk.initializers.Constant(-273.0)
        # weight_init = hk.initializers.TruncatedNormal(stddev=10.0, mean=0.0)
        # h = hk.Linear(output_size=1, w_init=weight_init, b_init=bias_init)(h)

        # bias_init = hk.initializers.Constant(0.0)
        # weight_init = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)
        # h = hk.Linear(output_size=1, w_init=weight_init, b_init=bias_init)(h)

        h = hk.Linear(output_size=1)(h)
        return h

    def resnet_p_fn(self, h):
        # h = self.positional_encoding_p(h)
        for _ in range(self.num_res_blocks):
            # start 1 resnet block
            h_i = hk.Linear(output_size=self.hidden_dim)(h)
            h_ = self.activation_p_fn(h_i)
            h_ = hk.Linear(output_size=self.hidden_dim)(h_)
            h = self.activation_p_fn(h_) + h_i
            # end 1 resnet block
        h = hk.Linear(output_size=1)(h)
        return h

    def resnet_m_fn(self, h):
        # h = self.positional_encoding_m(h)
        for _ in range(self.num_res_blocks):
            # start 1 resnet block
            h_i = hk.Linear(output_size=self.hidden_dim)(h)
            h_ = self.activation_m_fn(h_i)
            h_ = hk.Linear(output_size=self.hidden_dim)(h_)
            h = self.activation_m_fn(h_) + h_i
            # end 1 resnet block
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
        return "0.3.0"


# return DoubleMLP
