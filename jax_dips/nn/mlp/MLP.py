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

from typing import Optional


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class DoubleMLP(hk.Module):
    def __init__(
        self,
        name=None,
        model_type: str = "mlp",
        mlp: dict = {
            "hidden_layers_m": 1,
            "hidden_dim_m": 1,
            "activation_m": "jnp.tanh",
            "hidden_layers_p": 2,
            "hidden_dim_p": 10,
            "activation_p": "jnp.tanh",
        },
        resnet: dict = {
            "res_blocks_m": 3,
            "res_dim_m": 40,
            "activation_m": "nn.tanh",
            "res_blocks_p": 3,
            "res_dim_p": 80,
            "activation_p": "nn.tanh",
        },
        **kwargs,
    ):
        super().__init__(name=name)
        if model_type == "mlp":
            # mlp
            self.num_hidden_layers_m = mlp["hidden_layers_m"]  # for mlp only
            self.hidden_dim_m = mlp["hidden_dim_m"]  # number of neurons per layer
            self.activation_m_fn = eval(mlp["activation_m"])  # nn.celu, jnp.sin, jnp.tanh, nn.swish, ...
            self.tr_normal_init_m = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)

            self.num_hidden_layers_p = mlp["hidden_layers_p"]  # for mlp only
            self.hidden_dim_p = mlp["hidden_dim_p"]  # number of neurons per layer
            self.activation_p_fn = eval(mlp["activation_p"])  # nn.celu, jnp.sin, jnp.tanh, nn.swish, ...
            self.tr_normal_init_p = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.0)
        elif model_type == "resnet":
            # resnet
            self.num_res_blocks_m = resnet["res_blocks_m"]  # for resnet only
            self.hidden_resnet_dim_m = resnet["res_dim_m"]
            self.activation_resnet_m_fn = eval(resnet["activation_m"])

            self.num_res_blocks_p = resnet["res_blocks_p"]  # for resnet only
            self.hidden_resnet_dim_p = resnet["res_dim_p"]
            self.activation_resnet_p_fn = eval(resnet["activation_p"])

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
        for _ in range(self.num_res_blocks_p):
            # start 1 resnet block
            h_i = hk.Linear(output_size=self.hidden_resnet_dim_p)(h)
            h_ = self.activation_resnet_p_fn(h_i)
            h_ = hk.Linear(output_size=self.hidden_resnet_dim_p)(h_)
            h = self.activation_resnet_p_fn(h_) + h_i
            # end 1 resnet block
        h = hk.Linear(output_size=1)(h)
        return h

    def resnet_m_fn(self, h):
        # h = self.positional_encoding_m(h)
        for _ in range(self.num_res_blocks_m):
            # start 1 resnet block
            h_i = hk.Linear(output_size=self.hidden_resnet_dim_m)(h)
            h_ = self.activation_resnet_m_fn(h_i)
            h_ = hk.Linear(output_size=self.hidden_resnet_dim_m)(h_)
            h = self.activation_resnet_m_fn(h_) + h_i
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
