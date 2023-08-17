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
from typing import Optional

import haiku as hk
from jax import config
from jax import numpy as jnp

from jax_dips.nn.discrete.utils import (
    trilinear_interpolation_per_point,
    nonoscillatory_quadratic_interpolation_per_point,
)
from jax_dips._jaxmd_modules.util import f32


config.update("jax_debug_nans", False)


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class discrete(hk.Module):
    def __init__(
        self,
        name=None,
        model_type: str = "discrete",
        discrete: dict = {
            "Nx": 32,
            "Ny": 32,
            "Nz": 32,
            "xmin": -1.0,
            "xmax": 1.0,
            "ymin": -1.0,
            "ymax": 1.0,
            "zmin": -1.0,
            "zmax": 1.0,
        },
        **kwargs,
    ):
        super().__init__(name=name)
        Nx = discrete["Nx"]
        Ny = discrete["Ny"]
        Nz = discrete["Nz"]
        xc = jnp.linspace(discrete["xmin"], discrete["xmax"], Nx, dtype=f32)
        yc = jnp.linspace(discrete["ymin"], discrete["ymax"], Ny, dtype=f32)
        zc = jnp.linspace(discrete["zmin"], discrete["zmax"], Nz, dtype=f32)

        if discrete["interpolant"] == "quadratic":
            trainables_m = hk.get_parameter("trainables_m", shape=[Nx, Ny, Nz], dtype=f32, init=jnp.zeros)
            trainables_p = hk.get_parameter("trainables_p", shape=[Nx, Ny, Nz], dtype=f32, init=jnp.zeros)
            self.interp_m_fn = nonoscillatory_quadratic_interpolation_per_point(trainables_m, xc, yc, zc)
            self.interp_p_fn = nonoscillatory_quadratic_interpolation_per_point(trainables_p, xc, yc, zc)
        elif discrete["interpolant"] == "trilinear":
            trainables_m = hk.get_parameter("trainables_m", shape=[Nx, Ny, Nz], dtype=f32, init=jnp.zeros)
            trainables_p = hk.get_parameter("trainables_p", shape=[Nx, Ny, Nz], dtype=f32, init=jnp.zeros)
            self.interp_m_fn = trilinear_interpolation_per_point(trainables_m, xc, yc, zc)
            self.interp_p_fn = trilinear_interpolation_per_point(trainables_p, xc, yc, zc)
        elif discrete["interpolant"] == "single_trilinear":
            trainables_mp = hk.get_parameter("trainables_mp", shape=[Nx, Ny, Nz], dtype=f32, init=jnp.zeros)
            self.interp_m_fn = trilinear_interpolation_per_point(trainables_mp, xc, yc, zc)
            self.interp_p_fn = self.interp_m_fn

    def __call__(self, r, phi_r):
        """
        Driver function for evaluating neural networks in appropriate regions
        based on the value of the level set function at the point.
        """
        return jnp.where(phi_r >= 0, self.interp_p_fn(r), self.interp_m_fn(r))

    @staticmethod
    def __version__():
        return "0.3.0"
