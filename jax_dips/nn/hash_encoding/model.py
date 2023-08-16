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

config.update("jax_debug_nans", False)

from jax_dips._jaxmd_modules.util import f32


class HashNetwork(hk.Module):
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

    def __call__(self, r, phi_r):
        """
        Driver function for evaluating neural networks in appropriate regions
        based on the value of the level set function at the point.
        """
        return jnp.where(phi_r >= 0, self.interp_p_fn(r), self.interp_m_fn(r))

    def build_encoding(self, cfg_encoding):
        if cfg_encoding.type == "fourier":
            encoding_dim = 6 * cfg_encoding.levels
        elif cfg_encoding.type == "hashgrid":
            # Build the multi-resolution hash grid.
            l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
            r_min, r_max = 2**l_min, 2**l_max
            num_levels = cfg_encoding.levels
            self.growth_rate = jnp.exp((jnp.log(r_max) - jnp.log(r_min)) / (num_levels - 1))
            config = dict(
                otype="HashGrid",
                n_levels=cfg_encoding.levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=2**cfg_encoding.hashgrid.min_logres,
                per_level_scale=self.growth_rate,
            )
            self.tcnn_encoding = tcnn.Encoding(3, config)
            self.resolutions = []
            for lv in range(0, num_levels):
                size = jnp.floor(r_min * self.growth_rate**lv).astype(int) + 1
                self.resolutions.append(size)
            encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim

    def encode(self, points_3D):
        if self.cfg_sdf.encoding.type == "fourier":
            points_enc = nerf_util.positional_encoding(points_3D, num_freq_bases=self.cfg_sdf.encoding.levels)
            feat_dim = 6
        elif self.cfg_sdf.encoding.type == "hashgrid":
            # Tri-linear interpolate the corresponding embeddings from the dictionary.
            vol_min, vol_max = self.cfg_sdf.encoding.hashgrid.range
            points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
            tcnn_input = points_3D_normalized.view(-1, 3)
            tcnn_output = self.tcnn_encoding(tcnn_input)
            points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
            feat_dim = self.cfg_sdf.encoding.hashgrid.dim
        else:
            raise NotImplementedError("Unknown encoding type")
        # Coarse-to-fine.
        if self.cfg_sdf.encoding.coarse2fine.enabled:
            mask = self._get_coarse2fine_mask(points_enc, feat_dim=feat_dim)
            points_enc = points_enc * mask
        points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,R,N,3+LD]
        return points_enc

    @staticmethod
    def __version__():
        return "0.3.0"
