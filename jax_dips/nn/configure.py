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

from jax_dips.nn.MLP import DoubleMLP
from jax_dips.nn.discrete import discrete
from jax_dips.nn.hash_encoding.model import make_hash_network


def get_model(model_dict, model_type: str = "mlp"):
    if model_type == "mlp" or model_type == "resnet":

        @hk.transform
        def forward(x, phi_x):
            model = DoubleMLP(**model_dict)
            return model(x, phi_x)

        return forward, "haiku"

    elif model_type == "discrete":

        @hk.transform
        def forward(x, phi_x):
            model = discrete(**model_dict)
            return model(x, phi_x)

        return forward, "haiku"

    elif model_type == "multiresolution_hash_network":
        """This is a FLAX model, so doesn't need to be transformed."""
        model = make_hash_network(**model_dict[model_type])
        return model, "flax"
