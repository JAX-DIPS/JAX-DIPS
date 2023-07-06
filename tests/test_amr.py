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
import os
import sys

from jax import numpy as jnp
from jax.config import config

from jax_dips.data import data_management

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def phi_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return jnp.sqrt(x * x + y * y + z * z) - 0.25


def test_amr():
    xmin = ymin = zmin = -1
    xmax = ymax = zmax = 1
    Nx = Ny = Nz = 32

    TD = data_management.TrainData(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)
    points = TD.gstate.R
    refined_points = TD.refine_LOD(phi_fn)


if __name__ == "__main__":
    test_amr()
