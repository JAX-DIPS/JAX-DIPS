import jax
from jax import jit, random
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
from src.util import f32, i32
from src import partition
from src import space
from mayavi import mlab
import pdb

key = random.PRNGKey(0)

# Setup some variables describing the system.
N = 500
dimension = 2
box_size = f32(25.0)


# Create helper functions to define a periodic box of some size.
displacement, shift = space.periodic(box_size)

metric = space.metric(displacement)

# Use JAX's random number generator to generate random initial positions.
key, split = random.split(key)

xmin = ymin = zmin = -1
xmax = ymax = zmax = 1
Nx = Ny = Nz = 10
xc = np.linspace(xmin, xmax, Nx)
yc = np.linspace(ymin, ymax, Ny)
zc = np.linspace(zmin, zmax, Nz)
X, Y, Z = np.meshgrid(xc, yc, zc)
X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()
R = np.column_stack((X, Y, Z))
mlab.points3d(X, Y, Z)
mlab.show()
pdb.set_trace()