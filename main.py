import jax
from jax import jit, random, lax, ops
import jax.numpy as jnp
import numpy as onp
from jax.config import config
from src.quantity import EnergyFn
config.update("jax_enable_x64", True)
from src.util import f32, i32
from src import partition, simulate
from src import space
from src import mesh
from src import visualization
import pdb


key = random.PRNGKey(0)

# Setup some variables describing the system.
N = 500
dimension = 2
box_size = f32(25.0)
dt = f32(0.001)


# Create helper functions to define a periodic box of some size.
displacement, shift = space.periodic(box_size)

metric = space.metric(displacement)

# Use JAX's random number generator to generate random initial positions.
key, split = random.split(key)

simulation_steps = 100

xmin = ymin = zmin = -1
xmax = ymax = zmax = 1
box_size = xmax - xmin
Nx = Ny = Nz = 10
xc = jnp.linspace(xmin, xmax, Nx)
yc = jnp.linspace(ymin, ymax, Ny)
zc = jnp.linspace(zmin, zmax, Nz)
X, Y, Z = jnp.meshgrid(xc, yc, zc)
X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()
R = jnp.column_stack((X, Y, Z))





displacement_fn, shift_fn = space.periodic(box_size)
neighbor_fn, energy_fn = mesh.mesh(displacement_fn, box_size)
# energy_fn is a function whose gradient gives velocity field.
key = random.PRNGKey(0)
init_fn, apply_fn = simulate.level_set(energy_fn, shift_fn, dt)

opt_nbrs = neighbor_fn(R)
opt_state = init_fn(key, R, neighbor=opt_nbrs)

@jit
def step_func(i, state_and_nbrs):
    state, nbrs, log = state_and_nbrs
    t = i * dt
    log['t'] = ops.index_update(log['t'], i, t)
    pos = state.position
    log['R'] = ops.index_update(log['R'], i, pos)
    vel = state.velocity
    log['V'] = ops.index_update(log['V'], i, vel)
    nbrs = neighbor_fn(state.position, nbrs)
    return apply_fn(state, neighbor=nbrs), nbrs, log

log = {'R' : jnp.zeros((simulation_steps,) + R.shape),
       'V' : jnp.zeros((simulation_steps,) + R.shape),
       't' : jnp.zeros((simulation_steps,))
      }
opt_state, opt_nbrs, log = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (opt_state, opt_nbrs, log))

opt_state.position.block_until_ready()

final_R = opt_state.position
final_V = opt_state.velocity
final_F = opt_state.force
visualization.animate(log)
pdb.set_trace()