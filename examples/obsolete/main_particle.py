import jax
from jax import jit, random, lax, ops
import jax.numpy as jnp
import numpy as onp
from jax.config import config
from src.jaxmd_modules import energy, partition
from src.jaxmd_modules.quantity import EnergyFn
config.update("jax_enable_x64", True)
from src.jaxmd_modules.util import f32, i32
from src.jaxmd_modules import space
from src import simulate_fields
from src import simulate_particles
from src import visualization


# Use JAX's random number generator to generate random initial positions.
key = random.PRNGKey(0)
#key, split = random.split(key)


xmin = ymin = zmin = f32(-1.0)
xmax = ymax = zmax = f32(1.0)
box_size = xmax - xmin
Nx = Ny = Nz = i32(10)
dimension = i32(2)
dt = f32(0.001)
simulation_steps = i32(1000)

#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx)
yc = jnp.linspace(ymin, ymax, Ny)
zc = jnp.linspace(zmin, zmax, Nz)
X, Y, Z = jnp.meshgrid(xc, yc, zc)
X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()
R = jnp.column_stack((X, Y, Z))

#---------------
# Create helper functions to define a periodic box of some size.
displacement_fn, shift_fn = space.periodic(box_size)
neighbor_fn, energy_fn = energy.energy(displacement_fn, box_size)              # energy_fn is a function whose gradient gives velocity field.
init_fn, apply_fn = simulate_particles.conservative(energy_fn, shift_fn, dt)
opt_nbrs = neighbor_fn(R)
opt_state = init_fn(key, R, neighbor=opt_nbrs)

@jit
def step_func(i, state_and_nbrs):
    state, nbrs, log = state_and_nbrs
    t = i * dt
    log['t'] = log['t'].at[i].set(t)
    pos = state.position
    log['R'] = log['R'].at[i].set(pos)
    vel = state.velocity
    log['V'] = log['V'].at[i].set(vel)
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
