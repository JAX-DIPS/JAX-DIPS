import jax
from jax import jit, random, lax, ops, grad
import jax.numpy as jnp
import numpy as onp
from jax.config import config
from src import quantity
from src.quantity import EnergyFn
config.update("jax_enable_x64", True)
from src.util import f32, i32
from src import partition, space
from src import simulate_fields
from src import energy, simulate_particles
from src import visualization
from src import mesh
from src import util
import pdb

# Use JAX's random number generator to generate random initial positions.
key = random.PRNGKey(0)
#key, split = random.split(key)

dim = i32(3)
xmin = ymin = zmin = f32(-1.0)
xmax = ymax = zmax = f32(1.0)
box_size = xmax - xmin
Nx = Ny = Nz = i32(10)
dimension = i32(2)
dt = f32(0.001)
simulation_steps = i32(100)



#--------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx)
yc = jnp.linspace(ymin, ymax, Ny)
zc = jnp.linspace(zmin, zmax, Nz)

#---------------
# Create helper functions to define a periodic box of some size.


init_mesh_fn, coord_at = mesh.construct(dim)
gstate = init_mesh_fn(xc, yc, zc)
R = gstate.R
sample_pnt = coord_at(gstate, [1,1,1])

displacement_fn, shift_fn = space.periodic(box_size)

#-- define velocity field as gradient of a scalar field
def energy_fn(r):
    engy =  0.5 * space.square_distance(r)
    return engy
velocity_fn = grad(jit(energy_fn))

init_fn, apply_fn = simulate_fields.level_set(velocity_fn, shift_fn, dt)

sim_state = init_fn(R)

@jit
def step_func(i, state_and_nbrs):
    state, log = state_and_nbrs
    t = i * dt
    log['t'] = ops.index_update(log['t'], i, t)
    sol = state.solution
    log['U'] = ops.index_update(log['U'], i, sol)
    return apply_fn(state, gstate), log

log = {
        'U' : jnp.zeros((simulation_steps,) + sim_state.solution.shape),
        't' : jnp.zeros((simulation_steps,))
      }
sim_state, log = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (sim_state, log))

sim_state.solution.block_until_ready()


final_U = sim_state.solution

pdb.set_trace()