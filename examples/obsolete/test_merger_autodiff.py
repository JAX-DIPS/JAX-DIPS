import os, sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)
import time
from functools import partial

import jax
import jax.profiler
from jax import jit, random, lax, numpy as jnp, vmap
from jax_dips.jaxmd_modules.util import f32, i32
from jax_dips.jaxmd_modules import space
from jax_dips import interpolate, solver_advection
from jax_dips import io
from jax_dips import mesh, level_set, compositions
from jax.config import config

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'

key = random.PRNGKey(0)
# key, split = random.split(key)
dim = i32(3)
xmin = ymin = zmin = f32(-2.0)
xmax = ymax = zmax = f32(2.0)
box_size = xmax - xmin
Nx = i32(128)
Ny = i32(128)
Nz = i32(128)
dimension = i32(3)
tf = f32(2 * jnp.pi)

# --------- Grid nodes
xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
dx = xc[1] - xc[0]
dt = dx * f32(0.9)
simulation_steps = i32(tf / dt)

# ---------------
# Create helper functions to define a periodic box of some size.
init_mesh_fn, coord_at = mesh.construct(dim)
gstate = init_mesh_fn(xc, yc, zc)
R = gstate.R
sample_pnt = coord_at(gstate, [1, 1, 1])

displacement_fn, shift_fn = space.periodic(box_size)


# -- define velocity field as gradient of a scalar field
@jit
def velocity_fn(r, time_=0.0):
    x = r[0]
    y = r[1]
    z = r[2]
    return jnp.array([-y, x, 0.0 * z], dtype=f32)


velocity_fn = vmap(velocity_fn, (0, None))


def phi_fn(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return jnp.min(
        jnp.array(
            [
                jnp.sqrt(x**2 + (y - 0.55) ** 2 + z**2) - 0.5,
                jnp.sqrt(x**2 + (y + 0.55) ** 2 + z**2) - 0.5,
            ]
        )
    )
    # return jnp.sqrt(x**2 + (y-0.55)**2 + z**2) - 0.5


(
    init_fn,
    apply_fn,
    reinitialize_fn,
    reinitialized_advect_fn,
) = solver_advection.level_set(phi_fn, shift_fn, dt)
sim_state = init_fn(velocity_fn, R)


# This section is autodiff for the spatial gradients rather than discretizations
# --------------------------------------
def add_null_argument(func):
    def func_(x, y=0):
        return func(x)

    return func_


# AUTODIFF:
def get_velocity_fn_AD(phi_fn, gstate):
    curve_phi_fn = jit(compositions.vec_curvature_fn(phi_fn))
    normal_phi_fn = jit(compositions.vec_normal_fn(phi_fn))
    curvature_phi_n = curve_phi_fn(gstate.R)
    normal_phi_n = normal_phi_fn(gstate.R)
    vel_AD = f32(0.01) * curvature_phi_n.reshape(-1, 1) * normal_phi_n
    velocity_fn_AD = add_null_argument(interpolate.vec_nonoscillatory_quadratic_interpolation(vel_AD, gstate))

    def velocity_at_point(x):
        return velocity_fn_AD(x[jnp.newaxis]).reshape(3)

    return velocity_at_point


# DISCRETE: get normal vector and mean curvature
def get_velocity_fn_DS(sim_state, gstate):
    normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature)
    normal, curve = normal_curve_fn(sim_state.solution, gstate)
    vel = f32(0.01) * curve.reshape(-1, 1) * normal
    velocity_fn_DS = add_null_argument(interpolate.vec_nonoscillatory_quadratic_interpolation(vel, gstate))

    def velocity_fn_node(x):
        return velocity_fn_DS(x[jnp.newaxis])

    return velocity_fn_node


velocity_fn_DS = jit(get_velocity_fn_DS(sim_state, gstate))


# --------
curve_phi_fn = jit(compositions.vec_curvature_fn(phi_fn))
normal_phi_fn = jit(compositions.vec_normal_fn(phi_fn))
curvature_phi_n = curve_phi_fn(gstate.R)
normal_phi_n = normal_phi_fn(gstate.R)
normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature)
normal, curve = normal_curve_fn(sim_state.solution, gstate)
io.write_vtk_manual(
    gstate,
    {
        "phi": sim_state.solution,
        "curvature auto": curvature_phi_n,
        "curvature discrete": curve,
        "normal x auto": normal_phi_n[:, 0],
        "normal y auto": normal_phi_n[:, 1],
        "normal z auto": normal_phi_n[:, 2],
        "normal x discrete": normal[:, 0],
        "normal y discrete": normal[:, 1],
        "normal z discrete": normal[:, 2],
    },
    "results/manual_dump_n",
)
# ---------

# velocity_fn_AD = jit(get_velocity_fn_AD(phi_fn, gstate))
# compose_fn = compositions.node_advect_one_step_autodiff
# phi_np1_fn = jit(partial(compose_fn(phi_fn, velocity_fn_AD), dt))
# phi_np2_fn = jit(partial(compose_fn(phi_np1_fn, velocity_fn_AD), dt))

# --------------------------------------


def updater(gstate, dt):
    def update_node_fn(phi_fn):
        velocity_fn_AD = get_velocity_fn_AD(phi_fn, gstate)
        return partial(compositions.node_advect_one_step_autodiff(phi_fn, velocity_fn_AD), dt)

    return update_node_fn


compose_fn = updater(gstate, dt)

# -------------------------------------
# 2 Steps
phi_np2_fn = compose_fn(compose_fn(phi_fn))
phi_np2 = vmap(phi_np2_fn)(gstate.R)


curve_phi_np2_fn = jit(compositions.vec_curvature_fn(phi_np2_fn))
curvature_phi_np2 = curve_phi_np2_fn(gstate.R)
normal_phi_np2_fn = jit(compositions.vec_normal_fn(phi_np2_fn))
normal_phi_np2 = normal_phi_np2_fn(gstate.R)

# --------
normal_curve_fn = jit(level_set.get_normal_vec_mean_curvature)
normal_np2, curve_np2 = normal_curve_fn(phi_np2, gstate)
io.write_vtk_manual(
    gstate,
    {
        "phi": phi_np2,
        "curvature auto": curvature_phi_np2,
        "curvature discrete": curve_np2,
        "normal x auto": normal_phi_np2[:, 0],
        "normal y auto": normal_phi_np2[:, 1],
        "normal z auto": normal_phi_np2[:, 2],
        "normal x discrete": normal_np2[:, 0],
        "normal y discrete": normal_np2[:, 1],
        "normal z discrete": normal_np2[:, 2],
    },
    "results/manual_dump_np2",
)
# ---------

phi_fin_fn = phi_fn
for i in range(2):
    print(i)
    phi_fin_fn = compose_fn(phi_fin_fn)


def timestepper(i, x):
    phi_fin_fn = compose_fn(phi_fin_fn)
    return phi_fin_fn(x)


phi_next = lax.fori_loop(timestepper, 0, 10, gstate.R[0])


# def get_vel_merger_fn(normal_curve_fn, sim_state, gstate):
#     normal, curve = normal_curve_fn(sim_state.solution, gstate)
#     vel = curve.reshape(-1, 1) * normal
#     return interpolate.vec_multilinear_interpolation(vel, gstate)


def add_null_argument(func):
    def func_(x, y):
        return func(x)

    return func_


log = {
    "U": jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
    "kappaM": jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
    "nx": jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
    "ny": jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
    "nz": jnp.zeros((simulation_steps,) + sim_state.solution.shape, dtype=f32),
    "t": jnp.zeros((simulation_steps,), dtype=f32),
}


@jit
def step_func(i, state_and_nbrs):
    state, log, dt = state_and_nbrs
    time_ = i * dt

    normal, curve = normal_curve_fn(state.solution, gstate)
    vel = f32(0.01) * curve.reshape(-1, 1) * normal
    velocity_fn = add_null_argument(interpolate.vec_nonoscillatory_quadratic_interpolation(vel, gstate))

    log["t"] = log["t"].at[i].set(time_)
    log["U"] = log["U"].at[i].set(state.solution)
    log["kappaM"] = log["kappaM"].at[i].set(curve)
    log["nx"] = log["nx"].at[i].set(normal[:, 0])
    log["ny"] = log["ny"].at[i].set(normal[:, 1])
    log["nz"] = log["nz"].at[i].set(normal[:, 2])

    state = reinitialize_fn(state, gstate)
    # state = lax.cond(i//10==0, lambda p: reinitialize_fn(p[0], p[1]), lambda p : p[0], (state, gstate))
    return apply_fn(velocity_fn, state, gstate, time_), log, dt


t1 = time.time()
sim_state, log, dt = lax.fori_loop(i32(0), i32(simulation_steps), step_func, (sim_state, log, dt))
sim_state.solution.block_until_ready()
t2 = time.time()
print(f"time per timestep is {(t2 - t1)/simulation_steps}")

jax.profiler.save_device_memory_profile("memory.prof")


io.write_vtk_log(gstate, log)
