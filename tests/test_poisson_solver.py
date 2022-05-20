
from jax.config import config
from src import io, poisson_solver, mesh
from src.util import f32, i32
from jax import (jit, numpy as jnp, vmap)
import jax
import jax.profiler
import pdb
import time
import os
import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", True)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def test_poisson_solver():

    dim = i32(3)
    xmin = ymin = zmin = f32(-2.0)
    xmax = ymax = zmax = f32(2.0)
    box_size = xmax - xmin
    Nx = i32(128)
    Ny = i32(128)
    Nz = i32(128)

    # --------- Grid nodes
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]

    # ---------------
    # Create helper functions to define a periodic box of some size.
    init_mesh_fn, coord_at = mesh.construct(dim)
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    # -- define velocity field as gradient of a scalar field

    def phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + (y)**2 + z**2) #- 0.5

    def mu_m_fn(r):
        """
        Diffusion coefficient function in $\Omega^-$
        """
        return 1.0

    def mu_p_fn(r):
        """
        Diffusion coefficient function in $\Omega^+$
        """
        return 1.0

    def k_m_fn(r):
        """
        Linear term function in $\Omega^-$
        """
        return 0.0

    def k_p_fn(r):
        """
        Linear term function in $\Omega^+$
        """
        return 0.0

    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return x*x*x

    def f_m_fn(r):
        """
        Source function in $\Omega^-$
        """
        return 1.0

    def f_p_fn(r):
        """
        Source function in $\Omega^+$
        """
        return 1.0

    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return 0.0

    def beta_fn(r):
        """
        Jump in flux at interface
        """
        return 0.0

    init_fn, solve_fn = poisson_solver.setup(
        initial_value_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)

    t1 = time.time()

    sim_state = solve_fn(gstate, sim_state)
    sim_state.solution.block_until_ready()

    t2 = time.time()


    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver.prof")

    log = {
        'U': sim_state.solution,
        'phi': sim_state.phi
    }
    io.write_vtk_manual(gstate, log)

    pdb.set_trace()


if __name__ == "__main__":
    test_poisson_solver()
