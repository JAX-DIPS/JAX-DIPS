
from jax.config import config
from src import io, poisson_solver_scalable, mesh, level_set
from src.jaxmd_modules.util import f32, i32
from jax import (jit, numpy as jnp, vmap, grad, lax)
import jax
import jax.profiler
import pdb
import time
import os
import sys
from functools import partial

COMPILE_BACKEND = 'gpu'
custom_jit = partial(jit, backend=COMPILE_BACKEND)

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'


def test_poisson_solver_with_jump_complex():

    dim = i32(3)
    xmin = ymin = zmin = f32(-1.0)
    xmax = ymax = zmax = f32(1.0)
    Nx = i32(1024)
    Ny = i32(1024)
    Nz = i32(1024)

    ALGORITHM = 0                   # 0: regression normal derivatives, 1: neural network normal derivatives
    SWITCHING_INTERVAL = 3

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

    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @custom_jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(2.0*x) * jnp.cos(2.0*y) * jnp.exp(z) 
  

    @custom_jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        yx3 = (y-x)/3.0
        return (16.0*yx3**5 - 20.0*yx3**3 + 5.0*yx3) * jnp.log(x+y+3) * jnp.cos(z) 
        

    @custom_jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @custom_jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        r0 = 0.483; ri = 0.151; re = 0.911
        n_1 = 3.0; beta_1 =  0.1; theta_1 = 0.5
        n_2 = 4.0; beta_2 = -0.1; theta_2 = 1.8
        n_3 = 7.0; beta_3 = 0.15; theta_3 = 0.0

        core  = beta_1 * jnp.cos(n_1 * (jnp.arctan2(y,x) - theta_1))
        core += beta_2 * jnp.cos(n_2 * (jnp.arctan2(y,x) - theta_2))
        core += beta_3 * jnp.cos(n_3 * (jnp.arctan2(y,x) - theta_3))

        phi_  = jnp.sqrt(x**2 + y**2 + z**2)
        phi_ += -1.0*r0 * (1.0 + ((x**2 + y**2)/(x**2 + y**2 + z**2))**2 * core ) 

        return phi_
    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @custom_jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @custom_jit
    def mu_m_fn(r):
        """
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 10.0 * (1 + 0.2 * jnp.cos(2*jnp.pi*(x+y)) * jnp.sin(2*jnp.pi*(x-y)) * jnp.cos(z) )

    @custom_jit
    def mu_p_fn(r):
        """
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @custom_jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @custom_jit
    def beta_fn(r):
        """
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r)*grad_u_p_fn(r)
        vec_2 = mu_m_fn(r)*grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)

    @custom_jit
    def k_m_fn(r):
        """
        Linear term function in $\Omega^-$
        """
        return 0.0

    @custom_jit
    def k_p_fn(r):
        """
        Linear term function in $\Omega^+$
        """
        return 0.0

    @custom_jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0 #evaluate_exact_solution_fn(r)



    @custom_jit
    def f_m_fn_(r):
        """
        Source function in $\Omega^-$
        """
        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p)*grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)
            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i]**2 + tangent[i]
            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)
        return laplacian_m_fn(r) * (-1.0)

    @custom_jit
    def f_p_fn_(r):
        """
        Source function in $\Omega^+$
        """
        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p)*grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)
            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i]**2 + tangent[i]
            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)
        return laplacian_p_fn(r) * (-1.0)


    @custom_jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        fm   = -1.0 * mu_m_fn(r) * (-7.0 * jnp.sin(2.0*x) * jnp.cos(2.0*y) * jnp.exp(z)) +\
               -4*jnp.pi*jnp.cos(z)*jnp.cos(4*jnp.pi*x) * 2*jnp.cos(2*x)*jnp.cos(2*y)*jnp.exp(z)   +\
               -4*jnp.pi*jnp.cos(z)*jnp.cos(4*jnp.pi*y) * (-2)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.exp(z) +\
                2*jnp.cos(2*jnp.pi*(x+y))*jnp.sin(2*jnp.pi*(x-y))*jnp.sin(z) * jnp.sin(2*x)*jnp.cos(2*y)*jnp.exp(z)
        
        return fm

    @custom_jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        f_p = -1.0 * ( 
            ( 16*((y-x)/3)**5 - 20*((y-x)/3)**3 + 5*(y-x)/3 ) * (-2)*jnp.cos(z) / (x+y+3)**2 +\
             2*( 16*5*4*(1.0/9.0)*((y-x)/3)**3 - 20*3*2*(1.0/9.0)*((y-x)/3) ) * jnp.log(x+y+3)*jnp.cos(z) +\
            -1*( 16*((y-x)/3)**5 - 20*((y-x)/3)**3 + 5*((y-x)/3) ) * jnp.log(x+y+3)*jnp.cos(z)
        )
        return f_p


    exact_sol = vmap(evaluate_exact_solution_fn)(R)


    init_fn, solve_fn = poisson_solver_scalable.setup(initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn, alpha_fn, beta_fn)
    sim_state = init_fn(R)
   
    t1 = time.time()

    
    sim_state, epoch_store, loss_epochs = solve_fn(gstate, sim_state, algorithm=ALGORITHM, switching_interval=SWITCHING_INTERVAL)
    # sim_state.solution.block_until_ready()

    t2 = time.time()

    print(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile("memory_poisson_solver_scalable.prof")

    log = {
        'phi': sim_state.phi,
        'U': sim_state.solution,
        'U_exact': exact_sol,
        'U-U_exact': sim_state.solution - exact_sol,
        'alpha': sim_state.alpha,
        'beta': sim_state.beta,
        'mu_m': sim_state.mu_m,
        'mu_p': sim_state.mu_p,
        'f_m': sim_state.f_m,
        'f_p': sim_state.f_p,
        'grad_um_x': sim_state.grad_solution[0][:,0],
        'grad_um_y': sim_state.grad_solution[0][:,1],
        'grad_um_z': sim_state.grad_solution[0][:,2],
        'grad_up_x': sim_state.grad_solution[1][:,0],
        'grad_up_y': sim_state.grad_solution[1][:,1],
        'grad_up_z': sim_state.grad_solution[1][:,2],
        'grad_um_n': sim_state.grad_normal_solution[0],
        'grad_up_n': sim_state.grad_normal_solution[1]
    }
    io.write_vtk_manual(gstate, log)

    L_inf_err = abs(sim_state.solution - exact_sol).max()
    rms_err = jnp.square(sim_state.solution - exact_sol).mean()**0.5

    print("\n SOLUTION ERROR\n")

    print(f"L_inf error on solution everywhere in the domain is = {L_inf_err} and root-mean-squared error = {rms_err} ")
    

    """
    MASK the solution over sphere only
    """
    print("\n GRADIENT ERROR\n")

    grad_um = sim_state.grad_solution[0].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up = sim_state.grad_solution[1].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_exact = vmap(grad(exact_sol_m_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up_exact = vmap(grad(exact_sol_p_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    mask_m = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 #-0.5*dx
    err_x_m = abs(grad_um[mask_m][:,0] - grad_um_exact[mask_m][:,0]).max()
    err_y_m = abs(grad_um[mask_m][:,1] - grad_um_exact[mask_m][:,1]).max()
    err_z_m = abs(grad_um[mask_m][:,2] - grad_um_exact[mask_m][:,2]).max()

    mask_p = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 #0.5*dx
    err_x_p = abs(grad_up[mask_p][:,0] - grad_up_exact[mask_p][:,0]).max()
    err_y_p = abs(grad_up[mask_p][:,1] - grad_up_exact[mask_p][:,1]).max()
    err_z_p = abs(grad_up[mask_p][:,2] - grad_up_exact[mask_p][:,2]).max()

    print(f"L_inf errors in grad u in Omega_minus x: {err_x_m}, \t y: {err_y_m}, \t z: {err_z_m}")
    print(f"L_inf errors in grad u in Omega_plus  x: {err_x_p}, \t y: {err_y_p}, \t z: {err_z_p}")
    

    
    #--- normal gradients over interface
    normal_fn = grad(phi_fn)
    normal_vec = vmap(normal_fn)(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_n = sim_state.grad_normal_solution[0].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]
    grad_up_n = sim_state.grad_normal_solution[1].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]

    mask_i_m = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 )
    mask_i_p = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 )
    
    grad_um_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_um_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))
    grad_up_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_up_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))

    err_um_n = abs(grad_um_n - grad_um_n_exact)[mask_i_m].max()
    err_up_n = abs(grad_up_n - grad_up_n_exact)[mask_i_p].max()

    
    print(f"L_inf error in normal grad u on interface minus: {err_um_n} \t plus: {err_up_n}")

    #----
    assert L_inf_err<0.2

    pdb.set_trace()


if __name__ == "__main__":
    test_poisson_solver_with_jump_complex()
