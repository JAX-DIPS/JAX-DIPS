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

from typing import Callable, TypeVar, Tuple

from jax import vmap

from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.solvers.poisson import poisson_kernels_cellular_core_algorithms_scalable
from jax_dips.solvers.simulation_states import PoissonSimStateFn, PoissonSimState

Array = util.Array

T = TypeVar("T")
InitFn = Callable[..., T]
SolveFn = Callable[[T, T], T]
Simulator = Tuple[InitFn, SolveFn]


i32 = util.i32
f32 = util.f32
f64 = util.f64


def setup(
    initial_value_fn: Callable[..., Array],
    dirichlet_bc_fn: Callable[..., Array],
    lvl_set_fn: Callable[..., Array],
    mu_m_fn_: Callable[..., Array],
    mu_p_fn_: Callable[..., Array],
    k_m_fn_: Callable[..., Array],
    k_p_fn_: Callable[..., Array],
    f_m_fn_: Callable[..., Array],
    f_p_fn_: Callable[..., Array],
    alpha_fn_: Callable[..., Array],
    beta_fn_: Callable[..., Array],
    nonlinear_op_m=None,
    nonlinear_op_p=None,
) -> Simulator:
    u_0_fn = vmap(initial_value_fn)
    dir_bc_fn = vmap(dirichlet_bc_fn)
    phi_fn = vmap(lvl_set_fn)
    mu_m_fn = vmap(mu_m_fn_)
    mu_p_fn = vmap(mu_p_fn_)
    k_m_fn = vmap(k_m_fn_)
    k_p_fn = vmap(k_p_fn_)
    f_m_fn = vmap(f_m_fn_)
    f_p_fn = vmap(f_p_fn_)
    alpha_fn = vmap(alpha_fn_)
    beta_fn = vmap(beta_fn_)

    if nonlinear_op_m is None:
        print("nonlinear_op_m(u) is not defined. Setting it to zero.")
        nonlinear_op_m = lambda x: 0.0
    if nonlinear_op_p is None:
        print("nonlinear_op_m(u) is not defined. Setting it to zero.")
        nonlinear_op_p = lambda x: 0.0
    sim_state_fn = PoissonSimStateFn(
        u_0_fn,
        dir_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        nonlinear_op_m,
        nonlinear_op_p,
    )

    def init_fn(R):
        PHI = sim_state_fn.phi_fn(R)
        DIRBC = None  # sim_state_fn.dir_bc_fn(R)
        U = None  # sim_state_fn.u_0_fn(R)
        MU_M = None  # sim_state_fn.mu_m_fn(R)
        MU_P = None  # sim_state_fn.mu_p_fn(R)
        K_M = None  # sim_state_fn.k_m_fn(R)
        K_P = None  # sim_state_fn.k_p_fn(R)
        F_M = None  # sim_state_fn.f_m_fn(R)
        F_P = None  # sim_state_fn.f_p_fn(R)
        ALPHA = None  # sim_state_fn.alpha_fn(R)
        BETA = None  # sim_state_fn.beta_fn(R)
        sim_state = PoissonSimState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None)
        return sim_state

    def solve_fn(
        gstate,
        eval_gstate,
        sim_state,
        algorithm=0,
        switching_interval=3,
        Nx_tr=32,
        Ny_tr=32,
        Nz_tr=32,
        num_epochs=1000,
        multi_gpu=False,
        batch_size=131072,
        checkpoint_dir="./checkpoints",
        checkpoint_interval=2,
        currDir="./",
    ):
        (
            U_sol,
            grad_u_mp,
            grad_u_mp_normal_to_interface,
            epoch_store,
            loss_epochs,
        ) = poisson_kernels_cellular_core_algorithms_scalable.poisson_solver(
            gstate,
            eval_gstate,
            sim_state,
            sim_state_fn,
            algorithm,
            switching_interval=switching_interval,
            Nx_tr=Nx_tr,
            Ny_tr=Ny_tr,
            Nz_tr=Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            currDir=currDir,
        )
        return (
            dataclasses.replace(
                sim_state,
                solution=U_sol,
                grad_solution=grad_u_mp,
                grad_normal_solution=grad_u_mp_normal_to_interface,
            ),
            epoch_store,
            loss_epochs,
        )

    return init_fn, solve_fn
