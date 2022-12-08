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

from typing import Callable, TypeVar, Union, Tuple, Dict, Optional
from src import data_management, solver_advection, solver_poisson_advection
from src.utils import print_architecture 
import jax
import optax
from jax import (pmap, vmap, numpy as jnp, random)

from src.jaxmd_modules import dataclasses, util


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as onp
from functools import partial
import time
import signal

Array = util.Array

T = TypeVar('T')
InitFn = Callable[..., T]
SolveFn = Callable[[T,T], T]
Simulator = Tuple[InitFn, SolveFn]


i32 = util.i32
f32 = util.f32
f64 = util.f64



stop_training = False
def signalHandler(signal_num, frame):
    global stop_training
    stop_training = True
    print("Signal:", signal_num, " Frame: ", frame)
    print('Training will stop after the completion of current epoch')
signal.signal(signal.SIGINT, signalHandler)




@dataclasses.dataclass
class SState:
    """A struct containing the state of the simulation.

    This tuple stores the state of a simulation.

    Attributes:
    u: An ndarray of shape [n, spatial_dimension] storing the solution value at grid points.
    """
    phi: Array
    solution: Array
    dirichlet_bc: Array
    mu_m: Array
    mu_p: Array
    k_m: Array
    k_p: Array
    f_m: Array
    f_p: Array
    alpha: Array
    beta: Array
    grad_solution: Array
    grad_normal_solution: Array
    velocity_nm1: Array






def advect_poisson_solver(gstate,
                   eval_gstate,
                   sim_state,
                   sim_state_fn,
                   algorithm=0,
                   switching_interval=3,
                   Nx_tr=32,
                   Ny_tr=32,
                   Nz_tr=32,
                   num_epochs=1000,
                   multi_gpu=False,
                   batch_size=131072,
                   checkpoint_dir="./checkpoints",
                   checkpoint_interval=2):
    global stop_training
    #--- Defining Optimizer
    learning_rate = 1e-2
    decay_rate_ = 0.975
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=100,
        decay_rate=decay_rate_)
    optimizer = optax.chain(
                            optax.clip_by_global_norm(1.0),
                            optax.scale_by_adam(),
                            optax.scale_by_schedule(scheduler),
                            optax.scale(-1.0)
    )
    #--------
    # optimizer = optax.adam(learning_rate)
    #--------
    # optimizer = optax.rmsprop(learning_rate)
    #---------------------
    """ Training Parameters """

    TD = data_management.TrainData(gstate.xmin(), gstate.xmax(), gstate.ymin(), gstate.ymax(), gstate.zmin(), gstate.zmax(), Nx_tr, Ny_tr, Nz_tr)
    train_points = TD.gstate.R
    train_dx = TD.gstate.dx
    train_dy = TD.gstate.dy
    train_dz = TD.gstate.dz

    trainer = solver_poisson_advection.AdvectionPoissonTrainer(gstate, sim_state, sim_state_fn, optimizer, algorithm)
    state = trainer.fetch_checkpoint(checkpoint_dir)
    epoch_start = 0
    if state is None:
        opt_state, params = trainer.init(); print_architecture(params)
    else:
        opt_state = state['opt_state']
        params = state['params']
        epoch_start = state['epoch']
        batch_size = state['batch_size']
        resolution = state['resolution']
        print(f'Resuming training from epoch {epoch_start} with batch_size {batch_size}, resolution {resolution}.')

    loss_epochs = []
    epoch_store = []

    if multi_gpu:
        """ Multi-GPU Training """
        n_devices = jax.local_device_count()
        train_dx = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dx)
        train_dy = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dy)
        train_dz = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dz)
        params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
        opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)
        DD = data_management.DatasetDict(batch_size=n_devices*batch_size, x_data=train_points, num_gpus=n_devices)
        batched_training_data = DD.get_batched_data()
        update_fn = pmap(trainer.update_multi_gpu, in_axes=(0, 0, 0, 0, 0, 0), axis_name='num_devices')
    else:
        """ Single GPU """
        DD = data_management.DatasetDict(batch_size=batch_size, x_data=train_points)
        batched_training_data = DD.get_batched_data()
        update_fn = trainer.update

    num_batches = batched_training_data.shape[1]
    start_time = time.time()

    def learn_one_batch(carry, data_batch):
        opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
        opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
        loss_epoch += loss_epoch_
        return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

    key = random.PRNGKey(758493)
    for epoch in range(epoch_start, num_epochs):
        if stop_training:
            break

        if multi_gpu:
            loss_epoch = jax.tree_map(lambda x: jnp.array([x] * n_devices), 0.0 )
            batched_training_data = random.shuffle(key, batched_training_data, axis=2)
            for i in range(num_batches):
                opt_state, params, loss_epoch_ = update_fn(opt_state, params, batched_training_data[:,i,...], train_dx, train_dy, train_dz)
                loss_epoch += loss_epoch_
        else:
            loss_epoch = 0.0
            train_dx, train_dy, train_dz = TD.alternate_res(epoch, train_dx, train_dy, train_dz)
            batched_training_data = random.shuffle(key, batched_training_data, axis=1)
            (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), _ = jax.lax.scan(learn_one_batch, (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), batched_training_data)

        loss_epoch /= num_batches
        print(f"Epoch # {epoch} loss is {jnp.mean(loss_epoch)}. Exit training: {stop_training}")  # mean is to support multi-gpu as well.
        loss_epochs.append(loss_epoch)
        epoch_store.append(epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            state = {
                'opt_state': opt_state,
                'params': params,
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'resolution': f'{Nx_tr}, {Ny_tr}, {Nz_tr}'
            }
            trainer.save_checkpoint(checkpoint_dir, state)

    if multi_gpu:
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))

    #for epoch in range(num_epochs):
        # #train_dx, train_dy, train_dz = TD.alternate_res(epoch, train_dx, train_dy, train_dz)
        # #train_points = TD.move_train_points(train_points, train_dx, train_dy, train_dz)
        # ds_iter = DD(train_points)

        # for x in ds_iter:
        #    if multi_gpu: x = jax.tree_map(split, x)
        #    opt_state, params, loss_epoch = update_fn(opt_state, params, x, train_dx, train_dy, train_dz)

        # print(f"epoch # {epoch} loss is {loss_epoch}")
        # loss_epochs.append(loss_epoch)
        # epoch_store.append(epoch)

    epoch_store = onp.array(epoch_store)
    loss_epochs = onp.array(loss_epochs)



    #----- HALF JITTED
    # BATCH_SIZE_A6000 = 100000
    # NUMDATA = len(eval_gstate.R)
    # num_batches = jnp.ceil(NUMDATA // BATCH_SIZE_A6000)
    # def learn_whole_batched(carry, batch_idx):
    #     opt_state, params, egstate = carry
    #     x = egstate.R[batch_idx: jnp.min(jnp.array([NUMDATA, batch_idx + BATCH_SIZE_A6000]))]
    #     opt_state, params, loss_epoch = trainer.update(opt_state, params, x, egstate)
    #     return (opt_state, params, loss_epoch, egstate), None
    # epoch_store = []
    # loss_epochs = []
    # loss_epochs = jnp.zeros(num_epochs)
    # batch_store = jnp.arange(num_batches)
    # for epoch in range(num_epochs):
    #     (opt_state, params, loss_epoch, eval_gstate), _ = jax.lax.scan(learn_whole_batched, (opt_state, params, eval_gstate), batch_store)
    #     loss_epochs.append(loss_epoch)
    #     epoch_store.append(epoch)
    #     print(f"epoch # {epoch} loss is {loss_epoch}")



    # def learn_whole(carry, epoch):
    #     opt_state, params, loss_epochs = carry
    #     opt_state, params, loss_epoch = trainer.update(opt_state, params, eval_gstate)
    #     loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
    #     return (opt_state, params, loss_epochs), None
    # loss_epochs = jnp.zeros(num_epochs)
    # epoch_store = jnp.arange(num_epochs)
    # (opt_state, params, loss_epochs), _ = jax.lax.scan(learn_whole, (opt_state, params, loss_epochs), epoch_store)


    end_time = time.time()
    print(f"solve took {end_time - start_time} (sec)")



    fig, ax = plt.subplots(figsize=(8,8))

    # plt.plot(epoch_store[epoch_store%switching_interval - 1 ==0], loss_epochs[epoch_store%switching_interval - 1 ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 <0], loss_epochs[epoch_store%switching_interval - 1 <0], color='b', label='negative domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 >0], loss_epochs[epoch_store%switching_interval - 1 >0], color='r', label='positive domain')

    # plt.plot(epoch_store[epoch_store%switching_interval ==0], loss_epochs[epoch_store%switching_interval ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[-1*( epoch_store%switching_interval) <0], loss_epochs[-1*(epoch_store%switching_interval) <0], color='b', label='negative domain')

    if TD.alt_res:
        ax.plot(epoch_store[epoch_store%4==0], loss_epochs[epoch_store%4==0], color='k', label=r'$\rm level=\ $'+str(TD.base_level    ))
        ax.plot(epoch_store[epoch_store%4==1], loss_epochs[epoch_store%4==1], color='b', label=r'$\rm level=\ $'+str(TD.base_level + 1))
        ax.plot(epoch_store[epoch_store%4==2], loss_epochs[epoch_store%4==2], color='r', label=r'$\rm level=\ $'+str(TD.base_level + 2))
        ax.plot(epoch_store[epoch_store%4==3], loss_epochs[epoch_store%4==3], color='g', label=r'$\rm level=\ $'+str(TD.base_level + 3))

    else:
        ax.plot(epoch_store, loss_epochs, color='k', label=r'$\rm level\ =\ $'+str(TD.base_level    ))

    ax.set_yscale('log')
    ax.set_xlabel(r'$\rm epoch$', fontsize=20)
    ax.set_ylabel(r'$\rm loss$', fontsize=20)
    plt.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.tight_layout()
    plt.savefig('tests/poisson_solver_loss.png')
    plt.close()

    final_solution = trainer.evaluate_solution_fn(params, eval_gstate.R).reshape(-1)

    #------------- Gradients of discovered solutions are below:
    if algorithm==0:
        grad_u_mp_normal_to_interface = trainer.compute_normal_gradient_solution_mp_on_interface(params, eval_gstate.R, eval_gstate.dx, eval_gstate.dy, eval_gstate.dz)
        grad_u_mp = trainer.compute_gradient_solution_mp(params, eval_gstate.R)
        grad_u_normal_to_interface = trainer.compute_normal_gradient_solution_on_interface(params, eval_gstate.R, eval_gstate.dx, eval_gstate.dy, eval_gstate.dz)
        grad_u = trainer.compute_gradient_solution(params, eval_gstate.R)
    elif algorithm==1:
        grad_u_mp_normal_to_interface = trainer.compute_normal_gradient_solution_mp_on_interface(None, params)
        grad_u_mp = trainer.compute_gradient_solution_mp(None, params)
        grad_u_normal_to_interface = None
        grad_u = None

    return final_solution, grad_u, grad_u_normal_to_interface, epoch_store, loss_epochs 
    # return final_solution, grad_u_mp, grad_u_mp_normal_to_interface, epoch_store, loss_epochs










def setup(initial_value_fn :  Callable[..., Array], 
          dirichlet_bc_fn  :  Callable[..., Array],
          lvl_set_fn       :  Callable[..., Array], 
          mu_m_fn_         :  Callable[..., Array], 
          mu_p_fn_         :  Callable[..., Array], 
          k_m_fn_          :  Callable[..., Array], 
          k_p_fn_          :  Callable[..., Array],
          f_m_fn_          :  Callable[..., Array],
          f_p_fn_          :  Callable[..., Array],
          alpha_fn_        :  Callable[..., Array],
          beta_fn_         :  Callable[..., Array]
          ) -> Simulator:

    u_0_fn   = vmap(initial_value_fn)
    dir_bc_fn= vmap(dirichlet_bc_fn)
    phi_fn   = vmap(lvl_set_fn)
    mu_m_fn  = vmap(mu_m_fn_)
    mu_p_fn  = vmap(mu_p_fn_)
    k_m_fn   = vmap(k_m_fn_)
    k_p_fn   = vmap(k_p_fn_)
    f_m_fn   = vmap(f_m_fn_)
    f_p_fn   = vmap(f_p_fn_)
    alpha_fn = vmap(alpha_fn_)
    beta_fn  = vmap(beta_fn_)
    
    def init_fn(R):
        PHI   = phi_fn(R)
        DIRBC = dir_bc_fn(R)
        U     = u_0_fn(R)
        MU_M  = mu_m_fn(R)
        MU_P  = mu_p_fn(R)
        K_M   = k_m_fn(R)
        K_P   = k_p_fn(R)
        F_M   = f_m_fn(R)
        F_P   = f_p_fn(R)
        ALPHA = alpha_fn(R)
        BETA  = beta_fn(R)
        return SState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None, None) 

    def solve_fn(gstate, sim_state, algorithm=0, switching_interval=3):
        U_sol, grad_u_mp, grad_u_mp_normal_to_interface, epoch_store, loss_epochs = advect_poisson_solver(gstate, sim_state, algorithm, switching_interval=switching_interval)

        return dataclasses.replace(sim_state, solution=U_sol, grad_solution=grad_u_mp, grad_normal_solution=grad_u_mp_normal_to_interface), epoch_store, loss_epochs
    
    return init_fn, solve_fn