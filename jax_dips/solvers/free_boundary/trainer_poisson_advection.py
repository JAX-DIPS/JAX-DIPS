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

import signal
import time
from functools import partial
from typing import Callable, Tuple, TypeVar

import jax
import optax
from jax import jit
from jax import numpy as jnp
from jax import pmap, random, vmap

from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.advection import solver_advection
from jax_dips.data import data_management
from jax_dips.solvers.free_boundary import solver_poisson_advection
from jax_dips.solvers.simulation_states import PoissonAdvectionSimState, PoissonAdvectionSimStateFn
from jax_dips.utils import print_architecture
from jax_dips.utils.visualization import plot_loss_epochs

Array = util.Array

T = TypeVar("T")
InitFn = Callable[..., T]
SolveFn = Callable[[T, T], T]
Simulator = Tuple[InitFn, SolveFn]


stop_training = False


def signalHandler(signal_num, frame):
    global stop_training
    stop_training = True
    print("Signal:", signal_num, " Frame: ", frame)
    print("Training will stop after the completion of current epoch")


signal.signal(signal.SIGINT, signalHandler)


class PoissonAdvectionSolve:
    def __init__(
        self,
        gstate,
        eval_gstate,
        sim_state,
        sim_state_fn,
        algorithm=0,
        switching_interval=3,
        Nx_tr=32,
        Ny_tr=32,
        Nz_tr=32,
        num_epochs=1000,
        dt=0.1,
        multi_gpu=False,
        batch_size=131072,
        checkpoint_dir="./checkpoints",
        checkpoint_interval=2,
        currDir="./",
    ) -> None:
        #########################################################################
        global stop_training
        self.key = random.PRNGKey(758493)
        self.eval_gstate = eval_gstate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.multi_gpu = multi_gpu
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.currDir = currDir
        self.Nx_tr = Nx_tr
        self.Ny_tr = Ny_tr
        self.Nz_tr = Nz_tr

        #########################################################################
        OPTZ = "custom"
        optimizer = None
        if OPTZ == "custom":
            learning_rate = 1e-2
            decay_rate_ = 0.975
            scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=decay_rate_)
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )
        elif OPTZ == "adam":
            optimizer = optax.adam(learning_rate)
        elif OPTZ == "rmsprop":
            optimizer = optax.rmsprop(learning_rate)

        #########################################################################
        self.TD = data_management.TrainData(
            gstate.xmin(),
            gstate.xmax(),
            gstate.ymin(),
            gstate.ymax(),
            gstate.zmin(),
            gstate.zmax(),
            Nx_tr,
            Ny_tr,
            Nz_tr,
        )
        self.train_points = self.TD.gstate.R
        self.train_dx = self.TD.gstate.dx
        self.train_dy = self.TD.gstate.dy
        self.train_dz = self.TD.gstate.dz

        #########################################################################
        self.trainer = solver_poisson_advection.PoissonAdvectionTrainer(
            gstate, sim_state, sim_state_fn, optimizer, algorithm
        )

        (
            _,
            phi_apply_fn,
            phi_reinitialize_fn,
            phi_reinitialized_advect_fn,
        ) = solver_advection.level_set(level_set_fn=sim_state_fn.phi_fn, dt=dt)
        sim_state.velocity_nm1 = sim_state_fn.velocity_fn(gstate.R)
        #########################################################################
        state = self.trainer.fetch_checkpoint(self.checkpoint_dir)
        self.epoch_start = 0
        if state is None:
            self.opt_state, self.params = self.trainer.init()
            print_architecture(self.params)
        else:
            self.opt_state = state["opt_state"]
            self.params = state["params"]
            self.epoch_start = state["epoch"]
            self.batch_size = state["batch_size"]
            self.resolution = state["resolution"]
            print(
                f"Resuming training from epoch {self.epoch_start} with batch_size {self.batch_size}, resolution {self.resolution}."
            )

        self.loss_epochs = jnp.zeros(self.num_epochs - self.epoch_start)
        self.epoch_store = jnp.arange(self.epoch_start, self.num_epochs)

        #########################################################################

    #############################-----------------------
    def solve(self):
        if self.multi_gpu:
            return self.multi_GPU_train(self, self.opt_state, self.params)
        else:
            return self.single_GPU_solve()

    #############################-----------------------
    def single_GPU_solve(self):
        self.DD = data_management.DatasetDict(batch_size=self.batch_size, x_data=self.train_points)
        start_time = time.time()
        (
            self.opt_state,
            self.params,
            self.epoch_store,
            self.loss_epochs,
        ) = self.single_GPU_train(self.opt_state, self.params)
        end_time = time.time()
        print(f"solve took {end_time - start_time} (sec)")
        plot_loss_epochs(
            self.epoch_store,
            self.loss_epochs,
            self.currDir,
            self.TD.base_level,
            self.TD.alt_res,
        )
        (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
        ) = self.evaluate_solution_and_gradients(self.params, self.eval_gstate)
        return (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
            self.epoch_store,
            self.loss_epochs,
        )

    #############################-----------------------
    @partial(jit, static_argnums=(0))
    def evaluate_solution_and_gradients(self, params, eval_gstate):
        final_solution = self.trainer.evaluate_solution_fn(params, eval_gstate.R).reshape(-1)
        grad_u_normal_to_interface = self.trainer.compute_normal_gradient_solution_on_interface(
            params, eval_gstate.R, eval_gstate.dx, eval_gstate.dy, eval_gstate.dz
        )
        grad_u = self.trainer.compute_gradient_solution(params, eval_gstate.R)
        return final_solution, grad_u, grad_u_normal_to_interface

    #############################-----------------------
    @partial(jit, static_argnums=(0))
    def single_GPU_train(self, opt_state, params):
        batched_training_data = self.DD.get_batched_data()
        update_fn = self.trainer.update
        num_batches = batched_training_data.shape[1]

        def learn_one_batch(carry, data_batch):
            opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
            opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
            loss_epoch += loss_epoch_
            return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

        def learn_one_epoch(carry, epoch):
            (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ) = carry
            train_dx, train_dy, train_dz = self.TD.alternate_res(epoch, train_dx, train_dy, train_dz)
            batched_training_data = random.permutation(key, batched_training_data, axis=1)
            loss_epoch = 0.0
            (
                opt_state,
                params,
                loss_epoch,
                train_dx,
                train_dy,
                train_dz,
            ), _ = jax.lax.scan(
                learn_one_batch,
                (opt_state, params, loss_epoch, train_dx, train_dy, train_dz),
                batched_training_data,
            )
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            return (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ), None

        (
            self.key,
            opt_state,
            params,
            self.loss_epochs,
            self.train_dx,
            self.train_dy,
            self.train_dz,
            batched_training_data,
        ), _ = jax.lax.scan(
            learn_one_epoch,
            (
                self.key,
                opt_state,
                params,
                self.loss_epochs,
                self.train_dx,
                self.train_dy,
                self.train_dz,
                batched_training_data,
            ),
            self.epoch_store,
        )
        self.loss_epochs /= num_batches
        return opt_state, params, self.epoch_store, self.loss_epochs

    #############################-----------------------
    def multi_GPU_train(self, opt_state, params):
        loss_epochs = []
        epoch_store = []
        n_devices = jax.local_device_count()
        self.train_dx = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dx)
        self.train_dy = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dy)
        self.train_dz = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dz)
        params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
        opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)
        DD = data_management.DatasetDict(
            batch_size=n_devices * self.batch_size,
            x_data=self.train_points,
            num_gpus=n_devices,
        )
        batched_training_data = DD.get_batched_data()
        update_fn = pmap(
            self.trainer.update_multi_gpu,
            in_axes=(0, 0, 0, 0, 0, 0),
            axis_name="devices",
        )

        num_batches = batched_training_data.shape[1]

        for epoch in range(self.epoch_start, self.num_epochs):
            if stop_training:
                break
            loss_epoch = jax.tree_map(lambda x: jnp.array([x] * n_devices), 0.0)
            batched_training_data = random.permutation(self.key, batched_training_data, axis=2)
            for i in range(num_batches):
                opt_state, params, loss_epoch_ = update_fn(
                    opt_state,
                    params,
                    batched_training_data[:, i, ...],
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                )
                loss_epoch += loss_epoch_
            print(
                f"Epoch # {epoch} loss is {jnp.mean(loss_epoch) / num_batches}. Exit training: {stop_training}"
            )  # mean is to support multi-gpu as well.
            loss_epochs.append(loss_epoch / num_batches)
            epoch_store.append(epoch)
            if (epoch + 1) % self.checkpoint_interval == 0:
                state = {
                    "opt_state": opt_state,
                    "params": params,
                    "epoch": epoch + 1,
                    "batch_size": self.batch_size,
                    "resolution": f"{self.Nx_tr}, {self.Ny_tr}, {self.Nz_tr}",
                }
                self.trainer.save_checkpoint(self.checkpoint_dir, state)
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))
        return opt_state, params, epoch_store, loss_epochs


def poisson_advection_solve(
    gstate,
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
    checkpoint_interval=2,
    currDir="./",
):
    global stop_training
    # --- Defining Optimizer
    learning_rate = 1e-2
    decay_rate_ = 0.975
    scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=decay_rate_)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )
    # --------
    # optimizer = optax.adam(learning_rate)
    # --------
    # optimizer = optax.rmsprop(learning_rate)
    # ---------------------
    """ Training Parameters """

    TD = data_management.TrainData(
        gstate.xmin(),
        gstate.xmax(),
        gstate.ymin(),
        gstate.ymax(),
        gstate.zmin(),
        gstate.zmax(),
        Nx_tr,
        Ny_tr,
        Nz_tr,
    )
    train_points = TD.gstate.R
    train_dx = TD.gstate.dx
    train_dy = TD.gstate.dy
    train_dz = TD.gstate.dz

    trainer = solver_poisson_advection.PoissonAdvectionTrainer(gstate, sim_state, sim_state_fn, optimizer, algorithm)
    state = trainer.fetch_checkpoint(checkpoint_dir)
    epoch_start = 0
    if state is None:
        opt_state, params = trainer.init()
        print_architecture(params)
    else:
        opt_state = state["opt_state"]
        params = state["params"]
        epoch_start = state["epoch"]
        batch_size = state["batch_size"]
        resolution = state["resolution"]
        print(f"Resuming training from epoch {epoch_start} with batch_size {batch_size}, resolution {resolution}.")

    start_time = time.time()

    key = random.PRNGKey(758493)

    if not multi_gpu:
        """Single-GPU Update Preparation"""

        DD = data_management.DatasetDict(batch_size=batch_size, x_data=train_points)
        batched_training_data = DD.get_batched_data()
        update_fn = trainer.update
        num_batches = batched_training_data.shape[1]

        def learn_one_batch(carry, data_batch):
            opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
            opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
            loss_epoch += loss_epoch_
            return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

        def learn_one_epoch(carry, epoch):
            (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ) = carry
            train_dx, train_dy, train_dz = TD.alternate_res(epoch, train_dx, train_dy, train_dz)
            batched_training_data = random.permutation(key, batched_training_data, axis=1)
            loss_epoch = 0.0
            (
                opt_state,
                params,
                loss_epoch,
                train_dx,
                train_dy,
                train_dz,
            ), _ = jax.lax.scan(
                learn_one_batch,
                (opt_state, params, loss_epoch, train_dx, train_dy, train_dz),
                batched_training_data,
            )
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            return (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ), None

        loss_epochs = jnp.zeros(num_epochs - epoch_start)
        epoch_store = jnp.arange(epoch_start, num_epochs)
        (
            key,
            opt_state,
            params,
            loss_epochs,
            train_dx,
            train_dy,
            train_dz,
            batched_training_data,
        ), _ = jax.lax.scan(
            learn_one_epoch,
            (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ),
            epoch_store,
        )
        loss_epochs /= num_batches

    elif multi_gpu:
        """Multi-GPU Update Preparation"""

        loss_epochs = []
        epoch_store = []
        n_devices = jax.local_device_count()
        train_dx = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dx)
        train_dy = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dy)
        train_dz = jax.tree_map(lambda x: jnp.array([x] * n_devices), train_dz)
        params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
        opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)
        DD = data_management.DatasetDict(batch_size=n_devices * batch_size, x_data=train_points, num_gpus=n_devices)
        batched_training_data = DD.get_batched_data()
        update_fn = pmap(
            trainer.update_multi_gpu,
            in_axes=(0, 0, 0, 0, 0, 0),
            axis_name="devices",
        )

        num_batches = batched_training_data.shape[1]

        for epoch in range(epoch_start, num_epochs):
            if stop_training:
                break
            loss_epoch = jax.tree_map(lambda x: jnp.array([x] * n_devices), 0.0)
            batched_training_data = random.permutation(key, batched_training_data, axis=2)
            for i in range(num_batches):
                opt_state, params, loss_epoch_ = update_fn(
                    opt_state,
                    params,
                    batched_training_data[:, i, ...],
                    train_dx,
                    train_dy,
                    train_dz,
                )
                loss_epoch += loss_epoch_
            print(
                f"Epoch # {epoch} loss is {jnp.mean(loss_epoch) / num_batches}. Exit training: {stop_training}"
            )  # mean is to support multi-gpu as well.
            loss_epochs.append(loss_epoch / num_batches)
            epoch_store.append(epoch)
            if (epoch + 1) % checkpoint_interval == 0:
                state = {
                    "opt_state": opt_state,
                    "params": params,
                    "epoch": epoch + 1,
                    "batch_size": batch_size,
                    "resolution": f"{Nx_tr}, {Ny_tr}, {Nz_tr}",
                }
                trainer.save_checkpoint(checkpoint_dir, state)
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))

    end_time = time.time()
    print(f"solve took {end_time - start_time} (sec)")
    plot_loss_epochs(epoch_store, loss_epochs, currDir, TD.base_level, TD.alt_res)

    final_solution = trainer.evaluate_solution_fn(params, eval_gstate.R).reshape(-1)

    # ------------- Gradients of discovered solutions:
    grad_u_normal_to_interface = trainer.compute_normal_gradient_solution_on_interface(
        params, eval_gstate.R, eval_gstate.dx, eval_gstate.dy, eval_gstate.dz
    )
    grad_u = trainer.compute_gradient_solution(params, eval_gstate.R)

    return final_solution, grad_u, grad_u_normal_to_interface, epoch_store, loss_epochs


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
    vel_fn_: Callable[..., Array],
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
    vel_fn = vmap(vel_fn_)
    sim_state_fn = PoissonAdvectionSimStateFn(
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
        vel_fn,
    )

    def init_fn(
        dt,
        gstate,
        eval_gstate,
        Nx_tr=32,
        Ny_tr=32,
        Nz_tr=32,
        num_epochs=1000,
        batch_size=131072,
        algorithm=0,
        switching_interval=3,
        multi_gpu=False,
        checkpoint_interval=1000,
        checkpoint_dir="./checkpoints",
        currDir="./",
    ):
        R = eval_gstate.R
        PHI = phi_fn(R)
        DIRBC = dir_bc_fn(R)
        U = u_0_fn(R)
        MU_M = mu_m_fn(R)
        MU_P = mu_p_fn(R)
        K_M = k_m_fn(R)
        K_P = k_p_fn(R)
        F_M = f_m_fn(R)
        F_P = f_p_fn(R)
        ALPHA = alpha_fn(R)
        BETA = beta_fn(R)
        VEL = vel_fn(R)

        def solve_fn(sim_state):
            PAS = PoissonAdvectionSolve(
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
                dt=dt,
                multi_gpu=multi_gpu,
                batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                currDir=currDir,
            )
            (
                final_solution,
                grad_u,
                grad_u_normal_to_interface,
                epoch_store,
                loss_epochs,
            ) = PAS.solve()
            sim_state = (
                dataclasses.replace(
                    sim_state,
                    solution=final_solution,
                    grad_solution=grad_u,
                    grad_normal_solution=grad_u_normal_to_interface,
                ),
                epoch_store,
                loss_epochs,
            )
            return sim_state

        return (
            PoissonAdvectionSimState(
                PHI,
                U,
                DIRBC,
                MU_M,
                MU_P,
                K_M,
                K_P,
                F_M,
                F_P,
                ALPHA,
                BETA,
                None,
                None,
                VEL,
                dt,
            ),
            solve_fn,
        )

    return init_fn
