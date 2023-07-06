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
import logging
import os
import pickle
import signal
import time
from functools import partial
from typing import Callable, Tuple, TypeVar

import numpy as onp

logger = logging.getLogger(__name__)

import jax
import optax
from jax import jit
from jax import numpy as jnp
from jax import pmap, random, vmap
from optax._src.base import GradientTransformation

from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.data import data_management
from jax_dips.domain.mesh import GridState
from jax_dips.solvers.poisson import nbm
from jax_dips.solvers.simulation_states import PoissonSimState, PoissonSimStateFn
from jax_dips.utils.inspect import print_architecture, progress_bar
from jax_dips.utils.visualization import plot_loss_epochs

Array = util.Array
i32 = util.i32
f32 = util.f32
f64 = util.f64

T = TypeVar("T")
SolveFn = Callable[[PoissonSimState], PoissonSimState]
InitFn = Callable[
    [GridState, GridState, int, int, int, int, int, int, int, bool, int, str, str, str],
    Tuple[PoissonSimState, SolveFn],
]
Simulator = Tuple[InitFn, SolveFn]


stop_training = False


def signalHandler(signal_num, frame):
    global stop_training
    stop_training = True
    logger.warning("Signal:", signal_num, " Frame: ", frame)
    logger.warning("Training will stop after the completion of current epoch")


signal.signal(signal.SIGINT, signalHandler)


class Trainer:
    """Trainer class for Poisson solvers. Performs all operations for multi/single GPU training.

    Uses loss residuals to train the neural network model. This class relies on the NBM oracle (nbm.py) to
    interact with the neural network surrogate model for evaulating the loss, solution, solution gradients.
    """

    def __init__(
        self,
        gstate: GridState,
        eval_gstate: GridState,
        sim_state: PoissonSimState,
        sim_state_fn: PoissonSimStateFn,
        algorithm: int = 0,
        switching_interval: int = 3,
        Nx_tr: int = 32,
        Ny_tr: int = 32,
        Nz_tr: int = 32,
        lvl_set_fn: Callable[..., Array] = None,
        num_epochs: int = 1000,
        multi_gpu: bool = False,
        batch_size: int = 131072,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 2,
        results_dir: str = "./",
        loss_plot_name: str = "solver_loss",
        optimizer: GradientTransformation = optax.adam(1e-2),
        restart: bool = False,
        print_rate: int = 1,
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
        self.results_dir = results_dir
        self.loss_plot_name = loss_plot_name
        self.Nx_tr = Nx_tr
        self.Ny_tr = Ny_tr
        self.Nz_tr = Nz_tr
        self.print_rate = print_rate
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
        train_points = self.TD.gstate.R
        # extra_points = self.TD.refine_normals(lvl_set_fn, max_iters=15)
        # self.train_points = jnp.concatenate((train_points, extra_points))

        # surface_points = self.TD.refine_LOD(lvl_set_fn, init_res=32, upsamples=3)
        # self.train_points = jnp.concatenate((train_points, surface_points))

        self.train_points = train_points

        self.train_dx = self.TD.gstate.dx
        self.train_dy = self.TD.gstate.dy
        self.train_dz = self.TD.gstate.dz
        #########################################################################

        self.model = nbm.Bootstrap(
            gstate,
            sim_state,
            sim_state_fn,
            optimizer,
            algorithm,
        )
        #########################################################################
        if restart:
            state = self.fetch_checkpoint(self.checkpoint_dir)
            self.opt_state = state["opt_state"]
            self.params = state["params"]
            self.epoch_start = state["epoch"]
            self.batch_size = state["batch_size"]
            self.resolution = state["resolution"]
            logger.info(
                f"Resuming training from epoch {self.epoch_start} with \
                batch_size {self.batch_size}, resolution {self.resolution}."
            )
        else:
            state = None
            self.opt_state, self.params = self.model.init()
            print_architecture(self.params)

        self.epoch_start = 0

        self.loss_epochs = jnp.zeros(self.num_epochs - self.epoch_start)
        self.epoch_store = onp.arange(self.epoch_start, self.num_epochs)
        #########################################################################

    def fetch_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            return None
        else:
            checkpoints = [p for p in os.listdir(checkpoint_dir) if "checkpoint_" in p]
            if checkpoints == []:
                return None
            checkpoint = os.path.join(checkpoint_dir, max(checkpoints))
            logger.info(f"Loading checkpoint {checkpoint}")
            with open(checkpoint, "rb") as f:
                state = pickle.load(f)
            return state

    def save_checkpoint(self, checkpoint_dir, state):
        if checkpoint_dir is None:
            logger.info("No checkpoint dir. specified. Skipping checkpoint.")
            return
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint_" + str(state["epoch"]))
        logger.info(f"Saving checkpoint {checkpoint}")
        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)
        return checkpoint

    # -----------------------
    def solve(self):
        start_time = time.time()
        if self.multi_gpu:
            n_devices = jax.local_device_count()
            self.DD = data_management.DatasetDict(
                batch_size=self.batch_size,
                x_data=self.train_points,
                num_gpus=n_devices,
            )
            (
                self.opt_state,
                self.params,
                self.epoch_store,
                self.loss_epochs,
            ) = self.multi_GPU_train(self.opt_state, self.params)
        else:
            self.DD = data_management.DatasetDict(
                batch_size=self.batch_size,
                x_data=self.train_points,
            )
            batched_training_data = self.DD.get_batched_data()
            (
                self.opt_state,
                self.params,
                self.epoch_store,
                self.loss_epochs,
            ) = self.single_GPU_train(self.opt_state, self.params, batched_training_data)

        end_time = time.time()
        logger.info(f"solve took {end_time - start_time} (sec)")

        state = {
            "opt_state": self.opt_state,
            "params": self.params,
            "epoch": self.epoch_store[-1] + 1,
            "batch_size": self.batch_size,
            "resolution": f"{self.Nx_tr}, {self.Ny_tr}, {self.Nz_tr}",
        }
        self.save_checkpoint(self.checkpoint_dir, state)

        plot_loss_epochs(
            self.epoch_store,
            self.loss_epochs,
            self.results_dir,
            self.TD.base_level,
            self.TD.alt_res,
            self.loss_plot_name,
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

    # -----------------------
    @partial(jit, static_argnums=(0))
    def single_GPU_train(self, opt_state, params, batched_training_data):
        # batched_training_data = self.DD.get_batched_data()
        update_fn = self.model.update
        num_batches = batched_training_data.shape[0]

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
            # train_dx, train_dy, train_dz = self.TD.alternate_res(epoch, train_dx, train_dy, train_dz) #TODO: automate this
            train_dx, train_dy, train_dz = self.TD.alternate_res_sequentially(
                self.num_epochs, epoch, train_dx, train_dy, train_dz
            )
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
            loss_epoch = loss_epoch / num_batches
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            loss_epoch = progress_bar((epoch, self.num_epochs, self.print_rate, loss_epoch), loss_epoch)
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
        # self.loss_epochs /= num_batches
        return opt_state, params, self.epoch_store, self.loss_epochs

    # ----------------------------------------------------------------

    def _multi_GPU_train(self, opt_state, params):
        """TODO: This is under development.

        Tensor Model Parallel Training using Positional Sharding

        Args:
            opt_state (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """

        from jax.experimental import mesh_utils
        from jax.experimental.maps import xmap
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P
        from jax.sharding import PositionalSharding

        n_devices = jax.local_device_count()
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, axis_names=("devices",))

        sharding = PositionalSharding(devices)

        batched_training_data = self.DD.get_batched_data()
        batched_training_data = jax.device_put(batched_training_data, sharding.reshape(n_devices, 1, 1, 1))
        num_batches = batched_training_data.shape[1]

        # -----
        update_fn = self.model.update_multi_gpu

        def learn_one_batch(carry, data_batch):
            opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
            opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
            loss_epoch += loss_epoch_
            return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

        def learn_one_epoch(
            epoch,
            opt_state,
            params,
            loss_epochs,
            train_dx,
            train_dy,
            train_dz,
            batched_training_data,
        ):
            # train_dx, train_dy, train_dz = self.TD.alternate_res(epoch, train_dx, train_dy, train_dz) #TODO: automate this
            train_dx, train_dy, train_dz = self.TD.alternate_res_sequentially(
                self.num_epochs, epoch, train_dx, train_dy, train_dz
            )
            # batched_training_data = random.permutation(key, batched_training_data, axis=1)
            # TODO: circular shift to the left: lhs = jax.lax.ppermute(lhs, axis_name='i', perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
            loss_epoch = 0.0
            for batch in range(num_batches):
                (
                    opt_state,
                    params,
                    loss_epoch,
                    train_dx,
                    train_dy,
                    train_dz,
                ), _ = learn_one_batch(
                    (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), batched_training_data[batch]
                )
            loss_epoch = loss_epoch / num_batches
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            return (
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ), None

        def per_device_update(opt_state, params, per_device_data, train_dx, train_dy, train_dz):
            batched_training_data = per_device_data.squeeze(0)
            for epoch in self.epoch_store:
                (
                    opt_state,
                    params,
                    self.loss_epochs,
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                    batched_training_data,
                ), _ = learn_one_epoch(
                    epoch,
                    opt_state,
                    params,
                    self.loss_epochs,
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                    batched_training_data,
                )
                loss_epoch = self.loss_epochs[epoch]
                # loss_epoch = progress_bar((epoch, self.num_epochs, self.print_rate, loss_epoch), loss_epoch)
            return opt_state, params, self.epoch_store, self.loss_epochs

        mgpu_train = jit(
            shard_map(
                per_device_update,
                mesh=mesh,
                in_specs=(P(), P(), P("devices", None, None, None), P(), P(), P()),
                out_specs=(P(), P(), P(), P()),
                check_rep=False,
            )
        )
        opt_state, params, self.epoch_store, self.loss_epochs = mgpu_train(
            opt_state, params, batched_training_data, self.train_dx, self.train_dy, self.train_dz
        )
        return opt_state, params, self.epoch_store, self.loss_epochs

    # -----------------------

    def multi_GPU_train(self, opt_state, params):
        """Data Parallel Training using pmap

        Args:
            opt_state (_type_): state of optimizer
            params (_type_): parameters of model

        Returns:
            _type_: optimized state, epochs, losses
        """
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
            self.model.update_multi_gpu,
            in_axes=(0, 0, 0, 0, 0, 0),
            axis_name="devices",
        )
        num_batches = batched_training_data.shape[1]
        t0 = time.time()
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
            if epoch % self.print_rate == 0:
                dt_avg = (time.time() - t0) / self.print_rate
                t0 = time.time()
                logger.info(
                    f"Epoch # {epoch} loss is {jnp.mean(loss_epoch) / num_batches} \t avg epoch time is {dt_avg} (sec)"
                )
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
                self.save_checkpoint(self.checkpoint_dir, state)
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))
        return opt_state, params, epoch_store, loss_epochs

    # -----------------------
    @partial(jit, static_argnums=(0))
    def evaluate_solution_and_gradients(self, params, eval_gstate):
        final_solution = self.model.evaluate_solution_fn(
            params,
            eval_gstate.R,
        ).reshape(-1)
        grad_u_normal_to_interface = self.model.compute_normal_gradient_solution_on_interface(
            params,
            eval_gstate.R,
            eval_gstate.dx,
            eval_gstate.dy,
            eval_gstate.dz,
        )
        grad_u = self.model.compute_gradient_solution(
            params,
            eval_gstate.R,
        )
        return final_solution, grad_u, grad_u_normal_to_interface


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
) -> InitFn:
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
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_m(x):
            return 0.0

    if nonlinear_op_p is None:
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_p(x):
            return 0.0

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

    def init_fn(
        gstate: GridState,
        eval_gstate: GridState,
        Nx_tr: int = 32,
        Ny_tr: int = 32,
        Nz_tr: int = 32,
        num_epochs: int = 1000,
        batch_size: int = 131072,
        algorithm: int = 0,
        switching_interval: int = 3,
        multi_gpu: bool = False,
        checkpoint_interval: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        results_dir: str = "./",
        loss_plot_name: str = "solver_loss",
        optimizer: GradientTransformation = optax.adam(1e-2),
        restart: bool = False,
        print_rate: int = 1,
    ) -> Tuple[PoissonSimState, SolveFn]:
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

        def solve_fn(sim_state: PoissonSimState) -> PoissonSimState:
            trainer = Trainer(
                gstate,
                eval_gstate,
                sim_state,
                sim_state_fn,
                algorithm,
                switching_interval=switching_interval,
                Nx_tr=Nx_tr,
                Ny_tr=Ny_tr,
                Nz_tr=Nz_tr,
                lvl_set_fn=lvl_set_fn,
                num_epochs=num_epochs,
                multi_gpu=multi_gpu,
                batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                results_dir=results_dir,
                loss_plot_name=loss_plot_name,
                optimizer=optimizer,
                restart=restart,
                print_rate=print_rate,
            )
            (
                final_solution,
                grad_u,
                grad_u_normal_to_interface,
                epoch_store,
                loss_epochs,
            ) = trainer.solve()
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
            PoissonSimState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None),
            solve_fn,
        )

    return init_fn
