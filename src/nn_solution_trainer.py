import haiku as hk
import jax
from jax import (numpy as jnp, vmap, jit, random, value_and_grad)
from functools import partial
import optax
import pdb
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from jax import config
config.update("jax_debug_nans", True)

from src.nn_solution_model import DoubleMLP




class Trainer:
    def __init__(self, compute_Ax_and_b_fn, grid_shape, optimizer=optax.adam):
        self.optimizer = optimizer
        self.compute_Ax_and_b_fn = compute_Ax_and_b_fn
        self.grid_shape = grid_shape
   

        
    @staticmethod
    @hk.transform
    def forward(x, phi_x):
        """
        Forward pass of the neural network.

        Args:
            x: input data

        Returns:
            output of the neural network
        """
        model = DoubleMLP()
        return model(x, phi_x)

    @partial(jit, static_argnums=0)
    def init(self, seed=42):
        rng = random.PRNGKey(seed)
        params = self.forward.init(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=0.1)  
        opt_state = self.optimizer.init(params)
        return opt_state, params


    def evaluation_fn(self, params):
        sol_fn =  partial(self.forward.apply, params, None)
        return sol_fn


    @partial(jit, static_argnums=(0))
    def weighted_loss_fn(self, phi_flat, lhs, rhs, sol_cube, dirichlet_cube, Vol_cell_nominal):
        """
            Weighted L2 loss with exp(-\phi^2) to emphasize error around boundaries
        """
        weight = jnp.exp(-1.0*jnp.square(phi_flat))
        tot_loss = jnp.mean(weight * optax.l2_loss(lhs, rhs))             
        tot_loss += jnp.square(sol_cube[ 0, :, :] - dirichlet_cube[ 0, :, :]).mean() * Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[-1, :, :] - dirichlet_cube[-1, :, :]).mean() * Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[: , 0, :] - dirichlet_cube[ :, 0, :]).mean() * Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[: ,-1, :] - dirichlet_cube[ :,-1, :]).mean() * Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[: , :, 0] - dirichlet_cube[ :, :, 0]).mean() * Vol_cell_nominal
        tot_loss += jnp.square(sol_cube[: , :,-1] - dirichlet_cube[ :, :,-1]).mean() * Vol_cell_nominal

        return tot_loss



    def loss(self, params, R_flat, phi_flat, dirichlet_cube, Vol_cell_nominal):
        """
            Loss function of the neural network
        """        
        sol_fn =  partial(self.forward.apply, params, None)
        pred_sol = vmap(sol_fn, (0,0))(R_flat, phi_flat)
        lhs_rhs = self.compute_Ax_and_b_fn(pred_sol)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        sol_cube = pred_sol.reshape(self.grid_shape)      
        tot_loss = self.weighted_loss_fn( phi_flat, lhs, rhs, sol_cube, dirichlet_cube, Vol_cell_nominal)
        return tot_loss


    def update(self, opt_state, params, R_flat, phi_flat, dirichlet_cube, Vol_cell_nominal):      
        loss, grads = value_and_grad(self.loss)(params, R_flat, phi_flat, dirichlet_cube, Vol_cell_nominal)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss






def train(optimizer, compute_Ax_and_b_fn, R_flat, phi_flat, grid_shape, dirichlet_cube, Vol_cell_nominal, num_epochs=10000):
    """
    Train the neural network.

    Args:
        optimizer: optimizer of the neural network
        config: Input parameters

    Returns:
        trained parameters of the neural network
    """

    trainer = Trainer(compute_Ax_and_b_fn, grid_shape, optimizer)
    
    opt_state, params = trainer.init()

    print('\n')
    print('Architecture Summary (trainable parameters):')
    
    num_params = 0
    for pytree in params:
        leaves = jax.tree_leaves(pytree)
        cur_shape = jax.tree_map(lambda x: x.shape, params[leaves[0]])
        print(f"{repr(pytree):<45} \t has trainable parameters:\t {cur_shape}")
        shapes = [val for key, val in cur_shape.items()]
        for val in shapes:
            res = 1
            for elem in val:
                res *= elem
            num_params += res
    
    print('\n')
    print(f"Total number of trainable parameters = {num_params} ...")
    print('\n')
    
    #----------------------------------------------------------------------    
    
    start_time = time.time()
    loss_epochs = []
    epoch_store = []
    for epoch in range(num_epochs):            
        opt_state, params, loss_epoch = trainer.update(opt_state, params, R_flat, phi_flat, dirichlet_cube, Vol_cell_nominal)            
        print(f"epoch # {epoch} loss is {loss_epoch}")
        loss_epochs.append(loss_epoch)
        epoch_store.append(epoch)


    end_time = time.time()
    print(f"solve took {end_time - start_time} (sec)")

    plt.figure(figsize=(8, 8))
    plt.plot(epoch_store, loss_epochs)
    plt.yscale('log')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.savefig('tests/poisson_solver_loss.png')
    plt.close()

    sol_fn = trainer.evaluation_fn(params)
    solution = vmap(sol_fn, (0,0))(R_flat, phi_flat)
    return solution.reshape(-1)
