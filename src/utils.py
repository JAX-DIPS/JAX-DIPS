import jax

def print_architecture(params):
        print('\n')
        print('Architecture Summary (trainable parameters):')

        num_params = 0
        for pytree in params:
            leaves = jax.tree_util.tree_leaves(pytree)
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
