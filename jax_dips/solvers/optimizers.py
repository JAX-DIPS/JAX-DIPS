import logging

import optax
from optax._src.base import GradientTransformation

import jaxopt
from jaxopt import LBFGS, LevenbergMarquardt
from jaxopt.loss import huber_loss

logger = logging.getLogger(__name__)


def get_scheduler(
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    transition_steps: int = 1000,
    **kwargs,
):
    if scheduler_name == "exponential":
        logger.info("Using Exponential Scheduler")
        scheduler = optax.exponential_decay(
            init_value=learning_rate, transition_steps=transition_steps, decay_rate=decay_rate, **kwargs
        )
    elif scheduler_name == "polynomial":
        logger.info("Using Polynomial Scheduler")
        scheduler = optax.polynomial_schedule(
            init_value=learning_rate, end_value=0.0, power=1, transition_steps=transition_steps, **kwargs
        )
    return scheduler


def chained_adam(
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    transition_steps: int = 1000,
    max_norm: float = 1.0,
    **kwargs,
) -> GradientTransformation:
    scheduler = get_scheduler(
        scheduler_name=scheduler_name,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
        **kwargs,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )
    return optimizer


def get_optimizer(
    optimizer_name: str = "custom",
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    max_norm: float = 1.0,
    loss_fn: object = None,
    **kwargs,
) -> GradientTransformation:
    if optimizer_name == "custom":
        logger.info("Using chained Adam optimizer")
        return chained_adam(
            scheduler_name=scheduler_name,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            max_norm=max_norm,
            **kwargs,
        )

    elif optimizer_name == "adam":
        logger.info("Using Adam optimizer")
        return optax.adam(
            learning_rate,
            **kwargs,
        )

    elif optimizer_name == "rmsprop":
        logger.info("Using RMSprop optimizer")
        return optax.rmsprop(
            learning_rate,
            **kwargs,
        )

    # elif optimizer_name == "lbfgs":
    #     logger.info("Using jaxopt.LBFGS optimizer")
    #     lbfgs = LBFGS_adapter(loss_fn)
    #     return lbfgs

    else:
        logger.error("Unknown optimizer: {}".format(optimizer_name))
        raise ValueError("Unknown optimizer: {}".format(optimizer_name))


# class LBFGS_adapter(LBFGS):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def init(self, params, *args, **kwargs):
#         params, opt_state = self.init_state(params, *args, **kwargs)
#         return opt_state
