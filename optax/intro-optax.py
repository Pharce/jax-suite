import optax
from optax import scale_by_rms, chain, clip_by_global_norm, scale_by_adam, scale, polynomial_schedule, scale_by_schedule, scale_and_decay
from optax import pathwise_jacobians
from optax import utils
import jax.numpy as jnp

def gradient_transforms(grads, params):
    """Transformations are composed of two functions: init, and update.
    Init intiailizes and empty set of statistics and update transforms a candidate gradient given statistics and'
    current parameter values. 

    Args:
        grads ([jnp]): gradients
        params ([jnp]): parameters
    """
    tx = scale_by_rms()
    state = tx.init(params)
    grads, state = tx.update(grads, state, params)

def combine(max_norm, learning_rate):
    chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(eps=1e-4),
        scale(-learning_rate)
    )


def schedule(learning_rate, max_norm):
    """Schedules contain time dependence of optimizations

    Args:
        learning_rate ([jnp.float32]): learning rate
        max_norm ([jnp.float32]): max normalization

    Returns:
        optimizer: chained optimizations
    """
    schedule_fn = polynomial_schedule(
        init_value=1., end_value=0., power=1, transition_steps=5
    )

    for step_count in range(6):
        print(schedule_fn(step_count)) # [1., 0.8, 0.6, 0.4, 0.2, 0.]
    
    schedule_fn = polynomial_schedule(
        init_value=-learning_rate, end_value=0., power=1, transition_steps=5
    )

    optimiser = chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(eps=1e-4),
        scale_by_schedule(schedule_fn)
    )

    return optimiser

def adamw(learning_rate, b1, b2, eps, weight_decay):
    """AdamW optimizer

    Args:
        learning_rate (float32): learning rate
        b1 (float32): mean weight
        b2 (float32): variance weight
        eps (float32): variance
        weight_decay (float32): decay of weights for learning rate

    Returns:
        optax.chain: optimizer with AdamW properties
    """
    optimizer = chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps),
        scale_and_decay(-learning_rate, weight_decay=weight_decay)
    )
    return optimizer

def apply_weights(grads, state, params, tx):
    """Change weights based on an optimizer

    Args:
        grads (jnp.float32): gradients
        state (jnp.float32): state of parameters
        params (jnp.float32): model parameters
        tx (optax.transform): transformations. 
    """
    updates, states = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

def mc_estimator(mean, log_scale, optim, weights, optim_state, rng, num_samples):
    """Montecarlo estimator of pathwise jacobians. Args are all hyperparameters. 

    Args:
        mean ([type]): [description]
        log_scale ([type]): [description]
        optim ([type]): [description]
        weights ([type]): [description]
        optim_state ([type]): [description]
        rng ([type]): [description]
        num_samples ([type]): [description]
    """
    dist_params = [mean, log_scale]
    function = lambda x: jnp.sum(x * weights)
    jacobians = pathwise_jacobians(
        function, dist_params,
        utils.multi_normal, rng, num_samples
    )

    mean_grads = jnp.mean(jacobians[0], axis=0)
    log_scale_grads = jnp.mean(jacobians[1], axis=1)
    grads = [mean_grads, log_scale_grads]
    optim_update, optim_state = optim.update(grads, optim_state)
    updated_dist_params = optax.apply_updates(dist_params, optim_update)
    
    return updated_dist_params
