import contextlib
import functools
from typing import Iterable, Mapping, Tuple

from absl import app, flags, logging
from jax.interpreters.masking import is_tracing
from jax.lib.xla_bridge import local_device_count
import haiku as hk
from examples.imagenet import dataset

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tree


# Types
OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[str, jnp.ndarray]

def _forward(
    batch: dataset.Batch,
    is_training: bool,
) -> jnp.ndarray:
    """Forward appliaction of resnet.

    Args:
        batch (dataset.Batch): batch of images
        is_training (bool): True, if training. 

    Returns:
        jnp.ndarray: array of logits
    """

    net = hk.nets.ResNet50(1000,
        resnet_v2=True,
        bn_config={'decay_rate': 0.9}
        )
    
    return net(batch['images'], is_training=is_training)

forward = hk.transform_with_state(_forward)


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
    """Linear scaling rule optimized for 90 epochs

    Args:
        step (jnp.ndarray): step number

    Returns:
        jnp.ndarray: learning rate
    """

    train_split = dataset.Split.from_string('TRAIN_AND_VALID')

    # get batch size
    total_batch_size = 128 * jax.device_count()
    steps_per_epoch = train_split.num_examples / total_batch_size

    current_epoch = step / steps_per_epoch # type:float
    lr = (0.1 * total_batch_size) / 256
    lr_linear_till = 5
    boundaries = jnp.array((30, 60, 90)) * steps_per_epoch
    values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

    # get index and learning rate
    index   = jnp.sum(boundaries < step)
    lr      = jnp.take(values, index)

    return lr * jnp.minimum(1., current_epoch / lr_linear_till)

def make_optimizer():
    """SGD with nesterov momentum and custom lr schedule
    """

    return optax.chain(
        optax.trace(
            decay=0.9,
            nesterov=True
        )
    )

def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
    """L2 Loss

    Args:
        params (Iterable[jnp.ndarray]): Parameters of a neural network

    Returns:
        jnp.ndarray: loss function
    """
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def softmax_cross_entropy(
    *,
    logits: jnp.ndarray,
    labels: jnp.ndarray) -> jnp.ndarray:
    """Softmax Cross Entropy

    Args:
        logits (jnp.ndarray): log predictions
        labels (jnp.ndarray): labels

    Returns:
        jnp.ndarray: softmax cross entropy. 
    """
    
    return - jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)

def smooth_labels(
    labels: jnp.ndarray,
    smoothing: jnp.ndarray) -> jnp.ndarray:
    """Smooth labels

    Args:
        labels (jnp.ndarray): array of labels
        smoothing (jnp.ndarray): smoothing factors

    Returns:
        jnp.ndarray: array of smoothed labels
    """

    smooth_positives = 1. - smoothing
    smooth_negatives = smoothing / 1000

    return smooth_positives * labels + smooth_negatives

def loss_fn(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch) -> Tuple[jnp.ndarray, hk.State]:
    """Computes regularized loss

    Args:
        params (hk.Params): network parameters
        state (hk.State): optimizer state
        batch (dataset.Batch): dataset batch
    Returns:
        Tuple[jnp.ndarray, hk.State]: loss, new state
    """

    logits, state = forward.apply(params, state, None, batch, is_training=True)
    labels = jax.nn.one_hot(batch['labels'], 1000)

    labels = smooth_labels(labels, smoothing=0.1)

    cat_loss = jnp.mean(softmax_cross_entropy(logits, labels))
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
        if 'batchnorm' not in mod_name]
        
    reg_loss = 0.9 * l2_loss(l2_params)
    loss = cat_loss + reg_loss
    return loss, state


@functools.partial(jax.pmap, axis_name='i', donate_argnums=(0, 1, 2))
def train_step(
    params: hk.Params,
    state: hk.State,
    opt_state : OptState,
    batch: dataset.Batch
) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies update to parameters and returns new state

    Args:
        params (hk.Params): neural network parameters
        state (hk.State): current state
        batch (dataset.Batch): current batch
        opt_state (OptState): State of optimizer

    Returns:
        Tuple[hk.Params, hk.State, OptState, Scalars]: new parameters, state, optimizer, scalars
    """

    (loss, state), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch)
    )

    # taking mean across all replicas
    grads = jax.lax.pmean(grads, axis_name='i')

    # Compute and apply updates via optimizer
    updates, opt_state = make_optimizer().update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Scalars to log
    scalars = {'train_loss' : loss}
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

def make_initial_state(
    rng: jnp.ndarray,
    batch: dataset.Batch
) -> Tuple[hk.Params, hk.State, OptState]:
    """Makes initial state

    Args:
        rng (jnp.ndarray): = seed
        batch (dataset.Batch): batch
    Returns:
        Tuple[hk.Params, hk.State, OptState]: initial params, state, optimizer
    """

    params, state   = forward.init(rng, batch, is_training=True)
    opt_state       = make_optimizer().init(params)

    return params, state, opt_state

@jax.jit
def eval_batch(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch
) -> jnp.ndarray:
    """Evaluates a batch

    Args:
        params (hk.Params): NN parameters
        state (hk.State): NN state
        batch (dataset.Batch): current batch

    Returns:
        jnp.ndarray: array to evaluate
    """

    logits, _ = forward.apply(params, state, None, batch, is_training=False)
    predicted_label = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(jnp.equal(predicted_label, batch['labels']))
    return correct.astype(jnp.float32)

def evaluate(
    split: dataset.Split,
    params: hk.Params,
    state: hk.State
) -> Scalars:
    """Evaluates model at given params state

    Args:
        split (dataset.Split): dataset split
        params (hk.Params): parameters
        state (hk.State): state

    Returns:
        Scalars: scalars
    """

    params, state   = jax.tree_map(lambda x: x[0], (params, state))
    test_dataset    = dataset.load(split,
        is_training=False,
        batch_dims=[1000])

    correct = jnp.array(0)
    total = 0
    for batch in test_dataset:
        correct += eval_batch(params, state, batch)
        total   += batch['images'].shape[0]
    
    return {'top_1_acc' : correct.item() / total}

@contextlib.contextmanager
def time_activity(activity_name: str):
    logging.info('[Timing] %s start.', activity_name)
    yield
    logging.info('[Timing] %s finished.', activity_name)

def main(argv):

    train_split = dataset.Split.from_string('TRAIN')
    eval_split  = dataset.Split.from_string('TEST')

    total_train_batch_size = 128 * jax.device_count()
    num_train_steps = (train_split.num_examples * 90) // total_train_batch_size

    local_device_count  = jax.local_device_count()
    train_dataset       = dataset.load(
        train_split,
        is_training=True,
        batch_dims=[local_device_count, 128]
    ) 

    rng = jax.random.PRNGKey(0)
    rng = jnp.broadcast_to(rng, (local_device_count, ) + rng.shape)

    # intialize with exampel input
    batch = next(train_dataset)
    params, state, opt_state = jax.pmap(make_initial_state)(rng, batch)

    eval_every = -1
    log_every = 1000

    with time_activity('train'):
        for step_num in range(num_train_steps):

            # take a single training step
            with jax.profiler.StepTraceContext('train', step_num=step_num):
                params, state, opt_state, train_scalars = (
                    train_step(params, state, opt_state, next(train_dataset))
                )

                if step_num and step_num % log_every == 0:
                    train_scalars = jax.tree_map(
                        lambda v: np.mean(v).item(),
                        jax.device_get(train_scalars)
                    )

                    logging.info('[Train %s/%s] %s',
                        step_num,
                        num_train_steps,
                        train_scalars
                    )
    
    with time_activity('final eval'):
        eval_scalars = evaluate(eval_split, params, state)
    logging.info('[Eval FINAL] %s', eval_scalars)

if __name__ == '__main__':
    app.run(main)

