import haiku as hk
import jax.numpy as jnp
import jax

def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(images, labels):
    mlp = hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(10),
    ])

    logits = mlp(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

def sgd(param, update):
    return param - 0.01 * update



if __name__ == '__main__':

    # converts into pure functions init, apply. 
    loss_fn_t = hk.transform(loss_fn)           # or transform_with_state if update. 
    loss_fn_t = hk.without_apply_rng(loss_fn_t) # deterministic post learning

    # key to set random numbers (required)
    rng = jax.random.PRNGKey(42)

    # pass in dummy
    images, labels = next(input_dataset)
    params = loss_fn_t.init(rng, images, labels)

    # get loss and grads
    loss = loss_fn_t.apply(params, images, labels)
    grads = jax.grad(loss_fn_t.apply)(params, images, labels)

    for images, labels in input_dataset:
        grads = jax.grad(loss_fn_t.apply)(params, images, labels)
        params = jax.tree_multimap(sgd, params, grads)