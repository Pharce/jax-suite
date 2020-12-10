import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def forward(x, is_training):
    net = hk.nets.ResNet50(1000)
    return net(x, is_training)

def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(inputs, labels):
    logits = hk.nets.MLP([8, 4 , 2])(inputs)
    return jnp.mean(softmax_cross_entropy(logits, labels))


if __name__ == '__main__':

    # forward function
    forward = hk.transform_with_state(forward)
    
    # rng, x
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1,1])

    # initialized with parameters and state
    params, state = forward.init(rng, x, is_training=True)

    # apply function takes both parameters and state. 
    logits, state = forward.apply(params, state, rng, x, is_training=True)


    loss_fn_t = hk.transform(loss_fn)
    loss_fn_t = hk.without_apply_rng(loss_fn_t)

    rng = jax.random.PRNGKey(428)
    sample_image, sample_label = next(input_dataset)
    params = loss_fn_t.init(rng, sample_image, sample_label)

    num_devices = jax.local_device_count()
    params = jax.tree_util.tree_map(lambda x: np.stack([x] * num_devices), params)

    def make_superbatch():

        superbatch = [next(input_dataset) for _ in range(num_devices)]
        superbatch_images, superbatch_labels = zip(*superbatch)

        superbatch_images = np.stack(superbatch_images)
        superbatch_labels = np.stack(superbatch_labels)
        return superbatch_images, superbatch_labels

    def sgd(params, grads, alpha):
        return params - grads * alpha


    def update(params, inputs, labels, axis_name='i'):
        grads = jax.grad(loss_fn_t.apply)(params, inputs, labels)
        grads = jax.lax.pmean(grads, axis_name)
        new_params = sgd(params, grads)
        return new_params
    
    for _ in range(10):
        superbatch_images, superbatch_labels = make_superbatch()
        params = jax.pmap(update, axis_name='i')(params, superbatch_images, superbatch_labels)

