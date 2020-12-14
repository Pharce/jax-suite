import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import tensorflow_datasets as tfds
import time


## Hyper-parameters
def random_layer_params(m, n, key, scale=1e-2):
    """[summary]

    Args:
        m (int): Dimension 1
        n (int): Dimension 2
        key (jax.random.NPRGKey): jax random key
        scale ([type], optional): [description]. Defaults to 1e-2.

    Returns:
        (rand_w, rand_b): Randomly initialized weights
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    """Initialized network parameters. 

    Args:
        sizes ([list]): List of sizes of neural network layers
        key ([jax.random.NPRGKey]): Jax-based seeding

    Returns:
        [list]: vectorized neural network
    """
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]



def relu(x):
    """[summary]

    Args:
        x ([jnp.float32]): unactivated inputs

    Returns:
        relu_x [jnp.float32]: ReLU activated inputs
    """
    return jnp.maximum(0, x)

def predict(params, image):
    """[summary]

    Args:
        params ([list]): List of parameters
        image ([jnp.float32]): Image parameters. 

    Returns:
        predictions [jnp.float32]: predictions for a NN on an image
    """
    activations = image

    # generate outputs and activations
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    # get final values and take inverse logit to get predictions. 
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


def one_hot(x, k, dtype=jnp.float32):
    """One hot encoding labels

    Args:
        x ([jnp.float32]): Vector of labels
        k ([int]): dimension of encoding
        dtype ([type], optional): [description]. Defaults to jnp.float32.

    Returns:
        [x_onehot]: Vector of one_hot encoded labels. 
    """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
  
def accuracy(params, images, targets):
    """Calculates accuracy of a neural network given parameters, images, and targets. 

    Args:
        params ([list]): List of parameters.
        images ([jnp.float32]): Flattenend images
        targets ([jnp.float32]): One hot encoded targets

    Returns:
        accuracy (jnp.float32): accuracy of predictions. 
    """
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    """Calculate loss of neural network. 

    Args:
        params ([list]): List of parameters.
        images ([jnp.float32]): Flattenend images
        targets ([jnp.float32]): One hot encoded targets

    Returns:
        loss (jnp.float32): loss of predictions. 
    """
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    """Update parameters of neural network.

    Args:
        params ([list]): List of parameters.
        x ([list]): Inputs
        y ([list]): outputs

    Returns:
        params_updated [list]: Updated list of parameters. 
    """
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
if __name__ == '__main__':
    layer_sizes = [784, 512, 512, 10]
    param_scale = 0.1
    step_size = 0.01
    num_epochs = 10
    batch_size = 128
    n_targets = 10
    params = init_network_params(layer_sizes, random.PRNGKey(0))

    data_dir = '../data/tfds'

    mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    num_labels = info.features['label'].num_classes
    h, w, c = info.features['image'].shape
    num_pixels = h * w * c

    # Full train set
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot(train_labels, num_labels)

    # Full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot(test_labels, num_labels)

    batched_predict = vmap(predict, in_axes=(None, 0))


    def get_train_batches():
        # load batches as numpy values. 
        ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
        ds = ds.batch(batch_size).prefetch(1)
        return tfds.as_numpy(ds)

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in get_train_batches():
            x = jnp.reshape(x, (len(x), num_pixels))
            y = one_hot(y, num_labels)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
