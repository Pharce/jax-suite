import os
import time

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random


from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus
from examples import datasets


def gaussian_kl(mu, sigmasq):
    """KL divergence from a diaganol Gaussian to the standard Gaussian. 

    Args:
        mu (jnp.float32): mean
        sigmasq (jnp.float32): variance
    """
    return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu **2. - sigmasq)

def gaussian_sample(rng, mu, sigmasq):
    """Sample a diagonal Gaussian.

    Args:
        rng (key): PRNGKey from haiku
        mu (jnp.float32): mean
        sigmasq (jnp.float32): standard deviation
    """
    return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)

def bernouilli_logpdf(logits, x):
    """Bernouilli log density of data x given logits P(x | logits)

    Args:
        logits ([jnp.float32]): List of log probabilities
        x ([jnp.loat32]): 
    """

    return - jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))

def elbo(rng, params, images):
    """Mounte Carlo estimate of the negative evidence lower bound.

    Args:
        rng (PRNGKey): random key generator 
        params (tuple): encoder, decoder parameters
        images (jnp.float32): float32 containing images
    """

    enc_params, dec_params = params             # split the tuple
    mu_z, sigmasq_z = encode(enc_params, images) # encode images with encoder parameters

    # get logits from the decoder and sampled from the encoded mean and standard deviation
    logits_x = decode(dec_params, gaussian_sample(rng, mu_z, sigmasq_z)) 

    # ELBO equation
    return bernouilli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)


def image_sample(rng, params, nrow, ncol):
    """Sample images from the decoder model

    Args:
        rng (PRNGKey): [description]
        params (tuple): [description]
        nrow (int): rows of images
        ncol (int): cols of images
    """
    _, dec_params       = params              # get the decoder parameters
    code_rng, img_rng   = random.split(rng)   # split the rng key

    # generate logits from a decoder using a randomly sampled nrow * ncol
    logits          = decode(dec_params, random.normal(code_rng, (nrow * ncol, 10)))

    # generate images via random sampling
    sampled_images  = random.bernouilli(img_rng, jnp.logaddexp(0., logits))

    return image_grid(nrow, ncol, sampled_images, (28, 28))

def image_grid(nrow, ncol, imagevecs, imshape):
    """Reshape a stack of image vectors into an image grid for plotting.

    Args:
        nrow (int): nrows of images
        ncol (int): n col of images
        imagevecs (list): list of images
        imshape (tuple): image shape
    """

    # reshape imagevecs into n-images x image-shape
    images = iter(imagevecs.reshape(-1) + imshape)
    return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
        for _ in range(nrow)]).T

encoder_init, encode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,

    # split into 2 
    FanOut(2),

    # generate means and standard deviations
    stax.parallel(Dense(10, stax.serial(Dense(10), Softplus)))
)

decoder_init, decode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    Dense(28*28)
)

if __name__ == '__main__':
    
    # hyperparameters
    step_size = 0.001
    num_epochs = 100
    batch_size = 32
    nrow, ncol = 10

    # PRNGEKey
    test_rng = random.PRNGKey(1)

    # get and batch dataset
    train_images, _, test_images, _ = datasets.mnist(permute_train=True)
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)

    # Initialize encoder and decoder parameters. 
    enc_init_rng, dec_init_rng = random.split(random.PRNGKey(2))
    _, init_encoder_params = encoder_init(enc_init_rng, (batch_size, 28*28))
    _, init_decoder_params = decoder_init(dec_init_rng, (batch_size, 10))
    init_params = init_encoder_params, init_decoder_params

    # Create optimizer
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)

    # train, test images
    train_images = jax.device_put(train_images)
    test_images  = jax.device_put(test_images)

    # binarize batches
    def binarize_batch(rng, i, images):
        i = i % num_batches 
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        return random.bernouilli(rng, batch)

    # run epochs
    @jit
    def run_epoch(rng, opt_state, images):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))           # split RNG
            batch = binarize_batch(data_rng, i, images)                         # binarize each batch
            loss = lambda params: -elbo(elbo_rng, params, batch) / batch_size   # get -elbo / batch size
            g = grad(loss)(get_params(opt_state))                               # get grads
            return opt_update(i, g, opt_state)                                  # update
        
        return lax.fori_loop(0, num_batches, body_fun, opt_state)               # iterate over batches

    @jit
    def evaluate(opt_state, images):
        params = get_params(opt_state)
        elbo_rng, data_rng, image_rng = random.split(test_rng,3)
        binarized_test = random.bernouilli(data_rng, images)
        test_elbo = elbo(elbo_rng, params, binarized_test) / images.shape[0]
        sampled_images = image_sample(image_rng, params, nrow, ncol)
        return test_elbo, sampled_images
    
    opt_state = opt_init(init_params)
    for epoch in range(num_epochs):
        tic = time.time()
        opt_state = run_epoch(random.PRNGKey(epoch), opt_state, train_images)
        test_elbo, sampled_images = evaluate(opt_state, test_images)
        print(f"Epoch {epoch}, Elbo {test_elbo}, Time - {time.time() - tic}")

    








