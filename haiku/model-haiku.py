from numpy.core.fromnumeric import shape
import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax

class MyLinear(hk.Module):

    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size
    
    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))

        w = hk.get_parameter("w", shape=[j, k], dtype = x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[k], dtype = x.dtype, init=jnp.zeros)

        return jnp.dot(x, w) + b
    
def forward_fn(x):
    # use output_size of 10. 
    model = MyLinear(10)
    return model(x)

class MyDropout(hk.Module):

    def __init__(self, rate = 0.5, name=None):
        super().__init__(name=name)
        self.rate = rate
    
    def __call__(self, x):
        key = hk.next_rng_key()
        p = jax.random.bernoulli(key, 1.0 - self.rate, shape=x.shape)
        return x * p / (1.0 - self.rate)


if __name__ == '__main__':
    # transform into haiku function
    forward = hk.transform(forward_fn)

    # initialize x
    x = jnp.ones([1, 1])

    # get randome key
    key = hk.PRNGSequence(42)
    params = forward.init(next(key), x)

    y = forward.apply(params, None, x)

    # using a stochastic model. 
    forward2 = hk.transform(lambda x: MyDropout()(x))

    key1, key2 = jax.random.split(jax.random.PRNGKey(42), 2)
    params2 = forward2.init(key1, x)
    prediction = forward2.apply(params, key2, x)