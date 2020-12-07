import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random 
from jax import device_put
from jax import jacfwd, jacrev
import timeit   
import numpy as np

def selu(x, alpha=1.67, lmbda=1.05):
    """Se Linear Unit

    Args:
        x ([type]): [input]
        alpha (float, optional): [hyperparameter]. Defaults to 1.67.
        lmbda (float, optional): [hyperparameter]. Defaults to 1.05.

    Returns:
        [x_activated]: [activated x]
    """
    # jnp.where returns lmbda * x when x > 0, else alpha * jnp.exp(x) - alpha
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def sum_logistic(x):

    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def finite_difference(f, x):
    eps = 1e-3
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

def hessian(fun):
    return jit(jacfwd(jacrev(fun)))

def apply_matrix(mat, v):
    return jnp.dot(mat, v)

def naive_batching(mat, v_batched):
    return jnp.stack([apply_matrix(mat, v) for v in v_batched])
    print('Naively batched')

def jax_batching(mat, v_batched):
    return jnp.dot(v_batched, mat.T)
    print('Jax batched')
    
def apply_matrix(v):
    return jnp.dot(mat, v)

def vmap_batching(v_batched):
    return vmap(apply_matrix)(v_batched)

if __name__ == '__main__':
    """Test commands from the JAX documentation
    """

    # For reproducbility, set the seed. 
    key = random.PRNGKey(0) # generate a random key 
    x = random.normal(key, (10,)) # normally distributed x variables. 
    # print(f'X  - {x}')


    # matrix multiplication, in 2 dimensions. 
    size = 30 # size of random number vector
    x = random.normal(key, (size, size), dtype=jnp.float32) # generate 30 x 30 random numbers
    # print("JNP ", jnp.dot(x, x.T)) # print the matrix multiplication. 

    # works on numpy matrices as well
    x = np.random.normal(size=(size, size)).astype(np.float32)
    # print("JNP ", jnp.dot(x, x.T)) # print the matrix multiplication. 

    # store on GPU and return when necessary. 
    x = device_put(x)
    # print("JNP ", jnp.dot(x, x.T)) # print the matrix multiplication. 

    # we can use the activation on x
    x = random.normal(key, (10,))
    selu_x = selu(x)
    selu_jit = jit(selu)

    selu_jit_x = selu_jit(x)

    # print("X ", x)
    # print('Selu X', selu_x)
    # print('Selu jit X', selu_jit_x)

    x_small = jnp.arange(3.)
    derivative_fn = grad(sum_logistic)
    # print('X-small', x_small)
    # print('logistic x-small', sum_logistic(x_small))
    # print('Derivative of X-small', derivative_fn(x_small))
    # print('Finite Difference of X-small', finite_difference(sum_logistic, x_small))

    # very easy to take gradients, just type grad. jit to speed up. 
    # print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

    # Take hessian of a function 
    # print('Hessian', hessian(sum_logistic))


    # Vectorization with vmap
    mat = random.normal(key, (15, 10))
    batched_x = random.normal(key, (1, 10))

    def apply_matrix(v):
        return jnp.dot(mat, v)

    def naive_batching(v_batched):
        return jnp.stack([apply_matrix(v) for v in v_batched])
        print('Naively batched')

    print('Naive batching', naive_batching(batched_x))


    def jax_batching(v_batched):
        return jnp.dot(v_batched, mat.T)
        print('Jax batched')
    
    print('JAX batching', jax_batching(batched_x))

    def vmap_batching(v_batched):
        return vmap(apply_matrix)(v_batched)

    print('VMAP batching', vmap_batching(batched_x))






    

