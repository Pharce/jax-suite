import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import value_and_grad



def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)



if __name__ == '__main__':

    # building a toy dataset
    inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])
    targets = jnp.array([True, True, False, True])

    # define loss function
    def loss(W, b):
        preds = predict(W, b, inputs)
        label_probabilites = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.sum(jnp.log(label_probabilites))

    # initialize keys
    key = random.PRNGKey(0)
    key, W_key, b_key = random.split(key, 3)
    W = random.normal(W_key, (3,))
    b = random.normal(b_key, ())

    # take gradients
    W_grad = grad(loss, argnums=0)(W, b)
    b_grad = grad(loss, argnums=1)(W, b)

    print('W grad', W_grad)
    print('b_grad', b_grad)

    # unpack as a tuple
    W_grad, b_grad = grad(loss, argnums=(0,1))(W, b)

    def loss2(params_dictionary):
        preds = predict(params_dictionary['W'], params_dictionary['b'], inputs)
        label_probabilites = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.sum(jnp.log(label_probabilites))
    
    logit_dictionary = {'W': W, 'b': b}
    print('Grad dictionary', grad(loss2)(logit_dictionary))

    loss_value, Wb_grad = value_and_grad(loss, (0, 1))(W, b)
    print('Loss value', loss_value)
    print('Loss value', loss(W, b))