import chex
import jax
import jax.numpy as jnp
from chex import assert_shape, assert_rank, assert_type, assert_equal_shape, assert_tree_all_close, assert_tree_all_finite, assert_numerical_grads, assert_devices_available, assert_tpu_available
from absl.testing import parameterized

def asserts(t1, t2, t3, t4, t5, x, y, z, tree_x, tree_y, f, j):

    # ensure that t1, t2, t3 are shaped the same
    chex.assert_equal_shape([t1, t2, t3])

    # assert that t4, t5 have the rank 2 and (3 or 4)
    chex.assert_rank([t4, t5], [2, {3, 4}])

    assert_shape(x, (2, 3)) # x has shape (2, 3)
    assert_shape([x, y], [(), (2, 3)]) # x is scalary and y has shape (2, 3)
    
    assert_rank(x, 0) # x is a scalar
    assert_rank([x, y], [0, 2]) # assert x is scalar and y is rank-2 array
    assert_rank([x, y], {0, 2}) # assert x and y are either scalar or rank-2 arrays

    assert_type(x, int) # x has type int
    assert_type([x, y], [int, float]) # x has type 'int' and y has type 'float'

    assert_equal_shape([x, y, z]) # assert equal shape

    assert_tree_all_close(tree_x, tree_y) # valuess and tree structures 
    assert_tree_all_finite(tree_x) # all tree_x leaves are finite

    assert_devices_available(2, 'gpu') # 2 GPU's avaialble
    assert_tpu_available() # at least 1 TPU available

    assert_numerical_grads(f, (x, y), j) # f^{(j)} (x, y) matches numerical grads. 

# make sure that jax.jit is not retracing more than n times.
def retracing():

    @jax.jit
    @chex.assert_max_traces(n=1)
    def fn_sum_jitted(x, y):
        return x + y
    
    z = fn_sum_jitted(jnp.zeros(3), jnp.zeros(3))
    t = fn_sum_jitted(jnp.zeros(6,7), jnp.zeros(6, 7)) # assertion error

    def fn_sub(x, y):
        return x - y

    # can be used with jax.pmap
    fn_sub_pmapped = jax.pmap(chex.assert_max_retraces(fn_sub), n = 10)

### Test Variants with and without jax
def fn(x, y):
    return x + y

class ExampleTest(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test(self):
        
        var_fn = self.variant(fn)
        # OR
        @self.variant
        def var_fn(x,y):
            return x + y

        self.assertEqual(fn(1,2), 3)
        self.assertEqual(var_fn(1,2), fn(1, 2))

## Paramterized Testing
class ExampleParameterizedTest(parameterized.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ('case_positive', 1, 2, 3),
        ('case_negative', -1, -2, -3)
    )
    def test(self, arg_1, arg_2, expected):
        @self.variant
        def var_fn(x, y):
            return x + y
        
        self.assertEqual(var_fn(arg_1, arg_2), expected)

## Fake jit and pmapping. 
def fake_pmap(inputs):
    with chex.fake_pmap():
        @jax.pmap
        def fn(inputs):
            #...
            return True

        # function will be vmapped over inputs. 
        fn(inputs)

    # this also works
    fake_pmap2 = chex.fake_map()
    fake_pmap2.start()
    # insert code here
    fake_pmap2.stop()

# faking set up of multi-device test environments
def setUpModule():
    chex.set_n_cpu_devices()