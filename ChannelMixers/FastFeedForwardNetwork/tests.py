from FastFeedForwardNetwork import FastFeedForwardNetwork
from jax import numpy as jnp
import jax

B = 32
T = 128
Din = 256
Dleaf = 128
depth = 4

dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]

jit_options = [False, True]
bias_options = [False, True]

key = jax.random.PRNGKey(123)
inp_key, param_key = jax.random.split(key)

x = jax.random.normal(inp_key, shape=(B, T, Din))

for dt in dtypes:
  for jit_op in jit_options:
    for bias_op in bias_options:
      fffn = FastFeedForwardNetwork(Dleaf, depth, dtype=dt, use_bias=bias_op)

      params = fffn.init(param_key, x, False)

      f = fffn.apply
      if jit_op:
        f = jax.jit(f)
      
      y = f(params, x, True)

    