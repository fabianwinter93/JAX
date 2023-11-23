"""
Fast Feedforward Networks: https://arxiv.org/pdf/2308.14711.pdf
Exponentially Faster Language Modeling: https://arxiv.org/pdf/2311.10770.pdf
"""
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers as init
from typing import (
  Callable
)
vmap = jax.vmap

class FastFFN(nn.Module):
  dim : int
  leaf_dim : int
  depth : int

  use_bias : bool = True


  dtype : jax.typing.DTypeLike = jnp.float32

  node_init : Callable = init.normal
  leaf_init : Callable = init.normal

  def setup(self):
    self.num_leaves = 2**self.depth
    self.num_nodes = self.num_leaves - 1

  @nn.compact
  def __call__(self, inputs, training, *args, **kwargs):
    B, T, D = inputs.shape

    
    mixture = jnp.ones((B, T, self.num_leaves))

    for curr_depth in range(self.depth):
      n_nodes = 2**curr_depth

      wnode = self.param(f"weight_node_{curr_depth}", self.leaf_init(1/jnp.sqrt(D)), (D, n_nodes), self.dtype)
      bnode = self.param(f"bias_node_{curr_depth}", init.zeros_init(), (n_nodes,), self.dtype)

      c = vmap(vmap(lambda u : nn.sigmoid(u @ wnode + bnode)))(inputs)
      if training:
        c = c - jax.lax.stop_gradient(c) + jax.lax.stop_gradient(jnp.rint(c))
      else:
        c = jnp.rint(c)

      nc = 1-c

      c_nc = jnp.stack([c, nc], -1)
      c_nc = jnp.reshape(c_nc, (B, T, -1))
      c_nc = jnp.expand_dims(c_nc, -1)

      mixture = mixture.reshape((B, T, n_nodes*2, self.num_leaves // (2*n_nodes)))
      mixture = mixture * c_nc

    mixture = mixture.reshape((B, T, self.num_leaves))

    w_leaves_1_a = [self.param(f"weight_leaf_1a_{i}", self.leaf_init(1/jnp.sqrt(D)), (D, self.leaf_dim), self.dtype) for i in range(self.num_leaves)]
    b_leaves_1_a = [self.param(f"bias_leaf_1a_{i}", init.zeros_init(), (self.leaf_dim,), self.dtype) for i in range(self.num_leaves)]
    w_leaves_1_b = [self.param(f"weight_leaf_1b_{i}", self.leaf_init(1/jnp.sqrt(D)), (D, self.leaf_dim), self.dtype) for i in range(self.num_leaves)]
    b_leaves_1_b = [self.param(f"bias_leaf_1b_{i}", init.zeros_init(), (self.leaf_dim,), self.dtype) for i in range(self.num_leaves)]

    w_leaves_2 = [self.param(f"weight_leaf_2_{i}", self.leaf_init(1/jnp.sqrt(self.leaf_dim)), (self.leaf_dim, D), self.dtype) for i in range(self.num_leaves)]
    b_leaves_2 = [self.param(f"bias_leaf_2_{i}", init.zeros_init(), (D,), self.dtype) for i in range(self.num_leaves)]

    w1a = jnp.stack(w_leaves_1_a, 0)
    b1a = jnp.stack(b_leaves_1_a, 0)

    w1b = jnp.stack(w_leaves_1_b, 0)
    b1b = jnp.stack(b_leaves_1_b, 0)

    leaf_f_1 = vmap(lambda u,wa,wb,ba,bb : (u @ wa + ba) * (u @ wb + bb), (None, 0, 0, 0, 0))
    y1 = vmap(vmap(lambda u : leaf_f_1(u, w1a, w1b, b1a, b1b)))(inputs)

    w2 = jnp.stack(w_leaves_2, 0)
    b2 = jnp.stack(b_leaves_2, 0)

    leaf_f_2 = vmap(lambda u,w,b : u @ w + b, (-2, 0, 0))
    y2 = vmap(vmap(lambda u : leaf_f_2(u, w2, b2)))(y1)

    y = y2 * jnp.expand_dims(mixture, -1)
    y = y.sum(-2)
    y = y.reshape((B, T, -1))

    return y