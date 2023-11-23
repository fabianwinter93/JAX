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


    ## layer logic ##

    mixture = jnp.ones((B, T, self.num_leaves))

    node_dense = nn.DenseGeneral((self.num_nodes,1), use_bias=self.use_bias, kernel_init=self.node_init(1/jnp.sqrt(D)), param_dtype=self.dtype)

    c = vmap(vmap(lambda u : nn.sigmoid(node_dense(u))))(inputs)

    """
    if training:
      c = c - jax.lax.stop_gradient(c) + jax.lax.stop_gradient(jnp.rint(c))
    else:
      c = jnp.rint(c)
      """

    nc = 1-c

    c_nc = jnp.stack([c, nc], -1)
    c_nc = jnp.reshape(c_nc, (B, T, -1))
    c_nc = jnp.expand_dims(c_nc, -1)

    for curr_depth in range(self.depth):
      n_nodes = 2**curr_depth

      mixture = mixture.reshape((B, T, n_nodes*2, self.num_leaves // (2*n_nodes)))
      mixture = mixture * c_nc[...,:n_nodes*2,:]

    mixture = mixture.reshape((B, T, self.num_leaves))

    l1a = nn.DenseGeneral((self.num_leaves, self.leaf_dim), kernel_init=self.leaf_init(1/jnp.sqrt(D)), use_bias=self.use_bias, param_dtype=self.dtype) 
    l1b = nn.DenseGeneral((self.num_leaves, self.leaf_dim), kernel_init=self.leaf_init(1/jnp.sqrt(D)), use_bias=self.use_bias, param_dtype=self.dtype) 
    l2 = nn.DenseGeneral((self.num_leaves, self.dim), axis=(-1,-2), kernel_init=self.leaf_init(1/jnp.sqrt(self.leaf_dim)), use_bias=self.use_bias, param_dtype=self.dtype) 

    y = vmap(vmap(lambda u : l2(l1a(u)*l1b(u))))(inputs)
    
    y = y * jnp.expand_dims(mixture, -1)
    y = y.sum(-2)
    y = y.reshape((B, T, -1))

    return y