"""
Fast Feedforward Networks: https://arxiv.org/pdf/2308.14711.pdf
Exponentially Faster Language Modeling: https://arxiv.org/pdf/2311.10770.pdf
"""
from functools import partial
import jax
from jax import numpy as jnp
from jax import random
from flax import linen as nn
from flax.linen import initializers as init
from typing import (
  Callable
)
vmap = jax.vmap

class FFFNMLP(nn.Module):
  dim : int
  use_bias : bool
  activation : Callable
  dtype : jax.typing.DTypeLike


  def param_init(self, scale):
    def f(key, shape, dtype):
      _scale = 1 / jnp.sqrt(scale)
      val = random.uniform(key, shape=shape, minval=-_scale, maxval=_scale)
      return val.astype(dtype)
    return f

  @nn.compact
  def __call__(self, inputs, training, *args, **kwargs):
    D = inputs.shape[0]

    w1 = self.param("kernel_1", self.param_init(D), (D, self.dim), self.dtype)
    w2 = self.param("kernel_2", self.param_init(self.dim), (self.dim, D), self.dtype)

    if self.use_bias:
      b1 = self.param("bias_1", self.param_init(D), (self.dim,), self.dtype)
      b2 = self.param("bias_2", self.param_init(self.dim), (D,), self.dtype)
      

    if self.use_bias:
      leaf_f_1 = lambda u : self.activation(u @ w1 + b1)
      leaf_f_2 = lambda u : u @ w2 + b2

    else:
      leaf_f_1 = lambda u : self.activation(u @ w1)
      leaf_f_2 = lambda u : u @ w2
      
    y = leaf_f_1(inputs)
    y = leaf_f_2(y)

    return y


class FastFeedForwardNetwork(nn.Module):
  leaf_dim : int
  depth : int
  
  activation : Callable = nn.relu
  use_bias : bool = True

  dtype : jax.typing.DTypeLike = jnp.float32


  def param_init(self, scale):
    def f(key, shape, dtype):
      _scale = 1 / jnp.sqrt(scale)
      val = random.uniform(key, shape=shape, minval=-_scale, maxval=_scale)
      return val.astype(dtype)
    return f

  def setup(self):
    self.num_leaves = 2**self.depth
    self.num_nodes = self.num_leaves - 1

  @nn.compact
  def __call__(self, inputs, training, *args, **kwargs):
    B, T, D = inputs.shape

    ## layer logic ##
    
    if training:
      def fwd(l, m, n):
        if m == self.depth:
          mlp = FFFNMLP(self.leaf_dim, self.use_bias, activation=self.activation, dtype=self.dtype)
          y = mlp(l, training=training)
          return y
        else:
          node = nn.Dense(1, kernel_init=self.param_init(D), bias_init=self.param_init(D), use_bias=self.use_bias, param_dtype=self.dtype)
          c = nn.sigmoid(node(l))
          return c * fwd(l, m+1, 2*n) + (1-c) * fwd(l, m+1, 2*n + 1)

      y = vmap(vmap(lambda u : fwd(u, 0, 0)))(inputs)
    
    else:

      def fwd(l, m, n):
        if m == self.depth:
          mlp = FFFNMLP(self.leaf_dim, self.use_bias, activation=self.activation, dtype=self.dtype)
          y = mlp(l, training=training)
          return y
        else:
          node = nn.Dense(1, kernel_init=self.param_init(D), bias_init=self.param_init(D), use_bias=self.use_bias, param_dtype=self.dtype)
          c = nn.sigmoid(node(l))
          return c * fwd(l, m+1, 2*n) + (1-c) * fwd(l, m+1, 2*n + 1)
      y = vmap(vmap(lambda u : fwd(u, 0, 0)))(inputs)

    return y







