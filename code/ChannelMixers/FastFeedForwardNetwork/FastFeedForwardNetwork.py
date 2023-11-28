"""
Fast Feedforward Networks: https://arxiv.org/pdf/2308.14711.pdf
Exponentially Faster Language Modeling: https://arxiv.org/pdf/2311.10770.pdf
"""
import jax
from jax import numpy as jnp
from jax import random
from flax import linen as nn
from flax.linen import initializers as init
from typing import (
  Callable
)
vmap = jax.vmap

class FastFeedForwardNetwork(nn.Module):
  output_dim : int
  leaf_dim : int
  depth : int

  activation : Callable = lambda u : u[u.shape[0]//2:] * u[:u.shape[0]//2]
  use_bias : bool = True
  use_ste : bool = True

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

    #self.leaves = [FFFNMLP(self.leaf_dim, self.use_bias, activation=self.activation, dtype=self.dtype) for _ in range(self.num_leaves)]



  @nn.compact
  def __call__(self, inputs, training, *args, **kwargs):
    B, T, D = inputs.shape


    w1 = self.param("kernel_1", self.param_init(D), (self.num_leaves, D*self.leaf_dim*2), self.dtype)
    w2 = self.param("kernel_2", self.param_init(self.leaf_dim), (self.num_leaves, self.leaf_dim*self.output_dim), self.dtype)

    if self.use_bias:
      b1 = self.param("bias_1", self.param_init(D), (self.num_leaves,self.leaf_dim*2), self.dtype)
      b2 = self.param("bias_2", self.param_init(self.leaf_dim), (self.num_leaves,self.output_dim), self.dtype)

    wnode = self.param("kernel_node", self.param_init(D), (self.num_nodes, D), self.dtype)
    bnode = self.param("bias_node", self.param_init(D), (self.num_nodes,1), self.dtype)

    ## layer logic ##

    def fwd_train(l, m, n):
      if m == self.depth:
        _w1 = w1[n,:].reshape((D, self.leaf_dim*2))
        _w2 = w2[n,:].reshape((self.leaf_dim, self.output_dim))
        
        if self.use_bias:
          _b1 = b1[n,:].reshape((self.leaf_dim*2,))
          _b2 = b2[n,:].reshape((self.output_dim,))

        y = l @ _w1
        if self.use_bias:
          y = y + _b1
        y = self.activation(y)
        y = y @ _w2
        if self.use_bias:
          y = y + _b2
        return y
      else:
        
        w = wnode[m,:]          
        b = bnode[m,:]
        c = l @ w + b

        c = nn.sigmoid(c)

        # STE
        if self.use_ste:
          c = c - jax.lax.stop_gradient(c) + jax.lax.stop_gradient(jnp.rint(c).astype(int))

        return c * fwd(l, m+1, 2*n) + (1-c) * fwd(l, m+1, 2*n + 1)

    def fwd_inference(l, m, n):
      if m == self.depth:
        _w1 = w1[n,:].reshape((D, self.leaf_dim*2))
        _w2 = w2[n,:].reshape((self.leaf_dim, self.output_dim))
        
        if self.use_bias:
          _b1 = b1[n,:].reshape((self.leaf_dim*2,))
          _b2 = b2[n,:].reshape((self.output_dim,))

        y = l @ _w1
        if self.use_bias:
          y = y + _b1
        y = self.activation(y)
        y = y @ _w2
        if self.use_bias:
          y = y + _b2
        return y
      else:

        w = wnode[m,:]
        b = bnode[m,:]
        c = l @ w + b

        c = nn.sigmoid(c)
        c = jnp.rint(c).astype(int)

        return fwd(l, m+1, 2*n + c)# + (1-c) * fwd(l, m+1, 2*n + 1)

    if training:
      fwd = fwd_train
    else:
      fwd = fwd_inference

    y = vmap(vmap(lambda u : fwd(u, 0, 0)))(inputs)


    return y