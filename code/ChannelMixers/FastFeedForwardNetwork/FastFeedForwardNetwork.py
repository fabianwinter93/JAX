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
    """
    mixture = jnp.ones((B, T, self.num_leaves))

    node_kernel = self.param("node_kernel", self.param_init(D), (D, self.num_nodes), self.dtype)
    
    if self.use_bias:
      node_bias = self.param("node_bias", self.param_init(D), (self.num_nodes,), self.dtype)
      node_fn = lambda u : nn.sigmoid(u @ node_kernel + node_bias)
    else:
      node_fn = lambda u : nn.sigmoid(u @ node_kernel)
      
    c = vmap(vmap(node_fn))(inputs)

  
    if training:
      c = c - jax.lax.stop_gradient(c) + jax.lax.stop_gradient(jnp.rint(c))
    else:
      c = jnp.rint(c)
    
      

    nc = 1-c

    c_nc = jnp.stack([c, nc], -1)
    c_nc = jnp.reshape(c_nc, (B, T, -1))
    c_nc = jnp.expand_dims(c_nc, -1)

    for curr_depth in range(self.depth):
      n_nodes = 2**curr_depth

      mixture = mixture.reshape((B, T, n_nodes*2, self.num_leaves // (2*n_nodes)))
      mixture = mixture * c_nc[...,:n_nodes*2,:]

    mixture = mixture.reshape((B, T, self.num_leaves))

    
      w_leaves_1_a = [self.param(f"weight_leaf_1a_{i}", self.param_init(D), (D, self.leaf_dim), self.dtype) for i in range(self.num_leaves)]
      w_leaves_2 = [self.param(f"weight_leaf_2_{i}", self.param_init(self.leaf_dim), (self.leaf_dim, D), self.dtype) for i in range(self.num_leaves)]
      
      w1a = jnp.stack(w_leaves_1_a, 0)
      w2 = jnp.stack(w_leaves_2, 0)

      if self.use_bias:
        b_leaves_1_a = [self.param(f"bias_leaf_1a_{i}", self.param_init(D), (self.leaf_dim,), self.dtype) for i in range(self.num_leaves)]
        b_leaves_2 = [self.param(f"bias_leaf_2_{i}", self.param_init(self.leaf_dim), (D,), self.dtype) for i in range(self.num_leaves)]
        
        b1a = jnp.stack(b_leaves_1_a, 0)
        b2 = jnp.stack(b_leaves_2, 0)

      
      if self.use_bias:
        _leaf_f_1 = vmap(lambda u,wa,ba : self.activation(u @ wa + ba), (None, 0, 0))
        _leaf_f_2 = vmap(lambda u,w,b : u @ w + b, (-2, 0, 0))
        leaf_f_1 = lambda u : _leaf_f_1(u, w1a, b1a)
        leaf_f_2 = lambda u : _leaf_f_2(u, w2, b2)

      else:
        _leaf_f_1 = vmap(lambda u,wa : self.activation(u @ wa), (None, 0))
        _leaf_f_2 = vmap(lambda u,w: u @ w, (-2, 0))
        leaf_f_1 = lambda u : _leaf_f_1(u, w1a)
        leaf_f_2 = lambda u : _leaf_f_2(u, w2)
        
      y = vmap(vmap(leaf_f_1))(inputs)
      y = vmap(vmap(leaf_f_2))(y)
    

    y = y * jnp.expand_dims(mixture, -1)
    y = y.sum(-2)
    y = y.reshape((B, T, -1))
    """

    #leaves = [FFFNMLP(self.leaf_dim, self.use_bias, activation=self.activation, dtype=self.dtype) for _ in range(self.num_leaves)]
    
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







