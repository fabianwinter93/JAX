"""
GATELOOP: FULLY DATA-CONTROLLED LINEAR RE-
CURRENCE FOR SEQUENCE MODELING: https://arxiv.org/pdf/2311.01927.pdf
"""

import jax
from jax import random
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers as init
from typing import (
    Callable
)
vmap = jax.vmap

@jax.checkpoint
def binary_op(ei, ej):
  #ai, bi = ei
  #aj, bj = ej
  decay_tm1, state_tm1 = ei
  decay_t, input_t = ej

  decay_prop = decay_t * decay_tm1
  decayed_state = decay_t * state_tm1
  new_state = decayed_state + input_t
  #return aj*ai, aj*bi + bj
  return decay_prop, new_state

class GateLoop(nn.Module):
  v_dim : int
  kq_dim : int
  kq_use_bias : bool = True
  v_use_bias : bool = True
  alpha_beta_use_bias : bool = True

  kernel_init : Callable = init.lecun_normal()
  bias_init : Callable = init.zeros_init()
  
  dtype : jax.typing.DTypeLike = jnp.float32
  complex_dtype : jax.typing.DTypeLike = jnp.complex64

  magnitude_activation : Callable = nn.sigmoid
  phase_activation : Callable = lambda x : x

  def init_states(self, batchdim):
    states = {}
    states["kv"] = jnp.zeros((batchdim, 1, self.kq_dim, self.v_dim)).astype(self.complex_dtype)
    states["a"] = jnp.ones((batchdim, 1, self.kq_dim, 1)).astype(self.complex_dtype)
    return states

  def setup(self):
    self.k_dense = nn.Dense(self.kq_dim, use_bias=self.kq_use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init, param_dtype=self.complex_dtype)
    self.v_dense = nn.Dense(self.v_dim, use_bias=self.v_use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init, param_dtype=self.dtype)
    self.q_dense = nn.Dense(self.kq_dim, use_bias=self.kq_use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init, param_dtype=self.complex_dtype)
    self.alpha_dense = nn.Dense(self.kq_dim, use_bias=self.alpha_beta_use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init, param_dtype=self.dtype)
    self.beta_dense = nn.Dense(self.kq_dim, use_bias=self.alpha_beta_use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init, param_dtype=self.dtype)

  @nn.compact
  def __call__(self, inputs, states, *args, **kwargs):
    B, T, D = inputs.shape

    k_dense = self.k_dense
    v_dense = self.v_dense
    q_dense = self.q_dense
    alpha_dense = self.alpha_dense
    beta_dense = self.beta_dense

    #     layer logic     #

    K = vmap(vmap(lambda u : k_dense(u)))(inputs)
    V = vmap(vmap(lambda u : v_dense(u)))(inputs)
    Q = vmap(vmap(lambda u : q_dense(u)))(inputs)
    alpha = vmap(vmap(lambda u : alpha_dense(u)))(inputs)
    beta = vmap(vmap(lambda u : beta_dense(u)))(inputs)
    
    kv_outer = lambda k, v : jnp.einsum("i,j->ij", k, v)
    #kv_outer = jax.checkpoint(kv_outer)

    KV = vmap(vmap(kv_outer))(K, V)

    A = self.magnitude_activation(alpha) * jnp.exp(1j*self.phase_activation(beta))
    A = jnp.expand_dims(A, -1)

    KV = jnp.concatenate([states["kv"], KV], 1)
    A = jnp.concatenate([states["a"], A], 1)

    _, H = vmap(jax.lax.associative_scan, (None, 0))(binary_op, (A, KV))

    H = H[:, 1:]

    y = vmap(vmap(lambda u,m : u @ m))(Q, H)

    new_states = {}
    new_states["kv"] = jnp.expand_dims(H[:, -1], 1)
    new_states["a"] = jnp.expand_dims(A[:, -1], 1)

    return y, new_states