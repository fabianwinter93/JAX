import numpy as np
import jax
from jax import random
from jax import numpy as jnp
import flax
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


class QLSTM(nn.Module):
  dim : int
  state_dim : int

  use_bias : bool = True

  kernel_init : Callable = init.lecun_normal()
  bias_init : Callable = init.zeros_init()

  forget_fn : Callable = nn.sigmoid
  input_fn : Callable = nn.sigmoid
  output_fn : Callable = nn.tanh
  canidate_fn : Callable = nn.tanh

  dtype : jax.typing.DTypeLike = jnp.float32

  def init_states(self, batchdim):
    states = {}
    states["c"] = jnp.zeros((batchdim, 1, self.state_dim)).astype(self.dtype)
    states["f"] = jnp.ones((batchdim, 1, self.state_dim)).astype(self.dtype)
    return states

  def setup(self):

    self.c_dense = nn.Dense(self.state_dim, use_bias=self.use_bias, 
                      kernel_init=self.kernel_init, bias_init=self.bias_init, 
                      param_dtype=self.dtype)
    self.i_dense = nn.Dense(self.state_dim, use_bias=self.use_bias, 
                      kernel_init=self.kernel_init, bias_init=self.bias_init, 
                      param_dtype=self.dtype)
    self.f_dense = nn.Dense(self.state_dim, use_bias=self.use_bias, 
                      kernel_init=self.kernel_init, bias_init=self.bias_init, 
                      param_dtype=self.dtype)
    self.o_dense = nn.Dense(self.state_dim, use_bias=self.use_bias, 
                      kernel_init=self.kernel_init, bias_init=self.bias_init, 
                      param_dtype=self.dtype)
    self.h_dense = nn.Dense(self.dim, use_bias=False, 
                      kernel_init=self.kernel_init, bias_init=self.bias_init, 
                      param_dtype=self.dtype)

  @nn.compact
  def __call__(self, inputs, states, *args, **kwargs):
    B, T, D = inputs.shape

    c_dense = self.c_dense
    f_dense = self.f_dense
    i_dense = self.i_dense
    o_dense = self.o_dense
    h_dense = self.h_dense

    c = vmap(vmap(lambda u : self.canidate_fn(c_dense(u))))(inputs)
    f = vmap(vmap(lambda u : self.forget_fn(f_dense(u))))(inputs)
    i = vmap(vmap(lambda u : self.input_fn(i_dense(u))))(inputs)
    o = vmap(vmap(lambda u : self.output_fn(o_dense(u))))(inputs)
    
    u = i * c

    f = jnp.concatenate([jnp.ones((B, 1, self.state_dim)), f], 1)
    u = jnp.concatenate([states["c"], u], 1)

    elements = (f, u)

    _, ct = vmap(jax.lax.associative_scan, (None, 0))(binary_op, elements)
    ct = ct[:, 1:]

    y = vmap(vmap(lambda u : h_dense(u)))(inputs)

    new_states = {}
    new_states["c"] = jnp.expand_dims(ct[:,-1], 1)

    return y, new_states
