"""
Resurrecting Recurrent Neural Networks for Long Sequences: https://arxiv.org/pdf/2303.06349.pdf
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


class LinearRecurrentUnit(nn.Module):
  dim : int

  r_min : float = 0. 
  r_max : float = 1. 
  max_phase : float = 6.28

  dtype : jax.typing.DTypeLike = jnp.float32
  complex_dtype : jax.typing.DTypeLike = jnp.complex64

  def init_states(self, batchdim):
    states = {}
    states["h"] = jnp.zeros((batchdim, 1, self.dim)).astype(self.complex_dtype)
    return states

  def B_init(self):
      def f(key, shape, dtype):
        return random.normal(key, shape)/jnp.sqrt(2*shape[0]).astype(dtype)
      return f

  def C_init(self):
    def f(key, shape, dtype):
      return random.normal(key, shape)/jnp.sqrt(shape[-1]).astype(dtype)
    return f

  def nu_init(self, r_min, r_max):
    def f(key, shape, dtype):
      u = random.uniform(key, shape)
      nu_log = jnp.log(-0.5*jnp.log(u*(r_max**2-r_min**2) + r_min**2))
      return nu_log.astype(dtype)
    return f

  def theta_init(self, max_phase):
    def f(key, shape, dtype):
      u = random.uniform(key, shape)
      theta_log = jnp.log(max_phase*u)
      return theta_log.astype(dtype)
    return f

  def gamma_init(self, nu_log, theta_log):
    def f(key, dtype):
      diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
      gamma_log = jnp.log(jnp.sqrt(1-jnp.abs(diag_lambda)**2))
      return gamma_log
    return f


  @nn.compact
  def __call__(self, inputs, states, *args, **kwargs):
    B, T, D = inputs.shape

    ## init ##
    nu_log = self.param("nu_log", self.nu_init(self.r_min, self.r_max), (self.dim,), self.dtype)
    theta_log = self.param("theta_log", self.theta_init(self.max_phase), (self.dim,), self.dtype)
 
    B_re = self.param("B_re", self.B_init(), (D,self.dim), self.dtype)
    B_im = self.param("B_im", self.B_init(), (D,self.dim), self.dtype)
    C_re = self.param("C_re", self.C_init(), (self.dim,D), self.dtype)
    C_im = self.param("C_im", self.C_init(), (self.dim,D), self.dtype)
    D = self.param("D", lambda k,s,d : random.normal(k, s).astype(d), (D,), self.dtype)
    
    gamma_log = self.param("gamma_log", self.gamma_init(nu_log, theta_log), self.dtype)
    gamma_log = jnp.expand_dims(gamma_log, 0)

    ### layer logic ###

    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))

    B_norm = (B_re + 1j*B_im) * jnp.exp(gamma_log)
    C = C_re + 1j*C_im

    # Running the LRU + output projection
    # For details on parallel scan, check discussion in Smith et al (2022).
    Lambda_elements = jnp.broadcast_to(Lambda, (B, T+1, Lambda.shape[-1]))
    Bu_elements = vmap(vmap(lambda u: u @ B_norm))(inputs)

    Bu_elements = jnp.concatenate([states["h"], Bu_elements], 1)


    elements = (Lambda_elements, Bu_elements)
    _, H = vmap(jax.lax.associative_scan, (None, 0))(binary_op, elements)
    H = H[:, 1:]
    y = vmap(vmap(lambda x, u: (x @ C).real + D * u))(H, inputs)


    new_states = {}
    new_states["h"] = jnp.expand_dims(H[:, -1], 1)

    return y, new_states
   