"""
Attention Is Not All You Need Anymore: https://arxiv.org/pdf/2308.07661.pdf
Interpretation of the Transformer and Improvement of the Extractor: https://arxiv.org/pdf/2311.12678v1.pdf
"""
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers as init
from typing import (
    Callable
)
vmap = jax.vmap

class SuperHighPerformanceExtractor(nn.Module):
  features_dim : int
  time_dim : int

  kernel_init : Callable = init.lecun_normal
  dtype : jax.typing.DTypeLike = jnp.float32

  @nn.compact
  def __call__(self, inputs, training, *args, **kwargs):
    B, T, D = inputs.shape

    ### init ###
    Wext = [self.param(f"Wext_{i}", self.kernel_init(dtype=self.dtype), (D, self.features_dim)) for i in range(self.time_dim)]
    Wext = jnp.stack(Wext, 0)

    Wadj = self.param("Wadj", self.kernel_init(dtype=self.dtype), (D, self.features_dim))

    ### layer logic ###
    f = lambda u,w : u @ w
    f = vmap(f)
    f = vmap(f, (0, None))

    xi = f(inputs, Wext)

    Xout_ext = vmap(jax.lax.associative_scan, (None, 0))(jnp.add, xi)
    scaling = 1 / jnp.sqrt(jnp.arange(1, T+1))
    Xout_ext = jnp.apply_along_axis(jnp.multiply, 1, Xout_ext, scaling)

    Xout_adj = vmap(vmap(lambda xin,xout : (xin @ Wadj) * xout))(inputs, Xout_ext)

    return Xout_adj
