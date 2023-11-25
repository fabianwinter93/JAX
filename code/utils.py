import jax

vmap = jax.vmap

vm = lambda x,w : x @ w
vmb = lambda x,w,b : x @ w + b

btvm = lambda w,b : jax.vmap(jax.vmap(lambda u : u @ w + b))
