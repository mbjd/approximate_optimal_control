#!/usr/bin/env python
import ipdb
import jax
import jax.numpy as np
import matplotlib.pyplot as pl


def f(a, b):
    return a**2 + b**2

def g(a, b):
    return a**2 + (np.floor(10000 * b)/10000)**2

def h(a, b):
    return a**2 + jax.lax.stop_gradient(b)**2

N = 32
for fct in (f, g, h):
    pl.figure(fct.__name__)
    xs = np.linspace(-2, 2, N)
    xx, yy = np.meshgrid(xs, xs)

    pl.subplot(121)
    fs = jax.vmap(fct)(xx.flatten(), yy.flatten()).reshape(N, N)
    pl.contour(xx, yy, fs)

    pl.subplot(122)
    grads = jax.vmap(jax.grad(fct, argnums=(0, 1)))(xx.flatten(), yy.flatten())
    uu = grads[0].reshape(N, N)
    vv = grads[1].reshape(N, N)
    pl.quiver(xx, yy, uu, vv)

pl.show()
