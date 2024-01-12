#!/usr/bin/env python

import jax.numpy as np
import jax

import matplotlib.pyplot as pl

# try to answer qualitatively the question: 
# for N points x_i, convex combination weights a = softmax(x) 
# and x just some distribution of positive weights, how does 
# the distribution of sum_i a_i x_i look like? 


N = 20  # number of points
d = 2  # dimensionality of points
N_samples = 2000  # number of samples


pts = jax.random.normal(jax.random.PRNGKey(666), shape=(N, d))

def randpt(k, stretch):
    key = jax.random.PRNGKey(k)

    # make positive numbers, some large, some small
    # the larger this constant, the further from uniform the distributions
    rand = stretch * jax.random.normal(key, shape=(N,))
    # x = jax.nn.softplus(rand)
    x = rand
    alpha = jax.nn.softmax(x)

    scaledpts = alpha[:, None] * pts
    pt = scaledpts.sum(axis=0)
    return pt


def plotfig(stretch):
    
    combinedpts = jax.vmap(randpt, in_axes=(0, None))(np.arange(N_samples), stretch)

    pl.scatter(combinedpts[:, 0], combinedpts[:, 1], color='black', alpha=.1)
    pl.scatter(pts[:, 0], pts[:, 1], color='red')


for i, stretch in enumerate(np.logspace(-1, 1, 16)):

    pl.subplot(4, 4, i+1)
    plotfig(stretch)
    pl.gca().set_title(f'stretch = {stretch:.2f}')

pl.show()
