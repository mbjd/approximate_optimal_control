#!/usr/bin/env python
import os

# for now, a minimal working example of supervised learning with a toy dataset

# somehow this doesn't work, but starting the script with
# JAX_PLATFORMS='' ./simplejax.py does...
os.environ['JAX_PLATFORMS'] = ''

os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import jax
import jax.numpy as jnp
from jax import grad

import optax

from flax import linen as nn

import numpy as np

import matplotlib.pyplot as pl
from matplotlib import cm


def generate_data(key, in_dim=2, n_pts=100):
    data_key, function_key = jax.random.split(key)

    # quadratic function x.T @ P @ x
    Phalf = jax.random.normal(function_key, (in_dim, in_dim))
    P = Phalf.T @ Phalf  # should be positive definite
    print(P)

    xs = jax.random.normal(data_key, shape=(n_pts, in_dim))
    ys = jax.vmap(lambda x: x.T @ P @ x, 0)(xs)

    # return training and test data.
    train_split = 0.8
    train_idx = int(train_split * xs.shape[0])

    xs_train = xs[0:train_idx, :]
    ys_train = ys[0:train_idx]

    xs_test = xs[train_idx:, :]
    ys_test = ys[train_idx:]

    return xs_train, ys_train, xs_test, ys_test


class flax_NN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.softplus(x)
        x = nn.Dense(features=64)(x)
        x = nn.softplus(x)
        x = nn.Dense(features=16)(x)
        x = nn.softplus(x)
        x = nn.Dense(features=1)(x)

net = flax_NN()

state = create_train_state(cnn, init_rng, learning_rate, momentum)
# print(net.tabulate(jax.random.PRNGKey(0), jnp.ones((2,))))


xs_train, ys_train, xs_test, ys_test = generate_data(
        jax.random.PRNGKey(0),
        in_dim = 2,
        n_pts = 100
)

import ipdb; ipdb.set_trace()


def loss_fn(model, xs, ys):
    pred_y = jax.vmap(model)(xs)
