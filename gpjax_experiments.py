#!/usr/bin/env python

import jax
import jax.numpy as np

import gpjax as gpx
import optax

import ipdb

import matplotlib
import matplotlib.pyplot as pl

if __name__ == '__main__':
    print('hi')

    # make samples of some function and its gradient

    # ultimately we have an almost zero noise situation, only from ODE solver and float math
    # but setting it too small seems to make the kernel parameter selection difficult
    noise_size = .05

    def f(x):
        return np.sin(x) + x**2/100

    grad_f = jax.grad(f)

    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)

    xs_f = 3 * jax.random.normal(k1, (32, 1))
    subkey, key = jax.random.split(key)
    fs = jax.vmap(f)(xs_f) + noise_size * jax.random.normal(subkey, xs_f.shape)

    xs_fgrads = 3 * jax.random.normal(k1, (32,))
    subkey, key = jax.random.split(key)
    fgrads = jax.vmap(grad_f)(xs_fgrads) + noise_size * jax.random.normal(subkey, xs_f.shape)

    D = gpx.Dataset(X=xs_f, y=fs)

    xs_plot = np.linspace(-10, 10, 201)
    fs_plot = jax.vmap(f)(xs_plot)
    fig1, ax1 = pl.subplots()
    ax1.plot(xs_plot, fs_plot, linestyle='--', alpha=.5, c='black')

    ax1.scatter(xs_f, fs)

    # basically the gpjax regression example
    cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)

    xtest = xs_plot.reshape(-1, 1)
    prior_dist = prior.predict(xtest)
    prior_mean = prior_dist.mean()
    prior_std = np.sqrt(prior_dist.variance())
    samples = prior_dist.sample(seed=key, sample_shape=(20,))

    fig, ax = pl.subplots()
    fig.canvas.manager.set_window_title('samples from GP prior')
    ax.plot(xtest, samples.T, alpha=0.5, color=cols[0], label="Prior samples")
    ax.plot(xtest, prior_mean, color=cols[1], label="Prior mean")
    ax.fill_between(
        xtest.flatten(),
        prior_mean - prior_std,
        prior_mean + prior_std,
        alpha=0.3,
        color=cols[1],
        label="Prior std. dev",
    )
    ax.legend(loc="best")



    likelihood = gpx.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    negative_mll = jax.jit(gpx.objectives.ConjugateMLL(negative=True))

    xtest = xs_plot.reshape(-1, 1)
    posterior_dist = posterior.predict(xtest, D)
    posterior_mean = posterior_dist.mean()
    posterior_std = np.sqrt(posterior_dist.variance())
    samples = posterior_dist.sample(seed=key, sample_shape=(20,))

    fig, ax = pl.subplots()
    fig.canvas.manager.set_window_title('samples from GP posterior (no training)')
    ax.plot(xtest, samples.T, alpha=0.5, color=cols[0], label="posterior samples")
    ax.plot(xtest, posterior_mean, color=cols[1], label="posterior mean")
    ax.fill_between(
        xtest.flatten(),
        posterior_mean - posterior_std,
        posterior_mean + posterior_std,
        alpha=0.3,
        color=cols[1],
        label="posterior std. dev",
    )
    ax.legend(loc="best")



    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=optax.adam(learning_rate=0.01),
        num_iters=500,
        safe=True,
        key=key,
    )

    xtest = xs_plot.reshape(-1, 1)
    posterior_dist = opt_posterior.predict(xtest, D)
    posterior_mean = posterior_dist.mean()
    posterior_std = np.sqrt(posterior_dist.variance())
    samples = posterior_dist.sample(seed=key, sample_shape=(20,))

    ax1.plot(xtest, samples.T, alpha=0.5, color=cols[0], label="posterior samples")
    ax1.plot(xtest, posterior_mean, color=cols[1], label="posterior mean")
    ax1.fill_between(
        xtest.flatten(),
        posterior_mean - 2*posterior_std,
        posterior_mean + 2*posterior_std,
        alpha=0.3,
        color=cols[1],
        label="2x posterior std. dev",
    )
    ax1.legend(loc="best")

    fig, ax = pl.subplots()
    ax.plot(history, color=cols[1])
    ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")

    pl.show()

    ipdb.set_trace()
