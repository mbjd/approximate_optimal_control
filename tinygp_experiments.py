#!/usr/bin/env python

import jax
import jax.numpy as np

import tinygp
import optax

import ipdb
import matplotlib.pyplot as pl

# recommended by tinygp docs...
jax.config.update("jax_enable_x64", True)



class DerivativeKernel(tinygp.kernels.Kernel):
    '''
    from https://tinygp.readthedocs.io/en/latest/tutorials/derivative.html

    this builds a kernel for incorporating derivative observations out of a
    regular kernel.

    is this just for one-dimesional functions? or for directional
    derivatives??

    if it wasn't already clear, this is in no way working or used yet. only
    first steps that work are in 'if name == main' below.
    '''
    def __init__(self, kernel):
        self.kernel = kernel

    def evaluate(self, X1, X2):
        t1, d1 = X1
        t2, d2 = X2

        # Differentiate the kernel function: the first derivative wrt x1
        Kp = jax.grad(self.kernel.evaluate, argnums=0)

        # ... and the second derivative
        Kpp = jax.grad(Kp, argnums=1)

        # Evaluate the kernel matrix and all of its relevant derivatives
        K = self.kernel.evaluate(t1, t2)
        d2K_dx1dx2 = Kpp(t1, t2)

        # For stationary kernels, these are related just by a minus sign, but we'll
        # evaluate them both separately for generality's sake
        dK_dx2 = jax.grad(self.kernel.evaluate, argnums=1)(t1, t2)
        dK_dx1 = Kp(t1, t2)

        return jnp.where(
            d1, jnp.where(d2, d2K_dx1dx2, dK_dx1), jnp.where(d2, dK_dx2, K)
        )



if __name__ == '__main__':

    # make samples of some function and its gradient

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


    def build_gp(params, X):
        # exp transform for >0 constraint.
        amp = np.exp(params['log_amp'])
        scale = np.exp(params['log_scale'])
        kernel = amp * tinygp.kernels.ExpSquared(scale)

        # small noise of same shape as x variable.
        diag = 1e-3 * np.ones_like(X)
        return tinygp.GaussianProcess(
                kernel, X,
                diag=np.exp(params['log_diag']),
                mean=params['mean'],
        )


    def nll(params, X, y):
        gp = build_gp(params, X)
        return -gp.log_probability(y)


    params = {
        'mean': np.float64(0),
        'log_amp': np.log(10),
        'log_scale': np.log(.01),
        'log_diag': np.log(0.1),
    }

    nll_value_grad = jax.jit(jax.value_and_grad(nll, argnums=0))

    v, g = nll_value_grad(params, xs_f, fs)

    print(v)
    print(g)

    # set up optimiser for neg log likelihood
    opti = optax.adam(learning_rate=.02)
    opt_state = opti.init(params)

    losses = []
    for i in range(500):
        loss_val, grads = nll_value_grad(params, xs_f, fs)
        losses.append(loss_val)
        updates, opt_state = opti.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    v, g = nll_value_grad(params, xs_f, fs)
    print(v)
    print(g)
    print(params)
    print(jax.tree_util.tree_map(np.exp, params))

    pl.plot(losses)
    pl.show()

    # this works...
    x_plot = np.linspace(-10, 10, 201)
    gp = build_gp(params, xs_f.squeeze())
    cond_gp = gp.condition(fs.squeeze(), x_plot).gp


    # this not. something with wrong broadcasting...
    # x_plot = np.linspace(-10, 10, 201).reshape(-1, 1)
    # gp = build_gp(params, xs_f)
    # ipdb.set_trace()
    # cond_gp = gp.condition(fs, x_plot).gp.reshape(-1, 1)

    mu, var = cond_gp.loc, cond_gp.variance



    # make a plot.
    pl.plot(x_plot, mu, color='C0', label='GP mean')
    pl.fill_between(x_plot, mu+1.96*np.sqrt(var), mu-1.96*np.sqrt(var), color='C0', alpha=.5, label='GP 95% interval')
    pl.scatter(xs_f, fs, color='black', label='data')
    pl.legend()
    pl.show()


    # next experiment: the same plus derivative info....
    # also multivariate data.
    ipdb.set_trace()
