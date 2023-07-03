#!/usr/bin/env python

import jax
import jax.numpy as np

import tinygp
import optax

import ipdb
import matplotlib.pyplot as pl
import tqdm

# recommended by tinygp docs...
jax.config.update("jax_enable_x64", True)

class GradientKernel(tinygp.kernels.Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    # instead of X1 = (data, is_derivative) we want to be able to accomodate
    # several dimensions with gradients. probably the easiest way is to
    # include each directional derivative along [0...1...0] as a separate
    # data point. lots of redundant computation but probably the cleanest
    # code wise. wait, probably not even redundant computation... :)

    # so let us construct the X1, X2 datapoints like so:
    # x1, g1 = X1
    # where g1 is an integer with the following meaning:
    # 0: the corresponding data point is a normal observation, no gradients
    # 1, ..., nx: the data point is an observation of df/dx_i, meaning
    # the directional derivative of f along coordinate i

    # then we still have to calculate all the possible derivatives (nx**2)
    # and select the correct one.

    def evaluate(self, X1, X2):

        x1, g1 = X1
        x2, g2 = X2

        # idea: build a big array of all observation-derivative-kernel
        # pairs. it will have size (1+nx, 1+nx), looking like
        #     k(x1, x2)        d/dx2_1 k(x1, x2)          ...   d/dx2_nx k(x1, x2)
        # d/dx1_1 k(x1, x2)  d/dx1_1 d/dx2_1 k(x1, x2)         d/dx1_1 d/dx2_nx k(x1, x2)
        #
        #        ...                 ...                                 ...
        #
        # d/dx1_nx k(x1, x2)         ...                       d/dx1_nx d/dx2_nx k(x1, x2)

        # then we can index this directly with the gradient flags :)))

        # documentation says the input will have shape (nx,) here, and vmap is done automatically.
        # https://tinygp.readthedocs.io/en/latest/api/summary/tinygp.kernels.Custom.html#tinygp.kernels.Custom

        # normal observation covariances.
        K = self.kernel.evaluate(x1, x2)

        nx = x1.shape[0]

        # observation-gradient covariances. row and col vecs respectively.
        K_grad_x1 = jax.jacobian(self.kernel.evaluate, argnums=0)(x1, x2).reshape(nx, 1)
        K_grad_x2 = jax.jacobian(self.kernel.evaluate, argnums=1)(x1, x2).reshape(1, nx)

        # will this do the right thing? maybe need some vmap additionally
        # derivatives commute in all reasonable cases
        K_grad_x1x2 = jax.jacobian(jax.jacobian(self.kernel.evaluate, argnums=0), argnums=1)(x1, x2)

        # first index = row index corresponds to gradient flag of x1
        # second index = col index corresponds to gradient flag of x2
        all_kernel_pairs = np.block([
            [K,         K_grad_x2  ],
            [K_grad_x1, K_grad_x1x2],
        ])

        # now this should select the right kernel based on the two gradient # flags.
        return all_kernel_pairs[g1, g2]



def build_gp(params, X, g):
    # exp transform for >0 constraint.
    amp = np.exp(params['log_amp'])
    scale_vec = np.exp(params['log_scale'])

    # ...just the kernel specification syntax took some getting used to
    # linear will first scale by the diagonal matrix diag(scale_vec),
    # then pass that to the kernel, ExpSquared.
    base_kernel = amp * tinygp.transforms.Linear(
            scale_vec, tinygp.kernels.ExpSquared()
    )
    # todo: incorporate an acutal linear map instead of jus

    kernel = GradientKernel(base_kernel)

    # the gradient flag is passed with the data, it supports data in pytree
    # format as long as the kernel function resolves it correctly, see
    # tinygp tutorial about derivative observations

    # as we are working with almost noise free data, it makes more sense to
    # set a slightly higher value here to make the matrices not too ill conditioned
    # instead of trying to estimate the actual numerical/ODE solver noise.
    noise_size = 0.1
    return tinygp.GaussianProcess(kernel, (X, g), diag=noise_size**2)


def get_optimised_gp(build_gp_fct, init_params, xs, ys, gradient_flags, steps=100, plot=True):

    # just one point estimate of kernel parameters that minimise nll
    # no fancy 'being bayesian about hyperparameters' here
    # sorry krause

    # nll should take in (params, xs, ys, gradient_flags)
    def nll(params, X, y, g):
        gp = build_gp_fct(params, X, g)
        return -gp.log_probability(y)

    nll_value_grad = jax.jit(jax.value_and_grad(nll, argnums=0))

    # set up optimiser for neg log likelihood
    opti = optax.adam(learning_rate=.05)
    opt_state = opti.init(init_params)

    def train_loop_iter(carry, inp):
        params, opt_state = carry
        loss_val, grads = nll_value_grad(params, xs, ys, gradient_flags)
        updates, opt_state = opti.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    init_carry = (init_params, opt_state)
    (params, opt_state), losses = jax.lax.scan(train_loop_iter, init_carry, xs=None, length=steps)

    if plot:
        pl.plot(losses, label='GP neg log likelihood')
        pl.legend()
        pl.show()

    trained_gp = build_gp(params, xs, gradient_flags)
    nll_value = nll(params, xs, ys, gradient_flags)

    return trained_gp, params, nll_value




def reshape_for_gp(ys):

    # needs as input an (N_pts, 2*nx+1) array with the extended state
    # vectors from the pontryagin solver. extended state = (x, λ, v).

    # puts the observations and derivative observations in the same array
    # in turn we make an extra array of 'gradient flags'
    # see more in the implementation of the gradient kernel.

    N_pts, N_extended_state = ys.shape
    N_dims = N_extended_state // 2  # N_extended_state = 2*N_dims + 1 so always uneven

    xs, grad_ys, ys = np.split(ys, [N_dims, 2*N_dims], axis=1)

    ys = ys.reshape(N_pts)  # flatten, has shape (N_pts, 1) otherwise

    assert xs.shape == (N_pts, N_dims)
    assert ys.shape == (N_pts,)
    assert grad_ys.shape == xs.shape

    # basically the xs but repeated for each observation type (y and all gradients)
    new_xs = np.kron(np.ones(1+N_dims)[:, None], xs)  # it just works

    # .T will give [x1, x1, ..., x2, x2, ... ], otherwise it is [x1 x2 x3 ... x1 x2 x3...]
    new_ys = np.concatenate([ys, grad_ys.T.reshape(-1)])
    grad_flags = np.kron(np.arange(1+N_dims), np.ones(N_pts)).astype(np.int8)

    return (new_xs, new_ys, grad_flags)



if __name__ == '__main__':

    # minimal working example

    key = jax.random.PRNGKey(0)

    N_pts = 256
    nx = 2

    noise_size = 0.1

    key, subkey = jax.random.split(key)
    xs_f = jax.random.normal(subkey, (N_pts, nx))

    def f(x):
        # return x[0]**2 + (x[1]-x[0]**2)**2
        return x[0]**2 + np.sin(3*x[1]+0.3*x[0])

    key, subkey = jax.random.split(key)
    fs = jax.vmap(f)(xs_f) + noise_size * jax.random.normal(subkey, (N_pts,))

    key, subkey = jax.random.split(key)
    grad_fs = jax.vmap(jax.grad(f))(xs_f) + noise_size * jax.random.normal(subkey, (N_pts, nx))


    # this has since been changed - pass the ys array directly
    xs, ys, grad_flags = gradient_gp.reshape_for_gp(xs_f, fs, grad_fs)

    initparams = {
            'log_amp': np.log(1),
            'log_scale': np.zeros(nx),
    }


    gp = gradient_gp.get_optimised_gp(
            gradient_gp.build_gp,
            initparams,
            xs,
            ys,
            grad_flags,
    )

    # then, do inference like this: (probably wrong variables here)
    #                              v data to condition on. must be same shape
    #                              v    v points to evaluate prediction
    pred_gp = trained_gp.condition(ys, (X_pred, pred_grad_flag)).gp
    y_pred = pred_gp.loc
    y_std = np.sqrt(pred_gp.variance)
