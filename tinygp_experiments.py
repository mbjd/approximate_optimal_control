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


def example_univariate():
    # make samples of some function and its gradient

    noise_size = 1e-4

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
    ipdb.set_trace()

def example_multivariate():

    # multivariate seems easy, it kind of just works

    key = jax.random.PRNGKey(0)

    N_pts = 256
    nx = 2

    noise_size = 1e-4

    key, subkey = jax.random.split(key)
    xs_f = jax.random.normal(subkey, (N_pts, nx))

    def f(x):
        return x[0]**2 + (x[1]-x[0]**2)**2

    key, subkey = jax.random.split(key)
    fs = jax.vmap(f)(xs_f) + noise_size * jax.random.normal(subkey, (N_pts,))


    def build_gp(params, X):
        # exp transform for >0 constraint.
        amp = np.exp(params['log_amp'])
        scale_vec = np.exp(params['log_scale'])

        # ...just the kernel specification syntax took some getting used to
        # linear will first scale by the diagonal matrix diag(scale_vec),
        # then pass that to the kernel, ExpSquared.
        kernel = amp * tinygp.transforms.Linear(
                scale_vec, tinygp.kernels.ExpSquared()
        )

        # small noise of same shape as x variable.
        diag = 1e-3 * np.ones_like(X)
        return tinygp.GaussianProcess(kernel, X, diag=noise_size**2)


    def nll(params, X, y):
        gp = build_gp(params, X)
        return -gp.log_probability(y)


    # no mean anymore.
    params = {
        'log_amp': np.log(1),
        'log_scale': np.zeros(nx),
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
    pl.plot(losses)
    pl.show()


    # do the inference, make a plot.
    trained_gp = build_gp(params, xs_f)

    extent = 3
    x_grid, y_grid = np.linspace(-extent, extent, 50), np.linspace(-extent, extent, 50)
    x_, y_ = np.meshgrid(x_grid, y_grid)
    X_pred = np.vstack((x_.flatten(), y_.flatten())).T
    y_true = jax.vmap(f)(X_pred).reshape(x_.shape)

    pred_gp = trained_gp.condition(fs, X_pred).gp
    y_pred = pred_gp.loc.reshape(y_true.shape)
    y_std = np.sqrt(pred_gp.variance.reshape(y_true.shape))

    fig, axs = pl.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].pcolor(x_, y_, y_pred, vmin=y_true.min(), vmax=y_true.max())
    axs[0].scatter(
            xs_f[:, 0], xs_f[:, 1],
            c=fs, ec='black',
            vmin=y_true.min(), vmax=y_true.max()
    )

    y_std = np.log(y_std)
    axs[1].pcolor(x_, y_, y_std, cmap='plasma', vmin=y_std.min(), vmax=y_std.max())
    pl.show()




    ipdb.set_trace()


class GradientKernel(tinygp.kernels.Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    # instead of X1 = (is_derivative, data) we want to be able to accomodate
    # several dimensions with gradients. probably the easiest way is to
    # include each directional derivative along [0...1...0] as a separate
    # data point. lots of redundant computation but probably the cleanest
    # code wise.

    # so let us construct the X1, X2 arrays like so:
    # g1, d1 = X1
    # where g1 is an integer with the following meaning:
    # 0: the corresponding data point is a normal observation, no # gradients
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

        '''
        univariate implementation for reference.
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
        '''


def example_gradient():

    # the standard gradient code only works for univariate derivatives.
    # there is this flag that indicates whether a data point has
    # observation or derivative info, and a custom kernel function that
    # decides based on this flag which derivative kernel to use.

    key = jax.random.PRNGKey(0)

    N_pts = 256
    nx = 2

    noise_size = 1e-2

    key, subkey = jax.random.split(key)
    xs_f = jax.random.normal(subkey, (N_pts, nx))

    def f(x):
        return x[0]**2 + (x[1]-x[0]**2)**2

    key, subkey = jax.random.split(key)
    fs = jax.vmap(f)(xs_f) + noise_size * jax.random.normal(subkey, (N_pts,))

    key, subkey = jax.random.split(key)
    grad_fs = jax.vmap(jax.grad(f))(xs_f) + noise_size * jax.random.normal(subkey, (N_pts, nx))


    # some reshaping magic to get data in required format.
    # we have: (N, nx) array of x values, (N,) array of observations, (N, # nx) array of observed gradients
    # we want:
    # - (N*(1+nx), nx) array of x values for all observations and gradients (just repeat)
    # - (N*(1+nx)) array of first observations, then gradients w.r.t x1, then x2, etc.
    extended_xs = np.kron(np.ones(3)[:, None], xs_f)  # it just works

    # .T will give [x1, x1, ..., x2, x2, ... ], otherwise it is [x1 x2 x3 ... x1 x2 x3...]
    extended_ys = np.concatenate([fs, grad_fs.T.reshape(-1)])
    extended_gradient_flags = np.kron(np.arange(1+nx), np.ones(N_pts)).astype(np.int8)

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

        kernel = GradientKernel(base_kernel)

        # small noise of same shape as x variable.
        # the gradient flag is read here as a global variable.
        # kind of shady right? what if we later have data with different
        # gradient flags?
        return tinygp.GaussianProcess(kernel, (X, g), diag=noise_size**2)


    def nll(params, X, y, g):
        gp = build_gp(params, X, g)
        return -gp.log_probability(y)


    # no mean anymore.
    params = {
        'log_amp': np.log(1),
        'log_scale': np.zeros(nx),
    }

    nll_value_grad = jax.jit(jax.value_and_grad(nll, argnums=0))

    # fs_expanded = np.column_stack([fs, gradient_flag])

    gradient_flag = np.zeros(xs_f.shape[0], dtype=np.int8)

    v, _ = nll_value_grad(params, xs_f, fs, gradient_flag)
    print('nll and params before optimisation:')
    print(v)
    print(params)

    # set up optimiser for neg log likelihood
    opti = optax.adam(learning_rate=.05)
    opt_state = opti.init(params)

    losses = []
    for i in tqdm.tqdm(range(100)):
        loss_val, grads = nll_value_grad(params, xs_f, fs, gradient_flag)
        losses.append(loss_val)
        updates, opt_state = opti.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    v, _ = nll_value_grad(params, xs_f, fs, gradient_flag)
    print('nll and params after optimisation:')
    print(v)
    print(params)
    pl.figure()
    pl.plot(losses)


    # do the inference
    # this is basically the same as during training, but with optimal # params.
    trained_gp = build_gp(params, xs_f, gradient_flag)

    extent = 3
    x_grid, y_grid = np.linspace(-extent, extent, 50), np.linspace(-extent, extent, 50)
    x_, y_ = np.meshgrid(x_grid, y_grid)
    X_pred = np.vstack((x_.flatten(), y_.flatten())).T
    y_true = jax.vmap(f)(X_pred).reshape(x_.shape)

    pred_grad_flag = np.zeros(X_pred.shape[0], dtype=np.int8)
    pred_gp = trained_gp.condition(fs, (X_pred, pred_grad_flag)).gp
    y_pred = pred_gp.loc.reshape(y_true.shape)
    y_std = np.sqrt(pred_gp.variance.reshape(y_true.shape))

    # make a plot.
    fig, axs = pl.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].pcolor(x_, y_, y_pred, vmin=y_true.min(), vmax=y_true.max())
    axs[0].scatter(
            xs_f[:, 0], xs_f[:, 1],
            c=fs, ec='black',
            vmin=y_true.min(), vmax=y_true.max()
    )
    axs[0].set_title('Data & GP posterior mean')

    pc = axs[1].pcolor(x_, y_, y_std, cmap='plasma', vmin=y_std.min(), vmax=y_std.max())
    fig.colorbar(pc, ax=axs[1])
    axs[1].set_title('GP posterior std deviation, no gradient info')


    # do it again, but with gradient data.
    # build the gradient data


    optimize_again = False
    if optimize_again:
        opti = optax.adam(learning_rate=.05)
        opt_state = opti.init(params)

        losses = []
        for i in tqdm.tqdm(range(100)):
            loss_val, grads = nll_value_grad(
                    params, extended_xs,
                    extended_ys, extended_gradient_flags
            )
            losses.append(loss_val)
            updates, opt_state = opti.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        v, _ = nll_value_grad(params, extended_xs, extended_ys, extended_gradient_flags)
        print('nll and params after 2nd optimisation:')
        print(v)
        print(params)
        pl.figure()
        pl.plot(losses)

    trained_gp = build_gp(params, extended_xs, extended_gradient_flags)

    extent = 3
    x_grid, y_grid = np.linspace(-extent, extent, 50), np.linspace(-extent, extent, 50)
    x_, y_ = np.meshgrid(x_grid, y_grid)
    X_pred = np.vstack((x_.flatten(), y_.flatten())).T
    y_true = jax.vmap(f)(X_pred).reshape(x_.shape)

    pred_grad_flag = np.zeros(X_pred.shape[0], dtype=np.int8)
    pred_gp = trained_gp.condition(extended_ys, (X_pred, pred_grad_flag)).gp
    y_pred = pred_gp.loc.reshape(y_true.shape)
    y_std_new = np.sqrt(pred_gp.variance.reshape(y_true.shape))

    # make a plot.
    fig, axs = pl.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].pcolor(x_, y_, y_pred, vmin=y_true.min(), vmax=y_true.max())
    axs[0].scatter(
            xs_f[:, 0], xs_f[:, 1],
            c=fs, ec='black',
            vmin=y_true.min(), vmax=y_true.max()
    )
    axs[0].set_title('Data & GP posterior mean')

    pc = axs[1].pcolor(x_, y_, y_std_new, cmap='plasma', vmin=y_std.min(), vmax=y_std.max())
    fig.colorbar(pc, ax=axs[1])
    axs[1].set_title('GP posterior std deviation, with gradient info')

    pl.figure()
    diff = y_std_new - y_std
    vmax = np.max(np.abs(diff))
    pc = pl.pcolor(diff, cmap='seismic', vmin=-vmax, vmax=vmax)
    pl.gca().set_title('difference in std before and after including gradient info')
    pl.colorbar(pc, ax=pl.gca())

    pl.show()

    ipdb.set_trace()


if __name__ == '__main__':

    # example_univariate()
    # example_multivariate()
    example_gradient()


