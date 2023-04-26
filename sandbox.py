#!/usr/bin/env python

# jax!
import jax
import jax.numpy as np

# jax libraries
import equinox as eqx
import optax

# cheating on equinox :/
import flax
from flax import linen as nn
from typing import Sequence, Optional

# other, trivial stuff
import numpy as onp
import matplotlib.pyplot as pl
import matplotlib

import ipdb

import time
from tqdm import tqdm


@jax.jit
def generate_data(key, in_dims=2, n_pts=100, P=None):
    data_key, function_key = jax.random.split(key)

    if P is None:
        # quadratic function x.T @ P @ x
        Phalf = jax.random.normal(function_key, (in_dims, in_dims))
        P = Phalf.T @ Phalf  # should be positive definite
        # print(P)

    xs = jax.random.normal(data_key, shape=(n_pts, in_dims))
    ys = jax.vmap(lambda x: x.T @ P @ x, 0)(xs)

    # return training and test data.
    train_split = 0.8
    train_idx = int(train_split * xs.shape[0])

    xs_train = xs[0:train_idx, :]
    ys_train = ys[0:train_idx]

    xs_test = xs[train_idx:, :]
    ys_test = ys[train_idx:]

    return xs_train, ys_train, xs_test, ys_test

class my_nn(eqx.Module):

    layers: list

    # basic usage, should give network output.
    #   nn = my_nn(jax.random.PRNGKey(0));
    #   nn(jax.numpy.array([.1, .2])

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)

        # it would be very nice to write the activation functions here too
        # but then the tree is somehow not compatible with jax datatypes
        # and it complains.
        self.layers = [
                eqx.nn.Linear(2, 16, key=key1),
                eqx.nn.Linear(16, 16, key=key1),
                eqx.nn.Linear(16, 8, key=key2),
                eqx.nn.Linear(8, 1, key=key2),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))

        # last line was originally: return self.layers[-1](x), another linear layer.
        # we might need this instead if we want to enforce positive definiteness of value functions.
        return self.layers[-1](x)
        return x.T @ x

    def without_last_layer(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))

        return x  # the input to the last layer




@jax.jit
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)


def plot_nn(nn, xlim, ylim, N_disc=50):

    x = np.linspace(*xlim, num=N_disc)
    y = np.linspace(*ylim, num=N_disc)
    X, Y = np.meshgrid(x, y)
    xy_vec = np.stack([X.ravel(), Y.ravel()]).T
    zs = jax.vmap(nn)(xy_vec)

    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, zs.reshape(X.shape), cmap='summer', alpha=0.4)
    return ax

def plot_nn_manualfig(nn, xlim, ylim, ax, N_disc=50):

    x = np.linspace(*xlim, num=N_disc)
    y = np.linspace(*ylim, num=N_disc)
    X, Y = np.meshgrid(x, y)
    xy_vec = np.stack([X.ravel(), Y.ravel()]).T
    # ipdb.set_trace()
    zs = jax.vmap(nn)(xy_vec)

    if len(zs.shape) == 2 and zs.shape[1] > 1:
        # we have a 'nn' callable with several outputs.
        # plot them all.
        for i_output in range(zs.shape[1]):
            ax.plot_surface(X, Y, zs[:, i_output].reshape(X.shape), alpha=0.08)
    else:
        # the usual.
        ax.plot_surface(X, Y, zs.reshape(X.shape), cmap='summer', alpha=0.8)

def plot_data(xs, ys, ax, label):
    # assumes that ax has already been created, see plot_nn

    assert xs.shape[1] == 2, 'need shape (n, 2)'
    ax.scatter3D(xs[:, 0], xs[:, 1], ys, label=label)



def learningrate_experiment(load=None):

    # try a selection of learning rates and plot the corresponding
    # train and test errors.

    N_lr = 50
    N_train = 500

    learningrates = onp.logspace(-2.5, -1, N_lr)
    train_losses = onp.zeros((N_train, N_lr))
    test_losses = onp.zeros((N_train, N_lr))

    if load is None:
        # do a new experiment
        key, datakey = jax.random.split(jax.random.PRNGKey(0))

        xs_train, ys_train, xs_test, ys_test = generate_data(
                datakey,
                in_dims=2,
                n_pts=1000
        )

        key, modelkey = jax.random.split(key)

        for i_lr, lr in enumerate(learningrates):
            print(f'learning rate = {lr} ({i_lr}/{N_lr})')

            # re-instantiate NN with same init
            nn = my_nn(modelkey)

            optim = optax.adam(lr)

            @jax.jit
            def make_train_step(model, opt_state, x, y):

                loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
                updates, opt_state = optim.update(grads, opt_state)
                model = eqx.apply_updates(model, updates)

                return model, opt_state, loss


            opt_state = optim.init(nn)

            # standard training loop.
            for i_train in tqdm(range(N_train)):
                nn, opt_state, old_loss_val = make_train_step(nn, opt_state, xs_train, ys_train)
                train_losses[i_train, i_lr] = old_loss_val
                test_losses[i_train, i_lr] = loss_fn(nn, xs_test, ys_test)

        t = int(time.time())
        np.save(f'./learningrate_experiment_outputs/test_{t}.npy', test_losses)
        np.save(f'./learningrate_experiment_outputs/train_{t}.npy', train_losses)

    else:
        t = load
        print(f'loading experiment {t}')
        test_losses = np.load(f'./learningrate_experiment_outputs/test_{t}.npy')
        train_losses = np.load(f'./learningrate_experiment_outputs/train_{t}.npy')
        assert test_losses.shape == train_losses.shape, 'wrong shapes!'
        N_train, N_lr = train_losses.shape




    # plot the stuff.
    plot_3d = True
    xx, yy = np.meshgrid(learningrates, np.arange(N_train))

    if plot_3d:
        fig = pl.figure()
        ax = fig.add_subplot(projection='3d')
        # surf = ax.plot_surface(np.log10(xx), yy, np.log10(train_losses), label='train loss', alpha=0.2)
        # # hack to make plot work with legend
        # surf._edgecolors2d = surf._edgecolor3d
        # surf._facecolors2d = surf._facecolor3d

        # only test loss is interesting...
        surf = ax.plot_surface(np.log10(xx), yy, np.log10(test_losses),
                cmap='viridis', linewidth=0, antialiased=False)

        ax.set_xlabel('log10(learning rate)')
        ax.set_ylabel('training iteration')
        ax.set_zlabel('log10(loss)')

        # pl.legend()
        pl.show()

    else:
        pl.contourf(np.log10(xx), yy, np.log10(test_losses), 100, cmap='jet_r')
        pl.xlabel('log10(learning rate)')
        pl.ylabel('training iteration')
        pl.show()

def sample_and_train(key, batch_size=512, n_pts_max=1e6, wall_time_max=60, do_plot=False):
    # train with continously generated data, similar to later when we want to solve PDEs.

    key = jax.random.PRNGKey(0)

    # first, generate 200 test points.
    n_pts = 200

    # parameterisation of quadratic function to fit.
    in_dims = 2
    key, function_key = jax.random.split(key)
    Phalf = jax.random.normal(function_key, (in_dims, in_dims))
    P = Phalf.T @ Phalf

    # cannot @jit because variable shape...
    def generate_data(key, n_pts):
        xs = jax.random.normal(key, shape=(n_pts, in_dims))
        ys = jax.vmap(lambda x: x.T @ P @ x, 0)(xs)
        return xs, ys

    key, data_key = jax.random.split(key)
    test_x, test_y = generate_data(data_key, n_pts)

    key, model_key = jax.random.split(key)
    nn = my_nn(model_key)

    n_pts_seen = 0

    @jax.jit
    def make_train_step(model, opt_state, x, y):

        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss


    def plot_and_save(nn, name, i, ts, encountered_losses, testlosses, save=True):
        extent = 2.5

        fig = pl.figure(figsize=(8, 4))
        ax = fig.add_subplot(121, projection='3d')
        ax.view_init(elev=None, azim=i/10000, roll=None)

        plot_nn_manualfig(nn, [-extent, extent], [-extent, extent], ax)
        plot_nn_manualfig(nn.without_last_layer, [-extent, extent], [-extent, extent], ax)

        plot_data(test_x, test_y, ax, 'test data')

        ax = fig.add_subplot(122)

        # where_valid = ts != 0
        where_valid = testlosses != 0

        ax.semilogy(encountered_losses[where_valid], label='MC training loss', alpha=0.3)
        ax.semilogy(testlosses[where_valid], label='test loss')
        pl.legend()
        pl.ylabel('test loss')
        pl.xlabel('training iteration')

        if save:
            pl.savefig(name, dpi=300)
            print(f'saved "{name}"')
        else:
            pl.show()

        pl.close()


    optim = optax.adam(0.001)
    opt_state = optim.init(nn)

    # standard training loop.
    # for i_train in tqdm(range(N_train)):
    #     nn, opt_state, old_loss_val = make_train_step(nn, opt_state, xs_train, ys_train)
    #     train_losses[i_train, i_lr] = old_loss_val
    #     test_losses[i_train, i_lr] = loss_fn(nn, xs_test, ys_test)

    pts_seen_steps = onp.arange(0, n_pts_max, batch_size)

    encountered_losses = onp.zeros_like(pts_seen_steps)
    test_losses = onp.zeros_like(pts_seen_steps)
    wall_times = onp.zeros_like(pts_seen_steps)

    t_init = time.time()
    for i, pts_seen in enumerate(tqdm(pts_seen_steps)):


        key, datakey = jax.random.split(key)
        xs, ys = generate_data(datakey, batch_size)

        nn, opt_state, encountered_loss = make_train_step(nn, opt_state, xs, ys)

        encountered_losses[i] = encountered_loss
        test_losses[i] = loss_fn(nn, test_x, test_y)
        wall_times[i] = time.time() - t_init

        # turned off again.
        if False and i%10 == 0:
            plot_and_save(
                    nn,
                    f'./equinox_figs/learningplot_{i:05d}.png',
                    pts_seen,
                    wall_times,
                    encountered_losses,
                    test_losses,
            )

        # stop training after that many seconds.
        if wall_times[i] >= wall_time_max:
            print(f'Training stopped after {wall_times[i]:.2f} s with {int(pts_seen)} points seen')
            break

    if do_plot:

        # pl.plot(pts_seen_steps, encountered_losses, label='loss at train data')
        # pl.plot(pts_seen_steps, test_losses, label='test loss')
        # pl.xlabel('data points seen')
        # pl.legend()
        # pl.show()

        plot_and_save(nn, 'fuck all', pts_seen, wall_times, encountered_losses, test_losses, save=False)

    return pts_seen_steps, encountered_losses, test_losses, wall_times, nn

def batchsize_sampletrain_experiment():

    # todo - maybe it makes more sense to plot this stuff against wall time
    # to find the practical sweet spot?
    batchsizes = 2**np.arange(7, 11)
    print(f'trying out batch sizes:\n{batchsizes}')
    bs_total = len(batchsizes)

    # use the same key all the time for same test data
    key = jax.random.PRNGKey(0)

    cmap = matplotlib.colormaps.get_cmap('viridis')

    for i, bs in enumerate(batchsizes):
        print(f'starting batch size {bs}')
        pts_seen, enc_loss, test_loss, wall_times, _ = sample_and_train(
                key,
                batch_size=bs,
                n_pts_max=1e8,  # many - we have wall time.
                wall_time_max=300
        )

        where_valid = wall_times != 0
        pl.plot(wall_times[where_valid], test_loss[where_valid], c=cmap(i/bs_total), label=f'batch size {bs}')

    pl.xlabel('wall time')
    pl.ylabel('test loss')
    pl.legend()
    pl.show()

def not_training_debugging():
    # when the last layer is return self.layers[-1](x) instead of return x.T @ x,
    # it somehow fails to train. here we find out why.

    # train with continously generated data, similar to later when we want to solve PDEs.

    key = jax.random.PRNGKey(0)

    # first, generate 200 test points.
    n_pts = 200

    # parameterisation of quadratic function to fit.
    in_dims = 2
    key, function_key = jax.random.split(key)
    Phalf = jax.random.normal(function_key, (in_dims, in_dims))
    P = Phalf.T @ Phalf

    def generate_data(key, n_pts):
        xs = jax.random.normal(key, shape=(n_pts, in_dims))
        ys = jax.vmap(lambda x: x.T @ P @ x, 0)(xs)
        return xs, ys

    key, data_key = jax.random.split(key)
    test_x, test_y = generate_data(data_key, n_pts)

    @jax.jit
    def make_train_step(model, opt_state, x, y):

        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, grads


    key, model_key = jax.random.split(key)
    nn = my_nn(model_key)

    optim = optax.adamaxw(0.0001)
    opt_state = optim.init(nn)

    steps = 2000
    batchsize = 1024

    encountered_losses = onp.zeros(steps,)
    test_losses = onp.zeros(steps,)
    flat_grads = []  # no clue how many parameters there are exactly...
    flat_params = []

    for i in tqdm(range(steps)):
        key, datakey = jax.random.split(key)
        xs, ys = generate_data(datakey, batchsize)

        nn, opt_state, encountered_loss, grads = make_train_step(nn, opt_state, xs, ys)

        grads_as_array = np.concatenate(
            jax.tree_util.tree_map(lambda z: z.flatten(),
                jax.tree_util.tree_flatten(grads)[0],
            )
        )
        flat_grads.append(grads_as_array)
        ipdb.set_trace()

        params_as_array = np.concatenate(
            jax.tree_util.tree_map(lambda z: z.flatten(),
                jax.tree_util.tree_flatten(nn)[0],
            )
        )

        flat_params.append(params_as_array)
        # ipdb.set_trace()


        encountered_losses[i] = encountered_loss
        test_losses[i] = loss_fn(nn, test_x, test_y)



    extent = 2.5

    fig = pl.figure()
    ax = fig.add_subplot(221, projection='3d')

    plot_nn_manualfig(nn, [-extent, extent], [-extent, extent], ax)
    plot_nn_manualfig(nn.without_last_layer, [-extent, extent], [-extent, extent], ax)

    plot_data(test_x, test_y, ax, 'test data')

    ax = fig.add_subplot(222)

    ax.semilogy(encountered_losses, label='MC training loss', alpha=0.3)
    ax.semilogy(test_losses, label='test loss')
    pl.legend()
    pl.ylabel('test loss')
    pl.xlabel('training iteration')

    ax = fig.add_subplot(223)
    pl.ylabel('gradients & magnitude')

    flat_grad_array = np.array(flat_grads)  # shape: n_train, n_params
    pl.plot(flat_grad_array, c='g', alpha=0.02)

    pl.plot(np.sqrt(np.square(flat_grad_array).sum(axis=1)), c='b')

    ax = fig.add_subplot(224)

    flat_param_array = np.array(flat_params)  # shape: n_train, n_params
    pl.plot(flat_param_array, c='g', alpha=0.02)


    pl.show()


def selfcontained_flax_example():

    key = jax.random.PRNGKey(0)

    class my_nn_flax(nn.Module):
        features: Sequence[int]
        output_dim: Optional[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(features=feat)(x)
                x = nn.softmax(x)

            if self.output_dim is not None:
                x = nn.Dense(features=self.output_dim)(x)
            return x


    # parameterisation of quadratic function to fit.
    in_dims = 2
    key, function_key = jax.random.split(key)
    Phalf = jax.random.normal(function_key, (in_dims, in_dims))
    P = Phalf.T @ Phalf

    def generate_data(key, n_pts):
        xs = jax.random.normal(key, shape=(n_pts, in_dims))
        ys = jax.vmap(lambda x: x.T @ P @ x, 0)(xs)
        return xs, ys

    key, data_key = jax.random.split(key)
    train_x, train_y = generate_data(data_key, 512)

    model = my_nn_flax(features=(32, 32, 32), output_dim=1)
    key, subkey = jax.random.split(key)

    # only works with jit if we include model fixed, not as argument.
    def point_loss(params, x, y):
        y_pred = model.apply(params, x)
        return (y_pred - y)**2

    def loss(params, xs, ys):
        losses = jax.vmap(point_loss, in_axes=(None, 0, 0))(params, xs, ys)
        return np.mean(losses)


    key, testdata_key = jax.random.split(key)
    test_x, test_y = generate_data(testdata_key, 512)

    @jax.jit
    def eval_test_loss(params):
        return loss(params, test_x, test_y)


    # optim = optax.adam(learning_rate=0.01)
    nstart = 1000
    lr_end = 0.001
    lr_fct = lambda i: np.minimum(0.02, 0.02/(i/3000))
    optim = optax.adam(learning_rate=lr_fct)

    loss_val_grad = jax.value_and_grad(loss, argnums=0)

    @jax.jit
    def update_step(xs, ys, opt_state, params):

        loss_val, loss_grad = loss_val_grad(params, xs, ys)

        updates, opt_state = optim.update(loss_grad, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params, loss_val

    key, initkey = jax.random.split(key)

    params = model.init(subkey, np.zeros(2))
    opt_state = optim.init(params)

    N_train = 10000
    testloss_interval = 10
    data_regen_interval = 100

    trainlosses = onp.zeros(N_train,)
    testlosses = onp.zeros(N_train,)

    for i in tqdm(range(N_train)):


        # continuously sample data.
        if i % data_regen_interval == 0:
            key, data_key = jax.random.split(key)
            train_x, train_y = generate_data(data_key, 512)

        opt_state, params, loss_value = update_step(
                train_x, train_y, opt_state, params)


        trainlosses[i] = loss_value

        if i%testloss_interval == 0:
            testlosses[i:i+testloss_interval] = eval_test_loss(params)


    pl.plot(trainlosses, label='training loss')
    pl.plot(testlosses, label='test loss')
    pl.plot(np.arange(N_train), lr_fct(np.arange(N_train)), label='learning rate')
    pl.legend()
    pl.show()

    ipdb.set_trace()




def flax_pde_example(n_unroll=1, factor=0.5, do_plot=False, save=False):
    key = jax.random.PRNGKey(0)

    class my_nn_flax(nn.Module):
        features: Sequence[int]
        output_dim: Optional[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(features=feat)(x)
                x = nn.softplus(x)

            if self.output_dim is not None:
                x = nn.Dense(features=self.output_dim)(x)
            return x


    # no train/test data.
    # instead we want to solve the PDE:
    # grad_x f(x) = 2x
    # f(0) = 0
    # to get the solution f(x) = x.T x.

    # input dim is assigned when passing first input at init.
    model = my_nn_flax(features=(32, 32, 32), output_dim=1)
    key, subkey = jax.random.split(key)

    # call still with (params, x)
    # jacrev maybe faster for wide jacobian? dunno...
    #   -> yes, about 2x faster than jacfwd
    model_output_grad = jax.jacrev(model.apply, argnums=1)


    # only works with jit if we include model fixed, not as argument.
    def point_pde_loss(params, x):
        pde_lhs = model_output_grad(params, x)
        pde_rhs = 2 * x
        return np.linalg.norm(pde_lhs - pde_rhs)

    def MC_pde_loss(params, xs):
        losses = jax.vmap(point_pde_loss, in_axes=(None, 0))(params, xs)
        return np.mean(losses)


    dims = 6
    zero = np.zeros(dims)
    def boundary_loss(params):
        return model.apply(params, zero)**2


    # for quadratic function:
    def full_loss(params, xs):
        # no idea how to balance the two. in principle they are both constraints...
        # could also enforce the boundary constraint structurally: f(x) = g(x)-g(0)
        # but it seems to not matter too much.
        return 10 * boundary_loss(params).reshape() + MC_pde_loss(params, xs)


    def sample_xs(key, N):
        return jax.random.normal(key, (N, dims))

    key, test_key = jax.random.split(key)
    test_x = sample_xs(test_key, 128)

    @jax.jit
    def eval_test_loss(params):
        return full_loss(params, test_x)


    @jax.jit
    def update_step(xs, opt_state, params):


        # 'unrolling' several iterations at the same time makes training
        # a lot faster per iteration but leads to longer jit times in the
        # start
        # with proper jax.lax control flow the jit is much shorter
        # implemented in update_step_multiple.
        for j in range(n_unroll):
            loss_val, loss_grad = jax.value_and_grad(full_loss)(params, xs)
            updates, opt_state = optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)

        def body_fun(opt_state, params, loss_val, test_loss):
            loss_val, loss_grad = jax.value_and_grad(full_loss)(params, xs)
            updates, opt_state = optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)
            test_loss = eval_test_loss(params)
            return opt_state, params, loss_val, test_loss


        test_loss = eval_test_loss(params)

        return opt_state, params, loss_val, test_loss

    def body_fun(i, args):
        opt_state, params, loss_val, test_loss, xs = args
        loss_val, loss_grad = jax.value_and_grad(full_loss)(params, xs)
        updates, opt_state = optim.update(loss_grad, opt_state)
        params = optax.apply_updates(params, updates)
        test_loss = eval_test_loss(params)
        return (opt_state, params, loss_val, test_loss, xs)

    @jax.jit
    def update_step_multiple(xs, opt_state, params):

        # multiple, 'unrolled' training steps in single jit region
        # compiles quite quickly with lax fori_loop, but requires
        # a separate function and some unnecessary arguments, because
        # input and output to body_fun need to be identical.

        test_loss = loss_val = 0
        init_args = (opt_state, params, loss_val, test_loss, xs)

        out_args = jax.lax.fori_loop(0, n_unroll, body_fun, init_args)

        (opt_state, params, loss_val, test_loss, xs) = out_args

        return opt_state, params, loss_val, test_loss






    # test the stuff....
    # @jax.jit
    # def eval_MC_loss(seed):
    #     xs = jax.random.normal(jax.random.PRNGKey(seed), (256, 2))
    #     return MC_pde_loss(params, xs)

    # results1 = jax.vmap(eval_MC_loss, 0)(np.arange(10000))

    # print(results1)
    # ipdb.set_trace()

    key, initkey = jax.random.split(key)
    params = model.init(subkey, np.zeros(dims))

    lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value = 0.01,
            peak_value = 0.2,
            warmup_steps = 500,
            transition_steps = 40,
            decay_rate = 0.9,
            end_value=1e-12
    )

    N_train = 5000
    # to remove unnecessary overparameterization...
    decay_per_iter = .9998
    transition_steps= (N_train // 10) * n_unroll;
    lr_schedule = optax.exponential_decay(
            init_value=0.05,
            transition_steps=transition_steps,
            decay_rate=decay_per_iter ** transition_steps,
            transition_begin=(N_train // 10) * n_unroll,
            staircase=False
    )

    # lr_schedule = optax.piecewise_constant_schedule(
    # lr_schedule = optax.piecewise_interpolate_schedule(
    #         interpolate_type='linear',
    #         init_value=0.1,
    #         boundaries_and_scales = {
    #             500   * n_unroll: factor,
    #             1000  * n_unroll: factor,
    #             1500  * n_unroll: factor,
    #             2000  * n_unroll: factor,
    #             3000  * n_unroll: factor,
    #             4000  * n_unroll: factor,
    #             5000  * n_unroll: factor,
    #             10000 * n_unroll: factor,
    #             15000 * n_unroll: factor,
    #             20000 * n_unroll: factor,
    #         }
    # )


    # somehow that compensation for n_unroll does not really work... dunno
    # why. anyway, putting several gradient steps in the jit region is
    # probably not a huuuuge speedup anyways, could probably achieve much
    # the same effect using larger batch sizes.
    # optim = optax.adam(learning_rate=lambda i: lr_schedule(i//n_unroll))
    optim = optax.adam(learning_rate=lr_schedule)
    opt_state = optim.init(params)

    data_regen_interval = 1


    # incorrect if n_unroll > 1.
    # if do_plot:
    #     pl.plot(np.arange(N_train), lr_schedule(np.arange(N_train)), label='learning rate schedule')
    #     pl.legend()
    #     pl.show()

    trainlosses = onp.zeros(N_train,)
    testlosses = onp.zeros(N_train,)
    learningrates = onp.zeros(N_train,)

    for i in tqdm(range(N_train)):

        # constantly sample data.
        if i==0 or i % data_regen_interval == 0:
            key, data_key = jax.random.split(key)
            train_x = sample_xs(data_key, 128)

        opt_state, params, loss_value, test_loss_value = update_step_multiple(
                train_x, opt_state, params
        )

        trainlosses[i] = loss_value
        testlosses[i] = test_loss_value
        learningrates[i] = lr_schedule(opt_state[1].count)


    if do_plot:
        # figzize = (w, h) in inches
        fig = pl.figure(figsize=(10, 7))
        ax = pl.subplot(221, projection='3d')

        func = None
        if dims == 2:
            func = lambda x: model.apply(params, x)
        if dims > 2:
            func = lambda x: model.apply(params, np.concatenate([x, np.zeros((dims-2))]))
        assert func is not None

        extent = 4
        plot_nn_manualfig(func, (-extent, extent), (-extent, extent), ax)

        pl.subplot(222)
        pl.semilogy(trainlosses, label='training loss', alpha=0.5)
        pl.semilogy(testlosses, label='test loss')
        pl.semilogy(learningrates, label='learning rate', linestyle='--',
                color='black', alpha=0.3)
        pl.grid(axis='both')
        pl.legend()

        def plot_sol_random_dir(model, key):
            # plot solution vs closed form in random directions...
            key, plotkey = jax.random.split(key)

            direction = jax.random.normal(plotkey, (dims,))
            direction = direction / np.linalg.norm(direction)

            xs_plot = np.linspace(-extent, extent, 101)
            xs = np.outer(xs_plot, direction)

            ys_target = jax.vmap(lambda x: x.T @ x)(xs)
            # ipdb.set_trace()

            model_full = lambda x: model.apply(params, x)

            # ys = [model_full(x) for x in xs]
            ys = jax.vmap(model_full)(xs).flatten()
            pl.plot(xs_plot, ys, label='neural solution')
            pl.plot(xs_plot, ys_target, label='closed form solution')
            with np.printoptions(precision=2):
                pl.gca().set_title(f'solution in direction {direction}')
            pl.legend()

        pl.subplot(223)
        key, plotkey = jax.random.split(key)
        plot_sol_random_dir(model, plotkey)

        pl.subplot(224)
        key, plotkey = jax.random.split(key)
        plot_sol_random_dir(model, plotkey)

        if save:
            figname = f'./unroll_figs/{factor}.png'
            print(f'saving "{figname}"')
            pl.savefig(figname, dpi=300)
            pl.close()
        else:
            pl.show()


    return np.mean(testlosses[-10:])


def hjb_solver_pinn(f, l, h, nx, nu):
    '''
    f: dynamics. vector-valued function of t, x, u.
    l: cost. scalar-valued function of t, x, u.
    h: terminal cost. scalar-valued function of x.
    '''

    # define NN first.
    key = jax.random.PRNGKey(0)

    class my_nn_flax(nn.Module):
        features: Sequence[int]
        output_dim: Optional[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(features=feat)(x)
                x = nn.softmax(x)

            if self.output_dim is not None:
                x = nn.Dense(features=self.output_dim)(x)
            return x



    # input dim is assigned when passing first input at init.
    V_nn = my_nn_flax(features=(32, 32, 32), output_dim=1)
    key, subkey = jax.random.split(key)


    # we want to find the value function V.
    # for that we want to approximately satisfy the hjb equation:
    #    0 = V_t(t, x) + inf_u { l(t, x, u) + V_x(t, x).T @ f(t, x, u) }
    #    V(x, T) = h(x)

    # for this first case let us consider
    #    f(t, x, u) = f_tx(t, x) + g(t, x) @ u
    #    l(t, x, u) = l_tx(t, x) + u.T @ R @ u.
    # so a control affine system with cost quadratic in u. this will make
    # the inner minimization simpler:

    #       argmin_u l(t, x, u) + V_x(t, x).T @ f(t, x, u)
    #     = argmin_u l_x(x) + u.T @ R @ u + V_x(t, x).T @ (f_tx(t, x) + g(t, x) @ u)
    #     = argmin_u          u.T @ R @ u + V_x(t, x).T @ (             g(t, x) @ u)

    # and, with A = V_x(t, x).T @ g(t, x):

    #     = argmin_u u.T @ R @ u + A @ u

    # which is an unconstrained convex QP, solved by setting the gradient to zero.

    #     = u s.t. 0 = (R + R.T) u + A
    #     = solution of linear system (R + R.T, -A)

    # this is implemented in the following - the pointwise minimization over u of the hamiltonian.
    def find_u_star_matrices(R, A):
        # A is a row vector here...
        return np.linalg.solve(R + R.T, -A.T)

    def find_u_star_functions(f, l, V, t, x):

        # assuming that l is actually of the form l(t, x, u) = l_tx(t, x) + u.T @ R @ u,
        # the hessian R is independent of u. R should be of shape (nu, nu).
        zero_u = np.zeros((1, 1))
        R = jax.hessian(l, argnums=2)(t, x, zero_u).reshape(1, 1)

        grad_V_x = jax.jacobian(V, argnums=1)(t, x).reshape(1, nx)
        grad_f_u = jax.jacobian(f, argnums=2)(t, x, zero_u).reshape(nx, nu)
        A = grad_V_x @ grad_f_u        # should have shape (1, nu)
        import ipdb; ipdb.set_trace()

        return find_u_star_matrices(R, A)

    # check. if u is 2d, we need 2x2 R and 1x2 A
    R = np.array([[1., 0], [0, 2]])
    A = np.array([[0.1, 0.3]])
    x = np.array([[1., 1.]]).T
    print(find_u_star_matrices(
        R, A
    ))


    # dummy inputs.
    V = lambda t, x: (x.T @ np.eye(2) @ x).reshape(())
    t = 0

    print(find_u_star_functions(f, l, V, t, x))



def hjb_pinn_experiment_simple():

    # simple control system. double integrator with friction term.
    def f(t, x, u):
        fricterm = (np.array([[0, 1]]) @ x)**3
        return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ u - fricterm

    def l(t, x, u):
        Q = np.eye(2)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 10 * np.eye(2)
        return x.T @ Qf @ x

    hjb_solver_pinn(f, l, h, 2, 1)

if __name__ == '__main__':

    # hjb_pinn_experiment_simple()

    flax_pde_example(n_unroll=1, do_plot=True, save=False)

    # selfcontained_flax_example()
    # not_training_debugging()
    # batchsize_sampletrain_experiment()

    import sys; sys.exit()

    pts_seen, _, test_loss, wall_ts, nn = sample_and_train(
            jax.random.PRNGKey(0),
            batch_size=64,
            n_pts_max=1e5,
            wall_time_max=120,
            do_plot=True
    )

    ipdb.set_trace()
