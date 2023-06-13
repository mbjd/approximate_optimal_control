# jax
import jax
import jax.numpy as np

# other, trivial stuff
import numpy as onp

import tk as tkinter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb

from functools import partial


def plot_fct(f, xbounds, ybounds, N_disc = 201):

    # just the business logic: construct grid, reshape, evaluate, plot
    # create the plot before and show it after this function

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    xgrid = np.linspace(xmin, xmax, N_disc)
    ygrid = np.linspace(ymin, ymax, N_disc)

    xx, yy = np.meshgrid(xgrid, ygrid)

    all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    all_outputs = jax.vmap(f)(all_inputs).reshape(xx.shape)  # need a lot of memory!
    pl.pcolor(xx, yy, all_outputs, cmap='viridis')
    # pl.pcolor(xx, yy, all_outputs, cmap='jet')


def plot_2d_V(V_nn_wrapper, nn_params, tbounds, xbounds):

    tmin, tmax = tbounds
    xmin, xmax = xbounds  # same bounds for both state dims for now.


    # make a figure with a slider to get a feel of the distribution of particles
    # over the state-space at some (adjustable) time.


    @partial(jax.jit, static_argnames=['return_grid'])
    def eval_V_grid(t, return_grid=False):
        N_disc = 51
        xgrid = ygrid = np.linspace(xmin, xmax, N_disc)
        xx, yy = np.meshgrid(xgrid, ygrid)

        nn_inputs = np.concatenate([
            t * np.ones((N_disc*N_disc, 1)),
            xx.reshape(N_disc*N_disc, 1),
            yy.reshape(N_disc*N_disc, 1),
        ], axis=1)

        zz = V_nn_wrapper(nn_inputs, nn_params)

        if return_grid:
            return xx, yy, zz.reshape(N_disc, N_disc)
        else:
            return zz.reshape(N_disc, N_disc)

    t_init = tmin

    xx, yy, V = eval_V_grid(t_init, return_grid=True)

    fig = pl.figure('value function')
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xx, yy, V)

    ax = fig.add_subplot(111)
    line = ax.contourf(xx, yy, V, levels=np.arange(30))

    fig.subplots_adjust(bottom=.25)
    ax_time = fig.add_axes([.25, .1, .65, .03])
    time_slider = matplotlib.widgets.Slider(
        ax=ax_time,
        label='time [s]',
        valmin=tmin,
        valmax=tmax,
        valinit=t_init,
    )

    def update(val):
        t_plot = time_slider.val

        print('clearing...')
        ax.clear()

        print('jitting/evaluating...')
        V = eval_V_grid(t_plot)
        print('plotting')
        line = ax.contourf(xx, yy, V, levels=np.arange(30))
        pl.draw()

    time_slider.on_changed(update)


def plot_V_over_time(V_nn_wrapper, nn_params, all_sols, all_ts, where_resampled, algo_params):

    # plots V at some random set of states, as a function of time, so we
    # can qualitatively assess how/whether the value function converges
    # to a steady state with growing horizon.

    N_states = 50

    nx = algo_params['x_sample_cov'].shape[0]  # get nx from whereever

    states = jax.random.multivariate_normal(
            jax.random.PRNGKey(666),  # very pseudorandom
            mean=np.zeros(nx,),
            cov=algo_params['x_sample_cov'],
            shape=(algo_params['n_trajectories'],)
    ).reshape(algo_params['n_trajectories'], nx)

    tbounds = (np.min(all_ts), np.max(all_ts))
    tgrid = np.linspace(*tbounds, 101)

    # make a 3D array containing first the time index, then the states
    # we want shape (n_t, n_trajectories, 3), last dim row to store (t, x)

    reshaped_ts = np.repeat(tgrid[:, None, None], algo_params['n_trajectories'], axis=1)
    reshaped_xs = np.repeat(states[None, :, :], tgrid.shape[0], axis=0)

    # this will work if shapes are (n_t, n_traj, 1) and (n_t, n_traj, 2)
    all_state_time_combs = np.concatenate([reshaped_ts, reshaped_xs], axis=2)

    # evaluate the NN :)
    values = V_nn_wrapper(
            all_state_time_combs.reshape(-1, 3), nn_params
    ).reshape(tgrid.shape[0], algo_params['n_trajectories'])


    # for plotting we don't really care about the particular state. but we
    # would like to have a (n_time, n_states) array to plot directly. this
    # is already our values array?!?

    statenorms = np.linalg.norm(states, axis=1)
    statenorms_normal = (statenorms - np.min(statenorms)) / (np.max(statenorms) - np.min(statenorms))

    # somehow i was not able to find out how to do this :(
    # colors = matplotlib.colormaps.get_cmap('cool')(statenorms_normal)

    # and another plot with value fct over simulated trajectories
    pl.figure('value function plots')
    pl.subplot(211)
    pl.title('V over time, at a selection of states')
    pl.plot(tgrid, values, alpha = .5)

    all_values = all_sols[:, :, -1]

    try:
        all_values = all_values.at[where_resampled].set(np.nan)
    except:
        all_values[where_resampled] = np.nan


    pl.subplot(212)
    pl.title('value function along simulated trajectories')
    pl.plot(all_ts, all_values.squeeze().T, alpha=0.1)
    pl.plot(all_ts, 0 * all_ts, color='black', alpha=0.5, linestyle='--')


def plot_2d(all_sols, all_ts, where_resampled, problem_params, algo_params):

    fig = pl.figure(figsize=(8, 3))

    ax0 = fig.add_subplot(111, projection='3d')
    # ax0 = fig.add_subplot(121, projection='3d')
    # ax1 = fig.add_subplot(122, projection='3d')


    nx = problem_params['nx']
    T = problem_params['T']
    all_xs       = all_sols[:, :,  0:nx   ]
    all_costates = all_sols[:, :,  nx:2*nx]
    all_values   = all_sols[:, :, -1      ]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algo_params['n_trajectories'], -1, 1)
    # when norm <= max_norm, scalings = 1, otherwise, scales to max_norm
    scalings = np.minimum(1, max_norm/x_norms)

    all_xs = all_xs * scalings

    a = 0.1 * 256 / (algo_params['n_trajectories'])
    if a > 1: a = 1

    # somehow i was not able to get it to work with a single call and data in matrix form.
    # ts_expanded = np.tile(all_ts[:, None], algo_params['n_trajectories'])
    # ipdb.set_trace()
    # ax0.plot(all_ts, all_xs[:, :, 0].squeeze().T, all_xs[:, :, 1].squeeze().T, color='black', alpha=a)
    # ax1.plot(all_xs[:, :, 0], all_xs[:, :, 1], all_values, color='black', alpha=a)

    # neat hack - if we set the xs to nan where resamplings occurred, it breaks up the plot and does not
    # connect the trajectories before and after resampling
    all_xs = all_xs.at[where_resampled, :].set(np.nan)

    # so now we go back to the stone age
    for i in range(algo_params['n_trajectories']):
        ax0.plot(all_ts, all_xs[i, :, 0].squeeze(), all_xs[i, :, 1].squeeze(), color='black', alpha=a)
        # ax1.plot(all_xs[i, :, 0], all_xs[i, :, 1], all_values[i], color='blue', alpha=a)

    ax0.set_xlabel('t')
    ax0.set_ylabel('x_0')
    ax0.set_zlabel('x_1')

    ax0.set_ylim([-5, 5])
    ax0.set_zlim([-2, 2])

    # ax1.set_xlabel('x_0')
    # ax1.set_ylabel('x_1')
    # ax1.set_zlabel('value')


    fig, ax = pl.subplots()

    # make a figure with a slider to get a feel of the distribution of particles
    # over the state-space at some (adjustable) time.

    t_plot = T

    # make the figure
    idx_plot = np.argmin(np.abs(all_ts - t_plot))
    sc = pl.scatter(all_xs[:, idx_plot, 0].squeeze(), all_xs[:, idx_plot, 1].squeeze())

    fig.subplots_adjust(bottom=.25)
    ax_time = fig.add_axes([.25, .1, .65, .03])
    time_slider = matplotlib.widgets.Slider(
        ax=ax_time,
        label='time [s]',
        valmin=0,
        valmax=T,
        valinit=t_plot,
    )

    def update(val):
        t_plot = time_slider.val
        idx_plot = np.argmin(np.abs(all_ts - t_plot))

        x = all_xs[:, idx_plot, 0].squeeze()
        y = all_xs[:, idx_plot, 1].squeeze()

        sc.set_offsets(np.c_[x, y])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)


def plot_1d(all_sols, all_ts, where_resampled, problem_params, algo_params):

    # just one figure, with the state trajectories, maybe colored according
    # to value?

    fig = pl.figure(figsize=(8, 3))

    nx = problem_params['nx']
    T = problem_params['T']
    all_xs       = all_sols[:, :,  0:nx   ]
    all_costates = all_sols[:, :,  nx:2*nx]
    all_values   = all_sols[:, :, -1      ]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algo_params['n_trajectories'], -1, 1, 1)
    # when norm <= max_norm, scalings = 1, otherwise, scales to max_norm
    scalings = np.minimum(1, max_norm/x_norms)

    all_xs = all_xs * scalings

    a = 0.1 * 256 / (algo_params['n_trajectories'])
    if a > 1: a = 1

    # neat hack - if we set the xs to nan where resamplings occurred, it breaks up the plot and does not
    # connect the trajectories before and after resampling
    all_xs = all_xs.at[where_resampled, :, :].set(np.nan)

    xs_plot = all_xs.squeeze().T
    costates_plot = all_costates.squeeze().T
    pl.plot(all_ts, xs_plot, color='b', label='state')
    pl.plot(all_ts, costates_plot, color='g', label='costate')
    pl.legend()

    pl.xlabel('t')
    pl.ylabel('x')



def plot_nn_train_outputs(outputs):

    pl.figure('NN training visualisation', figsize=(15, 10))

    # outputs is a dict where keys are different outputs, and the value
    # is an array containing that output for ALL relevant training iterations,

    # training subplot
    ax1 = pl.subplot(211)

    make_nice = lambda s: s.replace('_', ' ')

    for k in outputs.keys():
        if 'train' in k:
            pl.semilogy(outputs[k], label=make_nice(k), alpha=0.8)
    pl.grid(axis='both')
    pl.ylim([1e-3, 1e3])
    pl.legend()

    # training subplot
    pl.subplot(212, sharex=ax1)
    pl.gca().set_prop_cycle(None)

    for k in outputs.keys():
        if 'test' in k:
            pl.semilogy(outputs[k], label=make_nice(k), alpha=1)
        if 'lr' in k:
            pl.semilogy(outputs[k], label='learning rate', linestyle='--', color='gray', alpha=.5)
    pl.grid(axis='both')
    pl.ylim([1e-3, 1e3])
    pl.legend()


