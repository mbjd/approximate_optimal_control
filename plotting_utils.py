# jax
import jax
import jax.numpy as np

# other, trivial stuff
import numpy as onp

import tk as tkinter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

def plot_2d_V(V_nn_wrapper, nn_params, tbounds, xbounds):

    tmin, tmax = tbounds
    xmin, xmax = xbounds  # same bounds for both state dims for now.


    # make a figure with a slider to get a feel of the distribution of particles
    # over the state-space at some (adjustable) time.


    # @partial(jax.jit, static_argnames=['return_grid'])
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


def plot_2d(all_sols, all_ts, where_resampled, problem_params, algo_params):

    fig = pl.figure(figsize=(8, 3))

    ax0 = fig.add_subplot(111, projection='3d')
    # ax0 = fig.add_subplot(121, projection='3d')
    # ax1 = fig.add_subplot(122, projection='3d')


    nx = problem_params['nx']
    T = problem_params['T']
    all_xs       = all_sols[:, :,  0:nx,    :]
    all_costates = all_sols[:, :,  nx:2*nx, :]
    all_values   = all_sols[:, :, -1,       :]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algo_params['n_trajectories'], -1, 1, 1)
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
    all_xs = all_xs.at[where_resampled, :, :].set(np.nan)

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

    # pl.show()

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

    pl.show()


def plot_1d(all_sols, all_ts, where_resampled, problem_params, algo_params):

    # just one figure, with the state trajectories, maybe colored according
    # to value?

    fig = pl.figure(figsize=(8, 3))

    nx = problem_params['nx']
    T = problem_params['T']
    all_xs       = all_sols[:, :,  0:nx,    :]
    all_costates = all_sols[:, :,  nx:2*nx, :]
    all_values   = all_sols[:, :, -1,       :]

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

    pl.show()

