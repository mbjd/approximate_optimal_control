# jax
import jax
import jax.numpy as np

# other, trivial stuff
import numpy as onp

# import tk as tkinter
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import time
import numpy as onp

from functools import partial
cmap = 'viridis'

def plot_controlcost_vs_traindata():

    # axis 0: which PRNG key was used?
    # axis 1: which point number?
    # axis 2: which type of data?
    #  index 0: number of training points used (-> x axis for plot)
    #  index 1: costate test loss
    #  index 2: mean control cost
    #  index 3: std. dev. control cost.

    # we swap here for easier plotting
    data = np.load('datasets/trainpts_controlcost_data.npy').swapaxes(0, 1)

    N_trainpts, costate_testloss, cost_mean, cost_std = np.split(data, np.arange(1, data.shape[2]), axis=2)

    # all the same
    N_trainpts = N_trainpts[:, 0]
    N_seeds = data.shape[1]

    labels = ['' for _ in range(N_seeds)]
    labels[0] = 'costate test loss'
    # pl.subplot(121)
    a = .3
    pl.loglog(N_trainpts, costate_testloss.squeeze(), c='tab:blue', marker='.', alpha=a, label=labels)


    labels[0] = 'mean control cost'
    pl.loglog(N_trainpts, cost_mean.squeeze(), c='tab:green', marker='.', alpha=a, label=labels)
    pl.xlabel('Training set size')


    pl.legend()

    # pl.subplot(122)

    # labels[0] = 'control cost vs. λ test loss'
    # pl.loglog(costate_testloss.squeeze(), cost_mean.squeeze(), c='tab:orange', marker='.', alpha=a, label=labels)
    # pl.xlabel('λ test loss')
    # pl.ylabel('control cost')

    pl.legend()
    pl.show()



    ipdb.set_trace()



def plot_2d_gp(gp, gp_ys, xbounds, ybounds, N_disc=101, save=False,
        savename=None):

    '''
    plot the 2d gp. makes three plots:
    - color plot of the function itself
    - color plot of the function variance
    - color plot of the gradient variance
      (how? largest? det cov?)
    '''

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    xgrid = np.linspace(xmin, xmax, N_disc)
    ygrid = np.linspace(ymin, ymax, N_disc)

    xx, yy = np.meshgrid(xgrid, ygrid)

    # TODO maybe package this type of batched, jitted eval functin
    # with each GP?
    def eval_gp_single(gp, gp_ys, input_x, input_gradflag):
        # evaluate gp conditioned on gp_ys at (input_x, input_gradflag).
        pred_gp = gp.condition(gp_ys, (input_x, input_gradflag)).gp
        return (pred_gp.loc, np.sqrt(pred_gp.variance))

    eval_gp_multiple = jax.vmap(eval_gp_single, in_axes=(None, None, 0, 0))
    eval_gp_multiple = jax.jit(eval_gp_multiple, static_argnums=[0])

    fig, ax = pl.subplots(ncols=3, layout='compressed', figsize=(9, 4))

    eval_xs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    eval_gradflags = np.zeros(eval_xs.shape[0], dtype=np.int8)

    # new try: keep (N_disc, N_disc, 2) shape, pass axes 1,2 into
    # eval_gp_single, vmap over axis 0
    eval_xs = np.stack([xx, yy], axis=2)
    eval_gradflags = np.zeros(xx.shape, dtype=np.int8)

    y_pred, y_std = eval_gp_multiple(gp, gp_ys, eval_xs, eval_gradflags)

    threeD = True
    if threeD:
        # ugly subplot over subplot...
        ax3d = fig.add_subplot(131, projection='3d')

        # just mean
        # ax3d.plot_surface(xx, yy, y_pred, alpha=.8, cmap='viridis')

        # confidence interval
        ax3d.plot_surface(xx, yy, np.clip(y_pred + 3*y_std, 0, 12), alpha=.6, color='red')
        ax3d.plot_surface(xx, yy, np.clip(y_pred - 3*y_std, 0, 12), alpha=.6, color='green')

    else:
        pl.subplot(131)
        pl.pcolor(xx, yy, y_pred)

    # plot training data
    # only get data where gradflag is 0, difficult to scatterplot the gradient
    train_x = gp.X[0][gp.X[1] == 0]
    train_V = gp_ys[gp.X[1] == 0]
    pl.gca().scatter(train_x[:, 0], train_x[:, 1], train_V)

    pl.gca().set_title('Value function V(x)')

    pl.subplot(132)

    outputs = y_std.reshape(xx.shape)
    pl.pcolor(xx, yy, y_std)
    pl.gca().set_title('std. dev. sqrt(Var(V(x)))')

    pl.subplot(133)

    nx = 2  # otherwise this function needs a complete rewrite anyway

    grad_preds = onp.zeros(eval_xs.shape)
    grad_stds =  onp.zeros(eval_xs.shape)

    # loop so we can reuse the jitted function
    for i in range(nx):
        grad_preds[:, :, i], grad_stds[:, :, i] = eval_gp_multiple(
                gp, gp_ys, eval_xs, eval_gradflags + i + 1
        )

    pl.pcolor(xx, yy, np.max(grad_stds, axis=2))
    pl.gca().set_title('std. dev max_i sqrt(Var(grad_{x_i} V(x)))')

    if save:
        assert type(savename) == str
        pl.savefig(savename, dpi=400)
    else:
        pl.show()




def value_lambda_scatterplot(x0s, v0s, lamTs, save=True):

    fig, ax = pl.subplots(ncols=2, layout='compressed', figsize=(6, 3))
    pl.subplot(121)
    pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, c=v0s)
    pl.gca().set_title(r'Sampled $x_0$')

    pl.subplot(122)
    sc = pl.scatter(*np.split(lamTs, [1], axis=1), cmap=cmap, c=v0s)
    pl.gca().set_title(r'Sampled $\lambda_T$')

    pl.colorbar(sc, label='Value $V(x_0)$')

    if save:
        t = int(time.time())
        pl.savefig(f'figs/value_lambda_scatter_{t}.png', dpi=400)
    else:
        pl.show()






def plot_ellipse(Q, N_pts=101):

    # plot ellipse S = {x | x.T Q x = 1}
    # x.T Q x = x.T Q^.5.T Q^.5 x = || Q^.5 x || == 1
    # so basically, Q^.5 x is the unit circle, for x in S.
    # Therefore, Q^(-1/2) (unit circle) = S

    thetas = np.linspace(0, 2*np.pi, N_pts)

    circle = jax.vmap(lambda t: np.array([np.cos(t), np.sin(t)]))(thetas).T

    # it should be positive definite for a unique solution
    # hopefully the user is smart enough
    Q_half_inv = jax.scipy.linalg.sqrtm(np.linalg.inv(Q)).real

    ellipse = Q_half_inv @ circle
    pl.plot(ellipse[0, :], ellipse[1, :], color='red', alpha=.5)


def plot_fct(f, xbounds, ybounds, N_disc = 401):

    # just the business logic: construct grid, reshape, evaluate, plot
    # create the plot before and show it after this function

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    xgrid = np.linspace(xmin, xmax, N_disc)
    ygrid = np.linspace(ymin, ymax, N_disc)

    xx, yy = np.meshgrid(xgrid, ygrid)

    all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    all_outputs = jax.vmap(f)(all_inputs).reshape(xx.shape)  # need a lot of memory!
    # clip at V_max used for data generation.
    all_outputs = np.clip(all_outputs, 0, 12)
    pl.pcolor(xx, yy, all_outputs, cmap='viridis')
    pl.contour(xx, yy, all_outputs, c='black')
    # pl.contourf(xx, yy, all_outputs, cmap='viridis')
    # pl.pcolor(xx, yy, all_outputs, cmap='jet')

def plot_fct_3d(f, xbounds, ybounds, N_disc = 401):

    fig = pl.figure('3d plot ooh')
    ax = fig.add_subplot(111, projection='3d')

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    xgrid = np.linspace(xmin, xmax, N_disc)
    ygrid = np.linspace(ymin, ymax, N_disc)

    xx, yy = np.meshgrid(xgrid, ygrid)

    all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    all_outputs = jax.vmap(f)(all_inputs).reshape(xx.shape)  # need a lot of memory!

    # clip at V_max used for data generation.
    all_outputs = np.clip(all_outputs, 0, 12)

    ax.plot_surface(xx, yy, all_outputs, cmap='viridis', alpha=.75)

def plot_nn_gradient_eval(V_nn, nn_params, xs, xs_test, ys, ys_test):

    # xs.shape == (N, nx)
    # ys.shape == (N, nx+1), first nx costates, then 1 value.
    # everything should be unnormalised here.

    # the idea is simple: plot scatterplot of actual values vs nn values.
    # three plots: value vs predicted value, λ0 vs pred λ0, λ1 vs pred λ1.
    # but two times, once (top row) with training data, once (bottom) test.

    # to keep it somewhat manageable.
    N_scatter = 512

    pl.figure(figsize=(12, 8))

    pl.subplot(231)
    pl.scatter(ys[0:N_scatter, -1], V_nn(nn_params, xs[0:N_scatter, :]))
    pl.gca().set_title('train value vs predicted value')
    pl.gca().set_aspect('equal', 'box')

    grad_pred_train = V_nn.apply_grad(nn_params, xs[0:N_scatter, :])

    pl.subplot(232)
    pl.scatter(ys[0:N_scatter, 0], grad_pred_train[:, 0])
    pl.gca().set_title('train λ0 vs predicted λ0')
    pl.gca().set_aspect('equal', 'box')

    pl.subplot(233)
    pl.scatter(ys[0:N_scatter, 1], grad_pred_train[:, 1])
    pl.gca().set_title('train λ1 vs predicted λ1')
    pl.gca().set_aspect('equal', 'box')

    pl.subplot(234)
    pl.scatter(ys_test[0:N_scatter, -1], V_nn(nn_params, xs_test[0:N_scatter, :]))
    pl.gca().set_title('test value vs predicted value')
    pl.gca().set_aspect('equal', 'box')

    grad_pred_test = V_nn.apply_grad(nn_params, xs_test[0:N_scatter, :])

    pl.subplot(235)
    pl.scatter(ys_test[0:N_scatter, 0], grad_pred_test[:, 0])
    pl.gca().set_title('test λ0 vs predicted λ0')
    pl.gca().set_aspect('equal', 'box')

    pl.subplot(236)
    pl.scatter(ys_test[0:N_scatter, 1], grad_pred_test[:, 1])
    pl.gca().set_title('test λ1 vs predicted λ1')
    pl.gca().set_aspect('equal', 'box')


    fig = pl.figure()
    # next figure: scatterplot of test data x0, x1, coloured by error.

    pl.subplot(131)
    V_pred = V_nn(nn_params, xs_test).squeeze()
    V_err = V_pred - ys_test[:, -1]
    m = np.abs(V_err).max()
    im = pl.gca().scatter(xs_test[:, 0], xs_test[:, 1], vmin=-m, vmax=m, c=V_err, cmap='RdYlGn')
    pl.colorbar(im)
    pl.gca().set_title('V error')

    grad_pred = V_nn.apply_grad(nn_params, xs_test)
    grad_err = grad_pred - ys_test[:, 0:2]

    pl.subplot(132)
    vm = np.abs(grad_err[:, 0]).max()
    im = pl.gca().scatter(xs_test[:, 0], xs_test[:, 1], vmin=-vm, vmax=vm, c=grad_err[:, 0], cmap='RdYlGn')
    pl.colorbar(im)
    pl.gca().set_title('λ0 = V_x0 error')

    pl.subplot(133)
    vm = np.abs(grad_err[:, 0]).max()
    im = pl.gca().scatter(xs_test[:, 0], xs_test[:, 1], vmin=-vm, vmax=vm, c=grad_err[:, 1], cmap='RdYlGn')
    pl.colorbar(im)
    pl.gca().set_title('λ1 = V_x1 error')


def plot_nn_ensemble(V_nn, nn_params, xbounds, ybounds, Nd = 101, save=None):

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    xgrid = np.linspace(xmin, xmax, Nd)
    ygrid = np.linspace(ymin, ymax, Nd)

    xx, yy = np.meshgrid(xgrid, ygrid)
    all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    means, stds = V_nn.ensemble_mean_std(nn_params, all_inputs)

    fig, axs = pl.subplots(2, 3, figsize=(6, 10))

    # pcolor likes everything to be (nx, ny) shaped
    names = ['Value std.', 'λ0 std.', 'λ1 std.']
    for i in range(3):
        im = axs[0, i].pcolormesh(xx, yy, means[:, i].reshape(Nd, Nd))
        axs[1, i].set_xlabel(names[i].replace('std', 'mean'))
        pl.colorbar(im)

        im = axs[1, i].pcolormesh(xx, yy, stds[:, i].reshape(Nd, Nd))
        axs[1, i].set_xlabel(names[i])
        pl.colorbar(im)
    pl.tight_layout()

    if save is not None:
        pl.savefig(save, dpi=400)
    else:
        pl.show()





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

    cs = ['tab:blue', 'tab:green', 'tab:red']
    c=0
    a = .2

    # training subplot
    ax1 = pl.subplot(211)

    make_nice = lambda s: s.replace('_', ' ')

    for k in outputs.keys():
        if 'train' in k:
            if len(outputs[k].shape) == 2:
                # NN ensemble
                N_ens, Nt = outputs[k].shape
                labels = ['' for i in range(N_ens)]
                labels[0] = make_nice(k)
                pl.semilogy(outputs[k].T, label=labels, alpha=a, c=cs[c])
                c = c+1
            else:
                pl.semilogy(outputs[k], label=make_nice(k), alpha=0.8)
    pl.grid(axis='both')
    pl.ylim([1e-3, 1e3])
    pl.legend()

    # training subplot
    pl.subplot(212, sharex=ax1)
    pl.gca().set_prop_cycle(None)
    c = 0

    for k in outputs.keys():
        if 'test' in k:
            if len(outputs[k].shape) == 2:
                # NN ensemble
                N_ens, Nt = outputs[k].shape
                labels = ['' for i in range(N_ens)]
                labels[0] = make_nice(k)
                pl.semilogy(outputs[k].T, label=labels, alpha=a, c=cs[c])
                c = c+1
            else:
                pl.semilogy(outputs[k], label=make_nice(k), alpha=0.8)
        if 'lr' in k:
            pl.semilogy(outputs[k], label='learning rate', linestyle='--', color='gray', alpha=.5)
    pl.grid(axis='both')
    pl.ylim([1e-3, 1e3])
    pl.legend()


