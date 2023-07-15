#!/usr/bin/env python

import jax
import jax.numpy as np

import diffrax

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl
import matplotlib.ticker as mticker

import ipdb
import os

import numpy as onp
import scipy

# global config
dpi = 400
halfwidth = 5  # in
subsample=True
save=True

def save_fig_wrapper(figname):
    if save:
        figpath = os.path.join('./figs/', figname)
        pl.savefig(figpath, dpi=dpi)
        print(f'saved fig: {figpath}')
        pl.close()
    else:
        pl.show()

def fig_train_data(sysname):

    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

    k = jax.random.PRNGKey(0)
    print(f'total pts: {y0s.shape[0]}')
    if subsample:
        Nmax = 2048
        rand_idx = jax.random.choice(k, np.arange(y0s.shape[0]), shape=(Nmax,))

        y0s = y0s[rand_idx, :]
        lamTs = lamTs[rand_idx, :]

    print(f'total pts to plot: {y0s.shape[0]}')
    x0s, lam0s, v0s = np.split(y0s, [2, 4], axis=1)

    cmap = 'viridis'

    fig, ax = pl.subplots(ncols=2, layout='compressed', figsize=(2*halfwidth, 2*halfwidth*.4))
    pl.subplot(121)
    pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, c=v0s)
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$')

    pl.subplot(122)
    sc = pl.scatter(*np.split(lamTs, [1], axis=1), cmap=cmap, c=v0s)
    pl.xlabel(r'$λ_T^{(0)}$')
    pl.ylabel(r'$λ_T^{(1)}$')
    pl.gca().set_title(r'Sampled $\lambda_T$')

    pl.colorbar(sc, label='Value $V(x_0)$')

    save_fig_wrapper(f'fig_train_data_{sysname}.png')

def fig_train_data_big(sysname):

    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

    full = False
    if full:
        all_y0s = np.load(f'datasets/mcmc_complete/last_y0s_{sysname}.npy')
        all_lamTs = np.load(f'datasets/mcmc_complete/last_lamTs_{sysname}.npy')
        y0s = all_y0s.reshape(-1, 5)
        lamTs = all_lamTs.reshape(-1, 2)

    k = jax.random.PRNGKey(1)
    print(f'total pts: {y0s.shape[0]}')
    if subsample:
        Nmax = 2048
        rand_idx = jax.random.choice(k, np.arange(y0s.shape[0]), shape=(Nmax,))
        # rand_idx = lamTs[:, 0] < -0.041  # override to plot just part of data

        y0s = y0s[rand_idx, :]
        lamTs = lamTs[rand_idx, :]

    print(f'total pts to plot: {y0s.shape[0]}')
    x0s, lam0s, v0s = np.split(y0s, [2, 4], axis=1)




    cmap = 'viridis'
    a = 1

    fig, ax = pl.subplots(ncols=2, layout='compressed', figsize=(2*halfwidth, 2*halfwidth*.8))
    pl.subplot(221)
    pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, alpha=a, c=v0s)
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $V(x_0)$')

    pl.subplot(222)
    sc = pl.scatter(*np.split(lamTs, [1], axis=1), cmap=cmap, alpha=a, c=v0s)
    pl.xlabel(r'$λ_T^{(0)}$')
    pl.ylabel(r'$λ_T^{(1)}$')
    pl.gca().set_title(r'Sampled $\lambda_T$, coloured by $V(PMP(λ_T))$')

    pl.colorbar(sc, label='Value $V(x_0)$')

    pl.subplot(223)
    sc = pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, alpha=a, c=y0s[:, 2])
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $λ^{(0)}(x_0)$')

    pl.subplot(224)
    sc = pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, alpha=a, c=y0s[:, 3])
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $λ^{(1)}(x_0)$')

    pl.colorbar(sc, label='Costate $λ(x_0)$')


    save_fig_wrapper(f'fig_train_data_big_{sysname}.png')

# def fig_controlcost_newformat(sysname):
#
#     # control cost/test loss/N training pts figure.
#
#     # we swap here for easier plotting
#     data = np.load(f'datasets/trainpts_controlcost_data_{sysname}.npy').swapaxes(0, 1)
#
#     # todo unpack more values here
#     N_trainpts, costate_testloss, cost_mean, cost_std = np.split(data, np.arange(1, data.shape[2]), axis=2)
#
#     # all the same
#     N_trainpts = N_trainpts[:, 0]
#     N_seeds = data.shape[1]
#
#     labels = ['' for _ in range(N_seeds)]
#     labels[0] = 'costate test loss'
#     pl.figure(figsize=(halfwidth, 1.*halfwidth))
#
#
#     pl.subplot(211)
#     a = .3
#     # pl.gca().set_yticks([.5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8])
#     pl.loglog(N_trainpts, costate_testloss.squeeze(), c='tab:blue', marker='.', alpha=a, label=labels)
#     # so it looks less weird
#     # pl.gca().set_ylim([0.4, 12])
#     # pl.gca().axes.xaxis.set_ticklabels([])
#     pl.legend()
#     pl.grid()
#
#
#     pl.subplot(212)
#
#     # plot lqr baseline:
#     if sysname == 'double_integrator_linear':
#         mean, std = np.load(f'datasets/controlcost_lqr_meanstd_{sysname}.npy')
#
#         labels[0] = 'control cost / LQR cost'
#         pl.loglog(N_trainpts, cost_mean.squeeze()/mean, c='tab:green', marker='.', alpha=a, label=labels)
#         pl.xlabel('Training set size')
#         # pl.loglog(N_trainpts, mean * np.ones_like(N_trainpts), c='black', alpha=2*a, linestyle='--', label='LQR cost')
#
#
#     else:
#         labels[0] = 'control cost'
#         pl.loglog(N_trainpts, cost_mean.squeeze(), c='tab:green', marker='.', alpha=a, label=labels)
#         pl.xlabel('Training set size')
#
#
#
#     pl.legend()
#     pl.grid()
#
#     pl.tight_layout()
#     pl.subplots_adjust(hspace=0)
#
#
#     save_fig_wrapper(f'fig_controlcost_{sysname}.png')

def fig_controlcost(sysname):

    # control cost/test loss/N training pts figure.

    base2 = True
    shaded_percentile = True

    def plot_data(xs, data, c, label_arr):
        # data.shape[0] = number of points per line
        # data.shape[1] = number of lines
        assert len(data.shape) == 2

        if not shaded_percentile:
            # previous plot
            pl.loglog(xs, data, color=c, marker='.', alpha=.3, label=label_arr)
            return

        # otherwise: plot shaded areas for percentiles.

        lower = np.percentile(data, 0, axis=1)
        upper = np.percentile(data, 100, axis=1)
        median = np.percentile(data, 50, axis=1)

        a = 0.3 # onp.clip(p/40 + 0.2, 0, 1)
        pl.fill_between(xs, lower, upper, color=c, alpha=a)
        pl.loglog(xs, median, color=c, marker='.', alpha=1, label=label_arr[0])


    # we swap here for easier plotting
    data = np.load(f'datasets/trainpts_controlcost_data_{sysname}.npy').swapaxes(0, 1)

    if data.shape[2] == 6:
        # throw it away :)
        data = data[:, :, np.array([0, 1, 4, 5])]
        # fig_controlcost_newformat(sysname)
        # return

    N_trainpts, costate_testloss, cost_mean, cost_std = np.split(data, np.arange(1, data.shape[2]), axis=2)

    # see if the data is complete...
    n_complete = (cost_mean > 0).sum()
    n_total = cost_mean.size
    if n_complete < n_total:
        print(f'warning: making fig_controlcost with incomplete data ({n_complete}/{n_total})')

    # all the same
    N_trainpts = N_trainpts[:, 0].squeeze()
    N_seeds = data.shape[1]

    labels = ['' for _ in range(N_seeds)]
    labels[0] = 'costate test loss'
    pl.figure(figsize=(halfwidth, 1.*halfwidth))


    pl.subplot(211)
    a = .3
    plot_data(N_trainpts, costate_testloss.squeeze(), 'tab:blue', labels)
    if base2:
        pl.gca().set_xscale('log', base=2)  # base 2 is love, base 2 is life
        pl.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
    # so it looks less weird
    # pl.gca().set_ylim([0.4, 12])
    # pl.gca().axes.xaxis.set_ticklabels([])
    pl.legend()
    pl.grid()


    pl.subplot(212)
    # plot lqr baseline:
    if sysname in ('double_integrator_linear', 'double_integrator_linear_corrected'):
        mean, std = np.load(f'datasets/controlcost_lqr_meanstd_{sysname}.npy')

        labels[0] = '(control cost - LQR cost) / LQR cost'
        plot_data(N_trainpts, (cost_mean.squeeze()-mean)/mean, 'tab:green', labels)

        # pl.loglog(N_trainpts, (cost_mean.squeeze()-mean)/mean, c='tab:green', marker='.', alpha=a, label=labels)
        if base2:
            pl.gca().set_xscale('log', base=2)
            pl.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())

        # this is clearly the worse visualisation, although the one above is maybe harder
        # to grasp intuitively, it shows a lot more useful info
        # labels[0] = 'control cost / LQR cost'
        # pl.loglog(N_trainpts, (cost_mean.squeeze())/mean, c='tab:green', marker='.', alpha=a, label=labels)
        pl.gca().set_ylim([1e-6, 3e-1])
        # pl.loglog(N_trainpts, mean * np.ones_like(N_trainpts), c='black', alpha=2*a, linestyle='--', label='LQR cost')


    else:

        labels[0] = 'control cost'
        plot_data(N_trainpts, cost_mean.squeeze(), 'tab:green', labels)
        if base2:
            pl.gca().set_xscale('log', base=2)
            pl.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())

    pl.xlabel('Training set size')


    pl.legend()
    pl.grid()

    pl.tight_layout()
    pl.subplots_adjust(hspace=0)


    save_fig_wrapper(f'fig_controlcost_{sysname}.png')

def fig_mcmc(sysname):

    # shape = (N_chains, N_iters, 2*nx+1)
    y0s = np.load(f'datasets/mcmc_complete/last_y0s_{sysname}.npy')

    # shape = (N_chains, N_iters, nx)
    lamTs = np.load(f'datasets/mcmc_complete/last_lamTs_{sysname}.npy')

    x0s = y0s[:, :, 0:2]

    # in theory: autocorrelation η(n) = E_k[ x_k^T x_{k+n} ]

    # n goes from 0 to this
    max_dist = 64

    # this does not work yet...
    # approx. expectation with empirical mean
    # this takes two sequences of shape (seq_len, nx)
    def correlate(seq_a, seq_b):
        all_corrs = jax.vmap(np.dot)(seq_a, seq_b)
        mean_corr = all_corrs.mean()
        return mean_corr

    def calc_autocorrs(offsets, subseq_len, data):
        # offsets: ns to use for evaluating η(n)
        # subseq_len: length of sequences passed to correlate(., .)
        # data: time series of shape (N_iters, nx)

        @jax.jit
        def calc_autocorr_n(n):
            seq_a = data[0:subseq_len]
            seq_b = data[n:subseq_len+n]
            return correlate(seq_a, seq_b)

        ipdb.set_trace()
        # return all_autocorrs

    autocorrs_0 = calc_autocorrs(np.arange(max_dist), x0s.shape[1]-max_dist-1, x0s[0])
    ipdb.set_trace()


if __name__ == '__main__':

    # new example with more correct prng key handling
    # fig_controlcost('double_integrator_unlimited')
    # fig_train_data_big('double_integrator_lofi')

    #fig_train_data_big('double_integrator_linear')    # 65536  pts
    #fig_train_data_big('double_integrator_linear_pontryagintest')    # 65536  pts
    # fig_train_data_big('double_integrator')           # 16384  pts
    # fig_train_data_big('double_integrator_bigsample') # 262144 pts
    # fig_controlcost('double_integrator_linear')
    # fig_controlcost('double_integrator')

    # fig_mcmc('double_integrator_linear_corrected')

    # fig_train_data_big('double_integrator_linear_corrected')
    # fig_controlcost('double_integrator_linear_corrected')
    fig_train_data_big('double_integrator_corrected')
    fig_controlcost('double_integrator_corrected')


    # the two are literally the exact same
    # fig_controlcost('double_integrator')
    # fig_train_data('double_integrator_unlimited')
    # fig_train_data('double_integrator')

