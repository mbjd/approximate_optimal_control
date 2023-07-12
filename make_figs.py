#!/usr/bin/env python

import jax
import jax.numpy as np

import diffrax

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import os

import numpy as onp
import scipy

# global config
dpi = 400
halfwidth = 5  # in
subsample=True
save=False

def save_fig_wrapper(figname):
    if save:
        pl.savefig(os.path.join('./figs/', figname), dpi=dpi)
        print(f'saved fig: {figname}')
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

'''
def fig_train_data_big(sysname):

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

    fig, ax = pl.subplots(ncols=2, layout='compressed', figsize=(2*halfwidth, 2*halfwidth*.8))
    pl.subplot(221)
    pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, c=v0s)
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $V(x_0)$')

    pl.subplot(222)
    sc = pl.scatter(*np.split(lamTs, [1], axis=1), cmap=cmap, c=v0s)
    pl.xlabel(r'$λ_T^{(0)}$')
    pl.ylabel(r'$λ_T^{(1)}$')
    pl.gca().set_title(r'Sampled $\lambda_T$, coloured by $V(PMP(λ_T))$')

    pl.colorbar(sc, label='Value $V(x_0)$')

    pl.subplot(223)
    sc = pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, c=y0s[:, 2])
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $λ^{(0)}(x_0)$')

    pl.subplot(224)
    sc = pl.scatter(*np.split(x0s, [1], axis=1), cmap=cmap, c=y0s[:, 3])
    pl.xlabel(r'$x_0^{(0)}$')
    pl.ylabel(r'$x_0^{(1)}$')
    pl.gca().set_title(r'Sampled $x_0$, coloured by $λ^{(1)}(x_0)$')

    pl.colorbar(sc, label='Costate $λ(x_0)$')


    save_fig_wrapper(f'fig_train_data_big_{sysname}.png')

def fig_controlcost_newformat(sysname):

    # control cost/test loss/N training pts figure.

    # we swap here for easier plotting
    data = np.load(f'datasets/trainpts_controlcost_data_{sysname}.npy').swapaxes(0, 1)

    # todo unpack more values here
    N_trainpts, costate_testloss, cost_mean, cost_std = np.split(data, np.arange(1, data.shape[2]), axis=2)

    # all the same
    N_trainpts = N_trainpts[:, 0]
    N_seeds = data.shape[1]

    labels = ['' for _ in range(N_seeds)]
    labels[0] = 'costate test loss'
    pl.figure(figsize=(halfwidth, 1.*halfwidth))


    pl.subplot(211)
    a = .3
    # pl.gca().set_yticks([.5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8])
    pl.loglog(N_trainpts, costate_testloss.squeeze(), c='tab:blue', marker='.', alpha=a, label=labels)
    # so it looks less weird
    # pl.gca().set_ylim([0.4, 12])
    # pl.gca().axes.xaxis.set_ticklabels([])
    pl.legend()
    pl.grid()


    pl.subplot(212)

    # plot lqr baseline:
    if sysname == 'double_integrator_linear':
        mean, std = np.load(f'datasets/controlcost_lqr_meanstd_{sysname}.npy')

        labels[0] = 'mean control cost / LQR cost'
        pl.loglog(N_trainpts, cost_mean.squeeze()/mean, c='tab:green', marker='.', alpha=a, label=labels)
        pl.xlabel('Training set size')
        # pl.loglog(N_trainpts, mean * np.ones_like(N_trainpts), c='black', alpha=2*a, linestyle='--', label='LQR cost')


    else:
        labels[0] = 'mean control cost'
        pl.loglog(N_trainpts, cost_mean.squeeze(), c='tab:green', marker='.', alpha=a, label=labels)
        pl.xlabel('Training set size')



    pl.legend()
    pl.grid()

    pl.tight_layout()
    pl.subplots_adjust(hspace=0)


    save_fig_wrapper(f'fig_controlcost_{sysname}.png')
'''

def fig_controlcost(sysname):

    # control cost/test loss/N training pts figure.

    # we swap here for easier plotting
    data = np.load(f'datasets/trainpts_controlcost_data_{sysname}.npy').swapaxes(0, 1)

    if data.shape[2] == 6:
        # throw it away :)
        data = data[:, :, np.array([0, 1, 4, 5])]
        # fig_controlcost_newformat(sysname)
        # return

    N_trainpts, costate_testloss, cost_mean, cost_std = np.split(data, np.arange(1, data.shape[2]), axis=2)

    # all the same
    N_trainpts = N_trainpts[:, 0]
    N_seeds = data.shape[1]

    labels = ['' for _ in range(N_seeds)]
    labels[0] = 'costate test loss'
    pl.figure(figsize=(halfwidth, 1.*halfwidth))


    pl.subplot(211)
    a = .3
    # pl.gca().set_yticks([.5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8])
    pl.loglog(N_trainpts, costate_testloss.squeeze(), c='tab:blue', marker='.', alpha=a, label=labels)
    # so it looks less weird
    # pl.gca().set_ylim([0.4, 12])
    # pl.gca().axes.xaxis.set_ticklabels([])
    pl.legend()
    pl.grid()


    pl.subplot(212)
    # plot lqr baseline:
    if sysname == 'double_integrator_linear':
        relative_cost = True
        if relative_cost:
            mean, std = np.load(f'datasets/controlcost_lqr_meanstd_{sysname}.npy')

            labels[0] = '(mean control cost - LQR cost) / LQR cost'
            pl.loglog(N_trainpts, (cost_mean.squeeze()-mean)/mean, c='tab:green', marker='.', alpha=a, label=labels)

            # this is clearly the worse visualisation, although the one above is maybe harder
            # to grasp intuitively, it shows a lot more useful info
            # labels[0] = 'mean control cost / LQR cost'
            # pl.loglog(N_trainpts, (cost_mean.squeeze())/mean, c='tab:green', marker='.', alpha=a, label=labels)
            pl.gca().set_ylim([1e-2, 1e3])
            pl.xlabel('Training set size')
            # pl.loglog(N_trainpts, mean * np.ones_like(N_trainpts), c='black', alpha=2*a, linestyle='--', label='LQR cost')
        else:
            # do the same as otherwise + lqr baseline  overlaid
            pl.subplot(212)
            labels[0] = 'mean control cost'
            pl.loglog(N_trainpts, cost_mean.squeeze(), c='tab:green', marker='.', alpha=a, label=labels)
            pl.xlabel('Training set size')

            # plot lqr baseline:
            if sysname == 'double_integrator_linear':
                mean, std = np.load(f'datasets/controlcost_lqr_meanstd_{sysname}.npy')
                pl.loglog(N_trainpts, mean * np.ones_like(N_trainpts), c='black', alpha=2*a, linestyle='--', label='LQR cost')


    else:
        labels[0] = 'mean control cost'
        pl.loglog(N_trainpts, cost_mean.squeeze(), c='tab:green', marker='.', alpha=a, label=labels)
        pl.xlabel('Training set size')


    pl.legend()
    pl.grid()

    pl.tight_layout()
    pl.subplots_adjust(hspace=0)


    save_fig_wrapper(f'fig_controlcost_{sysname}.png')


if __name__ == '__main__':

    # new example with more correct prng key handling
    # fig_controlcost('double_integrator_unlimited')
    # fig_train_data_big('double_integrator_lofi')

    # for paper:
    #fig_train_data_big('double_integrator_linear')    # 65536  pts
    #fig_train_data_big('double_integrator_linear_pontryagintest')    # 65536  pts
    # fig_train_data_big('double_integrator')           # 16384  pts
    # fig_train_data_big('double_integrator_bigsample') # 262144 pts
    fig_controlcost('double_integrator_linear')
    fig_controlcost('double_integrator_tuning')

    # the two are literally the exact same
    # fig_controlcost('double_integrator')
    # fig_train_data('double_integrator_unlimited')
    # fig_train_data('double_integrator')

