#!/usr/bin/env python

import jax
import jax.numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import os

# global config
dpi = 400
halfwidth = 5  # in
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
    Nmax = 1024
    print(f'total pts: {y0s.shape[0]}')
    rand_idx = jax.random.choice(k, np.arange(y0s.shape[0]), shape=(Nmax,))

    y0s = y0s[rand_idx, :]
    lamTs = lamTs[rand_idx, :]

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

def fig_controlcost(sysname):

    # control cost/test loss/N training pts figure.

    # we swap here for easier plotting
    data = np.load(f'datasets/trainpts_controlcost_data_{sysname}.npy').swapaxes(0, 1)

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
    pl.gca().set_ylim([0.4, 12])
    pl.gca().axes.xaxis.set_ticklabels([])
    pl.legend()
    pl.grid()


    pl.subplot(212)
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
    # fig_controlcost('double_integrator')

    # fig_controlcost('double_integrator')

    # the two are literally the exact same
    fig_train_data('double_integrator_test')
    fig_train_data('double_integrator_unlimited')
    fig_train_data('double_integrator')

