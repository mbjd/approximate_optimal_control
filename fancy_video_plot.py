#!/usr/bin/env python

import jax
import jax.numpy as np

import sampling
import pontryagin_utils
import plotting_utils
import nn_utils
import eval_utils

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

import numpy as onp

from jax.config import config
jax.config.update("jax_enable_x64", True)
# make a video visualising how data points transition from λT to x0.

# simple control system. double integrator with friction term.
def f(t, x, u):
    # p' = v
    # v' = f
    # f = -v**3 + u
    # clip so we don't have finite escape time when integrating backwards
    v_cubed = (np.array([[0, 1]]) @ x)**3
    v_cubed = np.clip(v_cubed, -10, 10)
    return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u - v_cubed)

def l(t, x, u):
    Q = np.eye(2)
    R = np.eye(1)
    return x.T @ Q @ x + u.T @ R @ u

def h(x):
    Qf = 1 * np.eye(2)
    return (x.T @ Qf @ x).reshape()


problem_params = {
        # 'system_name': 'double_integrator_unlimited',
        # 'system_name': 'double_integrator',
        'system_name': 'double_integrator_tuning',  # data copied from double_integrator
        'f': f,
        'l': l,
        'h': h,
        'T': 8,
        'nx': 2,
        'nu': 1,
        'U_interval': [-1, 1],
        'terminal_constraint': True,  # not tested with False for a long time
        'V_max': 16,
}

x_sample_scale = np.diag(np.array([1, 3]))
x_sample_cov = x_sample_scale @ x_sample_scale.T

# algo params copied from first resampling characteristics solvers
# -> so some of them might not be relevant
algo_params = {
        'pontryagin_solver_dt': 1/64,
        'pontryagin_solver_dense': False,

        'sampler_dt': 1/64,
        'sampler_burn_in': 0,
        'sampler_N_chains': 4,  # with pmap this has to be 4
        'sampler_samples': 2**8,  # actual samples = N_chains * samples
        'sampler_steps_per_sample': 4,
        'sampler_plot': True,
        'sampler_tqdm': False,
        # 'sampler_x_proposal_cov': np.array([[3.5, -4.5], [-4.5, 12]]),

        'x_sample_cov': x_sample_cov,
        'x_max_mahalanobis_dist': 2,

        'load_last': True,

        'nn_layersizes': [64, 64, 64],
        'nn_V_gradient_penalty': 50,
        'nn_batchsize': 128,
        'nn_N_max': 8192,
        'nn_N_epochs': 10,
        'nn_progressbar': True,
        'nn_testset_fraction': 0.1,
        'nn_ensemble_size': 16,
        'lr_staircase': False,
        'lr_staircase_steps': 4,
        'lr_init': 0.05,
        'lr_final': 0.005,

        'sim_T': 16,
        'sim_dt': 1/16,
        'sim_N': 32,
}

sysname = problem_params['system_name']

y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

# random subsample of data
N_subsample = 1024
idx = np.arange(y0s.shape[0])
idx_shuf = jax.random.permutation(jax.random.PRNGKey(0), idx)
# idx_subsample = idx_shuf[0:N_subsample]
idx_subsample = idx  # everything!

y0s_subsample = y0s[idx_subsample]
lamTs_subsample = lamTs[idx_subsample]

# do the integration again (we never saved the full data...)
solver = pontryagin_utils.make_pontryagin_solver_wrapped(problem_params, algo_params)

sols, _ = solver(lamTs_subsample)


# lets not bother with interpolation for the time being.

step = 1

valid_idx = np.abs(sols.ts[0]) < np.inf

# shape = (N_t)
sol_ts = sols.ts[0, valid_idx]
# shape = (N_trajectories, N_t, 5)
sol_ys = sols.ys[:, valid_idx, :]

vmin=0
vmax=16

for i in tqdm.tqdm(np.arange(sol_ys.shape[1])[::step]):

    pl.figure(figsize=(10, 6))

    pl.scatter(sol_ys[:, i, 0], sol_ys[:, i, 1], cmap='viridis', c=sol_ys[:, -1, 4], vmin=vmin, vmax=vmax, alpha=1/6)
    pl.savefig(f'animation/{i:04d}.png', dpi=300)
    pl.close('all')

