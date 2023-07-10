#!/usr/bin/env python

import jax
import jax.numpy as np

import gradient_gp
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



def train_single(problem_params, algo_params, key):

    sysname = problem_params['system_name']

    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

    # choose initial states for sim. the same each time.
    key, x0key = jax.random.split(key)

    # shuffle data to 'get rid of' interdependence
    key, subkey = jax.random.split(key)
    y0_idx = np.arange(y0s.shape[0])
    permuted_idx = jax.random.permutation(subkey, y0_idx)
    y0s = y0s[permuted_idx, :]
    lamTs = lamTs[permuted_idx, :]

    nx = problem_params['nx']

    V_nn = nn_utils.nn_wrapper(
            input_dim = nx,
            layer_dims = algo_params['nn_layersizes'],
            output_dim = 1,
    )

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)
    N_max = algo_params['nn_N_max']
    y0s = y0s[0:N_max]  # only 16384 training pts
    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)

    # shuffle once for each PRNG key
    y0s_shuffled = jax.random.permutation(key, y0s, axis=0, independent=False)


    # 'subsample' data = take first point_N and repeat
    nn_xs, nn_ys = np.split(y0s_shuffled, [nx], axis=1)

    # train NN ensemble
    nn_params, outputs = V_nn.ensemble_init_and_train(
            jax.random.split(key, algo_params['nn_ensemble_size']), nn_xs, nn_ys, algo_params
    )

    # evaluate test loss
    mean_pred, std_pred = V_nn.ensemble_mean_std(nn_params, xs_eval)
    costate_test_loss = np.mean((mean_pred[:, 1:] - gradients_eval)**2)

    plotting_utils.plot_nn_train_outputs(outputs)
    plotting_utils.plot_nn_ensemble(V_nn, nn_params, (-5, 5), (-10, 10), Nd=201)
    pl.show()

    ipdb.set_trace()


if __name__ == '__main__':

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
            'system_name': 'double_integrator',
            'f': f,
            'l': l,
            'h': h,
            'T': 8,
            'nx': 2,
            'nu': 1,
            'terminal_constraint': True,  # not tested with False for a long time
            'V_max': 16,
    }

    x_sample_scale = np.diag(np.array([1, 3]))
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'pontryagin_solver_dt': 1/64,

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
            'nn_ensemble_size': 8,
            'lr_staircase': False,
            'lr_staircase_steps': 4,
            'lr_init': 0.05,
            'lr_final': 0.005,

            'sim_T': 16,
            'sim_dt': 1/16,
            'sim_N': 32,
    }

    k = jax.random.PRNGKey(0)
    train_single(problem_params, algo_params, k)
