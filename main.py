#!/usr/bin/env python

import jax
import jax.numpy as np

import gradient_gp
import sampling
import pontryagin_utils
import plotting_utils
import nn_utils

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

from jax.config import config
# config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)

import numpy as onp

# import matplotlib
# import matplotlib.pyplot as pl
# pl.rcParams['text.usetex'] = True
#
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })



def fit_value_fct(problem_params, algo_params, key):

    if algo_params['load_last']:

        sysname = problem_params['system_name']

        y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
        lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

        # seen this
        # plotting_utils.value_lambda_scatterplot(y0s[:, 0:2], y0s[:, -1], lamTs, save=False)

    else:

        key, subkey = jax.random.split(key)
        y0s, lamTs = sample_uniform(problem_params, algo_params, subkey)

    # shuffle data to 'get rid of' interdependence
    key, subkey = jax.random.split(key)
    permuted_idx = jax.random.permutation(subkey, np.arange(y0s.shape[0]))

    y0s = y0s[permuted_idx, :]
    lamTs = lamTs[permuted_idx, :]




    nx = problem_params['nx']

    V_nn = nn_utils.nn_wrapper(
            input_dim = nx,
            layer_dims = algo_params['nn_layersizes'],
            output_dim = 1,
    )

    key, subkey = jax.random.split(key)
    nn_params = V_nn.init_nn_params(subkey)


    batchsize = algo_params['nn_batchsize']
    N_epochs = algo_params['nn_N_epochs']


    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256])


    # for N_pts in 2**np.arange(4, 14):

    # easier to repeat data so we have the same size everytime and
    # the same training dynamics...
    # N_pts = 128
    # repeated_idx = np.mod(np.arange(y0s.shape[0]), N_pts)

    # nn_xs, nn_ys = np.split(y0s[repeated_idx], [nx], axis=1)
    nn_xs, nn_ys = np.split(y0s, [nx], axis=1)

    # normalise. just to experiment. todo, later un-normalise for inference.
    # nn_xs = nn_xs / nn_xs.std(axis=0)[None, -1]

    # this is kind of dumb. we need to normalise the gradient with the same factor
    # used for the value. otherwise the function we would be looking for would not
    # exist.

    # nn_ys = nn_ys / nn_ys[:, -1].std()

    nn_params, outputs = V_nn.train(
            nn_xs, nn_ys, nn_params, algo_params, key
    )





    # recreate train/test set here - kind of inelegant i know
    # logic copied from V_nn.train.
    testset_fraction = algo_params['nn_testset_fraction']
    testset_exists = testset_fraction > 0

    if not testset_exists:
        warnings.warn('this is untested and should not be done anyway')

    N_datapts = nn_xs.shape[0]
    xs_test = ys_test = None

    split_idx_float = (1-testset_fraction) * N_datapts
    split_idx_float_array = onp.array([split_idx_float])
    split_idx_int_array = split_idx_float_array.astype(onp.int32)

    xs, xs_test = np.split(nn_xs, split_idx_int_array)
    ys, ys_test = np.split(nn_ys, split_idx_int_array)


    plotting_utils.plot_nn_gradient_eval(V_nn, nn_params, xs, xs_test, ys, ys_test)

    plotting_utils.plot_nn_train_outputs(outputs)

    pl.figure()
    extent=5
    plotting_utils.plot_fct(partial(V_nn.nn.apply, nn_params),
            (-extent, extent), (-extent, extent), N_disc=256)

    plotting_utils.plot_fct_3d(partial(V_nn.nn.apply, nn_params),
            (-extent, extent), (-extent, extent), N_disc=256)

    pl.gca().scatter(nn_xs[0:100, 0], nn_xs[0:100, 1], nn_ys[0:100, -1])


    pl.show()
    ipdb.set_trace()


def evaluate_closedloop(V_nn, problem_params, algo_params, key):

    # obtain a monte carlo estimate of the control cost
    # expectation_x0~p(x0)

    N_sims = 100
    nx = problem_params['nx']

    x0s = jax.random.normal(key, shape=(N_sims, nx))



def sample_uniform(problem_params, algo_params, key):

    # so nice, exactly from paper
    Q_S = algo_params['x_Q_S']
    nx = problem_params['nx']

    # reward_fct = lambda x: -50 * np.maximum(0, x.T @ Q_S @ x - 1) + 5 * np.sqrt(0.01 + x.T @ np.array([[1,0],[0,0]]) @ x)
    # reward_fct = lambda x: -100 * np.maximum(0, x.T @ Q_S @ x - 1) + 10 * np.sqrt(0.01 + np.square(np.array([3, 1]) @ x))
    reward_fct = lambda x: -10 * np.maximum(0, x.T @ Q_S @ x - 1)
    reward_fct = lambda y: -10 * np.maximum(0, y[0:nx].T @ Q_S @ y[0:nx] - 1)  # S = some ellipse

    Vmax = problem_params['V_max']

    reward_fct = lambda y: -10 * np.maximum(0, y[-1] - Vmax)  # S = value sublevel set

    integrate = pontryagin_utils.make_pontryagin_solver_wrapped(problem_params, algo_params)

    y0s, λTs = sampling.geometric_mala_2(integrate, reward_fct, problem_params, algo_params, key)

    return y0s, λTs



if __name__ == '__main__':

    # simple control system. double integrator with friction term.
    def f(t, x, u):
        # p' = v
        # v' = f
        # f = -v**3 + u
        # clip so we don't have finite escape time when integrating backwards
        v_cubed = np.clip((np.array([[0, 1]]) @ x)**3, -10, 10)
        return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u - v_cubed)

    def l(t, x, u):
        Q = np.eye(2)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 1 * np.eye(2)
        return (x.T @ Qf @ x).reshape()


    problem_params = {
            'system_name': 'double_integrator',
            'f': f,
            'l': l,
            'h': h,
            'T': 8,
            'nx': 2,
            'nu': 1,
            'terminal_constraint': True,  # not tested with False for a long time
            'V_max': 15,
    }

    x_sample_scale = np.diag(np.array([1, 3]))
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'pontryagin_solver_dt': 1/64,

            # 'pontryagin_sampler_n_trajectories': 32,
            # 'pontryagin_sampler_n_iters': 8,
            # 'pontryagin_sampler_n_extrarounds': 2,
            # 'pontryagin_sampler_strategy': 'importance',
            # 'pontryagin_sampler_deterministic': False,
            # 'pontryagin_sampler_plot': False,  # plotting takes like 1000x longer than the computation
            # 'pontryagin_sampler_returns': 'functions',

            'sampler_dt': 1/64,
            'sampler_burn_in': 0,
            'sampler_N_chains': 4,  # with pmap this has to be 4
            'sampler_samples': 2**12,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 4,
            'sampler_plot': True,
            'sampler_tqdm': True,

            'x_sample_cov': x_sample_cov,
            'x_max_mahalanobis_dist': 2,

            # 'gp_iters': 100,
            # 'gp_train_plot': True,
            # 'N_learning_iters': 200,

            'load_last': True,

            'nn_layersizes': [64, 64, 64],
            'nn_V_gradient_penalty': 50,
            'nn_batchsize': 128,
            'nn_N_epochs': 10,
            'nn_progressbar': True,
            'nn_testset_fraction': 0.1,
            'lr_staircase': False,
            'lr_staircase_steps': 4,
            'lr_init': 0.01,
            'lr_final': 0.0001,
    }

    # the matrix used to define the relevant state space subset in the paper
    #   sqrt(x.T @ Σ_inv @ x) - max_dist
    # = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
    # so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
    algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    # to re-make the sample:
    # sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(10101))

    fit_value_fct(problem_params, algo_params, key=jax.random.PRNGKey(0))
