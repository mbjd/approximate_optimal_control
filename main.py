#!/usr/bin/env python

import jax
import jax.numpy as np

import gradient_gp
import sampling
import pontryagin_utils

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import tqdm
from functools import partial

from jax.config import config
# config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

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



def uniform_sampling_learning(problem_params, algo_params, key):

    if algo_params['load_last']:

        y0s = np.load('datasets/last_y0s.npy')
        lamTs = np.load('datasets/last_lamTs.npy')

    else:

        key, subkey = jax.random.split(key)
        y0s, lamTs = sample_uniform(problem_params, algo_params, subkey)

    # shuffle data to 'get rid of' interdependence
    key, subkey = jax.random.split(key)
    permuted_idx = jax.random.permutation(subkey, np.arange(y0s.shape[0]))

    y0s = y0s[permuted_idx, :]
    lamTs = lamTs[permuted_idx, :]

    # partition data into init, main, eval.
    # init = data for fitting first GP and optimising GP hyperparams.
    # main = big pool to sample from when fitting GP.
    # eval = holdout set to do MC estimate of approximation error.
    init_y0s, main_y0s, eval_y0s = np.split(y0s, [32, y0s.shape[0]-256], axis=0)


    # make GP, optimise hyperparams.
    nx = problem_params['nx']
    initparams = {
            'log_amp': np.log(1),
            'log_scale': np.zeros(nx),
    }

    gp_xs, gp_ys, gp_gradflags = gradient_gp.reshape_for_gp(init_y0s)

    gp, opt_params = gradient_gp.get_optimised_gp(
            gradient_gp.build_gp,
            initparams,
            gp_xs,
            gp_ys,
            gp_gradflags,
    )

    main_xs, main_ys, main_gradflags = gradient_gp.reshape_for_gp(main_y0s)
    eval_xs, eval_ys, eval_gradflags = gradient_gp.reshape_for_gp(eval_y0s)

    N_iters = algo_params['N_learning_iters']

    mean_stds = onp.zeros(N_iters)
    rmses = onp.zeros(N_iters)

    for i in range(N_iters):
        # just add one more sample to the GP.
        # None to match shapes
        gp_xs = np.concatenate([gp_xs, main_xs[None, i]], axis=0)
        gp_ys = np.concatenate([gp_ys, main_ys[None, i]], axis=0)
        gp_gradflags = np.concatenate([gp_gradflags, main_gradflags[None, i]], axis=0)

        gp = gradient_gp.build_gp(opt_params, gp_xs, gp_gradflags)

        # evaluate approximation accuracy.
        # this takes ages -- cannot really jit because we are always
        # changing size.
        pred_gp = gp.condition(gp_ys, (eval_xs, eval_gradflags)).gp
        y_pred = pred_gp.loc
        y_std = np.sqrt(pred_gp.variance)

        mean_std = np.mean(y_std)
        rmse = np.sqrt(np.mean(np.square(y_pred - eval_ys)))
        print(f'i = {i}')
        print(f'mean standard deviation (hehe): {mean_std}')
        print(f'rmse: {rmse}')

        mean_stds[i] = mean_std
        rmses[i] = rmse

    pl.plot(mean_stds, label='mean std')
    pl.plot(rmses, label='rmses')
    pl.legend()
    pl.show()

    ipdb.set_trace()


    # pred_gp = trained_gp.condition(ys, (X_pred, pred_grad_flag)).gp
    # y_pred = pred_gp.loc
    # y_std = np.sqrt(pred_gp.variance)








def sample_uniform(problem_params, algo_params, key):

    # so nice, exactly from paper
    Q_S = algo_params['x_Q_S']
    nx = problem_params['nx']

    # reward_fct = lambda x: -50 * np.maximum(0, x.T @ Q_S @ x - 1) + 5 * np.sqrt(0.01 + x.T @ np.array([[1,0],[0,0]]) @ x)
    # reward_fct = lambda x: -100 * np.maximum(0, x.T @ Q_S @ x - 1) + 10 * np.sqrt(0.01 + np.square(np.array([3, 1]) @ x))
    reward_fct = lambda x: -10 * np.maximum(0, x.T @ Q_S @ x - 1)
    reward_fct = lambda y: -10 * np.maximum(0, y[0:nx].T @ Q_S @ y[0:nx] - 1)  # S = some ellipse

    Vmax = 10
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
            'f': f,
            'l': l,
            'h': h,
            'T': 8,
            'nx': 2,
            'nu': 1,
            'terminal_constraint': True,  # not tested with False for a long time
    }

    x_sample_scale = np.diag(np.array([1, 3]))
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'pontryagin_solver_dt': 1/16,

            # 'pontryagin_sampler_n_trajectories': 32,
            # 'pontryagin_sampler_n_iters': 8,
            # 'pontryagin_sampler_n_extrarounds': 2,
            # 'pontryagin_sampler_strategy': 'importance',
            # 'pontryagin_sampler_deterministic': False,
            # 'pontryagin_sampler_plot': False,  # plotting takes like 1000x longer than the computation
            # 'pontryagin_sampler_returns': 'functions',

            'sampler_dt': 1/128,
            'sampler_burn_in': 0,
            'sampler_N_chains': 4,  # with pmap this has to be 4
            'sampler_samples': 2**13,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 1,
            'sampler_plot': True,
            'sampler_tqdm': True,

            'x_sample_cov': x_sample_cov,
            'x_max_mahalanobis_dist': 2,

            'gp_iters': 100,
            'gp_train_plot': False,
            'N_learning_iters': 100,

            'load_last': True,
    }

    # the matrix used to define the relevant state space subset in the paper
    #   sqrt(x.T @ Σ_inv @ x) - max_dist
    # = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
    # so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
    algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    uniform_sampling_learning(problem_params, algo_params, key=jax.random.PRNGKey(0))

    sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(121))
