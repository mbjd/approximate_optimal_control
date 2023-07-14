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

from main import sample_uniform, experiment_controlcost_vs_traindata, ode_dt_sweep

from jax.config import config
jax.config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


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
            # 'system_name': 'double_integrator',
            'system_name': 'double_integrator_corrected',  # data copied from double_integrator
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
            'pontryagin_solver_dt': 1/8,
            'pontryagin_solver_dense': False,

            'sampler_dt': 1/64,
            'sampler_burn_in': 8,
            'sampler_N_chains': 64,  # with pmap this has to be 4
            'samper_samples_per_chain': 2**6,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 4,
            'sampler_plot': False,
            'sampler_tqdm': False,

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

    # the matrix used to define the relevant state space subset in the paper
    #   sqrt(x.T @ Σ_inv @ x) - max_dist
    # = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
    # so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
    algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    # to re-make the sample:
    sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(0))


    key = jax.random.PRNGKey(0)

    ode_dt_sweep(problem_params, algo_params)
    # experiment_controlcost_vs_traindata(problem_params, algo_params, key)
