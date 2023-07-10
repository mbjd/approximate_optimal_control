#!/usr/bin/env python
import jax
import jax.numpy as np
import diffrax

import gradient_gp
import sampling
import pontryagin_utils
import plotting_utils
import nn_utils
import eval_utils

from main import sample_uniform, experiment_controlcost_vs_traindata

import ipdb
import scipy
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

import numpy as onp

from jax.config import config


def experiment_controlcost_vs_traindata_lqr_comparison(problem_params, algo_params, key):


    sysname = problem_params['system_name']

    assert sysname == 'double_integrator_linear', 'otherwise lqr comparison makes no sense'

    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

    # choose initial states for sim. the same each time.
    key, x0key = jax.random.split(key)

    # shuffle data to 'get rid of' interdependence
    key, subkey = jax.random.split(key)
    permuted_idx = jax.random.permutation(subkey, np.arange(y0s.shape[0]))
    y0s = y0s[permuted_idx, :]
    lamTs = lamTs[permuted_idx, :]

    nx = problem_params['nx']

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)
    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)

    # x0s from test set.
    # confirmed they are actually the same as the ones used in the experiment.
    # (if seed=0 in the main function)
    x0s = jax.random.choice(x0key, xs_eval, shape=(algo_params['sim_N'],))


    print('making comparison to closed form LQR solution')
    # too lazy to properly extract these from the problem description...
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0, 1]]).T
    Q = np.eye(2)
    R = np.eye(1)
    invR = R  # only if R = eye
    N = np.zeros((2, 1))

    # define matrix ricatti diff. eq.
    def P_dot(t, P, args=None):
        return -( A.T @ P + P @ A - (P @ B + N) @ invR @ (B.T @ P + N.T) + Q )

    P_T = 1 * np.eye(2)  # large to approximate terminal constraint

    def get_P0(P_T):
        term = diffrax.ODETerm(P_dot)
        solver = diffrax.Tsit5()

        # maybe easier to control the timing intervals like this?
        saveat = diffrax.SaveAt(steps=True)

        dt = -1/1024  # really small

        max_steps = int(problem_params['T'] / np.abs(dt))

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=problem_params['T'], t1=0., dt0=dt, y0=P_T,
                saveat=saveat, max_steps=max_steps,
        )

        return solution

    P_sol = get_P0(np.eye(2))

    P0 = P_sol.ys[-1]
    K0 = invR @ (B.T @ P0 + N.T)

    # compare with inf-horizon solution
    # from http://www.mwm.im/lqr-controllers-with-python/
    def lqr(A,B,Q,R):
        """Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """
        #ref Bertsekas, p.151

        #first, try to solve the ricatti equation
        X = onp.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

        #compute the LQR gain
        K = onp.matrix(scipy.linalg.inv(R)*(B.T*X))

        eigVals, eigVecs = scipy.linalg.eig(A-B*K)

        return K, X, eigVals

    K0_inf, P0_inf, eigvals = lqr(A, B, Q, R)
    K0_inf = invR @ (B.T @ P0_inf + N.T)

    # they are basically the same, we are good.
    # so now we step to calculating the cost. just like in the experiment itself :)

    ustar_fct = lambda x: -K0 @ x
    all_sols = eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)

    # extract cost...
    cost_mean = all_sols.ys[:, -1, nx].mean()
    cost_std = all_sols.ys[:, -1, nx].std()

    mean_std = np.array([cost_mean, cost_std])

    np.save(f'datasets/controlcost_lqr_meanstd_{sysname}.npy', mean_std)





# simple control system. double integrator with friction term.
def f(t, x, u):
    # p' = v
    # v' = f
    # f =  u

    # v_cubed = (np.array([[0, 1]]) @ x)**3
    return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u)

def l(t, x, u):
    Q = np.eye(2)
    R = np.eye(1)
    return x.T @ Q @ x + u.T @ R @ u

def h(x):
    Qf = 1 * np.eye(2)
    return (x.T @ Qf @ x).reshape()


problem_params = {
        # 'system_name': 'double_integrator_unlimited',
        'system_name': 'double_integrator_linear',
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
        'pontryagin_solver_dt': 1/32,

        # 'pontryagin_sampler_n_trajectories': 32,
        # 'pontryagin_sampler_n_iters': 8,
        # 'pontryagin_sampler_n_extrarounds': 2,
        # 'pontryagin_sampler_strategy': 'importance',
        # 'pontryagin_sampler_deterministic': False,
        # 'pontryagin_sampler_plot': False,  # plotting takes like 1000x longer than the computation
        # 'pontryagin_sampler_returns': 'functions',

        'sampler_dt': 1/32,
        'sampler_burn_in': 0,
        'sampler_N_chains': 4,  # with pmap this has to be 4
        'sampler_samples': 2**14,  # actual samples = N_chains * samples
        'sampler_steps_per_sample': 4,
        'sampler_plot': False,
        'sampler_tqdm': True,
        # 'sampler_x_proposal_cov': np.array([[3.5, -4.5], [-4.5, 12]]),

        'x_sample_cov': x_sample_cov,
        'x_max_mahalanobis_dist': 2,

        # 'gp_iters': 100,
        # 'gp_train_plot': True,
        # 'N_learning_iters': 200,

        'load_last': True,

        'nn_layersizes': [64, 64, 64],
        'nn_V_gradient_penalty': 50,
        'nn_batchsize': 128,
        'nn_N_epochs': 2,
        'nn_progressbar': True,
        'nn_testset_fraction': 0.1,
        'nn_ensemble_size': 16,
        'lr_staircase': False,
        'lr_staircase_steps': 4,
        'lr_init': 0.01,
        'lr_final': 0.0001,

        'sim_T': 16,
        'sim_dt': 1/16,
        'sim_N': 32,
}

# the matrix used to define the relevant state space subset in the paper
#   sqrt(x.T @ Σ_inv @ x) - max_dist
# = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
# so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

# sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(0))

key = jax.random.PRNGKey(0)
# experiment_controlcost_vs_traindata(problem_params, algo_params, key)
experiment_controlcost_vs_traindata_lqr_comparison(problem_params, algo_params, key)
