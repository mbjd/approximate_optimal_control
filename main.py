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


def fit_ensemble_value(problem_params, algo_params, key):


    if algo_params['load_last']:

        sysname = problem_params['system_name']

        y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
        lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

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

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)

    # only use part of the data.
    repeated_idx = np.mod(np.arange(y0s.shape[0]), algo_params['nn_N_trainpts'])
    nn_xs, nn_ys = np.split(y0s[repeated_idx], [nx], axis=1)

    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)



    # train the nns
    nn_params, outputs = V_nn.ensemble_init_and_train(
            jax.random.split(key, algo_params['nn_ensemble_size']), nn_xs, nn_ys, algo_params
    )

    # means, stds = V_nn.ensemble_mean_std(nn_params, xs_eval)

    # make some plots.
    # plotting_utils.plot_nn_ensemble(V_nn, nn_params, (-3, 3), (-8, 8))

    return V_nn, nn_params




def experiment_controlcost_vs_traindata(problem_params, algo_params, key):

    # train the NN with different amounts of training data.
    # each time, evaluate:
    #  - mean λ approx. error
    #  - control cost


    sysname = problem_params['system_name']

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

    V_nn = nn_utils.nn_wrapper(
            input_dim = nx,
            layer_dims = algo_params['nn_layersizes'],
            output_dim = 1,
    )

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)
    N_max = algo_params['nn_N_max']
    y0s = y0s[0:N_max]  # only N_max training pts
    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)

    # x0s from test set.
    x0s = jax.random.choice(x0key, xs_eval, shape=(algo_params['sim_N'],))

    point_Ns = 2**np.arange(3, int(1 + np.log2(N_max)))
    print(point_Ns)
    N_keys = 16

    # light version :)
    # point_Ns = 2**np.arange(8, int(1 + np.log2(N_max)))
    # N_keys = 2

    # axis 0: which PRNG key was used?
    # axis 1: which point number?
    # axis 2: which type of data?
    #  index 0: number of training points used (-> x axis for plot)
    #  index 1: costate test loss
    #  index 2: costate prediction std. maximum <- this is new! TODO: adapt plotting function
    #  index 3: costate prediction std. mean    <-
    #  index 4: mean control cost
    #  index 5: std. dev. control cost.

    out_data = onp.zeros((N_keys, point_Ns.shape[0], 6))

    for i in range(N_keys):

        key = jax.random.PRNGKey(i)

        # shuffle once for each PRNG key
        y0s_shuffled = jax.random.permutation(key, y0s, axis=0, independent=False)

        for j, point_N in enumerate(point_Ns):

            NN_key, key = jax.random.split(key)

            out_data[i, j, 0] = point_N

            # 'subsample' data = take first point_N and repeat
            repeated_idx = np.mod(np.arange(y0s_shuffled.shape[0]), point_N)
            nn_xs, nn_ys = np.split(y0s_shuffled[repeated_idx], [nx], axis=1)

            # train NN ensemble
            nn_params, outputs = V_nn.ensemble_init_and_train(
                    jax.random.split(NN_key, algo_params['nn_ensemble_size']), nn_xs, nn_ys, algo_params
            )

            # evaluate test loss
            mean_pred, std_pred = V_nn.ensemble_mean_std(nn_params, xs_eval)
            costate_test_loss = np.mean((mean_pred[:, 1:] - gradients_eval)**2)

            out_data[i, j, 1] = costate_test_loss
            out_data[i, j, 2] = std_pred[:, 1:].max()
            out_data[i, j, 3] = std_pred[:, 1:].mean()

            # simulate closed loop
            all_sols = eval_utils.closed_loop_eval_nn_ensemble(problem_params, algo_params, V_nn, nn_params, x0s)

            # record control cost.
            cost_mean = all_sols.ys[:, -1, nx].mean()
            cost_std = all_sols.ys[:, -1, nx].std()

            out_data[i, j, 4] = cost_mean
            out_data[i, j, 5] = cost_std

            # just because i wanna be extra extra sure to not lose data
            np.save(f'datasets/trainpts_controlcost_data_{sysname}.npy', out_data)


    try:
        np.save(f'datasets/trainpts_controlcost_data_{sysname}.npy', out_data)
    except:
        print('it went wrong :( handle this ipdb session with care')

    # ipdb.set_trace()




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

def ode_dt_sweep(problem_params, algo_params):

    '''
    as an exception, make the figure here as well, since otherwise it is a mess to store data
    of varying size
    '''

    from make_figs import dpi, halfwidth, save, save_fig_wrapper

    fig, axs = pl.subplots(3, 1, figsize=(halfwidth, halfwidth))

    # dts = 2.0 ** -np.arange(2, 6)
    dts = np.logspace(-2, -8, 10, base=2)

    lamT = np.array([0.1, -.05])  # should be ok for linear and nonlinear example in report
    lamT = np.array([-.05, -.025])  # should be ok for linear and nonlinear example in report
    yT = np.concatenate([np.zeros(2), lamT, np.zeros(1)])

    T = problem_params['T']

    for i, dt in enumerate(dts):
        print(f'dt = {dt}')

        algo_params['pontryagin_solver_dt'] = dt
        solver = pontryagin_utils.make_single_pontryagin_solver(problem_params, algo_params)

        sol, _  = solver(yT, T, 0.)

        a = i/dts.shape[0]

        red, green, _ = np.eye(3)
        c = a * red + (1-a) * green
        c_tuple = [i.item() for i in c]

        axs[0].plot(sol.ts, sol.ys[:, 2], c=c_tuple)
        axs[1].plot(sol.ts, sol.ys[:, 3], c=c_tuple)
        axs[2].plot(sol.ts, sol.ys[:, 4], c=c_tuple)

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
            'samper_samples_per_chain': 2**8,  # actual samples = N_chains * samples
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

    # the matrix used to define the relevant state space subset in the paper
    #   sqrt(x.T @ Σ_inv @ x) - max_dist
    # = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
    # so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
    algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    # to re-make the sample:
    # sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(0))


    key = jax.random.PRNGKey(0)

    ode_dt_sweep(problem_params, algo_params)
    # experiment_controlcost_vs_traindata(problem_params, algo_params, key)
