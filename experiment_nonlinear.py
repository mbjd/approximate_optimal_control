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



def experiment_baseline(problem_params, algo_params, key):

    ### this is exactly copied from experiment_controlcost_vs_traindata in main
    ### so hopefully we also get the same x0s with the same key :)
    sysname = problem_params['system_name']
    print(f'starting experiment baseline calculation for system "{sysname}"')

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

    # V_nn = nn_utils.nn_wrapper(
    #         input_dim = nx,
    #         layer_dims = algo_params['nn_layersizes'],
    #         output_dim = 1,
    # )

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)
    N_max = algo_params['nn_N_max']
    y0s = y0s[0:N_max]      # only N_max training pts
    lamTs = lamTs[0:N_max]  # https://tinyurl.com/2ywjf6pv
    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)

    # x0s from test set.
    x0s = jax.random.choice(x0key, xs_eval, shape=(algo_params['sim_N'],))

    # confirmed the same as used in controlcost experiments.
    # print('x0s for nonlinear baseline:')
    # print(x0s)

    '''
    rough plan:

    define ustar_fct that solves BVP to get optimal controls, maybe with
    even smaller dt, with initialisation from training data

    evaluate (just like lqr):
        all_sols = eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)

        # extract cost...
        cost_mean = all_sols.ys[:, -1, nx].mean()
        cost_std = all_sols.ys[:, -1, nx].std()

        mean_std = np.array([cost_mean, cost_std])
        print(f'mean cost: {cost_mean}')
        print(f'std. cost: {cost_std}')

        np.save(f'datasets/controlcost_bvp_meanstd_{sysname}.npy', mean_std)
    '''

    # VERY weird - if we load this new here it works perfectly. whyyy
    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')


    # alternatively:
    complete = False
    if complete:
        all_y0s = np.load(f'datasets/mcmc_complete/last_y0s_{sysname}.npy')
        all_lamTs = np.load(f'datasets/mcmc_complete/last_lamTs_{sysname}.npy')
        y0s = all_y0s.reshape(-1, 5)
        lamTs = all_lamTs.reshape(-1, 2)

    T = problem_params['T']

    # takes (yT, T, 0)
    # returns (sol, y0)
    solver = pontryagin_utils.make_single_pontryagin_solver(problem_params, algo_params)
    lamT_to_y0 = lambda lamT: solver(np.concatenate([np.zeros(2), lamT, np.zeros(1)]), T, 0)[1]

    # for better solutions :)

    algo_params_fancy = algo_params.copy()
    algo_params_fancy['pontryagin_solver_dt'] = 1/1024
    solver_fancy = pontryagin_utils.make_single_pontryagin_solver(problem_params, algo_params_fancy)
    lamT_to_y0_fancy = lambda lamT: solver(np.concatenate([np.zeros(2), lamT, np.zeros(1)]), T, 0)[1]


    # basically, evaluates PMP(λT) and dPMP(λT)/dλT, but with full extended
    # output y0 = (x0, lam0, v0)
    def y0_and_jac(lamT):

        y0 = lamT_to_y0(lamT)
        dy0_dlamT = jax.jacobian(lamT_to_y0)(lamT)
        return y0, dy0_dlamT

    def y0_and_jac_fancy(lamT):
        y0 = lamT_to_y0(lamT)
        dy0_dlamT = jax.jacobian(lamT_to_y0_fancy)(lamT)
        return y0, dy0_dlamT


    def newton_step(lamT, x0_goal, step):

        # adjusts lamT so that hopefully PMP(lamT) == x0_goal
        # returns: (new lamT, last y0)

        # newton iteration: basically solve the equation if replacing
        # function with its linear approx.
        # ---> solve this linear system for "new lamT":
        # dPMP(lamT)/dlamT (new lamT - current lamT)  = (x0_goal - x0 current)
        # dx0_dlamT         lamT_update               =  x0_error

        y0, dy0_dlamT = y0_and_jac(lamT)
        x0 = y0[0:2]
        dx0_dlamT = dy0_dlamT[0:2, :]
        assert dx0_dlamT.shape == (2, 2)

        x0_error = x0_goal - x0

        lamT_update = np.linalg.solve(dx0_dlamT, x0_error)

        new_lamT = lamT + step * lamT_update
        return new_lamT, y0

    def newton_step_fancy(lamT, x0_goal):

        y0, dy0_dlamT = y0_and_jac_fancy(lamT)
        x0 = y0[0:2]
        dx0_dlamT = dy0_dlamT[0:2, :]
        assert dx0_dlamT.shape == (2, 2)

        x0_error = x0_goal - x0

        lamT_update = np.linalg.solve(dx0_dlamT, x0_error)

        new_lamT = lamT + lamT_update
        return new_lamT, y0

    batch_newton_step = jax.jit(jax.vmap(newton_step, in_axes=(0, None, None)))

    def ustar_fct(x):
        assert x.shape == (2,)

        # find the k closest x0s in training data according to 2-norm
        k = 16

        diffs = y0s[:, 0:2] - x[None, :]
        sq_distances = np.square(diffs).sum(axis=1)
        values, idx_of_closest = jax.lax.top_k(-sq_distances, k)
        lamTs_closest = lamTs[idx_of_closest]

        # alternative:
        # - find w such that w.T @ closest_x0s = x
        # - then initialise with lamT = w.T @ closest_lamTs
        x0s_closest = y0s[idx_of_closest, 0:2]
        # w, residuals, rank, svals = np.linalg.lstsq(x0s_closest.T, x0)
        # lamT_guess = w @ lamTs_closest

        # # add some noise to it for different inits
        # next_lamTs = lamT_guess[None, :] + 0.001 * jax.random.normal(jax.random.PRNGKey(0), (k, 2))

        # first, do batched newton iteration to get them all closer to x0
        next_lamTs = lamTs_closest + 0.0001 * jax.random.normal(jax.random.PRNGKey(0), lamTs_closest.shape)
        all_lamTs = [next_lamTs]
        all_y0s = []


        # # newton with smaller step size initially - seems to not help a lot
        # for i in range(10):
        #     next_lamTs, next_y0s = batch_newton_step(next_lamTs, x, .5)
        #     # next_lamT, y0 = newton_step(next_lamT, x)

        #     all_lamTs.append(next_lamTs)
        #     all_y0s.append(next_y0s)
        #     print(f'finished newton iter {i}')

        for i in range(10):

            next_lamTs, next_y0s = batch_newton_step(next_lamTs, x, 1)
            # next_lamT, y0 = newton_step(next_lamT, x)

            all_lamTs.append(next_lamTs)
            all_y0s.append(next_y0s)
            print(f'finished newton iter {i}')

        plot_newton_iter = False
        if plot_newton_iter:
            all_lamTs = np.array(all_lamTs) # (N_steps,   k, nx)
            all_y0s = np.array(all_y0s)     # (N_steps-1, k, 2*nx+1)
            all_x0_errors = all_y0s[:, :, 0:2] - x[None, None, :]
            x0_errnorms = np.linalg.norm(all_x0_errors, axis=2)

            lamT_updates_norms = np.linalg.norm(np.diff(all_lamTs, axis=0), axis=2)

            pl.figure()
            pl.semilogy(x0_errnorms, color='tab:blue', alpha=.3)
            pl.semilogy(lamT_updates_norms, color='tab:green', alpha=.3)

            pl.figure()
            pl.plot(all_y0s[:, :, 0], all_y0s[:, :, 1], color='tab:blue', alpha=.1)
            pl.scatter(x[0], x[1], c='tab:red')
            for j in range(all_y0s.shape[0]):
                pl.scatter(all_y0s[j, :, 0], all_y0s[j, :, 1], label=f'iteration {j}', alpha=.1)
            pl.legend()


        # find the one that got closest.
        x0_errs = next_y0s[:, 0:2] - x[None, :]
        x0_errnorms = np.linalg.norm(x0_errs, axis=1)
        best_sol_idx = x0_errnorms.argmin()
        best_lamT = all_lamTs[-1][best_sol_idx] # -1 or -2 correct? dunno

        best_lam0 = next_y0s[best_sol_idx][2:4]
        ustar = pontryagin_utils.u_star_new(x, best_lam0, problem_params)

        print(f'x0 err: {x0_errs[best_sol_idx]}')


        '''
        # noticed that this literally changes the input u* by either 0 or
        # like 1e-15. not worth doing
        # ...and improve it further using fancy ode solver
        best_lamT_fancy, y0 = newton_step_fancy(best_lamT, x)
        best_lamT_fancy, y0 = newton_step_fancy(best_lamT_fancy, x)

        new_err = y0[0:2] - x
        print(f'x0 err after fancy solver: {new_err}')
        ipdb.set_trace()

        lam0 = y0[2:4]
        ustar = pontryagin_utils.u_star_new(x, lam0, problem_params)
        '''

        return ustar

    fancy_batch_sol = jax.vmap(lamT_to_y0_fancy)

    # this works with jit hallelujah
    # now, does it give nice enough control inputs? specifically, smooth?
    # if not, maybe weight the data for lstsq according to # exp(-distance**2)?
    def ustar_fct_alt(x, k):
        # find the k closest x0s in training data according to 2-norm
        # typical closeness length scale is about 0.05

        diffs = y0s[:, 0:2] - x[None, :]
        sq_distances = np.square(diffs).sum(axis=1)
        values, idx_of_closest = jax.lax.top_k(-sq_distances, k)

        lamTs_closest = lamTs[idx_of_closest]

        y0s_closest = y0s[idx_of_closest, :]
        lam0s_closest = y0s_closest[:, 2:4]

        # some jitter, small but deterministic, to improve linear fit
        lamTs_closest = lamTs_closest + .0001 * jax.random.normal(jax.random.PRNGKey(0), (k, 2))

        # refine with fancy solver.
        y0s_refined = fancy_batch_sol(lamTs_closest)

        # print(lamT_to_y0_fancy)

        # fit a linear function to those closest costates to hopefully
        # predict at x.

        # this here was not linear function fitting, but instead some kind
        # of accidental kernel method or something.  {{{

        # new_sq_dists = np.linalg.norm(y0s_refined[:, 0:2] - x[None, :], axis=1)
        # length scale for k=32. probably proportiaonal to k^(1/n) or # something
        # penalty_diagonal = new_sq_dists / 0.02

        # we want w such that: w.T @ xs_closest = x
        # change the problem into l2 regularised regression to place
        # emphasis on closer points.
        # so we want: argmin w || A w - b ||_2^2 + || diag(penalty) @ w ||_2^2
        # same if we expand A <- [A; diag(penalty) and b <- [b;  0]
        # ipdb.set_trace()

        # A = np.row_stack([y0s_refined[:, 0:2].T, np.diag(penalty_diagonal)])
        # b = np.concatenate([x, np.zeros(k)])
        # A = y0s_refined[:, 0:2].T
        # b = x
        # ws, _, _, _ = np.linalg.lstsq(A, b)
        # lam_pred = ws @ y0s_refined[:, 2:4]
        # print(f'ws: {ws}')

        # is this even the right way of fitting a linear function??? why
        # are we searching over 32 parameters??
        # we want: lambda(x) = a + b x_0 + c x_1
        # what we are doing is different...
        # }}}

        # proper linear regression. A @ w = costate, where A = [1s x0s x1s]
        A = np.column_stack([np.ones(k), y0s_refined[:, 0:2]])
        b = y0s_refined[:, 2:4]
        w, _, _, _ = np.linalg.lstsq(A, b)

        lam_pred = np.concatenate([np.ones(1), x]) @ w

        ustar = pontryagin_utils.u_star_new(x, lam_pred, problem_params)

        plot=False # for jit set this false ofc
        if plot:
            fig = pl.figure()
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            ax1.scatter(y0s_refined[:, 0], y0s_refined[:, 1], y0s_refined[:, 2], c='tab:blue')
            ax1.scatter(x[0], x[1], lam_pred[0], c='tab:red')

            ax2.scatter(y0s_refined[:, 0], y0s_refined[:, 1], y0s_refined[:, 3], c='tab:blue')
            ax2.scatter(x[0], x[1], lam_pred[1], c='tab:red')
            pl.show()
            # pl.scatter(x[0], x[1], c='tab:red')

        return ustar


    '''
    ideas for more ustar fcts
    - importance sampling style: find closest x0s, evaluate PMP at
      corresponding lambdas, set IS weights to some gaussian centered at x,
      repeat with smaller and smaller region
    - just apply adam to try and get them closer? maybe that followed by
      linear interpolation?
    - the current linear fit approach, but instead of taking the n k
      closest x0s and their lambdaTs which leads to discontinuity, instead:
       - evaluate some kernel weight that is high when x is close to x0
       - interpret those weights over lambda_T as a distribution
       - find its mean and covariance
       - take a normal sample of λTs with fixed key for that mean and cov
       - propagate PMP from those λTs to find great data close to x0
       - linear fit around that, predict λ0, get u*
    '''

    # # just the ones that cause problems
    # idx = np.array([4, 9, 26], dtype=np.int32) - 1

    # for x0 in x0s[idx]:
    #     # u_newton = ustar_fct(x0)
    #     u_lstsq = ustar_fct_alt(x0)

    #     print(u_lstsq)
    #     # print(f'u newton: {u_newton}         u lstsq: {u_lstsq}')


    # ustar_batched = jax.jit(jax.vmap(ustar_fct_alt, in_axes=(0, None)), static_argnums=1)

    # randvec = np.array([.3, -1])

    # for k in [4, 8, 16, 32, 64]:

    #     alphas = np.linspace(-.6, -.59, 501)
    #     xs = alphas[:, None] * randvec[None, :]
    #     us = ustar_batched(xs, k)

    #     pl.plot(alphas, us, label=f'k = {k}', alpha=.5)
    # pl.legend()
    # pl.show()
    # ipdb.set_trace()


    # i_test = 102
    # err = lamT_to_y0(lamTs[i_test]) - y0s[i_test]
    # print(f'error = {err}')

    # ipdb.set_trace()

    # # pl.show()

    # # pl.scatter(x0s[:, 0], x0s[:, 1])
    # # pl.show()
    # # ipdb.set_trace()

    # just to try something overnight
    import time
    print(time.time())
    ustar_fct = lambda x: ustar_fct_alt(x, 16)
    eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)
    print(time.time())

    # extract cost...
    cost_mean = all_sols.ys[:, -1, nx].mean()
    cost_std = all_sols.ys[:, -1, nx].std()

    mean_std = np.array([cost_mean, cost_std])
    print(f'mean cost: {cost_mean}')
    print(f'std. cost: {cost_std}')

    np.save(f'datasets/controlcost_bvp_meanstd_{sysname}.npy', mean_std)
    print(time.time())



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
            'pontryagin_solver_dt': 1/32,
            'pontryagin_solver_dense': False,

            'sampler_dt': 1/512,
            'sampler_burn_in': 8,
            'sampler_N_chains': 32,  # with pmap this has to be 4
            'samper_samples_per_chain': 2**6,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 16,
            'sampler_plot': True,
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
    # sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(0))


    key = jax.random.PRNGKey(0)

    # ode_dt_sweep(problem_params, algo_params)
    # experiment_controlcost_vs_traindata(problem_params, algo_params, key)
    experiment_baseline(problem_params, algo_params, key)
