#!/usr/bin/env python

import jax
import jax.numpy as np

import gradient_gp
import sampling

import ipdb
import matplotlib.pyplot as pl
import tqdm


def run_algo(problem_params, algo_params, key=None):

    if key is None:
        key = jax.random.PRNGKey(0)

    # first we generate a bunch of trajectories with the pontryagin sampler.
    key, subkey = jax.random.split(key)
    sample, integrate = sampling.pontryagin_sampler(problem_params, algo_params, key=subkey)

    key, subkey = jax.random.split(key)
    sol_obj, ys = integrate(sample(subkey, algo_params['sampler_n_trajectories']))

    nx = problem_params['nx']

    # fit a GP to the data.
    gp_params = {
            'log_amp': np.log(1),
            'log_scale': np.zeros(nx),
    }

    xs_gp, ys_gp, grad_flags_gp = gradient_gp.reshape_for_gp(ys)

    gp, gp_params = gradient_gp.get_optimised_gp(
            gradient_gp.build_gp,
            gp_params,
            xs_gp,
            ys_gp,
            grad_flags_gp,
            steps=algo_params['gp_iters'],
    )

    all_ys = ys

    for i in range(4):

        # idea:
        # - find points where the approximation is inaccurate
        # - guess which terminal conditions we might use to get useful new data
        #   at these points (hoping for help from the GP here)
        # - get the new data with the pontryagin solver
        # - re-fit the GP.

        # as a first attempt, we just generate loads of random states
        # and choose the ones with hightest GP uncertainty to take the
        # next samples.

        # not sure if this is guaranteed to do anything smart, probably to get
        # decent convergence guarantees we have to sample over a bounded domain,
        # this way it will just choose the outermost, most unlikely draws for
        # new samples because there we have no data yet.

        n_eval = 256
        xs_eval = jax.random.multivariate_normal(
                subkey,
                mean=np.zeros(nx),
                cov=algo_params['x_sample_cov'],
                shape=(n_eval,)
        )

        # no gradients! only the function.
        # although, should we not mainly care about accurate representation
        # of the gradient.....?
        gradflags_eval = np.zeros((n_eval,), dtype=np.int8)

        # condition the GP on available data & evaluate uncertainty at xs_eval.
        pred_gp = gp.condition(ys_gp, (xs_eval, gradflags_eval)).gp

        # find xs with largest value uncertainty. for simplicity again the same
        # number of trajectories as sampling -> less re-jitting.
        k = algo_params['sampler_n_trajectories']
        largest_vars, idx = jax.lax.top_k(pred_gp.variance, k)

        # these are the states we'd like to know more about.
        uncertain_xs = xs_eval[idx]
        uncertain_xs_gradflag = np.ones(k, dtype=np.int8)

        # find the weights the gp used to predict y = w.T @ ys_train
        # TODO next week: continue with this.
        X_pred = (uncertain_xs, uncertain_xs_gradflag)

        # very brute force for now...
        K_pt = gp.kernel(X_pred, gp.X)
        K_pp = gp.kernel(gp.X, gp.X)
        ws_pred = K_pt @ np.linalg.inv(K_pp)

        # somehow this is very nonzero
        print(ws_pred @ ys_gp - pred_gp.mean[idx])
        ipdb.set_trace()

        # ws = gradient_gp.get_gp_prediction_weights(gp, pred_gp, X_pred)












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
            'terminal_constraint': True,
    }

    x_sample_scale = 1 * np.eye(2)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'sampler_n_trajectories': 32,
            'sampler_dt': 1/16,
            'sampler_n_iters': 8,
            'sampler_n_extrarounds': 2,
            'sampler_strategy': 'importance',
            'sampler_deterministic': False,
            'sampler_plot': False,  # plotting takes like 1000x longer than the computation
            'sampler_returns': 'sampling_fct',

            'x_sample_cov': x_sample_cov,

            'gp_iters': 100,

    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    run_algo(problem_params, algo_params)
