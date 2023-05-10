#!/usr/bin/env python
# jax
import jax
import jax.numpy as np
import optax
import diffrax

# other, trivial stuff
import numpy as onp

import tk as tkinter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import time
from tqdm import tqdm
from functools import partial

import pontryagin_utils
import plotting_utils
import array_juggling

# https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
from nn_utils import nn_wrapper
jax.tree_util.register_pytree_node(nn_wrapper,
                                   nn_wrapper._tree_flatten,
                                   nn_wrapper._tree_unflatten)

import yappi



def importance_sampling_bvp(problem_params, algo_params):

    '''

    we have the following problem. there is some distribution of states at
    t=0 for which we want to know the optimal controls. We can do this with
    the pontryagin principle. but we need to know the terminal (co)state
    distribution* that leads to the desired outcome at t=0.

    this is a first attempt at this by importance sampling.

    Idea:

    let κ = some desired distribution of states
    let μ_0 = κ

    loop for k=0, 1, ...:
        - sample x_i(T) ~ μ_k, find characteristic curves from there
        - decide for each characteristic curve how much we like it
            - concretely: define weights w_i for each trajectory i
            - maybe w_i ~ κ(x_i(0))
            - or w_i ~ κ(x_i(0)) / (something compensating for μ_i)
        - update μ to approximate the re-weighted distribution Σ_i w_i δ(x - x_i(T))
            - simplest method: gaussian with appropriate mean and cov.
            - or gaussian mixture model
            - or just MC style, like particle filter.

    the obvious problem with this is that the distributions can only
    shrink, not grow.


    *state for terminal value, costate for terminal constraint

    (all in problem_params:)
    f: dynamics. vector-valued function of t, x, u.
    l: cost. scalar-valued function of t, x, u.
    h: terminal cost. scalar-valued function of x.
    T: time horizon > 0. problem is solved over t ∈ [0, T] with V(T, x) = h(x).
    nx, nu: dimensions of state and input spaces. obvs should match f, l, h.
    '''

    key = jax.random.PRNGKey(0)

    # do this with keyword arguments and **dict instead?
    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']



    # define the dynamics of optimal trajectories according to PMP.
    f_forward = pontryagin_utils.define_extended_dynamics(problem_params)

    # solve pontryagin backwards, for vampping later.
    def pontryagin_backward_solver(y0, t0, t1):

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs
        saveat = diffrax.SaveAt(steps=True)
        steps = algo_params['integ_steps']
        dt = (t1 - t0)/(steps)  # negative if t1 < t0, backward integration just works

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt, y0=y0,
                saveat=saveat, max_steps=steps,
        )

        return solution, solution.ys[-1]

    # vmap = gangster!
    # vmap only across first argument.
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None, None)
    ))

    # helper function, expands the state vector x to the extended state vector y = [x, λ, v]
    # λ is the costate in the pontryagin minimum principle
    # h is the terminal value function
    def x_to_y(x, h):
        costate = jax.grad(h)(x)
        v = h(x)
        y = np.vstack([x, costate, v])

        return y

    # construct the initial (=terminal boundary condition...) extended state
    x_to_y_vmap = jax.vmap(lambda x: x_to_y(x, h))

    # later, when we have a (N_trajectories, nx) shaped vector of # terminal states x...
    # all_y_T = x_to_y_vmap(all_x_T)
    # init_ys = all_y_T


    key, subkey = jax.random.split(key)
    x_T = jax.random.multivariate_normal(
            subkey,
            mean=np.zeros(nx,),
            cov=algo_params['x_sample_cov'],
            shape=(algo_params['n_trajectories'],)
    ).reshape(algo_params['n_trajectories'], nx, 1)

    y_T = x_to_y_vmap(x_T)

    # very coarse example of importance sampling to find out which
    # terminal conditions lead to relevant initial states.

    desired_mean = np.zeros((nx,))
    desired_cov = np.eye(nx)
    desired_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, desired_mean, desired_cov)

    @jax.jit
    def scan_fct(carry, inp):

        # input = (t1)  # END of integration interval. t1 < t0 = T
        # carry = (sampling_mean, sampling_cov, key)
        # output = (sol_object)

        sampling_mean, sampling_cov, key = carry

        # sampling step first.
        # nicer for code but maybe reverse it for practical application?
        key, subkey = jax.random.split(key)
        x_T = jax.random.multivariate_normal(
                subkey,
                mean=sampling_mean,
                cov=sampling_cov,
                shape=(algo_params['n_trajectories'],)
        ).reshape(algo_params['n_trajectories'], nx, 1)

        y_T = x_to_y_vmap(x_T)

        t1 = inp

        sol_object, sol_vec = batch_pontryagin_backward_solver(y_T, T, t1)

        desired_likelihoods = desired_pdf(sol_vec[:, 0:nx, 0])
        sampling_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, sampling_mean, sampling_cov)
        sampling_likelihoods = sampling_pdf(x_T[:, 0])

        # importance sampling weight
        resampling_weights = desired_likelihoods / (1e-9 + sampling_likelihoods)

        # rejection sampling weights
        # resampling_weights = desired_likelihoods >= 0.001

        # cheap in between weights
        # resampling_weights = desired_likelihoods

        resampling_weights = resampling_weights / np.sum(resampling_weights)

        xs_flat = x_T.reshape(-1, nx)

        sampling_mean = np.mean(resampling_weights[:, None] * xs_flat, axis=0)  # or just zero?

        # subtract mean
        xs_zm = xs_flat - sampling_mean[None, :]

        # for zero mean RV x: cov(x) = E[x x'] = Σ_x p(x) * x x' = Σ_x (x*sqrt(p(x)) * (x*sqrt(p(x))'
        # so basically in comparison to the 'regular' covariance formula, we need to scale the
        # data points by the square root of their weights.
        # but where did the n-1 go? bessel correction mean estimate what?
        sqrt_scaled_datapts = np.sqrt(1e-9 + resampling_weights[:, None]) * xs_flat

        # then we have (wide matrix of column vecs) @ (tall matrix of row vecs)
        sampling_cov = sqrt_scaled_datapts.T @ sqrt_scaled_datapts

        # just for plotting, also put the state distribution at t1 in the output.
        t1_states = sol_vec[:, 0:nx, 0]
        t1_mu = np.mean(t1_states, axis=0)
        t1_states_zm = t1_states - t1_mu[None, :]

        t1_cov = t1_states.T @ t1_states / (algo_params['n_trajectories'] + 1)

        output = (sol_object, sampling_cov, resampling_weights, t1_cov)
        new_carry = (sampling_mean, sampling_cov, key)

        return new_carry, output

    N=10

    # a couple ehrenrunden at the end.
    t1s = np.concatenate([np.linspace(0, T, N, endpoint=False)[::-1], np.zeros([4])])

    N = t1s.shape[0]

    init_carry = (np.zeros(nx,), algo_params['x_sample_cov'], key)
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, t1s)

    sol_object = outputs[0]
    all_ys = sol_object.ys

    sampling_covs = outputs[1]
    resampling_ws = outputs[2]
    t1_covs = outputs[3]

    resampling_ws = (resampling_ws / resampling_ws.max(axis=1)[:, None])

    fig = pl.figure(figsize=(8, 3))
    ax0 = fig.add_subplot(111, projection='3d')

    for i in range(N):
        print(f'plotting iteration {i}...')

        t_plot = sol_object.ts[i, 0]
        x_plot = sol_object.ys[i, :, :, 0, 0].T
        y_plot = sol_object.ys[i, :, :, 1, 0].T

        c = matplotlib.colormaps.get_cmap('cividis')(i/N)

        # bc. apparently alpha cannot be a vector :(
        for j in range(x_plot.shape[1]):
            ax0.plot(t_plot, x_plot[:, j], y_plot[:, j], color=c, alpha=resampling_ws[i, j].item())


    pl.figure()
    def plot_covs(covs, col):
        pl.subplot(221)
        pl.plot(covs[:, 0, 0], c=col)
        pl.subplot(222)
        pl.plot(covs[:, 1, 0], c=col)
        pl.subplot(223)
        pl.plot(covs[:, 0, 1], c=col)
        pl.subplot(224)
        pl.plot(covs[:, 1, 1], c=col)

    plot_covs(sampling_covs, color='blue')
    plot_covs(t1_covs, color='green')
    pl.legend()
    pl.show()
    ipdb.set_trace()




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
            'T': 16,
            'nx': 2,
            'nu': 1
    }

    x_sample_scale = 1 * np.eye(2)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # resample when the mahalanobis distance (to the sampling distribution) is larger than this.
    resample_mahalanobis_dist = 2

    Q_x = np.linalg.inv(x_sample_cov) / resample_mahalanobis_dist**2

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'n_trajectories': 128,
            # 'dt': 1/128,
            # 'resample_interval': 1/4,
            # 'resample_type': 'minimal',
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,
            'integ_steps': 256,

            # 'nn_layersizes': (64, 64, 64, 64),
            # 'nn_batchsize': 128,
            # 'nn_N_epochs': 5,
            # 'nn_testset_fraction': 0.2,
            # 'nn_plot_training': True,
            # 'nn_train_lookback': 1/4,
            # 'nn_V_gradient_penalty': 100,

            # 'nn_retrain_final': True,
            # 'nn_progressbar': False,

            # 'lr_init': 1e-2,
            # 'lr_final': 5e-4,
            # 'lr_staircase': False,
            # 'lr_staircase_steps': 5,

            # 'plot_final': True
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    output = importance_sampling_bvp(problem_params, algo_params)
