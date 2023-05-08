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
    def pontryagin_backward_solver(y_final, tstart):

        '''
        to assemble the y vector, do this, but outside and with adjustments for vmap:
        # final condition (integration goes backwards...)
        costate_T = jax.grad(h)(x_T)
        v_T = h(x_T)
        y_final = np.vstack([x_T, costate_T, v_T])

        here we have t0 > t1, and negative dt (already defined here), to do backward integration

        when vmapping this we get back not a vector of solution objects, but a single
        solution object with an extra dimension in front of all others inserted in all arrays

        '''

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs
        saveat = diffrax.SaveAt(steps=True)
        dt = algo_params['dt']
        # WARNING this has different meaning than in the other script!
        integ_time = algo_params['resample_interval']
        max_steps = int(integ_time / dt)

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=tstart, t1=tstart-integ_time, dt0=-dt, y0=y_final,
                saveat=saveat, max_steps=max_steps
        )

        return solution, solution.ys[-1]

    # vmap = gangster!
    # in_axes[1] = None caueses it to not do dumb stuff with the second argument. dunno exactly why
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None)
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

    sampling_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(
            x, np.zeros(nx,), algo_params['x_sample_cov'])

    fig = pl.figure(figsize=(8, 3))
    ax0 = fig.add_subplot(111, projection='3d')

    for i in range(4):
        # backward sim.
        sol_object, sol_vec = batch_pontryagin_backward_solver(y_T, T)

        a = .1 * i

        for j in range(algo_params['n_trajectories']):
            if j==0:
                ax0.plot(sol_object.ys[j, :, 0].squeeze(), sol_object.ys[j, :, 1].squeeze(), color='black', alpha=a, label=f'iteration {i}')
            else:
                ax0.plot(sol_object.ys[j, :, 0].squeeze(), sol_object.ys[j, :, 1].squeeze(), color='black', alpha=a)

        # we are only interested in where the states end up, not the costates.
        earlier_likelihoods_desired = desired_pdf(sol_vec[:, 0:nx, 0])
        later_likelihoods_sampling = sampling_pdf(x_T[:, 0])

        # intuition behind resampling weights:
        # - if a trajectory lands at a high value of the desired pdf, it is
        #   good and we want more similar trajectories
        # - if a trajectory does so while also being somewhere where the
        #   sampling pdf is low, we want even more of that trajectory.
        resampling_weights = earlier_likelihoods_desired # / later_likelihoods_sampling

        reweighted_xs_flat = resampling_weights[:, None] * x_T[:, :, 0]

        # 'collapse' the re-weighted distribution into a gaussian again bc nice
        # and tractable. might later replace with more general class of distributions.

        sampling_mean = np.mean(reweighted_xs_flat, axis=0)  # or just zero?
        rw_zm = reweighted_xs_flat - sampling_mean[None, :]
        sampling_cov = rw_zm.T @ rw_zm / (algo_params['n_trajectories'] - 1)

        # ... and sample new terminal conditions.
        key, subkey = jax.random.split(key)
        x_T = jax.random.multivariate_normal(
                subkey,
                mean=sampling_mean,
                cov=sampling_cov,
                shape=(algo_params['n_trajectories'],)
        ).reshape(algo_params['n_trajectories'], nx, 1)

        y_T = x_to_y_vmap(x_T)

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
            'T': 2,
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
            'n_trajectories': 256,
            'dt': 1/128,
            'resample_interval': 1/4,
            'resample_type': 'minimal',
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,

            'nn_layersizes': (64, 64, 64, 64),
            'nn_batchsize': 128,
            'nn_N_epochs': 5,
            'nn_testset_fraction': 0.2,
            'nn_plot_training': True,
            'nn_train_lookback': 1/4,
            'nn_V_gradient_penalty': 100,

            'nn_retrain_final': True,
            'nn_progressbar': False,

            'lr_init': 1e-2,
            'lr_final': 5e-4,
            'lr_staircase': False,
            'lr_staircase_steps': 5,

            'plot_final': True
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'
    output = importance_sampling_bvp(problem_params, algo_params)
