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

import frozendict



def hjb_characteristics_solver(problem_params, algo_params):
    '''
    (all in problem_params:)
    f: dynamics. vector-valued function of t, x, u.
    l: cost. scalar-valued function of t, x, u.
    h: terminal cost. scalar-valued function of x.
    T: time horizon > 0. problem is solved over t ∈ [0, T] with V(T, x) = h(x).
    nx, nu: dimensions of state and input spaces. obvs should match f, l, h.

    algo_params: dictionary with algorithm tuning parameters.
    '''


    # do this with keyword arguments and **dict instead?
    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']


    # define NN first.
    key = jax.random.PRNGKey(0)
    V_nn = nn_wrapper(
            input_dim  = 1 + nx,   # inputs are vectors containing (t, x)
            layer_dims = algo_params['nn_layersizes'],
            output_dim = 1,
    )

    key, subkey = jax.random.split(key)
    nn_params = V_nn.init_nn_params(subkey)

    # the algo params relevant for nn training.
    # so we have only immutable stuff left.
    is_nn_key = lambda key_and_val: key_and_val[0].startswith('nn_')
    algo_params_nn = frozendict.frozendict(filter(is_nn_key, algo_params.items()))




    # generate initial state points according to the specified distribution.
    key, subkey = jax.random.split(key)
    all_x_T = jax.random.multivariate_normal(
            subkey,
            mean=np.zeros(nx,),
            cov=algo_params['x_sample_cov'],
            shape=(algo_params['n_trajectories'],)
    ).reshape(algo_params['n_trajectories'], nx, 1)

    max_steps = int(T / algo_params['dt'])








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
    all_y_T = x_to_y_vmap(all_x_T)
    init_ys = all_y_T








    # calculate all the time and index parameters in advance to avoid the mess.

    dt = algo_params['dt']
    t = T

    # image for visualisation:

    # 0                                                                     T
    # |---------------------------------------------------------------------|
    # |         |         |         |         |         |         |         | resample_intervals
    # | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | timesteps

    # all arrays defined here contain all but the last (at T) steps.

    # somehow with % it is inexact sometimes.
    # remainder = algo_params['resample_interval'] % dt
    # so we calculate explicitly the difference between the quotient and the nearest integer.
    remainder = algo_params['resample_interval'] / dt - int(0.5 + algo_params['resample_interval'] / dt)
    assert np.allclose(remainder, 0, atol=1e-3), 'dt does not evenly divide resample_interval; problems may arise.'

    n_resamplings = int(T / algo_params['resample_interval'])
    timesteps_per_resample = int(0.5 + algo_params['resample_interval'] / dt)

    resampling_indices = np.arange(n_resamplings) * timesteps_per_resample
    resampling_ts = resampling_indices * dt

    n_timesteps = n_resamplings * timesteps_per_resample + 1 # so we have the boundary condition as well
    all_indices = onp.arange(n_timesteps)
    all_ts = all_indices * dt




    # make space for output data
    # full solution array. onp so we can modify in place.
    # every integration step fills a slice all_sols[:, i:i+timesteps_per_resample, :, :], for i in resampling_indices.
    # final ..., 1) in shape so we have nice column vectors just as everywhere else.
    all_sols = onp.zeros((algo_params['n_trajectories'], n_timesteps, 2*nx+1, 1))
    where_resampled = onp.zeros((algo_params['n_trajectories'], n_timesteps), dtype=bool)

    # the terminal condition separately
    # technically redundant but might make function fitting nicer later
    all_sols[:, -1, :, :] = all_y_T

    train_loss_curves = []
    test_loss_curves = []

    total_start = time.time()
    integration_time = 0
    nn_time = 0

    # the main loop.
    # basically, do backwards continuous-time approximate dynamic programming.
    # alternate between particle-based, batched pontryagin/characteristic step,
    # followed by resampling step.

    it = zip(resampling_indices[::-1], resampling_ts[::-1])
    # for (resampling_i, resampling_t) in it:
    for (resampling_i, resampling_t) in tqdm(it, total=n_resamplings):

        # we integrate backwards over the time interval [resampling_t, resampling_t + resample_interval].
        # corresponding data saved with save_idx = [:, resampling_i:resampling_i + timesteps_per_resample, : :]
        # at all_sols[save_idx]

        integ_start = time.time()
        start_t = resampling_t + algo_params['resample_interval']
        integration_time += time.time() - integ_start

        # the main dish.
        # integrate the pontryagin necessary conditions backward in time, for a time period of
        # algo_params['resample_interval'], for a whole batch of terminal conditions.
        sol_object, final_ys = batch_pontryagin_backward_solver(init_ys, start_t)


        # sol_object.ys is the full solution array. it has shape (n_trajectories, n_timesteps, 2*nx+1, 1).
        # the 2*nx+1 comes from the 'extended state' = (x, λ, v).

        # sol object info is saved the other way around, going from initial=large to final=small times.
        save_idx = np.arange(resampling_i, resampling_i+timesteps_per_resample)
        save_idx_rev = save_idx[::-1]
        # tdiffs = sol_object.ts[0,] - all_ts[save_idx_rev]
        # assert(np.allclose(tdiffs, 0, atol=1e-3)), 'time array not like expected'

        # so we save it in the big solution array with reversed indices.
        all_sols[:, save_idx_rev, :, :] = sol_object.ys

        train_inputs, train_labels = array_juggling.sol_array_to_train_data(
                all_sols, all_ts, resampling_i, n_timesteps, algo_params
        )

        key, train_key = jax.random.split(key)

        # for this to work with jit, we need algo_params to be hashable
        # so we can pass it as static argument
        # for this in turn we cannot have jax arrays in there.
        # luckily the NN training does not need the arrays.
        # therefore, we take them out

        nn_start = time.time()
        nn_params, train_losses, test_losses = V_nn.train(
                train_inputs, train_labels, nn_params, algo_params_nn, train_key
        )
        nn_time += time.time() - nn_start


        if algo_params['nn_plot_training']:
            train_loss_curves.append(train_losses)
            test_loss_curves.append(test_losses)


        # resampling step.
        # which type of resampling (minimal or all) is baked into the resample function
        # via algo_params['resample_type'].
        key, subkey = jax.random.split(key)

        ys_resampled, resampling_mask = array_juggling.resample(
                final_ys, resampling_t,
                V_nn.nn.apply, nn_params, algo_params,
                subkey)

        where_resampled[:, resampling_i, None, None] = resampling_mask

        # finished (x, λ, v) vectors for the next integration step.
        init_ys = ys_resampled

    total_end = time.time()

    full_time = total_end - total_start

    frac_integration = integration_time / full_time
    frac_nn = nn_time / full_time
    rest_time = full_time - integration_time - nn_time
    print(f' --- timing summary ---')
    print(f' total:         {       full_time:.1f}s ')
    print(f' integration:   {integration_time:.1f}s = {int(100*frac_integration)}% ')
    print(f' NN training:   {         nn_time:.1f}s = {int(100*frac_nn         )}% ')
    print(f' rest:          {       rest_time:.1f}s = {int(100*(1-frac_integration-frac_nn))}% ')
    plotting_utils.plot_2d_V(V_nn, nn_params, (0, T), (-3, 3))

    if algo_params['nn_plot_training']:
        pl.figure()
        pl.semilogy(np.array(train_loss_curves).T, label='training loss', c='gray')
        pl.semilogy(np.array(test_loss_curves).T, label='test loss', c='blue')
        pl.legend()
        pl.show()


    return all_sols, all_ts, where_resampled




def characteristics_experiment_even_simpler():
    # 1d test case.

    # SINGLE integrator.
    def f(t, x, u):
        return u

    def l(t, x, u):
        Q = np.eye(1)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 1 * np.eye(1)
        return (x.T @ Qf @ x).reshape()

    problem_params = {
            'f': f,
            'l': l,
            'h': h,
            'T': 5,
            'nx': 1,
            'nu': 1
    }

    x_sample_scale = 1 * np.eye(1)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # resample when the mahalanobis distance (to the sampling distribution) is larger than this.
    resample_mahalanobis_dist = 2
    Q_x = np.linalg.inv(x_sample_cov) / resample_mahalanobis_dist**2

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.
    algo_params = {
            'n_trajectories': 256,
            'dt': 1/256,
            'resample_interval': 1/16,
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,
            'nn_layersizes': (128, 128, 128),
            'nn_batchsize': 128,
            'nn_N_epochs': 10,
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'
    all_sols, all_ts, where_resampled = hjb_characteristics_solver(problem_params, algo_params)

    plotting_utils.plot_1d(all_sols, all_ts, where_resampled, problem_params, algo_params)



def characteristics_experiment_simple():

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
        Qf = 0 * np.eye(2)
        return (x.T @ Qf @ x).reshape()


    problem_params = {
            'f': f,
            'l': l,
            'h': h,
            'T': 4,
            'nx': 2,
            'nu': 1
    }

    x_sample_scale = 1 * np.eye(2)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # resample when the mahalanobis distance (to the sampling distribution) is larger than this.
    resample_mahalanobis_dist = 2

    # to calculate mahalanobis dist: d = sqrt(x.T inv(Σ) x) -- basically the argument to exp(.) in the pdf.
    # x domain defined as X = {x in R^n: x.T @ Q_x @ x <= 1}
    # so we want to find Q_x to achieve
    #     sqrt(x.T inv(Σ) x) <= d_max  <=>  x.T @ Q_x @ x <= 1
    #                                  <=>  sqrt(x.T @ Q_x @ x) <= 1                       (bc. sqrt monotonous)
    #                                  <=>  sqrt(x.T @ Q_x @ x) d_max <= d_max             (scaling LHS changes nothing)
    #                                  <=>  sqrt(x.T @ Q_x @ x * d_max**2) <= d_max        (d_max into sqrt)
    #                                  <=>  sqrt(x.T @ (Q_x * d_max**2) @ x) <= d_max      (reordering)
    # these are now basically the same, with inv(Σ) = Q_x * d_max**2 or Q_x = inv(Σ) / d_max**2
    # why could I not simply guess this?
    Q_x = np.linalg.inv(x_sample_cov) / resample_mahalanobis_dist**2

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.

    algo_params = {
            'n_trajectories': 32,
            'dt': 1/64,
            'resample_interval': 1/4,
            'resample_type': 'minimal',
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,

            'nn_layersizes': (64, 64, 64),
            'nn_batchsize': 128,
            'nn_N_epochs': 5,
            'nn_testset_fraction': 0.1,
            'nn_plot_training': False,
            'nn_train_lookback': 1/4,
            'nn_V_gradient_penalty': 1.,
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'
    output = hjb_characteristics_solver(problem_params, algo_params)

    plotting_utils.plot_2d(*output, problem_params, algo_params)

    pl.show()


if __name__ == '__main__':

    # characteristics_experiment_even_simpler()
    characteristics_experiment_simple()
