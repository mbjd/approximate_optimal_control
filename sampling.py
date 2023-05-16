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

# no bullshit
# this will throw an error when trying to automatically promote rank.
from jax import config
import warnings
config.update("jax_numpy_rank_promotion", "warn")
warnings.filterwarnings('error', message='.*Following NumPy automatic rank promotion.*')


def pontryagin_sampler(problem_params, algo_params):

    '''
    we have the following problem. there is some distribution of states at
    t=0 for which we want to know the optimal controls. We can do this with
    the pontryagin principle. but we need to sample from a suitable
    distribution over terminal boundary conditions (state or costate) to
    get interesting data.

    This function does that by importance sampling. Very rough description:

    - We sample from some distribution of terminal BC's and simulate
      optimal trajectories according to the PMP, up to t=t1.
    - We assign a weight to each of the trajectories that quantifies 'how
      much we like them'. the technicalities of how this weight is set
      basically define the method: importance sampling, rejection sampling,
      etc.
    - We find a new distribution (gaussian) over terminal BC's that most
      closely matches the reweighted version of the previous sampling
      distribution.

    These steps are iterated, and gradually t1 is stepped from T towards 0,
    so each time the distribution only changes a bit.

    It seems to work well. But due to the fact that we always 'collapse'
    the distribution over terminal BCs back to a gaussian, and due to some
    other approximations, we have no guarantees that the desired state
    distribution at t=0 is actually reached.

    problem_params: same as usual
    algo_params: dict with algorithm tuning parameters. see code.
    '''

    key = jax.random.PRNGKey(0)

    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']

    # define the dynamics of optimal trajectories according to PMP.
    # TODO in here the input constraints are defined. move them to problem_params somehow?
    f_forward = pontryagin_utils.define_extended_dynamics(problem_params)

    # solve pontryagin backwards, for vampping later.
    # slightly differently parameterised than in other version.
    def pontryagin_backward_solver(y0, t0, t1):

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs

        # negative if t1 < t0, backward integration just works
        assert algo_params['dt'] > 0
        dt = algo_params['dt'] * np.sign(t1 - t0)

        # what if we accept that we could create NaNs?
        max_steps = int(1 + problem_params['T'] / algo_params['dt'])

        # maybe easier to control the timing intervals like this?
        saveat = diffrax.SaveAt(steps=True)

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt, y0=y0,
                saveat=saveat, max_steps=max_steps,
        )

        # this should return the last calculated (= non-inf) solution.
        return solution, solution.ys[solution.stats['num_accepted_steps']-1]

    # vmap = gangster!
    # vmap only across first argument.
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None, None)
    ))

    # construct the terminal extended state

    # helper function, expands the state vector x to the extended state vector y = [x, λ, v]
    # λ is the costate in the pontryagin minimum principle
    # h is the terminal value function
    def x_to_y_terminalcost(x, h):
        costate = jax.grad(h)(x)
        v = h(x)
        y = np.vstack([x, costate, v])

        return y

    # slight abuse here: the same argument is used for λ as for x, be aware
    # that x_to_y in that case is a misnomer, it should be λ_to_y.
    # but as with a terminal constraint we are really just searching a
    # distribution over λ instead of x, but the rest stays the same.
    def x_to_y_terminalconstraint(λ, h=None):
        x = np.zeros((nx, 1))
        costate = λ
        v = 0
        y = np.vstack([x, costate, v])
        return y

    if problem_params['terminal_constraint']:
        # we instead assume a zero terminal constraint.
        x_to_y = x_to_y_terminalconstraint
    else:
        x_to_y = x_to_y_terminalcost

    x_to_y_vmap = jax.vmap(lambda x: x_to_y(x, h))

    # this is the initial distribution. choosing the desired state distribution here
    # has proven to work well for the toy example. but if units are weird or something
    # we may have to find something smarter.
    key, subkey = jax.random.split(key)
    x_T = jax.random.multivariate_normal(
            subkey,
            mean=np.zeros(nx,),
            cov=algo_params['x_sample_cov'],
            shape=(algo_params['n_trajectories'],)
    ).reshape(algo_params['n_trajectories'], nx, 1)

    y_T = x_to_y_vmap(x_T)


    # the desired distribution of states at t=0.
    desired_mean = np.zeros((nx,))
    desired_cov = algo_params['x_sample_cov']
    desired_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, desired_mean[None, :], desired_cov)


    # main loop iteration, for jax.lax.scan.

    # basically, this does the following:
    # - sample terminal boundary conditions ~ N(sampling_mean, sampling_cov)
    # - simulate optimal trajectories backwards using PMP, from T to t1.
    # - weigh them according to how much we like them. here we have several schemes:
    #   - importance sampling
    #   - cheap importance sampling without compensation for sampling likelihood
    #   - rejection sampling
    # - find the gaussian distribution over terminal boundary conditions that leads to
    #   the re-weighted distribution at t1.

    # to make the problem easier we solve this iteratively, each time making the integration
    # interval a bit longer, so the distribution does not change too wildly.

    def scan_fct(carry, inp):

        # input = t1 = END of integration interval. t1 < t0 = T
        # carry = (sampling_mean, sampling_cov, key)
        # output = dict with various useful entries, see for yourself :)

        sampling_mean, sampling_cov, key = carry

        # sampling step first.
        # if deterministic, we choose the same sample each time.
        if algo_params['deterministic']:
            subkey = key
        else:
            key, subkey = jax.random.split(key)

        x_T = jax.random.multivariate_normal(
                subkey,
                mean=sampling_mean,
                cov=sampling_cov,
                shape=(algo_params['n_trajectories'],)
        ).reshape(algo_params['n_trajectories'], nx, 1)

        y_T = x_to_y_vmap(x_T)

        t1 = inp

        # find optimal trajectories by pontryagin principle.
        sol_object, last_sol_vec = batch_pontryagin_backward_solver(y_T, T, t1)

        # find the relevant likelihoods.
        # IMPORTANT: the sampling likelihood is wrong, to be correct we would have to multiply by
        # the inverse jacobian determinant of the mapping from terminal BCs to initial state.
        # we find that omittig this gives satisfactory results. because the weights are normalised anyway,
        # it does not matter too much.
        desired_likelihoods = desired_pdf(last_sol_vec[:, 0:nx, 0])
        sampling_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, sampling_mean[None, :], sampling_cov)
        sampling_likelihoods = sampling_pdf(x_T.reshape(algo_params['n_trajectories'], nx))

        # importance sampling weight - some different choices.
        resampling_weights_importance = desired_likelihoods / (1e-9 + sampling_likelihoods)

        # rejection sampling weights
        resampling_weights_rejection = desired_likelihoods >= 0.001

        # cheap in between weights
        # weight by likelihood at t1, but don't compensate for sampling distribution.
        resampling_weights_between = desired_likelihoods

        # if we are in the final iterations (t1=0), take the importance sampling weighting.
        # otherwise, 'between' ones seem to work better.
        is_final = np.allclose(t1, 0)
        resampling_weights_switched = is_final * resampling_weights_importance + (1-is_final) * resampling_weights_between

        if algo_params['sampling_strategy'] == 'importance':
            resampling_weights = resampling_weights_importance
        elif algo_params['sampling_strategy'] == 'rejection':
            resampling_weights = resampling_weights_rejection
        elif algo_params['sampling_strategy'] == 'between':
            resampling_weights = resampling_weights_between
        elif algo_params['sampling_strategy'] == 'switched':
            resampling_weights = resampling_weights_switched
        else:
            st = algo_params['sampling_strategy']
            raise ValueError(f'Invalid sampling strategy "{st}"')

        resampling_weights = resampling_weights / np.sum(resampling_weights)

        xs_flat = x_T.reshape(-1, nx)

        # calculate mean and covariance of the re-weighted terminal BC distribution.
        sampling_mean = np.mean(resampling_weights[:, None] * xs_flat, axis=0) * 0  # or just zero?

        # subtract mean
        xs_zm = xs_flat - sampling_mean[None, :]

        # for zero mean RV x: cov(x) = E[x x'] = Σ_x p(x) * x x' = Σ_x (x*sqrt(p(x)) * (x*sqrt(p(x))'
        # so basically to apply the 'regular' covariance formula, we need to scale the
        # data points by the square root of their weights.
        # but where did the n-1 go? bessel correction mean estimate what?
        sqrt_scaled_datapts = np.sqrt(1e-9 + resampling_weights[:, None]) * xs_flat

        # (wide matrix of column vecs) @ (tall matrix of row vecs)
        sampling_cov = sqrt_scaled_datapts.T @ sqrt_scaled_datapts

        # do the same but for the y_T vector - the covariance we get at t1.
        # just for output bookkeeping.
        last_sol_zm = last_sol_vec[:, 0:nx, 0]
        last_sol_zm = last_sol_zm - last_sol_zm.mean(axis=0)[None, :]  # (N_traj, nx) - (1, nx) promotes correctly
        last_sol_cov = last_sol_zm.T @ last_sol_zm / (algo_params['n_trajectories'] - 1)


        # output = (sol_object, sampling_cov, resampling_weights)
        # output as dict is way nicer. scan automatically 'transposes' it into
        # a dict of arrays, not an array of dicts :)
        output = {
                'solutions': sol_object,
                'sampling_means': sampling_mean,
                'sampling_covs': sampling_cov,
                't1_covs': last_sol_cov,
                'resampling_ws': resampling_weights,
        }

        new_carry = (sampling_mean, sampling_cov, key)

        return new_carry, output

    # the sequence of times to integrate. always from T to t1s[i].
    # at the end we iterate a couple of times at t1=0, just to be sure.
    t1s = np.concatenate([
        np.linspace(0, T, algo_params['n_resampling_iters'], endpoint=False)[::-1],
        np.zeros(algo_params['n_extrarounds']),
    ])

    N = t1s.shape[0]

    init_carry = (np.zeros(nx,), algo_params['x_sample_cov'], key)

    # do it twice so we see how fast it actually is...

    start = time.perf_counter()
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, t1s, length=N)
    end = time.perf_counter()
    print(f'time for jit/first run: {end-start}')
    start = time.perf_counter()
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, t1s, length=N)
    end = time.perf_counter()
    print(f'time for second run: {end-start}')

    sol_object = outputs['solutions']
    all_ys = sol_object.ys

    sampling_covs = outputs['sampling_covs']
    t1_covs = outputs['t1_covs']
    resampling_ws = outputs['resampling_ws']


    if algo_params['plot']:
        fig = pl.figure(figsize=(8, 3))

        dims = 2
        if dims==3:
            ax3 = fig.add_subplot(111, projection='3d')
        else:
            assert dims==2
            ax0 = fig.add_subplot(211)
            ax1 = fig.add_subplot(212)

        # basically the same but in a phase plot
        fig = pl.figure(figsize=(4, 4))
        ax_phase = fig.add_subplot(111)

        for i in range(N):
            print(f'plotting iteration {i}...')
            t_plot = sol_object.ts[i, 0]

            # the two state vectors...
            x_plot = sol_object.ys[i, :, :, 0, 0].T
            y_plot = sol_object.ys[i, :, :, 1, 0].T

            # color basically shows iteration number
            # alpha shows likelihood of trajectory for resampling?
            c = matplotlib.colormaps.get_cmap('gist_gray')(i/N)
            a = onp.array(resampling_ws[i] / np.max(resampling_ws[i]))

            if i==N-1:
                # visualise final distribution differently...
                c = 'red'
                a = onp.ones_like(a) * .25

            # colors_with_alpha = onp.zeros((algo_params['n_trajectories'], 4))
            # colors_with_alpha[:, 3] = a
            # colors_with_alpha[:, 0:3] = onp.array(c)[None, 0:3]


            # bc. apparently alpha cannot be a vector :(
            for j in range(x_plot.shape[1]):
                if dims==3:
                    ax3.plot(t_plot, x_plot[:, j], y_plot[:, j], color=c, alpha=a[j])
                else:
                    ax0.plot(t_plot, x_plot[:, j], color=c, alpha=a[j])
                    ax1.plot(t_plot, y_plot[:, j], color=c, alpha=a[j])
                ax_phase.plot(x_plot[:, j], y_plot[:, j], color=c, alpha=a[j]/5)
                ax_phase.scatter(x_plot[0, j], y_plot[0, j])


        pl.figure()
        pl.subplot(211)
        pl.plot(sampling_covs.reshape(N, -1), label='sampling cov. entries')
        pl.legend()
        pl.subplot(212)
        pl.plot(t1_covs.reshape(N, -1), label='t1 cov. entries')
        pl.legend()
        pl.show()


    # previously for benchmarking
    # return end-start

    last_sampling_mean = outputs['sampling_means'][-1]
    last_sampling_cov = outputs['sampling_covs'][-1]

    return (last_sampling_mean, last_sampling_cov)




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
            'n_trajectories': 128,
            'dt': 1/16,
            'x_sample_cov': x_sample_cov,
            'n_resampling_iters': 8,
            'sampling_strategy': 'switched',
            'n_extrarounds': 2,
            'deterministic': True,
            'plot': True,
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    pontryagin_sampler(problem_params, algo_params)

    # nts = 2**np.arange(4, 18)
    # ts = []
    # for nt in nts:
    #     print(f'trying n_trajectories = {nt}')
    #     algo_params['n_trajectories'] = nt
    #     ts.append(importance_sampling_bvp(problem_params, algo_params))

    # pl.plot(nts, np.array(ts))
    # pl.show()


