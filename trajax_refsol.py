import functools
import gzip
import os
import pprint
import subprocess
import sys
import time
from operator import itemgetter

import diffrax
import equinox
import flax
import ipdb
import jax
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as onp
import tqdm

import nn_utils
import plotting_utils
import pontryagin_utils
import visualiser
import wandb
from misc import *

import trajax


def refsol_homotopy(xs, sol0, v_nn, nn_params, problem_params, algo_params, dt=0.05, N=100):

    # both of the other in one so we can loop efficiently with jax.lax.scan

    assert dt*N <= sol0.t1 - sol0.t0, 'solution too short to initialise'

    # 1. get initial guess U0 {{{

    def v_mean(x, vmap_params):

        # find (empirical) mean and std. dev of value function.
        vs_ensemble = jax.vmap(v_nn, in_axes=(0, None))(vmap_params, x)
        return vs_ensemble.mean()

    vx_mean = jax.jacobian(v_mean, argnums=0)

    ts = np.arange(N) * dt
    sol_ys = jax.vmap(sol0.evaluate)(ts)
    sol_xs = sol_ys['x']
    sol_vxs = jax.vmap(vx_mean, in_axes=(0, None))(sol_xs, nn_params)
    # def u_star_general(x, costate, problem_params):

    U0 = jax.vmap(pontryagin_utils.u_star_general, in_axes=(0, 0, None))(
        sol_xs, sol_vxs, problem_params
    )

    x0 = sol_xs[0]
    # }}}

    # convert problem description {{{

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    x_eq = problem_params['x_eq']
    V_f = lambda x: 0.5 * (x - x_eq).T @ P_lqr @ (x - x_eq)
    u_lqr_fct = lambda x: -K_lqr @ (x - problem_params['x_eq']) + problem_params['u_eq']


    # discrete time dynamics & cost function
    dynamics_cont = lambda x, u, t: problem_params['f'](x, u)

    def dynamics_disc(x, u, k):
        xn = trajax.integrators.rk4(dynamics_cont, dt=dt)(x, u, k)
        return problem_params['project_M'](xn)

    def cost_disc(x, u, k):
        # implementation like https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L495
        stage_cost = problem_params['l'](x, u) * dt
        terminal_cost = V_f(x)  # add input cost too???
        return np.where(k == N, terminal_cost, stage_cost)

    # x0 = np.array([-1., 0., 0, 1., 5., 0, 0.])
    # U0 = np.ones((N, problem_params['nu'])) * problem_params['u_eq'][0]
    u_lower, u_upper = problem_params['U_interval']

    # inspired by https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L713
    def control_constraint(x, u, k):
        # functions that should be <= 0.
        control_limits = np.concatenate([u_lower - u, u - u_upper])
        return np.where(k == N, 0 * control_limits, control_limits)

    # }}}

    # define computation w scan {{{

    def body(U, x0):

        X, U, dual_equality, dual_inequality, penalty, equality_constraints, inequality_constraints, max_constraint_violation, obj, gradient, iteration_ilqr, iteration_al = trajax.optimizers.constrained_ilqr(
            cost_disc, dynamics_disc,
            x0, U,
            inequality_constraint=control_constraint
        )

        return U, (X, obj)

    # }}}

    # do it :)
    last_U, (Xs, objs) = jax.lax.scan(body, U0, xs)
    ipdb.set_trace()
    return objs, Xs


def refsol(sol0, v_nn, nn_params, problem_params, algo_params, dt=0.05, N=100, plot=False):

    # wrapper for trajax ilqr optimiser.
    # - get initial guess from continuous-time solution (as given by diffrax) (not yet)
    # - call other, lower level wrapper.

    assert dt*N <= sol0.t1 - sol0.t0, 'solution too short to initialise'

    # 1. get initial guess U {{{

    def v_mean(x, vmap_params):

        # find (empirical) mean and std. dev of value function.
        vs_ensemble = jax.vmap(v_nn, in_axes=(0, None))(vmap_params, x)
        return vs_ensemble.mean()

    vx_mean = jax.jacobian(v_mean, argnums=0)

    ts = np.arange(N) * dt
    sol_ys = jax.vmap(sol0.evaluate)(ts)
    sol_xs = sol_ys['x']
    sol_vxs = jax.vmap(vx_mean, in_axes=(0, None))(sol_xs, nn_params)
    # def u_star_general(x, costate, problem_params):

    sol_us = jax.vmap(pontryagin_utils.u_star_general, in_axes=(0, 0, None))(
        sol_xs, sol_vxs, problem_params
    )

    x0 = sol_xs[0]

    return refsol_from_us(x0, sol_us, problem_params, algo_params, dt=dt, N=N, plot=plot)


def refsol_from_us(x0, U0, problem_params, algo_params, dt=0.05, N=100, plot=False):

    # - discretise cost&dynamics
    # - solve problem with constrained_ilqr
    # - return only the objective.

    # U0 an array of initial inputs, (N_t, nu)


    # define & solve (discrete time) problem {{{
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    x_eq = problem_params['x_eq']
    V_f = lambda x: 0.5 * (x - x_eq).T @ P_lqr @ (x - x_eq)
    u_lqr_fct = lambda x: -K_lqr @ (x - problem_params['x_eq']) + problem_params['u_eq']


    # discrete time dynamics & cost function
    dynamics_cont = lambda x, u, t: problem_params['f'](x, u)

    def dynamics_disc(x, u, k):
        xn = trajax.integrators.rk4(dynamics_cont, dt=dt)(x, u, k)
        return problem_params['project_M'](xn)

    def cost_disc(x, u, k):
        # implementation like https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L495
        stage_cost = problem_params['l'](x, u) * dt
        terminal_cost = V_f(x)  # add input cost too???
        return np.where(k == N, terminal_cost, stage_cost)

    # x0 = np.array([-1., 0., 0, 1., 5., 0, 0.])
    # U0 = np.ones((N, problem_params['nu'])) * problem_params['u_eq'][0]
    u_lower, u_upper = problem_params['U_interval']

    # inspired by https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L713
    def control_constraint(x, u, k):
        # functions that should be <= 0.
        control_limits = np.concatenate([u_lower - u, u - u_upper])
        return np.where(k == N, 0 * control_limits, control_limits)

    # cannot decide if trajax is beautiful or utterly deranged
    X, U, dual_equality, dual_inequality, penalty, equality_constraints, inequality_constraints, max_constraint_violation, obj, gradient, iteration_ilqr, iteration_al = trajax.optimizers.constrained_ilqr(
        cost_disc, dynamics_disc,
        x0, U0,
        inequality_constraint=control_constraint
    )
    # }}}

    # basic unconstrained ilqr
    # X, U, obj, gradient, adjoints, lqr, iteration = trajax.optimizers.ilqr(
    #     cost_disc, dynamics_disc,
    #     x0, U0,
    # )

    if plot:
        pl.subplot(211)
        ts = np.arange(N+1) * dt
        pl.plot(ts, X, '.-', label=problem_params['state_names'])
        pl.subplot(212)
        ts = np.arange(N) * dt
        pl.plot(ts, U, '.-', label=('u1', 'u2'))

        ys = jax.vmap(lambda x: np.concatenate([x[0:2], np.array([np.arctan2(x[2], x[3])]), x[4:]]))(X)
        visualiser.plot_trajectories_meshcat({'t': ts, 'x': ys})

        pl.show()

        # ipdb.set_trace()

    return obj, U
