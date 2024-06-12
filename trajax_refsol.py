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

def refsol(sol0, problem_params, algo_params):

    # wrapper for trajax ilqr optimiser.
    # - get initial guess from continuous-time solution (as given by diffrax) (not yet)
    # - discretise cost&dynamics
    # - return only the objective.

    dt = 0.02
    N = 200

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    x_eq = problem_params['x_eq']
    V_f = lambda x: 0.5 * (x - x_eq).T @ P_lqr @ (x - x_eq)
    u_lqr_fct = lambda x: -K_lqr @ (x - problem_params['x_eq']) + problem_params['u_eq']


    # discrete time dynamics & cost function
    dynamics_cont = lambda x, u, t: problem_params['f'](x, u)

    def dynamics_disc(x, u, t):
        xn = trajax.integrators.rk4(dynamics_cont, dt=dt)(x, u, t)
        return problem_params['project_M'](xn)

    def cost_disc(x, u, t):
        # implementation like https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L495
        stage_cost = problem_params['l'](x, u) * dt
        terminal_cost = V_f(x)  # add input cost too???
        return np.where(t == N, terminal_cost, stage_cost)

    x0 = np.array([-1., 0., 1., 0., 0., 0., 0.])
    U0 = np.ones((N, problem_params['nu'])) * problem_params['u_eq'][0]
    u_lower, u_upper = problem_params['U_interval']

    # inspired by https://github.com/google/trajax/blob/main/tests/optimizers_test.py#L713
    def control_constraint(x, u, t):
        # functions that should be <= 0.
        return np.concatenate([u_lower - u, u - u_upper])

    # cannot decide if trajax is beautiful or utterly deranged
    X, U, dual_equality, dual_inequality, penalty, equality_constraints, inequality_constraints, max_constraint_violation, obj, gradient, iteration_ilqr, iteration_al = trajax.optimizers.constrained_ilqr(
        cost_disc, dynamics_disc,
        x0, U0,
        inequality_constraint=control_constraint
    )

    # basic unconstrained ilqr
    # X, U, obj, gradient, adjoints, lqr, iteration = trajax.optimizers.ilqr(
    #     cost_disc, dynamics_disc,
    #     x0, U0,
    # )

    plot = True
    if plot:
        pl.subplot(211)
        ts = np.arange(N+1) * dt
        pl.plot(ts, X, '.-', label=problem_params['state_names'])
        pl.subplot(212)
        ts = np.arange(N) * dt
        pl.plot(ts, U, '.-', label=('u1', 'u2'))
        pl.show()
        ipdb.set_trace()

    return obj
