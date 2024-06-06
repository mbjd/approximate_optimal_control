#!/usr/bin/env python
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
import optax
import tqdm

import nn_utils
import plotting_utils
import pontryagin_utils
import visualiser
import wandb
from misc import *


def trajopt(V_f, problem_params):

    # direct single shooting trajectory optimisaiton. only shitty version
    # that hopefully works locally.

    # time discretisation.
    T = 3
    N = 30

    # fixed t grid for everything. what a welcome change
    tgrid = np.linspace(0, T, N)

    def interp_u(us, t):
        # np.interp only wants single output functions
        u = jax.vmap(np.interp, in_axes=(None, None, 0))(t, tgrid, us)
        return u

    us = np.ones((2, N)) * problem_params['u_eq'][0]  # hacky
    interp_u(us, 0.2)

    # RHS for forward simulation.
    def forwardsim_rhs(t, y, args):

        x = y['x']
        cost = y['cost']

        us = args
        u = interp_u(us, t)

        u_clipped = np.clip(u, umin, umax)

        return {
                'x': problem_params['f'](x, u_clipped),
                'cost': problem_params['l'](x, u),
        }

    umin = problem_params['U_interval'][0][0]
    umax = problem_params['U_interval'][1][0]

    def sim(us, dense=False):


        term = diffrax.ODETerm(forwardsim_rhs)

        # projection only for state, not cost ofc
        # if not manifold, project is just lambda x: x and we are fine :)
        project = lambda y: {'x': problem_params['project_M'](y['x']), 'cost': y['cost']}
        solver = pontryagin_utils.ProjectionSolver(project=project)
        adjoint = diffrax.DirectAdjoint

        ctrl = diffrax.StepTo(tgrid)

        y0 = {
                # TODO make this x0 arg
                'x': np.array([-1., -1, 0, 1, 0, 0, 0.]),
                'cost': 0.
        }


        # forwardsim_rhs(0.1, y0, us)

        if not dense:
            # calculate barebones sol, save only final state, return cost
            saveat = diffrax.SaveAt(t1=True)
        else:
            # calculate sol with interpolation
            saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        sol = diffrax.diffeqsolve(
            term, solver, t0=0., t1=T,
            y0=y0,
            max_steps=N+1,
            dt0=None,
            saveat=saveat,
            args=us,  # <- here the input enters!
            stepsize_controller=ctrl
        )

        reg_cost = np.sum(np.square(np.diff(us, axis=1))) * 1.
        # reg_cost = 0
        if not dense:
            x_f = sol.ys['x'][0]  # only one y
            terminal_cost = V_f(x_f)
            return sol.ys['cost'].squeeze() + terminal_cost + reg_cost
        else:
            return sol


    # sol = sim(us, dense=True)
    cost = sim(us, dense=False)

    # mess up the us
    # us = us.at[1, 10].set(100.)
    us = us + jax.random.normal(jax.random.PRNGKey(0), shape=us.shape) * 0.001
    # us = np.array([1.01, 0.99]) @ us

    # define obj & grad
    cost_fn = jax.jit(sim)
    grad_fn = jax.jit(jax.grad(sim))

    # scipy bfgs?
    scipy_cost = lambda u_flat: cost_fn(u_flat.reshape(us.shape))
    from jax.scipy import optimize

    us_scipy = us.flatten()

    umin = problem_params['U_interval'][0][0]
    umax = problem_params['U_interval'][1][0]
    bounds = np.column_stack([np.ones_like(us_scipy) * umin, np.ones_like(us_scipy) * umax])

    # own thing?????????????????????
    # def step(us):

    # original scipy...?
    # import scipy
    # oup = scipy.optimize.minimize(
    #         jax.jit(scipy_cost),
    #         jac=jax.jit(jax.grad(scipy_cost)),
    #         x0=us_scipy,
    #         bounds=bounds,
    #         method='L-BFGS-B'
    # )
    # ipdb.set_trace()


    # standard adam?!?
    lr_init = .1
    lr_final = 0.001
    N_steps = 10000

    lr_schedule = optax.exponential_decay(
            init_value = 0.001,
            transition_steps = N_steps,
            decay_rate = lr_final / lr_init,
            end_value=lr_final,
            staircase=False
    )

    opti = optax.adam(learning_rate=lr_schedule, nesterov=True,
            b1=.98, b2=.98)

    params = us
    opt_state = opti.init(params)


    costs = []
    for j in range(N_steps):
        grad = grad_fn(params)
        updates, opt_state = opti.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        cost = cost_fn(params)
        print(cost)
        costs.append(cost)
    pl.semilogy(costs)

    pl.figure()
    pl.plot(us.T, c='C0')
    pl.plot(params.T, c='C1')
    pl.show()

    sol = sim(params, dense=True)
    ipdb.set_trace()







    # u = argmin_u obj(u).


if __name__ == '__main__':

    from flatquad_landing_experiment import define_problem_params
    problem_params = define_problem_params()

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    x_eq = problem_params['x_eq']
    V_f = lambda x: 0.5 * (x - x_eq).T @ P_lqr @ (x - x_eq)

    trajopt(V_f, problem_params)
