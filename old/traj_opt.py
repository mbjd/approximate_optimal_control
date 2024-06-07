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


def trajopt(problem_params):

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    x_eq = problem_params['x_eq']
    V_f = lambda x: 0.5 * (x - x_eq).T @ P_lqr @ (x - x_eq)
    u_lqr_fct = lambda x: -K_lqr @ (x - problem_params['x_eq']) + problem_params['u_eq']

    # direct single shooting trajectory optimisaiton. only shitty version
    # that hopefully works locally.

    # time discretisation.
    T = 5
    N = 30

    # fixed t grid for everything. what a welcome change
    tgrid = np.linspace(0, T, N)
    # fancy uneven t grid
    # tgrid = np.concatenate([np.linspace(0, T/3, N//2)[:-1], np.linspace(T/3, T, N//2)])
    # N = tgrid.shape[0]

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

        us, use_lqr = args
        u_interp = interp_u(us, t)
        u_lqr = u_lqr_fct(x)

        u = jax.lax.select(use_lqr, u_lqr, u_interp)

        # u_clipped = np.clip(u, umin, umax)

        return {
                'x': problem_params['f'](x, u),
                'cost': problem_params['l'](x, u),
        }

    umin = problem_params['U_interval'][0][0]
    umax = problem_params['U_interval'][1][0]

    def sim(us, dense=False, lqr=False):


        term = diffrax.ODETerm(forwardsim_rhs)

        # projection only for state, not cost ofc
        # if not manifold, project is just lambda x: x and we are fine :)
        project = lambda y: {'x': problem_params['project_M'](y['x']), 'cost': y['cost']}
        solver = pontryagin_utils.ProjectionSolver(project=project)
        adjoint = diffrax.DirectAdjoint

        ctrl = diffrax.StepTo(tgrid)

        y0 = {
                # TODO make this x0 arg
                'x': np.array([-1., -0, 0, 1, 0, 0, 0.]),
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
            args=(us, lqr),  # <- here the input enters!
            stepsize_controller=ctrl
        )

        reg_cost = np.sum(np.square(np.diff(us, axis=1))) * 1.
        # reg_cost = 0
        if not dense:
            x_f = sol.ys['x'][0]  # only one y
            terminal_cost = V_f(x_f)
            return (sol.ys['cost'].squeeze() + terminal_cost + reg_cost)
        else:
            return sol


    # get initial sol with lqr.
    sol_init = sim(us, dense=True, lqr=True)
    us_lqr = jax.vmap(u_lqr_fct)(sol_init.ys['x'])[0:N].T
    us = us_lqr

    pl.plot(sol_init.ts, sol_init.ys['x'], label=problem_params['state_names'])
    pl.plot(sol_init.ts, sol_init.ys['cost'], label='cost')
    pl.legend()
    pl.show()
    # ipdb.set_trace()
    # sol = sim(us, dense=True)
    cost = sim(us, dense=False)

    # mess up the us
    # us = us.at[1, 10].set(100.)
    # us = us + jax.random.normal(jax.random.PRNGKey(0), shape=us.shape) * 0.001
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

    opt = 'jaxopt'



    if opt == 'own':
        # own thing?????????????????????
        # def step(us):
        pass

    if opt == 'jaxopt':
        import jaxopt

        bounds = (np.ones_like(us) * umin, np.ones_like(us) * umax)
        params0 = us

        fun = jax.jit(jax.value_and_grad(sim))
        N_steps = 20000

        opti = jaxopt.LBFGSB(
                fun=fun, value_and_grad=True,
                history_size=20,
                stepsize = lambda i: 1e-4,
                # linesearch_init='current',
                # max_stepsize=1e-3,
                # min_stepsize=1e-6,
                # maxiter=N_steps,
                # tol=0.0001,
                )

        state = opti.init_state(params0, bounds)
        params = params0

        update = jax.jit(opti.update)
        costs = []
        for j in tqdm.tqdm(range(N_steps)):
            (params, state) = update(params, state, bounds)
            costs.append(cost_fn(params))
        pl.plot(costs)

        # ipdb.set_trace()

        # params_opt, state = opti.run(params0, bounds)
        us_opt = params

    elif opt == 'scipy':
        # original scipy...?
        import scipy
        oup = scipy.optimize.minimize(
                jax.jit(scipy_cost),
                jac=jax.jit(jax.grad(scipy_cost)),
                x0=us_scipy,
                bounds=bounds,
                method='L-BFGS-B'
        )

        us_opt = oup.x.reshape(us.shape)

    elif opt == 'adam':
        # standard adam?!?
        lr_init = .01
        lr_final = 0.0001
        N_steps = 100000

        lr_schedule = optax.exponential_decay(
                init_value = lr_init,
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

        us_opt = params

    pl.figure()
    pl.subplot(121)
    pl.plot(tgrid, us.T, c='C0', label=('us initial', None))
    pl.plot(tgrid, us_opt.T, '.-', c='C1', label=('us optimised', None))
    pl.legend()

    sol = sim(us_opt, dense=True)
    pl.subplot(122)
    pl.plot(sol.ts, sol.ys['x'], label=problem_params['state_names'])
    pl.plot(sol.ts, sol.ys['cost'], label='cost')
    pl.legend()

    pl.show()

    ipdb.set_trace()






    # u = argmin_u obj(u).


if __name__ == '__main__':

    from flatquad_landing_experiment import define_problem_params
    problem_params = define_problem_params()


    trajopt(problem_params)
