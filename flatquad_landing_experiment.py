#!/usr/bin/env python

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import rrt_sampler
import pontryagin_utils
import ddp_optimizer
import visualiser
import ct_basics

import ipdb
import time
import numpy as onp
import tqdm

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf


def current_weird_experiment(problem_params, algo_params):
    
    # initial couple lines from rrt_sample

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'


    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.

    # the key leads to an interesting failure case in evaluating the derivative of the interpolation.
    # basically, the input goes from saturated to non-saturated, introducing a nondifferentiable point.
    # because basically input = d/dt state (differential flatness you know) the time derivative of the
    # state should have a similar corner. It does if we evaluate rhs(solution(t)), but if we instead
    # evaluate d/dt solution(t) we get wiggling artefacts due to polynomial interpolation.
    key = jax.random.PRNGKey(767667633)

    key = jax.random.PRNGKey(int(time.time()*100000))

    normal_pts = jax.random.normal(key, shape=(algo_params['sampling_N_trajectories'], problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    '''
    # instead, replace w/ linear interpolation between first two points
    # followed by re-normalisation. 
    a = unitsphere_pts[0][None, :]
    b = unitsphere_pts[1][None, :]
    alphas = np.linspace(0, 1, algo_params['sampling_N_trajectories'])[:, None]

    unitsphere_pts = a * alphas + b * (1-alphas)
    norms = np.linalg.norm(unitsphere_pts, axis=1)
    unitsphere_pts = unitsphere_pts / norms[:, None]
    '''

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01  # this gives rise to terminal value lvel vT


    # test if it worked
    xf_to_Vf = lambda x: x @ P_lqr @ x.T
    vTs =  jax.vmap(xf_to_Vf)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    xf_to_lamf = lambda x: x @ P_lqr * 2
    lamTs = jax.vmap(xf_to_lamf)(xTs)

    yTs = np.hstack([xTs, lamTs, np.zeros((algo_params['sampling_N_trajectories'], 1))])

    # pontryagin_solver = jax.jit(pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params))

    # vmap_pontryagin_solver = jax.jit(jax.vmap(pontryagin_solver, in_axes=(0, 0, None)))
    # sols = vmap_pontryagin_solver(yTs, vTs, problem_params['V_max'])

    # # newest experiment: "split" trajectories and see how it goes. 
    # idx = 0  # just split off this one trajectory for now. 
    # sol = jax.tree_util.tree_map(lambda node: node[idx], sols)

    # # so we have our reference solution: sol. 
    # y0 = sol.evaluate(sol.t1) 
    # x0, lam0, t0 = np.split(y0, [problem_params['nx'], 2*problem_params['nx']])

    # to have correct time info, use non-reparam solver here
    pontryagin_solver_orig = jax.jit(pontryagin_utils.make_pontryagin_solver(problem_params, algo_params))
    sol_orig = pontryagin_solver_orig(yTs[0], 4., 0.) # [0] bc. not vmapped


    # small test: plot the solution trajectory and its time derivatives.
    # looks good at first glance, no discontinuities etc.
    # second time derivative looks pretty ugly though 

    # after a closer look we see though that anything discontinuous is not 
    # well represented at all. for example if the initial input is full torque 
    # because we rotate quickly, then the derivative of the omega state is obviously
    # constant. After that it *should* start decreasing. At the nondifferentiable point
    # however the interpolating polynomial's derivative oscillates about a bit and even 
    # overshoots the (previous) maximum.

    # this is making me think that indeed we will just have to evaluate the RHS again 
    # instead of relying on the derivative of the interpolation. Patrick also warns 
    # of this here: https://docs.kidger.site/diffrax/api/solution/

    # or do we still "cheat" our way around it and d/dt the solution? my thinking is this: 
    # we mostly don't care if the solution "wiggles" about slightly (which will alter the 
    # v and vxx information we are propagating, but alter them right back when wiggling back)
    # as long as it generally goes into the right direction. And the RHS derivatives we have to 
    # evaluate anyway are evaluated at u* according to lambda (which we are finding in
    # the backward pass) and x (which we have from the forward solution). the only reason we are
    # interested in the time derivative of the forward solution is so we can adjust v and v_x 
    # according to the difference in the two derivatives. 

    # maybe the second is actually even better because we actually follow exactly the interpolated 
    # solution so from that perspective it would be more "coherent" to also adjust the taylor 
    # expansion using that info. but then again, the wiggling of the interpolation is just an artefact
    # of the ODE solution method, we know that the actual trajectory follows the RHS precisely, and so
    # might guess that the ODE solution over longer time periods is also more accurately represented by
    # that RHS than the interpolation's time derivative

    # this has turned into a rambling mess. probably either approach will work okay-ish, and we already
    # have other numerical errors anyway. 

    if True:
        ts_eval = np.linspace(sol_orig.t0, sol_orig.t1, 2001)

        interp_sol = jax.vmap(sol_orig.evaluate)(ts_eval)

        # plot the interpolation as line, and the nodes as points. 
        pl.subplot(121)
        pl.plot(ts_eval, interp_sol[:, 0:6], label=problem_params['state_names'])
        pl.gca().set_prop_cycle(None)
        # basically pl.scatter but that one has trouble understanding arrays
        pl.plot(sol_orig.ts, sol_orig.ys[:, 0:6], marker='.', linestyle='')
        pl.xlabel('state trajectory (interpolated)')
        pl.legend()


        pl.subplot(122)
        # now the same but for the derivative of the solution. 

        # the difference between those looks like white noise about
        # 10^5 times smaller than the trajectories themselves.
        # so basically very minor numerical differences.
        # interp_ddt = jax.vmap(jax.jacobian(sol_orig.evaluate))(ts_eval)
        interp_ddt = jax.vmap(sol_orig.derivative)(ts_eval)

        # higher derivative just for fun
        # interp_ddt = jax.vmap(jax.jacobian(jax.jacobian(jax.jacobian(sol_orig.evaluate))))(ts_eval)

        labels = ['d/dt ' + name for name in problem_params['state_names']]
        pl.plot(ts_eval, interp_ddt[:, 0:6], label=labels, linestyle='--', alpha=.5)
        pl.xlabel('d/dt [state trajectory (interpolated)]')


        pl.xlabel('ODE RHS at interpolated state trajectory')

        pl.gca().set_prop_cycle(None)
        f = pontryagin_utils.define_extended_dynamics(problem_params)
        interp_rhs = jax.vmap(f, in_axes=(None, 0))(0., interp_sol)
        labels = ['rhs(' + name + ')' for name in problem_params['state_names']]
        pl.plot(ts_eval, interp_rhs[:, 0:6], label=labels)


        pl.legend()
        pl.show()
        # ipdb.set_trace()

    ddp_optimizer.ddp_main(problem_params, algo_params, sol_orig)
    ipdb.set_trace()

    
    
    



if __name__ == '__main__':

    # classic 2D quad type thing. 6D state.

    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

    # remove time arguments sometime now that we're mostly treating 
    # infinite horizon, time-invariant problems? 
    def f(t, x, u):

        # unpack for easier names
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        xdot = np.array([
            vx,
            vy,
            omega,
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr-Fl) * r / I,
        ])

        return xdot

    def l(t, x, u):
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        state_length_scales = np.array([1, 1, np.deg2rad(10), 1, 1, np.deg2rad(45)])
        Q = np.diag(1/state_length_scales**2)
        state_cost = x.T @ Q @ x  

        # can we just set an input penalty that is zero at hover?
        # penalise x acc, y acc, angular acc here.
        # this here is basically a state-dependent linear map of the inputs, i.e. M(x) u with M(x) a 3x2 matrix.
        # the overall input cost will be acc.T M(x).T Q M(x) acc, so for each state it is still a nice quadratic in u.
        accelerations = np.array([
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr - Fl) * r / I,
        ])

        accelerations_lengthscale = np.array([1, 1, 1])

        input_cost = accelerations.T @ np.diag(1/accelerations_lengthscale**2) @ accelerations

        return state_cost + input_cost


    def h(x):
        # irrelevant if terminal constraint or infinite horizon
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    problem_params = {
        'system_name': 'flatquad',
        'f': f,
        'l': l,
        'h': h,
        'T': 2,
        'nx': 6,
        'state_names': ("x", "y", "Phi", "vx", "vy", "omega"),
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'terminal_constraint': True,
        'V_max': 1000.,
        'u_eq': np.ones(2) * m * g / 2,
        'x_eq': np.zeros(6),
    }


    algo_params = {
            'sampling_N_trajectories': 1,
            'sampling_N_iters': 10,
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-5,
            'pontryagin_solver_rtol': 1e-5,
            'pontryagin_solver_maxsteps': 4192, # nice if it is not waaay too much
            'pontryagin_solver_dense': False,
    }

        
        

    current_weird_experiment(problem_params, algo_params)
    # all_sols = rrt_sampler.rrt_sample(problem_params, algo_params)

    # visualiser.plot_trajectories_meshcat(all_sols, colormap='viridis')

    # otherwise visualiser closes immediately
    ipdb.set_trace()

    


