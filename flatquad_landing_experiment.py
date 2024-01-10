#!/usr/bin/env python

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import rrt_sampler
import pontryagin_utils
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
    # just experiment around a bit. 
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

    pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)

    vmap_pontryagin_solver = jax.jit(jax.vmap(pontryagin_solver, in_axes=(0, 0, None)))
    sols = vmap_pontryagin_solver(yTs, vTs, problem_params['V_max'])

    sol = pontryagin_solver(yTs[0], vTs[0], problem_params['V_max'])
    f = lambda y: y[0:6].T @ P_lqr @ y[0:6] - 20
    # TODO get this to work tomorrow. 
    val = ct_basics.find_zero_on_trajectory(sol, f, 2e-4, 400.)
    ipdb.set_trace()
    
    

    # not working yet 
    # # try to re-sample the trajectories that got highest, because right now we only
    # # get ones where the goal is approached from below. 
    # for i in range(10):
    #     # replace inf with -inf to not mess with maximisation. 
    #     last_ys_noinf = jax.lax.select(sols.ys == np.inf, -np.inf * np.ones_like(sols.ys), sols.ys)

    #     max_y_per_trajectory = np.max(last_ys_noinf[:, :, 1], axis=1)
    #     
    #     # transform to 0 mean and unit variance. 
    #     max_y_normalised = (max_y_per_trajectory - max_y_per_trajectory.mean()) / max_y_per_trajectory.std()

    #     # transform to resampling probabilities.
    #     ps = jax.nn.softmax(max_y_normalised)

    #     key = jax.random.PRNGKey(i)
    #     resampled_yfs = jax.random.choice(key, last_ys_noinf[:, 0, 0:6], p, axis=0)



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
        state_cost = x.T @ Q @ x + np.maximum(-posy, 0)**2  # state penalty? 

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
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'terminal_constraint': True,
        'V_max': 500,
        'u_eq': np.ones(2) * m * g / 2,
        'x_eq': np.zeros(6),
    }


    algo_params = {
            'sampling_N_trajectories': 2**8,
            'sampling_N_iters': 10,
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 128, # nice if it is not waaay too much
            'pontryagin_solver_dense': False,
    }

        
        

    current_weird_experiment(problem_params, algo_params)
    # all_sol_ys = rrt_sampler.rrt_sample(problem_params, algo_params)

    visualiser.plot_trajectories_meshcat(sols.ys)

    # otherwise visualiser closes immediately
    ipdb.set_trace()


