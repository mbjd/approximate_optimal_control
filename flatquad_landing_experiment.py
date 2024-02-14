#!/usr/bin/env python

import jax
import jax.numpy as np
import diffrax

import rrt_sampler
import pontryagin_utils
import ddp_optimizer
import visualiser
import ct_basics

import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf

import ipdb
import time
import numpy as onp
import tqdm


def current_weird_experiment(problem_params, algo_params):
    
    # initial couple lines from rrt_sample

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'

    # get initial "easy" solution with LQR for state close to goal. 
    x0 = jax.random.normal(jax.random.PRNGKey(0), shape=(problem_params['nx'],)) * 0.1



    # forward sim w/ "optimal" controller based on quadratic LQR value function
    # but while respecting state constraints
    def forwardsim_rhs(t, state, args):

        x = state  

        lam_x = 2 * P_lqr @ x
        u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
        dot_x = problem_params['f'](t, x, u)

        return dot_x

    # code stolen from DDP forward pass. 
    term = diffrax.ODETerm(forwardsim_rhs)

    # same tolerance parameters used in pure unguided backward integration of whole PMP
    step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
    saveat = diffrax.SaveAt(steps=True, dense=True) 

    init_sol = diffrax.diffeqsolve(
        term, diffrax.Tsit5(), t0=0., t1=problem_params['T'], dt0=0.1, y0=x0,
        stepsize_controller=step_ctrl, saveat=saveat,
        max_steps = algo_params['pontryagin_solver_maxsteps'],
    )

    # this trajectory should be rather boring
    # visualiser.plot_trajectories_meshcat(init_sol)

    ddp_optimizer.ddp_main(problem_params, algo_params, init_sol)
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
        'T': 10,
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
            'pontryagin_solver_maxsteps': 512, # nice if it is not waaay too much
            'pontryagin_solver_dense': False,
    }

        
        

    current_weird_experiment(problem_params, algo_params)
    # all_sols = rrt_sampler.rrt_sample(problem_params, algo_params)

    # visualiser.plot_trajectories_meshcat(all_sols, colormap='viridis')

    # otherwise visualiser closes immediately
    ipdb.set_trace()

    


