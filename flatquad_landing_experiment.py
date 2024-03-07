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
from operator import itemgetter


def sqrtm_vs_cholesky():

    thetas = np.linspace(0, 2*np.pi, 100) 
    xs = np.vstack([np.cos(thetas), np.sin(thetas)])

    pl.plot(xs[0], xs[1], 'o-', label='circle')

    ellipse_P_sqrt = jax.random.normal(jax.random.PRNGKey(0), shape=(2, 2))
    ellipse_P = ellipse_P_sqrt @ ellipse_P_sqrt.T

    # ellipse is now x: x.T @ ellipse_P @ x <= 1.

    P_eigv, P_eigvec = np.linalg.eigh(ellipse_P)

    Phalf_sqrtm = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T

    Phalf_cholesky = np.linalg.cholesky(ellipse_P)


    # the .T is needed because it is the other way around as 
    # in the actual code, where we have xs.T @ P_sqrtm, where
    # xs.T is a tall matrix of state space points in row vector form.
    ellipse_orig = np.linalg.inv(ellipse_P_sqrt.T) @ xs
    ellipse_sqrtm = np.linalg.inv(Phalf_sqrtm.T) @ xs
    ellipse_cholesky = np.linalg.inv(Phalf_cholesky.T) @ xs

    # the three ellipses are definitely the same so we are in luck. 
    # also the distribution of points looks to be the same for all three. 
    # so it doesn't really matter which one we use -- in that case 
    # cholesky is probably the nicest. 

    pl.plot(ellipse_orig[0], ellipse_orig[1], 'o-', label='ellipse orig', alpha=.3)
    pl.plot(ellipse_sqrtm[0], ellipse_sqrtm[1], 'o-', label='ellipse sqrtm', alpha=.3)
    pl.plot(ellipse_cholesky[0], ellipse_cholesky[1], 'o-', label='ellipse cholesky', alpha=.3)
    pl.legend()
    pl.show()




def u_star_debugging(problem_params, algo_params):

    # found this failure case randomly during ddp testing. it does appear to 
    # show up right before things go south. not sure if it is the culprit. 

    # this is pretty ugly, and i'm not exactly sure why it happens. in general
    # there seems to be some numerical noise on the u_star, although generally 
    # only like 1e-6 or 1e-5

    # alright, fixed it by a cheap hack. reason is that we are comparing 
    # objective values for different candidate solutions. if the candidates are 
    # close with small distance d, then the objective differences are O(d^2). 
    # this causes float inaccuracies to be significant. 

    # fixed it by re-evaluating only the difference to the optimum in a second 
    # round of comparison. the numerical "chatter" is still here but only in a 
    # much smaller region close to the active set changes. 

    # in a future "proper" implementation we should just take the KKT conditions
    # which IIRC are all linear-ish instead of quadratic. 

    x = np.array([4.538097,6.19019,-0.10679064,-2.4105127,-3.6202462,0.04513513])
    lam = np.array([17.425396  , 22.878305  ,  0.73656803, -2.9136074 , -6.116059  ,0.28396177])

    # for pdb
    print(pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=True))

    def ustar_fct(x, lam):
        return pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=True)

    def ustar_fct_nonsmooth(x, lam):
        return pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=False)

    ustar = pontryagin_utils.u_star_2d(x, lam, problem_params)
    ustar_vmap = jax.vmap(ustar_fct, in_axes=(0, 0))
    ustar_vmap_nonsmooth = jax.vmap(ustar_fct_nonsmooth, in_axes=(0, 0))

    # go in random directions a bit and plot. 
    k = jax.random.PRNGKey(2)
    direction = jax.random.normal(k, shape=(12,))
    direction = direction / np.linalg.norm(direction)
    xdir, lamdir = np.split(direction, [6])
    xdir = 0 * xdir  # just to check if piecewise linear...

    alphas = np.linspace(0, .3, 10001)[:, None]

    xs = x + alphas * xdir
    lams = lam + alphas * lamdir


    def ustar_fct(x, lam): 
        return pontryagin_utils.u_star_2d(x, lam, problem_params, debug_oups=True)

    ustar_vmap = jax.vmap(ustar_fct, in_axes=(0, 0))

    (ustars, debug_oups), (grads, debug_oups_grads) = jax.vmap(lambda x, lam: jax.jvp(ustar_fct, (x, lam), (lam, lamdir)))(xs, lams)

    pl.subplot(211)
    pl.plot(ustars, label='u*')
    pl.legend()
    pl.subplot(212)
    pl.plot(grads, label='(du*/dx).T (x direction of sweep)')
    pl.legend()

    # ipdb.set_trace()
    # this works only for the modified function with debug outputs!
    # ustars_nonsmooth, oups = ustar_vmap_nonsmooth(xs, lams)


    # pl.plot(oups['all_Hs_adjusted'] - oups['all_Hs_adjusted'].min(axis=1)[:, None], alpha=.5, label=('H unconstrained - H*', 'H1 - H*', 'H2 - H*', 'H3 - H*', 'H4 - H*'))
    # pl.gca().set_prop_cycle(None)
    # pl.plot(oups['all_Hs_adjusted_new'] - oups['all_Hs_adjusted_new'].min(axis=1)[:, None], alpha=.5, linestyle='--', label=('Ht unconstrained - Ht*', 'Ht1 - Ht*', 'Ht2 - Ht*', 'Ht3 - Ht*', 'Ht4 - Ht*'))
    # pl.legend(); pl.show()
    # ipdb.set_trace()


    # plot the curve traced by u*
    pl.figure()
    pl.plot(ustars[:, 0], ustars[:, 1], color='red', label='u*')

    cands = debug_oups['all_candidates']

    for j in range(cands.shape[1]):
        pl.plot(cands[:, j, 0], cands[:, j, 1], alpha=.3)

    lowerbounds = problem_params['U_interval'][0]
    upperbounds = problem_params['U_interval'][1]
    pl.plot([lowerbounds[0], lowerbounds[0], upperbounds[0], upperbounds[0], lowerbounds[0]],
            [lowerbounds[1], upperbounds[1], upperbounds[1], lowerbounds[1], lowerbounds[1]], 
            color='black', label='constraints', alpha=0.2)
    

    pl.show()

    ipdb.set_trace()




def current_weird_experiment(problem_params, algo_params):

    # get initial "easy" solution with LQR for state close to goal. 
    x0 = jax.random.normal(jax.random.PRNGKey(0), shape=(problem_params['nx'],)) * 0.1
    print(x0)

    # this trajectory should be rather boring
    # visualiser.plot_trajectories_meshcat(init_sol)

    ddp_optimizer.ddp_main(problem_params, algo_params, x0)
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
    def f(x, u):

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

    def l(x, u):
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        # state_length_scales = np.array([1, 1, np.deg2rad(10), 1, 1, np.deg2rad(45)])
        state_length_scales = np.array([1, 1, np.deg2rad(30), .5, .5, np.deg2rad(120)])
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
        # OR we could right in here put the terminal quadratic cost
        # plus some exception or +infinity cost if outside terminal set...
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    problem_params = {
        'system_name': 'flatquad',
        'f': f,
        'l': l,
        'h': h,
        'T': 10.,  # problems come up if not float
        'nx': 6,
        'state_names': ("x", "y", "Phi", "vx", "vy", "omega"),
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'V_f': 0.001,
        'V_max': 1000.,
        'u_eq': np.ones(2) * m * g / 2,
        'x_eq': np.zeros(6),
    }


    algo_params = {
            # 'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            # 'pontryagin_solver_adaptive': True,  always adaptivee
            'pontryagin_solver_atol': 1e-5,
            'pontryagin_solver_rtol': 1e-5,
            'pontryagin_solver_maxsteps': 4096, # nice if it is not waaay too much
            # causes it not to quit when hitting maxsteps. probably still all subsequent 
            # results will be unusable due to evaluating solutions outside their domain giving NaN
            'throw': False,  
    }

        
        
    import levelsets
    levelsets.main(problem_params, algo_params)

    # backward_with_hessian(problem_params, algo_params)
    # current_weird_experiment(problem_params, algo_params)
    # u_star_debugging(problem_params, algo_params)
    ipdb.set_trace()

    # all_sols = rrt_sampler.rrt_sample(problem_params, algo_params)

    # visualiser.plot_trajectories_meshcat(all_sols, colormap='viridis')

    # otherwise visualiser closes immediately

    


