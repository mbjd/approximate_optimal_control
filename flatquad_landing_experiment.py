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



def backward_with_hessian(problem_params, algo_params):

    # find terminal LQR controller and value function. 
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'

    key = jax.random.PRNGKey(0)

    # two unit norm points
    normal_pts = jax.random.normal(key, shape=(2, problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # interpolate between them
    alphas = np.linspace(0, 1, 200)[:, None]
    pts_interp = alphas * unitsphere_pts[0] + (1-alphas) * unitsphere_pts[1]

    # put back on unit sphere. 
    pt_norms = jax.vmap(np.linalg.norm)(pts_interp)[:, None]
    pts_interp = pts_interp / pt_norms

    # different strategy: purely random ass points. 
    normal_pts = jax.random.normal(key, shape=(200, problem_params['nx']))
    pts_interp = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]


    # map to boundary of Xf. 
    xfs = pts_interp @ np.linalg.inv(Phalf) * 0.01




    # test if it worked
    xf_to_Vf = lambda x: x @ P_lqr @ x.T
    vfs =  jax.vmap(xf_to_Vf)(xfs)

    # define RHS of the backward system, including 
    def f_extended(t, state, args=None):

        # state variables = x and quadratic taylor expansion of V.
        x       = state['x']
        v       = state['v']
        costate = state['vx']
        S       = state['vxx']

        H = lambda x, u, λ: l(t, x, u) + λ.T @ f(t, x, u)

        # RHS of the necessary conditions without the hessian. 
        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)
            nx = problem_params['nx']

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # its derivatives, for propagation of Vxx.
        fx, gx = jax.jacobian(pmp_rhs, argnums=0)(x, costate)
        flam, glam = jax.jacobian(pmp_rhs, argnums=1)(x, costate)

        # we calculate this here one extra time, could be optimised
        u_star = pontryagin_utils.u_star_2d(x, costate, problem_params)

        # calculate all the RHS terms
        v_dot = -problem_params['l'](t, x, u_star)
        x_dot, costate_dot = pmp_rhs(x, costate)
        S_dot = gx + glam @ S - S @ fx - S @ flam @ S 

        # and pack them in a nice dict for the state.
        state_dot = dict()
        state_dot['x'] = x_dot
        state_dot['v'] = v_dot
        state_dot['vx'] = costate_dot
        state_dot['vxx'] = S_dot

        return state_dot

    def solve_backward(x_f):

        # still unsure about these factors of 2 or 1/2 or 1
        v_f = x_f.T @ P_lqr @ x_f
        vx_f = 2 * P_lqr @ x_f
        vxx_f = P_lqr 

        state_f = {
            'x': x_f,
            'v': v_f,
            'vx': vx_f, 
            'vxx': vxx_f,
        }

        term = diffrax.ODETerm(f_extended)

        relax_factor = 1.
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
            dtmin = 0.005,  # just to avoid getting stuck completely
            dtmax = 0.5,
        )

        # try this, maybe it works better \o/
        # step_ctrl_fixed = diffrax.StepTo(prev_forward_sol.ts)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        T = 4.

        backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=-T, dt0=-0.1, y0=state_f,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
        )

        return backward_sol

    # backward_sol = solve_backward(xfs[0])
    sols = jax.vmap(solve_backward)(xfs)

    def plot_sol(sol):

        # adapted from plot_forward_backward in ddp_optimizer
        
        ts = np.linspace(sol.t0, sol.t1, 5001)

        # plot the state trajectory of the forward pass, interpolation & nodes. 
        ax1 = pl.subplot(311)

        pl.plot(sol.ts, sol.ys['x'], marker='.', linestyle='', alpha=1, label=problem_params['state_names'])
        interp_ys = jax.vmap(sol.evaluate)(ts)
        pl.gca().set_prop_cycle(None)
        pl.plot(ts, interp_ys['x'], alpha=0.5)
        pl.legend()


        pl.subplot(312, sharex=ax1)
        us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(
            sol.ys['x'], sol.ys['vx'], problem_params
        )
        def u_t(t):
            state_t = sol.evaluate(t)
            return pontryagin_utils.u_star_2d(state_t['x'], state_t['vx'], problem_params)

        us_interp = jax.vmap(u_t)(ts)

        pl.plot(sol.ts, us, linestyle='', marker='.')
        pl.gca().set_prop_cycle(None)
        pl.plot(ts, us_interp, label=('u_0', 'u_1'))
        pl.legend()


        # plot the eigenvalues of S from the backward pass.
        pl.subplot(313, sharex=ax1)

        # eigenvalues at nodes. 
        sorted_eigs = lambda S: np.sort(np.linalg.eig(S)[0].real)

        S_eigenvalues = jax.vmap(sorted_eigs)(sol.ys['vxx'])
        eigv_label = ['S(t) eigenvalues'] + [None] * (problem_params['nx']-1)
        pl.semilogy(sol.ts, S_eigenvalues, color='C0', marker='.', linestyle='', label=eigv_label)
        # also as line bc this line is more accurate than the "interpolated" one below if timesteps become very small
        pl.semilogy(sol.ts, S_eigenvalues, color='C0')  

        # eigenvalues interpolated. though this is kind of dumb seeing how the backward
        # solver very closely steps to the non-differentiable points. 
        sorted_eigs_interp = jax.vmap(sorted_eigs)(interp_ys['vxx'])
        pl.semilogy(ts, sorted_eigs_interp, color='C0', linestyle='--', alpha=.5)
        pl.legend()



    # plot_sol(backward_sol)
    # visualiser.plot_trajectories_meshcat(backward_sol)
    pl.rcParams['figure.figsize'] = (15, 8)

    '''
    for j, _ in tqdm.tqdm(enumerate(xfs)):
        plot_sol(jax.tree_util.tree_map(itemgetter(j), sols))
        pl.savefig(f'tmp/backward_{j:03d}.png')
        pl.close('all')
    '''

    # random backward trajectory + local feedback ?!?!??!??!!!?!?
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
        'T': 10.,  # problems come up if not float
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
            # 'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            # 'pontryagin_solver_adaptive': True,  always adaptivee
            'pontryagin_solver_atol': 1e-5,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 4096, # nice if it is not waaay too much
            # causes it not to quit when hitting maxsteps. probably still all subsequent 
            # results will be unusable due to evaluating solutions outside their domain giving NaN
            'throw': False,  
    }

        
        
    backward_with_hessian(problem_params, algo_params)
    ipdb.set_trace()

    # current_weird_experiment(problem_params, algo_params)
    # all_sols = rrt_sampler.rrt_sample(problem_params, algo_params)

    # visualiser.plot_trajectories_meshcat(all_sols, colormap='viridis')

    # otherwise visualiser closes immediately

    


