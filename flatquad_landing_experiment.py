#!/usr/bin/env python

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import main
import pontryagin_utils

import ipdb

if __name__ == '__main__':

    # classic 2D quad type thing. 6D state.
    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

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
        # irrelevant if terminal constraint
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    state_length_scales = np.array([2, 1, 2, 1, np.deg2rad(45), np.deg2rad(45)])
    x_sample_cov = np.diag(state_length_scales**2)

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
        'V_max': 2,
    }


    algo_params = {
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 1024,
            'pontryagin_solver_dense': False,
    }

    # similar to orbits example: find local lqr value function


    # important to linearise about equilibrium, not zero!
    # that linearisation would be uncontrollable in this case.
    # equilibrium in y direction:
    #   0 = np.cos(Phi) * (Fl + Fr) / m - g,
    #   0 =               (Fl + Fr) / m - g,
    #   Fl + Fr = mg
    # so, to also have no torque, we need: Fl = Fr = mg/2

    u_eq = np.ones(problem_params['nu']) * m * g / 2

    x_eq = np.zeros(problem_params['nx'])
    A = jax.jacobian(f, argnums=1)(0., x_eq, u_eq)
    B = jax.jacobian(f, argnums=2)(0., x_eq, u_eq).reshape((problem_params['nx'], problem_params['nu']))
    Q = jax.hessian(l, argnums=1)(0., x_eq, u_eq)
    R = jax.hessian(l, argnums=2)(0., x_eq, u_eq)

    # cheeky controllability test
    ctrb =  np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(problem_params['nx'])])
    if np.linalg.matrix_rank(ctrb) < problem_params['nx']:
        raise ValueError('linearisation not controllable aaaaah what did you do idiot')

    K0_inf, P0_inf, eigvals = pontryagin_utils.lqr(A, B, Q, R)

    # compute sqrtm of P, the value function matrix.
    # this is to later map points x from a small unit sphere to points Phalf x which are on a small value level set.
    P_eigv, P_eigvec = np.linalg.eigh(P0_inf)
    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T

    assert np.allclose(Phalf @ Phalf, P0_inf), 'sqrtm(P) calculation failed or inaccurate'

    N_trajectories = 1024

    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.
    key = jax.random.PRNGKey(0)
    normal_pts = jax.random.normal(key, shape=(N_trajectories, problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01
    # test if it worked
    vTs =  jax.vmap(lambda x: x @ P0_inf @ x.T)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    lamTs = jax.vmap(lambda x: x @ P0_inf * 2)(xTs)

    yTs = np.hstack([xTs, lamTs])


    pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)

    # sols = jax.vmap(pontryagin_solver)
    sol = pontryagin_solver(yTs[0], vTs[0], 10)
    sols = jax.vmap(pontryagin_solver, in_axes=(0, None, None))(yTs, vTs[0], 100)



    # having lots of trouble using sols.evaluate or sols.interpolation.evaluate in any way. thus, just use sol.ts \o/
    def plot_trajectories(ts, ys, color='green', alpha='.1'):

        # plot trajectories.
        pl.plot(ys[:, :, 0].T, ys[:, :, 1].T, color='green', alpha=.1)

        # sols.ys.shape = (N_trajectories, N_ts, 2*nx+1)
        # plot attitude with quiver.
        arrow_x = ys[:, :, 0].reshape(-1)
        arrow_y = ys[:, :, 1].reshape(-1)
        attitudes = ys[:, :, 4].reshape(-1)
        arrow_len = 0.5
        u = np.sin(-attitudes) * arrow_len
        v = np.cos(attitudes) * arrow_len

        pl.quiver(arrow_x, arrow_y, u, v, color='green', alpha=0.1)

    plot_trajectories(sols.ts, sols.ys)
    pl.show()

    ipdb.set_trace()

