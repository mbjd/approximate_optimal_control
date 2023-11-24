#!/usr/bin/env python
import jax
import jax.numpy as np
import diffrax

import sampling
import pontryagin_utils
import plotting_utils
import nn_utils
import eval_utils

from main import sample_uniform, experiment_controlcost_vs_traindata, ode_dt_sweep

import ipdb
import scipy
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

import numpy as onp

from jax.config import config



# example from idea dump. orbit-like thing where move in circles. if outside the
# unit circle, move in one direction, inwards other. input u moves "orbit radius".
# aim to stabilise (0, 1)
def f(t, x, u):
    rotspeed = (x[0]**2 + x[1]**2 - 1).reshape(u.shape)
    # mat = np.diag([u, u]) * u + np.array([[0, -1], [1, 0]]) * rotspeed

    mat = np.array([[u, -rotspeed], [rotspeed, u]]).squeeze()

    return mat @ x  # weird pseudo linear thing


def l(t, x, u):
    Q = np.eye(2)
    err = x - np.array([0, 1])
    distpenalty = err.T @ Q @ err
    rotspeed = x[0]**2 + x[1]**2 - 1
    vpenalty = (x[0]**2 + x[1]**2 - 1)**2
    inp_penalty = 10 * u**2
    return (vpenalty + 0.1 * distpenalty + inp_penalty).reshape()


# from http://www.mwm.im/lqr-controllers-with-python/
def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151
    # first, try to solve the ricatti equation
    # somehow this stuff only works with old np.matrix types and * multiplication
    X = onp.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    # compute the LQR gain
    K = onp.matrix(scipy.linalg.inv(R) * (B.T * X))
    eigVals, eigVecs = scipy.linalg.eig(A - B * K)
    return K, X, eigVals



xf = np.array([0., 1.])

# linearise around equilibrium.
# terminal lqr controller for inf horizon approximation
# store these somewhere? or hope for jit to save us?
zero_u = np.zeros(1)
A = jax.jacobian(f, argnums=1)(0., xf, zero_u)
B = jax.jacobian(f, argnums=2)(0., xf, zero_u).reshape((2, 1))
Q = jax.hessian(l, argnums=1)(0., xf, zero_u)
R = jax.hessian(l, argnums=2)(0., xf, zero_u)

K0_inf, P0_inf, eigvals = lqr(A, B, Q, R)


def h(x):

    # terminal value.
    return (x.T @ P0_inf @ x).reshape()


problem_params = {
    'system_name': 'orbits',
    'f': f,
    'l': l,
    'h': h,
    'T': np.inf,
    'nx': 2,
    'nu': 1,
    'U_interval': [-.2, .2],
    'terminal_constraint': False,
    'V_max': 16,
}

algo_params = {
    'pontryagin_solver_dt': 1 / 16,
    'pontryagin_solver_adaptive': True,
    'pontryagin_solver_dense': False,
    'pontryagin_solver_rtol': 1e-6,
    'pontryagin_solver_atol': 1e-4,
    'pontryagin_solver_maxsteps': 1024,
}


# try this pro move, very mockup like, probably wrong
# integrate along value axis, that is, with value as independent axis
# this is achieved by dividing the whole vector field from the hamiltonian dynamics by -l(x, u) = dv/dt
# then assume quadratic lqr value function on some small subset
# and integrate "up" the value axis starting from points on a small value level set.
# see what happens \o/

# basically it works, BUT we see that when approximating state constraints with smooth penalty functions,
# the sensitivity issue becomes MUCH more pronounced. just to visualise a kind of iterative search procedure
# here, that just resamples more closely around the point that reached the smallest x[0].
pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)



thetas = np.linspace(0, 2 * np.pi, 128)[:-1]
circle_xs = np.row_stack([np.sin(thetas), np.cos(thetas)])
x0s = 0.01 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ circle_xs + xf[:, None]  # sqrtm "covariance" maps circle to ellipse.


x0 = x0s[:, 0]
v0 = (lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
v1 = 4

# test single solution first

lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
y0 = np.concatenate([x0, lam0])
# sol = pontryagin_solver(y0, v0, v1)

load=False
if not load:
    def get_sol(theta):
        direction = np.array([np.sin(theta), np.cos(theta)])
        x0 = 0.01 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ direction + xf
        lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
        y0 = np.concatenate([x0, lam0])
        sol = pontryagin_solver(y0, v0, v1)
        return sol

    @jax.jit
    def sol_single_and_dxdtheta(theta):

        def get_sol(theta):
            direction = np.array([np.sin(theta), np.cos(theta)])
            x0 = 0.01 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ direction + xf
            # lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
            lam0 = 2 * P0_inf @ (x0-xf)
            y0 = np.concatenate([x0, lam0])
            sol = pontryagin_solver(y0, v0, v1)
            return sol

        get_dsol = jax.jacobian(get_sol)

        # finite difference = bad they said
        dtheta = .00001
        sol = get_sol(theta)
        solp = get_sol(theta + dtheta)

        dsol_final = (solp.evaluate(v1) - sol.evaluate(v1))/dtheta

        vs = np.linspace(np.sqrt(v0), np.sqrt(v1), 41)**2
        ys = jax.vmap(sol.evaluate)(vs)
        return vs, ys, dsol_final

    sol_single_and_dxdtheta(.1)

    ax = pl.figure().add_subplot(projection='3d')
    ax2d = pl.figure().add_subplot()
        # ax.plot(ys[:, :, 0].reshape(-1), ys[:, :, 1].reshape(-1), vs[:, :].reshape(-1), color='green', alpha=.5)

    theta = 0
    target_x_steplen = .2
    all_vs = []
    all_ys = []

    i=0
    while theta < 2*np.pi:
        i = i+1
        print(theta)
        ts, ys, dfinaldtheta = sol_single_and_dxdtheta(theta)

        all_vs.append(ts)
        all_ys.append(ys)

        cmap = matplotlib.colormaps['hsv']
        # ax.plot(ys[:, 0].reshape(-1), ys[:, 1].reshape(-1), ts.reshape(-1), color='green', alpha=.1)

        if i % 40 == 0:
            ax2d.plot(ys[:, 0], ys[:, 1], color='black', alpha=.3)
            ax.plot(ys[:, 0], ys[:, 1], ts, color='black', alpha=.3)

        dx_final = dfinaldtheta[0:2]
        steplen_per_theta = np.linalg.norm(dfinaldtheta)
        thetastep = target_x_steplen / steplen_per_theta
        theta = theta + thetastep


    all_ys = np.array(all_ys)
    all_vs = np.array(all_vs)

    # make basically the same plot, but with the data transposed, so we plot value level sets
    # instead of trajectories.
    # ax = pl.figure().add_subplot(projection='3d')
    # for each value level set:
    for vlevel in range(all_vs.shape[1]):
        vvec = all_vs[:, vlevel]
        x0vec = all_ys[:, vlevel, 0]
        x1vec = all_ys[:, vlevel, 1]

        #
        ax2d.plot(x0vec, x1vec, color=matplotlib.colormaps['jet'](vvec[0]/v1), alpha=.3)
        ax.plot(x0vec, x1vec, vvec, color=matplotlib.colormaps['jet'](vvec[0]/v1), alpha=.3)

    thetas = np.linspace(0, 2*np.pi, 501)
    ax2d.plot(np.sin(thetas), np.cos(thetas), color='black')
    ax.plot(np.sin(thetas), np.cos(thetas), 0 * thetas, color='black')
    ax2d.scatter([0], [1], [0], color='black')


    pl.show()
    ipdb.set_trace()

    # np.save('ys', all_ys)
    # np.save('vs', all_vs)

else:
    all_ys = np.load('ys')
    all_vs = np.load('vs')

extent = 1.5
N_linspace = 81
x1s = x2s = np.linspace(-extent, extent, N_linspace)
vs = onp.ones((N_linspace, N_linspace)) * v1

# map float between -extent and extent to int from 0 to N_linspace-1
state_to_idx = lambda x: int((x + extent) / (2*extent) * (N_linspace-1))

for v, ys in tqdm.tqdm(zip(all_vs, all_ys), total=all_vs.shape[0]):
    try:
        idx_x1 = state_to_idx(ys[0])
        idx_x2 = state_to_idx(ys[1])
        if vs[idx_x1, idx_x2] >= v:
            vs[idx_x1, idx_x2] = v
    except:
        # either ys is nan somewhere, or indices out of bounds.
        pass

pl.imshow(vs[:, ::-1].T, extent=(-extent, extent, -extent, extent), aspect='equal')
thetas = np.linspace(0, 2*np.pi, 501)
pl.plot(np.sin(thetas), np.cos(thetas))
pl.scatter([0], [1], color='red')
pl.show()


ipdb.set_trace()

@jax.jit
def x0_to_vs_ys(x0):

    xf = np.array([0., 1.])
    lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
    y0 = np.concatenate([x0, lam0])
    sol = pontryagin_solver(y0, v0, v1)

    # vs = np.linspace(np.sqrt(v0), np.sqrt(v1), 201)**2
    # ys = jax.vmap(sol.evaluate)(vs)
    # return vs, ys
    return sol.ts, sol.ys




pl.show()


print('done')
