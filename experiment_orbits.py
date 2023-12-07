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
        # 'V_max': 3.8,
        'V_max': 8,
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
v1 = problem_params['V_max']

# test single solution first

lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
y0 = np.concatenate([x0, lam0])
# sol = pontryagin_solver(y0, v0, v1)

@jax.jit
def sol_single_and_dxdtheta(theta):


    vs = np.linspace(np.sqrt(v0), np.sqrt(v1), 101)**2

    def get_sol(theta):
        direction = np.array([np.sin(theta), np.cos(theta)])
        x0 = 0.01 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ direction + xf
        # lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
        lam0 = 2 * P0_inf @ (x0-xf)
        y0 = np.concatenate([x0, lam0])
        sol = pontryagin_solver(y0, v0, v1)

        ys = jax.vmap(sol.evaluate)(vs)

        return ys

    get_dsol = jax.jacfwd(lambda theta: get_sol(theta)[-2, 0:2])

    # okay maybe doing this with nice autodiff still doesnt work.
    # other, finite-diff implementation still in lqr_statepenalty example
    sol = get_sol(theta)
    dsol_final = get_dsol(theta)

    # dsol_final = (solp.evaluate(v1) - sol.evaluate(v1))/dtheta

    return vs, sol, dsol_final

sol_single_and_dxdtheta(.1)

ax = pl.figure().add_subplot(projection='3d')
ax2d = pl.figure().add_subplot()
ax2d.axis('equal')
    # ax.plot(ys[:, :, 0].reshape(-1), ys[:, :, 1].reshape(-1), vs[:, :].reshape(-1), color='green', alpha=.5)

theta = 0
target_x_steplen = .02
all_vs = []
all_ys = []
all_thetas = []

# cmap = matplotlib.colormaps['hsv']
cmap = matplotlib.colormaps['viridis']
# cmap = matplotlib.colormaps['jet']

t_alpha = .8
v_alpha = .4

i=0
while theta < 2*np.pi:
    i = i+1
    print(theta, end='\r')
    all_thetas.append(theta)
    ts, ys, dfinaldtheta = sol_single_and_dxdtheta(theta)

    all_vs.append(ts)
    all_ys.append(ys)

    # ax.plot(ys[:, 0].reshape(-1), ys[:, 1].reshape(-1), ts.reshape(-1), color='green', alpha=.1)

    if i % 100 == 0:
        ax2d.plot(ys[:, 0], ys[:, 1], color='black', alpha=t_alpha)
        ax.plot(ys[:, 0], ys[:, 1], ts, color='black', alpha=t_alpha)

    dx_final = dfinaldtheta[0:2]
    steplen_per_theta = np.linalg.norm(dfinaldtheta)
    thetastep = target_x_steplen / steplen_per_theta
    theta = theta + thetastep
print('')


all_ys = np.array(all_ys)
all_vs = np.array(all_vs)

pl.figure()

# make basically the same plot, but with the data transposed, so we plot value level sets
# instead of trajectories.
# ax = pl.figure().add_subplot(projection='3d')
# for each value level set:
for vlevel in tqdm.tqdm(range(all_vs.shape[1])):
    try:
        vvec = all_vs[:, vlevel]
        x0vec = all_ys[:, vlevel, 0]
        x1vec = all_ys[:, vlevel, 1]

        pl.plot(x0vec, x1vec, color=cmap(vvec[0]/v1), alpha=v_alpha)

        # ax2d.plot(x0vec, x1vec, color=cmap(vvec[0]/v1), alpha=v_alpha)
        # ax.plot(x0vec, x1vec, vvec, color=cmap(vvec[0]/v1), alpha=v_alpha)
    except:
        # sometimes the last entries are NaN. Don't care
        pass

    # ipdb.set_trace()
    pl.savefig(f'animation_figs/orbits_{vlevel:05d}.png', dpi=400)

thetas = np.linspace(0, 2*np.pi, 501)
ax2d.plot(np.sin(thetas), np.cos(thetas), color='black')
ax.plot(np.sin(thetas), np.cos(thetas), 0 * thetas, color='black')
ax2d.scatter([0], [1], [0], color='black')


pl.show()
ipdb.set_trace()

print('done')
