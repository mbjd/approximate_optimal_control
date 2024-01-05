#!/usr/bin/env python
import jax
import jax.numpy as np
import diffrax

import pontryagin_utils

import ipdb
import scipy
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

import numpy as onp

from jax.config import config



# simple control system. double integrator with friction term.
def f(t, x, u):
    # p' = v
    # v' = f
    # f =  u

    return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u)

def l(t, x, u):
    Q = np.eye(2)
    R = np.eye(1)
    a = .01  # penalty function steepness parameter. --> 0 steep, --> +inf flat
    z = x[1] - 1  # x[1] <= 10   -->   z <= 0
    # state_penalty = .5 * (np.sqrt(1 + (z/a)**2) - z/a)
    state_penalty = np.maximum(0, z)**2 / a + np.maximum(0, x[0]-1)**2 / a
    # state_penalty = np.maximum(0, z) / a
    return x.T @ Q @ x + u.T @ R @ u + state_penalty


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



xf = np.array([0., 0.])

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
    Qf = 1 * np.eye(2)
    return (x.T @ Qf @ x).reshape()


problem_params = {
    'system_name': 'double_integrator_linear_statepenalty',
    'f': f,
    'l': l,
    'h': h,
    'T': 8,
    'nx': 2,
    'nu': 1,
    # 'U_interval': [-np.inf, np.inf],
    'U_interval': [-1, 1],
    'terminal_constraint': False,  # not tested with False for a long time
    'V_max': 30,
}

x_sample_scale = np.diag(np.array([1, 3]))
x_sample_cov = x_sample_scale @ x_sample_scale.T

# algo params copied from first resampling characteristics solvers
# -> so some of them might not be relevant
algo_params = {
    'pontryagin_solver_dt': 1 / 16,
    'pontryagin_solver_adaptive': True,
    'pontryagin_solver_dense': False,
    'pontryagin_solver_rtol': 1e-10,
    'pontryagin_solver_atol': 1e-10,
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
x0s = 0.1 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ circle_xs + xf[:, None]  # sqrtm "covariance" maps circle to ellipse.


x0 = x0s[:, 0]
v0 = (lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
v1 = problem_params['V_max']

# test single solution first

lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
y0 = np.concatenate([x0, lam0])
# sol = pontryagin_solver(y0, v0, v1)

def sol_single_and_dxdtheta(theta):

    def get_vs_ys(theta):
        direction = np.array([np.sin(theta), np.cos(theta)])
        x0 = 0.01 * np.linalg.inv(scipy.linalg.sqrtm(P0_inf)) @ direction + xf
        # lam0 = jax.jacobian(lambda x: (x-xf).T @ P0_inf @ (x-xf))(x0)
        lam0 = 2 * P0_inf @ (x0-xf)
        y0 = np.concatenate([x0, lam0])
        # same as experiment_orbits -- breaking change in pontryagin_utils lets this fail
        sol = pontryagin_solver(y0, v0, v1)
        vs = np.linspace(np.sqrt(v0), np.sqrt(v1), 101)**2
        ys = jax.vmap(sol.evaluate)(vs)
        return vs, ys

    def get_final_x(theta):
        vs, ys = get_vs_ys(theta)
        return ys[-1, 0:2]


    # finite difference = bad they said
    vs, ys = get_vs_ys(theta)
    # dvs, dys = jax.jacfwd(get_vs_ys)(theta)
    dsol_final = jax.jacrev(get_final_x)(theta)

    return vs, ys, dsol_final

# sol_single_and_dxdtheta(.1)

ax = pl.figure().add_subplot(projection='3d')
ax2d = pl.figure().add_subplot()
ax2d.axis('equal')
    # ax.plot(ys[:, :, 0].reshape(-1), ys[:, :, 1].reshape(-1), vs[:, :].reshape(-1), color='green', alpha=.5)

theta = 0.
target_x_steplen = .05
all_vs = []
all_ys = []
all_thetas = []

# cmap = matplotlib.colormaps['hsv']
cmap = matplotlib.colormaps['viridis']
# cmap = matplotlib.colormaps['jet']

t_alpha = .2
v_alpha = .1

i=0
while theta < 2*np.pi:
    i = i+1
    print(theta, end='\r')

    vs, ys, dfinaldtheta = sol_single_and_dxdtheta(theta)

    all_vs.append(vs)
    all_ys.append(ys)
    all_thetas.append(theta)

    # ax.plot(ys[:, 0].reshape(-1), ys[:, 1].reshape(-1), vs.reshape(-1), color='green', alpha=.1)

    if i % 1 == 0:
        ax2d.plot(ys[:, 0], ys[:, 1], color='black', alpha=t_alpha)
        ax.plot(ys[:, 0], ys[:, 1], vs, color='black', alpha=t_alpha)

    dx_final = dfinaldtheta[0:2]
    steplen_per_theta = np.linalg.norm(dfinaldtheta)

    thetastep = target_x_steplen / steplen_per_theta

    if thetastep < 0 or thetastep > 1:
        ipdb.set_trace()

    thetastep = np.clip(thetastep, 1e-8, 1)

    theta = theta + thetastep
print('')


all_ys = np.array(all_ys)
all_vs = np.array(all_vs)
# ipdb.set_trace()

# make basically the same plot, but with the data transposed, so we plot value level sets
# instead of trajectories.
# ax = pl.figure().add_subplot(projection='3d')
# for each value level set:
for vlevel in range(all_vs.shape[1]):
    try:
        vvec = all_vs[:, vlevel]
        x0vec = all_ys[:, vlevel, 0]
        x1vec = all_ys[:, vlevel, 1]

        ax2d.plot(x0vec, x1vec, color=cmap(vvec[0]/v1), alpha=v_alpha)
        ax.plot(x0vec, x1vec, vvec, color=cmap(vvec[0]/v1), alpha=v_alpha)
    except:
        # sometimes the last entries are NaN. Don't care
        pass

# thetas = np.linspace(0, 2*np.pi, 501)
# ax2d.plot(np.sin(thetas), np.cos(thetas), color='black')
# ax.plot(np.sin(thetas), np.cos(thetas), 0 * thetas, color='black')
# ax2d.scatter([0], [1], [0], color='black')

ax2d.plot([-5, 5], [1, 1], color='black', linestyle='--', alpha=.2)


pl.show()
ipdb.set_trace()

print('done')
