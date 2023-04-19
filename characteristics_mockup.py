#!/usr/bin/env python

# jax
import jax
import jax.numpy as np
import optax
import diffrax

# cheating on equinox :/
import flax
from flax import linen as nn
from typing import Sequence, Optional

# other, trivial stuff
import numpy as onp

import matplotlib.pyplot as pl
import matplotlib

import ipdb

import time
from tqdm import tqdm

def hjb_characteristics_solver(f, l, h, T, nx, nu, algoparams):
    '''
    f: dynamics. vector-valued function of t, x, u.
    l: cost. scalar-valued function of t, x, u.
    h: terminal cost. scalar-valued function of x.
    T: time horizon > 0. problem is solved over t ∈ [0, T] with V(T, x) = h(x).
    nx, nu: dimensions of state and input spaces. obvs should match f, l, h.

    algoparams: dictionary with algorithm tuning parameters.
      nn_layersizes: sizes of NN layers from input to output, e.g. (64, 64, 16)
    '''

    # define NN first.
    key = jax.random.PRNGKey(0)

    class my_nn_flax(nn.Module):
        features: Sequence[int]
        output_dim: Optional[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(features=feat)(x)
                x = nn.softmax(x)

            if self.output_dim is not None:
                x = nn.Dense(features=self.output_dim)(x)
            return x



    # input dim is assigned when passing first input at init.
    V_nn = my_nn_flax(features=algoparams['nn_layersizes'], output_dim=1)


    # we want to find the value function V.
    # for that we want to approximately satisfy the hjb equation:
    #    0 = V_t(t, x) + inf_u { l(t, x, u) + V_x(t, x).T @ f(t, x, u) }
    #    V(x, T) = h(x)

    # for this first case let us consider
    #    f(t, x, u) = f_tx(t, x) + g(t, x) @ u
    #    l(t, x, u) = l_tx(t, x) + u.T @ R @ u.
    # so a control affine system with cost quadratic in u. this will make
    # the inner minimization simpler:

    #       argmin_u l(t, x, u) + V_x(t, x).T @ f(t, x, u)
    #     = argmin_u l_x(x) + u.T @ R @ u + V_x(t, x).T @ (f_tx(t, x) + g(t, x) @ u)
    #     = argmin_u          u.T @ R @ u + V_x(t, x).T @ (             g(t, x) @ u)

    # and, with A = V_x(t, x).T @ g(t, x):

    #     = argmin_u u.T @ R @ u + A @ u

    # which is an unconstrained convex QP, solved by setting the gradient to zero.

    #     = u s.t. 0 = (R + R.T) u + A
    #     = solution of linear system (R + R.T, -A)

    # this is implemented in the following - the pointwise minimization over u of the hamiltonian.
    def find_u_star_matrices(R, A):
        # A is a row vector here...
        u_star_unconstrained = np.linalg.solve(R + R.T, -A.T)
        return np.clip(u_star_unconstrained, -.1, .1)

    def find_u_star_functions(f, l, V, t, x):
        # assuming that l is actually of the form l(t, x, u) = l_tx(t, x) + u.T @ R @ u,
        # the hessian R is independent of u. R should be of shape (nu, nu).
        zero_u = np.zeros((1, 1))
        R = jax.hessian(l, argnums=2)(t, x, zero_u).reshape(1, 1)

        grad_V_x = jax.jacobian(V, argnums=1)(t, x).reshape(1, nx)
        grad_f_u = jax.jacobian(f, argnums=2)(t, x, zero_u).reshape(nx, nu)
        A = grad_V_x @ grad_f_u        # should have shape (1, nu)

        return find_u_star_matrices(R, A)

    def find_u_star_costate(f, l, costate, t, x):
        zero_u = np.zeros((1, 1))
        R = jax.hessian(l, argnums=2)(t, x, zero_u).reshape(1, 1)

        # costate = grad_V_x.T (costate colvec, grad_V_x col vec)
        # grad_V_x = jax.jacobian(V, argnums=1)(t, x).reshape(1, nx)
        grad_f_u = jax.jacobian(f, argnums=2)(t, x, zero_u).reshape(nx, nu)
        A = costate.T @ grad_f_u        # should have shape (1, nu)

        return find_u_star_matrices(R, A)

    # check. if u is 2d, we need 2x2 R and 1x2 A
    R = np.array([[1., 0], [0, 2]])
    A = np.array([[0.1, 0.3]])
    x = np.array([[1., 1.]]).T
    print(find_u_star_matrices(
        R, A
    ))


    # dummy inputs.
    # reshape() gives a shape () array, whereas item() gives a float
    V = lambda t, x: (x.T @ np.eye(2) @ x).reshape()
    t = 0

    print(find_u_star_functions(f, l, V, t, x))

    # the dynamics governing state, costate and value evolution according to
    # pontryagin minimum principle. normal in forward time.
    def f_forward(t, y, args=None):

        # unpack. english names to distinguish from function arguments...
        state   = y[0:nx]
        costate = y[nx:2*nx]
        value   = y[2*nx]

        # define ze hamiltonian for that time.
        H = lambda x, u, costate: l(t, x, u) + costate.T @ f(t, x, u)

        u_star = find_u_star_costate(f, l, costate, t, state)

        state_dot   = jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx, 1)
        costate_dot = jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx, 1)
        value_dot   = -l(t, state, u_star)

        y_dot = np.vstack([state_dot, costate_dot, value_dot])
        return y_dot

    f_forward = jax.jit(f_forward)

    key, subkey = jax.random.split(key)

    # scale by scale matrix. data will be distributed ~ N(0, scale.T @ scale)
    scale = algoparams['x_sample_scale']
    all_x_T = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))

    max_steps = int(T / algoparams['dt'])

    def resample(ys, key):
        # all the ys that have left the interesting region, we want to put back into it.

        all_xs = ys[:, 0:nx, :]

        # parameterise x domain as ellipse: X = {x in R^n: x.T @ Q_x @ x <= 1}
        # 'x_domain': Q_x,
        ellipse_membership_fct = lambda x: x.T @ algoparams['x_domain'] @ x
        ellipse_memberships = jax.vmap(ellipse_membership_fct)(all_xs)

        # just resample ALL the xs...
        all_x_T = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))

        # make some mask that tells us where to update and were not...



    # solve pontryagin backwards, for vampping later.
    def solve_single(x_T):

        # final condition (integration goes backwards...)
        costate_T = jax.grad(h)(x_T)
        v_T = h(x_T)
        y_T = np.vstack([x_T, costate_T, v_T])

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(steps=True)
        dt = algoparams['dt']
        max_steps = int(T / dt)

        solution = diffrax.diffeqsolve(term, solver, t0=T, t1=0, dt0=-dt, y0=y_T, saveat=saveat, max_steps=max_steps)
        return solution


    # when vmapping this we get back not a vector of solution objects, but a single
    # solution object with an extra dimension in front of all others inserted in all arrays

    # vmap = gangster!
    solve_multiple = jax.vmap(solve_single)
    all_sols = solve_multiple(all_x_T)

    key, subkey = jax.random.split(key)
    ys_resampled = resample(all_sols.ys[:, -1, :, :], subkey)

    fig = pl.figure(figsize=(8, 3))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    ts = all_sols.ts[0, :]

    all_xs = all_sols.ys[:, :, 0:nx, :]
    all_costates = all_sols.ys[:, :, nx:2*nx, :]
    all_values = all_sols.ys[:, :, -1, :]

    max_norm = 1000

    # -1 becomes n_trajectories
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(-1, max_steps, 1, 1)
    # when norm <= max_norm, scalings = 1, otherwise, scales to max_norm
    scalings = np.minimum(1, max_norm/x_norms)

    all_xs = all_xs * scalings

    a = 0.1 * 256 / (algoparams['n_trajectories'])

    # somehow i was not able to get it to work with a single call and data in matrix form.
    # ts_expanded = np.tile(ts[:, None], algoparams['n_trajectories'])
    # ipdb.set_trace()
    # ax0.plot(ts, all_xs[:, :, 0].squeeze().T, all_xs[:, :, 1].squeeze().T, color='black', alpha=a)
    # ax1.plot(all_xs[:, :, 0], all_xs[:, :, 1], all_values, color='black', alpha=a)

    # so now we go back to the stone age
    for i in range(algoparams['n_trajectories']):
        ax0.plot(ts, all_xs[i, :, 0].squeeze(), all_xs[i, :, 1].squeeze(), color='black', alpha=a)
        ax1.plot(all_xs[i, :, 0], all_xs[i, :, 1], all_values[i], color='blue', alpha=a)

    ax0.set_xlabel('t')
    ax0.set_ylabel('x_0')
    ax0.set_zlabel('x_1')

    ax1.set_xlabel('x_0')
    ax1.set_ylabel('x_1')
    ax1.set_zlabel('value')

    pl.show()




def characteristics_experiment_simple():

    # simple control system. double integrator with friction term.
    def f(t, x, u):
        # p' = v
        # v' = f
        # f = -v**3 + u
        # clip so we don't have finite escape time when integrating backwards
        v_cubed = np.clip((np.array([[0, 1]]) @ x)**3, -100, 100)
        return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u - v_cubed)

    def l(t, x, u):
        Q = np.eye(2)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 1 * np.eye(2)
        return (x.T @ Qf @ x).reshape()

    T = .1
    # parameterise x domain as ellipse: X = {x in R^n: x.T @ Q_x @ x <= 1}
    # if q diagonal, these numbers are the ellipse axes length along each state dim.
    Q_sqrt = np.diag(1/np.array([10, 10]))
    Q_x = Q_sqrt @ Q_sqrt.T

    x_sample_scale = np.diag(np.array([10, 10]))

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.
    algoparams = {
            'nn_layersizes': (32, 32, 32),
            'n_trajectories': 64,
            'dt': 0.01,
            'resample_interval': 0.2,
            'x_domain': Q_x,
            'x_sample_scale': x_sample_scale,  # basically sqrtm(cov)
    }


    hjb_characteristics_solver(f, l, h, T, 2, 1, algoparams)

if __name__ == '__main__':

    characteristics_experiment_simple()
