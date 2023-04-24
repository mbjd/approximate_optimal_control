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

    key = jax.random.PRNGKey(0)

    '''
    # define NN first.

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
    '''


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
    # we have a fake version of function overloading -- these do the same thing but with different inputs.
    # mostly the same though, only cosmetic difference, in the end they all reduce down to the first one.
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
    # R = np.array([[1., 0], [0, 2]])
    # A = np.array([[0.1, 0.3]])
    # x = np.array([[1., 1.]]).T
    # u_star_test = find_u_star_matrices( R, A )

    # the dynamics governing state, costate and value evolution according to
    # pontryagin minimum principle. normal in forward time.
    @jax.jit
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

    key, subkey = jax.random.split(key)

    # scale by scale matrix. data will be distributed ~ N(0, scale.T @ scale)
    scale = algoparams['x_sample_scale']
    all_x_T = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))

    max_steps = int(T / algoparams['dt'])

    @jax.jit
    def resample(ys, key):
        # all the ys that have left the interesting region, we want to put back into it.

        all_xs = ys[:, 0:nx, :]

        # parameterise x domain as ellipse: X = {x in R^n: x.T @ Q_x @ x <= 1}
        # 'x_domain': Q_x,
        ellipse_membership_fct = lambda x: x.T @ algoparams['x_domain'] @ x
        ellipse_memberships = jax.vmap(ellipse_membership_fct)(all_xs)

        # resample where this mask is 1.
        resample_mask = (ellipse_memberships > 1).astype(np.float32)
        # keep old sample where that mask is 1.
        not_resample_mask = 1 - resample_mask

        # just resample ALL the xs...
        resampled_xs = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))

        new_xs = resample_mask * resampled_xs + \
                not_resample_mask * all_xs

        # circumventing jax's immutable objects.
        # if docs are to be believed, after jit this will do an efficient in place update.
        ys.at[:, 0:nx, :].set(new_xs)
        return ys




    # solve pontryagin backwards, for vampping later.
    def pontryagin_backward_solver(y_final, tstart):

        '''
        to assemble the y vector, do this, but outside and with adjustments for vmap:
        # final condition (integration goes backwards...)
        costate_T = jax.grad(h)(x_T)
        v_T = h(x_T)
        y_final = np.vstack([x_T, costate_T, v_T])

        here we have t0 > t1, and negative dt (already defined here), to do backward integration

        when vmapping this we get back not a vector of solution objects, but a single
        solution object with an extra dimension in front of all others inserted in all arrays

        '''

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs
        saveat = diffrax.SaveAt(steps=True)
        dt = algoparams['dt']
        integ_time = algoparams['resample_interval']
        max_steps = int(integ_time / dt)

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=tstart, t1=tstart-integ_time, dt0=-dt, y0=y_final,
                saveat=saveat, max_steps=max_steps
        )

        return solution, solution.ys[-1]

    # vmap = gangster!
    # in_axes[1] = None caueses it to not do dumb stuff with the second argument. dunno exactly why
    batch_pontryagin_backward_solver = jax.vmap(pontryagin_backward_solver, in_axes=(0, None))

    # helper function, expands the state vector x to the extended state vector y = [x, λ, v]
    # λ is the costate in the pontryagin minimum principle
    # h is the terminal value function
    def x_to_y(x, h):
        costate = jax.grad(h)(x)
        v = h(x)
        y = np.vstack([x, costate, v])

        return y

    x_to_y_vmap = jax.vmap(lambda x: x_to_y(x, h))
    all_y_T = x_to_y_vmap(all_x_T)

    dt = algoparams['dt']
    t = T  # np.array([T])

    init_ys = all_y_T

    # calculate all the time and index parameters in advance to avoid the mess.
    # image for visualisation:

    # 0                                                                     T
    # |---------------------------------------------------------------------|
    # |         |         |         |         |         |         |         | resample_intervals
    # | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | timesteps

    # all arrays defined here contain all but the last (at T) steps.

    remainder = algoparams['resample_interval'] % dt
    assert np.allclose(remainder, 0), 'dt does not evenly divide resample_interval; problems may arise.'

    n_resamplings = int(T / algoparams['resample_interval'])
    timesteps_per_resample = int(0.5 + algoparams['resample_interval'] / dt)

    resampling_indices = np.arange(n_resamplings) * timesteps_per_resample
    resampling_ts = resampling_indices * dt

    n_timesteps = n_resamplings * timesteps_per_resample + 1 # so we have the boundary condition as well
    all_indices = onp.arange(n_timesteps)
    all_ts = all_indices * dt

    # full solution array. onp so we can modify in place.
    # every integration step fills a slice all_sols[:, i:i+timesteps_per_resample, :, :], for i in resampling_indices.
    # final ..., 1) in shape so we have nice column vectors just as everywhere else.
    all_sols = onp.zeros((algoparams['n_trajectories'], n_timesteps, 2*nx+1, 1))

    # the terminal condition separately
    # technically redundant but might make function fitting nicer later
    all_sols[:, -1, :, :] = all_y_T

    # the main loop.
    # basically, do backwards continuous-time approximate dynamic programming.
    # alternate between particle-based, batched pontryagin/characteristic step,
    # followed by resampling step.

    print('Main loop progress:')
    it = zip(resampling_indices[::-1], resampling_ts[::-1])
    for (resampling_i, resampling_t) in tqdm(it, total=n_resamplings):

        # we integrate backwards over the time interval [resampling_t, resampling_t + resample_interval].
        # corresponding data saved with save_idx = [:, resampling_i:resampling_i + timesteps_per_resample, : :]
        # at all_sols[save_idx]

        start_t = resampling_t + algoparams['resample_interval']

        # the main dish.
        # integrate the pontryagin necessary conditions backward in time, for a time period of
        # algoparams['resample_interval'], for a whole batch of terminal conditions.
        sol_object, final_ys = batch_pontryagin_backward_solver(init_ys, start_t)

        # sol_object.ys is the full solution array. it has shape (n_trajectories, n_timesteps, 2*nx+1, 1).
        # the 2*nx+1 comes from the 'extended state' = (x, λ, v).

        # sol object info is saved the other way around, going from initial=large to final=small times.
        save_idx = np.arange(resampling_i, resampling_i+timesteps_per_resample)
        save_idx_rev = save_idx[::-1]
        tdiffs = sol_object.ts[0,] - all_ts[save_idx_rev]
        assert(np.allclose(tdiffs, 0, atol=1e-3)), 'time array not like expected'

        # so we save it in the big solution array with reversed indices.
        all_sols[:, save_idx_rev, :, :] = sol_object.ys
        # ipdb.set_trace()

        # TODO fit GP/NN to current data to provide good start for resampling

        # resampling step
        key, subkey = jax.random.split(key)
        ys_resampled = resample(final_ys, subkey)

        # for next loop iteration:
        init_ys = ys_resampled


    # key, subkey = jax.random.split(key)
    # ys_resampled = resample(all_sols.ys[:, -1, :, :], subkey)

    fig = pl.figure(figsize=(8, 3))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    ts = all_ts

    all_xs       = all_sols[:, :, 0:nx, :]
    all_costates = all_sols[:, :, nx:2*nx, :]
    all_values   = all_sols[:, :, -1, :]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algoparams['n_trajectories'], -1, 1, 1)
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

    T = 5

    x_sample_scale = 100 * np.eye(2)

    # parameterise x domain as ellipse: X = {x in R^n: x.T @ Q_x @ x <= 1}
    # if q diagonal, these numbers are the ellipse axes length along each state dim.
    Q_sqrt = np.diag(1/np.array([10, 10]))

    # ... or some sublevel set of the state distribution.
    Q_sqrt = np.diag(1 / np.diag(10 * x_sample_scale))
    Q_x = Q_sqrt @ Q_sqrt.T

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.
    algoparams = {
            'nn_layersizes': (32, 32, 32),
            'n_trajectories': 32,
            'dt': 0.01,
            'resample_interval': 0.1,
            'x_domain': Q_x,
            'x_sample_scale': x_sample_scale,  # basically sqrtm(cov)
    }


    hjb_characteristics_solver(f, l, h, T, 2, 1, algoparams)

if __name__ == '__main__':

    characteristics_experiment_simple()
