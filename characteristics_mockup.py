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

import tk as tkinter

import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as pl

import ipdb

import time
from tqdm import tqdm

class my_nn_flax(nn.Module):

    # simple, fully connected NN class.
    # for bells & wistles -> nn_wrapper class :)

    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.softplus(x)

        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)

        return x



class nn_wrapper():

    # all the usual business logic around the NN.
    # initialisation, data loading, training, loss plotting

    def __init__(self, input_dim, layer_dims, output_dim, key):

        self.input_dim  = input_dim
        self.layer_dims = layer_dims
        self.output_dim = output_dim

        self.nn = my_nn_flax(features=layer_dims, output_dim=output_dim)

        # could make these arguments as well...
        factor = .1
        lr_schedule = optax.piecewise_constant_schedule(
                init_value=0.05,
                boundaries_and_scales = {
                    500: factor,
                    1000: factor,
                    2000: factor,
                }
        )

        self.optim = optax.adam(learning_rate=lr_schedule)

        self.loss_val_grad = jax.value_and_grad(self.loss, argnums=0)

        # turns out if we remove the last dimension (which made everything a
        # column vector) we stop having the weird problem of wrong output shape.
        self.params = self.nn.init(key, np.zeros((self.input_dim,)))

        self.opt_state = self.optim.init(self.params)

    # so we can use it like a function :)
    def __call__(self, x):
        return self.nn.apply(self.params, x)

    # just an example loss function, very standard.
    def point_loss(self, params, x, y):
        y_pred = self.nn.apply(params, x)
        return (y_pred - y)**2

    def loss(self, params, xs, ys):
        losses = jax.vmap(self.point_loss, in_axes=(None, 0, 0))(params, xs, ys)
        return np.mean(losses)


    def train(self, xs, ys, key, batchsize=64, N_epochs=100, plot=False):


        # make minibatches
        N_datapts = xs.shape[0]


        N_datapts_equal_batches = N_datapts - N_datapts % batchsize

        N_batches = N_datapts_equal_batches / batchsize
        assert N_batches == int(N_batches), 'N_batches size not integer'
        N_batches = int(N_batches)

        @jax.jit
        def generate_minibatch_index_array(key):

            # discard a bit of data to make equally sized batches
            data_idx_shuffled = jax.random.permutation(key, np.arange(N_datapts))[0:N_datapts_equal_batches]

            # batched_data_idx[i, :] = data indices for batch number i.
            batched_data_idx = data_idx_shuffled.reshape(N_batches, batchsize)
            return batched_data_idx


        total_iters = N_batches * N_epochs
        losses = onp.zeros(total_iters)

        @jax.jit
        def update_step(xs, ys, opt_state, params):

            loss_val, loss_grad = self.loss_val_grad(params, xs, ys)

            updates, opt_state = self.optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)

            return opt_state, params, loss_val

        for i in tqdm(range(total_iters)):

            # re-shuffle batch after epoch.
            i_batch = i % N_batches
            if i_batch == 0:
                key, subkey = jax.random.split(key)
                batched_data_idx = generate_minibatch_index_array(subkey)

            # get the batch
            batch_idx = batched_data_idx[i_batch, :]

            xs_batch = xs[batch_idx]
            ys_batch = ys[batch_idx]

            self.opt_state, self.params, loss_val = update_step(
                    xs_batch, ys_batch, self.opt_state, self.params
            )

            losses[i] = loss_val

        if plot:
            pl.plot(losses)
            pl.show()



def hjb_characteristics_solver(problemparams, algoparams):
    '''
    (all in problemparams:)
    f: dynamics. vector-valued function of t, x, u.
    l: cost. scalar-valued function of t, x, u.
    h: terminal cost. scalar-valued function of x.
    T: time horizon > 0. problem is solved over t ∈ [0, T] with V(T, x) = h(x).
    nx, nu: dimensions of state and input spaces. obvs should match f, l, h.

    algoparams: dictionary with algorithm tuning parameters.
      nn_layersizes: sizes of NN layers from input to output, e.g. (64, 64, 16)
    '''


    # do this with keyword arguments and **dict instead?
    f  = problemparams['f' ]
    l  = problemparams['l' ]
    h  = problemparams['h' ]
    T  = problemparams['T' ]
    nx = problemparams['nx']
    nu = problemparams['nu']


    # define NN first.

    key = jax.random.PRNGKey(0)
    V_nn = nn_wrapper(
            input_dim  = 1 + nx,   # inputs are vectors containing (t, x)
            layer_dims = algoparams['nn_layersizes'],
            output_dim = 1,
            key        = key
    )

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
        return np.clip(u_star_unconstrained, -1, 1)

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
        H = lambda x, u, λ: l(t, x, u) + λ.T @ f(t, x, u)

        u_star = find_u_star_costate(f, l, costate, t, state)

        # the first line is just a restatement of the dynamics
        # but doesn't it look cool with those partial derivatives??
        state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx, 1)
        costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx, 1)
        value_dot   = -l(t, state, u_star)

        y_dot = np.vstack([state_dot, costate_dot, value_dot])
        return y_dot

    key, subkey = jax.random.split(key)

    # scale by scale matrix. data will be distributed ~ N(0, scale.T @ scale)
    # scale = algoparams['x_sample_scale']
    # all_x_T = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))

    # ... not anymore. all parameterized with covariance.
    all_x_T = jax.random.multivariate_normal(
            key,
            mean=np.zeros(nx,),
            cov=algoparams['x_sample_cov'],
            shape=(algoparams['n_trajectories'],)
    ).reshape(algoparams['n_trajectories'], nx, 1)

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
        # resampled_xs = scale @ jax.random.normal(key, (algoparams['n_trajectories'], nx, 1))
        resampled_xs = jax.random.multivariate_normal(
                key,
                mean=np.zeros(nx,),
                cov=algoparams['x_sample_cov'],
                shape=(algoparams['n_trajectories'],)
        ).reshape(algoparams['n_trajectories'], nx, 1)

        new_xs = resample_mask * resampled_xs + \
                not_resample_mask * all_xs

        # circumventing jax's immutable objects.
        # if docs are to be believed, after jit this will do an efficient in place update.
        ys = ys.at[:, 0:nx, :].set(new_xs)
        return ys, resample_mask

    @jax.jit
    def resample_all(ys, key):
        # just generate all the states according to the state distribution

        resampled_xs = jax.random.multivariate_normal(
                key,
                mean=np.zeros(nx,),
                cov=algoparams['x_sample_cov'],
                shape=(algoparams['n_trajectories'],)
        ).reshape(algoparams['n_trajectories'], nx, 1)

        ys = ys.at[:, 0:nx, :].set(resampled_xs)

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
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None)
    ))

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

    # somehow with % it is inexact sometimes.
    # remainder = algoparams['resample_interval'] % dt
    # so we calculate explicitly the difference between the quotient and the nearest integer.
    remainder = algoparams['resample_interval'] / dt - int(0.5 + algoparams['resample_interval'] / dt)
    assert np.allclose(remainder, 0, atol=1e-3), 'dt does not evenly divide resample_interval; problems may arise.'

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
    where_resampled = onp.zeros((algoparams['n_trajectories'], n_timesteps), dtype=bool)

    # the terminal condition separately
    # technically redundant but might make function fitting nicer later
    all_sols[:, -1, :, :] = all_y_T

    # the main loop.
    # basically, do backwards continuous-time approximate dynamic programming.
    # alternate between particle-based, batched pontryagin/characteristic step,
    # followed by resampling step.

    print('Main loop progress:')
    it = zip(resampling_indices[::-1], resampling_ts[::-1])
    # for (resampling_i, resampling_t) in tqdm(it, total=n_resamplings):
    for (resampling_i, resampling_t) in it:

        # we integrate backwards over the time interval [resampling_t, resampling_t + resample_interval].
        # corresponding data saved with save_idx = [:, resampling_i:resampling_i + timesteps_per_resample, : :]
        # at all_sols[save_idx]

        start_t = resampling_t + algoparams['resample_interval']

        # the main dish.
        # integrate the pontryagin necessary conditions backward in time, for a time period of
        # algoparams['resample_interval'], for a whole batch of terminal conditions.
        # ipdb.set_trace()
        sol_object, final_ys = batch_pontryagin_backward_solver(init_ys, start_t)

        # sol_object.ys is the full solution array. it has shape (n_trajectories, n_timesteps, 2*nx+1, 1).
        # the 2*nx+1 comes from the 'extended state' = (x, λ, v).

        # sol object info is saved the other way around, going from initial=large to final=small times.
        save_idx = np.arange(resampling_i, resampling_i+timesteps_per_resample)
        save_idx_rev = save_idx[::-1]
        # tdiffs = sol_object.ts[0,] - all_ts[save_idx_rev]
        # assert(np.allclose(tdiffs, 0, atol=1e-3)), 'time array not like expected'

        # so we save it in the big solution array with reversed indices.
        all_sols[:, save_idx_rev, :, :] = sol_object.ys
        # ipdb.set_trace()

        # TODO fit GP/NN to current data to provide good start for resampling

        # training data terminology:

        # input = [t, state], label = V

        # reshape - during training we do not care which is the time and which the
        # trajectory index, they are all just pairs of ((t, x), V) regardless

        # before we have a shape (n_traj, n_time, 2*nx+1, 1).
        # here we remove the last dimension - somehow it makes NN stuff easier.
        train_states = all_sols[:, resampling_i:-1, 0:nx, :].reshape(-1, nx)

        # make new axes to match train_state
        train_ts = all_ts[None, resampling_i:-1, None]
        # repeat for all trajectories (axis 0), which all have the same ts.
        train_ts = np.repeat(train_ts, algoparams['n_trajectories'], axis=0)
        # flatten the same way as train_states
        train_ts = train_ts.reshape(-1, 1)

        # assemble into main data arrays
        # these are of shape (n_datpts, n_inputfeatures) and (n_datapts, 1)
        train_inputs = np.concatenate([train_ts, train_states], axis=1)
        train_labels = all_sols[:, resampling_i:-1, 2*nx, :].reshape(-1, 1)

        key, train_key = jax.random.split(key)
        V_nn.train(
                train_inputs, train_labels, key,
                batchsize=algoparams['batchsize'], N_epochs=algoparams['N_epochs'],
                plot=True
        )
        break




        # TODO how to handle this?
        # we have some intermediate value function approximation at time t: Vt_fct
        # we cannot pass it as an argument to the resample functions because then they
        # cannot be jit-ted (probably only minor performance loss though??)
        # so should we implement the bulk of the resampling logic here anyway?
        # think tomorrow.

        # something like:
        # resampled_values = jax.vmap(Vt_fct)(resampled_xs)
        # resampled_costates = jax.vmap(jax.jacobian(Vt_fct))(resampled_xs)
        # followed by appropriate masking & updating main solution.

        resampling_type = 'minimal'

        if resampling_type == 'all':
            # put ALL points at random new locations, and update their value
            # and costate info according to some value function approximation.

            ys_resampled = resample_all(final_ys, subkey)
            where_resampled[:, resampling_i, None, None] = True

            init_ys = ys_resampled

        elif resampling_type == 'minimal':
            # only put the points that have left the interesting state domain
            # at new locations, and only update these state's value/costate info.
            ys_resampled, resampling_mask = resample(final_ys, subkey)
            where_resampled[:, resampling_i, None, None] = resampling_mask

            init_ys = ys_resampled

        elif resampling_type == 'none':
            init_ys = final_ys

        else:
            raise RuntimeError(f'Invalid resampling type "{resampling_type}"')

    return all_sols, all_ts, where_resampled


def plot_2d(all_sols, all_ts, where_resampled, problemparams, algoparams):

    fig = pl.figure(figsize=(8, 3))

    ax0 = fig.add_subplot(111, projection='3d')
    # ax0 = fig.add_subplot(121, projection='3d')
    # ax1 = fig.add_subplot(122, projection='3d')


    nx = problemparams['nx']
    T = problemparams['T']
    all_xs       = all_sols[:, :,  0:nx,    :]
    all_costates = all_sols[:, :,  nx:2*nx, :]
    all_values   = all_sols[:, :, -1,       :]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algoparams['n_trajectories'], -1, 1, 1)
    # when norm <= max_norm, scalings = 1, otherwise, scales to max_norm
    scalings = np.minimum(1, max_norm/x_norms)

    all_xs = all_xs * scalings

    a = 0.1 * 256 / (algoparams['n_trajectories'])
    if a > 1: a = 1

    # somehow i was not able to get it to work with a single call and data in matrix form.
    # ts_expanded = np.tile(all_ts[:, None], algoparams['n_trajectories'])
    # ipdb.set_trace()
    # ax0.plot(all_ts, all_xs[:, :, 0].squeeze().T, all_xs[:, :, 1].squeeze().T, color='black', alpha=a)
    # ax1.plot(all_xs[:, :, 0], all_xs[:, :, 1], all_values, color='black', alpha=a)

    # neat hack - if we set the xs to nan where resamplings occurred, it breaks up the plot and does not
    # connect the trajectories before and after resampling
    all_xs = all_xs.at[where_resampled, :, :].set(np.nan)

    # so now we go back to the stone age
    for i in range(algoparams['n_trajectories']):
        ax0.plot(all_ts, all_xs[i, :, 0].squeeze(), all_xs[i, :, 1].squeeze(), color='black', alpha=a)
        # ax1.plot(all_xs[i, :, 0], all_xs[i, :, 1], all_values[i], color='blue', alpha=a)

    ax0.set_xlabel('t')
    ax0.set_ylabel('x_0')
    ax0.set_zlabel('x_1')

    ax0.set_ylim([-5, 5])
    ax0.set_zlim([-2, 2])

    # ax1.set_xlabel('x_0')
    # ax1.set_ylabel('x_1')
    # ax1.set_zlabel('value')

    # pl.show()

    fig, ax = pl.subplots()

    # make a figure with a slider to get a feel of the distribution of particles
    # over the state-space at some (adjustable) time.

    t_plot = T

    # make the figure
    idx_plot = np.argmin(np.abs(all_ts - t_plot))
    sc = pl.scatter(all_xs[:, idx_plot, 0].squeeze(), all_xs[:, idx_plot, 1].squeeze())

    fig.subplots_adjust(bottom=.25)
    ax_time = fig.add_axes([.25, .1, .65, .03])
    time_slider = matplotlib.widgets.Slider(
        ax=ax_time,
        label='time [s]',
        valmin=0,
        valmax=T,
        valinit=t_plot,
    )

    def update(val):
        t_plot = time_slider.val
        idx_plot = np.argmin(np.abs(all_ts - t_plot))

        x = all_xs[:, idx_plot, 0].squeeze()
        y = all_xs[:, idx_plot, 1].squeeze()

        sc.set_offsets(np.c_[x, y])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    pl.show()


def plot_1d(all_sols, all_ts, where_resampled, problemparams, algoparams):

    # just one figure, with the state trajectories, maybe colored according
    # to value?

    fig = pl.figure(figsize=(8, 3))

    nx = problemparams['nx']
    T = problemparams['T']
    all_xs       = all_sols[:, :,  0:nx,    :]
    all_costates = all_sols[:, :,  nx:2*nx, :]
    all_values   = all_sols[:, :, -1,       :]

    max_norm = 1e5

    # -1 becomes number of time steps
    # ipdb.set_trace()
    x_norms = np.linalg.norm(all_xs, axis=2).reshape(algoparams['n_trajectories'], -1, 1, 1)
    # when norm <= max_norm, scalings = 1, otherwise, scales to max_norm
    scalings = np.minimum(1, max_norm/x_norms)

    all_xs = all_xs * scalings

    a = 0.1 * 256 / (algoparams['n_trajectories'])
    if a > 1: a = 1

    # neat hack - if we set the xs to nan where resamplings occurred, it breaks up the plot and does not
    # connect the trajectories before and after resampling
    all_xs = all_xs.at[where_resampled, :, :].set(np.nan)

    xs_plot = all_xs.squeeze().T
    costates_plot = all_costates.squeeze().T
    pl.plot(all_ts, xs_plot, color='b', label='state')
    pl.plot(all_ts, costates_plot, color='g', label='costate')
    pl.legend()

    pl.xlabel('t')
    pl.ylabel('x')

    pl.show()


def characteristics_experiment_simple():

    # simple control system. double integrator with friction term.
    def f(t, x, u):
        # p' = v
        # v' = f
        # f = -v**3 + u
        # clip so we don't have finite escape time when integrating backwards
        v_cubed = np.clip((np.array([[0, 1]]) @ x)**3, -10, 10)
        return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u - v_cubed)

    def l(t, x, u):
        Q = np.eye(2)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 1 * np.eye(2)
        return (x.T @ Qf @ x).reshape()


    problemparams = {
            'f': f,
            'l': l,
            'h': h,
            'T': 4,
            'nx': 2,
            'nu': 1
    }

    x_sample_scale = 1 * np.eye(2)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # resample when the mahalanobis distance (to the sampling distribution) is larger than this.
    resample_mahalanobis_dist = 2

    # to calculate mahalanobis dist: d = sqrt(x.T inv(Σ) x) -- basically the argument to exp(.) in the pdf.
    # x domain defined as X = {x in R^n: x.T @ Q_x @ x <= 1}
    # so we want to find Q_x to achieve
    #     sqrt(x.T inv(Σ) x) <= d_max  <=>  x.T @ Q_x @ x <= 1
    #                                  <=>  sqrt(x.T @ Q_x @ x) <= 1                       (bc. sqrt monotonous)
    #                                  <=>  sqrt(x.T @ Q_x @ x) d_max <= d_max             (scaling LHS changes nothing)
    #                                  <=>  sqrt(x.T @ Q_x @ x * d_max**2) <= d_max        (d_max into sqrt)
    #                                  <=>  sqrt(x.T @ (Q_x * d_max**2) @ x) <= d_max      (reordering)
    # these are now basically the same, with inv(Σ) = Q_x * d_max**2 or Q_x = inv(Σ) / d_max**2
    # why could I not simply guess this?
    Q_x = np.linalg.inv(x_sample_cov) / resample_mahalanobis_dist**2

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.
    algoparams = {
            'n_trajectories': 512,
            'dt': 1/256,
            'resample_interval': 1/4,
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,

            'nn_layersizes': (32, 32, 32),
            'batchsize': 64,
            'N_epochs': 10,
    }

    # problemparams are parameters of the problem itself
    # algoparams contains the 'implementation details'
    output = hjb_characteristics_solver(problemparams, algoparams)

    plot_2d(*output, problemparams, algoparams)

def characteristics_experiment_even_simpler():
    # 1d test case.

    # SINGLE integrator.
    def f(t, x, u):
        return u

    def l(t, x, u):
        Q = np.eye(1)
        R = np.eye(1)
        return x.T @ Q @ x + u.T @ R @ u

    def h(x):
        Qf = 1 * np.eye(1)
        return (x.T @ Qf @ x).reshape()

    problemparams = {
            'f': f,
            'l': l,
            'h': h,
            'T': 5,
            'nx': 1,
            'nu': 1
    }

    x_sample_scale = 1 * np.eye(1)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # resample when the mahalanobis distance (to the sampling distribution) is larger than this.
    resample_mahalanobis_dist = 2
    Q_x = np.linalg.inv(x_sample_cov) / resample_mahalanobis_dist**2

    # IDEA for simpler parameterisation. same matrix for x sample cov and x domain.
    # then just say we resample when x is outside of like 4 sigma or similar.
    algoparams = {
            'n_trajectories': 256,
            'dt': 1/256,
            'resample_interval': 1/16,
            'x_sample_cov': x_sample_cov,
            'x_domain': Q_x,
            'nn_layersizes': (128, 128, 128),
            'batchsize': 128,
            'N_epochs': 100,
    }

    # problemparams are parameters of the problem itself
    # algoparams contains the 'implementation details'
    all_sols, all_ts, where_resampled = hjb_characteristics_solver(problemparams, algoparams)

    plot_1d(all_sols, all_ts, where_resampled, problemparams, algoparams)



if __name__ == '__main__':

    # characteristics_experiment_even_simpler()
    characteristics_experiment_simple()
