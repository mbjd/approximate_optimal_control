import jax
import jax.numpy as np
import ipdb

from functools import partial

from equinox import filter_jit


# filter_jit from equinox is really cool. it will treat jax arrays dynamically (so jit
# only re-runs when they have new shape), whereas other types (ints, bools, strings, ...)
# will trigger re-jitting when changed, as if they were marked as static_argnums/names in
# jax.jit.

# this allows us to have jax arrays in algo_params without too much hassle.

@filter_jit
def resample(ys, t, nn_apply_fct, nn_params, algo_params, key):

    # all the ys that have left the interesting region, we want to put back into it.

    # this is the first mockup of the 'actual' resampling function.
    # It will not only put points in new locations, but also put in the correct value
    # and costate information, based on the value function approximation given by
    # (nn_apply_fct, nn_params). nn_apply_fct should be the apply function of the nn
    # object itself, NOT the wrapper which is mostly used here. This is to separate
    # NN architecture info (-> in nn object) from the parameters nn_params, which
    # otherwise would have been in the same nn_wrapper object, perventing jit.

    # variable naming scheme:
    # old      : before resampling, whatever came out of the last pontryagin integration step
    # resampled: ALL resampled variables, whether we want them or not
    # new      : resampled where we want to resample, old where we don't


    # does this work with jit?
    nx = (ys.shape[1] - 1) // 2

    old_xs = ys[:, 0:nx, :]
    old_V_grads = ys[:, nx:2*nx, :]
    old_Vs = ys[:, -1:, :]  # range to keep shapes consistent


    # depending on the resampling condition, set the resampling mask to 1 for the trajectories
    # we want to resample.
    if algo_params['resample_type'] == 'minimal':

        # parameterise x domain as ellipse: X = {x in R^n: x.T @ Q_x @ x <= 1}
        # 'x_domain': Q_x,
        ellipse_membership_fct = lambda x: x.T @ algo_params['x_domain'] @ x
        ellipse_memberships = jax.vmap(ellipse_membership_fct)(old_xs)

        # resample where this mask is 1.
        resample_mask = (ellipse_memberships > 1).astype(np.float32)

    elif algo_params['resample_type'] == 'all':
        # additional shape for compatibility with column vector style
        resample_mask = np.ones((algo_params['n_trajectories'], 1, 1))

    else:
        raise RuntimeError(f'Invalid resampling type "{algo_params["resampling_type"]}"')


    # keep old sample where that mask is 1.
    not_resample_mask = 1 - resample_mask

    # just resample ALL the xs anyway, and later let the mask decide which to keep.
    # resampled_xs = scale @ jax.random.normal(key, (algo_params['n_trajectories'], nx, 1))
    resampled_xs = jax.random.multivariate_normal(
            key,
            mean=np.zeros(nx,),
            cov=algo_params['x_sample_cov'],
            shape=(algo_params['n_trajectories'],)
    ).reshape(algo_params['n_trajectories'], nx, 1)

    new_xs = resample_mask * resampled_xs + \
            not_resample_mask * old_xs

    # every trajectory is at the same time...
    ts = t * np.ones((algo_params['n_trajectories'], 1))

    # input vector of the NN
    # squeeze removes the last 1. (n_traj, nx, 1) -> (n_traj, nx).
    # because the NN doesn't work with explicit (n, 1) shaped column vectors
    resampled_ts_and_xs = np.concatenate([ts, resampled_xs.squeeze()], axis=1)

    # these value gradients are with respect to (t, x). so we also have time
    # gradients, not only the ones w.r.t. state. can we do something with those as well?

    # we have d/dt V(t, x(t)) = V_t(t, x(t)) + V_x(t, x(t)).T d/dt x(t)
    # or d/dt V(t, x(t)) = [V_t(t, x(t))  V_x(t, x(t))] @ [1  d/dt x(t)]
    # this is basically a linear system which (probably?) we can solve for
    # V_t(t, x(t)) (partial derivative in first argument, not total time derivative)
    # so, TODO, also try to incorporate time gradient during training.

    # do some jax magic
    V_fct = lambda z: nn_apply_fct(nn_params, z).reshape()
    V_fct_and_grad = jax.value_and_grad(V_fct)
    V_fct_and_grad_batch = jax.vmap(V_fct_and_grad)

    resampled_Vs, resampled_V_grads = V_fct_and_grad_batch(resampled_ts_and_xs)

    # weird [None] tricks so it doesn't do any wrong broadcasting
    # V grads [1:] because the NN output also is w.r.t. t which at the moment we don't need
    new_Vs = resample_mask * resampled_Vs[:, None, None] + not_resample_mask * old_Vs
    new_V_grads = resample_mask * resampled_V_grads[:, 1:, None] + not_resample_mask * old_V_grads

    # circumventing jax's immutable objects.
    # if docs are to be believed, after jit this will do an efficient in place update.
    ys = ys.at[:, 0:nx, :].set(new_xs)
    ys = ys.at[:, nx:2*nx, :].set(new_V_grads)
    ys = ys.at[:, -1:, :].set(new_Vs)

    return ys, resample_mask




def sol_array_to_train_data(all_sols, all_ts, resampling_i, n_timesteps, algo_params):

    # basically, this takes the data in all_sols and all_ts, slices out the
    # relevant part between the current resampling (resampling_i) and the
    # lookback horizon, and reshapes it accordingly so the NN can handle it.

    # training data terminology:

    # input = [t, state], label = V

    # reshape - during training we do not care which is the time and which the
    # trajectory index, they are all just pairs of ((t, x), V) regardless

    # before we have a shape (n_traj, n_time, 2*nx+1, 1).
    # here we remove the last dimension - somehow it makes NN stuff easier.

    nx = (all_sols.shape[2]-1) // 2

    if algo_params['nn_train_lookback'] == np.inf:
        # use ALL the training data available
        train_time_idx = np.arange(resampling_i, n_timesteps)
    else:
        lookback_indices = int(algo_params['nn_train_lookback'] / algo_params['dt'])
        upper_train_time_idx = resampling_i + lookback_indices
        if upper_train_time_idx > n_timesteps:
            # If available data <= lookback, just use whatever we have.
            upper_train_time_idx = n_timesteps

        train_time_idx = np.arange(resampling_i, upper_train_time_idx)

    train_states = all_sols[:, train_time_idx, 0:nx, :].reshape(-1, nx)

    # make new axes to match train_state
    train_ts = all_ts[None, train_time_idx, None]
    # repeat for all trajectories (axis 0), which all have the same ts.
    train_ts = np.repeat(train_ts, algo_params['n_trajectories'], axis=0)
    # flatten the same way as train_states
    train_ts = train_ts.reshape(-1, 1)

    # assemble into main data arrays
    # these are of shape (n_datpts, n_inputfeatures) and (n_datapts, 1)
    train_inputs = np.concatenate([train_ts, train_states], axis=1)
    train_labels = all_sols[:, train_time_idx, 2*nx, :].reshape(-1, 1)

    return train_inputs, train_labels

