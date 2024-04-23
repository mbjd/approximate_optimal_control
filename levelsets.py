import jax
import jax.numpy as np
import numpy as onp
import diffrax
import equinox

import nn_utils
import plotting_utils
import pontryagin_utils
import ddp_optimizer
import visualiser
import ct_basics
from misc import *

import matplotlib
import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf

import ipdb
import time
import tqdm
from operator import itemgetter



# many of these functions probably don't work. they were hacked together within the testbed function
# and depend on some variables there. if needed again, put back there or include variables as proper arguments.


def plot_calibration(all_ys, pred_v_means, pred_v_stds):

    # calibration plot = plot of true frequency of data in each confidence band
    # vs predicted frequency.

    # although it may be questioned if this plot is at all relevant for us. we
    # basically have deterministic data (except ODE solver error) and just want
    # to distinguish between "inside" the known set and "outside" of it.

    sigmas = np.linspace(-5, 5, 300)
    predicted_fractions = jax.scipy.stats.norm.cdf(sigmas)

    # the error between predicted and label, scaled by the std dev.
    # if model is well calibrated, this should be normally distributed.
    normalised_predictions = (pred_v_means.flatten() - all_ys['v'].flatten()) / pred_v_stds.flatten()

    where_usable = ~np.isnan(normalised_predictions)
    normalised_predictions = normalised_predictions[where_usable]

    observed_fractions = np.mean(normalised_predictions[:, None] < sigmas, axis=0)

    pl.plot(predicted_fractions, observed_fractions, '.-')
    pl.plot([0, 1], [0, 1], '--', c='black', alpha=.1)
    pl.xlabel('predicted fraction')
    pl.ylabel('observed fraction')






def plot_distributions(ys_n):

    pl.figure()

    # try to display the distribution of the different (normalised) variables.
    # plot essentially the cdf of them.

    norm_xs = np.linspace(-5, 5, 201)
    norm_ys = jax.scipy.stats.norm.cdf(norm_xs)

    def plot_data(arr, label):

        # plot cdf of all data points. can be of any shape -- first axis
        # is assumed to

        # +1 for NaN we are adding
        N = arr.shape[0] + 1
        pt_shape = arr.shape[1:]

        # sort and add a nan to each member of point.
        sorted_arr = arr.sort(axis=0)
        nan_pt = arr[0:1] * np.nan  # range to keep leading axis
        sorted_arr = np.concatenate([sorted_arr, nan_pt], axis=0)

        ys = np.linspace(np.zeros(pt_shape), np.ones(pt_shape), N)
        pl.plot(sorted_arr.ravel(order='F'), ys.ravel(order='F'), alpha=.5, label=label)
        pl.plot(norm_xs, norm_ys, c='black', linestyle='--', alpha=.5)
        pl.legend()



    pl.subplot(221)
    plot_data(ys_n['x'], 'x entries')
    pl.subplot(222)
    plot_data(ys_n['v'], 'v')
    pl.subplot(223)
    plot_data(ys_n['vx'], 'vx entries')
    if 'vxx' in ys_n:
        pl.subplot(224)
        plot_data(ys_n['vxx'], 'vxx entries')


def sobolev_weight_gridsearch():

    # perform a 2d grid search over a range of vx and vxx sobolev weights.
    # v weight = 1 always -- because we normalise the weights this explores all degrees of freedom.
    # from initial experiments we guess the rough range where interesting things happen

    # we save the raw loss terms for v, vx, vxx which are the same regardless of weights.
    # not exactly sure what our end goal to optimise for should be though.

    N = 16

    vx_weights = np.logspace(-2, 2, N)
    vxx_weights = np.logspace(-2, 2, N)

    all_testlossterms = np.zeros((N, N, 3))

    key = jax.random.PRNGKey(0)

    for i, vx_w in enumerate(vx_weights):
        for j, vxx_w in enumerate(vxx_weights):

            # train nn with those hyperparams.
            algo_params['nn_sobolev_weights'] = np.array([1., vx_w, vxx_w])
            init_key, train_key, key = jax.random.split(key, 3)
            params_sobolev = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))
            params_sobolev, oups_sobolev = v_nn.train_sobolev(train_key, ys_n, params_sobolev, algo_params)

            # evaluate & store test loss.
            _, test_lossterms = v_nn.sobolev_loss_batch_mean(key, params_sobolev, test_ys_n, algo_params)
            all_testlossterms = all_testlossterms.at[i, j, :].set(test_lossterms)

            # ipdb.set_trace()

    norm = matplotlib.colors.LogNorm(all_testlossterms.min(), all_testlossterms.max())

    ax = pl.subplot(131)
    pl.imshow(all_testlossterms[:, :, 0], norm=norm)
    ax.set_title('v test loss')
    ax = pl.subplot(132)
    pl.imshow(all_testlossterms[:, :, 1], norm=norm)
    ax.set_title('vx test loss')
    ax = pl.subplot(133)
    pl.imshow(all_testlossterms[:, :, 2], norm=norm)
    ax.set_title('vxx test loss')
    pl.colorbar()
    pl.show()

    ipdb.set_trace()

    # middle of the pack seems to look nicest here... so both 1.36???


def vxx_weight_sweep():
    # new sobolev training method.
    pl.rcParams['figure.figsize'] = (16, 9)
    vxx_weights = np.concatenate([np.zeros(1,), np.logspace(-1, 5, 128)])
    hessian_rnds = np.zeros_like(vxx_weights)
    final_training_errs = np.zeros((vxx_weights.shape[0], 3))
    test_errs = np.zeros((vxx_weights.shape[0], 3))

    key = jax.random.PRNGKey(0)

    for i, vxx_weight in tqdm.tqdm(enumerate(vxx_weights)):

        algo_params['nn_sobolev_weights'] = algo_params['nn_sobolev_weights'].at[2].set(vxx_weight)

        init_key, key = jax.random.split(key)
        params_sobolev = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))

        train_key, key = jax.random.split(key)
        params_sobolev, oups_sobolev = v_nn.train_sobolev(train_key, ys_n, params_sobolev, algo_params)

        # the value function back in the "unnormalised" domain, ie. the actual state space.
        v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
        # and its hessian at 0 just as a sanity check.
        hess_vnn_unnormalised = lambda params, x: jax.hessian(v_nn_unnormalised, argnums=1)(params, x).squeeze()
        H0 = hess_vnn_unnormalised(params_sobolev, np.zeros(6,))

        # compute some statistics :)
        hess_rnd = rnd(H0, P_lqr)
        loss_means = oups_sobolev['loss_terms'][-100:, :].mean(axis=0)
        # print(hess_rnd)
        hessian_rnds = hessian_rnds.at[i].set(hess_rnd)
        final_training_errs = final_training_errs.at[i, :].set(loss_means)
        _, test_lossterms = v_nn.sobolev_loss_batch_mean(key, params_sobolev, test_ys_n, algo_params)
        test_errs = test_errs.at[i, :].set(test_lossterms)

        '''
        pl.figure(f'vxx weight: {vxx_weight:.8f}')
        pl.suptitle(f'vxx weight: {vxx_weight:.8f}')
        pl.loglog(oups_sobolev['loss_terms'], label=('v', 'vx', 'vxx'), alpha=.5)
        pl.grid('on')
        pl.ylim([1e-7, 1e1])
        pl.legend()
        figpath = f'./tmp/losses_{i:04d}_{vxx_weight:.3f}.png'
        pl.savefig(figpath)
        print(f'saved "{figpath}"')
        pl.close('all')
        '''


# look at normalised data.
def data_normalisation_experiment():

    N = 100
    vs = np.logspace(0, 3, N)
    for i, v_k in enumerate(vs):

        # extract corresponding data points for NN fitting.
        # this is a multi dim bool index! indexing a multidim array with it effectively flattens it.
        bool_train_idx = sols_orig.ys['v'] < v_k

        # value "band" instead.
        bool_train_idx = np.logical_and(sols_orig.ys['v'] < v_k, sols_orig.ys['v'] > 0.1)

        vs_flat = sols_orig.ys['v'].flatten()
        argsort_vs_flat = np.argsort(vs_flat)

        pl.plot()

        all_ys = jax.tree_util.tree_map(lambda node: node[bool_train_idx], sols_orig.ys)

        # split into train/test set.

        train_ys, test_ys = nn_utils.train_test_split(all_ys)

        print(f'dataset size: {bool_train_idx.sum()} points (= {bool_train_idx.mean()*100:.2f}%)')
        n_data = count_floats(train_ys)
        print(f'corresponding to {n_data} degrees of freedom')

        v_nn = nn_utils.nn_wrapper(
            input_dim=problem_params['nx'],
            layer_dims=algo_params['nn_layerdims'],
            output_dim=1
        )


        normaliser = nn_utils.data_normaliser(train_ys)

        ys_n = normaliser.normalise_all_dict(train_ys)
        test_ys_n = normaliser.normalise_all_dict(test_ys)

        means = jax.tree_util.tree_map(lambda n: np.expand_dims(n.mean(axis=0), 0), ys_n)
        stds  = jax.tree_util.tree_map(lambda n: np.expand_dims(n.std(axis=0), 0), ys_n)
        if i==0:
            all_means = means
            all_stds = stds
        else:
            concat0 = lambda a, b: np.concatenate([a, b], axis=0)
            all_means = jax.tree_util.tree_map(concat0, all_means, means)
            all_stds = jax.tree_util.tree_map(concat0, all_stds, stds)


    # append a nan to break up plot lines.
    all_means = jax.tree_util.tree_map(concat0, all_means, jax.tree_util.tree_map(lambda n: n * np.nan, means))
    all_stds = jax.tree_util.tree_map(concat0, all_stds, jax.tree_util.tree_map(lambda n: n * np.nan, stds))
    N = N+1  # because of the NaN at the end



    pl.figure('data normalisation plot')

    nx = problem_params['nx']
    xs_one = np.arange(N)
    xs_nx = np.arange(N * nx) % (N)
    xs_nx_sq = np.arange(N * nx**2) % N

    plot_wrt_v = True
    if plot_wrt_v:
        xs_one = vs[xs_one]
        xs_nx = vs[xs_nx]
        xs_nx_sq = vs[xs_nx_sq]

    pl.subplot(211)
    pl.plot(xs_one, all_means['v'], label='v mean', alpha=.5)

    vx_means = all_means['vx'].T.flatten()
    pl.plot(xs_nx, vx_means, label='vx mean', alpha=.5)

    vxx_means = all_means['vxx'].swapaxes(0, 2).flatten()
    pl.plot(xs_nx_sq, vxx_means, label='vxx mean', alpha=.5)
    pl.legend()

    pl.subplot(212)
    pl.plot(xs_one, all_stds['v'], label='v std', alpha=.5)

    vx_means = all_stds['vx'].T.flatten()
    pl.plot(xs_nx, vx_means, label='vx std', alpha=.5)

    vxx_means = all_stds['vxx'].swapaxes(0, 2).flatten()
    pl.plot(xs_nx_sq, vxx_means, label='vxx std', alpha=.5)
    pl.legend()
    pl.show()


def plot_taylor_sanitycheck(sol, problem_params):

    # possibly this stopped working when i moved it outside of the testbed function
    # but hopefully we won't be needing this so much anyway.
    # see if the lambda = vx and S = vxx are remotely plausible.

    pl.figure(f'taylor sanity check')

    # does a part of both of the above functions plus the retrieval of the solutions.
    # plot v(x(t)) alone
    interp_ts = np.linspace(sol.t0, sol.t1, 1000)
    interp_ys = jax.vmap(sol.evaluate)(interp_ts)
    pl.plot(sol.ys['t'], sol.ys['v'], linestyle='', marker='.', color='C0', alpha=0.4)
    pl.plot(interp_ts, interp_ys['v'], linestyle='--', label='v(x(t))', color='C0', alpha=0.4)

    us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(sol.ys['x'], sol.ys['vx'], problem_params)
    fs = jax.vmap(problem_params['f'], in_axes=(0, 0))(sol.ys['x'], us)

    # the total time derivative of v we're after
    v_ts = jax.vmap(np.dot)(sol.ys['vx'], fs)

    # for each of the derivatives, plot a small line.

    def line_params(t, v, v_t):

        line_len = 0.1
        diffvec_unscaled = np.array([1, v_t])
        diffvec = line_len * diffvec_unscaled / np.linalg.norm(diffvec_unscaled)

        # nan in between to break up lines.
        xs = np.array([t-diffvec[0], t+diffvec[0], np.nan])
        ys = np.array([v-diffvec[1], v+diffvec[1], np.nan])
        return xs, ys

    xs, ys = jax.vmap(line_params)(sol.ys['t'], sol.ys['v'], v_ts)

    pl.plot(xs.flatten(), ys.flatten(), label='d/dt v(x(t))', color='C1', alpha=0.5)


    # now the hessians, def the hardest part...

    def line_params_hessian(ode_state, f_value):

        t_len = 0.02
        n = 20

        dts = np.concatenate([np.linspace(-t_len, t_len, n), np.array([np.nan])])
        dxs = np.vstack([np.linspace(-f_value * t_len, +f_value * t_len, n), np.nan * np.ones(problem_params['nx'])])

        xs = ode_state['x'] + dxs
        ts = ode_state['t'] + dts

        vs_taylor = jax.vmap(lambda dx: ode_state['v'] + ode_state['vx'] @ dx + 0.5 * dx.T @ ode_state['vxx'] @ dx)(dxs)

        return ts, vs_taylor

    def line_params_twice_hessian(ode_state, f_value):

        t_len = 0.02
        N = 20

        dts = np.concatenate([np.linspace(-t_len, t_len, N), np.array([np.nan])])
        dxs = np.vstack([np.linspace(-f_value * t_len, +f_value * t_len, N), np.nan * np.ones(problem_params['nx'])])

        xs = ode_state['x'] + dxs
        ts = ode_state['t'] + dts

        vs_taylor = jax.vmap(lambda dx: ode_state['v'] + ode_state['vx'] @ dx + 0.5 * 2 * dx.T @ ode_state['vxx'] @ dx)(dxs)

        return ts, vs_taylor

    ts, vs = jax.vmap(line_params_hessian, in_axes=(0, 0))(sol.ys, fs)

    pl.plot(ts.flatten(), vs.flatten(), alpha=.5, color='C2', label='hessian')

    # also interesting:
    pl.figure()
    idx = 20
    state_idx = jax.tree_util.tree_map(itemgetter(idx), sol.ys)
    ts, vs = line_params_hessian(state_idx, fs[idx])
    pl.plot(ts, vs, label='v taylor')
    ts, vstwice = line_params_twice_hessian(state_idx, fs[idx])
    pl.plot(ts, vstwice, label='v taylor but twice hessian')
    pl.scatter(state_idx['t'], state_idx['v'], color='C0')
    pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'], label='v sol')
    pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'] - vs, label='diff')
    pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'] - vstwice, label='diff with twice hessian')


    pl.legend()
    # ipdb.set_trace()


def debug_nan_sol(sols_orig, problem_params, algo_params):

    if not 'vxx' in sols_orig.ys:
        print('debug_nan_sol highlights possible issues in the vxx trajectory.')
        print('however, there is no vxx here. returning without doing anything. ')
        return

    # long debugging session.
    solve_backward, f_extended = pontryagin_utils.define_backward_solver(
        problem_params, algo_params
    )

    # conclusion: if NaNs pop up or the vxx terms become unreasonably large,
    # try decreasing dtmin a bit. fixed everything in this case.

    # just the first one
    all_nan_idx = np.where(np.isnan(sols_orig.ys['vxx']).any(axis=(1,2,3)))[0]
    nan_idx = all_nan_idx[0]
    print(f'NaN in solutions {all_nan_idx}, plotting details for {nan_idx}')
    bad_sol = jax.tree_util.tree_map(itemgetter(nan_idx), sols_orig)

    # serialise solution to analyse after switching to 64 bit.
    # import flax
    # bs = flax.serialization.msgpack_serialize(bad_sol.ys)
    # f = open('tmp/bad_ys.msgpack', 'wb')
    # f.write(bs)
    # f.close()

    # f = open('tmp/bad_ys.msgpack', 'rb')
    # bs = f.read()
    # bad_ys = flax.serialization.msgpack_restore(bs)

    '''
    def plot_badsol_from_idx(idx):

        y = jtm(itemgetter(idx), bad_sol.ys)

        # recreate the solution in 64 bit precision.
        newsol = solve_backward(y)

        plotting_utils.plot_sol(newsol, problem_params)

    for idx in (50, 100, 200, 400):
        pl.figure(idx)
        plot_badsol_from_idx(idx)

    '''

    '''
    def estimate_vxx_lstsq(dx_size, t_eval=-4.7):

        # perturb state at index 50 slightly and see what happens.
        # evaluate the resulting solutions at t_eval and estimate vxx there.

        dxs = jax.random.normal(jax.random.PRNGKey(0), shape=(200, 6)) * dx_size

        def solve_perturbed(y, dx):

            y_perturbed = {
                't': y['t'],
                'x': y['x'] + dx,
                'v': y['v'] + y['vx'] @ dx,
                'vx': y['vx'] + y['vxx'] @ dx,
                'vxx': y['vxx']  # no info here.
            }

            newsol = solve_backward(y_perturbed)

            return newsol

        y = jtm(itemgetter(50), bad_sol.ys)
        newsols = jax.vmap(solve_perturbed, in_axes=(None, 0))(y, dxs)


        # for new sols (which start at y with t=0...)
        t_eval_new = t_eval - y['t']

        xs = jax.vmap(lambda sol: sol.evaluate(t_eval_new)['x'])(newsols)
        vxs = jax.vmap(lambda sol: sol.evaluate(t_eval_new)['vx'])(newsols)

        x_mean = xs.mean(axis=0)
        vx_mean = vxs.mean(axis=0)

        # try to estimate the hessian (= jacobian of map x -> vx) from data.
        # taylor expansion: vx(x + dx) \approx vx(x) + vxx dx
        # transposing     : vx.T(x + dx) \approx vx.T(x) + dx.T vxx.T
        # subtracting mean: vx.T(x+dx) - vx.T(x) \approx dx.T vxx
        # for lstsq         b                            A    x
        # so, lstsq(dxs as stacked row vecs, vxs-mean stacked as row vecs)
        # should do the trick.
        vxx_est, _, _, _ = np.linalg.lstsq(dxs, vxs - vxs.mean(axis=0)[None, :])

        return vxx_est

    t_eval = -4.7
    vxx_sol = bad_sol.evaluate(t_eval)['vxx']
    vxx_est = estimate_vxx_lstsq(.001, t_eval=t_eval)
    # wtf these don't seem to be similar in any way...
    # is it because the hessian also changes "quickly" and is thus not well
    # represented by the "discrete difference" of vx?
    '''




    ipdb.set_trace()


    # start sol with higher precision from a state close to the last one.
    restart_state_idx = bad_sol.stats['num_accepted_steps'].item() - 5

    # not really needed, still fails, rhs actually does return very high values :(
    # restart_y = jax.tree_util.tree_map(itemgetter(restart_state_idx), bad_sol.ys)

    # algo_params_tight = algo_params.copy()
    # algo_params_tight['pontryagin_solver_atol'] = 1e-7
    # algo_params_tight['pontryagin_solver_rtol'] = 1e-7

    # # not sure if this still works, changed function signature twice since running it
    # sol_tight = solve_backward(restart_y)

    # # evaluate the right hand side again to see where it produces shit.
    # rhs_evals = jax.vmap(f_extended, in_axes=(0, 0, None))(sol_tight.ts, sol_tight.ys, None)
    rhs_evals_orig = jax.vmap(f_extended, in_axes=(0, 0, None))(bad_sol.ts, bad_sol.ys, None)

    rhs_evals, aux_outputs = jax.vmap(f_extended, in_axes=(0, 0, None))(bad_sol.ts, bad_sol.ys, 'debug')

    pl.figure()
    ax = pl.subplot(211)
    steps = bad_sol.ts.shape[0]
    pl.plot(bad_sol.ts, bad_sol.ys['vxx'].reshape(steps, -1), '.-', c='C0', alpha=.5)
    pl.ylabel('vxx trajectory components')
    pl.subplot(212, sharex=ax)
    pl.plot(bad_sol.ts, rhs_evals['vxx'].reshape(steps, -1), '.-', c='C0', alpha=.5)
    pl.ylabel('vxx rhs components')

    pl.figure()
    plotting_utils.plot_sol(bad_sol, problem_params)



    # another plot.
    # plot v on the x axis against ||vxx|| on the y axis.
    all_vs = sols_orig.ys['v'].reshape(-1)
    all_vxxs = sols_orig.ys['vxx'].reshape((-1, 6, 6))
    pl.figure()
    pl.loglog(all_vs, np.linalg.norm(all_vxxs, axis=(1,2)), alpha=.2)

    pl.show()


    ipdb.set_trace()


    # even in the original one we see clearly a spike at the end, where it goes from
    # about 5e3 up to 1e8 in 3 steps.




def main(problem_params, algo_params):
    pass

def testbed(problem_params, algo_params):

    # possibly cleaner implementation of this.
    # idea: learn V(x) for some level set V(x) <= v_k.
    # once we have that, increase v_k.

    key = jax.random.PRNGKey(1)

    # find terminal LQR controller and value function.
    # ultimately generate the function unitsphere_to_dXf

    # in manifold case, this is still something which we should do purely
    # on the tangent space...
    if problem_params['m'] is not None:
        K_lqr, P_lqr, Proj_tangent = pontryagin_utils.get_terminal_lqr(problem_params, return_tangent_projection=True)

        # find the LQR controller in tangent space basis
        # essentially undo what we did inside the lqr function...
        P_lqr_tangent = Proj_tangent @ P_lqr @ Proj_tangent.T
        K_lqr_tangent = K_lqr @ Proj_tangent.T

        # here we can do cholesky just fine
        L_lqr_tangent = np.linalg.cholesky(P_lqr_tangent)

        assert rnd(L_lqr_tangent @ L_lqr_tangent.T, P_lqr_tangent) < 1e-6, 'cholesky decomposition wrong or inaccurate'

        # same as below, except we project the (ambient space) point to the tangent space
        # and then back. this has no affine part though and the nonzero x_eq is disregarded,
        # so we change it to a lambda function which includes that.
        # and finally, we project back to the manifold using the provided function.
        unitsphere_to_dXf_linear = Proj_tangent.T @ np.linalg.inv(L_lqr_tangent) @ Proj_tangent * np.sqrt(problem_params['V_f']) * np.sqrt(2)
        unitsphere_to_dXf = lambda x: problem_params['project_M'](problem_params['x_eq'] + x.T @ unitsphere_to_dXf_linear)


    else:
        # state space R^n
        K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

        # find a matrix mapping from the unit circle to the value level set
        # previously done with eigendecomposition -> sqrt of eigenvalues.
        # with the cholesky decomp the trajectories look about the same
        # qualitatively. it is nicer so we'll keep that.
        # cholesky decomposition says: P = L L.T but not L.T L
        L_lqr = np.linalg.cholesky(P_lqr)


        assert rnd(L_lqr @ L_lqr.T, P_lqr) < 1e-6, 'cholesky decomposition wrong or inaccurate'

        # linear map from the hypersphere to the ellipse V_lqr(x) == V_f
        unitsphere_to_dXf = lambda x: x.T @ np.linalg.inv(L_lqr) * np.sqrt(problem_params['V_f']) * np.sqrt(2)


    # purely random ass points for initial batch of trajectories.
    normal_pts = jax.random.normal(key, shape=(algo_params['initial_batchsize'], problem_params['nx']))
    unitsphere_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]
    xfs = jax.vmap(unitsphere_to_dXf)(unitsphere_pts)

    # test if it worked
    V_f = lambda x: 0.5 * x.T @ P_lqr @ x
    vfs = jax.vmap(V_f)(xfs)

    # this is not that precise in the manifold case.
    # nevertheless we continue and assume that for small enough V_f it will still kind of work :)
    # assert np.allclose(vfs, problem_params['V_f']), 'wrong terminal value...'

    pl.rcParams['figure.figsize'] = (16, 10)


    solve_backward, f_extended = pontryagin_utils.define_backward_solver(
        problem_params, algo_params
    )

    def solve_backward_lqr(x_f, algo_params):

        # P_lqr = hessian of value fct.
        # everything else follows from usual differentiation rules.

        v_f = 0.5 * x_f.T @ P_lqr @ x_f
        vx_f = P_lqr @ x_f

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        if algo_params['pontryagin_solver_vxx']:
            vxx_f = P_lqr
            state_f['vxx'] = vxx_f

        return solve_backward(state_f, v_upper=500)

    sols_orig = jax.vmap(solve_backward_lqr, in_axes=(0, None))(xfs, algo_params)


    def find_min_l(ys, v_lower, v_upper, problem_params):

        # find the smallest value of l(x, u) in the given value band
        # in the dataset. brute force -- calculates every l(x, u).

        # in principle we should be able to find this info based on
        # what we already calculated during the ODE solving, or even
        # keep some "running min" that is updated at basically no cost.
        # but this on the other hand is much simpler implementation wise.

        def l_of_y(y):
            x = y['x']
            vx = y['vx']
            u = pontryagin_utils.u_star_2d(x, vx, problem_params)
            return problem_params['l'](x, u)

        # double vmap because we have N_trajectories x N_timesteps ys
        all_ls = jax.vmap(jax.vmap(l_of_y))(ys)

        # single vmap in case we store it flattened again (N_pts,) or (N_pts, nx)
        # all_ls = jax.vmap(l_of_y)(ys)

        # add NaN to every x with v(x) > v_k
        is_outside_valueband = ~np.logical_and(v_lower <= ys['v'], ys['v'] <= v_upper)
        all_ls_masked = all_ls + (is_outside_valueband * np.inf)

        min_l = np.nanmin(all_ls_masked)
        return min_l




    def select_train_pts(value_interval, sols):

        # ideas for additional functionality:
        # - include not only strictly the value interval, but at least n_min pts from each trajectory.
        #   so that if no points happen to be within the value band we include a couple (lower) ones
        #   to still hopefully improve the fit.
        # - return only a random subsample of the data (with a specified fraction)
        # - throw away points of the same trajectory that are closer than some threshold (in time or state space?)
        #   this is also a form of subsampling but maybe better than random.

        v_lower, v_upper = value_interval

        v_finite = np.logical_and(~np.isnan(sols.ys['v']), ~np.isinf(sols.ys['v']))

        v_in_interval = np.logical_and(sols.ys['v'] >= v_lower, sols.ys['v'] <= v_upper)

        # sols.ys['vxx'].shape == (N_trajectories, N_ts, nx, nx)
        # get the frobenius norms of the hessian & throw out large ones.
        if 'vxx' in sols.ys:
            vxx_norms = np.linalg.norm(sols.ys['vxx'], axis=(2, 3))
            vxx_acceptable = vxx_norms < algo_params['vxx_max_norm']  # some random upper bound based on looking at a plot of v vs ||vxx||

            bool_train_idx = np.logical_and(v_in_interval, vxx_acceptable)
        else:
            bool_train_idx = v_in_interval

        all_ys = jtm(lambda node: node[bool_train_idx], sols.ys)

        perc = 100 * bool_train_idx.sum() / v_finite.sum()

        print(f'full (train+test) dataset size: {bool_train_idx.sum()} points (= {perc:.2f}% of valid points)')
        n_data = count_floats(all_ys)
        print(f'corresponding to {n_data} degrees of freedom')

        # check if there are still NaNs left -- should not be the case.
        contains_nan = jtm(lambda n: np.isnan(n).any(), all_ys)
        contains_nan_any = jax.tree_util.tree_reduce(operator.or_, contains_nan)

        if contains_nan_any:
            print('There are still NaNs in training data. dropping into debugger. have fun')
            ipdb.set_trace()

        return all_ys






    def v_meanstd(x, vmap_params):

        # find (empirical) mean and std. dev of value function.
        vs_ensemble = jax.vmap(v_nn_unnormalised, in_axes=(0, None))(vmap_params, x)

        v_mean = vs_ensemble.mean()
        v_std = vs_ensemble.std()

        return v_mean, v_std

    def vx_meanstd(x, vmap_params):

        # vmap for nn ensemble.
        vx_fct = jax.jacobian(v_nn_unnormalised, argnums=1)
        ensemble_vxs = jax.vmap(vx_fct, in_axes=(0, None))(vmap_params, x)

        # now we have all_vxs.shape == (N_ensemble, nx)
        # we want ensemble mean and std across axis 0.
        # stds will be individual for each coordinate, sum/mean whatever later if you want.
        vx_mean = ensemble_vxs.mean(axis=0)
        vx_std = ensemble_vxs.std(axis=0)

        return vx_mean, vx_std


    v_meanstds = jax.jit(jax.vmap(v_meanstd, in_axes=(0, None)))
    vx_meanstds = jax.jit(jax.vmap(vx_meanstd, in_axes=(0, None)))


    def plot_v_along_lines(test_pts, v_nn, params_sobolev_ens, v_k):

        # choose random pairs of points in the currently "known" set.
        # plot v along their connecting line.

        v_means, v_stds = v_meanstds(test_pts, params_sobolev_ens)
        # select points from a thin value band.
        usable = (v_means + 2*v_stds <= v_k) & (v_means - 2*v_stds >= v_k / 2)
        ps = usable / usable.sum()

        for j in range(20):

            # select two points
            pts = jax.random.choice(jax.random.PRNGKey(j), test_pts, shape=(2,), replace=False, p=ps)

            # find the line connecting them
            N = 201
            xs = np.linspace(pts[0], pts[1], N)
            xs = jax.vmap(problem_params['project_M'])(xs)
            ts = np.linspace(0, 1, N)

            # evaluate the value function along the line
            line_ms, line_stds = v_meanstds(xs, params_sobolev_ens)

            pl.plot(ts, line_ms, alpha=.1, c='C0')
            pl.fill_between(ts, line_ms - line_stds, line_ms + line_stds, color='C0', alpha=.1)




    def plot_manifold(v_nn, vmap_params, problem_params):

        # visualise the value function when just changing the angle, leaving
        # the rest ("cartesian" states) fixed.

        thetas = np.linspace(-np.pi, np.pi, 300)

        xs = jax.vmap(lambda theta: np.array([0, 0, np.sin(theta), np.cos(theta), 0, 0, 0]))(thetas)

        mus, sigmas = v_meanstds(xs, vmap_params)

        ax = pl.subplot(211)
        pl.plot(thetas, mus, label='value mean')
        pl.fill_between(thetas, mus - sigmas, mus + sigmas, color='C0', alpha=.2, label=f'value 1σ confidence')
        pl.legend()

        vx_mu, vx_sigma = vx_meanstds(xs, vmap_params)

        pl.subplot(212, sharex=ax)
        pl.plot(thetas, vx_mu, label=problem_params['state_names'])

        pl.gca().set_prop_cycle(None)

        for j in range(7):
            pl.fill_between(thetas, vx_mu[:, j] - vx_sigma[:, j], vx_mu[:, j] + vx_sigma[:, j], alpha=.2)

        pl.legend()









    def forward_sim_nn(x0, params, vmap=False):

        if vmap:
            # we have a whole NN ensemble. use the mean here.
            # v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
            # mean across only axis resulting in a scalar. differentiate later.
            v_fct = lambda x: jax.vmap(v_nn_unnormalised, in_axes=(0, None))(params, x).mean()

        else:
            v_fct = lambda x: v_nn_unnormalised(params, x)

        def forwardsim_rhs(t, x, args):

            lam_x = jax.jacobian(v_fct)(x).squeeze()
            # lam_x = P_lqr @ x  # <- for lqr instead
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](x, u)


        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'], dtmin=.05)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        # simulate for pretty damn long
        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=10., dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
            throw=algo_params['throw'],
        )

        return forward_sol


    def meshcat_forward_sims(x0s, nn_params):

        # just a couple of steps I find myself doing in pdb all the time
        trajs = jax.vmap(forward_sim_nn, in_axes=(0, None, None))(x0s, nn_params, True)

        # convert to old (theta) repr. ugly hardcoded i know
        ys = jax.vmap(jax.vmap(lambda x: np.concatenate([x[0:2], np.array([np.arctan2(x[2], x[3])]), x[4:]])))(trajs.ys)

        solsdict = {'t': trajs.ts, 'x': ys}

        visutils.plot_trajectories(solsdict)



    def forward_sim_nn_until_value(x0, params, v_k, vmap=False):

        # also simulates forward, but stops once we are with high probability
        # inside the value level set v_k AND we have sufficiently low sigma.

        # only vmap=True is tested as of now.

        if vmap:
            # we have a whole NN ensemble. use the mean here.
            # v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
            # mean across only axis resulting in a scalar. differentiate later.
            v_fct = lambda x: jax.vmap(v_nn_unnormalised, in_axes=(0, None))(params, x).mean()

        else:
            v_fct = lambda x: v_nn_unnormalised(params, x)

        def forwardsim_rhs(t, x, args):

            lam_x = jax.jacobian(v_fct)(x).squeeze()
            # lam_x = P_lqr @ x  # <- for lqr instead
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](x, u)


        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'], dtmin=.05)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        # additionally, terminating event.
        # only works for vmapped NN ensemble!
        if not vmap:
            raise NotImplementedError('only vmapped (NN ensemble) case implemented here.')

        def event_fn(state, **kwargs):
            # another stopping condition could be much more simply: v_std < some limit?
            # then we continue a bit if it happens to not be that way right at the edge
            # of the value level set.
            v_mean, v_std = v_meanstd(state.y, params)

            # we only quit once we're very sure that we're in the value level set.
            # thus we take an upper confidence band = overestimated value function = inner approx of level set
            # return (v_mean + 2 * v_std <= v_k).item()   # if meanstd returns arrays of shape (), not floats
            is_very_likely_in_Vk = v_mean + 2 * v_std <= v_k

            sigma_max = algo_params['sigma_max'](v_mean)

            has_low_sigma = v_std <= sigma_max

            # return is_very_likely_in_Vk

            return np.logical_and(is_very_likely_in_Vk, has_low_sigma)

        terminating_event = diffrax.DiscreteTerminatingEvent(event_fn)

        # simulate for pretty damn long
        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=10., dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
            throw=algo_params['throw'],
            discrete_terminating_event=terminating_event,
        )

        return forward_sol


    def solve_backward_nn_ens(x_f, vmap_params, v_upper, problem_params, algo_params):

        v_fct = lambda x: jax.vmap(v_nn_unnormalised, in_axes=(0, None))(vmap_params, x).mean()

        v_f = v_fct(x_f)
        vx_f = jax.jacobian(v_fct)(x_f)

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        # if manifold, backproject here.
        if problem_params['m'] is not None:

            # easy part: project x to the manifold.
            state_f['x'] = problem_params['project_M'](x_f)

            # now we want to set the costate to 0 in the "irrelevant" normal direction.
            # get normal & tangent space projections just like in nn_utils
            B = jax.jacobian(problem_params['m'])(x_f)
            assert B.shape == (problem_params['nx'],), 'only manifolds of codimension 1 supported rn'
            B = B / np.linalg.norm(B)

            # orthogonal projection to normal space at current x
            P_normal = np.outer(B, B)
            # orthogonal projection to tangent space at current x
            P_tangent = np.eye(problem_params['nx']) - P_normal

            # from this construction we have P_normal + P_tangent = I. can we
            # thus just project a costate onto the tangent space? will this
            # work out?

            # the costate is in T*xM, the cotangent space, whereas the state
            # derivative is in TxM. Together they can form the inner product
            # <lambda, xdot> as they often do, which equals d/dt V(x(t)).

            # We decompose lambda:
            # lambda = (P_normal + P_tangent) lambda = lambda_normal + lambda_tangent.
            # the inner product becomes <lambda, xdot> =
            #   = <P_normal lambda, xdot> + <P_tangent lambda, xdot>
            #   = lambda.T P_normal.T xdot + lambda.T P_tangent.T xdot     | writing it out in R^n standard basis
            #   = <lambda, P_normal.T xdot> + <lambda, P_tangent.T xdot>   | changing parentheses without effect & writing as inner product again
            #   = <lambda, P_normal xdot> + <lambda, P_tangent xdot>       | projection matrices symmetric
            #   = 0                       + <lambda, P_tangent xdot>       | normal space is orthogonal to tangent space of which xdot is an element

            # thus, we see that we can arbitrarily modify the costate in
            # normal direction without affecting the relevant inner products.
            # this is kind of obvious right? more formally this means
            # (something like) the canonical map from T*x R^n to T*x M is a
            # surjection, with all lambda in T*x R^n differing only by a
            # vector in normal direction mapping to the same element of T*x M.

            state_f['vx'] = P_tangent @ vx_f


        if algo_params['pontryagin_solver_vxx']:
            vxx_f = jax.hessian(v_nn_unnormalised)(x_f)
            state_f['vxx'] = vxx_f

        return solve_backward(state_f, v_upper=v_upper)


    '''
    # cover a couple different magnitudes
    x0s = np.concatenate([
        # jax.random.normal(jax.random.PRNGKey(0), shape=(100, 6)) * .1,
        # jax.random.normal(jax.random.PRNGKey(1), shape=(100, 6)) * .3,
        jax.random.normal(jax.random.PRNGKey(2), shape=(100, 6)) * 1,
        # jax.random.normal(jax.random.PRNGKey(3), shape=(100, 6)) * 3,
        jax.random.normal(jax.random.PRNGKey(4), shape=(100, 6)) * 10,
    ], axis=0)

    # sol = forward_sim_nn(x0s[0], params)
    # sols         = jax.vmap(forward_sim_nn, in_axes=(0, None))(x0s, params)
    # sols_sobolev = jax.vmap(forward_sim_nn, in_axes=(0, None))(x0s, params_sobolev)
    # sols_sobolev_ens = jax.vmap(forward_sim_nn, in_axes=(0, None, None))(x0s, params_sobolev_ens, True)

    # visualiser.plot_trajectories_meshcat(sols, color=(.5, .7, .5))
    # visualiser.plot_trajectories_meshcat(sols_sobolev)
    # visualiser.plot_trajectories_meshcat(sols_sobolev_ens)
    # visualiser.plot_trajectories_meshcat(sols_lqr, color=(.4, .8, .4))
    '''



    def set_value_target(all_ys, v_k):

        # also, in an initial step we should verify that v_k represents an accurate value level set
        # (or just start with it really low, maybe just lqr solution too.)

        # be generous!
        # value_interval = [0., v_k*2]

        # instead calculate a more educated guess like this:
        # TODO make this configurable via algo_params
        # and get the time constant from problemparams...
        fastestpole_tau = .49  # from LQR solution.
        T = 5 * fastestpole_tau

        # use actual previous value level instead?
        min_l = find_min_l(all_ys, v_k/2, v_k, problem_params)

        print(f'min dv/dt = {min_l:.3f}, max dt/dv = {1/min_l:.3f}')

        # so min value step to ensure horizon <= T is T * smallest dv/dt
        # min l = min dv/dt
        v_step = T * min_l
        v_next = v_k + v_step

        print(f'v_k+1 target = {v_next:.3f}')

        return v_next


    def propose_pts(key, v_k, v_next, vmap_nn_params, x_extent):

        value_interval = [v_k, v_next]

        # ~~~ b) find uniformly sampled points from value band w/ rejection sampling ~~~

        # to "approximate" all kinds of global optimisation & sampling
        # operations over that set. this is a bit ugly and certainly not
        # jit-able... can this be done in a better way?

        all_valueband_pts = np.zeros((0, problem_params['nx']))

        # we want that many points inside the value band, from which we
        # can then select the proposals.
        N_pts_desired = 128 * algo_params['active_learning_batchsize']

        # here just use testpts? or another similar but constant set?
        # with log_min_scale getting enough samples should be easy enough.

        i=0
        while all_valueband_pts.shape[0] < N_pts_desired and i < 1000:

            i = i + 1   # a counter so we return if it never happens.

            newkey, key = jax.random.split(key)

            x_pts = algo_params['sample_states_batched'](
                newkey, 10000, x_extent, log_min_scale=-2
            )

            # what kind of points do we propose? we want points x such that:
            # - x is withing the value band V_{k+1} \ V_k
            # - from those, we want the ones with largest uncertainty.

            v_means, v_stds = v_meanstds(x_pts, vmap_nn_params)

            # optimistic_vs = v_means - 2 * v_stds
            # optimistic_vs = v_means - 1 * v_stds

            # not optimistic hehe
            optimistic_vs = v_means

            # is_in_range = np.logical_and(value_interval[0] <= optimistic_vs, optimistic_vs <= value_interval[1])

            # only be optimistic for the outer boundary instead.
            # inner boundary is a) relatively low-σ and b) nothing happens
            # if we are a bit wrong about it.
            # or, actually, should we be optimistic there too? then we get
            # an outer approximation of the lower sublevel set, meaning we
            # don't propose points *right* at the boundary which could be
            # good right?
            is_in_range = np.logical_and(value_interval[0] <= v_means, optimistic_vs <= value_interval[1])

            interesting_x0s = x_pts[is_in_range]

            all_valueband_pts = np.concatenate([all_valueband_pts, interesting_x0s], axis=0)

        if all_valueband_pts.shape[0] < N_pts_desired:

            # this has never happened since we started using non-uniform sample
            # concentrated around equilibrium here too

            print('did not find enough points!')
            ipdb.set_trace()

            # one possibility: "pad" the points with the ones that are not
            # within the value interval necessarily, but above the lower bound.
            N_missing = N_pts_desired - all_valueband_pts.shape[0]
            arr, idx = jax.lax.top_k(-v_means - np.inf * (v_means < value_interval[0]), N_missing)
            all_valueband_pts = np.concatenate([all_valueband_pts, x_pts[idx]], axis=0)


        all_valueband_pts = all_valueband_pts[0:N_pts_desired, :]
        assert all_valueband_pts.shape == (N_pts_desired, problem_params['nx'])

        # ~~~ c) find a sensible subset of that points to use as proposals ~~~
        # now we have 1000 points that satisfy the first requirement (be inside of the value band).
        # as a first attempt we just sample without replacement according to acquisition function style weights.

        # things to consider afterwards:
        # - ensure the samples are not very close (some literature about this "batched active learning", max kernel distance etc.)

        # though: the highest-uncertainty ones also tend to be high value (= far from the current data set)
        # is this a problem? if V_k+1 is higher than it should be it might take a long time to learn
        # forget this for now maybe its even a good thing.

        v_means, v_stds = v_meanstds(all_valueband_pts, vmap_nn_params)
        sigma_maxs = jax.vmap(algo_params['sigma_max'])(v_means)


        N_proposals = algo_params['active_learning_batchsize']


        proposal_strategy = algo_params['proposal_strategy']

        # every one of these just needs to set proposal_idxs - the indices of
        # proposed points in the array all_valueband_pts.

        do_replace = False

        if proposal_strategy == 'max_sigma':

            # very simple.
            # possible problem: we select only "far" points with very large sigma, while neglecting
            # the ones that are closer which maybe we should do first to even reach the far points
            _, proposal_idxs = jax.lax.top_k(v_stds / sigma_maxs, N_proposals)

        if proposal_strategy == 'max_kernel':

            # very experimental implementation. we choose the max sigma point,
            # then mark all points in a given radius as unusable. maybe this is
            # not maxkernel at all but should ensure that we can propose
            # high-sigma points without them all being in the same region.


            def scan_fct(sigmas, inp):

                # carry the array of standard deviations. replace ones we don't
                # want to use with -inf or scale them down somehow. that way we
                # can just select the argmax every time :)

                # find max sigma.
                proposal_idx = np.argmax(sigmas)
                proposal = all_valueband_pts[proposal_idx]

                # mark close ones as unused.

                # say we have some kernel function k(x, y) satisfying:
                # 0 <= k(x, y) <= 1
                # k(x, x) = 1

                # like this cute RBF kernel here.
                # how to tune this length scale in a smart way??
                # ideas for kernels:
                #  - the lqr state cost matrix
                #  - something from the NN? NN tangent kernel???
                #  - instead max determinant stuff from lenart?
                #  - no clue tbh.
                # probably this should be part of algo_params.
                lengthscale = .5
                k = lambda x, y: np.exp(-np.sum(((x-y) / lengthscale)**2))

                # then we just scale everything by 1-that kernel?
                weights = jax.vmap(lambda x: 1 - k(x, proposal))(all_valueband_pts)

                carry = sigmas * weights

                # oup = (proposal_idx, carry)  # just to look at the data :)
                oup = proposal_idx

                return carry, oup

            sigma_relative = v_stds / sigma_maxs

            final_carry, oups = jax.lax.scan(scan_fct, sigma_relative, None, length=N_proposals)
            proposal_idxs = oups



        elif proposal_strategy == 'max_sigma_and_uniform':

            # mix max_sigma with uniform strategy.
            # first propose 50% of points like max_sigma,
            # then add uniform selection of remaining uncertain points.

            maxsigma_frac = .5
            N_maxsigma = int(N_proposals * maxsigma_frac)
            N_uniform = N_proposals - N_maxsigma

            # first choose *some* max sigma points.
            _, proposal_idxs_max_sigma = jax.lax.top_k(v_stds / sigma_maxs, N_proposals)

            # then find all points p which
            #  a) have uncertainty above max value
            #  b) we have not already chosen in the max_sigma step above.
            where_uncertain = v_stds > sigma_maxs
            where_available = where_uncertain.at[proposal_idxs_max_sigma].set(False)

            N_available = where_available.sum()
            ps = where_available / N_available  # this casts to float :)

            # from those remaining points, get the uniform sample.
            # same move to avoid undefined behaviour as before

            # is there something smarter though? because that way we end up putting the same trajectory in the dataset several times.
            # i guess this is not that bad if we try to avoid this situation by tuning.

            do_replace = N_uniform > N_available
            proposal_idxs_uniform = jax.random.choice(key, all_valueband_pts.shape[0], shape=(N_proposals,), replace=do_replace, p=ps)

            proposal_idxs = np.concatenate([proposal_idxs_max_sigma, proposal_idxs_uniform])


        elif proposal_strategy == 'lowest_v_uncertain':

            # probably would do the same without vmap by relying on broadcasting...
            where_uncertain = v_stds > sigma_maxs

            # replaces the v_means where we are certain enough by inf
            # that way once we multiply by -1 we have -inf and negative values
            # the largest negative values = the smallest positive values
            v_means_uncertain = v_means + ~where_uncertain * np.inf
            _, proposal_idxs = jax.lax.top_k(-v_means_uncertain, N_proposals)

            # what if that way we make too few proposals???
            # include ones with lower uncertainty? raise some sort of signal that the value
            # step can be increased???


        elif proposal_strategy == 'uniform_uncertain':
            # other, simpler idea: among the points with excessive uncertainty just choose
            # a uniform subsample
            # probably this will also biased towards the upper level set but maybe not overly so.
            # this seems to make slower progress than just maximum uncertainty...
            is_uncertain = v_stds > sigma_maxs
            N_uncertain = np.sum(is_uncertain)
            ps = is_uncertain / N_uncertain

            # if we want more samples than points sampling without replacement is undefined.
            # but generally we prefer no replacement (otw the same point is repeated!)
            do_replace = N_proposals > N_uncertain

            proposal_idxs = jax.random.choice(key, N_pts_desired, shape=(N_proposals,), replace=do_replace, p=ps)

        # these two "softmax" strategies can be understood as an interpolation
        # between uniform_among_uncertain (= softmax_but_only_uncertain as the
        # scale we divide by inside the softmax goes to +inf) and max_sigma
        # (= softmax as that scale goes to 0)

        elif proposal_strategy == 'softmax':

            # by scaling with sigma_maxs we hit the right range for the softmax function hopefully.
            # the factor just makes the distribution a bit closer to uniform.
            scale = 10
            ps = jax.nn.softmax(v_stds / sigma_maxs / scale)
            # proposals = jax.random.choice(key, all_valueband_pts, shape=(N_proposals,), replace=False, p=ps)

            # passing an int (N_proposals) is understood as choosing from arange(0, N_proposals)
            proposal_idxs = jax.random.choice(key, all_valueband_pts.shape[0], shape=(N_proposals,), replace=False, p=ps)

        elif proposal_strategy == 'softmax_uncertain':

            # same as above, but after the softmax we modify the weights to
            # place 0 probability on the samples already below sigma, instead
            # of just low probability. though in short ipdb experiments this
            # changes almost nothing, as the points are already VERY unlikely

            scale = 10
            ps = jax.nn.softmax(v_stds / sigma_maxs / scale)

            is_certain = v_stds < sigma_maxs
            ps = ps.at[is_certain].set(0)
            ps = ps / ps.sum()

            do_replace = N_proposals > (~is_certain).sum()

            proposal_idxs = jax.random.choice(key, all_valueband_pts.shape[0], shape=(N_proposals,), replace=False, p=ps)

        # but that's kind of dumb, we give a small probability of also selecting points with exactly

        else:
            raise ValueError(f'unknown proposal strategy "{proposal_strategy}"')


        if do_replace:
            print('warning -- had too few points to sammple, resorting to choice(replace=True)')

        # make some global --loglevel style algoparam to decide what to plot?
        # and/or switching between savefig and show? especially useful for euler...
        plot=True
        if plot:
            # pl.plot(v_means, v_stds, '. ', label='candidates')
            pl.plot(v_means[proposal_idxs], v_stds[proposal_idxs], '. ', label='proposals', alpha=.2, color='green')
            pl.legend()

            pl.xlim([1e-1, 1e4])
            pl.ylim([1e-2, 1e3])


        proposed_states = all_valueband_pts[proposal_idxs]

        return proposed_states



    @equinox.filter_jit
    def batched_oracle(proposals, v_k, v_next, vmap_nn_params, problem_params):

        # forward simulation. this stops if BOTH of these conditions hold.
        # - v_mean + 2 * v_sigma <= v_k
        # - v_sigma <= 0.5
        # so we can be quite sure the information at that point is usable.
        # (also stops if time horizon ends. )

        # project proposals to manifold. should not be needed if sampling fct
        # is properly designed. still here just in case :)jk
        proposals = jax.vmap(problem_params['project_M'])(proposals)

        forward_sols = jax.vmap(forward_sim_nn_until_value, in_axes=(0, None, None, None))(
            proposals,
            vmap_nn_params,
            v_k,
            True
        )

        xfs = jax.vmap(lambda sol: sol.ys[sol.stats['num_accepted_steps']])(forward_sols)


        # the solutions that stopped due to DiscreteTerminatingEvent
        stopped_bc_terminatingevent = forward_sols.result == 1

        # sanity check: this should be the same. literally just checking the terminatingevent
        # conditions as well. *maybe* there is some edge case where the condition is True at the
        # last step and the solver quits anyway, so it doesn't report quitting "due to" the event?
        mus, sigs = v_meanstds(xfs, vmap_nn_params)
        sig_maxs = algo_params['sigma_max'](mus)
        is_usable = np.logical_and(mus + 2 * sigs <= v_k, sigs <= sig_maxs)

        # this assertion never failed since the last change of making
        # sigma_max a function specified in algo_params. should we still
        # somehow try to do it? is chex the tool for this?
        # assert (stopped_bc_terminatingevent == is_usable).all(), 'shit happened'


        # if we have a different amount every time, we cannot jit the simulation.
        # therefore we just mark it as nan and try to tune the algo such that not too many
        # of them are nan.
        # usable_xfs = xfs.at[~is_usable].set(np.nan)

        # turns out that was itself not jittable. this should work:
        usable_xfs = np.where(is_usable[:, None], xfs, np.nan * xfs)

        # as we kind of would expect, is_usable correlates clearly (negatively) with the amount of
        # solver steps. so the most effort is spent calculating solutions which we're never going to use.
        # could we somehow avoid this? maybe stop after 3/4 of solutions have terminated? probably but
        # then the implementation becomes messier, because plain vmap doesn't allow cross communication.
        # more simply: just set a rather low step limit and be fine with a couple more solutions being
        # thrown out.

        # or just don't care, forward sim is cheaper than backward anyway. (is
        # it? with vmapped nn ensemble maybe not..., certainly not if backward sim
        # is without vxx.)


        # generous upper bound for value we're interested in rn.
        # integration of trajectories stops once we pass this threshold.
        v_upper = v_next + 10 * (v_next - v_k)

        # TODO jit this.
        backward_sols_new = jax.vmap(solve_backward_nn_ens, in_axes=(0, None, None, None, None))(
            usable_xfs, vmap_nn_params, v_upper, problem_params, algo_params
        )



        # TODO global switch for plot+show/plot+savefig/no plot
        plot=False
        if plot:
            # plot 0th forward and backward sol in same plot.
            pl.figure()
            sol0 = solve_backward_nn_ens(usable_xfs[0], vmap_nn_params, v_upper, problem_params, algo_params)
            sol0_fwd = jtm(itemgetter(0), forward_sols)
            fwd_ts_adjusted = sol0_fwd.ts - sol0_fwd.ts[sol0_fwd.stats['num_accepted_steps']]

            pl.subplot(221)
            pl.plot(fwd_ts_adjusted, sol0_fwd.ys)
            pl.gca().set_prop_cycle(None)
            plotting_utils.plot_sol(sol0, problem_params)

        return backward_sols_new

    def prune_and_train_simple(key, params_sobolev_ens, all_ys, v_interval, previously_suboptimal, algo_params, warmstart=False):

        # what if we first do a simpler version of this prune_and_train thing?
        # consisting of just one step instead of a loop with sub-valuesteps.
        #  a) prune the (parts of) solutions that are clearly suboptimal
        #     (optimally just enough to avoid conflicts...)
        #  a) add to training data, train.

        # mark clearly suboptimal data.

        v_lower, v_upper = v_interval

        v_nn_means, v_nn_stds = jax.vmap(v_meanstds, in_axes=(0, None))(all_ys['x'], params_sobolev_ens)

        # now without the extra dim the vmap we already did is sufficient
        # v_nn_means, v_nn_stds = v_meanstds(all_ys['x'], params_sobolev_ens)


        if algo_params['pruning_strategy'] == 'conservative':

            # be conservative: only prune POINTS (not trajectories) that
            # definitely (with high prob) are outside of value level set
            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v_lower
            trajectory_outside_levelset = v_lower < all_ys['v']

            is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset

        elif algo_params['pruning_strategy'] == 'conservative_future':

            # same as conservative, BUT also mark points as suboptimal that
            # "lead" to a suboptimal trajectory segment in the future. this
            # should be strictly better than 'conservative'.

            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v_lower
            trajectory_outside_levelset = v_lower < all_ys['v']

            is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset

            # time goes from 0.0 at idx 0 to negative values at idx 1, 2, ... so
            # cumsum marks as suboptimal the PRECEDING points in physical time even
            # though in array indices they are the subsequent ones. all correct.
            is_suboptimal = np.cumsum(is_suboptimal, axis=1) > 0

        elif algo_params['pruning_strategy'] == 'conservative_bidirectional':

            # same as conservative_future BUT also remove trajectory segments that in the close past have been suboptimal
            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v_lower
            trajectory_outside_levelset = v_lower < all_ys['v']

            is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset
            is_suboptimal = np.cumsum(is_suboptimal, axis=1) > 0



        elif algo_params['pruning_strategy'] == 'generous':

            # start with pointwise pruning mask from conservative strategy.
            # delete not only the points preceding any suboptimal point, but
            # also the ones after it, as long as they are above the currently
            # known value level.

            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v_lower
            trajectory_outside_levelset = v_lower < all_ys['v']

            point_is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset

            # clear out everything above the lower value level if there is a suboptimal point in the trajectory.
            is_suboptimal = point_is_suboptimal.any(axis=1)[:, None] & (all_ys['v'] >= v_lower)


        elif algo_params['pruning_strategy'] == 'bayesian':

            # same inequality as above but without the vk sandwiched in. this
            # means we prune a strict superset of the points pruned with the
            # 'bayesian_future' strategy. in fact, the condition there is:

            # (v_nn_means + 3 * v_nn_stds < v_lower) & (v_lower < all_ys['v'])

            # which is equivalent to is_suboptimal (below) &

            is_suboptimal = (v_nn_means + 3 * v_nn_stds < all_ys['v']) & (v_lower < all_ys['v'])

            # time goes from 0.0 at idx 0 to negative values at idx 1, 2, ... so
            # cumsum marks as suboptimal the PRECEDING points in physical time even
            # though in array indices they are the subsequent ones. all correct.
            is_suboptimal = np.cumsum(is_suboptimal, axis=1) > 0

            #

        # keep suboptimal points marked suboptimal
        is_suboptimal = np.logical_or(previously_suboptimal, is_suboptimal)


        print(f'pruning {is_suboptimal.sum():.3f} = {100 * is_suboptimal.mean():.3f}% points')

        # next step: build training data out of this pruned mess.
        in_band = (all_ys['v'] <= v_upper)


        if algo_params['thin_data']:

            # much simpler strategy: just exclude way past data.
            v_cutoff = v_lower / 100
            in_band = in_band & (v_cutoff <= all_ys['v'])

            '''
            # test strategy: take fixed number of highest value points (leading to
            # upper, densely sampled value band) & fixed number subsample of lower
            # points.

            N_band = algo_params['N_band']
            N_lower = algo_params['N_lower']

            if usable_ys['v'].shape[0] >= N_band + N_lower:

                # thin out the data, if we have more data than N_band + N_lower

                # should we not instead of the top k values take the ones just
                # above the known level set? or close above&below? with the
                # current way we might sample less densely in the region we
                # need to learn first

                arr, top_idx = jax.lax.top_k(usable_ys['v'], N_band)
                print('thinning out data')
                print(f'densely sampled value interval = [{arr.min()}, {arr.max()}]')

                if arr.min() >= v_interval[0]:
                    print('warning: densely sampled interval smaller than value interval')

                # nicer to work with boolean indices.
                bot_bool_idx = np.ones_like(usable_ys['v'], dtype=bool).at[top_idx].set(False)
                top_bool_idx = ~bot_bool_idx

                # this should hold based on the if above. if not, documentation says that choice
                # without replacements is undefined.
                assert bot_bool_idx.sum() >= N_lower, 'not enough datapoints'

                # we take a random subsample. bool indices serve as probabilities -> choose only from bottom subset.
                # we return the indices, so we can choose those elements from each dict member, thus the int (-> arange).
                key, sample_key = jax.random.split(key)
                bot_subsample_idx = jax.random.choice(
                        sample_key, usable_ys['v'].shape[0], shape=(N_lower,), replace=False, p=bot_bool_idx.astype(float)
                )

                all_idx = np.concatenate([top_idx, bot_subsample_idx])

                usable_ys = jtm(lambda node: node[all_idx], usable_ys)
            '''



        bool_train_idx = in_band & ~is_suboptimal

        # b) train the NN again, while ignoring data marked as suboptimal.
        # easiest thing to do here: extract training data like in mockup, make new array.
        # surely we can optimise this and keep fixed shapes for jit.
        usable_ys = jax.tree_util.tree_map(lambda node: node[bool_train_idx], all_ys)


        print(f'total data points: {usable_ys["v"].shape[0]}')



        # split into train/test set.
        train_ys, test_ys = nn_utils.train_test_split(usable_ys)

        # use these instead if we somehow need the normaliser again
        # ys_n = normaliser.normalise_all_dict(train_ys)
        # test_ys_n = normaliser.normalise_all_dict(test_ys)

        init_key, key = jax.random.split(key)
        params_init = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))

        # only count each individual NN's params to assess under/overparameterisation
        n_params = count_floats(params_init)
        n_data = count_floats(train_ys)
        print(f'params/data ratio = {n_params/n_data:.4f}')

        params_old = params_sobolev_ens

        if warmstart:
            # continue from previous params, only last portion of training.
            params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble_warmstarted(
                train_key, train_ys, params_sobolev_ens, problem_params, algo_params
            )
        else:
            # training from scratch
            params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble(
                train_key, train_ys, problem_params, algo_params
            )


        '''
        # so, here we could maybe detect if collisions made training worse.
        # might we also identify the particular datapoints where it occured?
        # basically the ones with maximum loss...

        # but that will be such a huge mess...

        # i think the key doesn't enter unless we have prior and/or stochastic vxx hvp approximation
        key, batchloss_key = jax.random.split(key)
        # evaluate this again too because when training we also include the prior

        train_lossmeans, train_terms = jax.vmap(v_nn.sobolev_loss_batch_mean, in_axes=(None, 0, None, None, None))(
            batchloss_key, params_sobolev_ens, train_ys, problem_params, algo_params
        )

        test_lossmeans, test_terms = jax.vmap(v_nn.sobolev_loss_batch_mean, in_axes=(None, 0, None, None, None))(
            batchloss_key, params_sobolev_ens, test_ys, problem_params, algo_params
        )
        '''


        return params_sobolev_ens, oups_sobolev_ens, is_suboptimal



    def prune_and_train_substeps(params_sobolev_ens, all_ys, v_interval):

        raise NotImplementedError('for now use prune_and_train simple plz')

        # pseudocode:
        # for v in linspace(v_k, v_k+1):
        #     prune clearly suboptimal solutions (and maybe a small segment before too)
        #     re-train NN with pruned dataset

        # -> this is probably dumb. the whole point of using pontryagin at all
        # is that we have two different time discretisations, a fine one (= ODE
        # integrator) for the trajectories responsible for local optimality,
        # and a larger one to prune non-globally-optimal solutions. let's
        # embrace (or disprove....) the fact that most practical problems work
        # just fine if pruning happens at a much smaller rate. if true that
        # enables us to minimise the main bottleneck which is repeated function
        # approximation.


        v_substeps = np.linspace(*v_interval, 10)



        for v in v_substeps:

            # an easy way to guarantee that this works would be the following:
            # prune not only the "clearly suboptimal" points, but also a small value
            # band below them. then set the value substep <= that valueband height.
            # does it follow from this that all collisons are handled properly? not
            # very sure...


            # a) mark data that is clearly suboptimal wrt the NN posterior as invalid.
            # prune only solutions where both of these hold:
            # v <= v(x) with high probability.
            # v_nn(x) <= v with high probability

            # v_meanstds is already vmapped. here we vmap it a second time for the
            # trajectories axis.
            v_nn_means, v_nn_stds = jax.vmap(v_meanstds, in_axes=(0, None))(all_ys['x'], params_sobolev_ens)

            trajectory_outside_levelset = v < all_ys['v']
            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v

               # these two together mean that with high probability (3 sigma for N(0, 1))
               # the given point is suboptimal.
               # instead of mu_v + 3 sigma_v < v < v_trajectory, we could also just ask
               # for mu_v + 3 sigma_v < v_trajectory. then we also classify these two additional situations as suboptimal:
               #  1. mu_v + 3 simga_v < v_trajectory < v.
               #  the trajectory is inside the level set and so has alredy been used for the NN fit. not interesting.
               #  2. v < mu_v + 3 simga_v < v_trajectory
               #  the nn solution is also outside the level set. despite the 3 sigma we choose to not prune
               #  based on that info, bc it is still an extrapolation.
               # not 100% sure if it smart to exclude these cases, so maybe it is smarter to ditch the v in the middle?

               # additional rule which we can use here: if a trajectory segment is globally suboptimal, everything
               # before it (wrt physical, forward time) is also suboptimal and we can ditch it. TODO.

            is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset

            is_suboptimal = v_nn_means + 3 * v_nn_stds < all_ys['v']

            ipdb.set_trace()

            v_min = v / 2  # or something... or set fixed number of pts and find with argpartition?
            bool_train_idx = (v_min <= all_ys['v']) & (all_ys['v'] <= v) & ~is_suboptimal

            # b) train the NN again, while ignoring data marked as suboptimal.
            # easiest thing to do here: extract training data like in mockup, make new array.
            all_ys = jax.tree_util.tree_map(lambda node: node[bool_train_idx], sols_orig.ys)

            # split into train/test set.
            train_ys, test_ys = nn_utils.train_test_split(all_ys)

            # call sobolev training fct...

            # but i'd rather keep everything constant sized... this will require changing
            # the traininng function quite heavily though... for this we need:
            # - a modified training function that takes in the (large) ys tree
            #   and also bool_train_idx.




        return is_optimal, params_sobolev_ens, v_next_actual

    # initial step:
    # - generate data from uniform backward shooting
    # - do train_and_prune step to find value nn and known value level.
    # - start loop



    # test points, with increased density towards origin.
    # 10000 points doesn't even look like all that much on a plot, maybe we need more...
    N_testpts = 100000

    # keep this "hardcoded" here? put in algoparams? make some heuristic to
    # adapt based on data?
    x_extent = np.array([
        20,  20,  # x and y, [m]
        1., 1.,   # sinPhi and cosPhi [1] (but irrelevant -- see sampling fct)
        20,  20,  # vx and vy, [m/s]
        20*np.pi  # omega [rad/s]
    ])

    test_pts = algo_params['sample_states_batched'](
        jax.random.PRNGKey(123), N_testpts, x_extent, log_min_scale=-2
    )

    # use this to "mark" test points that AT SOME POINT were below the sigma limit.
    test_pts_known = np.zeros((N_testpts,)).astype(bool)


    # @jax.jit
    def estimate_value_level(test_pts, test_pts_known, params_sobolev_ens, upper_v=np.inf):

        # this function could also try to detect learning failure...

        # estimate "known" value level based on finite test points set.
        v_means, v_stds = v_meanstds(test_pts, params_sobolev_ens)

        sigma_maxs = jax.vmap(algo_params['sigma_max'])(v_means)
        sigma_small_enough = v_stds <= sigma_maxs

        sigma_small_enough = np.logical_or(test_pts_known, sigma_small_enough)


        if algo_params['vk_estimator'] == 'strict':

            # replace everything where sigma is small enough by infinity.
            # then we can take the minimum to find the lowest-v point with
            # sigma too high. This becomes our v_k.

            # this still "profits" from the points being marked as known going
            # into the infmask. So the value level *can* decrease from one
            # iteration to the next, but probably not by much. let's see how it
            # does.

            v_means_infmasked = v_means + np.inf * sigma_small_enough
            v_k = v_means_infmasked.min()

        elif algo_params['vk_estimator'] == 'k_exceptions':

            k = 5

            # same as 'strict' but ignores the first k uncertain points.
            v_means_infmasked = v_means + np.inf * sigma_small_enough
            _, smallest_k_idx = jax.lax.top_k(-v_means_infmasked, k)
            v_means_infmasked = v_means_infmasked.at[smallest_k_idx].set(np.inf)

            v_k = v_means_infmasked.min()

        elif algo_params['vk_estimator'] == 'relaxed':

            # be relaxed about *a few* high-σ points being inside our set.
            # particularly, find the highest v_k such that the fraction of
            # uncertain (σ > σ_max(v)) points inside Vk is <= a threshold.

            # this is probably n log(n) (sorting algo certainly, then only linear
            # stuff). the whole thing could be found directly by bisection which
            # would also be nlogn

            idx = np.argsort(v_means)

            v_means_sorted = v_means[idx]
            v_stds_sorted = v_stds[idx]
            sigma_small_enough_sorted = sigma_small_enough[idx]

            # for each k, this is the fraction
            #
            #      #(j: v[j] <= v[k] and σ[j] < threshold)
            #      ―――――――――――――――――――――――――――――――――――――――
            #                #(j: v[j] <= v[k])

            # ...probably. i think there might be some sort of mistake in here
            # frac_certain_inside = 1 - np.cumsum(1 - sigma_small_enough[idx]) / sigma_small_enough.shape[0]

            # this is correct i think. we want to divide by the number of smaller vs which in the sorted version is just the index.
            frac_certain_inside = 1 - np.cumsum(1 - sigma_small_enough[idx]) / (np.arange(test_pts.shape[0]) + .0001)

            # now, find the largest index k for which that fraction is above the threshold
            threshold = algo_params['frac_certain_in_Vk']

            # because the function is monotonously decreasing, we may equivalently find the
            # SMALLEST frac_certain_inside that is still above the limit.
            k_accept = np.argmin(frac_certain_inside + np.inf * (frac_certain_inside < threshold))

            v_k = v_means_sorted[k_accept]

        else:
            est = algo_params['vk_estimator']
            raise ValueError(f'v_k estimator "{est}" undefined!')

        # clip it to upper_v in case we estimate something higher...
        v_k = np.minimum(upper_v, v_k)




        # the new "known points" buffer. We consider points known if:

        # a) they are below the sigma threshold, and clearly (2 sigma) within the currently estimated level set
        new_testpts_known = (sigma_small_enough * (v_means + 2 * v_stds <= v_k))

        # b) they are above the sigma threshold, and VERY clearly (2 sigma) within the currently estimated level set
        new_testpts_known = np.logical_or(new_testpts_known, v_means + 10 * v_stds <= v_k)

        new_testpts_known = new_testpts_known


        print(f'estimated known value level: {v_k:.3f}')
        pl.figure()

        pl.xlabel('v mean')
        pl.ylabel('v std')

        pl.loglog(v_means + (np.nan * new_testpts_known), v_stds, '. ', alpha=.1, c='C1', label='unknown points')
        pl.loglog(v_means + (np.nan * ~new_testpts_known), v_stds, '. ', alpha=.1, c='C0', label='known points')
        # pl.loglog(v_means + (np.nan * ~newly_known), v_stds, '. ', alpha=.1, c='red', label='newly known points')
        pl.loglog([v_k, v_k], [v_stds.min(), v_stds.max()], linestyle='--', color='black', alpha=.2, label='v_k')

        vmax = v_means.max()
        plot_vs = np.logspace(-4, np.log10(vmax+1), 200)
        plot_sig_maxs = jax.vmap(algo_params['sigma_max'])(plot_vs)
        pl.loglog(plot_vs, plot_sig_maxs, linestyle='--', alpha=.5, label='$σ_{max}(v)$')


        print(f'test points known: {100*new_testpts_known.mean():.2f}%')

        # estimate actual state space volume with second half of test points.
        # this only works if the sampling function actually puts the uniformly
        # sampled subset there. specifically I think if log_min_scale > 0 the
        # distribution of points will still be usable but with uniform points
        # in first half. so avoid that.

        half = test_pts.shape[0] // 2
        print(f'state space volume known: {100*new_testpts_known[half:].mean():.6f}%')

        return v_k, new_testpts_known



    # choose initial value level. should we just blindly assume that below
    # this value level we only have globally optimal solutions? then we could
    # rapidly fill that sublevel set instead of being careful about
    # collisions... but no way to verify the assumption besides praying

    v_k = 50

    # to get a feel for when the linearisation stops being accurate.
    # important: this is only valid when we have a good covering of the
    # sublevel set, which with initial data we don't. also maybe doing
    # something like this for vx would be more meaningful?
    '''
    solution_vs = sols_orig.ys['v'].reshape(-1)
    lqr_vs = jax.vmap(V_f)(sols_orig.ys['x'].reshape(-1, problem_params['nx']))
    pl.figure('lqr V vs trajectory V')
    pl.loglog(lqr_vs, solution_vs, '. ', alpha=.2)
    '''

    all_ys = select_train_pts([v_k/1000, v_k], sols_orig)

    # what if instead we don't really constrain this initial data set to a
    # low-ish value level, and just let the v_k estimator figure out up to
    # which level it worked? -> value level is the same but extrapolation seems
    # to improve which makes sense. equivalently we can just set v_k a bit
    # higher initially
    # all_ys = select_train_pts([0.001, 500], sols_orig)

    # split into train/test set.

    # put in the lqr solution hehehe (for v_k like 5 it is practically the same...)
    # fake_ys = all_ys.copy()
    # fake_ys['v'] = jax.vmap(V_f)(all_ys['x'])
    # fake_ys['vx'] = jax.vmap(jax.jacobian(V_f))(all_ys['x'])
    # train_ys, test_ys = nn_utils.train_test_split(fake_ys, train_frac=algo_params['nn_train_fraction'])

    train_ys, test_ys = nn_utils.train_test_split(all_ys, train_frac=algo_params['nn_train_fraction'])


    v_nn = nn_utils.nn_wrapper(
        input_dim=problem_params['nx'],
        layer_dims=algo_params['nn_layerdims'],
        output_dim=1
    )


    # normaliser = nn_utils.data_normaliser(train_ys, problem_params, algo_params)
    # ys_n = normaliser.normalise_all_dict(train_ys)
    # test_ys_n = normaliser.normalise_all_dict(test_ys)

    init_key, key = jax.random.split(key)
    params_init = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))


    # test loss fct to pdb with concrete values.
    sol = jtm(itemgetter(12), sols_orig)
    y = jtm(itemgetter(12), sol.ys)
    loss = v_nn.sobolev_loss(key, y, params_init, problem_params, algo_params)
    extent = np.array([20, 20, 0., 0., 20, 20, 10])
    priorloss = v_nn.sobolev_loss_with_prior(key, y, params_init, None, extent, problem_params, algo_params)


    # to get a feel for over/underparameterisation.
    n_params = count_floats(params_init)
    n_data = count_floats(train_ys)
    print(f'params/data ratio = {n_params/n_data:.4f}')
    print(f'nn params: {n_params}')

    train_key, key = jax.random.split(key)

    params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble(
        train_key, train_ys, problem_params, algo_params, ys_test=test_ys
    )

    # ipdb.set_trace()

    # shouldn't it have been called "de-normalised" anyway?
    # v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
    # because that one is actually "unnormalised":
    v_nn_unnormalised = v_nn

    idx = 20
    sol = jax.tree_util.tree_map(itemgetter(idx), sols_orig)

    # pl.figure()
    # plotting_utils.plot_trajectory_vs_nn(sol, params_sobolev, v_nn_unnormalised)

    pl.figure('trajectory vs NN')
    plotting_utils.plot_trajectory_vs_nn_ensemble(sol, params_sobolev_ens, v_nn_unnormalised)

    # misuse the plotting function to compare trajectories w/ lqr solution.
    # it seems like all the optimal control stuff checks out indeed -- we
    # do have V_lqr(x(t)) ≈ v(t) along the initial part of the solutions.
    # pl.figure('trajectory vs LQR value fct')
    # plotting_utils.plot_trajectory_vs_nn(sol, P_lqr, lambda P, x: 0.5 * x.T @ P @ x)

    pl.figure('training run')
    plotting_utils.plot_nn_train_outputs(oups_sobolev_ens)

    pl.figure('nn calibration, initial run')
    means, stds = jax.vmap(v_meanstds, in_axes=(0, None))(sols_orig.ys['x'], params_sobolev_ens)
    plot_calibration(sols_orig.ys, means, stds)

    pl.figure('manifold')
    plot_manifold(v_nn, params_sobolev_ens, problem_params)

    pl.show()


    def print_solver_stats(sols):
        pass



    def flat_sol_ys(sols):
        # for a sols object with ys dict, reshape each member of the ys dict
        # from (N_trajectories, N_t_per_trajectory, x) to (N_pts, x)
        # (where x is either nx or nothing for scalar values)
        # also removes nan or inf values.

        # would be cool: if we still include a single nan point between the solutions
        # to make plotting "everything at once" nicer bc it breaks the line.

        where_usable = np.logical_and(~np.isnan(sols.ys['v']), ~np.isinf(sols.ys['v']))
        flat_ys = jtm(lambda node: node[where_usable], sols.ys)
        return flat_ys



    # all_ys = flat_sol_ys(sols_orig)

    all_ys = sols_orig.ys
    is_suboptimal = np.zeros_like(all_ys['v']).astype(bool)


    # more detailed plots w/ savefig.
    pl.rcParams['figure.figsize'] = (16, 9)


    v_next_target = np.inf

    vks = []

    for k in range(100):

        # active learning with level-set ideas embedded.
        # first pseudocode algo in idea dump.
        # seems to finally work alright!!! (after fixing mostly nn training issues)

        # this has got to be a bit fancy
        print('\n\n\n')
        print(f' ~~~~ active learning iteration {k} ~~~~')

        # why did we split up estimate_value_level and propose_pts?
        # don't we just calculate the whole mean/std at test pts twice?

        vk_prev = v_k
        v_k, test_pts_known = estimate_value_level(test_pts, test_pts_known, params_sobolev_ens, upper_v=v_next_target)

        if v_k < vk_prev and k > 0:
            print('warning: level set shrinking.')

        vks.append(v_k)

        # set next value target :)
        v_next_target = set_value_target(all_ys, v_k)

        key = jax.random.PRNGKey(k)

        proposed_pts = propose_pts(key, v_k, v_next_target, params_sobolev_ens, x_extent)

        # try this random move. propose points with higher level set, but only
        # learn with smaller step.

        # this actually seems to work alright (or the general settings right
        # now seem to work alright). counterintuitively we give less info to
        # the extrapolated, 'half-learned' regime. but maybe that's the price
        # we pay for actual pruning of suboptimal solutions.
        v_next_target = v_k + (v_next_target - v_k)*0.2

        # this figure is opened in estimate_value_level and further written to in propose_pts...
        # and here too...
        if algo_params['savefigs']:
            pl.savefig(f'tmp/meanstds_{k:04d}.png')

        # ~~~~ ORACLE ~~~~
        backward_sols_new = batched_oracle(proposed_pts, v_k, v_next_target, params_sobolev_ens, problem_params)

        # is_usable = ~np.isnan(jax.vmap(lambda sol: sol.ys['v'][sol.stats['num_accepted_steps']])(backward_sols_new))
        # simpler way:
        is_usable = backward_sols_new.stats['num_accepted_steps'] > 0
        print(f'{100*is_usable.mean():.2f}% of forward simulations reached lower value level set AND low sigma.')

        '''
        # find out how close we got.
        # this depends on the specific timesteps too...
        sol_min_dist = lambda x, sol: np.min(np.linalg.norm(sol.ys['x'] - x[None, :], axis=1))
        sol_dists = lambda x, sol: np.linalg.norm(sol.ys['x'] - x[None, :], axis=1)

        min_dists = jax.vmap(sol_min_dist, in_axes=(0, 0))(proposed_pts, backward_sols_new)
        dists = jax.vmap(sol_dists, in_axes=(0, 0))(proposed_pts, backward_sols_new)
        ipdb.set_trace()
        '''

        # append new data to main data set, now in flattened shape. would it be
        # smarter to keep this in some sort of dict to avoid reallocation?
        # -> probably marginal gains

        # truly unhinged idea: keep the big dataset sorted by ys['v'], so we can
        # do binary search to find the relevant value ranges?

        print_solver_stats(backward_sols_new)

        # new_ys = flat_sol_ys(backward_sols_new)

        new_ys = backward_sols_new.ys
        all_ys = jtm(lambda a, b: np.concatenate([a, b], axis=0), all_ys, new_ys)

        # ipdb.set_trace()
        is_suboptimal = np.concatenate([is_suboptimal, np.zeros_like(new_ys['v']).astype(bool)], axis=0)

        prev_params_sobolev_ens = params_sobolev_ens

        train_key = key  # yolo
        params_sobolev_ens, oups, is_suboptimal = prune_and_train_simple(
            train_key,
            params_sobolev_ens,
            all_ys,
            [v_k, v_next_target],
            is_suboptimal,
            algo_params,
            warmstart=algo_params['nn_warm_start']
        )
        all_oups = oups

        '''
        else:
            all_oups = None

            # store somewhere which solutions we've pruned already?

            # or just "blindly":
            n_pruned = []
            v_next_list = np.linspace(v_k, v_next_target, 20)
            for i, vnext in enumerate(v_next_list):

                print(f' ~~~~ prune&train substep {i}, vnext = {vnext:.3f} ~~~~')
                params_sobolev_ens, oups, is_suboptimal_new = prune_and_train_simple(
                    train_key,
                    params_sobolev_ens,
                    all_ys,
                    [v_k, vnext],
                    is_suboptimal,
                    algo_params,
                    warmstart=True
                )

                n_pruned.append(is_suboptimal_new.sum())

                # remove pruned points. once marked suboptimal we definitely won't need it again.
                all_ys = jtm(lambda node: node.at[is_suboptimal].set(np.inf), all_ys)

                v_k = vnext  # blindly accept? we need that to prune in the next substep...

                if all_oups is None:
                    all_oups = oups
                else:
                    # these shapes are (N_ensemble, N_trainsteps) apparently so axis=1
                    all_oups = jtm(lambda a, b: np.concatenate([a, b], axis=1), all_oups, oups)

                pl.figure(f'n pruned, iter {k}')
                pl.plot(n_pruned)
                if algo_params['savefigs']:
                    pl.savefig(f'tmp/n_pruned_{k:04d}.png')
        '''





        # here:
        # if NN training went like shit:
        #     repeat prune&train with smaller v_next_target.

        # or even better:
        # while oups['train_loss'] > threshold:
        #     v_next_target = v_k + (v_next_target - v_k) / 2
        #     repeat prune_and_train_simple, starting from PREVIOUS params if warmstart bc current ones are messed up


        pl.figure(f'nn training iter {k}')
        plotting_utils.plot_nn_train_outputs(all_oups, subsample=8)
        pl.ylim([1e-4, 1e3])
        if algo_params['savefigs']:
            pl.savefig(f'tmp/trainplot_{k:04d}.png')


        pl.figure(f'random trajectory, iter {k}')
        plotting_utils.plot_trajectory_vs_nn_ensemble(sol, params_sobolev_ens, v_nn_unnormalised)
        if algo_params['savefigs']:
            pl.savefig(f'tmp/trajectory_{k:04d}.png')

        pl.figure('manifold')
        plot_manifold(v_nn, params_sobolev_ens, problem_params)

        pl.figure(f'nn calibration, iter {k}')
        means, stds = jax.vmap(v_meanstds, in_axes=(0, None))(all_ys['x'], params_sobolev_ens)
        plot_calibration(all_ys, means, stds)

        pl.figure(f'value lines, iter {k}')
        plot_v_along_lines(test_pts, v_nn, params_sobolev_ens, v_next_target)


        if k % 20 == 0:
            ipdb.set_trace()


        if algo_params['savefigs']:
            pl.savefig(f'tmp/calibration_{k:04d}.png')
            pl.close('all')
        else:
            pl.show()
            # pass



    pl.figure()
    pl.plot(vks, label='known value level')
    pl.legend()

    pl.figure('off manifold straying m(x)')
    pl.plot(jax.vmap(problem_params['m'])(all_ys['x'].reshape(-1, 7)))

    pl.show()
    ipdb.set_trace()



