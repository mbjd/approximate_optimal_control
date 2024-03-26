import jax
import jax.numpy as np
import numpy as onp
import diffrax

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




# many of these functions probably don't work. they were hacked together within the testbed function
# and depend on some variables there. if needed again, put back there or include variables as proper arguments.

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

# sobolev_weight_gridsearch()



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

    # find terminal LQR controller and value function.
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # find a matrix mapping from the unit circle to the value level set
    # previously done with eigendecomposition -> sqrt of eigenvalues.
    # with the cholesky decomp the trajectories look about the same
    # qualitatively. it is nicer so we'll keep that.
    # cholesky decomposition says: P = L L.T but not L.T L
    L_lqr = np.linalg.cholesky(P_lqr)

    assert rnd(L_lqr @ L_lqr.T, P_lqr) < 1e-6, 'cholesky decomposition wrong or inaccurate'

    key = jax.random.PRNGKey(1)

    # purely random ass points for initial batch of trajectories.
    normal_pts = jax.random.normal(key, shape=(100, problem_params['nx']))
    unitsphere_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]
    xfs = unitsphere_pts @ np.linalg.inv(L_lqr) * np.sqrt(problem_params['V_f']) * np.sqrt(2)

    # test if it worked
    V_f = lambda x: 0.5 * x.T @ P_lqr @ x
    vfs = jax.vmap(V_f)(xfs)

    assert np.allclose(vfs, problem_params['V_f']), 'wrong terminal value...'


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

        return solve_backward(state_f)

    sols_orig = jax.vmap(solve_backward_lqr, in_axes=(0, None))(xfs, algo_params)


    def find_min_l(sols, v_lower, v_upper, problem_params):

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
        all_ls = jax.vmap(jax.vmap(l_of_y))(sols.ys)

        # add NaN to every x with v(x) > v_k 
        is_outside_valueband = ~np.logical_and(v_lower <= sols.ys['v'], sols.ys['v'] <= v_upper)
        all_ls_masked = all_ls + (is_outside_valueband * np.nan)

        min_l = np.nanmin(all_ls_masked)    
        return min_l



    def find_min_l_alt(sols, v_lower, v_upper, problem_params):

        # first extract the data, then calculate u* and l. 
        # seems to actually be slower than the other one :(

        def l_of_y(y):
            x = y['x']
            vx = y['vx']
            u = pontryagin_utils.u_star_2d(x, vx, problem_params)
            return problem_params['l'](x, u)

        # add NaN to every x with v(x) > v_k 
        is_outside_valueband = ~np.logical_and(v_lower <= sols.ys['v'], sols.ys['v'] <= v_upper)

        ys_relevant = jtm(lambda n: n[~is_outside_valueband], sols.ys)

        # double vmap because we have N_trajectories x N_timesteps ys
        all_ls = jax.vmap(l_of_y)(ys_relevant)

        min_l = np.nanmin(all_ls)    
        return min_l
    


    # lmin = find_min_l(sols_orig, 10, 20, problem_params)
    # ipdb.set_trace()
    
    # visualiser.plot_trajectories_meshcat(sols_orig)
    # ipdb.set_trace()

    # debug_nan_sol(sols_orig, problem_params, algo_params)
    # ipdb.set_trace()

    
    # choose initial value level.
    # v_k = 1000 * problem_params['V_f']
    # v_k = np.inf  # fullest gas
    v_k = 20

    # thin band
    # value_interval = [v_k/3, v_k]
    # thick band :) 
    value_interval = [v_k/5000, v_k]
    print(f'value interval: {value_interval}')


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
        vxx_norms = np.linalg.norm(sols.ys['vxx'], axis=(2, 3))
        vxx_acceptable = vxx_norms < algo_params['vxx_max_norm']  # some random upper bound based on looking at a plot of v vs ||vxx||

        bool_train_idx = np.logical_and(v_in_interval, vxx_acceptable)
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

    all_ys = select_train_pts(value_interval, sols_orig)

    # split into train/test set.

    train_ys, test_ys = nn_utils.train_test_split(all_ys, train_frac=algo_params['nn_train_fraction'])


    v_nn = nn_utils.nn_wrapper(
        input_dim=problem_params['nx'],
        layer_dims=algo_params['nn_layerdims'],
        output_dim=1
    )


    normaliser = nn_utils.data_normaliser(train_ys)

    ys_n = normaliser.normalise_all_dict(train_ys)
    test_ys_n = normaliser.normalise_all_dict(test_ys)

    # plot_distributions(ys_n)
    pl.show()
    # ipdb.set_trace()





    init_key, key = jax.random.split(key)
    params_init = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))

    # to get a feel for over/underparameterisation.
    n_params = count_floats(params_init)
    n_data = count_floats(train_ys)
    print(f'params/data ratio = {n_params/n_data:.4f}')

    # train once with new sobolev loss...
    train_key, key = jax.random.split(key)
    params_sobolev, oups_sobolev = v_nn.train_sobolev(
        train_key, ys_n, params_init, algo_params, ys_test=test_ys_n
    )


    # # and once without
    # algo_params_fake = algo_params.copy()
    # algo_params_fake['nn_sobolev_weights'] = algo_params['nn_sobolev_weights'].at[2].set(0.)
    # train_key, key = jax.random.split(key)
    # params, oups = v_nn.train_sobolev(
    #         train_key, ys_n, params_init, algo_params_fake, ys_test=test_ys_n
    # )

    pl.figure('sobolev with vxx')
    plotting_utils.plot_nn_train_outputs(oups_sobolev)

    params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble(
        train_key, ys_n, params_init, algo_params
    )

    v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

    idx = 20
    sol = jax.tree_util.tree_map(itemgetter(idx), sols_orig)

    pl.figure()
    plotting_utils.plot_trajectory_vs_nn(sol, params_sobolev, v_nn_unnormalised)

    pl.figure()
    plotting_utils.plot_trajectory_vs_nn_ensemble(sol, params_sobolev_ens, v_nn_unnormalised)
    pl.show()


    '''
    # sweep over epoch size.
    # conclusion: longer is better in this case up to about 2048. (512 almost as good though). after that
    # test loss goes up again. 
    for ep in 2**np.arange(12):
        print(ep)
        algo_params['nn_N_epochs'] = ep
        params, oups = v_nn.train_sobolev(train_key, ys_n, params_init, algo_params, ys_test=test_ys_n)
        plotting_utils.plot_nn_train_outputs(oups, legend=ep==1)
    pl.show()

    # or batch size

    train_key, key = jax.random.split(key)
    for bs in 2 ** np.arange(10):
        print(bs)
        algo_params['nn_batchsize'] = bs
        params, oups = v_nn.train_sobolev(train_key, ys_n, params_init, algo_params, ys_test=test_ys_n)
        plotting_utils.plot_nn_train_outputs(oups, legend=ep==1, alpha=0.1)
    '''



    # pl.savefig(f'tmp/vk_sweep_{i}_{v_k}.png')
    # pl.close('all')
    # pl.figure('sobolev without vxx')
    # plotting_utils.plot_nn_train_outputs(oups)
    '''

    # instead go straight to training a whole ensemble. 
    # surprisingly the progress bar just works now!
    train_key, key = jax.random.split(key)
    params_vmap, oups_vmap = v_nn.train_sobolev_ensemble(
        train_key, ys_n, params_init, algo_params, ys_test=test_ys_n,
    )

    # plot results
    for j in range(algo_params['nn_ensemble_size']):
        oups = jax.tree_util.tree_map(itemgetter(j), oups_vmap)
        plotting_utils.plot_nn_train_outputs(oups, alpha=.1, legend=j==0)
    '''


    if 'params' in oups_sobolev:

        # experimentally we output ALL parameters encountered during the whole training. 
        # this gives us effectively a huge NN ensemble, with (maybe) some sort of useful
        # model diversity. update: no, not really any useful model diversity, the extrapolation
        # tends to stay almost precis

        sliced_params = jtm(lambda node: node[-1024::128], oups_sobolev['params'])
        plotting_utils.plot_trajectory_vs_nn_ensemble(sol, sliced_params, v_nn_unnormalised)
        pl.show()







        ax = pl.subplot(211)

        # define smoothing kernel for nicer plots
        N = vxx_weights.shape[0] // 20
        smooth = lambda x: np.convolve(x, np.ones(N)/N, mode='valid')
        smooth_vmap = lambda x: jax.vmap(smooth)(x.T).T  # .T mess because pyplot plots columns not rows

        pl.loglog(vxx_weights, final_training_errs, linestyle='', marker='.', label=('v train loss', 'vx train loss', 'vxx train loss'), alpha=.5)
        pl.gca().set_prop_cycle(None)
        pl.loglog(smooth(vxx_weights), smooth_vmap(final_training_errs))
        pl.legend()

        pl.subplot(212, sharex=ax, sharey=ax)
        pl.loglog(vxx_weights, test_errs, linestyle='', marker='.', label=('v test loss', 'vx test loss', 'vxx test loss'), alpha=.5)
        pl.loglog(vxx_weights, hessian_rnds, linestyle='', marker='.', label='hessian rel norm diff at x=0', alpha=.5)
        pl.gca().set_prop_cycle(None)
        pl.loglog(smooth(vxx_weights), smooth_vmap(test_errs))
        pl.loglog(smooth(vxx_weights), smooth(hessian_rnds))
        pl.legend()
        pl.show()

        ipdb.set_trace()

    def v_meanstd(x, vmap_params):
        # evaluate the lower (beta<0) or upper (beta>0) confidence band of the value function. 
        # this serves as a *probable* overapproximation of the true value sublevel set.
        vs_ensemble = jax.vmap(v_nn_unnormalised, in_axes=(0, None))(vmap_params, x)

        v_mean = vs_ensemble.mean()
        v_std = vs_ensemble.std()

        return v_mean, v_std

    v_meanstds = jax.jit(jax.vmap(v_meanstd, in_axes=(0, None)))


    def forward_sim_lqr(x0):

        def forwardsim_rhs(t, x, args):

            lam_x = P_lqr @ x  # <- for lqr instead
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

    def forward_sim_nn(x0, params, vmap=False):

        if vmap:
            # we have a whole NN ensemble. use the mean here. 
            v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
            # mean across only axis resulting in a scalar. differentiate later.
            v_nn_unnormalised = lambda x: jax.vmap(v_nn_unnormalised_single, in_axes=(0, None))(params, x).mean()

        else:
            v_nn_unnormalised = lambda x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

        def forwardsim_rhs(t, x, args):

            lam_x = jax.jacobian(v_nn_unnormalised)(x).squeeze()
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


    def forward_sim_nn_until_value(x0, params, v_k, vmap=False):

        # also simulates forward, but stops once we are with high probability
        # inside the value level set v_k. (implemented as 2 sigma upper confidence band)

        # only vmap=True is implemented.

        if vmap:
            # we have a whole NN ensemble. use the mean here. 
            v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
            # mean across only axis resulting in a scalar. differentiate later.
            v_nn_unnormalised = lambda x: jax.vmap(v_nn_unnormalised_single, in_axes=(0, None))(params, x).mean()

        else:
            v_nn_unnormalised = lambda x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

        def forwardsim_rhs(t, x, args):

            lam_x = jax.jacobian(v_nn_unnormalised)(x).squeeze()
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
            has_low_sigma = v_std <= 0.5

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


    def solve_backward_nn_ens(x_f, vmap_params, algo_params):

        # modified from solve_backward_lqr. 

        # change this (lexical closure of normaliser) if during main iteration we change 
        # the data normalisation!!!

        v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
        # mean across only axis resulting in a scalar. differentiate later.
        v_nn_unnormalised = lambda x: jax.vmap(v_nn_unnormalised_single, in_axes=(0, None))(vmap_params, x).mean()

        v_lqr = lambda x: 0.5 * x.T @ P_lqr @ x

        # see if v_nn and v_lqr match up even here. 
        # all good, same-ish jacobian and hessian at 0. 
        # ipdb.set_trace()

        # v_f = 0.5 * x_f.T @ P_lqr @ x_f
        # vx_f = P_lqr @ x_f
        v_f = v_nn_unnormalised(x_f)
        vx_f = jax.jacobian(v_nn_unnormalised)(x_f)

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        if algo_params['pontryagin_solver_vxx']:
            vxx_f = jax.hessian(v_nn_unnormalised)(x_f)
            state_f['vxx'] = vxx_f

        return solve_backward(state_f)


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
    # sols_lqr = jax.vmap(forward_sim_lqr)(x0s)

    # visualiser.plot_trajectories_meshcat(sols, color=(.5, .7, .5))
    # visualiser.plot_trajectories_meshcat(sols_sobolev)
    # visualiser.plot_trajectories_meshcat(sols_sobolev_ens)
    # visualiser.plot_trajectories_meshcat(sols_lqr, color=(.4, .8, .4))



    def propose_pts(key, v_k, vmap_nn_params):

        # ~~~ a) estimate the "sensible" next value step ~~~

        # in the end we should verify if we actually reached that (= achieved low uncertainty within it)
        # if not, adjust it down so v_k always reflects the value level which we know to given tol.
        # this is probably best handled by the TrainAndPrune function. 

        # also, in an initial step we should verify that v_k represents an accurate value level set
        # (or just start with it really low, maybe just lqr solution too.)

        # be generous!
        # value_interval = [0., v_k*2]

        # instead calculate a more educated guess like this: 
        fastestpole_tau = .49  # from LQR solution. 
        T = 5 * fastestpole_tau
        min_l = find_min_l(sols_orig, v_k/2, v_k, problem_params)

        # min l = min dv/dt
        v_step = T * min_l
        v_next = v_k + v_step

        value_interval = [v_k, v_next]




        # ~~~ b) find uniformly sampled points from value band w/ rejection sampling ~~~
        # to "approximate" all kinds of global optimisation & sampling
        # operations over that set. 

        all_valueband_pts = np.zeros((0, problem_params['nx']))

        N_pts_desired = 1000  # we want 1000 points that are inside the value band. 
        i=0
        # key = jax.random.PRNGKey(k)
        while all_valueband_pts.shape[0] < N_pts_desired and i < 1000:

            i = i + 1   # a counter so we return if it never happens. 

            newkey, key = jax.random.split(key)

            # sample from "the whole state space".
            # this is kind of icky because we have to select a bounded region
            # also already in 6D if we specify a large region we have very few
            # samples actually in the value band. hence the loop outside. 
            # nicer solution would be some sort of mcmc thingy...? 
            # slightly better hack: set this region to like 1.5 * min and max
            # of current data...
            x_pts = jax.random.uniform(
                key=newkey,
                shape=(10000, problem_params['nx']),
                minval=np.array([-20, -20, -10*np.pi, -20, -20, -20*np.pi])/8,
                maxval=np.array([ 20,  20,  10*np.pi,  20,  20,  20*np.pi])/8,
            )

            # v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

            # what kind of points do we propose? we want points x such that: 
            # - x is withing the value band V_{k+1} \ V_k
            # - from those, we want the ones with largest uncertainty.

            v_means, v_stds = v_meanstds(x_pts, vmap_nn_params)

            optimistic_vs = v_means - 2 * v_stds

            # is_in_range = np.logical_and(value_interval[0] <= optimistic_vs, optimistic_vs <= value_interval[1])

            # only be optimistic for the outer boundary instead. 
            is_in_range = np.logical_and(value_interval[0] <= v_means, optimistic_vs <= value_interval[1])

            interesting_x0s = x_pts[is_in_range]

            all_valueband_pts = np.concatenate([all_valueband_pts, interesting_x0s], axis=0)
            print(all_valueband_pts.shape[0])

        if all_valueband_pts.shape[0] < N_pts_desired:
            print('did not find enough points!')
            ipdb.set_trace()

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
        # ipdb.set_trace()



        N_proposals = 200
        '''
        # scale -> 0 results in just the N_proposals points with highest std being chosen.
        # scale -> infinity results in the proposals being sampled uniformly at random. 
        std_scale = 3
        std_scale = .5

        # weights = np.exp(v_stds / std_scale)
        # ps = weights / np.sum(weights)
        # is this not just a softmax function?
        # -> yes, it is :)

        ps = jax.nn.softmax(v_stds / std_scale)

        proposals = jax.random.choice(key, all_valueband_pts, shape=(N_proposals,), replace=False, p=ps)

        all_idxs = np.arange(N_proposals)
        proposal_idxs = jax.random.choice(key, all_idxs, shape=(N_proposals,), replace=False, p=ps)

        proposed_pts = all_valueband_pts[proposal_idxs]

        # but that's kind of dumb, we give a small probability of also selecting points with exactly 
        '''

        # much simpler. 
        # possible problem: we select only "far" points with very large sigma, while neglecting
        # the ones that are closer which maybe we should do first to even reach the far points
        _, proposal_idxs = jax.lax.top_k(v_stds, N_proposals)


        # next idea: set some maximum acceptable sigma, and take the lowest-v k points that exceed it. 
        # or a uniform subsample of all the ones that exceed it. 
        v_std_min = 0.5
        where_uncertain = v_stds > v_std_min

        # replaces the v_means where we are certain enough by inf
        # that way once we multiply by -1 we have -inf and negative values
        # the largest negative values = the smallest positive values
        v_means_uncertain = v_means + ~where_uncertain * np.inf
        _, proposal_idxs = jax.lax.top_k(-v_means_uncertain, N_proposals)


        # visualise the selection of points. 
        pl.plot(v_means, v_stds, '. ', label='all points in value band')
        pl.plot(v_means[proposal_idxs], v_stds[proposal_idxs], '. ', label='selected points')


        proposed_states = all_valueband_pts[proposal_idxs]

        return proposed_states, v_next



    def batched_oracle(proposals, v_k, vmap_nn_params):

        # forward simulation. this stops if BOTH of these conditions hold.
        # - v_mean + 2 * v_sigma <= v_k
        # - v_sigma <= 0.5
        # so we can be quite sure the information at that point is usable.
        # (also stops if time horizon ends. )
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
        is_usable = np.logical_and(mus + 2 * sigs <= v_k, sigs <= 0.5)

        assert (stopped_bc_terminatingevent == is_usable).all(), 'shit happened'

        print(f'{100*is_usable.mean():.2f}% of forward simulations reached lower value level set AND low sigma.')

        # if we have a different amount every time, we cannot jit the simulation.
        # therefore we just mark it as nan and try to tune the algo such that not too many 
        # of them are nan. 
        usable_xfs = xfs.at[~is_usable].set(np.nan)

        # as we kind of would expect, is_usable correlates clearly (negatively) with the amount of 
        # solver steps. so the most effort is spent calculating solutions which we're never going to use. 
        # could we somehow avoid this? maybe stop after 3/4 of solutions have terminated? probably but 
        # then the implementation becomes messier, because plain vmap doesn't allow cross communication.
        # more simply: just set a rather low step limit and be fine with a couple more solutions being 
        # thrown out. 

        # or just don't care, forward sim is cheaper than backward anyway. (is
        # it? with vmapped nn ensemble maybe not..., certainly not if backward sim
        # is without vxx.)

        # here put some new limit on how far we solve backward? 
        # stop at like 5*value_interval[1]? 
        # or use a fixed multiple of the forward sim time? 
        # both? 
        # decrease pontryagin_solver_T overall??
        sol0 = solve_backward_nn_ens(usable_xfs[0], vmap_nn_params, algo_params)

        # TODO jit this. 
        backward_sols_new = jax.vmap(solve_backward_nn_ens, in_axes=(0, None, None))(usable_xfs, vmap_nn_params, algo_params)


        # plot 0th forward and backward sol in same plot.
        pl.figure()
        pl.subplot(221)
        sol0_fwd = jtm(itemgetter(0), forward_sols)
        fwd_ts_adjusted = sol0_fwd.ts - sol0_fwd.ts[sol0_fwd.stats['num_accepted_steps']]
        

        pl.plot(fwd_ts_adjusted, sol0_fwd.ys)
        pl.gca().set_prop_cycle(None)
        plotting_utils.plot_sol(sol0, problem_params)



        return backward_sols_new

    # initial step: 
    # - generate data from uniform backward shooting
    # - do train_and_prune step to find value nn and known value level.
    # - start loop

    for k in range(5):

        # active learning with level-set ideas embedded. 
        # first pseudocode algo in idea dump

        k = jax.random.PRNGKey(k)
        proposed_pts, v_next_target = propose_pts(k, v_k, params_sobolev_ens)

        # ~~~~ ORACLE ~~~~ 
        # now that we've proposed a batch of points, we call the oracle.
        # it consists of these steps:
        # - forward simulation, with approximate optimal input from NN, 
        #   until we reach known value level
        # - backward PMP shooting as usual.
        backward_sols_new = batched_oracle(proposed_pts, v_k, params_sobolev_ens)

  








        # maybe easier to write one function for the whole oracle step, and then vmap it?
        # instead of vmapping the forward solve and backward solve separately...


        ipdb.set_trace()

        
        # this just a very short pseudocode-like draft from the beginning.
        # x_proposals = propse_points(v_nn, vk)
        # sols = jax.vmap(solve_backward, in_axes=(0, None))(x_proposals, algo_params)

        # somehow store the solutions in our big dataset
        data = np.concatenate([data, sols.ys])

        params, v_known = train_and_prune(v_nn, params, v_known, data)

    '''
    '''



    # plot_taylor_sanitycheck(jax.tree_util.tree_map(itemgetter(0), sols_orig))
    pl.show()
    ipdb.set_trace()



