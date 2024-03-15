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


def debug_nan_sol(sols_orig, solve_backward):
    # long debugging session.

    # conclusion: if NaNs pop up or the vxx terms become unreasonably large,
    # try decreasing dtmin a bit. fixed everything in this case.

    # just the first one
    # nan_idx = np.where(np.isnan(sols_orig.ys['vxx']).any(axis=(1,2,3)))[0].item()
    nan_idx = 153
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

    def plot_badsol_from_y(y):

        # recreate the solution in 64 bit precision.
        newsol = solve_backward(y, algo_params)

        plotting_utils.plot_sol(newsol, problem_params)

    # plot_badsol_from_idx(30)
    # pl.show()

    # ipdb.set_trace()


    # start sol with higher precision from a state close to the last one.
    restart_state_idx = bad_sol.stats['num_accepted_steps'].item() - 5

    # not really needed, still fails, rhs actually does return very high values :(
    # restart_y = jax.tree_util.tree_map(itemgetter(restart_state_idx), bad_sol.ys)

    # algo_params_tight = algo_params.copy()
    # algo_params_tight['pontryagin_solver_atol'] = 1e-7
    # algo_params_tight['pontryagin_solver_rtol'] = 1e-7

    # # first arg irrelevant if last given
    # sol_tight = solve_backward(restart_y['x'], algo_params_tight, y_f=restart_y)

    # # evaluate the right hand side again to see where it produces shit.
    # rhs_evals = jax.vmap(f_extended, in_axes=(0, 0, None))(sol_tight.ts, sol_tight.ys, None)
    rhs_evals_orig = jax.vmap(f_extended, in_axes=(0, 0, None))(bad_sol.ts, bad_sol.ys, None)

    rhs_evals, aux_outputs = jax.vmap(f_extended, in_axes=(0, 0, None))(bad_sol.ts, bad_sol.ys, 'debug')

    ax = pl.subplot(211)
    pl.plot(bad_sol.ts, bad_sol.ys['vxx'].reshape(257, -1), '.-', c='C0', alpha=.5)
    pl.ylabel('vxx trajectory components')
    pl.subplot(212, sharex=ax)
    pl.plot(bad_sol.ts, rhs_evals['vxx'].reshape(257, -1), '.-', c='C0', alpha=.5)
    pl.ylabel('vxx rhs components')

    pl.figure()
    plotting_utils.plot_sol(bad_sol)

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
    normal_pts = jax.random.normal(key, shape=(200, problem_params['nx']))
    unitsphere_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]
    xfs = unitsphere_pts @ np.linalg.inv(L_lqr) * np.sqrt(problem_params['V_f']) * np.sqrt(2)

    # test if it worked
    V_f = lambda x: 0.5 * x.T @ P_lqr @ x
    vfs = jax.vmap(V_f)(xfs)

    assert np.allclose(vfs, problem_params['V_f']), 'wrong terminal value...'












    solve_backward, f_extended = pontryagin_utils.define_backward_solver(
        problem_params, algo_params
    )

    def solve_backward_lqr(x_f, algo_params):

        # P_lqr = hessian of value fct.
        # everything else follows from usual differentiation rules.

        v_f = 0.5 * x_f.T @ P_lqr @ x_f
        vx_f = P_lqr @ x_f
        vxx_f = P_lqr

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
            'vxx': vxx_f,
        }

        return solve_backward(state_f, algo_params)

    sols_orig = jax.vmap(solve_backward_lqr, in_axes=(0, None))(xfs, algo_params)


    '''
    # this is only really useful if computing much more than N_plot trajectories...
    # plot only the N_plot ones with lowest initial cost-to-go.
    v0s = jax.vmap(lambda s: s.evaluate(s.t1)['v'])(sols_orig)
    N_plot = 50

    # the highest v0 we're going to plot
    v0s = np.nan_to_num(v0s, nan=np.inf)
    v0_cutoff = np.percentile(v0s, N_plot/v0s.shape[0] * 100)

    bool_plot_idx = v0s <= v0_cutoff

    sols_plot = jax.tree_util.tree_map(lambda node: node[bool_plot_idx], sols_orig)
    sols_notplot = jax.tree_util.tree_map(lambda node: node[~bool_plot_idx], sols_orig)

    visualiser.plot_trajectories_meshcat(sols_plot)

    pl.plot(sols_plot.ys['t'].flatten(), sols_plot.ys['v'].flatten(), alpha=.2, c='C0', label='plotted sols')
    pl.plot(sols_notplot.ys['t'].flatten(), sols_notplot.ys['v'].flatten(), alpha=.2, c='C1', label='not plotted sols')
    pl.legend()
    pl.show()

    '''
    # visualiser.plot_trajectories_meshcat(sols_orig)

    # find some way to alert if NaN occurs in this whole pytree?
    # wasn't there a jax option that halts on nan immediately??


    '''
    for j in range(10, 15):
        pl.figure(str(j))
        plotting_utils.plot_sol(jax.tree_util.tree_map(itemgetter(j), sols_orig), problem_params)
    pl.show()
    '''

    # choose initial value level.
    v_k = 1000 * problem_params['V_f']
    v_k = np.inf  # fullest gas
    v_k = 2000
    print(f'target value level: {v_k}')
    pl.rcParams['figure.figsize'] = (16, 10)


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


    v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

    idx = 20
    sol = jax.tree_util.tree_map(itemgetter(idx), sols_orig)

    pl.figure()
    plotting_utils.plot_trajectory_vs_nn(sol, params_sobolev, v_nn_unnormalised)

    pl.show()





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

    # vxx_weight_sweep()



    def forward_sim_nn(x0, params):

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
        )

        return forward_sol



    x0s = jax.random.normal(jax.random.PRNGKey(0), shape=(200, 6)) * .2

    # sol = forward_sim_nn(x0s[0], params)
    # sols         = jax.vmap(forward_sim_nn, in_axes=(0, None))(x0s, params)
    sols_sobolev = jax.vmap(forward_sim_nn, in_axes=(0, None))(x0s, params_sobolev)

    # visualiser.plot_trajectories_meshcat(sols, color=(.5, .7, .5))
    visualiser.plot_trajectories_meshcat(sols_sobolev)

    ipdb.set_trace()

    # initial step: 
    # - generate data from uniform backward shooting
    # - do train_and_prune step to find value nn and known value level.
    # - start loop

    for k in range(5):

        # active learning with level-set ideas embedded. 
        # first pseudocode algo in idea dump

        x_proposals = propse_points(v_nn, vk)

        sols = jax.vmap(solve_backward, in_axes=(0, None))(x_proposals, algo_params)

        # somehow store the solutions in our big dataset
        data = np.concatenate([data, sols.ys])

        params, v_known = train_and_prune(v_nn, params, v_known, data)

    '''
    '''



    # plot_taylor_sanitycheck(jax.tree_util.tree_map(itemgetter(0), sols_orig))
    pl.show()
    ipdb.set_trace()



