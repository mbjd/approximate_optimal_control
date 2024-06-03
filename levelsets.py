import gzip
import os
import pprint
import subprocess
import sys
import time
from operator import itemgetter

import diffrax
import equinox
import flax
import ipdb
import jax
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as onp
import tqdm

import nn_utils
import plotting_utils
import pontryagin_utils
import visualiser
import wandb
from misc import *

# helper functions {{{


def def_v_meanstds(v_nn):

    def v_meanstd(x, vmap_params):

        # find (empirical) mean and std. dev of value function.
        vs_ensemble = jax.vmap(v_nn, in_axes=(0, None))(vmap_params, x)

        v_mean = vs_ensemble.mean()
        v_std = vs_ensemble.std()

        return v_mean, v_std

    def vx_meanstd(x, vmap_params):

        # vmap for nn ensemble.
        vx_fct = jax.jacobian(v_nn, argnums=1)
        ensemble_vxs = jax.vmap(vx_fct, in_axes=(0, None))(vmap_params, x)

        # now we have all_vxs.shape == (N_ensemble, nx)
        # we want ensemble mean and std across axis 0.
        # stds will be individual for each coordinate, sum/mean whatever later if you want.
        vx_mean = ensemble_vxs.mean(axis=0)
        vx_std = ensemble_vxs.std(axis=0)

        return vx_mean, vx_std

    v_meanstds = jax.vmap(v_meanstd, in_axes=(0, None))
    vx_meanstds = jax.vmap(vx_meanstd, in_axes=(0, None))
    return v_meanstds, vx_meanstds


def find_min_l(ys, v_lower, v_upper, problem_params):

    # find the smallest value of l(x, u) in the given value band
    # in the dataset.

    # yet another alternative: from each trajectory find the highest-v
    # point below v_upper. not constant size but let's not care about those
    # superficialities

    # same as above
    all_traj_idx = np.arange(ys['v'].shape[0])
    top_per_traj_idx = np.argmax(ys['v'] * (ys['v'] < v_upper), axis=1)
    ys_top = jtm(lambda node: node[all_traj_idx, top_per_traj_idx], ys)

    # but only use the trajectories that did not yet stop.
    # = trajectories that have some v > v_upper.
    crosses_v_upper = ((ys['v'] >= v_upper) & (ys['v'] < np.inf)).any(axis=1)

    # but now caluclate all those l(x, u).

    def l_of_y(y):
        x = y['x']
        vx = y['vx']
        u = pontryagin_utils.u_star_general(x, vx, problem_params)
        return problem_params['l'](x, u)

    ls = jax.vmap(l_of_y)(ys_top)
    ls_relevant = ls + np.nan * (~crosses_v_upper)
    min_l = np.nanmin(ls_relevant)

    return min_l


def set_value_target(all_ys, v_k, problem_params, algo_params):

    # value target = largest possible value target that still
    # contains trajectories with time duration <= T_value_target.
    # with this data based min_l as a surrogate for the actual
    # min l over the next level band.

    # use actual previous value level instead?
    min_l = find_min_l(all_ys, v_k/2, v_k, problem_params)

    # so min value step to ensure horizon <= T is T * smallest dv/dt
    # min l = min dv/dt
    v_step = algo_params['T_value_target'] * min_l
    v_next = v_k + v_step

    return v_next


def forward_sim_nn(x0, v_nn, params, problem_params, algo_params, ensemble=True, T=10.):

    # now simulates with state = {'x': system state, 'cost': control cost}.

    if ensemble:
        # we have a whole NN ensemble. use the mean here.
        # v_nn_unnormalised_single = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
        # mean across only axis resulting in a scalar. differentiate later.
        v_fct = lambda x: jax.vmap(v_nn, in_axes=(0, None))(params, x).mean()

    else:
        v_fct = lambda x: v_nn(params, x)

    def forwardsim_rhs(t, y, args):

        x = y['x']
        cost = y['cost']

        lam_x = jax.jacobian(v_fct)(x).squeeze()
        # lam_x = P_lqr @ x  # <- for lqr instead
        u = pontryagin_utils.u_star_general(x, lam_x, problem_params)
        return {
                'x': problem_params['f'](x, u),
                'cost': problem_params['l'](x, u),
        }


    term = diffrax.ODETerm(forwardsim_rhs)

    step_ctrl = diffrax.PIDController(
        atol=algo_params['pontryagin_solver_atol'],
        rtol=algo_params['pontryagin_solver_rtol'],
        dtmin=algo_params['dtmin'],
        dtmax=algo_params['dtmax'],
    )

    saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

    if problem_params['m'] is not None and algo_params['project_manifold']:
        # projection only for state, not cost ofc
        project = lambda y: {'x': problem_params['project_M'](y['x']), 'cost': y['cost']}
        solver = pontryagin_utils.ProjectionSolver(project=project)
    else:
        solver = diffrax.Tsit5()

    # start at 0 cost. the incurred cost is integrated up. in the end
    # to estimate inf horizon cost, either integrate for very long, or add terminal LQR.

    y0 = {
        'x': x0,
        'cost': 0.,
    }

    forward_sol = diffrax.diffeqsolve(
        term, solver, t0=0., t1=T, dt0=0.01, y0=y0,
        stepsize_controller=step_ctrl, saveat=saveat,
        max_steps = algo_params['pontryagin_solver_maxsteps'],
        throw=algo_params['throw'],
    )

    return forward_sol


def meshcat_forward_sims(x0s, v_nn, nn_params, problem_params, algo_params):

    # just a couple of steps I find myself doing in pdb all the time

    sim = lambda x0: forward_sim_nn(x0, v_nn, nn_params, problem_params, algo_params)
    trajs = jax.vmap(sim)(x0s)

    # convert to old (theta) repr. ugly hardcoded i know
    ys = jax.vmap(jax.vmap(lambda x: np.concatenate([x[0:2], np.array([np.arctan2(x[2], x[3])]), x[4:]])))(trajs.ys['x'])

    solsdict = {'t': trajs.ts, 'x': ys}

    visualiser.plot_trajectories_meshcat(solsdict)

    # also plot initial values.
    # pl.figure('meshcat sims: initial v mean/std')
    # v_means, v_stds = v_meanstds(x0s, nn_params)
    # ts = np.linspace(0, 1, x0s.shape[0])
    # pl.plot(ts, v_means, c='C0', label='v mean')
    # pl.fill_between(ts, v_means-v_stds, v_means+v_stds, color='C0', alpha=.2, label='1σ confidence')
    # pl.legend()
    # pl.show()


# }}}



# define main active learning ingredients: prune & train function {{{

def prune_and_train(key, v_nn, params_sobolev_ens, all_ys, v_interval, previously_suboptimal, problem_params, algo_params, warmstart=False, is_final=False):

    # these steps:
    # 1. mark data which we already know to be suboptimal as such
    #    (based on knowing a better solution at that point already)
    # 2. train the nn for the 1st time
    #    while gradually expanding the domain of training data (algo_params['nn_value_sweep'])
    #    with huber type losses to not break everything on conflicting data
    # 3. remove (= mark suboptimal) all the data that falls into the linear huber regions
    #    meaning the NN could not fit it (easily enough).
    # 4. train a second time with this "cleaned" dataset, just to remove the
    #    artefacts from outlier data in first training round, by settling into the
    #    equilibrium between gradient & weight decay.

    # use the is_final flag to do the final training round. changes these things if True:
    # - no pruning before training, all data & suboptimality flags are used as is
    # - overrides thin_data, we need the whole dataset

    # 0. redefining functions that were previously stolen from main's scope {{{

    def def_v_meanstds(v_nn):

        def v_meanstd(x, vmap_params):

            # find (empirical) mean and std. dev of value function.
            vs_ensemble = jax.vmap(v_nn, in_axes=(0, None))(vmap_params, x)

            v_mean = vs_ensemble.mean()
            v_std = vs_ensemble.std()

            return v_mean, v_std

        def vx_meanstd(x, vmap_params):

            # vmap for nn ensemble.
            vx_fct = jax.jacobian(v_nn, argnums=1)
            ensemble_vxs = jax.vmap(vx_fct, in_axes=(0, None))(vmap_params, x)

            # now we have all_vxs.shape == (N_ensemble, nx)
            # we want ensemble mean and std across axis 0.
            # stds will be individual for each coordinate, sum/mean whatever later if you want.
            vx_mean = ensemble_vxs.mean(axis=0)
            vx_std = ensemble_vxs.std(axis=0)

            return vx_mean, vx_std

        v_meanstds = jax.vmap(v_meanstd, in_axes=(0, None))
        vx_meanstds = jax.vmap(vx_meanstd, in_axes=(0, None))
        return v_meanstds, vx_meanstds

    v_meanstds, vx_meanstds = def_v_meanstds(v_nn)

    # }}}

    # 1. mark clearly suboptimal data. {{{

    v_lower, v_upper = v_interval


    # now without the extra dim the vmap we already did is sufficient
    # v_nn_means, v_nn_stds = v_meanstds(all_ys['x'], params_sobolev_ens)


    pruning_metrics = {}

    if is_final:

        # in the final round just use the suboptimality flag as is
        is_suboptimal = previously_suboptimal

    else:

        v_nn_means, v_nn_stds = jax.vmap(v_meanstds, in_axes=(0, None))(all_ys['x'], params_sobolev_ens)

        # otherwise we might throw out some data already here if we know a better solution
        if algo_params['pruning_strategy'] in ('conservative', 'conservative_past'):

            # these two are now the same -- the cumsum step which
            # differentiated them is now done after all these strategies.  no
            # matter how we conclude suboptimality of any point, the preceding
            # ones will also be suboptimal due to dynamic programming principle

            # be conservative: only prune POINTS (not trajectories) that
            # definitely (with high prob) are outside of value level set
            nn_v_likely_in_levelset = v_nn_means + 3 * v_nn_stds < v_lower
            trajectory_outside_levelset = v_lower < all_ys['v']

            is_suboptimal = trajectory_outside_levelset & nn_v_likely_in_levelset



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

        else:
            pruning_strategy = algo_params['pruning_strategy']
            raise ValueError(f'unknown pruning strategy "{pruning_strategy}"')


        # do this cumsum step here? for ANY pruning strategy this is the
        # reasonable last step...
        # time goes from 0.0 at idx 0 to negative values at idx 1, 2, ... so
        # cumsum marks as suboptimal the PRECEDING points in physical time even
        # though in array indices they are the subsequent ones. all correct.
        is_suboptimal = np.cumsum(is_suboptimal, axis=1) > 0

        # keep suboptimal points marked suboptimal
        is_suboptimal = np.logical_or(previously_suboptimal, is_suboptimal)

    # next step: build training data out of this pruned mess.
    in_band = (all_ys['v'] <= v_upper)

    # this has proven not to be a great idea...
    if algo_params['include_future_data']:
        # just randomly throw in a bit more data for training.
        v_upper_train = v_upper + (v_upper - v_lower)
        in_band = (all_ys['v'] <= v_upper_train)


    if algo_params['thin_data'] and not is_final:
        # much simpler strategy: just exclude way past data.
        v_cutoff = v_lower / algo_params['thin_data_denominator']
        in_band = in_band & (v_cutoff <= all_ys['v'])

    bool_train_idx = in_band & ~is_suboptimal
    # }}}

    # 2. train the NN for the first time. {{{

    usable_ys = jax.tree_util.tree_map(lambda node: node[bool_train_idx], all_ys)

    # split into train/test set.
    train_ys, test_ys = nn_utils.train_test_split(usable_ys, train_frac=algo_params['nn_train_fraction'])

    train_key, key = jax.random.split(key)

    params_old = params_sobolev_ens

    # final training round flag. in the final training round we have saved
    # data from the run and want to retrain the nn with ALL data. but we
    # have no nn params, we cannot warmstart.
    # if is_final:
    #     warmstart = False
    # do not do this ^^ anymore, decide for yourself with the warmstart flag if you want it

    if warmstart:
        # continue from previous params, only last portion of training.
        # since we are doing this sweep, can we do EVERYTHING with tiny learning rate instead?
        params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble_warmstarted(
            train_key, train_ys, v_lower, v_upper, params_sobolev_ens, problem_params, algo_params
        )
    else:
        # training from scratch
        # raise NotImplementedError('are you sure? not really doing this anymore. plz implement v sweep here too')
        # BUT with twice v_upper -- these two are for the sweep, NOT the entire value interval.
        # and in this case we want no sweep, we want all data at once.
        params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble(
            train_key, train_ys, v_upper, v_upper, problem_params, algo_params
        )

    n_params = count_floats(params_sobolev_ens) / algo_params['nn_ensemble_size']
    n_data = count_floats(train_ys)
    pruning_metrics['params_data_ratio'] = n_params / n_data

    # mean of the last couple iterations.
    final_trainloss = oups_sobolev_ens['lossterms']['total_loss'][:, -100:].mean()

    # and loss over test set.
    test_losses, test_lossterms = jax.vmap(v_nn.sobolev_loss_batch_mean, in_axes=(None, 0, None, None, None))(key, params_sobolev_ens, test_ys, problem_params, algo_params)
    final_testloss = np.mean(test_losses)

    # weight norm is not stochastically approximated so we can use just the last one.
    final_weightnorm = oups_sobolev_ens['weight_norm'][:, -1].mean()

    pruning_metrics['final_trainloss'] = final_trainloss
    pruning_metrics['final_testloss'] = final_testloss
    pruning_metrics['final_weightnorm'] = final_weightnorm
    # }}}


    # in the final training round we already know all the suboptimality
    # flags and presume they are correct. so this is not needed anymore.
    if not is_final:

        # 3. classify outliers
        # {{{

        # evaluate all this stuff again yolo
        v_means_trained, v_stds_trained = v_meanstds(usable_ys['x'], params_sobolev_ens)
        vx_means_trained, vx_stds_trained = vx_meanstds(usable_ys['x'], params_sobolev_ens)

        # sobolev loss inner must be vmapped along axes y, v_pred, vx_pred.
        # this means: in_axes = (None, 0, 0, 0, None, None)

        all_losses, all_auxs = jax.vmap(v_nn.sobolev_loss_inner, in_axes = (None, 0, 0, 0, None, None))(
            key, usable_ys, v_means_trained, vx_means_trained, problem_params, algo_params
        )

        v_outliers = all_auxs['v_loss_linear']
        print(f'v outliers:      {100*v_outliers.mean():.3f}%')
        vx_outliers = all_auxs['vx_loss_linear']
        print(f'vx outliers:     {100*vx_outliers.mean():.3f}%')
        print(f'either outliers: {100*(vx_outliers|v_outliers).mean():.3f}%')
        print(f'both outliers:   {100*(vx_outliers&v_outliers).mean():.3f}%')

        # these boolean idxs are all with respect to usable_ys.
        is_outlier = all_auxs['vx_loss_linear'] | all_auxs['v_loss_linear']  # is & better here?
        is_new = (v_lower <= usable_ys['v']) & (usable_ys['v'] <= v_upper)

        new_suboptimal = is_outlier & is_new


        # now: update the full is_suboptimal array with these new indices
        # is_suboptimal[bool_train_idx] = new_suboptimal
        # (by construction of train idx, is_suboptimal[bool_train_idx] == False
        is_suboptimal = is_suboptimal.at[bool_train_idx].set(new_suboptimal)

        # then the cumsum thing
        is_suboptimal = np.cumsum(is_suboptimal, axis=1) > 0

        # }}}

        # 4. second training run.

        # {{{
        # TODO last thing: switch huber loss to quadratic loss here.
        # --> probably not relevant if removing outliers anyway.

        bool_train_idx = in_band & ~is_suboptimal
        usable_ys = jax.tree_util.tree_map(lambda node: node[bool_train_idx], all_ys)
        train_ys, test_ys = nn_utils.train_test_split(usable_ys, train_frac=algo_params['nn_train_fraction'])

        # shorter second training run, just to find an equilibrium of data vs weight decay.
        algo_params_second = algo_params.copy()
        algo_params_second['lr_init'] = algo_params['lr_final']
        algo_params_second['nn_N_epochs'] = algo_params['nn_N_epochs'] / 8

        if warmstart:
            # continue from previous params, only last portion of training.
            # also don't do the sweep anymore -- always sample up to v_upper.
            params_sobolev_ens, oups_sobolev_ens_new = v_nn.train_sobolev_ensemble_warmstarted(
                train_key, train_ys, v_upper, v_upper, params_sobolev_ens, problem_params, algo_params_second
            )
        else:
            # training from scratch
            raise NotImplementedError('are you sure? not really doing this anymore. plz implement v sweep here too')
            params_sobolev_ens, oups_sobolev_ens_new = v_nn.train_sobolev_ensemble(
                train_key, train_ys, problem_params, algo_params_second
            )


        # mean of the last couple iterations.
        final_trainloss = oups_sobolev_ens_new['lossterms']['total_loss'][:, -100:].mean()

        # and loss over test set.
        test_losses, test_lossterms = jax.vmap(v_nn.sobolev_loss_batch_mean, in_axes=(None, 0, None, None, None))(key, params_sobolev_ens, test_ys, problem_params, algo_params)
        final_testloss = np.mean(test_losses)

        final_weightnorm = oups_sobolev_ens_new['weight_norm'][:, -1].mean()

        pruning_metrics['final_trainloss_second'] = final_trainloss
        pruning_metrics['final_testloss_second'] = final_testloss

        pruning_metrics['final_weightnorm_second'] = final_weightnorm

        # all these shapes are (N_nn_ensemble, N_trainsteps) -- ofc we want concatenation along trainsteps
        oups_sobolev_ens = jtm(lambda a, b: np.concatenate([a, b], axis=1), oups_sobolev_ens, oups_sobolev_ens_new)
        # }}}

    return params_sobolev_ens, oups_sobolev_ens, is_suboptimal, pruning_metrics
    # }}}


def main(problem_params, algo_params):


    print(f'jax default backend = {jax.default_backend()}')
    pl.rcParams['figure.figsize'] = (16, 10)

    key = jax.random.PRNGKey(algo_params['seed'])

    # find terminal LQR & xfs {{{

    # then define a function unitsphere_to_dXf, which we then feed with uniform
    # points from the unitsphere to arrive at boundary conditions for first
    # backward shooting step. (if initial_shooting == 'lqr' not quite)


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
        unitsphere_to_dXf = lambda x: problem_params['x_eq'] + x.T @ np.linalg.inv(L_lqr) * np.sqrt(problem_params['V_f']) * np.sqrt(2)


    # set xfs for initial batch of trajectories, depending on chosen
    # method.

    if algo_params['initial_shooting'] == 'uniform':

        # purely random ass points for initial batch of trajectories.
        normal_pts = jax.random.normal(key, shape=(algo_params['initial_batchsize'], problem_params['nx']))
        unitsphere_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]
        xfs = jax.vmap(unitsphere_to_dXf)(unitsphere_pts)

    elif algo_params['initial_shooting'] == 'lqr':

        def forward_sim_lqr_until_value(x0, P_lqr, v_goal):

            # simulate forward using LQR value function.
            # stop once we hit the desired value.

            def forwardsim_rhs(t, x, args):

                lam_x = P_lqr @ (x - problem_params['x_eq'])  # <- for lqr instead
                u = pontryagin_utils.u_star_general(x, lam_x, problem_params)

                # u = -K_lqr @ (x - problem_params['x_eq'])
                return problem_params['f'](x, u)


            term = diffrax.ODETerm(forwardsim_rhs)
            step_ctrl = diffrax.PIDController(
                atol=algo_params['pontryagin_solver_atol'],
                rtol=algo_params['pontryagin_solver_rtol'],
                dtmin=algo_params['dtmin'],
                dtmax=algo_params['dtmax'],
            )

            saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)


            def event_fn(state, **kwargs):

                x_err = state.y - problem_params['x_eq']
                lqr_value = 0.5 * x_err @ P_lqr @ x_err
                return lqr_value <= v_goal

            terminating_event = diffrax.DiscreteTerminatingEvent(event_fn)

            if problem_params['m'] is not None and algo_params['project_manifold']:
                solver = pontryagin_utils.ProjectionSolver(project=problem_params['project_M'])
            else:
                solver = diffrax.Tsit5()

            forward_sol = diffrax.diffeqsolve(
                term, solver, t0=0., t1=10., dt0=0.01, y0=x0,
                stepsize_controller=step_ctrl, saveat=saveat,
                max_steps = algo_params['pontryagin_solver_maxsteps'],
                throw=algo_params['throw'],
                discrete_terminating_event=terminating_event,
            )

            return forward_sol

        # sample uniform random points from surface of unit ball ||x|| = 1
        key, normalkey = jax.random.split(key)
        normal_pts = jax.random.normal(key, shape=(algo_params['initial_batchsize'], problem_params['nx']))
        unitball_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]

        # probably interior is more 'correct' here but surface should work
        # too. to make it the interior i think it is multiplication by
        # Unif([0, 1]) ** (1/n).

        # copied from above but with higher value level
        if problem_params['m'] is not None:
            unitsphere_to_dV_linear = Proj_tangent.T @ np.linalg.inv(L_lqr_tangent) @ Proj_tangent * np.sqrt(algo_params['v_init']) * np.sqrt(2)
            unitsphere_to_dV = lambda x: problem_params['project_M'](problem_params['x_eq'] + x.T @ unitsphere_to_dV_linear)
        else:
            unitsphere_to_dV = lambda x: problem_params['x_eq'] + x.T @ np.linalg.inv(L_lqr) * np.sqrt(algo_params['v_init']) * np.sqrt(2)

        x0s = jax.vmap(unitsphere_to_dV)(unitball_pts)
        sols = jax.vmap(forward_sim_lqr_until_value, in_axes=(0, None, None))(x0s, P_lqr, problem_params['V_f'])

        # pl.figure('forward solver m(x)')
        # pl.plot(jax.vmap(jax.vmap(problem_params['m']))(sols.ys).T, c='black', alpha=.1)
        # pl.show()

        xfs_unprojected = jax.vmap(lambda sol: sol.ys[sol.stats['num_accepted_steps']])(sols)
        xfs = jax.vmap(problem_params['project_M'])(xfs_unprojected)

        # mark the ones that stopped due to time or step limit as unusable
        # because only the ones stopped due to DiscreteTerminatingEvent reached
        # the low value sublevel set where we accept the LQR solution.
        stopped_bc_terminatingevent = sols.result == 1
        xfs = xfs.at[~stopped_bc_terminatingevent].set(np.nan)


    else:

        name = algo_params['initial_shooting']
        raise ValueError(f'initial shooting method {name} does not exist')




    # test if it worked
    V_f = lambda x: 0.5 * (x - problem_params['x_eq']).T @ P_lqr @ (x - problem_params['x_eq'])
    vfs = jax.vmap(V_f)(xfs)

    # }}}


    # define lots of boring ass functions  {{{

    solve_backward, f_extended = pontryagin_utils.define_backward_solver(
        problem_params, algo_params
    )

    def solve_backward_lqr(x_f, algo_params):

        # P_lqr = hessian of value fct.
        # everything else follows from usual differentiation rules.

        # V_f = lambda x: 0.5 * (x - problem_params['x_eq']).T @ P_lqr @ (x - problem_params['x_eq'])
        v_f = V_f(x_f)
        vx_f = jax.jacobian(V_f)(x_f)


        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        if algo_params['pontryagin_solver_vxx']:
            vxx_f = P_lqr
            state_f['vxx'] = vxx_f

        return solve_backward(state_f, v_upper=10. * algo_params['v_init'])

    sols_orig = jax.vmap(solve_backward_lqr, in_axes=(0, None))(xfs, algo_params)


    if problem_params['system_name'] == 'orbits' and algo_params['showfigs']:
        thetas = np.linspace(-np.pi, np.pi, 300)
        circle = np.array([np.sin(thetas), np.cos(thetas)]).T
        pl.plot(circle[:, 0], circle[:, 1], c='black', alpha=.1, linestyle='--')
        if algo_params['initial_shooting'] == 'lqr':
            pl.plot(*np.split(sols.ys.reshape(-1, 2), [1], axis=1), '.-', label='forward sols', alpha=.1)
        pl.plot(*np.split(sols_orig.ys['x'].reshape(-1, 2), [1], axis=1), '.-', label='backward sols', alpha=.1)


        pl.ylim([0.9, 1.1]); pl.xlim([-0.3, 0.3])

        pl.legend()
        pl.show()



    def select_train_pts(value_interval, sols):

        # this is basically repeated in prune_and_train, should we always just use that one?

        # (old) ideas for additional functionality:
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
        vs_ensemble = jax.vmap(v_nn, in_axes=(0, None))(vmap_params, x)

        v_mean = vs_ensemble.mean()
        v_std = vs_ensemble.std()

        return v_mean, v_std

    def vx_meanstd(x, vmap_params):

        # vmap for nn ensemble.
        vx_fct = jax.jacobian(v_nn, argnums=1)
        ensemble_vxs = jax.vmap(vx_fct, in_axes=(0, None))(vmap_params, x)

        # now we have all_vxs.shape == (N_ensemble, nx)
        # we want ensemble mean and std across axis 0.
        # stds will be individual for each coordinate, sum/mean whatever later if you want.
        vx_mean = ensemble_vxs.mean(axis=0)
        vx_std = ensemble_vxs.std(axis=0)

        return vx_mean, vx_std

    v_meanstds = jax.jit(jax.vmap(v_meanstd, in_axes=(0, None)))
    vx_meanstds = jax.jit(jax.vmap(vx_meanstd, in_axes=(0, None)))





    def plot_decision_boundary(v_nn, vmap_params, problem_params):

        # this x0 i got from random idpb experimentation by finding the
        # points with (0, -1) angle and lowest value. among those it is the
        # one with largest x.

        x0 = np.array([ 2.398837  ,  0.06769013,  0.        , -1.        , -1.166603  , 3.3332477 , -5.1929855 ], dtype=float)
        x1 = x0 * np.array([-1, 1, -1, 1, -1, 1, -1])

        ts = np.linspace(-1, 1, 200)
        xs = np.linspace(x0, x1, 200)

        mus, sigmas = v_meanstds(xs, vmap_params)

        ax = pl.subplot(211)

        pl.plot(ts, mus, label='value mean')
        pl.fill_between(ts, mus - sigmas, mus + sigmas, color='C0', alpha=.2, label=f'value 1σ confidence')
        pl.legend()

        vx_mu, vx_sigma = vx_meanstds(xs, vmap_params)

        pl.subplot(212, sharex=ax)
        pl.plot(ts, vx_mu, label=problem_params['state_names'])

        pl.gca().set_prop_cycle(None)

        for j in range(7):
            pl.fill_between(ts, vx_mu[:, j] - vx_sigma[:, j], vx_mu[:, j] + vx_sigma[:, j], alpha=.2)

        pl.legend()










    def plot_v_vx_line(xs, vmap_params):

        # xs = problem_params['project_M'](np.linspace(x0, x1, N))

        mus, sigmas = v_meanstds(xs, vmap_params)
        vx_mu, vx_sigma = vx_meanstds(xs, vmap_params)

        ax = pl.subplot(211)
        pl.plot(thetas, mus, label='value mean')
        pl.fill_between(thetas, mus - sigmas, mus + sigmas, color='C0', alpha=.2, label=f'value 1σ confidence')
        pl.legend()

        pl.subplot(212, sharex=ax)
        pl.plot(thetas, vx_mu, label=problem_params['state_names'])

        pl.gca().set_prop_cycle(None)
        for j in range(7):
            pl.fill_between(thetas, vx_mu[:, j] - vx_sigma[:, j], vx_mu[:, j] + vx_sigma[:, j], alpha=.2)

        pl.legend()

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
            u = pontryagin_utils.u_star_general(x, lam_x, problem_params)
            return problem_params['f'](x, u)


        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(
            atol=algo_params['pontryagin_solver_atol'],
            rtol=algo_params['pontryagin_solver_rtol'],
            dtmin=algo_params['dtmin'],
            dtmax=algo_params['dtmax'],
        )

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

            sigma_max = algo_params['sigma_max_abs'] + v_mean * algo_params['sigma_max_rel']

            has_low_sigma = v_std <= sigma_max

            # return is_very_likely_in_Vk

            return np.logical_and(is_very_likely_in_Vk, has_low_sigma)

        terminating_event = diffrax.DiscreteTerminatingEvent(event_fn)

        if problem_params['m'] is not None and algo_params['project_manifold']:
            solver = pontryagin_utils.ProjectionSolver(project=problem_params['project_M'])
        else:
            solver = diffrax.Tsit5()


        forward_sol = diffrax.diffeqsolve(
            term, solver, t0=0., t1=10., dt0=0.01, y0=x0,
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

        v_f_lqr = V_f(x_f)
        vx_f_lqr = jax.jacobian(V_f)(x_f)

        # if v_f_lqr < problem_params['V_f'], use that information instead.
        use_lqr = v_f_lqr < problem_params['V_f']
        v_f = jax.lax.select(use_lqr, v_f_lqr, v_f)
        vx_f = jax.lax.select(use_lqr, vx_f_lqr, vx_f)

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

    # }}}



    # define main active learning ingredients: proposals, oracle {{{

    def propose_pts(key, v_k, v_next, vmap_nn_params, all_ys, data_extent, algo_params):

        value_interval = [v_k, v_next]

        # ~~~  find uniformly sampled points from value band w/ rejection sampling ~~~

        # to "approximate" all kinds of global optimisation & sampling
        # operations over that set. this is a bit ugly and certainly not
        # jit-able... can this be done in a better way?

        if algo_params['proposal_sampling_distribution'] == 'uniform_scale_mixture':
            # this is the approach used initially. works for flatquad, fails for orbits with too few samples appearing
            # in the interesting regions where we don't have data. maybe similar things happen with flatquad too but
            # are fixed by brute-forcing a high active learning batchsize.
            sample_fct = lambda key, N: algo_params['sample_states_batched'](key, N, problem_params['x_extent'], log_min_scale=-2)
        elif algo_params['proposal_sampling_distribution'] == 'uniform':
            # instead, we want an actual uniform distribution. to make sure we still hit the interesting region appropriately,
            # we set the extent based on the data we curently have.
            # still this can bite us in the ass if test_pts suffer from similar issues of not covering the interesting "edge" regions
            # where we would want new data most urgently. hopefully the factor of 1.5 fixes this \o/ (while increasing sampling iters by 1.5**nx...)
            sample_fct = lambda key, N: algo_params['sample_states_batched'](key, N, data_extent * 1.5, log_min_scale=0)
        else:
            raise ValueError(f'unknown proposal sampling distribution {algo_params["proposal_sampling_distribution"]}')



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

            # sampling function, especially extent now fixed outside.
            x_pts = sample_fct(newkey, 100000)

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
            # good right?  -> this doesn't happen most of the time anyway
            is_in_range = np.logical_and(value_interval[0] <= v_means, optimistic_vs <= value_interval[1])

            interesting_x0s = x_pts[is_in_range]

            all_valueband_pts = np.concatenate([all_valueband_pts, interesting_x0s], axis=0)

        metrics = dict()
        metrics['proposal_sampling_iters'] = i

        if all_valueband_pts.shape[0] < N_pts_desired:

            # this has never happened since we started using non-uniform sample
            # concentrated around equilibrium here too

            print('did not find enough points!')
            if all_valueband_pts.shape[0] < algo_params['active_learning_batchsize']:
                raise ValueError('this is definitely not going to work')

            # one possibility: "pad" the points with the ones that are not
            # within the value interval necessarily, but above the lower bound.
            # N_missing = N_pts_desired - all_valueband_pts.shape[0]
            # arr, idx = jax.lax.top_k(-v_means - np.inf * (v_means < value_interval[0]), N_missing)
            # all_valueband_pts = np.concatenate([all_valueband_pts, x_pts[idx]], axis=0)


        all_valueband_pts = all_valueband_pts[0:N_pts_desired, :]
        # assert all_valueband_pts.shape == (N_pts_desired, problem_params['nx'])

        # ~~~ find a sensible subset of those points to use as proposals ~~~

        # now we have 1000 points that satisfy the first requirement (be
        # inside of the value band). as a first attempt we just sample
        # without replacement according to acquisition function style
        # weights.

        # things to consider afterwards:
        # - ensure the samples are not very close (some literature about this "batched active learning", max kernel distance etc.)

        # though: the highest-uncertainty ones also tend to be high value (= far from the current data set)
        # is this a problem? if V_k+1 is higher than it should be it might take a long time to learn
        # forget this for now maybe its even a good thing.

        v_means, v_stds = v_meanstds(all_valueband_pts, vmap_nn_params)

        sigma_maxs = algo_params['sigma_max_abs'] + v_means * algo_params['sigma_max_rel']


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

        elif proposal_strategy == 'max_kernel':

            # very experimental implementation. we choose the max sigma
            # point, then assume that close sigmas decrease based on that
            # according to some kernel function.

            # not maxkernel at all but should ensure that we can propose
            # high-sigma points without them all being in the same region.

            # maybe this kernel should somehow scale with the data "scale"?
            # if data is "spread out" a tiny kernel does nothing.

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

                # kernel_scaling < 1 will cause more samples to come from the uncertain region
                # and less of a uniform distribution.
                k = lambda x, y: algo_params['proposal_kernel_scaling'] * np.exp(-np.sum(((x-y) / lengthscale)**2))

                # this multiplication by .5 makes everything look nicer in low
                # dims. it will sample repeatedly from high-uncertainty regions
                # but not exctly at the same point. in higher dims, i think it
                # is less smart to do this because there always enough
                # different high-uncertainty points to choose from.
                # k = lambda x, y: 0.5 * np.exp(-np.sum(((x-y) / lengthscale)**2))
                # k = lambda x, y: 0.75 * np.exp(-np.sum(((x-y) / lengthscale)**2))

                # k = lambda x, y: np.exp(-np.sum(np.abs((x-y) / lengthscale)))

                # then we just scale everything by 1-that kernel?
                weights = jax.vmap(lambda x: 1 - k(x, proposal))(all_valueband_pts)

                carry = sigmas * weights

                # oup = (proposal_idx, carry)  # just to look at the data :)
                oup = proposal_idx

                return carry, oup

            sigma_relative = v_stds / sigma_maxs

            final_carry, oups = jax.lax.scan(scan_fct, sigma_relative, None, length=N_proposals)
            proposal_idxs = oups

        elif proposal_strategy == 'max_kernel_adaptive':

            # same as above, but adapts the size of the kernel (roughly) to
            # the extent of the data set.

            data_ranges = all_valueband_pts.ptp(axis=0)

            def scan_fct(sigmas, inp):

                # find max sigma.
                proposal_idx = np.argmax(sigmas)
                proposal = all_valueband_pts[proposal_idx]

                # adaptive kernel scales with extent of value levelset.
                lengthscales = (1/4) * data_ranges

                k = lambda x, y: algo_params['proposal_kernel_scaling'] * np.exp(-np.sum(((x-y) / lengthscales)**2))

                weights = jax.vmap(lambda x: 1 - k(x, proposal))(all_valueband_pts)

                carry = sigmas * weights

                # oup = (proposal_idx, carry)  # just to look at the data :)
                oup = proposal_idx

                return carry, oup

            sigma_relative = v_stds / sigma_maxs

            # ys_in_valueinterval = np.logical_and(all_ys['v'] >= value_interval[0], all_ys['v'] <= value_interval[1])

            # TODO finish this or scrap it.
            # TODO also do this in non-adaptive version?
            # for each data point x we already have, reduce sigma the same as w/ proposals.
            # k = lambda x, y: algo_params['proposal_kernel_scaling'] * np.exp(-np.sum(((x-y) / lengthscales)**2))
            # huge kernel matrix
            # ks = jax.vmap(k, jax.vmap(k, in_axes=(0, None)), in_axes=(None, 0))(all_valueband_pts, xs)



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

            proposal_idxs = jax.random.choice(key, all_valueband_pts.shape[0], shape=(N_proposals,), replace=do_replace, p=ps)

        elif proposal_strategy == 'uniform_all':

            # just uniform subsample of proposals yolo

            proposal_idxs = jax.random.choice(key, all_valueband_pts.shape[0], shape=(N_proposals,), replace=False)

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



        proposed_states = all_valueband_pts[proposal_idxs]

        # oops did many of these on euler. but i think it just ignores it
        if algo_params['showfigs']:
            pl.figure('proposals')
            if problem_params['nx'] == 2:
                pl.plot(all_valueband_pts[:, 0], all_valueband_pts[:, 1], '.', label='all points')
                pl.plot(proposed_states[:, 0], proposed_states[:, 1], 'o', label='proposed points')
            elif problem_params['nx'] == 7:
                pl.subplot(131)
                # x/y
                pl.plot(all_valueband_pts[:, 0], all_valueband_pts[:, 1], '.', label='all points')
                pl.plot(proposed_states[:, 0], proposed_states[:, 1], 'o', label='proposed points')
                pl.subplot(132)
                # vx/vy
                pl.plot(all_valueband_pts[:, 4], all_valueband_pts[:, 5], '.', label='all points')
                pl.plot(proposed_states[:, 4], proposed_states[:, 5], 'o', label='proposed points')
                pl.subplot(133)
                # Phi/omega
                x_to_phi = jax.vmap(lambda x: np.arctan2(x[2], x[3]))
                pl.plot(x_to_phi(all_valueband_pts), all_valueband_pts[:, 6], '.', label='all points')
                pl.plot(x_to_phi(proposed_states), proposed_states[:, 6], 'o', label='proposed points')
            pl.show()

        return proposed_states, v_means[proposal_idxs], v_stds[proposal_idxs], metrics



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

        metrics = dict()

        forward_sols = jax.vmap(forward_sim_nn_until_value, in_axes=(0, None, None, None))(
            proposals,
            vmap_nn_params,
            v_k,
            True
        )

        if problem_params['m'] is not None:
            all_ms = jax.vmap(jax.vmap(problem_params['m']))(forward_sols.ys)
            trajectory_max_m = np.abs(all_ms * (all_ms < np.inf)).max(axis=1)
            metrics['oracle_forward_max_m'] = trajectory_max_m.max()

        xfs = jax.vmap(lambda sol: sol.ys[sol.stats['num_accepted_steps']])(forward_sols)

        # the solutions that stopped due to DiscreteTerminatingEvent
        stopped_bc_terminatingevent = forward_sols.result == 1

        # sanity check: this should be the same. literally just checking the terminatingevent
        # conditions as well. *maybe* there is some edge case where the condition is True at the
        # last step and the solver quits anyway, so it doesn't report quitting "due to" the event?
        mus, sigs = v_meanstds(xfs, vmap_nn_params)



        sig_maxs = algo_params['sigma_max_abs'] + mus * algo_params['sigma_max_rel']

        # is_usable = np.logical_and(mus + 2 * sigs <= v_k, sigs <= sig_maxs)

        # this assertion never failed since the last change of making
        # sigma_max a function specified in algo_params. should we still
        # somehow try to do it? is chex the tool for this?
        # assert (stopped_bc_terminatingevent == is_usable).all(), 'shit happened'

        is_usable = stopped_bc_terminatingevent

        metrics['oracle_frac_usable'] = is_usable.mean()


        # if we have a different amount every time, we cannot jit the simulation.
        # therefore we just mark it as nan and try to tune the algo such that not too many
        # of them are nan.
        # usable_xfs = xfs.at[~is_usable].set(np.nan)

        # turns out that was itself not jittable. this should work:
        usable_xfs = np.where(is_usable[:, None], xfs, np.nan * xfs)

        # generous upper bound for value we're interested in rn.
        # integration of trajectories stops once we pass this threshold.
        v_upper = v_next + 50 * (v_next - v_k)

        backward_sols = jax.vmap(solve_backward_nn_ens, in_axes=(0, None, None, None, None))(
            usable_xfs, vmap_nn_params, v_upper, problem_params, algo_params
        )

        if problem_params['m'] is not None:
            all_ms = jax.vmap(jax.vmap(problem_params['m']))(backward_sols.ys['x'])
            metrics['oracle_backward_max_m'] = np.abs(all_ms * (all_ms < np.inf)).max()

        # find out how close we got.
        # ys['x'].shape == (N_proposals, N_steps, nx)
        # proposals.shape == (N_proposals, nx)
        # so to broadcast along the time axis, None in the middle.
        pointwise_dists_to_proposal = np.linalg.norm(backward_sols.ys['x'] - proposals[:, None, :], axis=-1)
        dists = np.min(pointwise_dists_to_proposal, axis=1)
        is_finite = dists < np.inf
        # worst dist is unaffected by changing inf to 0.
        metrics['oracle_worst_dist'] = np.nanmax(dists * is_finite)
        # mean has to be adjusted.
        metrics['oracle_mean_dist'] = np.nanmean(dists * is_finite) / np.nanmean(is_finite)

        return forward_sols, backward_sols, metrics



    # test points, with increased density towards origin.
    # 10000 points doesn't even look like all that much on a plot, maybe we need more...
    N_testpts = 100000

    # keep this "hardcoded" here? put in algoparams? make some heuristic to
    # adapt based on data?
    test_pts = algo_params['sample_states_batched'](
        jax.random.PRNGKey(123), N_testpts, problem_params['x_extent'], log_min_scale=-2
    )

    # use this to "mark" test points that AT SOME POINT were below the sigma limit.
    test_pts_known = np.zeros((N_testpts,)).astype(bool)


    @jax.jit
    def estimate_value_level(v_means, v_stds, test_pts_known, upper_v=np.inf):

        # estimate "known" value level based on finite test points set.

        sigma_maxs = algo_params['sigma_max_abs'] + v_means * algo_params['sigma_max_rel']
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
            frac_certain_inside = 1 - np.cumsum(1 - sigma_small_enough[idx]) / (np.arange(test_pts_known.shape[0]) + .0001)

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

        # b) they are above the sigma threshold, and VERY clearly (10 sigma) within the currently estimated level set
        new_testpts_known = np.logical_or(new_testpts_known, v_means + 10 * v_stds <= v_k)

        # c) they were known in the previous iteration.
        new_testpts_known = np.logical_or(test_pts_known, new_testpts_known)




        metrics = dict()
        metrics['frac_testpts_known'] = new_testpts_known.mean()

        # estimate actual state space volume with second half of test points.
        # this only works if the sampling function actually puts the uniformly
        # sampled subset there. specifically I think if log_min_scale > 0 the
        # distribution of points will still be usable but with uniform points
        # in first half. so avoid that.

        half = test_pts_known.shape[0] // 2
        metrics['frac_volume_known'] = new_testpts_known[half:].mean()

        return v_k, new_testpts_known, metrics

    # }}}



    # initial training run {{{

    # choose initial value level.

    v_k = algo_params['v_init']

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

    all_ys = select_train_pts([0., v_k], sols_orig)

    # split into train/test set.
    train_ys, test_ys = nn_utils.train_test_split(all_ys, train_frac=algo_params['nn_train_fraction'])

    v_nn = nn_utils.nn_wrapper(problem_params, algo_params)

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

    # initial NN training.
    # v_k both times for uniform (not sweep-style) minibatch sampling.
    params_sobolev_ens, oups_sobolev_ens = v_nn.train_sobolev_ensemble(
        train_key, train_ys, v_k, v_k, problem_params, algo_params
    )

    # no normalisation anywhere anymore
    v_nn_unnormalised = v_nn

    # first non-nan index
    sol_idx = np.argmax(~np.isnan(sols_orig.ys['v']))
    sol = jax.tree_util.tree_map(itemgetter(sol_idx), sols_orig)

    if algo_params['showfigs']:
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
        plotting_utils.plot_calibration(sols_orig.ys, means, stds)

        if problem_params['m'] is not None:
            pl.figure('manifold')
            plotting_utils.plot_manifold(v_meanstds, vx_meanstds, params_sobolev_ens, problem_params)

        pl.show()


    # }}}



    # set up data saving & tracking stuff {{{

    # euler scratch directory structure:
    # $SCRATCH
    #     flatquad_runs
    #         <run ID>
    #             figures
    #                 stuff.png
    #             all_data.msgpack.gz
    #     orbits_runs
    #         <same>
    #     ...

    euler = 'SCRATCH' in os.environ
    if euler:
        # assume we are on euler, save wandb files in scratch!
        sys = problem_params['system_name']
        save_dir = os.path.join(os.environ['SCRATCH'], f'{sys}_runs')
        os.makedirs(save_dir, exist_ok=True)
    else:
        # for quick local runs we don't have too much data
        save_dir = '.'


    if algo_params['wandb']:

        # start a new wandb run to track this script
        projectname = 'levelsets_' + problem_params['system_name']


        algo_params_clean = {k: v for k, v in algo_params.items() if not callable(v)}
        wandb.init(
            # set the wandb project where this run will be logged
            project=projectname,

            # track hyperparameters and run metadata
            config=algo_params_clean,
            dir=save_dir,
        )

        # save figures on euler scratch or locally to not destroy wandb storage
        run_dir = os.path.join(save_dir, str(wandb.run.id))
        if not euler:
            run_dir = os.path.join(save_dir, 'local_runs', str(wandb.run.id))

    else:
        # still make a local folder for figs & pickles, name it with timestamp.
        t = int(time.time())
        run_dir = os.path.join(save_dir, 'local_runs', f'run_{t}')
        os.makedirs(run_dir, exist_ok=True)


    if algo_params['savefigs']:
        fig_dir = os.path.join(run_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        print(f'saving figures in {fig_dir}')

    # }}}



    # main active learning loop!! {{{

    all_ys = sols_orig.ys
    is_suboptimal = np.zeros_like(all_ys['v']).astype(bool)
    v_next_target = algo_params['v_init']

    for k in range(100):


        print(f'\n\n\n ~~~~ active learning iteration {k} ~~~~')

        # estimate known value level
        vk_prev = v_k
        v_means, v_stds = v_meanstds(test_pts, params_sobolev_ens)
        v_k, test_pts_known, estimator_metrics = estimate_value_level(v_means, v_stds, test_pts_known, upper_v=v_next_target)

        if v_k >= problem_params['V_max']:
            # just quit here, data and nn params that lead to this were saved last iteration
            # def eval_controlcost(key, v_sim, v_nn, nn_params, all_ys, is_suboptimal, problem_params, algo_params, x0s=None):

            # or, rather call eval_directly here? with eval_controlcost and
            # also data saving right inside... but then we'd also like to put
            # some controlcost evals on wandb.

            # prob. should retrain here if going higher with thin_data=True...
            costs, x0s, v_means, v_stds = eval_controlcost(key, problem_params['V_max'], v_nn, params_sobolev_ens, all_ys, is_suboptimal, problem_params, algo_params)

            all_data = {'vk': v_k, 'ys': all_ys, 'is_suboptimal': is_suboptimal, 'nn_params': params_sobolev_ens}
            # TODO wandb run id and some place to store on euler
            # evaluate_directly(all_data, 'test_run_id', problem_params, algo_params)

            # what fraction of the points is below the given ratio of achieved / estimated cost?
            def frac_below(ratio):
                return ((costs / v_means) <= ratio).mean()

            cost_dict = {
                    'frac_ratio_005': frac_below(1 + 0.05),
                    'frac_ratio_050': frac_below(1 + 0.50),
                    'frac_ratio_500': frac_below(1 + 5.00),
            }
            if algo_params['wandb']:
                wandb.log(cost_dict)
            else:
                print('final cost dict:')
                pprint.pprint(cost_dict)

            break  # aaaand exit everything.

        # set next value target
        v_next_target = set_value_target(all_ys, v_k, problem_params, algo_params)

        # estimate extent of next level set based on test pts and extrapolation
        is_in_Vnext = v_means <= v_next_target
        inside_xs = test_pts * is_in_Vnext[:, None]
        # data extent in sampling fct is with respect to x_eq!
        # data_extent = np.abs(inside_xs - problem_params['x_eq'][None, :]).max(axis=0)
        # this more correct?
        data_extent = np.abs(inside_xs).max(axis=0)

        OK = False
        i = 0

        while True:
            print(f'estimated v_k = {v_k:.3f}, next target = {v_next_target:.3f}')

            # propose interesting points
            proposal_key, key = jax.random.split(key)
            proposed_pts, proposal_vmeans, proposal_vstds, proposal_metrics = propose_pts(
                proposal_key, v_k, v_next_target, params_sobolev_ens, all_ys, data_extent, algo_params
            )

            # obtain optimal trajectories close to those points
            forward_sols_new, backward_sols_new, oracle_metrics = batched_oracle(
                proposed_pts, v_k, v_next_target, params_sobolev_ens, problem_params
            )

            OK = oracle_metrics['oracle_frac_usable'] > 0.5

            if OK or i > 5:
                break

            v_next_target = v_k + (v_next_target - v_k) / 2
            i += 1


        # append data & suboptimality flag to previous data
        new_ys = backward_sols_new.ys
        all_ys = jtm(lambda a, b: np.concatenate([a, b], axis=0), all_ys, new_ys)
        is_suboptimal = np.concatenate([is_suboptimal, np.zeros_like(new_ys['v']).astype(bool)], axis=0)


        # prune suboptimal data & train NN
        prev_params_sobolev_ens = params_sobolev_ens
        train_key, key = jax.random.split(key)
        params_sobolev_ens, oups, is_suboptimal, pruning_metrics = prune_and_train(
            train_key,
            v_nn,
            params_sobolev_ens,
            all_ys,
            [v_k, v_next_target],
            is_suboptimal,
            problem_params,
            algo_params,
            warmstart=algo_params['nn_warm_start']
        )

        all_oups = oups


        # tracking, saving, plotting type stuff {{{

        # if a key is repeated apparently the latter one is used. but don't repeat keys!
        full_logdict = {
            **{ 'vk': v_k, 'v_next_target': v_next_target, },
            **pruning_metrics,
            **proposal_metrics,
            **estimator_metrics,
            **oracle_metrics,
        }

        if algo_params['wandb']:
            print('doing wandb log')
            wandb.log(full_logdict, step=k)

            # serialise nn parameters.
            # probably incorrect. not sure how to read documentation.
            '''
            with nn_params_artefact.new_file('params.msgpack', 'wb') as params_file:
                params_bytes = flax.serialization.msgpack_serialize(params_sobolev_ens)
                params_file.write(params_bytes)
            '''
        else:
            pprint.pprint(full_logdict)

        # save dataset in scratch.
        all_data = {
            'step': k,
            'vk': v_k,
            'ys': all_ys,
            'is_suboptimal': is_suboptimal,
            'nn_params': params_sobolev_ens,
        }

        bs = flax.serialization.msgpack_serialize(all_data)

        with gzip.open(os.path.join(run_dir, 'all_data.msgpack.gz'), 'wb') as f:
            f.write(bs)

        # figure plotting :))
        if algo_params['savefigs'] or algo_params['showfigs'] or algo_params['wandbfigs']:


            # get this boring stuff done once and for all, never confusing any names anymore
            class plot_saver():

                def __init__(self, name):
                    self.name = name

                def __enter__(self):
                    self.fig = pl.figure(self.name)

                def __exit__(self, exception_type, exception_value, exception_traceback):

                    if algo_params['savefigs']:
                        pl.savefig(os.path.join(fig_dir, f'{self.name}_{k:04d}.png'))

                    if algo_params['wandb'] and algo_params['wandbfigs']:
                        wandb.log({self.name : wandb.Image(self.fig)}, step=k)


            if problem_params['system_name'] == 'orbits':
                with plot_saver('orbits_all'):
                    xs = ys = np.linspace(-2, 2, 201)
                    xx, yy = np.meshgrid(xs, ys)
                    plot_v_means, plot_v_stds = jax.vmap(v_meanstds, in_axes=(0, None))(np.stack([xx, yy], axis=-1), prev_params_sobolev_ens)
                    _, plot_v_stds_new = jax.vmap(v_meanstds, in_axes=(0, None))(np.stack([xx, yy], axis=-1), params_sobolev_ens)
                    plotting_utils.orbits_plot_all(xx, yy, plot_v_means, plot_v_stds, plot_v_stds_new, v_k, v_next_target, proposed_pts, forward_sols_new, backward_sols_new, problem_params, algo_params)

            with plot_saver('meanstds'):
                plotting_utils.plot_proposals(v_means, v_stds, test_pts_known, proposal_vmeans, proposal_vstds, v_k, v_next_target, algo_params)

            with plot_saver('training'):
                plotting_utils.plot_nn_train_outputs(all_oups, subsample=64)
                pl.ylim([1e-4, 1e3])

            with plot_saver('trajectory'):
                plotting_utils.plot_trajectory_vs_nn_ensemble(sol, params_sobolev_ens, v_nn_unnormalised)

            if problem_params['m'] is not None:
                with plot_saver('manifold'):
                    plotting_utils.plot_manifold(v_meanstds, vx_meanstds, params_sobolev_ens, problem_params)

            if problem_params['system_name'] == 'flatquad':

                with plot_saver('decision_boundary'):
                    plot_decision_boundary(v_nn, params_sobolev_ens, problem_params)

                '''
                with plot_saver('value_lines'):
                    plot_v_along_lines(test_pts, v_nn, params_sobolev_ens, v_next_target)
                '''


            # sift out the data that we used during training.
            v_cutoff = v_k / algo_params['thin_data_denominator'] if algo_params['thin_data'] else 0.
            is_in_band = np.logical_and(all_ys['v'] <= v_next_target, all_ys['v'] >= v_cutoff)
            is_unusable = np.logical_or(all_ys['v'] == np.inf, np.isnan(all_ys['v']))
            is_relevant = (~is_suboptimal) & (~is_unusable) & is_in_band

            with plot_saver('nn_calibration'):

                relevant_ys = jtm(lambda node: node[is_relevant], all_ys)
                means, stds = v_meanstds(relevant_ys['x'], params_sobolev_ens)
                vx_means, vx_stds = vx_meanstds(relevant_ys['x'], params_sobolev_ens)
                plotting_utils.plot_calibration(relevant_ys, means, stds)


            with plot_saver('loss_distributions'):

                all_losses, aux = jax.vmap(
                    jax.vmap(v_nn.sobolev_loss, in_axes=(None, 0, None, None, None)),
                    in_axes=(None, None, 0, None, None)
                )(key, relevant_ys, params_sobolev_ens, problem_params, algo_params)

                aux_mean = jtm(lambda z: z.mean(axis=0), aux['lossterms'])
                plotting_utils.plot_loss_distribution(aux_mean)


            if algo_params['showfigs']:
                pl.show()

            pl.close('all')

        if algo_params['ipdb_interval'] > 0 and k % algo_params['ipdb_interval'] == 0:
            ipdb.set_trace()

        # }}}

    # }}}




def evaluate(run_dir, problem_params, algo_params):

    # given this run dir:
    #  - get all data from corresponding msgpack serialisation
    #  - fit nn to data to get value fct and controller
    #  - do some closed loop sims, uniformly from the sublevel set or something like that

    filepath = os.path.join(run_dir, 'all_data.msgpack.gz')
    run_id = run_dir.split('/')[-1]

    if not os.path.isfile(filepath):
        print(f'{filepath} does not exist. trying to pull from euler')
        cmd = ['./pull_run.sh', problem_params['system_name'], run_id]
        output = subprocess.run(cmd)
        if output.returncode != 0:
            print(f'failed to pull run from euler with exit code {output.returncode}')
            sys.exit(1)
        if not os.path.isfile(filepath):
            print('path still does not exist')
            sys.exit(1)

    with gzip.open(filepath, 'rb') as f:
        bs = f.read()

    all_data = flax.serialization.msgpack_restore(bs)

    evaluate_directly(all_data, run_id, problem_params, algo_params)

def eval_controlcost(key, v_sim, v_nn, nn_params, all_ys, is_suboptimal, problem_params, algo_params, x0s=None):

    # forward simulation & control cost calculation for supplied initial states x0s,
    # or if x0s=None we sample some here.

    v_meanstds, vx_meanstds = def_v_meanstds(v_nn)

    if x0s is None:
        # find extent for sampling.
        xkey, key = jax.random.split(key)
        inside = all_ys['v'] <= v_sim
        ys_relevant = jtm(lambda n: n[inside & ~is_suboptimal], all_ys)
        extent = np.abs(ys_relevant['x']).max(axis=0) * 1.5

        # sample uniform states from extent box.
        xs = algo_params['sample_states_batched'](xkey, 10000, extent, log_min_scale=-2)

        # find out which ones are within given sublevel set.
        v_means, v_stds = v_meanstds(xs, nn_params)
        idx = v_means < v_sim
        x0s = xs[idx]
        v_means = v_means[idx]
        v_stds = v_stds[idx]


    else:
        print('using supplied x0s.')
        v_means, v_stds = v_meanstds(x0s, nn_params)

    sim = lambda x0: forward_sim_nn(x0, v_nn, nn_params, problem_params, algo_params, T=10.)
    sols = jax.vmap(sim)(x0s)

    if not (sols.stats['num_steps'] < algo_params['pontryagin_solver_maxsteps']).all():
        print('eval_controlcost_common: warning, solver step limit reached, plz increase')

    solver_steps = sols.stats['num_steps']

    # last_ys = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)
    # difference: if t1 not reached this is still a valid state, not NaN
    # so the overall estimated cost will just be high
    last_xs = jax.vmap(lambda sol: sol.ys['x'][sol.stats['num_accepted_steps']])(sols)

    traj_costs = (sols.ys['cost'] * (sols.ys['cost'] != np.inf)).max(axis=1)

    # correct for inf horizon with lqr.
    if problem_params['m'] is not None:
        # in manifold case this should work the same.
        # P and K are wrt ambient space so we can 'blindly' use them here.
        # but check again to be sure.
        pass
        # ipdb.set_trace()

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)
    eq = problem_params['x_eq']
    lqr_terminalcosts = jax.vmap(lambda x: 0.5 * (x-eq).T @ P_lqr @ (x-eq))(last_xs)
    costs = (traj_costs + lqr_terminalcosts)
    return costs, x0s, v_means, v_stds




def evaluate_directly(all_data, run_id, problem_params, algo_params):

    # should these be arguments?

    # instead of ensemble 'distill' only to a single NN.
    single = True


    # restoring this gives us a Pytree with numpy array (not jax.numpy!)
    # leaves, so we convert it here.
    # wasted half an hour digging through so much code to find this out
    all_ys = jtm(np.array, all_data['ys'])
    is_suboptimal = np.array(all_data['is_suboptimal'])

    vk = all_data['vk']
    # this vk ^^ is from the penultimate round. so really a bit crappy to use this.
    # v_train = problem_params['V_max']
    # v_sim = v_train * 4/5
    v_sim = v_train = all_data['vk']

    key = jax.random.PRNGKey(0)

    # long training, quadratic (not huber) losses, low learning rate.
    # shouldn't we rather get the algoparams from the actual experiment?
    # so we have the same 'smoothness prior' mainly.
    # algo_params['lr_init'] = np.sqrt(algo_params['lr_init'] * algo_params['lr_final'])
    # algo_params['v_loss_d'] = algo_params['vx_loss_d'] = 100.
    algo_params['nn_value_sweep'] = False
    algo_params['lr_staircase'] = True
    # push it
    algo_params['lr_final'] = algo_params['lr_final'] / 10
    algo_params['lr_init'] = algo_params['lr_final'] * 2
    algo_params['nn_N_epochs'] = algo_params['nn_N_epochs'] / 10 # just for developping stuff below

    if single:
        algo_params['nn_ensemble_size'] = 1

    v_nn = nn_utils.nn_wrapper(problem_params, algo_params)


    trainkey, key = jax.random.split(key)
    if 'nn_params' in all_data:
        print('got nn params, training warm-started')
        nn_params = jtm(np.array, all_data['nn_params'])
        nn_params = jtm(lambda z: z[0:1], nn_params)
        # ipdb.set_trace()
        nn_params, training_oups, is_suboptimal, pruning_metrics = prune_and_train(
            trainkey,
            v_nn,
            nn_params,
            all_ys,            # all data
            [0., v_train],          # everything used
            is_suboptimal,     # but only the good parts
            problem_params,
            algo_params,
            is_final=True,
            warmstart=True,
        )
    else:
        print('got no nn params, training from scratch')
        nn_params, training_oups, is_suboptimal, pruning_metrics = prune_and_train(
            trainkey,
            v_nn,
            None,
            all_ys,            # all data
            [0., v_train],          # everything used
            is_suboptimal,     # but only the good parts
            problem_params,
            algo_params,
            is_final=True,
            warmstart=False,
        )

    plotting_utils.plot_nn_train_outputs(training_oups)
    pl.show()

    # here do all the nice evaluation metrics we can imagine.
    # for each metric:
    # - do experiment
    # - write data in some big output dict
    # - save that dict, again in msgpack format so we can later plot data.

    # take this in algoparams?
    eval_meshcat = False
    eval_controlcost_common = True
    eval_controlcost_2d = problem_params['system_name'] == 'orbits'

    eval_outputs = dict()
    data_dir = 'plot_data'

    # make another context manager thing to DRY the file output?

    if eval_meshcat:
        # usual upside down thing
        xs = jax.vmap(lambda x: np.array([x, 0, 0, -1, 0, 5, 0]))(np.linspace(-10, 10, 201))
        meshcat_forward_sims(xs, v_nn, nn_params, problem_params, algo_params)

        # same but faster
        xs = jax.vmap(lambda x: np.array([x, 0, 0, -1, 0, 15, 0]))(np.linspace(-10, 10, 201))
        meshcat_forward_sims(xs, v_nn, nn_params, problem_params, algo_params)

        # grid, upright, only where v < vk
        # xs = jax.vmap(lambda x: np.array([2 * (x%10 - 4.5), 2 * ((x//10)%10 - 4.5), 0, 1, 0, 0, 0]))(np.arange(100))

        x = np.linspace(-20, 20, 80)
        y = np.linspace(-20, 20, 80)
        xx, yy = np.meshgrid(x, y)
        xs = jax.vmap(lambda x, y: np.array([x, y, 0, 1, 0, 0, 0]), in_axes=(0, 0))(xx.flatten(), yy.flatten())

        vs = jax.vmap(v_nn, in_axes=(None, 0))(jtm(itemgetter(0), nn_params), xs)
        xs_inside = xs[vs < v_train]

        meshcat_forward_sims(xs_inside, v_nn, nn_params, problem_params, algo_params)

    if eval_controlcost_2d:

        # - make 2d grid covering the state space
        # - evaluate v_mean and v_std on that grid
        # - forward simulate and record control cost
        #   ("infinite horizon" = long horizon + terminal lqr)

        # for the orbits example.
        assert problem_params['system_name'] == 'orbits'

        N_grid = 256
        # extent of 2 is good enough for plot
        x = np.linspace(-2, 2, N_grid)
        y = np.linspace(-2, 2, N_grid)
        xx, yy = np.meshgrid(x, y)
        xs = np.column_stack([xx.flatten(), yy.flatten()])

        # just above what it needs empirically
        algo_params['pontryagin_solver_maxsteps'] = 180
        sim = lambda x0: forward_sim_nn(x0, v_nn, nn_params, problem_params, algo_params, T=30.)
        sols = jax.vmap(sim)(xs)

        if not (sols.stats['num_steps'] < algo_params['pontryagin_solver_maxsteps']).all():
            print('eval_controlcost_2d: warning, solver step limit reached, plz increase')

        solver_steps = sols.stats['num_steps'].reshape(N_grid, N_grid)

        last_ys = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)
        last_costs = last_ys['cost']

        # correct for inf horizon with lqr. to make this better, stop ODE
        # solver once low? though for 2D problem seems to be good like this
        K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)
        eq = problem_params['x_eq']
        lqr_terminalcosts = jax.vmap(lambda x: 0.5 * (x-eq).T @ P_lqr @ (x-eq))(last_ys['x'])
        costs = (last_costs + lqr_terminalcosts).reshape(N_grid, N_grid)

        # 1 size ensemble
        params = jtm(itemgetter(0), nn_params)
        vs = jax.vmap(v_nn, in_axes=(None, 0))(params, xs).reshape(N_grid, N_grid)

        # then, in the plotting script do this but nicer:
        # levels=np.linspace(0, 1000, 50)
        # ax = pl.subplot(211)
        # pl.contour(xx, yy, vs, levels=levels)
        # pl.xlabel('learned v')
        # ax = pl.subplot(212, sharex=ax, sharey=ax)
        # pl.contour(xx, yy, costs, levels=levels)
        # pl.xlabel('incurred cost')
        # pl.show()

        eval_outputs['xx'] = xx
        eval_outputs['yy'] = yy
        eval_outputs['learned_v'] = vs
        eval_outputs['controlcost'] = costs
        eval_outputs['v_train'] = v_train

        bs = flax.serialization.msgpack_serialize(eval_outputs)
        sysname = problem_params['system_name']
        fpath = os.path.join(data_dir, f'{sysname}_{run_id}_controlcosts_2d.msgpack.gz')
        with gzip.open(fpath, 'wb') as f:
            f.write(bs)
        print(f'eval_controlcost_2d: wrote to {fpath}')

    if eval_controlcost_common:

        # def eval_controlcost(key, v_sim, v_nn, nn_params, all_ys, problem_params, algo_params, x0s=None):
        evalkey, key = jax.random.split(key)
        costs, x0s, v_means, v_stds = eval_controlcost(evalkey, v_sim, v_nn, nn_params, all_ys, is_suboptimal, problem_params, algo_params)

        # what data do we want?
        eval_outputs = {
            'x0s': x0s,
            'v_mean': v_means,
            'v_stds': v_stds,
            'costs': costs,
        }

        # put run id in this file name too?
        bs = flax.serialization.msgpack_serialize(eval_outputs)
        sysname = problem_params['system_name']
        fpath = os.path.join(data_dir, f'{sysname}_{run_id}_controlcosts_common.msgpack.gz')
        with gzip.open(fpath, 'wb') as f:
            f.write(bs)
        print(f'eval_controlcost_common: wrote to {fpath}')

        pl.plot(v_means[v_means < v_sim], costs, '. ')
        pl.show()


