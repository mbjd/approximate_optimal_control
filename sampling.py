#!/usr/bin/env python
# jax
import jax
import jax.numpy as np
import optax
import diffrax

# other, trivial stuff
import numpy as onp

import tk as tkinter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import time
from functools import partial

from tqdm import tqdm
import jax_tqdm

import pontryagin_utils
import plotting_utils
import array_juggling

# https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
from nn_utils import nn_wrapper
jax.tree_util.register_pytree_node(nn_wrapper,
                                   nn_wrapper._tree_flatten,
                                   nn_wrapper._tree_unflatten)

import yappi

# no bullshit
# this will throw an error when trying to automatically promote rank.
from jax import config
import warnings
config.update("jax_numpy_rank_promotion", "warn")
warnings.filterwarnings('error', message='.*Following NumPy automatic rank promotion.*')


def geometric_mala_2(integrate_fct, reward_fct_y0, problem_params, algo_params, key=jax.random.PRNGKey(0)):
    '''
    version 2 of the 'geometric MALA' sampler. difference is the following:
    - previously we basically set up a MCMC sampler at x(0) and tried to 'follow along' approximately with λ(T)
    - now, we do the whole MCMC in λ(T) but with the proposal distribution adjusted to fit the warped geometry,
      i.e. adjust λ(T) such that x0 = PMP(λ(T)) makes a sensible jump.

    it can be argued the difference is only cosmetic, if not it is certainly only minor. but it enables us to do
    the complete MCMC algo in terms of λ(T) including a correct M-H accept/reject step, which will give it the
    correct stationary distribution :)
    '''

    nx = problem_params['nx']


    integrate_fct_reshaped = lambda tc: integrate_fct(tc.reshape(1, nx))[1][:, 0:nx].reshape(nx)
    full_integrate_fct_reshaped = lambda tc: integrate_fct(tc.reshape(1, nx))[1].reshape(-1)

    # DO NOT confuse these two or it will cause days of bug hunting
    logpdf = lambda tc: reward_fct_y0(full_integrate_fct_reshaped(tc))
    logpdf_y0 = reward_fct_y0
    # logpdf_x0 = reward_fct_x0

    def run_single_chain(key, tc0, jump_sizes, chain_length):
        '''
        key: jax PRNG key
        tc0: (nx,) shape array of initial terminal condition (what an experssion)
        jump_sizes: (nx,) shape array containing the relevant lengh scales
         in state space. this will influence the proposal dist, but that will be
         scaled down by algo_params['sampler_dt'].
        '''

        def mala_transition_info(tc, dt):
            # this function does most of the heavy lifting here, we
            # hope that aside from calling this twice* we don't have to
            # do much of anything
            # *could be optimised to once per iteration by passing the previous result..?
            # **done :)

            # returns:
            # mean, cov:    parameterisation of the transition density
            #               p(tc' | tc) =  N(mean, cov)
            # logpdf_value: the unnormalised log probability density of tc
            #               this is calculated *with* the jacobian
            #               correction, i.e. the actual logpdf we want
            #               to sample λ (aka tc) from.

            # we later need to both sample the transition and evaluate
            # their density. therefore we return (mean, cov) here and
            # not just the sample or something.

            # similarly to the v1 sampler we find a good proposal distribution
            # by relating everything to the state space at x(0). the difference
            # is that now, more stuff is in this transition density function.

            # first we calculate all the derivatives :)
            # x0 = integrate_fct_reshaped(tc)

            y0 = full_integrate_fct_reshaped(tc)
            x0 = y0[0:nx]

            dy0_dtc = jax.jacobian(full_integrate_fct_reshaped)(tc)
            dx0_dtc = dy0_dtc[0:nx, :]
            dv0_dtc = dy0_dtc[-1:, :]
            dtc_dx0 = np.linalg.inv(dx0_dtc)  # by implicit function thm :)

            # if reward_fct depends both on x0 and v0 we cannot do it this simply.
            # dlogpdf_dx0 = jax.jacobian(logpdf_x0)(x0)
            jacobian_logpdf_y0 = jax.jacobian(logpdf_y0)(y0).reshape(1, 2 * nx + 1)

            dlogpdf_dx0_partial = jacobian_logpdf_y0[:, 0:nx]
            dlogpdf_dv0_partial = jacobian_logpdf_y0[:, -1:]

            # instead, reverse the computation graph between x0 and λT:
            # (which is possible because we assumed that PMP: λT -> x0 is bijective)
            # (i am pretty sure this is possible in some much much easier way with jax)

            #        <--- x0 <---
            # logpdf              λT
            #        <--- v0 <---

            # to look like this:

            #        <--- x0 --->
            # logpdf              λT
            #        <--- v0 <---

            # then, simply use the sum rule to get:

            dlogpdf_dx0 = (dlogpdf_dx0_partial + dlogpdf_dv0_partial @ dv0_dtc @ dtc_dx0).reshape(nx)
            dlogpdf_dtc = dlogpdf_dx0 @ dx0_dtc
            # jax.debug.print('dlogpdf_tc manual = {x}', x=dlogpdf_dtc)
            # confirmed that these two are the same.
            # dlogpdf_dtc_auto = jax.jacobian(lambda tc: logpdf_y0(full_integrate_fct_reshaped(tc)))(tc)
            # jax.debug.print('dlogpdf_dtc auto = {x}\n\n\n', x=dlogpdf_dtc_auto)

            # propose a jump in x0, then transform it to λT.
            # this is kind of sketchy...
            x0_jump_mean = dt * dlogpdf_dx0 * jump_sizes

            # as before, scale down the jump if very large.
            max_x0_jump_norm = 0.5
            correction = np.maximum(1, np.linalg.norm(x0_jump_mean) / max_x0_jump_norm)
            x0_jump_mean = x0_jump_mean / correction

            tc_jump_mean = dtc_dx0 @ x0_jump_mean
            new_tc_mean = tc + tc_jump_mean

            # covariances change under linear maps like this;
            # https://stats.stackexchange.com/questions/113700/
            x0_jump_cov = np.diag(2 * dt * jump_sizes**2) # = (sqrt(2dt) jump_sizes)**2
            tc_jump_cov = dtc_dx0 @ x0_jump_cov @ dtc_dx0.T
            new_tc_cov = tc_jump_cov  # bc. cov(tc) = 0

            # dx0_normalised_dtc = np.diag(1 / jump_sizes) @ dx0_dtc
            # dtc_normalised_x0 = np.linalg.inv(dx0_normalised_dtc)
            # # cov(Ax) = A.T cov(x) A
            # # could also store sqrtm(cov) for more efficient implementation but probably peanuts
            # new_tc_cov = np.sqrt(2 * dt) * (dtc_normalised_x0.T @ dtc_normalised_x0)
            # new_tc_mean = 0.5 * new_tc_cov @ dlogpdf_dtc

            # as for the logpdf: we know (from pdf change of variable, see report):
            #   g(λT) = p(PMP(λT)) * abs(det(dPMP(λT)/dλT))
            # thus:
            #   log g(λT) = log ( p(PMP(λT)) * abs(det(dPMP(λT)/dλT)) )
            #             = log p(PMP(λT)) + log abs(det(dPMP(λT)/dλT))

            # We want to adjust the jacobian formula to account for the fact that logpdf depends on v too.
            # we may need another 'compuational graph juggling' trick.
            # because *actually* V is a function of x, the graph looks like this:

            #        <--- x0 <---
            # logpdf      v       λT
            #        <--- v0

            # can we not just consider the whole logpdf to be still in terms of x, like this:

            # (        <--- x0 ) <---
            # ( logpdf      v  )      λT
            # (        <--- v0 )

            # then we can still find the jacobian: dlogpdf_dx total = dlogpdf_dx partial + dlogpdf_dv0_partial @ dv0_dx
            # conveniently, we have dv0_dx = λ(0).
            # but do we even need this?
            # if still logpdf is a function of x0 and x0 a function of λT, can we not just keep it the same?
            # at least, in the old formula we only need the jacobian dx0/dtc, not dlogpdf/dx0.
            # why is this? well because the fomula says so...
            # and we do calculate the full total dlogpdf/dx0 just above...
            # lets roll with this for the moment, not 100% sure about it though.

            logpdf_value_y0 = logpdf_y0(y0)
            detjac = np.linalg.det(dx0_dtc)

            # + small number to avoid gradients of log(0)
            # but if PMP is invertible as assumed then abs(detjac) > 0 anyways
            logpdf_value = logpdf_value_y0 + np.log(1e-9 + np.abs(detjac))

            # this is all wrong!!! we calculate the correct logpdf value only here
            # for later use in the MCMC accept/reject step. BUT we should already be
            # calculating it within here, for calculating the relevant gradient.
            # the detjac also changes with x! and should be used to inform the proposal.
            # i am pretty sure this is the fundamental reason we are suffering low accept rate.

            return new_tc_mean, new_tc_cov, logpdf_value, y0

        def mala_transition_info_new(tc, dt):
            # this function does most of the heavy lifting here, we
            # hope that aside from calling this twice* we don't have to
            # do much of anything
            # *could be optimised to once per iteration by passing the previous result..?
            # **done :)

            # returns:
            # mean, cov:    parameterisation of the transition density
            #               p(tc' | tc) =  N(mean, cov)
            # logpdf_value: the unnormalised log probability density of tc
            #               this is calculated *with* the jacobian
            #               correction, i.e. the actual logpdf we want
            #               to sample λ (aka tc) from.

            # nuke everything and start from scratch, previous version was shit
            # this time we want the proper pre-conditioned langevin diffusion:
            # dx(t) = .5 A nabla log p(x) dt + A^(1/2) dB_t

            # does this mean we calculate a gradient of gradient?
            # logpdf clearly depends on the jacobian dx0_dtc.
            # and we need the gradient dlogpdf/dtc.
            # obviously the algorithm i wrote in the paper kind of needs that.

            # dont care for the moment, hope that jax optimises it to a bearable state..
            # could we somehow pull some hessian out of stuff we already calculated?
            # and then use that to make an even better proposal distribution?

            # or probably we could also estimate one of the components from previous iterates...

            def y0_and_logpdf(tc):
                y0 = full_integrate_fct_reshaped(tc)
                logpdf_value_y0 = logpdf_y0(y0)

                dy0_dtc = jax.jacobian(full_integrate_fct_reshaped)(tc)
                dx0_dtc = dy0_dtc[0:nx, :]
                # dv0_dtc = dy0_dtc[-1:, :]

                detjac = np.linalg.det(dx0_dtc)

                # correct for the volume change due to PMP function
                logpdf_value_tc = logpdf_value_y0 + np.log(1e-9 + np.abs(detjac))
                return y0, logpdf_value_tc

            y0, logpdf_value = y0_and_logpdf(tc)

            dy0_dtc, dlogpdf_dtc = jax.jacobian(y0_and_logpdf)(tc)

            # calculate the noise covariance = gradient preconditioner
            dx0_dtc = dy0_dtc[0:nx, :]  # also calculated twice but \o/
            dx0_dnormalised_tc = np.diag(1 / jump_sizes) @ dx0_dtc
            dnormalised_tc_dx0 = np.linalg.inv(dx0_dnormalised_tc)

            new_tc_cov = np.sqrt(2 * dt) * (dnormalised_tc_dx0.T @ dnormalised_tc_dx0)
            new_tc_mean = tc + dt * new_tc_cov @ dlogpdf_dtc

            return new_tc_mean, new_tc_cov, logpdf_value, y0

        def scan_fct(state, dt):

            tc, current_transition_info, key = state

            key, noise_key, alpha_key = jax.random.split(key, 3)


            # calculate all the transition properties at once.
            # f for forward transition.
            f_transition_mean, f_transition_cov, tc_logpdf, y0_current = current_transition_info
            # f_transition_mean, f_transition_cov, tc_logpdf, y0_current = mala_transition_info(tc, dt)

            # sample from that distribution
            tc_proposed = jax.random.multivariate_normal(noise_key, mean=f_transition_mean, cov=f_transition_cov)

            # and get the info for the backward transition for ensuring detailed balance.
            b_transition_info = mala_transition_info(tc_proposed, dt)
            b_transition_mean, b_transition_cov, tc_next_logpdf, _ = b_transition_info


            # now, evaluate the transition probability densities
            f_transition_logdensity = jax.scipy.stats.multivariate_normal.logpdf(
                    tc_proposed, mean=f_transition_mean, cov=f_transition_cov
            )

            b_transition_logdensity = jax.scipy.stats.multivariate_normal.logpdf(
                    tc, mean=b_transition_mean, cov=b_transition_cov
            )

            # hastings acceptance probability
            # H = (next_density * b_transition_density) / (current_density * f_transition_density)
            # log H = log(next_density)    + log(b_transition_density)
            #       - log(current_density) - log(f_transition_density)

            H = np.exp(tc_next_logpdf - tc_logpdf + b_transition_logdensity - f_transition_logdensity)

            jax.debug.print('logpdf at proposal: {tc_next_logpdf}', tc_next_logpdf=tc_next_logpdf)
            jax.debug.print('logpdf at current: {tc_logpdf}', tc_logpdf=tc_logpdf)
            jax.debug.print('back transition log density: {b_transition_logdensity}', b_transition_logdensity=b_transition_logdensity)
            jax.debug.print('fwd transition log density: {f_transition_logdensity}', f_transition_logdensity=f_transition_logdensity)
            jax.debug.print('accept prob: {H}\n\n', H=H)

            # but how does this make sense? the backwards transition is
            # always going to be much less likely than the forward
            # transition as long as the gradient is not very small...

            u = jax.random.uniform(alpha_key)

            # say H = .9, then we accept with a probability of .9
            # = probability that u ~ U([0, 1]) < 0.9
            # the min(1, H) can be left out, if H>1 we accept always anyway
            do_accept = u < H

            next_tc = np.where(do_accept, tc_proposed, tc)

            # if do_accept, we go to proposed tc, therefore b_transition applies
            # otherwise we keep the current transition info
            # but in this hacky way because jax.lax.select is only for arrays not pytrees
            combine = lambda a, b: do_accept * a + (1-do_accept) * b
            next_transition_info = jax.tree_util.tree_map(combine, b_transition_info, current_transition_info)
            # next_transition_info = jax.lax.select(do_accept, b_transition_info, current_transition_info)

            next_state = (next_tc, next_transition_info, key)

            # looooots of outputs for debugging
            output = {
                    'key': key,
                    'tc': tc,
                    'tc_proposed': tc_proposed,
                    'current_transition_info': current_transition_info,
                    'b_transition_info': b_transition_info,
                    'H': H,
                    'u': u,
                    'tc': tc,
                    'y0': y0_current,
                    'do_accept': do_accept,
            }

            return next_state, output

        init_transition_info = mala_transition_info(tc0, algo_params['sampler_dt'])
        init_state = (tc0, init_transition_info, key)

        # TODO maybe: longer dt for burn in, shorter after?
        dts = algo_params['sampler_dt'] * np.ones(chain_length)

        if algo_params['sampler_tqdm']:
            # but why does it not work this time? even when just calling run_single_chain
            # without outer jit/vamp/pmap... :(
            scan_fct = jax_tqdm.scan_tqdm(n=chain_length)(scan_fct)

        # ipdb.set_trace()
        final_state, outputs = jax.lax.scan(scan_fct, init_state, dts)
        # return outputs['tc'], outputs['x0'], outputs['do_accept']
        return outputs


    # import sys; sys.exit()

    # vectorise & speed up :)
    run_multiple_chains = jax.vmap(run_single_chain, in_axes=(0, 0, None, None))
    # run_multiple_chains = jax.jit(run_multiple_chains, static_argnums=(3,))
    run_multiple_chains_nojit = jax.vmap(run_single_chain, in_axes=(0, 0, None, None))

    N_chains = algo_params['sampler_N_chains']
    keys = jax.random.split(key, N_chains)

    # starting exactly at 0 seems to mess stuff up...
    inits = 1e-3 * jax.random.normal(key, shape=(N_chains, nx))
    # basically accept the first proposal instead...?
    inits = np.array([-0.038, -0.047,  0.012,  0.016,  0.021,  0.002]).reshape(1, -1)

    burn_in = algo_params['sampler_burn_in']
    steps_per_sample = algo_params['sampler_steps_per_sample']
    samples = algo_params['samper_samples_per_chain']

    N_steps = burn_in + samples * steps_per_sample

    # heuristically found to represent the problem dimensions nicely
    jumpsizes = np.diag(np.sqrt(algo_params['x_sample_cov'])) # * algo_params['x_max_mahalanobis_dist']) * np.sqrt(problem_params['V_max']/10)

    # test run:
    # test_outputs = run_single_chain(key, inits[0, :], jumpsizes, 1024)
    # print('test run done')

    # visualise.
    def plot_trajectories(ts, ys, color='green', alpha=.1):

        # plot trajectories.
        pl.plot(ys[:, :, 0].T, ys[:, :, 1].T, color=color, alpha=alpha)

        # sols.ys.shape = (N_trajectories, N_ts, 2*nx+1)
        # plot attitude with quiver.
        arrow_x = ys[:, :, 0].reshape(-1)
        arrow_y = ys[:, :, 1].reshape(-1)
        attitudes = ys[:, :, 4].reshape(-1)
        arrow_len = 0.5
        u = np.sin(-attitudes) * arrow_len
        v = np.cos(attitudes) * arrow_len

        pl.quiver(arrow_x, arrow_y, u, v, color=color, alpha=alpha)

    # wrapped_pontryagin_solver = pontryagin_utils.make_pontryagin_solver_wrapped(problem_params, algo_params)
    # sol_p, y0s_p = wrapped_pontryagin_solver(test_outputs['tc_proposed'])
    # pl.subplot(211)
    # plot_trajectories(sol_p.ts, sol_p.ys, color='red')
    # pl.xlabel('proposed trajectories')
    #
    # sol_p, y0s_p = wrapped_pontryagin_solver(test_outputs['tc'])
    # pl.subplot(212)
    # plot_trajectories(sol_p.ts, sol_p.ys)
    # pl.xlabel('accepted trajectories')
    # pl.show()


    print(f'sampler: jumpsizes = {jumpsizes}')
    print(f'sampler will generate')
    print(f'    {N_steps*N_chains} samples = {N_chains} chains * ({burn_in} burn in + {samples} usable samples/chain * {steps_per_sample} subsampling)')
    print(f'and return a total of ')
    print(f'    {samples*N_chains}')
    print(f'usable samples. good luck :)')

    # all_tcs, all_x0s, accept = run_multiple_chains(keys, inits, np.ones(2), N_steps)
    # t0 = time.perf_counter()
    # outputs = run_multiple_chains_nojit(keys, inits, np.ones(2), N_steps)
    # t1 = time.perf_counter()
    # print(f'time no-jit run: {t1-t0:.4f}')

    t0 = time.perf_counter()
    outputs = run_multiple_chains(keys, inits, jumpsizes, N_steps)
    t1 = time.perf_counter()

    print(f'accept rate: {outputs["do_accept"].mean():.3f}')

    print(f'time for 1st jit run: {t1-t0:.4f}')
    print(f'time per sample     : {(t1-t0)/(samples*N_chains):.4f}')

    # t0 = time.perf_counter()
    # outputs = run_multiple_chains(keys, inits, jumpsizes, N_steps)
    # t1 = time.perf_counter()
    # print(f'time for 2nd jit run: {t1-t0:.4f}')
    # print(f'time per sample     : {(t1-t0)/(samples*N_chains):.4f}')

    all_tcs = outputs['tc']

    # outputs['y0'].shape == (N_chains, N_steps, nx)
    all_x0s = outputs['y0'][:, :, 0:nx]
    all_y0s = outputs['y0'][:, :, nx:2*nx]
    all_v0s = outputs['y0'][:, :, -1]
    accept = outputs['do_accept']

    # discard some burn-in and subsample for approximate independence.

    subsampled_y0s = outputs['y0'][:, burn_in::steps_per_sample, :].reshape(-1, 2*nx+1)
    subsampled_x0s = subsampled_y0s[:, 0:nx]
    subsampled_λ0s = subsampled_y0s[:, nx:2*nx]
    subsampled_v0s = subsampled_y0s[:, -1]

    subsampled_λTs = outputs['tc'][:, burn_in::steps_per_sample, :].reshape(-1, nx)

    # also write the dataset into a csv so we have it ready the next time.
    sysname = problem_params['system_name']

    np.save(f'datasets/last_y0s_{sysname}.npy', subsampled_y0s)
    np.save(f'datasets/last_lamTs_{sysname}.npy', subsampled_λTs)
    np.save(f'datasets/mcmc_complete/last_y0s_{sysname}.npy', outputs['y0'])
    np.save(f'datasets/mcmc_complete/last_lamTs_{sysname}.npy', outputs['tc'])

    if algo_params['sampler_plot']:

        pl.figure('V(x) and V(x(λ))')
        pl.subplot(121)
        extent = 1.2 * np.max(np.abs(subsampled_x0s))
        # plotting_utils.plot_fct(lambda x: np.exp(reward_fct_x0(x)), (-extent, extent), (-extent, extent))

        pl.subplot(122)
        extent = 1.2 * np.max(np.abs(subsampled_λTs))
        plotting_utils.plot_fct(lambda λ: np.exp(logpdf(λ)), (-extent, extent), (-extent, extent))


        # make the normal plot

        trajectory_alpha = .2
        scatter_alpha = .2

        # here we plot the samples and desirability function over x(0)
        pl.subplot(121)
        for x0 in all_x0s:
            pl.plot(x0[burn_in:, 0], x0[burn_in:, 1], color='grey', alpha=trajectory_alpha)

        # plot ellipse.
        plotting_utils.plot_ellipse(algo_params['x_Q_S'])

        # pl.scatter(subsampled_x0s[:, 0], subsampled_x0s[:, 1], color='red', alpha=scatter_alpha)

        # and here as a function of λ(T)
        pl.subplot(122)

        for tc in all_tcs:
            pl.plot(tc[burn_in:, 0], tc[burn_in:, 1], color='grey', alpha=trajectory_alpha)

        # pl.scatter(all_tcs_flat[:, 0], all_tcs_flat[:, 1], color='red', alpha=scatter_alpha)

        # again basically the same plot but as a coloured scatterplot.


        plotting_utils.value_lambda_scatterplot(subsampled_x0s, subsampled_v0s, subsampled_λTs)


        pl.figure('acceptance probabilities (for each chain)')
        pl.hist(accept.mean(axis=1))


        # make mcmc diagnostics plot.

        # time series
        pl.figure()

        # all_x0s.shape == (N_chains, N_samples, nx)
        for x0 in all_x0s:
            pl.subplot(211)
            pl.plot(x0[:, 0], alpha=.3)
            pl.subplot(212)
            pl.plot(x0[:, 1], alpha=.3)


        '''
        this takes AGES
        for x0 in all_x0s:
            N_pts = x0.shape[0]
            corrmat = x0 @ x0.T
            autocorrs = np.array([np.diagonal(corrmat, offset).mean() for offset in range(N_pts)])
            pl.plot(autocorrs/autocorrs[0], color='black', alpha=0.2)
        '''

        pl.show()


    return subsampled_y0s, subsampled_λTs






if __name__ == '__main__':

    raise NotImplementedError('this code too old')

    # minimal example of how to use the sampler.
    # explanation in docstring of pontryagin_sampler.
    # (this is old code, use geometric_mala_2, like in main file)

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


    problem_params = {
            'f': f,
            'l': l,
            'h': h,
            'T': 8,
            'nx': 2,
            'nu': 1,
            'terminal_constraint': True,
    }

    x_sample_scale = 1 * np.eye(2)
    x_sample_cov = x_sample_scale @ x_sample_scale.T

    # algo params copied from first resampling characteristics solvers
    # -> so some of them might not be relevant
    algo_params = {
            'sampler_n_trajectories': 128,
            'sampler_dt': 1/16,
            'sampler_n_iters': 8,
            'sampler_n_extrarounds': 2,
            'sampler_strategy': 'importance',
            'sampler_deterministic': False,
            'sampler_plot': True,
            'sampler_returns': 'sampling_fct',

            'x_sample_cov': x_sample_cov,
    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    sample, integrate = pontryagin_sampler(problem_params, algo_params)

    sol_obj, ys = integrate(sample(jax.random.PRNGKey(0), 10))
    with np.printoptions(precision=3, linewidth=np.inf, suppress=True):
        print(ys)


