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


def geometric_mala_2(integrate_fct, reward_fct_x0, problem_params, algo_params, key=jax.random.PRNGKey(0)):
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
    # dt = algo_params['sampler_dt']


    integrate_fct_reshaped = lambda tc: integrate_fct(tc.reshape(1, nx))[1][:, 0:nx].reshape(nx)

    # DO NOT confuse these two or it will cause days of bug hunting
    logpdf = lambda tc: reward_fct_x0(integrate_fct_reshaped(tc))
    logpdf_x0 = reward_fct_x0

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
            x0 = integrate_fct_reshaped(tc)
            dx0_dtc = jax.jacobian(integrate_fct_reshaped)(tc)
            dtc_dx0 = np.linalg.inv(dx0_dtc)  # by implicit function thm :)

            dlogpdf_dx0 = jax.jacobian(logpdf_x0)(x0)

            # propose a jump in x0, then transform it to λT.
            x0_jump_mean = dt * dlogpdf_dx0 * jump_sizes

            # as before, scale down the jump if very large.
            max_x0_jump_norm = 0.5
            correction = np.maximum(1, np.linalg.norm(x0_jump_mean)/max_x0_jump_norm)
            x0_jump_mean = x0_jump_mean / correction

            tc_jump_mean = dtc_dx0 @ x0_jump_mean

            new_tc_mean = tc + tc_jump_mean

            # covariances change under linear maps like this;
            # https://stats.stackexchange.com/questions/113700/
            x0_jump_cov = np.diag(2 * dt * jump_sizes**2) # = (sqrt(2dt) jump_sizes)**2
            tc_jump_cov = dtc_dx0 @ x0_jump_cov @ dtc_dx0.T
            new_tc_cov = tc_jump_cov  # bc. cov(tc) = 0

            # as for the logpdf: we know (from pdf change of variable, see report):
            #   g(λT) = p(PMP(λT)) * abs(det(dPMP(λT)/dλT))
            # thus:
            #   log g(λT) = log ( p(PMP(λT)) * abs(det(dPMP(λT)/dλT)) )
            #             = log p(PMP(λT)) + log abs(det(dPMP(λT)/dλT))

            logpdf_value_x0 = logpdf_x0(x0)
            detjac = np.linalg.det(dx0_dtc)

            # + small number to avoid gradients of log(0)
            # but if PMP is invertible as assumed then abs(detjac) > 0 anyways
            logpdf_value = logpdf_value_x0 + np.log(1e-9 + np.abs(detjac))

            return new_tc_mean, new_tc_cov, logpdf_value, x0

        def scan_fct(state, dt):

            tc, current_transition_info, key = state

            key, noise_key, alpha_key = jax.random.split(key, 3)


            # calculate all the transition properties at once.
            # f for forward transition.
            f_transition_mean, f_transition_cov, tc_logpdf, x0_current = current_transition_info
            # f_transition_mean, f_transition_cov, tc_logpdf, x0_current = mala_transition_info(tc, dt)

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
                    # 'key': key,
                    # 'noise': noise,
                    # 'jac_value': jac_value,
                    # 'grad_logpdf_now': grad_logpdf_now,
                    # 'x0_jump_desired': x0_jump_desired,
                    # 'proposed_x0_desired': proposed_x0_desired,
                    # 'proposed_x0': proposed_x0,
                    # 'next_tc': next_tc,
                    # 'H': H,
                    # 'u': u,
                    'tc': tc,
                    'x0': x0_current,
                    'do_accept': do_accept,
            }

            return next_state, output

        init_transition_info = mala_transition_info(tc0, algo_params['sampler_dt'])
        init_state = (tc0, init_transition_info, key)

        # TODO maybe: longer dt for burn in, shorter after?
        dts = algo_params['sampler_dt'] * np.ones(chain_length)

        if algo_params['sampler_tqdm']:
            # why does this still not result in a progress bar?
            scan_fct = jax_tqdm.scan_tqdm(n=chain_length)(scan_fct)

        # ipdb.set_trace()
        final_state, outputs = jax.lax.scan(scan_fct, init_state, dts)
        # return outputs['tc'], outputs['x0'], outputs['do_accept']
        return outputs


    # test run:
    # test_tc, test_x0, test_accept = run_single_chain(key, np.array([0.001, 0.001]), np.ones(2), 100)
    # ipdb.set_trace()

    # import sys; sys.exit()


    # vectorise & speed up :)
    run_multiple_chains = jax.vmap(run_single_chain, in_axes=(0, 0, None, None))
    run_multiple_chains = jax.jit(run_multiple_chains, static_argnums=(3,))
    run_multiple_chains_nojit = jax.vmap(run_single_chain, in_axes=(0, 0, None, None))

    N_chains = algo_params['sampler_N_chains']
    keys = jax.random.split(key, N_chains)
    inits = 0.0 * jax.random.normal(key, shape=(N_chains, nx))

    # try to initialise at some weird point
    # inits = np.kron(np.array([-14.8, 15.8]), np.ones((N_chains, 1)))

    burn_in = algo_params['sampler_burn_in']
    steps_per_sample = algo_params['sampler_steps_per_sample']
    samples = algo_params['sampler_samples']

    N_steps = burn_in + samples * steps_per_sample

    jumpsizes = np.diag(algo_params['x_sample_cov'] * algo_params['x_max_mahalanobis_dist'])

    # all_tcs, all_x0s, accept = run_multiple_chains(keys, inits, np.ones(2), N_steps)
    # t0 = time.perf_counter()
    # outputs = run_multiple_chains_nojit(keys, inits, np.ones(2), N_steps)
    # t1 = time.perf_counter()
    # print(f'time no-jit run: {t1-t0:.4f}')

    t0 = time.perf_counter()
    outputs = run_multiple_chains(keys, inits, jumpsizes, N_steps)
    t1 = time.perf_counter()
    print(f'time for 1st jit run: {t1-t0:.4f}')

    # t0 = time.perf_counter()
    # outputs = run_multiple_chains(keys, inits, jumpsizes, N_steps)
    # t1 = time.perf_counter()
    # print(f'time for 2nd jit run: {t1-t0:.4f}')

    all_tcs = outputs['tc']
    all_x0s = outputs['x0']
    accept = outputs['do_accept']

    # discard some burn-in and subsample for approximate independence.
    all_tcs_flat = all_tcs[:, burn_in::steps_per_sample, :].reshape(-1, nx)
    all_x0s_flat = all_x0s[:, burn_in::steps_per_sample, :].reshape(-1, nx)

    # ipdb.set_trace()

    if algo_params['sampler_plot']:

        pl.subplot(121)
        extent = 1.2 * np.max(np.abs(all_x0s_flat))
        plotting_utils.plot_fct(lambda x: np.exp(reward_fct_x0(x)), (-extent, extent), (-extent, extent))

        pl.subplot(122)
        extent = 1.2 * np.max(np.abs(all_tcs_flat))
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

        # pl.scatter(all_x0s_flat[:, 0], all_x0s_flat[:, 1], color='red', alpha=scatter_alpha)

        # and here as a function of λ(T)
        pl.subplot(122)

        for tc in all_tcs:
            pl.plot(tc[burn_in:, 0], tc[burn_in:, 1], color='grey', alpha=trajectory_alpha)

        # pl.scatter(all_tcs_flat[:, 0], all_tcs_flat[:, 1], color='red', alpha=scatter_alpha)

        pl.figure('acceptance probabilities (for each chain)')
        pl.hist(accept.mean(axis=1))


        # make mcmc diagnostics plot.

        # time series
        pl.figure()
        pl.subplot(211)

        # all_x0s.shape == (N_chains, N_samples, nx)
        for x0 in all_x0s:
            pl.plot(x0, alpha=.3)
        # pl.plot(all_x0s[0, :, 0])
        # pl.plot(all_x0s[0, :, 1])

        # autocorrelation function
        pl.subplot(212)


        '''
        this takes AGES
        for x0 in all_x0s:
            N_pts = x0.shape[0]
            corrmat = x0 @ x0.T
            autocorrs = np.array([np.diagonal(corrmat, offset).mean() for offset in range(N_pts)])
            pl.plot(autocorrs/autocorrs[0], color='black', alpha=0.2)
        '''

        pl.show()
        ipdb.set_trace()




def adam_uncertainty_sampler(logpdf, init_population, algo_params, key=jax.random.PRNGKey(666)):
    '''
    another heuristic global optimiser. maybe this one is better?
    built a bit differently than the ULA version. here we write a single
    chain first and then do one vmap to do it N times.
    '''

    dt = algo_params['sampler_dt']
    N_chains, n = init_population.shape

    grad_logpdf = jax.grad(logpdf)

    def run_single_chain(key, x0, noise_schedule):

        lr = 0.01

        opti = optax.adam(learning_rate=0.01)

        '''
        def update(x, opt_state, key, noise_scale):
            noise = jax.random.normal(key, shape=(n,)) * noise_scale
            return x + dt * grad_logpdf(x) + np.sqrt(2 * dt) * noise
        '''

        def scan_fct(state, inp):

            # unpack state
            x, opt_state, key = state
            noise_scale = inp

            # optax update - we take the negative gradient for maximisation
            logpdf_value, logpdf_grad = jax.value_and_grad(logpdf)(x)
            updates, opt_state = opti.update(-logpdf_grad, opt_state, x)
            next_x = optax.apply_updates(x, updates)

            # plus noise
            key, noise_key = jax.random.split(key)
            next_x_noisy = next_x + noise_scale * np.sqrt(2*lr) * jax.random.normal(noise_key, shape=x.shape)

            # re-pack state
            next_state = (next_x, opt_state, key)
            return next_state, (next_x_noisy, logpdf_value)

        opt_state = opti.init(x0)
        init_state = (x0, opt_state, key)

        _, (xs, logpdfs) = jax.lax.scan(scan_fct, init_state, noise_schedule)

        return xs, logpdfs

    # build noise schedule and output indices.
    burn_in = algo_params['sampler_burn_in']
    samples = algo_params['sampler_samples']
    steps_per_sample = algo_params['sampler_steps_per_sample']

    N_steps = burn_in + samples * steps_per_sample

    burn_in_noise_schedule = algo_params['sampler_burn_in_noise'] * np.ones(burn_in)
    init_noise = algo_params['sampler_init_noise']
    final_noise = algo_params['sampler_final_noise']
    sample_noise_schedule = np.logspace(np.log10(init_noise), np.log10(final_noise), steps_per_sample)

    noise_schedule = np.concatenate([burn_in_noise_schedule] + samples * [sample_noise_schedule])
    output_idx = burn_in + np.arange(1, samples+1) * steps_per_sample - 1

    # to actually just run a single chain, do this
    # and get xs (N_steps, nx) and logpdfs (N_steps,)
    # xs, logpdfs = run_single_chain(key, init_population[0], noise_schedule)

    run_multiple_chains = jax.vmap(run_single_chain, in_axes=(0, 0, None))
    keys = jax.random.split(key, init_population.shape[0])
    all_xs, all_logpdfs = run_multiple_chains(keys, init_population, noise_schedule)

    # we copied the plotting code but the other algo spits out data in
    # different format, so we transpose accordingly here
    # _, (all_samples, logpdf_values) = jax.lax.scan(scan_fct, init_population, inputs)
    # #  (2560, 32, 2) (2560, 32)  <- shapes for one particular run
    all_samples = all_xs.swapaxes(0, 1)
    logpdf_values = all_logpdfs.swapaxes(0, 1)  # equivalent to .T

    if algo_params['sampler_plot']:
        print('making mcmc plot...')

        pl.subplot(121)
        # find out the logpdf for a reasonable range of norm_tcs & plot it
        extent = 5
        xs = ys = np.linspace(-extent, extent, 512)

        xx, yy = np.meshgrid(xs, ys)

        all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        all_logpdfs = jax.vmap(logpdf)(all_inputs)

        # pl.pcolor(xx, yy, all_logpdfs.reshape(xx.shape), cmap='jet')
        pl.pcolor(xx, yy, np.exp(all_logpdfs/10).reshape(xx.shape), cmap='jet')

        # above that, plot the evolution of the MCMC chains
        pl.plot(all_samples[:, :, 0], all_samples[:, :, 1], color='grey', alpha=.25)
        pl.scatter(all_samples[output_idx, :, 0].squeeze(), all_samples[output_idx, :, 1].squeeze(), color='red')
        pl.scatter(all_samples[0, :, 0], all_samples[0, :, 1], color='green')

        pl.subplot(122)
        pl.plot(np.linalg.norm(all_samples, axis=2), color='blue', alpha=.1)
        pl.plot(logpdf_values, color='green', alpha=.1)

        pl.show()

    ipdb.set_trace()



def sample_from_logpdf(logpdf, init_population, algo_params, key=jax.random.PRNGKey(666)):
    '''
    simple unadjusted langevin algorithm.
    https://www.jeremiecoullon.com/2020/11/10/mcmcjax3ways/

    this is really not something that actually samples from the given pdf. this is
    more just a heuristic noisy gradient, multiple initialisation based approximate
    global optimiser. but ofc also its global optimisation capabilities are quite
    questionable.

    however for our purposes it might work \o/ completely depending on tuning.

    logpdf: jax function that takes a variable x in R^n and calculates
    its unnormalised log probability density
    init_population: (N_chains, n) array with initial samples.
    algo_params: dict containing whatever entries are needed here.
    '''


    dt = algo_params['sampler_dt']
    N_chains, n = init_population.shape

    grad_logpdf = jax.grad(logpdf)

    def update(x, key, noise_scale):
        noise = jax.random.normal(key, shape=(n,)) * noise_scale
        return x + dt * grad_logpdf(x) + np.sqrt(2 * dt) * noise

    # same noise scale for all chains.
    update_vmap = jax.vmap(update, in_axes=(0, 0, None))

    def scan_fct(xs, inp):
        key, noise_scale = inp
        keys_vmap = jax.random.split(key, N_chains)

        next_xs = update_vmap(xs, keys_vmap, noise_scale)
        logpdf_value = jax.vmap(logpdf)(next_xs)
        return next_xs, (next_xs, logpdf_value)

    # here we have the glorious mcmc tuning variables...
    # first, we 'burn in' for some number of steps.
    # then, we want <samples> samples per chain, and we iterate
    # the markov chain for <steps_per_sample> in between to make
    # it approximately independent.
    burn_in = algo_params['sampler_burn_in']
    samples = algo_params['sampler_samples']
    steps_per_sample = algo_params['sampler_steps_per_sample']

    N_steps = burn_in + samples * steps_per_sample

    # maybe worth tweaking this?
    # okay, I did. constant noise for burn in, then exponential decrease before every sample.
    burn_in_noise_schedule = algo_params['sampler_burn_in_noise'] * np.ones(burn_in)
    init_noise = algo_params['sampler_init_noise']
    final_noise = algo_params['sampler_final_noise']
    sample_noise_schedule = np.logspace(np.log10(init_noise), np.log10(final_noise), steps_per_sample)
    noise_scales = np.concatenate([burn_in_noise_schedule] + samples * [sample_noise_schedule])
    # pl.plot(noise_scales); pl.show()

    keys = jax.random.split(key, N_steps)
    inputs = (keys, noise_scales)

    # do the actual computation.
    # _ == all_samples[-1] so we don't really need it again
    _, (all_samples, logpdf_values) = jax.lax.scan(scan_fct, init_population, inputs)
    #  (2560, 32, 2) (2560, 32)  <- shapes for one particular run

    output_idx = burn_in + np.arange(1, samples+1) * steps_per_sample - 1


    if algo_params['sampler_plot']:
        print('making mcmc plot...')

        pl.subplot(121)
        # find out the logpdf for a reasonable range of norm_tcs & plot it
        extent = 5
        xs = ys = np.linspace(-extent, extent, 501)

        xx, yy = np.meshgrid(xs, ys)

        all_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        all_logpdfs = jax.vmap(logpdf)(all_inputs)

        # pl.pcolor(xx, yy, all_logpdfs.reshape(xx.shape), cmap='jet')
        pl.pcolor(xx, yy, np.exp(all_logpdfs/10).reshape(xx.shape), cmap='jet')

        # above that, plot the evolution of the MCMC chains
        pl.plot(all_samples[:, :, 0], all_samples[:, :, 1], color='grey', alpha=.25)
        pl.scatter(all_samples[output_idx, :, 0].squeeze(), all_samples[output_idx, :, 1].squeeze(), color='red')
        pl.scatter(all_samples[0, :, 0], all_samples[0, :, 1], color='green')

        pl.subplot(122)
        pl.plot(np.linalg.norm(all_samples, axis=2), color='blue', alpha=.1)
        pl.plot(logpdf_values, color='green', alpha=.1)

        pl.show()

    ipdb.set_trace()







def pontryagin_sampler(problem_params, algo_params, key=jax.random.PRNGKey(1337)):

    '''
    we have the following problem. there is some distribution of states at
    t=0 for which we want to know the optimal controls. We can do this with
    the pontryagin principle. but we need to sample from a suitable
    distribution over terminal boundary conditions (state or costate) to
    get interesting data.

    This function does that by importance sampling. Very rough description:

    - We sample from some distribution of terminal BC's and simulate
      optimal trajectories according to the PMP, up to t=t1.
    - We assign a weight to each of the trajectories that quantifies 'how
      much we like them'. the technicalities of how this weight is set
      basically define the method: importance sampling, rejection sampling,
      etc.
    - We find a new distribution (gaussian) over terminal BC's that most
      closely matches the reweighted version of the previous sampling
      distribution.

    These steps are iterated, and gradually t1 is stepped from T towards 0,
    so each time the distribution only changes a bit.

    It seems to work well. But due to the fact that we always 'collapse'
    the distribution over terminal BCs back to a gaussian, and due to some
    other approximations, we have no guarantees that the desired state
    distribution at t=0 is actually reached.

    problem_params: same as usual
    algo_params: dict with algorithm tuning parameters. see code.

    it will return different things based on algo_params configuration.

    if algo_params['pontryagin_sampler_returns'] == 'distribution_params', it will return
    a tuple containing mean and cov of the final sampling distribution.

    if algo_params['pontryagin_sampler_returns'] == 'sampling_fct', it will return a tuple of
    functions: (sample, integrate).

        sample takes a PRNGKey and an integer n specifying the number of
        samples, and returns a (n, nx) array of terminal conditions drawn
        from the final distribution.

        integrate is the function taking terminal conditions to optimal
        trajectories according to PMP. It returns a tuple containing the
        full solution object and the extended state at time 0, i.e. an
        array of shape (n, 2*nx+1), with the last dimension indexing
        state, costate, and value.

    '''

    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']

    # define the dynamics of optimal trajectories according to PMP.
    # TODO in here the input constraints are defined. move them to problem_params somehow?
    f_forward = pontryagin_utils.define_extended_dynamics(problem_params)

    # solve pontryagin backwards, for vampping later.
    # slightly differently parameterised than in other version.
    def pontryagin_backward_solver(y0, t0, t1):

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs

        # negative if t1 < t0, backward integration just works
        assert algo_params['pontryagin_sampler_dt'] > 0
        dt = algo_params['pontryagin_sampler_dt'] * np.sign(t1 - t0)

        # what if we accept that we could create NaNs?
        max_steps = int(1 + problem_params['T'] / algo_params['pontryagin_sampler_dt'])

        # maybe easier to control the timing intervals like this?
        saveat = diffrax.SaveAt(steps=True)

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt, y0=y0,
                saveat=saveat, max_steps=max_steps,
        )

        # this should return the last calculated (= non-inf) solution.
        return solution, solution.ys[solution.stats['num_accepted_steps']-1]

    # vmap = gangster!
    # vmap only across first argument.
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None, None)
    ))

    # construct the terminal extended state

    # helper function, expands the state vector x to the extended state vector y = [x, λ, v]
    # λ is the costate in the pontryagin minimum principle
    # h is the terminal value function
    def x_to_y_terminalcost(x, h):
        costate = jax.grad(h)(x)
        v = h(x)
        ipdb.set_trace()
        y = np.concatenate([x, costate, v])

        return y

    # slight abuse here: the same argument is used for λ as for x, be aware
    # that x_to_y in that case is a misnomer, it should be λ_to_y.
    # but as with a terminal constraint we are really just searching a
    # distribution over λ instead of x, but the rest stays the same.
    def x_to_y_terminalconstraint(λ, h=None):
        x = np.zeros(nx)
        costate = λ
        v = np.zeros(1)
        y = np.concatenate([x, costate, v])
        return y

    if problem_params['terminal_constraint']:
        # we instead assume a zero terminal constraint.
        x_to_y = x_to_y_terminalconstraint
    else:
        x_to_y = x_to_y_terminalcost

    x_to_y_vmap = jax.vmap(lambda x: x_to_y(x, h))


    def sample_terminal_conditions(key, mean, cov, n_samples):
        return jax.random.multivariate_normal(
                key,
                mean=mean,
                cov=cov,
                shape=(n_samples,)
        )

    # this is the initial distribution. choosing the desired state distribution here
    # has proven to work well for the toy example. but if units are weird or something
    # we may have to find something smarter.
    key, subkey = jax.random.split(key)
    x_T = sample_terminal_conditions(subkey, np.zeros(nx,), algo_params['x_sample_cov'], algo_params['pontryagin_sampler_n_trajectories'])


    y_T = x_to_y_vmap(x_T)

    # the desired distribution of states at t=0.
    desired_mean = np.zeros((nx,))
    desired_cov = algo_params['x_sample_cov']
    desired_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, desired_mean[None, :], desired_cov)


    '''
    # bit of a hack - comment this all out because we only need the (sample, integrate) functions anyway.
    # main loop iteration, for jax.lax.scan.

    # basically, this does the following:
    # - sample terminal boundary conditions ~ N(sampling_mean, sampling_cov)
    # - simulate optimal trajectories backwards using PMP, from T to t1.
    # - weigh them according to how much we like them. here we have several schemes:
    #   - importance sampling
    #   - cheap importance sampling without compensation for sampling likelihood
    #   - rejection sampling
    # - find the gaussian distribution over terminal boundary conditions that leads to
    #   the re-weighted distribution at t1.

    # to make the problem easier we solve this iteratively, each time making the integration
    # interval a bit longer, so the distribution does not change too wildly.

    def scan_fct(carry, inp):

        # input = t1 = END of integration interval. t1 < t0 = T
        # carry = (sampling_mean, sampling_cov, key)
        # output = dict with various useful entries, see for yourself :)

        sampling_mean, sampling_cov, key = carry

        # sampling step first.
        # if deterministic, we choose the same sample each time.
        if algo_params['pontryagin_sampler_deterministic']:
            subkey = key
        else:
            key, subkey = jax.random.split(key)

        x_T = sample_terminal_conditions(key, sampling_mean, sampling_cov, algo_params['pontryagin_sampler_n_trajectories'])
        y_T = x_to_y_vmap(x_T)

        t1 = inp

        # find optimal trajectories by pontryagin principle.
        sol_object, last_sol_vec = batch_pontryagin_backward_solver(y_T, T, t1)

        # find the relevant likelihoods.
        # IMPORTANT: the sampling likelihood is wrong, to be correct we would have to multiply by
        # the inverse jacobian determinant of the mapping from terminal BCs to initial state.
        # we find that omittig this gives satisfactory results. because the weights are normalised anyway,
        # it does not matter too much.
        desired_likelihoods = desired_pdf(last_sol_vec[:, 0:nx])
        sampling_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, sampling_mean[None, :], sampling_cov)
        sampling_likelihoods = sampling_pdf(x_T.reshape(algo_params['pontryagin_sampler_n_trajectories'], nx))

        # importance sampling weight - some different choices.
        resampling_weights_importance = desired_likelihoods / (1e-9 + sampling_likelihoods)

        # rejection sampling weights
        resampling_weights_rejection = desired_likelihoods >= 0.001

        # cheap in between weights
        # weight by likelihood at t1, but don't compensate for sampling distribution.
        resampling_weights_between = desired_likelihoods

        # if we are in the final iterations (t1=0), take the importance sampling weighting.
        # otherwise, 'between' ones seem to work better.
        is_final = np.allclose(t1, 0)
        resampling_weights_switched = is_final * resampling_weights_importance + (1-is_final) * resampling_weights_between

        if algo_params['pontryagin_sampler_strategy'] == 'importance':
            resampling_weights = resampling_weights_importance
        elif algo_params['pontryagin_sampler_strategy'] == 'rejection':
            resampling_weights = resampling_weights_rejection
        elif algo_params['pontryagin_sampler_strategy'] == 'between':
            resampling_weights = resampling_weights_between
        elif algo_params['pontryagin_sampler_strategy'] == 'switched':
            resampling_weights = resampling_weights_switched
        else:
            st = algo_params['pontryagin_sampler_strategy']
            raise ValueError(f'Invalid sampling strategy "{st}"')

        resampling_weights = resampling_weights / np.sum(resampling_weights)

        xs_flat = x_T.reshape(-1, nx)

        # calculate mean and covariance of the re-weighted terminal BC distribution.
        sampling_mean = np.mean(resampling_weights[:, None] * xs_flat, axis=0) * 0  # or just zero?

        # subtract mean
        xs_zm = xs_flat - sampling_mean[None, :]

        # for zero mean RV x: cov(x) = E[x x'] = Σ_x p(x) * x x' = Σ_x (x*sqrt(p(x)) * (x*sqrt(p(x))'
        # so basically to apply the 'regular' covariance formula, we need to scale the
        # data points by the square root of their weights.
        # but where did the n-1 go? bessel correction mean estimate what?
        sqrt_scaled_datapts = np.sqrt(1e-9 + resampling_weights[:, None]) * xs_flat

        # (wide matrix of column vecs) @ (tall matrix of row vecs)
        sampling_cov = sqrt_scaled_datapts.T @ sqrt_scaled_datapts

        # do the same but for the y_T vector - the covariance we get at t1.
        # just for output bookkeeping.
        last_sol_zm = last_sol_vec[:, 0:nx]
        last_sol_zm = last_sol_zm - last_sol_zm.mean(axis=0)[None, :]  # (N_traj, nx) - (1, nx) promotes correctly
        last_sol_cov = last_sol_zm.T @ last_sol_zm / (algo_params['pontryagin_sampler_n_trajectories'] - 1)


        # output = (sol_object, sampling_cov, resampling_weights)
        # output as dict is way nicer. scan automatically 'transposes' it into
        # a dict of arrays, not an array of dicts :)
        output = {
                'solutions': sol_object,
                'sampling_means': sampling_mean,
                'sampling_covs': sampling_cov,
                't1_covs': last_sol_cov,
                'resampling_ws': resampling_weights,
        }

        new_carry = (sampling_mean, sampling_cov, key)

        return new_carry, output

    # the sequence of times to integrate. always from T to t1s[i].
    # at the end we iterate a couple of times at t1=0, just to be sure.
    t1s = np.concatenate([
        np.linspace(0, T, algo_params['pontryagin_sampler_n_iters'], endpoint=False)[::-1],
        np.zeros(algo_params['pontryagin_sampler_n_extrarounds']),
    ])

    N = t1s.shape[0]

    init_carry = (np.zeros(nx,), algo_params['x_sample_cov'], key)

    # do it twice so we see how fast it actually is...

    start = time.perf_counter()
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, t1s, length=N)
    end = time.perf_counter()
    print(f'time for jit/first run: {end-start}')
    start = time.perf_counter()
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, t1s, length=N)
    end = time.perf_counter()
    print(f'time for second run: {end-start}')

    sol_object = outputs['solutions']
    all_ys = sol_object.ys

    sampling_covs = outputs['sampling_covs']
    t1_covs = outputs['t1_covs']
    resampling_ws = outputs['resampling_ws']
    '''


    if algo_params['pontryagin_sampler_plot']:
        fig = pl.figure(figsize=(8, 3))

        dims = 2
        if dims==3:
            ax3 = fig.add_subplot(111, projection='3d')
        else:
            assert dims==2
            ax0 = fig.add_subplot(211)
            ax1 = fig.add_subplot(212)

        # basically the same but in a phase plot
        fig = pl.figure(figsize=(4, 4))
        ax_phase = fig.add_subplot(111)

        for i in range(N):
            print(f'plotting iteration {i}...')
            t_plot = sol_object.ts[i, 0]

            # the two state vectors...
            x_plot = sol_object.ys[i, :, :, 0].T
            y_plot = sol_object.ys[i, :, :, 1].T

            # color basically shows iteration number
            # alpha shows likelihood of trajectory for resampling?
            c = matplotlib.colormaps.get_cmap('gist_gray')(i/N)
            a = onp.array(resampling_ws[i] / np.max(resampling_ws[i]))

            if i==N-1:
                # visualise final distribution differently...
                c = 'red'
                a = onp.ones_like(a) * .25

            # colors_with_alpha = onp.zeros((algo_params['pontryagin_sampler_n_trajectories'], 4))
            # colors_with_alpha[:, 3] = a
            # colors_with_alpha[:, 0:3] = onp.array(c)[None, 0:3]


            # bc. apparently alpha cannot be a vector :(
            for j in range(x_plot.shape[1]):
                if dims==3:
                    ax3.plot(t_plot, x_plot[:, j], y_plot[:, j], color=c, alpha=a[j])
                else:
                    ax0.plot(t_plot, x_plot[:, j], color=c, alpha=a[j])
                    ax1.plot(t_plot, y_plot[:, j], color=c, alpha=a[j])
                ax_phase.plot(x_plot[:, j], y_plot[:, j], color=c, alpha=a[j]/5)

        # sol_object.ys indexed by [iter, trajectory, timeindex, extended state dim]
        # note that time index is 0 at T and N at 0, so reversed.
        last_valid_index = sol_object.stats['num_accepted_steps'][-1, 0]-1
        init_states = sol_object.ys[-1, :, last_valid_index, 0:2]
        ax_phase.scatter(init_states[:, 0], init_states[:, 1], c='red', alpha=1/4)


        pl.figure()
        pl.subplot(211)
        pl.plot(sampling_covs.reshape(N, -1), label='sampling cov. entries')
        pl.legend()
        pl.subplot(212)
        pl.plot(t1_covs.reshape(N, -1), label='t1 cov. entries')
        pl.legend()
        pl.show()


    # previously for benchmarking
    # return end-start

    # last_sampling_mean = outputs['sampling_means'][-1]
    # last_sampling_cov = outputs['sampling_covs'][-1]

    sampling_fct = lambda key, n: sample_terminal_conditions(key, last_sampling_mean, last_sampling_cov, n)
    integrate_fct = lambda xT: batch_pontryagin_backward_solver(x_to_y_vmap(xT), T, 0)

    if algo_params['pontryagin_sampler_returns'] == 'distribution_params':
        # now, return the parameters of the terminal condition distribution
        return (last_sampling_mean, last_sampling_cov)
    elif algo_params['pontryagin_sampler_returns'] == 'functions':
        return (sampling_fct, integrate_fct)
    elif algo_params['pontryagin_sampler_returns'] == 'both':
        # put it all in a dictionary
        return {
                'mean': last_sampling_mean,
                'cov': last_sampling_cov,
                'sampling_fct': sampling_fct,
                'integrate_fct': integrate_fct,
        }
    else:
        ret = algo_params['pontryagin_sampler_returns']
        raise ValueError(f'invalid sampler return value specification "{ret}"')




if __name__ == '__main__':

    # minimal example of how to use the sampler.
    # explanation in docstring of pontryagin_sampler.

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


