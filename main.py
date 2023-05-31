#!/usr/bin/env python

import jax
import jax.numpy as np

import gradient_gp
import sampling

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import tqdm
from functools import partial


def run_algo(problem_params, algo_params, key=None):

    if key is None:
        key = jax.random.PRNGKey(0)

    # first we generate a bunch of trajectories with the pontryagin sampler.
    key, subkey = jax.random.split(key)
    sampler_output = sampling.pontryagin_sampler(problem_params, algo_params, key=subkey)

    sample =        sampler_output['sampling_fct']
    integrate =     sampler_output['integrate_fct']
    sampling_mean = sampler_output['mean']
    sampling_cov =  sampler_output['cov']

    key, subkey = jax.random.split(key)
    sol_obj, ys = integrate(sample(subkey, algo_params['sampler_n_trajectories']))

    nx = problem_params['nx']

    # fit a GP to the data.
    gp_params = {
            'log_amp': np.log(1),
            'log_scale': np.zeros(nx),
    }

    xs_gp, ys_gp, grad_flags_gp = gradient_gp.reshape_for_gp(ys)

    print('optimising GP...')
    gp, gp_params = gradient_gp.get_optimised_gp(
            gradient_gp.build_gp,
            gp_params,
            xs_gp,
            ys_gp,
            grad_flags_gp,
            steps=algo_params['gp_iters'],
            plot=algo_params['gp_train_plot'],
    )

    all_ys = ys

    # we want to locally optimise the uncertainty in the V approximation,
    # with respect to the terminal condition. so basically we compose the
    # pontryagin solver with the uncertainty model to obtain model
    # uncertainty directly as a function of terminal condition.
    # the pontryagin solver is already in nice functional form as integrate


    def V_var(terminal_cond, gp, ys_gp):
        '''
        same as the other one, but only for one point. so we can first get
        the gradient mapping, then vmap. hoping that jit will compensate
        for the unnecessary vmap that already is in the integrate fct.

        inputs:
        gp:             tinygp gaussian process object for V approximation
        ys_gp:          data to condition gp on (V and V_x,
                        gp.X = (xs, gradflags) define which are where)
        terminal_conds: (nx,) array with either of:
                        x(T) if using terminal state + state cost
                        λ(T) if using terminal state constraint at 0

        output: posterior uncertainty in V approx.
          the units or σ vs σ**2 don't matter too much, we just search for
          high uncertainty.


        '''

        # reshape because it is vmapped already inside...
        sol_obj, init_ys = integrate(terminal_cond.reshape(1, nx))

        init_states = init_ys[:, 0:nx]

        # condition the GP on available data & evaluate uncertainty at xs_eval.
        gradflags = np.zeros(1, dtype=np.int8) # only 1 data point here
        pred_gp = gp.condition(ys_gp, (init_states, gradflags)).gp

        # reshape to make scalar.
        return pred_gp.variance.reshape()

    # these are to make the sensitivity problem somewhat less severe, and
    # to make the tuning of gradient descent type optimisers easier.
    # instead of directly taking x(T) or λ(T) as input for everything, we
    # take a re-scaled version, which we can sample from N(0,1) and get
    # sensible results.

    def normalise_term_cond(term_cond):
        Σ_half_inv = jax.scipy.linalg.sqrtm(sampling_cov).real
        return Σ_half_inv @ term_cond

    def unnormalise_term_cond(term_cond_normalised):
        Σ_half = jax.scipy.linalg.sqrtm(sampling_cov).real
        return Σ_half @ term_cond_normalised

    def V_var_normalised(term_cond_normalised, gp, ys_gp):
        term_cond = unnormalise_term_cond(term_cond_normalised)
        return V_var(term_cond, gp, ys_gp)


    # now including the state constraints as a penalty term.
    # otherwise the same as V_var_normalised.
    def desirability_fct(term_cond_normalised, gp, ys_gp):
        term_cond = unnormalise_term_cond(term_cond_normalised)

        sol_obj, init_ys = integrate(terminal_cond.reshape(1, nx))
        init_states = init_ys[:, 0:nx]
        gradflags = np.zeros(1, dtype=np.int8) # only 1 data point here
        pred_gp = gp.condition(ys_gp, (init_states, gradflags)).gp
        predictive_variance = pred_gp.variance.reshape()

        # come to think of it: is sqrt(x.T Σ x) = x.T @ Σ^(1/2) @ x ??
        # feels like it shouldn't be that way
        # certainly || Σ^(-1/2) x ||_2^2 == x.T @ Σ^-1 @ x
        Σ_inv = algo_params['x_sample_cov']
        mahalanobis_dist = np.sqrt(init_states.T @ Σ_inv @ init_states)

        # penalise large distances to stay within interesting state region
        # square again so larger adjustments are made if very far outside
        state_penalty = 100 * np.max(0, mahalanobis_dist - 2)**2

        # we are maximising to get the most desirable terminal condition
        # large predictive variance = good
        # large state penalty = bad
        desirability = predictive_variance - state_penalty

        return desirability



    # new version: gradient based search for high uncertainty in terms
    # of terminal conditions. hopefully nicer to implement, and more
    # obvious scaling to larger state spaces & NNs.
    # each loop iteration consists of these steps:
    # a) find terminal conditions that generate new data at x(0) which
    #    currently have an uncertain V estimate
    # b) use that data to update the GP

    # We separate the sampling method into two components.

    # 1. define a scalar function over x(0) that specifies how interested
    # we are in obtaining information about V, λ, u* at this loation. Say
    # desirability(x0).
    #
    # 2. find terminal conditions that lead to high desirability at x0.
    # straightforwardly we can compose the desirability function with the
    # PMP solver to obtain the desirability as a function of terminal
    # condition. then we are free to choose any algorithm to maximise
    # desirability w.r.t. terminal condition, e.g. gradient descent with
    # many random initialisations, fancier batched local/approximate global
    # optimisers, sampling from distributions ~ exp(desirability) with
    # SGLD, HMC, or anything of that sort.

    # this here just as a small test.
    key, subkey = jax.random.split(key)
    tcs = sample(key, algo_params['sampler_n_trajectories'])

    # all based on normalised terminal conditions!!
    # so be sure to scale by cov^(1/2) with functions above
    V_var_grad = jax.grad(V_var_normalised, argnums=0)
    V_var_vmap = jax.vmap(V_var_normalised, in_axes=(0, None, None))
    V_var_grad_vmap = jax.vmap(V_var_grad, in_axes=(0, None, None))


    N = algo_params['active_learning_iters']

    for i in range(N):

        gradient = True
        if gradient:

            # a) sample terminal conditions, push them in direction of
            # increasing desirability. but keep the points within the state
            # space region of interest.

            # also, maybe build in a check of some sort that exist when the
            # uncertainty appears to be below some threshold?

            # also, maybe better to use adam instead of plain gradient descent?

            key, subkey = jax.random.split(key)
            tcs_norm = jax.random.normal(subkey, shape=(algo_params['sampler_n_trajectories'], nx))

            def gradient_step(tcs_norm, lr):

                V_var_grads = V_var_grad_vmap(tcs_norm, gp, ys_gp)
                V_vars = V_var_vmap(tcs_norm, gp, ys_gp)  # to debug

                return (tcs_norm + lr * V_var_grads, V_vars)

            lrs = np.logspace(-1, -2, 50)  # decreasing step size?
            print('finding uncertain points...')
            tcs_norm, V_vars = jax.lax.scan(gradient_step, tcs_norm, lrs)

            new_tcs = jax.vmap(unnormalise_term_cond)(tcs_norm)
            # did this already in V_var_grad_vmap but messy to keep solution
            print('integrating PMP again...')
            new_sol_obj, new_ys = integrate(new_tcs)

        else:
            # this we can probably throw away wlog. Just set iterations=0
            # for the gradient type algorithms and we have this already.
            # maybe easier to just generate very many many trajectories and
            # chooose the ones with highest uncertainty?
            # just an idea, not tested, this is pseudocode!
            # but maybe if we fully embrace the sampling-type algorithms
            # everything will be less shitty

            tcs = sample_lots_of_terminal_conditions(n=4096)
            sol_object, ys = integrate(tcs)
            V_vars = V_var_vmap(tcs, gp, ys_gp)
            highest_vars, idx = jax.lax.top_k(V_vars, k=n_trajectories)
            new_ys = ys[idx]
            # then continue the same way.




        cmap = matplotlib.cm.get_cmap('viridis')
        pl.scatter(new_ys[:, 0], new_ys[:, 1], color=cmap(i/N))

        # b) add the newly found data to the training set.
        print('stacking data...')
        all_ys = np.concatenate([all_ys, new_ys], axis=0)
        xs_gp, ys_gp, grad_flags_gp = gradient_gp.reshape_for_gp(ys)

        # something like this apparently does not work because the gp can
        # not change its x points
        # gp_cond = gp.condition(ys, (xs, gradflags)).gp

        # so instead:
        print('conditioning GP...')
        gp = gradient_gp.build_gp(gp_params, xs_gp, grad_flags_gp)

    pl.show()
    ipdb.set_trace()











    if False:
        for i in range(4):


            # older version. idea:
            # - find points where the approximation is inaccurate
            # - guess which terminal conditions we might use to get useful new data
            #   at these points (hoping for help from the GP here)
            # - get the new data with the pontryagin solver
            # - re-fit the GP.

            # as a first attempt, we just generate loads of random states
            # and choose the ones with hightest GP uncertainty to take the
            # next samples.

            # not sure if this is guaranteed to do anything smart, probably to get
            # decent convergence guarantees we have to sample over a bounded domain,
            # this way it will just choose the outermost, most unlikely draws for
            # new samples because there we have no data yet.


            n_eval = 256
            key, subkey = jax.random.split(key)
            xs_eval = jax.random.multivariate_normal(
                    subkey,
                    mean=np.zeros(nx),
                    cov=algo_params['x_sample_cov'],
                    shape=(n_eval,)
            )

            # no gradients! only the function.
            # although, should we not mainly care about accurate representation
            # of the gradient.....?
            gradflags_eval = np.zeros((n_eval,), dtype=np.int8)

            # condition the GP on available data & evaluate uncertainty at xs_eval.
            pred_gp = gp.condition(ys_gp, (xs_eval, gradflags_eval)).gp

            # find xs with largest value uncertainty. for simplicity again the same
            # number of trajectories as sampling -> less re-jitting.
            k = algo_params['sampler_n_trajectories']
            largest_vars, idx = jax.lax.top_k(pred_gp.variance, k)

            # these are the states we'd like to know more about.
            uncertain_xs = xs_eval[idx]
            uncertain_xs_gradflag = np.ones(k, dtype=np.int8)

            # find the weights the gp used to predict y = w.T @ ys_train
            # TODO next week: continue with this.
            X_pred = (uncertain_xs, uncertain_xs_gradflag)

            ws_pred = gradient_gp.get_gp_prediction_weights(gp, pred_gp, X_pred)

            print(ws_pred @ ys_gp)

            ipdb.set_trace()












if __name__ == '__main__':

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
            'sampler_n_trajectories': 32,
            'sampler_dt': 1/16,
            'sampler_n_iters': 8,
            'sampler_n_extrarounds': 2,
            'sampler_strategy': 'importance',
            'sampler_deterministic': False,
            'sampler_plot': False,  # plotting takes like 1000x longer than the computation
            'sampler_returns': 'both',

            'x_sample_cov': x_sample_cov,

            'gp_iters': 100,
            'gp_train_plot': False,
            'active_learning_iters': 4,

    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    run_algo(problem_params, algo_params)
