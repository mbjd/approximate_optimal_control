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

from jax.config import config
# config.update("jax_debug_nans", True)

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
    sol_obj, ys = integrate(sample(subkey, algo_params['pontryagin_sampler_n_trajectories']))

    nx = problem_params['nx']

    # fit a GP to the data.
    gp_params = {
            'log_amp': np.log(1),
            'log_scale': np.zeros(nx),
    }

    xs_gp, ys_gp, grad_flags_gp = gradient_gp.reshape_for_gp(ys)

    params_known = True
    if not params_known:
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
        print(gp_params)
    else:
        gp_params = {'log_amp': 2.645, 'log_scale': np.array([0.727, 0.911])}
        gp = gradient_gp.build_gp(gp_params, xs_gp, grad_flags_gp)


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
        Σ_half_inv = jax.scipy.linalg.sqrtm(np.linalg.inv(sampling_cov)).real
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

        # calculate initial state according to PMP
        sol_obj, init_ys = integrate(term_cond.reshape(1, nx))
        init_states = init_ys[:, 0:nx]  # will give (1, nx) 'row vector'
        return desirability_fct_x0(init_states, gp, ys_gp)

    # here we actually take into account the 'change of variables' from terminal
    # condition to x0 to correct for volume change.
    def desirability_fct_jac_corrected(term_cond_normalised, gp, ys_gp):
        '''
        https://en.wikipedia.org/wiki/Probability_density_function#Vector_to_vector

        for an invertible change of variable y=H(x), and x ~ f, we have

        y ~ g(y) = f(H^-1(y)) | det d/dz H^-1 (z) |

        but we want it the other way around: we know the density at the output and
        would like to know a density for the input from which we can sample.
        so, can we just multiply by the inverse of that factor? i.e.:
        for an invertible change of variable y=H(x), and y ~ g, do we have:

        x ~ f(x) = g(H(x)) | det d/dx H(x) |

        also, is the det of the inverse mapping the inverse det of the mapping?
        -> probably yes, by implicit function theorem the jacobian of the inverse
           the inverse of the jacobian, and det(A^-1) = det(A)^-1.

        let's try it and worry about the proof later. the mapping is not necessarily
        invertible anyways so theory goes out of the window...

        here we basically replace:
         H -> tc_norm_to_x0
         g -> desirability_fct_x0
         f -> (what we are trying to find out)

        ooops, forgot an important thing, the desirability fct corresponds to a log pdf,
        not a pdf!!! so, instead we need:

        exp(desirability(xT)) = exp(desirability(x0(xT))) * | det d/dxT x0(xT) |

        putting a log on both sides:

        log(exp(desirability(xT))) = log(exp(desirability(x0(xT))) *     | det d/dxT x0(xT) |)
                desirability(xT)   =         desirability(x0(xT))  + log | det d/dxT x0(xT) |

        somehow it doesn't really work yet? seems to produce exactly the same function as
        the uncorrected version but shifted by a constant amount. but log(abs(det(jac)))
        should not be constant over the state space....?
        '''

        def tc_norm_to_x0(tc_norm):
            tc = unnormalise_term_cond(term_cond_normalised)
            sol_obj, init_ys = integrate(tc.reshape(1, nx))
            x0 = init_ys[:, 0:nx]  # will give (1, nx) 'row vector'
            # we reshape to (nx,) to make jacobian nice, but later reshape to (1, nx) again
            return x0.reshape((nx,))


        term_cond = unnormalise_term_cond(term_cond_normalised)

        x0 = tc_norm_to_x0(term_cond_normalised)
        jac = jax.jacobian(tc_norm_to_x0)(term_cond_normalised)

        jac_correction_factor = np.abs(np.linalg.det(jac))

        return desirability_fct_x0(x0.reshape(1, nx), gp, ys_gp) + np.log(1e-5 + jac_correction_factor)


    def desirability_fct_x0(x0, gp, ys_gp):
        # because of weirdness x0 needs to be (1, nx) shaped
        x0 = x0.reshape(1, nx)
        # find GP variance at initial state
        gradflags = np.zeros(1, dtype=np.int8) # only 1 data point here
        pred_gp = gp.condition(ys_gp, (x0, gradflags)).gp
        predictive_variance = pred_gp.variance

        # calculate penalty for leaving interesting state region
        # outer penalty function - we allow being slightly outside, we
        # would rather have no interference within the interesting region
        Σ_inv = np.linalg.inv(algo_params['x_sample_cov'])
        # add small number so the gradient at 0 is not NaN
        mahalanobis_dist = np.sqrt(1e-8 + x0 @ Σ_inv @ x0.T)

        max_dist = algo_params['x_max_mahalanobis_dist']
        state_penalty = 100 * np.maximum(0, mahalanobis_dist - max_dist)
        # state_penalty = 10 * np.log(1 + np.exp(mahalanobis_dist - max_dist))

        # we are maximising to get the most desirable terminal condition
        # large predictive variance = good
        # large state penalty = bad
        desirability = predictive_variance.reshape() - state_penalty.reshape()

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
    tcs = sample(key, algo_params['pontryagin_sampler_n_trajectories'])

    # all based on normalised terminal conditions!!
    # so be sure to scale by cov^(1/2) with functions above
    V_var_grad = jax.grad(V_var_normalised, argnums=0)
    V_var_vmap = jax.vmap(V_var_normalised, in_axes=(0, None, None))
    V_var_grad_vmap = jax.vmap(V_var_grad, in_axes=(0, None, None))

    # same but for the new desirability function.
    desirability_grad = jax.grad(desirability_fct, argnums=0)
    desirability_vmap = jax.vmap(desirability_fct, in_axes=(0, None, None))
    desirability_grad_vmap = jax.vmap(desirability_grad, in_axes=(0, None, None))


    N = algo_params['active_learning_iters']

    for i in range(N):

        sampler = True


        if sampler:
            # use sampling algorithm to find good terminal conditions

            logpdf = lambda tc_norm: desirability_fct_jac_corrected(tc_norm, gp, ys_gp)
            logpdf = lambda tc_norm: desirability_fct(tc_norm, gp, ys_gp)

            # normalised terminal conditions ~ N(0, I) :)
            init_norm_tcs = jax.random.normal(subkey, shape=(algo_params['pontryagin_sampler_n_trajectories'], nx))

            key, sampling_key = jax.random.split(key)
            # new_norm_tcs = sampling.sample_from_logpdf(logpdf, init_norm_tcs, algo_params, key=sampling_key)
            # new_norm_tcs = sampling.adam_uncertainty_sampler(logpdf, init_norm_tcs, algo_params, key=sampling_key)
            desirability = lambda x0: desirability_fct_x0(x0, gp, ys_gp)
            sampling.geometric_mala(integrate, desirability, problem_params, algo_params, key)

            ipdb.set_trace()
            new_sol_obj, new_ys = integrate(new_tcs)


        else:
            # use simpler gradient descent.

            key, subkey = jax.random.split(key)
            tcs_norm = jax.random.normal(subkey, shape=(algo_params['pontryagin_sampler_n_trajectories'], nx))

            def gradient_step(tcs_norm, lr):

                V_var_grads = V_var_grad_vmap(tcs_norm, gp, ys_gp)
                V_vars = V_var_vmap(tcs_norm, gp, ys_gp)  # to debug

                return (tcs_norm + lr * V_var_grads, V_vars)

            def gradient_step_desirability(tcs_norm, lr):

                V_var_grads = desirability_grad_vmap(tcs_norm, gp, ys_gp)
                V_vars = desirability_vmap(tcs_norm, gp, ys_gp)  # to debug

                return (tcs_norm + lr * V_var_grads, V_vars)

            lrs = np.logspace(-1, -2, 50)  # decreasing step size?
            print('finding uncertain points...')
            tcs_norm, V_vars = jax.lax.scan(gradient_step_desirability, tcs_norm, lrs)

            new_tcs = jax.vmap(unnormalise_term_cond)(tcs_norm)
            # did this already in V_var_grad_vmap but messy to keep solution
            print('integrating PMP again...')
            new_sol_obj, new_ys = integrate(new_tcs)





        cmap = matplotlib.colormaps.get_cmap('viridis')
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

    def plot_gp_uncertainty_2d(gp, ys_gp, extent, N=50):

        x_grid = y_grid = np.linspace(-extent, extent, N)
        x_, y_ = np.meshgrid(x_grid, y_grid)
        x_pred = np.vstack((x_.flatten(), y_.flatten())).T

        grid_shape = x_.shape

        X_pred = (x_pred, np.ones(x_pred.shape[0], dtype=np.int8))

        pred_gp = gp.condition(ys_gp, X_pred).gp
        y_pred = pred_gp.loc.reshape(grid_shape)
        y_std = np.sqrt(pred_gp.variance.reshape(grid_shape))

        pl.pcolor(x_, y_, y_std)

    def plot_desirability_2d(gp, ys_gp, extent, N=50):

        x_grid = y_grid = np.linspace(-extent, extent, N)
        x_, y_ = np.meshgrid(x_grid, y_grid)
        x_pred = np.vstack((x_.flatten(), y_.flatten())).T

        grid_shape = x_.shape

        X_pred = (x_pred, np.ones(x_pred.shape[0], dtype=np.int8))

        desirabilities = jax.vmap(desirability_fct_x0, in_axes=(0, None, None))(x_pred[:, None, :], gp, ys_gp)

        pl.pcolor(x_, y_, desirabilities.reshape(grid_shape))


    pl.figure('GP uncertainty and initial state desirability')
    pl.subplot(211)
    plot_gp_uncertainty_2d(gp, ys_gp, 4)

    # also plot an ellipse showing the interesting state distribution.
    thetas = np.linspace(0, 2*np.pi, 201)
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    unitcircle = np.row_stack([xs, ys])
    Σ_half = jax.scipy.linalg.sqrtm(algo_params['x_sample_cov']).real
    max_dist = algo_params['x_max_mahalanobis_dist']
    scaled_circle = Σ_half @ unitcircle * max_dist
    pl.plot(scaled_circle[0, :], scaled_circle[1, :])

    pl.subplot(212)
    plot_desirability_2d(gp, ys_gp, 4)
    pl.plot(scaled_circle[0, :], scaled_circle[1, :])
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
            k = algo_params['pontryagin_sampler_n_trajectories']
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
            'pontryagin_sampler_n_trajectories': 32,
            'pontryagin_sampler_dt': 1/16,
            'pontryagin_sampler_n_iters': 8,
            'pontryagin_sampler_n_extrarounds': 2,
            'pontryagin_sampler_strategy': 'importance',
            'pontryagin_sampler_deterministic': False,
            'pontryagin_sampler_plot': False,  # plotting takes like 1000x longer than the computation
            'pontryagin_sampler_returns': 'both',

            'sampler_dt': 0.05,  # this dt is basically a gradient descent stepsize
            'sampler_burn_in': 128,
            'sampler_burn_in_noise': 1,
            'sampler_init_noise': 1,
            'sampler_final_noise': 0.0001,
            'sampler_N_chains': 32,
            'sampler_samples': 16,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 16,
            'sampler_plot': True,

            'x_sample_cov': x_sample_cov,
            'x_max_mahalanobis_dist': 2,

            'gp_iters': 100,
            'gp_train_plot': False,
            'active_learning_iters': 4,

    }

    # problem_params are parameters of the problem itself
    # algo_params contains the 'implementation details'

    run_algo(problem_params, algo_params, key=jax.random.PRNGKey(121))
