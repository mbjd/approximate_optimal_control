#!/usr/bin/env python

import jax
import jax.numpy as np
import diffrax

import levelsets
import pontryagin_utils
import visualiser
from misc import *

import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf

import ipdb
import time
import numpy as onp
import tqdm
import argparse
from operator import itemgetter

# from jax import config
# config.update("jax_enable_x64", True)



def lqr_sanitycheck(problem_params, algo_params):

    # is LQR as optimal as it says?
    # let LQR value function be x.T P_lqr x.
    # then costate = Vx = 2 * P_lqr x.

    def linear_forward_sim(x0, P_lqr):

        def forwardsim_rhs(t, state, args):
            x, v = state
            lam_x = 2 * P_lqr @ x
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](x, u), -problem_params['l'](x, u)

        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol']/10, atol=algo_params['pontryagin_solver_atol']/10)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        # simulate for pretty damn long
        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=60., dt0=0.1, y0=(x0, 0.),
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps']*8,
        )

        return forward_sol



    # find terminal LQR controller and value function.
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)
    V_lqr = lambda x: x.T @ P_lqr @ x

    x0 = 0.1 * np.ones(6)

    sol = linear_forward_sim(x0, P_lqr)


    pl.plot(sol.ts, sol.ys[1] - sol.ys[1].min(), label='experienced cost-to-go')
    pl.plot(sol.ts, jax.vmap(V_lqr)(sol.ys[0]), label='LQR value fct.')
    pl.legend()

    pl.figure()

    V_lqr = lambda x: x.T @ (0.5 * P_lqr) @ x

    x0 = 0.1 * np.ones(6)

    sol = linear_forward_sim(x0, 0.5 * P_lqr)


    pl.plot(sol.ts, sol.ys[1] - sol.ys[1].min(), label='experienced cost-to-go, P/2')
    pl.plot(sol.ts, jax.vmap(V_lqr)(sol.ys[0]), label='LQR value fct., P/2')
    pl.legend()

    # therefore, we have LQR value = 0.5 x.T P_lqr x definitely.


    # also, second sanity check. The LQR controller is u = -Kx.
    # if we construct the map
    #     x -> u*(x, lambda(x))
    #        = u*(x, Vx(x))
    #        = u*(x, P_lqr @ x)
    # and differentiate it at the equilibrium, then we should
    # get the same control gain (up to the minus)

    # this is all using the correct (hopefully) V_lqr = 0.5 x.T P_lqr x.
    # so if this checks out we are probably good

    # we should really find a cleaner/more standard way to handle x_eq != 0
    lqr_controller = lambda x: pontryagin_utils.u_star_2d(x, jax.jacobian(V_lqr)(x-problem_params['x_eq']), problem_params)
    lqr_controller_manual = lambda x: pontryagin_utils.u_star_2d(x, P_lqr @ (x-problem_params['x_eq']), problem_params)

    K = jax.jacobian(lqr_controller)(problem_params['x_eq'])
    K_manual = jax.jacobian(lqr_controller_manual)(problem_params['x_eq'])

    print(f'relative norm diff -LQR controller and linearised u*(x, Vx(x)) (jax jacobian): {rnd(-K_lqr, K)}')
    print(f'relative norm diff -LQR controller and linearised u*(x, P_lqr x) (manual jacobian):  {rnd(-K_lqr, K_manual)}')
    pl.show()




def manifold_testing(problem_params, algo_params):

    # first, adapt this function to work at all.
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)
    # okay, it works :) returns K and P defined for full state space but only
    # calculates them in tangent space.

    # sanity checked (in pdb): u* stays the same if we change the costate along the
    # "irrelevant" direction [0, 0, 0, 1, 0, 0, 0] (in normal space)
    # this is because when considered members of T*x M they are the same.

    # test whether (forward) baumgarte stabilisation works.
    # seems that it does :)
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

    # make random initial states (which are in the state manifold)
    x0s = problem_params['x_eq'][None, :] + jax.random.normal(jax.random.PRNGKey(0), shape=(100, 7))*.1
    x0s = jax.vmap(problem_params['project_M'])(x0s)

    sols = jax.vmap(forward_sim_lqr)(x0s)

    pl.plot(sols.ts.flatten(), jax.vmap(problem_params['m'])(sols.ys.reshape(-1, 7)), alpha=.2, label='no baumgarte stabilisation')

    def forward_sim_lqr_baumgarte(x0):

        def forwardsim_rhs(t, x, args):

            lam_x = P_lqr @ x  # <- for lqr instead
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            xdot = problem_params['f'](x, u)

            # make a stabilisation term orthogonal to the dynamics, meaning: in the direction of the normal space.

            m_eval = problem_params['m'](x)

            # this is an "outward" normal -- it points to the outside of the unit circle.
            # might as well just choose (sinPhi, cosPhi)...
            # maybe we can not worry about inward/outward if instead we define a potential
            # like ||m(x)||^2 and then travel down its gradient?
            normal_dir = jax.jacobian(problem_params['m'])(x)

            # it could be that the extension to the ambient space already
            # has stable/unstable behaviour making it harder to choose
            # baumgarte time constant.

            # in that case, the proper way would consist not in ADDING a
            # stabilisation term in normal direction, but by REPLACING the
            # whole rhs in that direction with something stable. Two main
            # ways of doing so:
            # - take output of vector field, project to tangent space.
            # - project state to manifold, evaluate vector field there. ie.
            #   modify vector field f_p(z) = f(project_M(z)).
            # both of these should result in the rhs being "parallel" to
            # the manifold even if we strayed off of it. resulting in nice
            # marginally stable behaviour, from there we can change it with
            # baumgarte.

            # let's postpone this for later and just keep a close eye on the plots of m(x(t)).
            baumgarte_stab_term = -1 * normal_dir * m_eval

            return xdot + baumgarte_stab_term


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

    sols = jax.vmap(forward_sim_lqr_baumgarte)(x0s)

    pl.plot(sols.ts.flatten(), jax.vmap(problem_params['m'])(sols.ys.reshape(-1, 7)), alpha=.2, label='with baumgarte stabilisation')


    pl.legend()




    # next step: backward shooting with PMP.
    # first, naively using the same function as before. should work for short times.

    solve_backward, f_extended = pontryagin_utils.define_backward_solver(
        problem_params, algo_params
    )

    def solve_backward_lqr(x_f, algo_params):

        # P_lqr = hessian of value fct.
        # everything else follows from usual differentiation rules.

        # this too becomes kind of hairy in the manifold case. the LQR value function
        # is naturally defined ONLY on the tangent space at x_eq. if we naively use
        # the formula here to extend it to the manifold and whole ambient space,
        # we are just defining the value function everywhere to be:
        #  projection \Delta x (x = x_eq + \Delta x) to tangent space
        #  evaluation of tangent space value function at that point
        # which seems reasonable for small, "linearisable" regions around x_eq and seems
        # to produce quite precisely the same trajectories as the old local coordinates approach.
        # so we are happy and can keep this code verbatim in the main version :)

        v_f = 0.5 * x_f.T @ P_lqr @ x_f
        vx_f = P_lqr @ x_f

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        # no vxx here.

        return solve_backward(state_f)


    xfs = problem_params['x_eq'][None, :] + jax.random.normal(jax.random.PRNGKey(0), shape=(3000, 7))*.0001
    xfs = jax.vmap(problem_params['project_M'])(xfs)

    sols_backward = jax.vmap(solve_backward_lqr, in_axes=(0, None))(xfs, algo_params)




    # do the same with the old version (local coordinates instead of R^n embedding.)
    # hopefully sols will be the same...
    old_problem_params, old_algo_params = old_params()

    solve_backward_old, f_extended_old = pontryagin_utils.define_backward_solver(
        old_problem_params, old_algo_params
    )

    K_lqr_old, P_lqr_old = pontryagin_utils.get_terminal_lqr(old_problem_params)

    def solve_backward_lqr_old(x_f, algo_params):

        # P_lqr = hessian of value fct.
        # everything else follows from usual differentiation rules.

        v_f = 0.5 * x_f.T @ P_lqr_old @ x_f
        vx_f = P_lqr_old @ x_f

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f,
        }

        # no vxx here.

        return solve_backward_old(state_f)

    # transform between "old" (= local coordinates) and "new" (= embedded in R^n)
    # representation.
    def old_to_new(x):
        return np.concatenate([
            x[0:2],  # posx, posy
            np.array([np.sin(x[2]), np.cos(x[2])]), # angle embedding
            x[3:]
        ])

    def new_to_old(x):
        # verified experimentally: thetas = np.arctan2(np.sin(thetas), np.cos(thetas))
        # because old_to_new is not globally invertible this inverts its restriction on
        # the domain -pi/2 < theata < pi/2 or something like that.
        return np.concatenate([
            x[0:2],  # posx, posy
            np.array([np.arctan2(x[2], x[3])]),
            x[4:]
        ])

    xfs_old = jax.vmap(new_to_old)(xfs)
    xfs_oldnew = jax.vmap(old_to_new)(xfs_old)

    print(f'if {rnd(xfs, xfs_oldnew)} is very low the two transformations invert each other')
    # ipdb.set_trace()

    sols_backward_old = jax.vmap(solve_backward_lqr_old, in_axes=(0, None))(xfs_old, old_algo_params)

    # compare the two trajectories. they match :)))
    interp_ts = np.linspace(sols_backward.t0[0], sols_backward.t1[0], 500)
    for j in range(10):

        sol_new = jtm(itemgetter(j), sols_backward)
        sol_old = jtm(itemgetter(j), sols_backward_old)

        pl.figure(f'old vs new backward sol #{j}')

        pl.plot(sol_new.ts, sol_new.ys['x'], '. ', c='C0', alpha=1/2)
        pl.plot(interp_ts, jax.vmap(sol_new.evaluate)(interp_ts)['x'], c='C0', alpha=1/2)

        # old solution also transformed to embedded manifold repr.
        pl.plot(sol_old.ts, jax.vmap(old_to_new)(sol_old.ys['x']), '. ', c='C1', alpha=1/2)
        pl.plot(interp_ts, jax.vmap(old_to_new)(jax.vmap(sol_old.evaluate)(interp_ts)['x']), c='C1', alpha=1/2)

    pl.figure()
    pl.plot(sols_backward_old.stats['num_steps'], sols_backward.stats['num_steps'], '. ')
    pl.xlabel('old steps')
    pl.ylabel('new steps')

    # for the new version, look at the "unnecessary" costate.


    ms = jax.vmap(jax.vmap(problem_params['m']))(sols_backward.ys['x'])

    # costate in normal space direction.
    shit_costate = lambda vx, x: np.dot(vx, jax.jacobian(problem_params['m'])(x))
    shit_costates = jax.vmap(shit_costate)(sols_backward.ys['vx'].reshape(-1, 7), sols_backward.ys['x'].reshape(-1, 7))

    pl.figure()
    pl.subplot(211)

    # extra dof of state trajectory
    pl.plot(sols_backward.ts.reshape(-1), ms.reshape(-1), c='C0', alpha=1/3, label='m(x(t))')
    pl.legend()

    pl.subplot(212)
    pl.plot(sols_backward.ts.reshape(-1), shit_costates, alpha=1/3, label='costate in normal direction')
    pl.legend()

    # new experiment: change terminal unnecessary costate and see what happens.
    xfs_new = xfs[0:512].at[:].set(xfs[0])
    # bypass the solve_backward_lqr function and construct vmapped dict state ourselves.

    tfs_new = np.zeros(512)
    vfs_new = np.ones(512) * (0.5 * xfs_new[0].T @ P_lqr @ xfs_new[0])

    # in actual normal direction
    vx_orig = P_lqr @ xfs[0]
    delta = jax.jacobian(problem_params['m'])(xfs[0]) * 100
    vxs_new = np.linspace(vx_orig - delta, vx_orig + delta, 512)

    states_f = {
        'x': xfs_new,
        't': tfs_new,
        'v': vfs_new,
        'vx': vxs_new,
    }

    sols_test = jax.vmap(solve_backward)(states_f)

    pl.figure('normal direction costate perturbation in precise normal direction')
    shit_costates = jax.vmap(shit_costate)(sols_test.ys['vx'].reshape(-1, 7), sols_test.ys['x'].reshape(-1, 7))
    pl.subplot(211)
    pl.plot(sols_test.ts.reshape(-1), sols_test.ys['x'].reshape(-1, 7), label='state trajectories')
    pl.legend()

    pl.subplot(212)
    pl.plot(sols_test.ts.reshape(-1), shit_costates, alpha=1/3, label='normal direction costate for different inits')
    pl.legend()

    '''
    import nn_utils

    v_nn = nn_utils.nn_wrapper(
        input_dim=problem_params['nx'],
        layer_dims=algo_params['nn_layerdims'],
        output_dim=1
    )

    key = jax.random.PRNGKey(0)
    params = v_nn.nn.init(key, np.zeros(problem_params['nx']))

    sol = jtm(itemgetter(12), sols_backward)
    y = jtm(itemgetter(12), sol.ys)

    # def sobolev_loss(self, key, y, params, problem_params, algo_params):
    loss = v_nn.sobolev_loss(key, y, params, problem_params, algo_params)
    '''

    pl.show()
    ipdb.set_trace()





def define_problem_params():

    # classic 2D quad type thing. 6D` state.
    # update, 6D manifold embedded in R^7.

    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

    # remove time arguments sometime now that we're mostly treating
    # infinite horizon, time-invariant problems?
    def f(x, u):

        # unpack for easier names
        Fl, Fr = u
        posx, posy, sinPhi, cosPhi, vx, vy, omega = x

        # Phi' = omega
        # d/dt sin(Phi) = cos(Phi) Phi' = cosPhi omega
        # d/dt cos(Phi) = -sin(Phi) Phi' = -sinPhi omega

        xdot = np.array([
            vx,
            vy,
            cosPhi * omega,
            -sinPhi * omega,
            -sinPhi * (Fl + Fr) / m,
            cosPhi * (Fl + Fr) / m - g,
            (Fr-Fl) * r / I,
        ])

        return xdot

    x_eq = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)

    def l(x, u):
        Fl, Fr = u
        posx, posy, sin_Phi, cos_Phi, vx, vy, omega = x

        # penalise deviation from cos(Phi)=1, sin(Phi)=0 just in cartesian ambient space
        # derivatives should be the same still (bc sin'(0) = 1)

        state_length_scales = np.array([0.3, 0.3, np.deg2rad(30), np.deg2rad(30), .5, .5, np.deg2rad(120)])
        Q = np.diag(1/state_length_scales**2)
        state_cost = (x - x_eq).T @ Q @ (x - x_eq)

        # can we just set an input penalty that is zero at hover?
        # penalise x acc, y acc [m/s^2], angular acc [rad/s^2] here
        # this here is basically a state-dependent linear map of the inputs, i.e. M(x) u with M(x) a 3x2 matrix.
        # the overall input cost will be acc.T M(x).T Q M(x) acc, so for each state it is still a nice quadratic in u.
        accelerations = np.array([
            -sin_Phi * (Fl + Fr) / m,
            cos_Phi * (Fl + Fr) / m - g,
            (Fr - Fl) * r / I,
        ])

        accelerations_lengthscale = np.array([1, 1, 1])

        input_cost = accelerations.T @ np.diag(1/accelerations_lengthscale**2) @ accelerations

        return state_cost + input_cost


    problem_params = {

        'system_name': 'flatquad',

        # dynamics X x U -> TxX, stage cost X x U -> R
        'f': f,
        'l': l,

        # state & input space dimensions
        # if manifold, the dimension of the ambient space, not the manifold!
        'nx': 7,
        'nu': 2,

        'state_names': ("x", "y", "sinPhi", "cosPhi", "vx", "vy", "omega"),

        'u_eq': np.ones(2) * m * g / 2,
        'x_eq': x_eq,


        # if ever treating slightly bigger systems it would pay to frame this
        # as a general convex polytope described by Ax <= b.
        'U_interval': [np.zeros(2), umax*np.ones(2)],

        # the value level below which we accept the LQR solution as correct.
        'V_f': 0.001,
        'V_max': 1500.,

        # constraint equation defining the state space manifold as its 0-levelset.
        # if R^n, set this to None
        # number of constraint equations = codimension of manifold.
        # atm only codimension 1 is supported, because this makes finding
        # an orthonormal basis for the normal space trivial.

        # in this case only the unit circle for angle parameterisation.
        # / 2 so its jacobian is normalised.
        'm': lambda x: (x[2]**2 + x[3]**2 - 1) / 2,

        # projection operation onto the manifold -- great for resetting if
        # we stray off the manifold due to numerical errors.
        'project_M': lambda x: x.at[2:4].set(x[2:4] / np.linalg.norm(x[2:4])),

        'x_extent': np.array([
            20,  20,  # x and y, [m]
            1., 1.,   # sinPhi and cosPhi [1] (but irrelevant -- see sampling fct)
            20,  20,  # vx and vy, [m/s]
            20*np.pi  # omega [rad/s]
        ]),


    }

    return problem_params



def base_algo_params():

    algo_params = {

        # PRNG seed
        'seed': 32,


        # ODE SOLVER PARAMS
        'pontryagin_solver_atol': 1e-4,
        'pontryagin_solver_rtol': 1e-4,
        'dtmin': 0.01,
        'dtmax': 0.5,

        # project back to manifold after each solver step. only possible if
        # problem_params['project_M'] correctly defined.
        'project_manifold': True,

        # with throw=True we can set this pretty tight - it will just stop early.
        # will have to make sure ourselves that this is not a problem
        'pontryagin_solver_maxsteps': 128,

        # not very relevant if we can just "resume" the trajectory in a later solve
        # also maybe it makes sense to stop based on value, like stop after we reach sth like 10x
        # the current value level? then we pervent spending lots of effort in "difficult" (=high l(x, u))
        # state space regions.
        'pontryagin_solver_T': 5.,

        # (this was not used for a long time)
        # in theory ||vxx|| can become infinite - meaning we solve an ODE with finite escape time.
        # this happenn when many optimal trajectories originate from a small region (or a point in the limit)
        # to avoid this we just stop calculating the trajectory once ||vxx|| exceeds this bound.
        # hopefully the state space will still be sufficiently covered. In regions where ||vxx|| would
        # have been very high we will just have to accept the interpolation instead.
        'pontryagin_solver_vxx': False,
        'vxx_max_norm': 1e4,

        # causes it not to quit when hitting maxsteps. probably still all subsequent
        # results will be unusable due to evaluating solutions outside their domain giving NaN
        'throw': False,



        # NN ARCHITECTURE & TRAINING
        'nn_type': 'leaky',
        # 'nn_layerdims': (16, 16, 16),
        # 'nn_layerdims': (32, 32, 32),
        # 'nn_layerdims': (64, 64, 64),
        # 'nn_layerdims': (128, 128, 128),
        # 'nn_layerdims': (256, 256, 256),
        # 'nn_layerdims': (512, 512, 512),
        # 'nn_layerdims': (32, 32, 32, 32),
        # 'nn_layerdims': (128, 8),
        # nicer for the launch script
        'nn_n_layers': 3,
        'nn_layer_dim': 256,
        'nn_batchsize': 32,
        'nn_N_epochs': 512,
        'nn_train_fraction': .98,

        'lr_staircase': False,
        'lr_staircase_steps': 8,
        'lr_init': 0.05,
        'lr_final': 0.0001,
        'weight_decay': .003,

        'nn_ensemble_size': 4,

        'nn_warm_start': True,
        'nn_warmstart_fraction': 1/4,

        'nn_value_sweep': True,

        'nn_progressbar': True,

        # NN LOSS FUNCTION
        # relative importance of the losses for v, vx, vxx.
        # mostly we care about representing vx with great accuracy,
        # the other two can be thought of as "hints"/priors/inductive biases
        # to fit the correct vx function.
        # update: vxx not used anymore, leave it at 0 or update lots of code
        'nn_sobolev_weight_v': 1.,
        'nn_sobolev_weight_vx': 10.,
        'nn_sobolev_weight_vxx': 0.,

        # width of the quadratic regions in smoothed huber loss.
        'vx_loss_d': 0.3,
        'v_loss_d': 0.1,

        # above those thresholds relative loss is used
        'min_important_v': 0.1,
        'min_important_vx': 0.1,

        # penalisation of the extra value derivative which is defined in the ambient space
        # but normal to the state manifold.
        'vx_normal_regularisation': 0.001,

        # this is not a proper "prior" in the bayesian sense, but rather
        # just an additional weak loss term that makes the value function
        # large-ish at the problematic state of being upside down but
        # otherwise at equilibrium.
        'prior_strength': 0.01,
        'v_prior': 1.,

        'inv_vx_loss_fadeout': 5.,

        # MAIN ALGO
        # only take a subsample of data for active learning. dense sample
        # close to current level set, less dense sample further down.

        # the uncertainty bound we wish to satisfy.
        # sigma_max(mu) = simga_max_abs + simga_max_rel * mu
        'sigma_max_abs': 0.5,
        'sigma_max_rel': 0.05,

        # value band for training = [v_k / thin_data_denominator, v_next_target]
        'thin_data': True,
        'thin_data_denominator': 5,

        # initial data generation. 'uniform' or 'lqr' for nicer distribution.
        'initial_shooting': 'lqr',
        # the value level we include in the initial learning round.
        'v_init': 50,

        # number of proposals per active learning iteration.
        # larger = nicer! but don't kill our poor RAM
        'initial_batchsize': 128,
        'active_learning_batchsize': 128,
        'include_future_data': False,

        # the max. time horizon by which we aim to grow the known level set
        # in one iteration.
        'T_value_target': 1.,

        'vk_estimator': 'k_exceptions',

        'pruning_strategy': 'conservative',

        'proposal_sampling_distribution': 'uniform',
        'proposal_strategy': 'max_kernel_adaptive',
        'proposal_kernel_scaling': .5,

        # the sublevel set Vk must contain at least this fraction of test points
        # which are below the sigma target to qualify as "learned".
        # only applies for 'vk_estimator' == 'relaxed'.
        'frac_certain_in_Vk': .99,


        # OUTPUT & VISUALISATION
        'wandb': False,

        'savefigs': True,
        'wandbfigs': False,
        'showfigs': True,

        'ipdb_interval': 8,
    }

    def sample_states_batched(key, N, extent, log_min_scale=0):

        # vmapped version of the above, but also scales down half the
        # points with logspace'd distribution.

        # maybe the "scaling" should also be part of the sampling fct?
        # stochastic not determinstic? probably only cosmetic though

        keys = jax.random.split(key, N)

        # if log_min_scale != 0, this will scale half the points down with a
        # logarithmically scaled factor, while the other half will stay the same.
        log_min_scale = -np.abs(log_min_scale)
        scales = np.clip(np.logspace(log_min_scale, -log_min_scale, N), -np.inf, 1.)[:, None]

        pts = jax.vmap(sample_state, in_axes=(0, None, 0))(keys, extent, scales)

        return pts

    def sample_state(key, extent, scale=1.):

        # sample points "uniformly" from "the whole state space".
        # problem specific function! thus outside in problem_params.

        # key: usual PRNG key
        # extent: np.array of shape (nx,). [-extent, extent] are the box
        #   bounds for uniform sampling. Manifold states cosPhi, sinPhi
        #   treated separately so their "extent" is irrelevant.
        # scale: scales the sample by some scalar.

        # TODO think about what happens when x_eq != 0 -- just add it here?

        # maybe (especially for higher dims) ellipsids are better? a bit like this:
        # - sample from unit normal
        # - transform magnitude of samples such that they are uniform within unit ball
        #   (inverse transform normcdf chi squared something, i think I did this once)
        # - squash with matrix A to transform to ellipse {z: || z.T inv(A).T inv(A) z || <= 1 }
        # - sample from different scaled versions of this ellipse to avoid the soap bubble effect :)

        # but this is just an intuitive hunch, because for a uniform box most
        # of the volume is at the corners, where we might not want it. also
        # these effects probably don't really kick in at like 6 to 12 dims.


        # separate the "flat" R^n part and the manifold part.
        rnkey, manifoldkey = jax.random.split(key)

        # generate uniform points from a box in R^n
        x_pt = jax.random.uniform(
            key=rnkey,
            shape=extent.shape,
            minval=-extent,
            maxval= extent,
        ) * scale


        # for the manifold part: uniform sampling from unit circle.

        # if we instead generate this by drawing Phi ~ U[-pi, pi]
        # we can apply the same scaling logic to the angle and only
        # then convert to ambient space representation...

        # generate 2D gaussian & normalise
        xy = jax.random.normal(key=manifoldkey, shape=(2,))
        xy = xy / np.linalg.norm(xy)

        # indices of sinPhi and cosPhi states.
        assert problem_params['state_names'][2] == 'sinPhi'
        assert problem_params['state_names'][3] == 'cosPhi'
        x_pt = x_pt.at[2:4].set(xy)

        return x_pt

    # just pass on the entire functions :)
    algo_params['sample_state'] = sample_state
    algo_params['sample_states_batched'] = sample_states_batched

    return algo_params


if __name__ == '__main__':

    problem_params = define_problem_params()
    algo_params = base_algo_params()


    # argparser based on the algo_params dict.
    parser = argparse.ArgumentParser()

    arg_types = (bool, int, float, str)

    # thanks stackoverflow
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Need bool; got %r' % s)
        return {'true': True, 'false': False}[s.lower()]

    def add_boolean_argument(parser, name, default=False):
        """Add a boolean argument to an ArgumentParser instance."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            name, nargs='?', default=default, const=True, type=_str_to_bool)
        group.add_argument('--no' + name, dest=name, action='store_false')


    for k in algo_params:
        t = type(algo_params[k])
        if t in arg_types:
            if t == bool:
                add_boolean_argument(parser, f'--{k}', default=algo_params[k])
            else:
                parser.add_argument(f'--{k}', type=type(algo_params[k]), default=algo_params[k])

    commandline_args = parser.parse_args()

    # now, put the arguments back into the algo_params dict
    for k in algo_params:

        if type(algo_params[k]) in arg_types:
            new_arg = getattr(commandline_args, k)
            old_arg = algo_params[k]

            if type(new_arg) != type(old_arg):
                raise ValueError(f'argument {k} has type {type(new_arg)} but should have type {type(old_arg)}')

            algo_params[k] = new_arg


    levelsets.evaluate('euler_runs/8dgpt7uo', problem_params, algo_params)
    # levelsets.evaluate('euler_runs/ff5mij89', problem_params, algo_params)
    # levelsets.main(problem_params, algo_params)





