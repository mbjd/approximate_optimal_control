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


def define_problem_params():

    # example from idea dump. orbit-like thing where move in circles. if outside the
    # unit circle, move in one direction, inwards other. input u moves "orbit radius".
    # aim to stabilise (0, 1)

    def f(x, u):
        rotspeed = (x[0]**2 + x[1]**2 - 1).reshape(u.shape)
        # mat = np.diag([u, u]) * u + np.array([[0, -1], [1, 0]]) * rotspeed

        mat = np.array([[u, -rotspeed], [rotspeed, u]]).squeeze()

        return mat @ x  # weird pseudo linear thing


    def l(x, u):
        Q = np.eye(2)
        err = x - np.array([0, 1])
        distpenalty = err.T @ Q @ err
        rotspeed = x[0]**2 + x[1]**2 - 1
        vpenalty = (x[0]**2 + x[1]**2 - 1)**2
        inp_penalty = 10 * u**2
        return 100 * (vpenalty + 0.1 * distpenalty + inp_penalty).reshape()



    problem_params = {

        'system_name': 'orbits',

        # dynamics X x U -> TxX, stage cost X x U -> R
        'f': f,
        'l': l,

        # state & input space dimensions
        # if manifold, the dimension of the ambient space, not the manifold!
        'nx': 2,
        'nu': 1,

        'state_names': ("x", "y"),

        'u_eq': np.zeros(1),
        'x_eq': np.array([0., 1]),


        # if ever treating slightly bigger systems it would pay to frame this
        # as a general convex polytope described by Ax <= b.
        'U_interval': [-0.2, 0.2],

        # the value level below which we accept the LQR solution as correct.
        'V_f': 0.01,

        # constraint equation defining the state space manifold as its 0-levelset.
        # if R^n, set this to None
        # number of constraint equations = codimension of manifold.
        # atm only codimension 1 is supported, because this makes finding
        # an orthonormal basis for the normal space trivial.

        # in this case only the unit circle for angle parameterisation.
        # / 2 so its jacobian is normalised.
        # 'm': lambda x: (x[2]**2 + x[3]**2 - 1) / 2,
        'm': None,

        # projection operation onto the manifold -- great for resetting if
        # we stray off the manifold due to numerical errors.
        'project_M': lambda x: x,

        'x_extent': np.array([4, 4]),
    }

    return problem_params



def base_algo_params():

    algo_params = {

        # PRNG seed
        'seed': 0,

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
        'pontryagin_solver_T': 10.,

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
        # big question: should we aim for over- or underparameterisation?
        # 'nn_layerdims': (256, 16),
        'nn_type': 'leaky',
        'nn_layerdims': (16, 16, 16),
        'nn_batchsize': 32,
        'nn_N_epochs': 1024,
        'nn_train_fraction': .98,
        'lr_staircase': True,
        'lr_staircase_steps': 8,
        'lr_init': 0.01,
        'lr_final': 0.0001,
        'weight_decay': .001,
        'nn_warmstart_fraction': 1.,

        'nn_ensemble_size': 4,
        'nn_warm_start': True,

        'nn_progressbar': True,

        # NN LOSS FUNCTION
        # relative importance of the losses for v, vx, vxx.
        # mostly we care about representing vx with great accuracy,
        # the other two can be thought of as "hints"/priors/inductive biases
        # to fit the correct vx function.
        # 'nn_sobolev_weights': np.array([0.1, 1., 0.001]),
        'nn_sobolev_weights': [1., 10.],

        # width of the quadratic regions in smoothed huber loss.
        'vx_loss_d': 0.1,
        'v_loss_d': 1.,

        # penalisation of the extra value derivative which is defined in the ambient space
        # but normal to the state manifold.
        'vx_normal_regularisation': 0.001,

        # this is not a proper "prior" in the bayesian sense, but rather
        # just an additional weak loss term that makes the value function
        # large-ish at the problematic state of being upside down but
        # otherwise at equilibrium.
        'prior_strength': 0.0,
        'v_prior': 500.,

        'inv_vx_loss_fadeout': 20.,

        # MAIN ALGO
        # only take a subsample of data for active learning. dense sample
        # close to current level set, less dense sample further down.

        # the uncertainty bound we wish to satisfy.
        # sigma_max(mu) = simga_max_abs + simga_max_rel * mu
        'sigma_max_abs': 0.5,
        'sigma_max_rel': 0.05,

        # value band for training = [v_k / thin_data_denominator, v_next_target]
        'thin_data': False,
        'thin_data_denominator': 10,

        # initial data generation. 'uniform' or 'lqr' for nicer distribution.
        'initial_shooting': 'uniform',
        # the value level we include in the initial learning round.
        'v_init': 5.,

        # number of proposals per active learning iteration.
        # larger = nicer! but don't kill our poor RAM
        'initial_batchsize': 64,
        'active_learning_batchsize': 64,
        'include_future_data': True,

        # the max. time horizon by which we aim to grow the known level set
        # in one iteration.
        'T_value_target': 3.,

        'vk_estimator': 'k_exceptions',

        'proposal_sampling_distribution': 'uniform',
        'proposal_strategy': 'max_kernel',
        'pruning_strategy': 'conservative',
        'L_v': np.inf,
        'L_vx': 2000,

        'second_pruning_sigma': 5.,


        # the sublevel set Vk must contain at least this fraction of test points
        # which are below the sigma target to qualify as "learned".
        # only applies for 'vk_estimator' == 'relaxed'.
        'frac_certain_in_Vk': .99,


        # OUTPUT & VISUALISATION
        'wandb': False,

        # save figures on filesystem.
        'savefigs': False,
        # track figures with aim.
        'wandbfigs': True,
        # show figures in UI (blocking!)
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
        x_pt = problem_params['x_eq'] + jax.random.uniform(
            key=rnkey,
            shape=extent.shape,
            minval=-extent,
            maxval= extent,
        ) * scale

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

    levelsets.testbed(problem_params, algo_params)





