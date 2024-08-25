#!/usr/bin/env python
import warnings
from functools import partial

import diffrax
import ipdb
import jax
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as pl
import numpy as onp
import scipy
import tqdm
from jax import config

import pontryagin_utils
from misc import *

config.update("jax_enable_x64", True)


def f(x, u):
    # dynamics from https://arxiv.org/pdf/2312.17467
    sinPhi, cosPhi, phidot = x

    m = l = 1
    b = 0
    g = 9.81


    sinPhi_dot = cosPhi * phidot
    cosPhi_dot = -sinPhi * phidot
    phidot_dot = -1/(m * l**2) * (b * phidot - m*g*l*sinPhi - u.reshape())

    return np.array([sinPhi_dot, cosPhi_dot, phidot_dot])


def l(x, u):

    sinPhi, cosPhi, phidot = x

    q1, q2, r = 1., 1., 2.

    cost = q1 * (sinPhi**2 + (cosPhi - 1)**2) + q2 * phidot**2 + r * u**2
    return cost.reshape()

# lots of unneeded things...
problem_params = {

    'system_name': 'inv_pendulum',

    # dynamics X x U -> TxX, stage cost X x U -> R
    'f': f,
    'l': l,

    # state & input space dimensions
    # if manifold, the dimension of the ambient space, not the manifold!
    'nx': 3,
    'nu': 1,

    'state_names': ("x", "y"),

    'u_eq': np.zeros(1),
    'x_eq': np.array([0., 1., 0]),


    # if ever treating slightly bigger systems it would pay to frame this
    # as a general convex polytope described by Ax <= b.
    'U_interval': [-0.5, 0.5],

    # the value level below which we accept the LQR solution as correct.
    'V_f': 0.01,

    # constraint equation defining the state space manifold as its 0-levelset.
    # if R^n, set this to None
    # number of constraint equations = codimension of manifold.
    # atm only codimension 1 is supported, because this makes finding
    # an orthonormal basis for the normal space trivial.

    # in this case only the unit circle for angle parameterisation.
    # / 2 so its jacobian is normalised.
    'm': lambda x: (x[0]**2 + x[1]**2 - 1) / 2,
    # 'm': None,

    # projection operation onto the manifold -- great for resetting if
    # we stray off the manifold due to numerical errors.
    # 'project_M': lambda x: x.at[2:4].set(x[2:4] / np.linalg.norm(x[2:4])),
    'project_M': lambda x: x.at[0:2].set(x[0:2] / np.linalg.norm(x[0:2])),

    'x_extent': np.array([4, 4]),
}

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
    'nn_N_epochs': 256,
    'nn_train_fraction': .98,
    'lr_staircase': False,
    'lr_staircase_steps': 8,
    'lr_init': 0.05,
    'lr_final': 0.005,
    'weight_decay': .0005,
    'nn_warmstart_fraction': 1.,

    'nn_ensemble_size': 4,
    'nn_warm_start': True,

    'nn_value_sweep': False,

    'nn_progressbar': True,

    # NN LOSS FUNCTION
    # relative importance of the losses for v, vx, vxx.
    # mostly we care about representing vx with great accuracy,
    # the other two can be thought of as "hints"/priors/inductive biases
    # to fit the correct vx function.
    'nn_sobolev_weight_v': 1.,
    'nn_sobolev_weight_vx': 10.,

    # width of the quadratic regions in smoothed huber loss.
    # both in terms of relative error, i.e. 0.1 means that above an
    # error of 10% we penalise less heavily.
    'vx_loss_d': 0.3,
    'v_loss_d': 0.2,

    # penalisation of the extra value derivative which is defined in the ambient space
    # but normal to the state manifold.
    'vx_normal_regularisation': 0.001,

    # this is not a proper "prior" in the bayesian sense, but rather
    # just an additional weak loss term that makes the value function
    # large-ish at the problematic state of being upside down but
    # otherwise at equilibrium.
    'prior_strength': 0.01,
    'v_prior': 200.,

    'inv_vx_loss_fadeout': 10,

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
    'initial_shooting': 'lqr',
    # the value level we include in the initial learning round.
    'v_init': 1.,

    # number of proposals per active learning iteration.
    # larger = nicer! but don't kill our poor RAM
    'initial_batchsize': 64,
    'active_learning_batchsize': 16,
    'include_future_data': True,

    # the max. time horizon by which we aim to grow the known level set
    # in one iteration.
    'T_value_target': 2.,

    'vk_estimator': 'k_exceptions',

    'proposal_sampling_distribution': 'uniform',
    'proposal_strategy': 'max_kernel_adaptive',
    'proposal_kernel_scaling': 0.5,

    'pruning_strategy': 'conservative',
    'L_v': np.inf,
    'L_vx': 2000,

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

# magic switch for value level sets.
algo_params['reparam'] = True
algo_params['dtmin'] = 0.001  # smaller for initial stuff
algo_params['dtmax'] = 1000.  # larger for high v
algo_params['pontryagin_solver_atol'] = 1e-5
algo_params['pontryagin_solver_rtol'] = 1e-5
algo_params['pontryagin_solver_maxsteps'] = 128

solve_backward, f_extended = pontryagin_utils.define_backward_solver(problem_params, algo_params)

u_star = pontryagin_utils.u_star_general(np.zeros(3), np.zeros(3), problem_params)

K_lqr, P_lqr, P_tangent = pontryagin_utils.get_terminal_lqr(
        problem_params, return_tangent_projection=True
)

P_lqr_tangent = P_tangent @ P_lqr @ P_tangent.T


eq = problem_params['x_eq']
V_f = lambda x: 0.5 * (x - eq).T @ P_lqr @ (x - eq)

thetas = np.linspace(0, 2 * np.pi, 2048)[:-1]
circle_xs = jax.vmap(lambda theta: np.array([np.sin(theta), np.cos(theta)]))(thetas)
xfs = (0.1 * circle_xs @ np.linalg.inv(scipy.linalg.sqrtm(P_lqr_tangent))) @ P_tangent + problem_params['x_eq'][None, :]
yfs = jax.vmap(lambda xf: dict(x=problem_params['project_M'](xf), v=V_f(xf), vx=jax.grad(V_f)(xf), t=0.))(xfs)


yf = jtm(itemgetter(0), yfs)
# ipdb.set_trace()
vf = yf['v']

# v_upper = 1000.

@jax.jit
def remesh(sols, frac):

    # the magic sauce.
    # if necessary: find here by bisection the largest v such that some
    # maximum distance between points is not exceeded.
    yfs = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)

    # yfs['x'] represents a closed curve in state space. consider it equal to
    # its piecewise linear interpolation for now. we want something like the
    # arclength parametrization, to redistribute the points equidistantly. how?

    # first get the distances between all the neighbors.
    xs = yfs['x']
    rolled_xs = np.roll(xs, -1, axis=0)
    lifted_arclen=True
    if lifted_arclen:
        # compute arclentghs in full (x, λ) space.
        # but normalise gradient. only care about direction.
        lams = yfs['vx'] / np.linalg.norm(yfs['vx'], axis=1)[:, None]
        rolled_lams = np.roll(lams, -1, axis=0)
        neighbor_dists = np.linalg.norm(np.hstack([xs, lams]) - np.hstack([rolled_xs, rolled_lams]), axis=1)
    else:
        neighbor_dists = np.linalg.norm(xs - rolled_xs, axis=1)
    arclengths = np.cumsum(neighbor_dists)

    # now, find a new set of points that are equidistant in arclength.
    # considering arclengths as a function of the index, we see it is monotonously increasing and thus invertible.
    # thus we can also view index as a function of arclength! and use standard numpy interp thing.
    N = yfs['x'].shape[0]
    arclengths_even = np.linspace(0, arclengths[-1], N)
    frac_idx = np.interp(arclengths_even, arclengths, np.arange(N))

    # now, we can use this fractional index to interpolate the new points.
    # maybe this could have been done in a single step???

    # instead use the previous ys for that interpolation.
    # t = (1-frac) * sols.t0[0] + (frac) * sols.t1[0]
    nx = yfs['x'].shape[1]
    ys_remesh = jax.vmap(lambda sol: sol.evaluate(sol.t0))(sols)
    new_v = np.interp(frac_idx, np.arange(N), ys_remesh['v'])
    new_x = np.array([np.interp(frac_idx, np.arange(N), ys_remesh['x'][:, i]) for i in range(nx)]).T
    new_vx = np.array([np.interp(frac_idx, np.arange(N), ys_remesh['vx'][:, i]) for i in range(nx)]).T
    new_t = np.interp(frac_idx, np.arange(N), ys_remesh['t'])

    new_y = dict(x=new_x, v=new_v, vx=new_vx, t=new_t)
    return new_y

solve_fast = jax.jit(jax.vmap(solve_backward, in_axes=(0, None)))
# solve_fast = jax.vmap(solve_backward, in_axes=(0, None))

vmax = 15
N=20
levels = np.logspace(0., np.log10(vmax), N)
levels = np.linspace(np.sqrt(2*vf), np.sqrt(vmax), N)**2
# levels = np.linspace(1, vmax, N)

sols = None
transform_plot = lambda x: np.concatenate([np.array([np.arctan2(x[0], x[1])]), x[2:]])

threed = False
if threed:
    ax = pl.figure().add_subplot(projection='3d')
else:
    # add nan to angles near +- pi to make plot nicer despite glued
    # together boundaries
    transform_plot = lambda x: np.concatenate([np.array([np.arctan2(x[0], x[1])]), x[2:]]) + np.nan * (x[1] < -0.95)

for v_upper in tqdm.tqdm(levels):

    # alright so it has to work a bit differently.
    # 1. get solutions starting at uniformly spaced points on dVk
    # 2. remesh them to be equidistant at dVk+1
    # 3. get solutions again.

    # 1. uniform solutions.
    sols_uniform = solve_fast(yfs, v_upper)
    # 2. remeshing
    yfs = remesh(sols_uniform, 0.0)
    # 3. remeshed solutions.
    sols = solve_fast(yfs, v_upper)

    # pl.plot(*sols_uniform.ys['x'].reshape(-1,2).T, alpha=.3, label='uniform')
    # pl.plot(*jax.vmap(transform_plot)(sols.ys['x']).reshape(-1,2).T, alpha=.3, label='remeshed')
    # pl.legend()
    viridis = matplotlib.colormaps['viridis']
    c = viridis(v_upper / levels[-1])
    if threed:
        pl.plot(*sols.ys['x'][::10,:,:].reshape(-1, 3).T, c=c, alpha=.1, label='remeshed')
    else:
        pl.plot(*jax.vmap(transform_plot)(sols.ys['x'][::10,:,:].reshape(-1, 3)).T, c=c, alpha=.1, label='remeshed')

    # yprev = yfs
    yfs = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)

    viridis = matplotlib.colormaps['viridis']
    if threed:
        pl.plot(*yfs['x'].T, '-', alpha=.5, color = c )
    else:
        pl.plot(*jax.vmap(transform_plot)(yfs['x']).T, '-', alpha=.5, color = c )

    # pl.plot(*sols.ys['x'].reshape(-1, 2).T, color='black', alpha=.1 )


pl.show()
ipdb.set_trace()


def find_collision_continuation(sols, v_upper):

    # approach it the opposite way, with trajectories.
    # assume all "collisions" happen in the value slice covered by sols.
    pl.plot(*sols.ys['x'][0])
    pass


def find_self_intersection(xs):

    # given a closed curve in 2d space, return all points where it intersects itself.
    # absolutely brute force. no apologies.

    # first, we want all pairs of neighboring points.
    first_idx = np.arange(xs.shape[0])
    second_idx = np.roll(first_idx, -1)

    # we want to know if the line segment between

    lfirst = xs[first_idx]
    lsecond = xs[second_idx]

    rfirst = xs[first_idx]
    rsecond = xs[second_idx]

    # for each index pari (i, j), we want to know if the line segment
    # between lfirst and lsecond intersects the one between rfirst and rsecond.

    # that is, concretely:
    #  1. find a, b such that: lfirst + a * (lsecond - lfirst) = rfirst + b * (rsecond - rfirst)
    #  2. check if a and b are between 0 and 1 - if so, we have an intersection.

    # first step, for single line pair.
    # lfirst + a * (lsecond - lfirst) = rfirst + b * (rsecond - rfirst)
    # a * (lsecond - lfirst) - b * (rsecond - rfirst)= -lfirst + rfirst
    # [lsecond-lfirst, rsecond-rfirst]  [a; b] = -lfirst + rfirst
    # [ldir, rdir] [a; b] = -lfirst + rfirst
    def single_intersection(lfirst, lsecond, rfirst, rsecond):
        A = np.array([lsecond - lfirst, rsecond - rfirst]).T
        b = -lfirst + rfirst
        ab = np.linalg.solve(A, b)
        return ab

    # so for each index (i, j), we need:
    # i, j = 1, 2
    # single_intersection(lfirst[i], lsecond[i], rfirst[j], rsecond[j])

    # now use vmap to do this for all pairs.
    all_abs = jax.vmap(jax.vmap(single_intersection, in_axes=(None, None, 0, 0)), in_axes=(0, 0, None, None))(lfirst, lsecond, rfirst, rsecond)

    is_inside = ((all_abs > 0.) & (all_abs < 1.)).all(axis=2)
    all_intersection_pts = lfirst + all_abs[0][:, None] * (lsecond - lfirst)
    ipdb.set_trace()


xs = yfs['x']
find_self_intersection(xs)


print('')


# pl.figure()

# make basically the same plot, but with the data transposed, so we plot value level sets
# instead of trajectories.
# ax = pl.figure().add_subplot(projection='3d')
# for each value level set:
for vlevel in tqdm.tqdm(range(all_vs.shape[1])):
    try:
        vvec = all_vs[:, vlevel]
        x0vec = all_ys[:, vlevel, 0]
        x1vec = all_ys[:, vlevel, 1]

        # pl.plot(x0vec, x1vec, color=cmap(vvec[0]/v1), alpha=v_alpha)

        ax2d.plot(x0vec, x1vec, color=cmap(vvec[0]/v1), alpha=v_alpha)
        ax.plot(x0vec, x1vec, vvec, color=cmap(vvec[0]/v1), alpha=v_alpha)
    except:
        # sometimes the last entries are NaN. Don't care
        pass

    # ipdb.set_trace()
    # pl.savefig(f'animation_figs/orbits_{vlevel:05d}.png', dpi=400)

thetas = np.linspace(0, 2*np.pi, 501)
ax2d.plot(np.sin(thetas), np.cos(thetas), color='black')
ax.plot(np.sin(thetas), np.cos(thetas), 0 * thetas, color='black')
ax2d.scatter([0], [1], [0], color='black')

def intersection(x1,x2,x3,x4,y1,y2,y3,y4):
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if d:
        xs = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / d
        ys = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / d
        if (xs >= min(x1,x2) and xs <= max(x1,x2) and
            xs >= min(x3,x4) and xs <= max(x3,x4)):
            return xs, ys

# this is, expectedly, slow as shit.
# make jitted version somehow or ignore completely?
find_intersections = False
if find_intersections:
    # find the points where each value curve self-intersects, to plot
    # the decision boundary between going left or right.

    # first only for maximum vlevel.
    vlevel = 101
    ntrajs = all_ys.shape[0]

    # iterate over all pairs of lines. very brute force :/
    for i, line_a in tqdm.tqdm(enumerate(all_ys)):
        # only j > i bc symmetry
        for j, line_b in enumerate(all_ys[i+1:]):
            xi, yi = all_ys[i, vlevel, 0:2]
            xip, yip = all_ys[(i+1) % ntrajs, vlevel, 0:2]

            xj, yj = all_ys[j, vlevel, 0:2]
            xjp, yjp = all_ys[(j+1) % ntrajs, vlevel, 0:2]

            out = intersection(xi, xip, xj, xjp, yi, yip, yj, yjp)
            # x[i],x[i+1],x[j],x[j+1],y[i],y[i+1],y[j],y[j+1]

            if out is not None:
                print(out)


# # bit less dense plot for writeup
# pl.figure()
# for idx, name in zip([10, 20, 30, 40, 50, 80, 90], ['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_k', 'v_{k+1}']):
#     pl.plot(all_ys[:, idx, 0], all_ys[:, idx, 1], label=name, c=pl.colormaps['plasma'](idx/120))
#
#
# for i in [50, 80, 90]:
#     # plot short trajectory segments too. shape = (n trajectories, n points per trajectory, nx=2)
#     plot_states = all_ys[:, i:i+5, 0:2]
#
#     # we would like the trajectories to have equal-ish distance.
#     # mask out with nan until distance is large enough
#     d_min = 0.1
#     prev_pt = plot_states[0, 0, :]
#     for j in range(1, plot_states.shape[0]):
#         dist = np.linalg.norm(plot_states[j, 0, :] - prev_pt)
#
#         if dist < d_min:
#             # set this point to nan and go to next.
#             plot_states = plot_states.at[j, :, :].set(np.nan)
#         else:
#             # use this point for plotting and mark as prev_pt
#             prev_pt = plot_states[j, 0, :]
#
#     # also set each last one to nan to not connect.
#     plot_states = plot_states.at[:, -1, :].set(np.nan)
#     plot_states = plot_states.reshape(-1, 2)
#
#     # pl.plot(plot_states[:, 0], plot_states[:, 1], c='black', alpha=0.7, label='optimal trajectories' if i==50 else None)
# pl.legend()
# pl.gca().set_aspect('equal')

# pl.show()
# ipdb.set_trace()


pl.figure()
pl.subplot(211)
for idx, name in zip([20, 40], ['v_k', 'v_{k+1}']):
    pl.plot(all_ys[:, idx, 0], all_ys[:, idx, 1], label=name, c=pl.colormaps['plasma'](idx/120))


traj_range = (20, 40)

# then, similar code as above. here for "uniform" sampling:
plot_states = all_ys[:, :, 0:2]
level = traj_range[0]

# all_ys.shape = (N trajs, N_ts, nx)
d_min = 0.1
prev_pt = plot_states[0, level, :]
for j in range(1, plot_states.shape[0]):
    dist = np.linalg.norm(plot_states[j, level, :] - prev_pt)

    if dist < d_min:
        # set this point to nan and go to next.
        plot_states = plot_states.at[j, :, :].set(np.nan)
    else:
        # use this point for plotting and mark as prev_pt
        prev_pt = plot_states[j, level, :]

pl.plot(plot_states[:, traj_range[0]:traj_range[1]+1, 0].flatten(), plot_states[:, traj_range[0]:traj_range[1]+1, 1].flatten(), label='uniformly sampled trajectories')
pl.legend()

print('trajectories plotted (uniform)')
print(np.sum(~np.isnan(plot_states[:, 0, 0])))


# and for better sampling.
pl.subplot(212)
for idx, name in zip([20, 40], ['v_k', 'v_{k+1}']):
    pl.plot(all_ys[:, idx, 0], all_ys[:, idx, 1], label=name, c=pl.colormaps['plasma'](idx/120))

traj_range = (20, 40)

plot_states = all_ys[:, :, 0:2]
level = traj_range[1]

# all_ys.shape = (N trajs, N_ts, nx)
d_min = 0.2
prev_pt = plot_states[0, level, :]
for j in range(1, plot_states.shape[0]):
    dist = np.linalg.norm(plot_states[j, level, :] - prev_pt)

    if dist < d_min:
        # set this point to nan and go to next.
        plot_states = plot_states.at[j, :, :].set(np.nan)
    else:
        # use this point for plotting and mark as prev_pt
        prev_pt = plot_states[j, level, :]

pl.plot(plot_states[:, traj_range[0]:traj_range[1]+1, 0].flatten(), plot_states[:, traj_range[0]:traj_range[1]+1, 1].flatten(), label='extrapolation guided sampling of trajectories')
pl.legend()

print('trajectories plotted (smarter)')
print(np.sum(~np.isnan(plot_states[:, 0, 0])))


pl.show()
ipdb.set_trace()


print('done')
