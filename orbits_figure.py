#!/usr/bin/env python

import diffrax
import flax
import jax
import jax.numpy as np
from jax import config

config.update("jax_enable_x64", True)

import gzip
import os
import pickle
import warnings
from functools import partial

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import numpy as onp
import scipy
import tqdm

import pontryagin_utils
from fig_config import *
from misc import *
from orbits_experiment import base_algo_params, define_problem_params

cmap = matplotlib.colormaps['viridis']
levelset_alpha=.7
traj_alpha = .7

N_trajs = 4096


# define basics {{{

problem_params = define_problem_params()
algo_params = base_algo_params()

# magic switch for value level sets.
algo_params['reparam'] = True
algo_params['dtmin'] = 0.001  # smaller for initial stuff
algo_params['dtmax'] = 1000.  # larger for high v
algo_params['pontryagin_solver_atol'] = 1e-5
algo_params['pontryagin_solver_rtol'] = 1e-5
algo_params['pontryagin_solver_maxsteps'] = 128

solve_backward, f_extended = pontryagin_utils.define_backward_solver(problem_params, algo_params)

K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

@partial(jax.jit, static_argnums=2)
def remesh(sols, frac, remesh_final):

    # the magic sauce.
    # if necessary: find here by bisection the largest v such that some
    # maximum distance between points is not exceeded.
    # frac=1 -> arclength at t1
    # frac=0 -> arclength at t0
    yfs = jax.vmap(lambda sol: sol.evaluate(sol.t0 * (1-frac) + sol.t1 * frac))(sols)

    # yfs['x'] represents a closed curve in state space. consider it equal to
    # its piecewise linear interpolation for now. we want something like the
    # arclength parametrization, to redistribute the points equidistantly. how?

    # first get the distances between all the neighbors.
    xs = yfs['x']
    rolled_xs = np.roll(xs, -1, axis=0)

    lifted_arclen=False
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
    # ys_remesh = jax.vmap(lambda sol: sol.evaluate(sol.t0 * (1-frac) + sol.t1 * frac))(sols)
    ys_remesh = jax.vmap(lambda sol: sol.evaluate(sol.t0))(sols)
    new_v = np.interp(frac_idx, np.arange(N), ys_remesh['v'])
    new_x = np.array([np.interp(frac_idx, np.arange(N), ys_remesh['x'][:, i]) for i in range(2)]).T
    new_vx = np.array([np.interp(frac_idx, np.arange(N), ys_remesh['vx'][:, i]) for i in range(2)]).T
    new_t = np.interp(frac_idx, np.arange(N), ys_remesh['t'])

    if remesh_final:
        new_v = jax.vmap(V_f)(new_x)
        new_vx = jax.vmap(jax.grad(V_f))(new_x)
        new_t = np.zeros_like(new_v)

    new_y = dict(x=new_x, v=new_v, vx=new_vx, t=new_t)
    return new_y

solve_fast = jax.jit(jax.vmap(solve_backward, in_axes=(0, None)))



eq = problem_params['x_eq']
V_f = lambda x: 0.5 * (x - eq).T @ P_lqr @ (x - eq)

thetas = np.linspace(0, 2 * np.pi, N_trajs)
circle_xs = jax.vmap(lambda theta: np.array([np.sin(theta), np.cos(theta)]))(thetas)
xfs = 0.1 * circle_xs @ np.linalg.inv(scipy.linalg.sqrtm(P_lqr)) + problem_params['x_eq'][None, :]
yfs = jax.vmap(lambda xf: dict(x=xf, v=V_f(xf), vx=jax.grad(V_f)(xf), t=0.))(xfs)

yf = jtm(itemgetter(0), yfs)
vf = yf['v']

vmax = 420
N=20
levels = np.logspace(np.log10(vf*2), np.log10(vmax), N)
levels = np.linspace(np.sqrt(vf*2), np.sqrt(vmax), N)**2
# solve_fast = jax.vmap(solve_backward, in_axes=(0, None))
# }}}


# make or read data {{{

fpath_sols = os.path.join('plot_data', 'orbits_refsol.msgpack.gz')
fpath_treedef = os.path.join('plot_data', 'orbits_refsol_treedef.pickle')

make_data = False
if make_data:


    remesh_final = True
    sols = None

    def find_min_l(yfs):

        def l_of_y(y):
            x = y['x']
            vx = y['vx']
            u = pontryagin_utils.u_star_general(x, vx, problem_params)
            return problem_params['l'](x, u)

        ls = jax.vmap(l_of_y)(yfs)

        min_l = np.min(ls)
        return min_l

    # stable manifold calculation. remeshing followed by recreating the
    # solutions all the way from Xf.

    with tqdm.tqdm(total=vmax) as pbar:
        for v_upper in levels:
            # otherwise it needs the increment not absolute progress.
            # https://github.com/tqdm/tqdm/issues/1264
            pbar.n = v_upper.item()
            pbar.refresh()

            # alright so it has to work a bit differently.
            # 1. get solutions starting at uniformly spaced points on dVk
            # 2. remesh them to be equidistant at dVk+1
            # 3. get solutions again.

            # step size selection just like the real thing
            min_l = find_min_l(yfs)
            vstep = 3. * min_l
            v_upper = v_upper + vstep

            # 1. uniform solutions.
            sols_uniform = solve_fast(yfs, v_upper)
            # 2. remeshing
            yfs = remesh(sols_uniform, 1.0, remesh_final)
            # 3. remeshed solutions.
            sols = solve_fast(yfs, v_upper)

            # pl.plot(*sols_uniform.ys['x'].reshape(-1,2).T, alpha=.3, label='uniform', c='grey')
            # pl.plot(*sols.ys['x'][::1].reshape(-1,2).T, alpha=.05, label='remeshed', c='black')
            # pl.legend()

            # yprev = yfs
            if remesh_final:
                yfs = jax.vmap(lambda sol: sol.evaluate(sol.t0))(sols)
            else:
                yfs = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)

            viridis = matplotlib.colormaps['viridis']
            # pl.plot(*yfs['x'].T, '-', alpha=.5, color = viridis(v_upper / levels[-1]) )

            # pl.plot(*sols.ys['x'].reshape(-1, 2).T, color='black', alpha=.1 )
    # now we have the sols object.

    print('saving sols...')
    # flattens into list of array leaves
    sols_flat, sols_shape = jax.tree_util.tree_flatten(sols)

    bs = flax.serialization.msgpack_serialize(sols_flat)
    with gzip.open(fpath_sols, 'wb') as f:
        f.write(bs)

    with open(fpath_treedef, 'wb') as f:
        pickle.dump(sols_shape, f)


else:
    print('make_data=False. instead reading from file.')

    with gzip.open(fpath_sols, 'rb') as f:
        bs = f.read()
    sols_flat = flax.serialization.msgpack_restore(bs)
    sols_flat = jtm(np.array, sols_flat)  # np array -> jax array

    with open(fpath_treedef, 'rb') as f:
        sols_shape = pickle.load(f)

    # i really did not think this would just work...
    sols = jax.tree_util.tree_unflatten(sols_shape, sols_flat)
# }}}


def plot_levelset(v, grey=False):
    ys = jax.vmap(lambda sol: sol.evaluate(v))(sols)

    if grey:
        color = 'black'
    else:
        color = cmap(v / levels[-1])

    pl.plot(*ys['x'].T, alpha=levelset_alpha, color = color)



# trajectories plot {{{
fig = pl.figure('trajectories', figsize=(pagewidth*.9, 0.3*pagewidth*.9), dpi=dpi)

v0, v1 = 2., 50.
eps = 0.001  # to certainly land in interior of domain of interpolation
# evaluate at nan too to break up line
vs_plot = np.concatenate([np.logspace(np.log10(v0+eps), np.log10(v1-eps), 51), np.array([np.nan])])
subsample = 32 * (N_trajs // 512)

# v0, v1 = (150., 300.)

ax = pl.subplot(121)
ax.set_aspect('equal')
plot_levelset(v0, grey=True)
plot_levelset(v1, grey=True)

# yfs at lower value level v0
yfs = jax.vmap(lambda sol: sol.evaluate(v0))(sols)
# trajectories from v0 to v1 (with distribution of last traj batch!)
sols_partial = solve_fast(yfs, v1)

yfs_uniform_lower = remesh(sols_partial, 0., False)
yfs_uniform_upper = remesh(sols_partial, 1., False)

sols_uniform_lower = solve_fast(yfs_uniform_lower, v1)
sols_uniform_upper = solve_fast(yfs_uniform_upper, v1)

plot_ys = jax.vmap(lambda sol: jax.vmap(sol.evaluate)(vs_plot))(sols_uniform_lower)
pl.plot(*plot_ys['x'][::subsample].reshape(-1,2).T, alpha=traj_alpha)

ax = pl.subplot(122, sharex=ax, sharey=ax)
ax.set_aspect('equal')
plot_ys = jax.vmap(lambda sol: jax.vmap(sol.evaluate)(vs_plot))(sols_uniform_upper)
plot_levelset(v0, grey=True)
plot_levelset(v1, grey=True)
pl.plot(*plot_ys['x'][::subsample].reshape(-1,2).T, alpha=traj_alpha)
fig.tight_layout()

pl.savefig(f'./{fig_dir}/trajectories.{fig_format}')

if show:
    pl.show()

# }}}

# levelset intersection calculation definition {{{

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

    # instead try simpler method. compute all distances between points,
    # find closest ones.

    # pairwise distance of all points.
    idx = first_idx
    point_dists = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    # "index distance" but mapped to a circle to respect circular nature.
    thetas = np.linspace(0, 2*np.pi, xs.shape[0])
    # scaled so that distance between indices is still about 1.
    circle = jax.vmap(lambda t: np.array([np.sin(t), np.cos(t)]))(thetas)
    idx_dists = np.linalg.norm(circle[:, None, :] - circle[None, :, :], axis=-1)

    # find the one with smallest point_dist over idx_dist. small offset to
    # avoid huge numbers.
    dist_ratios = point_dists / (0.001 + idx_dists)

    # only upper triangular part to remove duplicates (a, b) = (b, a)
    dist_ratios = np.triu(dist_ratios, 1)
    dist_ratios = np.where(dist_ratios == 0, np.inf, dist_ratios)

    # "remove" points closer to 10 in index distance.
    min_idx_dist = 10 * (xs.shape[0] / 1024)
    min_circle_dist = (min_idx_dist / xs.shape[0]) * 2*np.pi
    dist_ratios = np.where(idx_dists < min_circle_dist, np.inf, dist_ratios)

    first_intersection = np.unravel_index(np.argmin(dist_ratios), dist_ratios.shape)
    first_dist = point_dists[first_intersection]

    # then, throw out everything close to these indices and try again.
    dist_ratios_old = dist_ratios
    dist_ratios = dist_ratios + (np.inf * (np.abs(first_idx - first_intersection[0]) < min_idx_dist)[None, :])
    dist_ratios = dist_ratios + (np.inf * (np.abs(first_idx - first_intersection[1]) < min_idx_dist)[None, :])
    dist_ratios = dist_ratios + (np.inf * (np.abs(first_idx - first_intersection[0]) < min_idx_dist)[:, None])
    dist_ratios = dist_ratios + (np.inf * (np.abs(first_idx - first_intersection[1]) < min_idx_dist)[:, None])

    second_intersection = np.unravel_index(np.argmin(dist_ratios), dist_ratios.shape)
    second_dist = point_dists[second_intersection]

    # rather miss a collision than show a wrong one. the edge cases we can
    # just brush under the rug & show a figure without them.
    if first_dist > 0.005 or second_dist > 0.005:
        return None, None, None, None

    # pl.figure('ratios')
    # pl.subplot(121); pl.imshow(dist_ratios_old)
    # pl.subplot(122); pl.imshow(dist_ratios)

    # pl.figure('curves')
    # pl.plot(*xs.T, '.-', alpha=.3)
    # pl.plot(*xs[np.array(first_intersection)].T, '. ', c='red')
    # pl.plot(*xs[np.array(second_intersection)].T, '. ', c='red')

    idx = np.concatenate([np.array(first_intersection), np.array(second_intersection)]).sort()

    # 'clean' those indices by removing ones that are too close.
    # -> not needed anymore, by construction we only have 4 of them.

    # so now it looks like we half-reliably find those intersection points.
    # what to do now?
    segments = np.split(xs, idx)
    segments[-1] = np.concatenate([segments[-1], segments[0]], axis=0)
    segments.pop(0)

    # now, any old heuristic for finding out which ones are optimal will
    # do. attempt 1: largest and smallest mean magnitude?
    mean_mags = np.array([np.linalg.norm(seg, axis=-1).mean() for seg in segments])

    min_seg = segments[np.argmin(mean_mags)]
    max_seg = segments[np.argmax(mean_mags)]

    # add the first point last again to wrap
    min_seg = min_seg[np.arange(min_seg.shape[0] + 1)]
    max_seg = max_seg[np.arange(max_seg.shape[0] + 1)]

    # pl.figure('curves')
    # pl.plot(*min_seg.T, '.-', c='orange')
    # pl.plot(*max_seg.T, '.-', c='orange')
    # pl.show()
    return min_seg, max_seg, xs[first_intersection[0]], xs[second_intersection[0]]


def plot_levelset_intersect(v, grey=False):
    ys = jax.vmap(lambda sol: sol.evaluate(v))(sols)
    xs = ys['x']

    seg_a, seg_b, xa, xb = find_self_intersection(xs)

    if seg_a is None or seg_b is None:
        seg_a = xs
        seg_b = xs * np.nan

    if grey:
        color = 'black'
    else:
        viridis = matplotlib.colormaps['viridis']
        color = viridis(v / levels[-1])

    pl.plot(*seg_a.T, alpha=levelset_alpha, color = color)
    pl.plot(*seg_b.T, alpha=levelset_alpha, color = color)
    return xa, xb

# }}}

# intersecting level sets plot {{{
pl.figure('levelsets', figsize=(pagewidth, 0.4*pagewidth), dpi=dpi)
exp = 0.75 # between sqrt and linear. looks nicest
vs_plot = np.linspace((vf*50)**exp, vmax**exp, 20)**(1/exp)
v_uppers = (300, np.inf)

ax = None
for k in range(2):
    ax = pl.subplot(131 + k, sharex=ax, sharey=ax)
    ax.set_aspect('equal')
    for v in vs_plot:
        if v < v_uppers[k]:
            plot_levelset(v)

# uniform-ish time grid for all sols.
ys = jax.vmap(lambda sol: jax.vmap(sol.evaluate)(np.linspace(np.sqrt(sol.t0+0.0001), np.sqrt(sol.t1-0.01), 128)**2))(sols)



# vs_plot = np.linspace(vf, vmax, 21)
# v_uppers = (300, 340, np.inf)

intersecs = []
ax = pl.subplot(133, sharex=ax, sharey=ax)
ax.set_aspect('equal')
for v in vs_plot:
    if v < v_uppers[k]:
        print(v)
        if v < v_uppers[0]:
            plot_levelset(v)
        else:
            xa, xb = plot_levelset_intersect(v)
            # if xa is not None and xb is not None:
                # intersecs.append(xa)
                # intersecs.append(xb)

# do it again for the intersections at finer resolution
pl.figure('shit')

sqrtspace = lambda a, b, n: np.linspace(np.sqrt(a), np.sqrt(b), n)**2

print('second round')
vs = 315 + sqrtspace(0, vmax-315, 30)
for v in vs:
    xa, xb = plot_levelset_intersect(v)
    if xa is not None and xb is not None:
        print(v)
        intersecs.append(xa)
        intersecs.append(xb)
intersecs = np.array(intersecs)
pl.plot(*intersecs.T, '. ', c='red')
pl.clf()

# now, sort this 'intersecs' array appropriately.
# the broad direction of the line. worked haha :)
ks = intersecs @ np.array([1, 1])
idx_sorted = np.argsort(ks)
pl.figure('levelsets')
pl.plot(*intersecs[idx_sorted].T, alpha=1., c='red')

fig.tight_layout()
pl.savefig(f'./{fig_dir}/levelsets.{fig_format}',  bbox_inches='tight')
# norm = matplotlib.colors.Normalize(vmin=0., vmax=vmax)
# fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=pl.gca(), orientation='vertical', label='Some Units')

if show:
    pl.show()

# }}}

# results plot {{{

sysname = problem_params['system_name']
fpath = os.path.join(data_dir, f'{sysname}_controlcosts.msgpack.gz')
with gzip.open(fpath, 'rb') as f:
    bs = f.read()
eval_outputs = jtm(np.array, flax.serialization.msgpack_restore(bs))

# here we kinda have to be smart to compare the reference sol in the right
# way...

# also generally, what do we plot?
# learned value? closed loop trajectories?
# ref trajectories?

def imshow_like_contourf(xx, yy, vals, **kwargs):
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    img = vals[::-1]  # flip in vertical axis
    if 'levels' in kwargs:
        levels = kwargs['levels']
        vmin, vmax = levels[0], levels[-1]
        pl.imshow(img, vmin=vmin, vmax=vmax, extent=extent)
    else:
        pl.imshow(img, extent=extent)


xx = eval_outputs['xx']
yy = eval_outputs['yy']
learned_v = eval_outputs['learned_v']
controlcost = eval_outputs['controlcost']

ax = pl.subplot(121)
ax.set_aspect('equal')
levels = np.linspace(-2.2, 3., 30)
pl.contour(xx, yy, np.log10(learned_v), levels=levels)
pl.xlabel('learned v, contour')

pl.subplot(122, sharex=ax, sharey=ax)
imshow_like_contourf(xx, yy, np.log10(learned_v), levels=levels)
pl.xlabel('learned v, imshow')

fig = pl.figure()
ratio = controlcost / learned_v
ratio = ratio.at[learned_v > 1000].set(np.nan)
imshow_like_contourf(xx, yy, np.log10(ratio))
pl.colorbar()
pl.xlabel('cost ratio')

# and the most interesting thing finally: control cost vs 'reference'
# solution.

# # unwrap the angle.
# xs = sols.ys['x']
# xs = np.where(xs == np.inf, np.nan, xs)
# phis = np.arctan2(xs[:, :, 0], xs[:, :, 1])
# phis = np.unwrap(phis, axis=1)
#
# idx = ~np.isnan(phis)
#
# # expand state z = (x, phi).
# zs = np.concatenate([xs[idx], phis[idx, None]], axis=-1)
# xs = xs[idx]
# vs = sols.ys['v'][idx]
# vxs = sols.ys['vx'][idx]
#
# # learn the function z -> v very nicely.
# # dumbest possible classifier: nearest neighbor.
#
# def v_ref(x):
#
#     phis = np.arctan2(x[0], x[1]) + np.arange(-1, 2) * 2 * np.pi
#
#     def v_local(phi):
#         z = np.concatenate([x, np.array([phi])])
#         dists = np.linalg.norm(zs - z[None, :], axis=1)
#         closest_idx = np.argmin(dists)
#         closest_v = vs[closest_idx]
#         closest_x = zs[closest_idx, 0:2]
#         closest_vx = vxs[closest_idx]
#         v = closest_v + closest_vx @ (x - closest_x)
#         return closest_v
#
#     v_candidates = jax.vmap(v_local)(phis)
#     return np.min(v_candidates)
#
#
# v_ref(np.array([0.1, 1.1]))
# ipdb.set_trace()
#
#
# fig.tight_layout()
# pl.savefig(f'./{fig_dir}/orbits_results.{fig_format}',  bbox_inches='tight')

# }}}
