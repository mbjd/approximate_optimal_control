#!/usr/bin/env python
import jax
import jax.numpy as np
import diffrax

import pontryagin_utils

import ipdb
import scipy
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

from misc import *

import numpy as onp

from jax import config
config.update("jax_enable_x64", True)

from orbits_experiment import define_problem_params, base_algo_params

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


eq = problem_params['x_eq']
V_f = lambda x: 0.5 * (x - eq).T @ P_lqr @ (x - eq)

thetas = np.linspace(0, 2 * np.pi, 1024)[:-1]
circle_xs = jax.vmap(lambda theta: np.array([np.sin(theta), np.cos(theta)]))(thetas)
xfs = 0.1 * circle_xs @ np.linalg.inv(scipy.linalg.sqrtm(P_lqr)) + problem_params['x_eq'][None, :]
yprev = yfs = jax.vmap(lambda xf: dict(x=xf, v=V_f(xf), vx=jax.grad(V_f)(xf), t=0.))(xfs)


yf = jtm(itemgetter(0), yfs)
vf = yf['v']

# v_upper = 1000.

@jax.jit
def remesh(yprev, yfs):

    # the magic sauce.
    # if necessary: find here by bisection the largest v such that some
    # maximum distance between points is not exceeded.
    # yfs = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)

    # yfs['x'] represents a closed curve in state space. consider it equal to
    # its piecewise linear interpolation for now. we want something like the
    # arclength parametrization, to redistribute the points equidistantly. how?

    # first get the distances between all the neighbors.
    xs = yfs['x']
    rolled_xs = np.roll(xs, -1, axis=0)
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
    yfs = yprev
    new_v = np.interp(frac_idx, np.arange(N), yfs['v'])
    new_x = np.array([np.interp(frac_idx, np.arange(N), yfs['x'][:, i]) for i in range(2)]).T
    new_vx = np.array([np.interp(frac_idx, np.arange(N), yfs['vx'][:, i]) for i in range(2)]).T
    new_t = np.interp(frac_idx, np.arange(N), yfs['t'])

    new_y = dict(x=new_x, v=new_v, vx=new_vx, t=new_t)
    return new_y

solve_fast = jax.jit(jax.vmap(solve_backward, in_axes=(0, None)))

vmax = 310
levels = np.logspace(0., np.log10(vmax), 120)
levels = np.linspace(1, np.sqrt(vmax), 50)**2
levels = np.linspace(1, vmax, 30)
levels = np.concatenate([levels, np.array([450])])

sols = None

for v_upper in tqdm.tqdm(levels):

    sols = solve_fast(yfs, v_upper)

    # mod_xs = sols.ys['x'].at[:, -1, :].set(np.nan)
    # pl.plot(*mod_xs.reshape(-1, 2).T, '.-', alpha=.3, color='C0')

    # yprev = yfs
    yfs = jax.vmap(lambda sol: sol.evaluate(sol.t1))(sols)

    viridis = matplotlib.colormaps['viridis']
    pl.plot(*yfs['x'].T, alpha=.5, color = viridis(v_upper / levels[-1]) )

    yfs = remesh(yprev, yfs)
    yprev = yfs

# somehow it does another thing than i meant for it to do. i thought i had
# coded a resampling strategy which restarted trajectories at the last level
# set, so as to reach a uniform distribution at upper level set. but it looks
# like they are restarted all the way at the bottom level set!! not sure why.
# but still works. actually is a lot better from the approximation error
# persepctive. then the blocky shapes can be explained by reaching the end of
# floating point precision at the terminal level set.

pl.plot(sols.ys['x'][:, :, 0].flatten(), sols.ys['x'][:, :, 1].flatten(), alpha=.1, color='black')

for v in np.linspace(310, 450, 10):

    ys = jax.vmap(lambda sol: sol.evaluate(v))(sols)
    viridis = matplotlib.colormaps['viridis']
    pl.plot(*ys['x'].T, alpha=.5, color = viridis(v / levels[-1]) )

ipdb.set_trace()
pl.show()

def find_collision_continuation(sols, v_upper):

    # approach it the opposite way, with trajectories.
    # assume all "collisions" happen in the value slice covered by sols.
    pl.plot(*sols.ys['x'][0])
    pass


def find_self_intersection(xs):

    # given a closed curve in 2d space, return all points where it intersects itself.
    # absolutely brute force. no apologies.

    # first, we want all pairs of neighboring points.
    left_idxs = np.arange(xs.shape[0])
    right_idxs = np.roll(left_idxs, -1)

    # we want to know if the line segment between

    xa = xs[None, :, :]
    xb = xs[:, None, :]

    # for each pair of line segments (x1, x2) and (z1, z2) we want to know if
    # the intersection between the two exists and



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
