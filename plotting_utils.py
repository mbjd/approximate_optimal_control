# jax
import jax
import jax.numpy as np

# other, trivial stuff
import numpy as onp

# import tk as tkinter
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl

import ipdb
import time
import numpy as onp

from functools import partial
cmap = 'viridis'

import pontryagin_utils

def plot_sol(sol, problem_params):

    # adapted from plot_forward_backward in ddp_optimizer
    # this works regardless of v/t reparameterisation.
    # all the x axes are physical times as stored in sol.ys['t']
    # all interpolations are done with ODE solver "t", so whatever independent
    # variable we happen to have

    interp_ts = np.linspace(sol.t0, sol.t1, 5001)

    # plot the state trajectory of the forward pass, interpolation & nodes.
    ax1 = pl.subplot(221)

    pl.plot(sol.ys['t'], sol.ys['x'], marker='.', linestyle='', alpha=1)
    # pl.plot(sol.ys['t'], sol.ys['v'], marker='.', linestyle='', alpha=1)
    interp_ys = jax.vmap(sol.evaluate)(interp_ts)
    pl.gca().set_prop_cycle(None)
    pl.plot(interp_ys['t'], interp_ys['x'], alpha=0.5, label=problem_params['state_names'])
    # pl.plot(interp_ys['t'], interp_ys['v'], alpha=0.5, label='v(x(t))')
    pl.legend()


    pl.subplot(222, sharex=ax1)
    us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(
        sol.ys['x'], sol.ys['vx'], problem_params
    )
    def u_t(t):
        state_t = sol.evaluate(t)
        return pontryagin_utils.u_star_2d(state_t['x'], state_t['vx'], problem_params)

    us_interp = jax.vmap(u_t)(interp_ts)

    pl.plot(sol.ys['t'], us, linestyle='', marker='.')
    pl.gca().set_prop_cycle(None)
    pl.plot(interp_ys['t'], us_interp, label=('u_0', 'u_1'))
    pl.legend()

    if 'vxx' not in sol.ys:
        # from here on we only plot hessian related stuff
        # so if that was not calculated, exit.
        return


    # plot the eigenvalues of S from the backward pass.
    pl.subplot(223, sharex=ax1)

    # eigenvalues at nodes.
    sorted_eigs = lambda S: np.sort(np.linalg.eig(S)[0].real)

    S_eigenvalues = jax.vmap(sorted_eigs)(sol.ys['vxx'])
    eigv_label = ['S(t) eigenvalues'] + [None] * (problem_params['nx']-1)

    eig_plot_fct = pl.plot  # = pl.semilogy

    eig_plot_fct(sol.ys['t'], S_eigenvalues, color='C0', marker='.', linestyle='', label=eigv_label)
    # also as line bc this line is more accurate than the "interpolated" one below if timesteps become very small
    eig_plot_fct(sol.ys['t'], S_eigenvalues, color='C0')

    # eigenvalues interpolated. though this is kind of dumb seeing how the backward
    # solver very closely steps to the non-differentiable points.
    sorted_eigs_interp = jax.vmap(sorted_eigs)(interp_ys['vxx'])
    eig_plot_fct(interp_ys['t'], sorted_eigs_interp, color='C0', linestyle='--', alpha=.5)

    # product of all eigenvalues = det(S)
    # dets = np.prod(S_eigenvalues, axis=1)
    # eig_plot_fct(sol.ys['t'], dets, color='C1', marker='.', label='prod(eigs(S))', alpha=.5)


    pl.legend()

    pl.subplot(224, sharex=ax1)
    # and raw Vxx entries.
    vxx_entries = interp_ys['vxx'].reshape(-1, problem_params['nx']**2)
    label = ['entries of Vxx(t)'] + [None] * (problem_params['nx']**2-1)
    pl.plot(interp_ys['t'], vxx_entries, label=label, color='green', alpha=.3)
    pl.legend()


    # or, pd-ness of the ricatti equation terms.
    # oups = jax.vmap(ricatti_rhs_eigenvalues)(sol.ys)

    # for j, k in enumerate(oups.keys()):
    #     # this is how we do it dadaTadadadaTada this is how we do it
    #     label = k # if len(oups[k].shape) == 1 else [k] + [None] * (oups[k].shape[1]-1)
    #     pl.plot(sol.ys['t'], oups[k], label=label, color=f'C{j}', alpha=.5)

    pl.legend()
    # ipdb.set_trace()



def plot_us(sols, problem_params, rotate=True, c='C0'):

    # plot all the u trajectories of a vmapped solutions object.

    # we flatten them here -- the inf padding breaks up the plot nicely
    all_xs = sols.ys['x'].reshape(-1, problem_params['nx'])
    all_lams = sols.ys['vx'].reshape(-1, problem_params['nx'])
    us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(all_xs, all_lams, problem_params)

    if rotate:
        diff_and_sum = np.array([[1, -1], [1, 1]]).T
        us = us @ diff_and_sum
        pl.xlabel('u0 - u1')
        pl.ylabel('u0 + u1')
    else:
        pl.xlabel('u0')
        pl.ylabel('u1')

    pl.plot(us[:, 0], us[:, 1], alpha=0.1, marker='.', c=c)
    pl.legend()




def plot_ellipse(Q, N_pts=101):

    # plot ellipse S = {x | x.T Q x = 1}
    # x.T Q x = x.T Q^.5.T Q^.5 x = || Q^.5 x || == 1
    # so basically, Q^.5 x is the unit circle, for x in S.
    # Therefore, Q^(-1/2) (unit circle) = S

    thetas = np.linspace(0, 2*np.pi, N_pts)

    circle = jax.vmap(lambda t: np.array([np.cos(t), np.sin(t)]))(thetas).T

    # it should be positive definite for a unique solution
    # hopefully the user is smart enough
    Q_half_inv = jax.scipy.linalg.sqrtm(np.linalg.inv(Q)).real

    ellipse = Q_half_inv @ circle
    pl.plot(ellipse[0, :], ellipse[1, :], color='red', alpha=.5)





def plot_nn_train_outputs_old(outputs):

    pl.figure('NN training visualisation', figsize=(15, 10))

    # outputs is a dict where keys are different outputs, and the value
    # is an array containing that output for ALL relevant training iterations,

    cs = ['tab:blue', 'tab:green', 'tab:red']
    c=0
    a = .2

    # training subplot
    ax1 = pl.subplot(211)

    make_nice = lambda s: s.replace('_', ' ')

    for k in outputs.keys():
        if 'train' in k:
            if len(outputs[k].shape) == 2:
                # NN ensemble
                N_ens, Nt = outputs[k].shape
                labels = ['' for i in range(N_ens)]
                labels[0] = make_nice(k)
                pl.semilogy(outputs[k].T, label=labels, alpha=a, c=cs[c])
                c = c+1
            else:
                pl.semilogy(outputs[k], label=make_nice(k), alpha=0.8)
    pl.grid(axis='both')
    # pl.ylim([1e-3, 1e3])
    pl.legend()

    # testing subplot
    pl.subplot(212, sharex=ax1, sharey=ax1)
    pl.gca().set_prop_cycle(None)
    c = 0

    for k in outputs.keys():
        if 'test' in k:
            if len(outputs[k].shape) == 2:
                # NN ensemble
                N_ens, Nt = outputs[k].shape
                labels = ['' for i in range(N_ens)]
                labels[0] = make_nice(k)
                pl.semilogy(outputs[k].T, label=labels, alpha=a, c=cs[c])
                c = c+1
            else:
                pl.semilogy(outputs[k], label=make_nice(k), alpha=0.8)
        if 'lr' in k:
            pl.semilogy(outputs[k], label='learning rate', linestyle='--', color='gray', alpha=.5)
    pl.grid(axis='both')
    # pl.ylim([1e-3, 1e3])
    pl.legend()



    

def plot_trajectory_vs_nn(sol, params, v_nn_unnormalised):

    # outside, do this: 
    # v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
    

    ax = pl.subplot(211)

    interp_ts = np.linspace(sol.t0, sol.t1, 200)

    xs = sol.ys['x']
    ts = sol.ys['t']
    vs = sol.ys['v']
    interp_ys = jax.vmap(sol.evaluate)(interp_ts)

    nx = sol.ys['x'].shape[-1]
    onelabel = lambda s: [s] + [None] * (nx-1)

    pl.plot(interp_ts, interp_ys['v'], alpha=.5, label='trajectory v(x(t))', c='C0')
    pl.plot(ts, vs, alpha=.5, linestyle='', marker='.', c='C0')

    # same with the NN
    vs = jax.vmap(v_nn_unnormalised, in_axes=(None, 0))(params, interp_ys['x'])
    pl.plot(interp_ts, vs, c='C1', label='NN v(x(t))')
    pl.legend()

    # and now the same for vx
    pl.subplot(212, sharex=ax)

    vxs = sol.ys['vx']
    pl.plot(interp_ts, interp_ys['vx'], alpha=.5, label=onelabel('trajectory vx(x(t))'), c='C0')
    pl.plot(ts, vxs, alpha=.5, linestyle='', marker='.', c='C0')

    nn_vx_fct = jax.jacobian(v_nn_unnormalised, argnums=1)
    nn_vxs = jax.vmap(nn_vx_fct, in_axes=(None, 0))(params, interp_ys['x'])
    pl.plot(interp_ts, nn_vxs, label=onelabel('NN v_x(x(t))'), c='C1')

    pl.legend()



def plot_trajectory_vs_nn_ensemble(sol, vmapped_params, v_nn_unnormalised):

    # outside, do this: 
    # v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))

    ax = pl.subplot(211)

    interp_ts = np.linspace(sol.t0, sol.t1, 200)

    xs = sol.ys['x']
    ts = sol.ys['t']
    vs = sol.ys['v']
    interp_ys = jax.vmap(sol.evaluate)(interp_ts)

    nx = sol.ys['x'].shape[-1]
    onelabel = lambda s: [s] + [None] * (nx-1)

    pl.plot(interp_ts, interp_ys['v'], alpha=.5, label='trajectory v(x(t))', c='C0')
    pl.plot(ts, vs, alpha=.5, linestyle='', marker='.', c='C0')

    confidence_width = 1  # make small to look better.

    # same with the NN
    # inner vmap for ys, outer vmap for params.
    vs = jax.vmap(jax.vmap(v_nn_unnormalised, in_axes=(None, 0)), in_axes=(0, None))(vmapped_params, interp_ys['x'])
    vs_mean = vs.mean(axis=0)
    vs_std = vs.std(axis=0)
    pl.plot(interp_ts, vs_mean, c='C1', label='NN v mean')
    pl.fill_between(interp_ts, vs_mean - confidence_width * vs_std, vs_mean + confidence_width * vs_std, color='C1', alpha=.2, label=f'NN v {confidence_width}σ band')
    pl.legend()

    # and now the same for vx
    pl.subplot(212, sharex=ax)

    vxs = sol.ys['vx']
    pl.plot(interp_ts, interp_ys['vx'], alpha=.5, label=onelabel('trajectory vx(x(t))'), c='C0')
    pl.plot(ts, vxs, alpha=.5, linestyle='', marker='.', c='C0')

    nn_vx_fct = jax.jacobian(v_nn_unnormalised, argnums=1)

    # shaped (N_params, N_ts, nx)
    vxs = jax.vmap(jax.vmap(nn_vx_fct, in_axes=(None, 0)), in_axes=(0, None))(vmapped_params, interp_ys['x'])

    # mean/std of NN ensemble
    vx_means = vxs.mean(axis=0)
    vx_stds = vxs.std(axis=0)

    nx = sol.ys['x'].shape[-1]
    # do this in a loop because fill_between does not like vectorised data
    for j in range(nx):

        label = 'NN vx mean' if j==0 else None
        pl.plot(interp_ts, vx_means[:, j], label=label, c='C1')

        lower = vx_means[:, j] - confidence_width*vx_stds[:, j]
        upper = vx_means[:, j] + confidence_width*vx_stds[:, j]
        label = f'NN vx {confidence_width}σ band' if j==0 else None
        pl.fill_between(interp_ts, lower, upper, color='C1', alpha=.2, label=label)

    pl.legend()



def plot_nn_train_outputs(outputs, alpha=.5, legend=True):

    # pl.figure('NN training visualisation', figsize=(15, 10))

    # newer version for order 2 sobolev nn. outputs is a dict with keys:
    # 'train_loss_terms', 'test_loss_terms':
    # both containing a Nx3 array with loss terms for v, vx, vxx
    # (just like returned by the training function)
    # 'lr': (N,) array of learning rates


    # training subplot
    ax = pl.subplot(211)
    ax.set_prop_cycle(None)
    pl.loglog(outputs['train_loss_terms'], label=('v', 'vx', 'vxx'), alpha=alpha)
    pl.ylabel('training losses')
    pl.grid('on')

    if legend:
        pl.legend()

    # training subplot
    ax = pl.subplot(212, sharex=ax, sharey=ax)
    ax.set_prop_cycle(None)
    pl.loglog(outputs['test_loss_terms'], label=('v', 'vx', 'vxx'), alpha=alpha)
    pl.ylabel('test losses (fixed PRNGKey)')
    pl.grid('on')

    pl.loglog(outputs['lr'], label='learning rate', linestyle='--', color='gray', alpha=.33)

    if legend:
        pl.legend()



