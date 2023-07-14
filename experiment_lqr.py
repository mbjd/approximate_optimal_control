#!/usr/bin/env python
import jax
import jax.numpy as np
import diffrax

import gradient_gp
import sampling
import pontryagin_utils
import plotting_utils
import nn_utils
import eval_utils

from main import sample_uniform, experiment_controlcost_vs_traindata, ode_dt_sweep

import ipdb
import scipy
import matplotlib
import matplotlib.pyplot as pl
import tqdm
import warnings
from functools import partial

import numpy as onp

from jax.config import config


def experiment_controlcost_vs_traindata_lqr_comparison(problem_params, algo_params, key):


    sysname = problem_params['system_name']

    assert sysname.startswith('double_integrator_linear'), 'otherwise lqr comparison makes no sense'

    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')

    # choose initial states for sim. the same each time.
    key, x0key = jax.random.split(key)

    # shuffle data to 'get rid of' interdependence
    key, subkey = jax.random.split(key)
    permuted_idx = jax.random.permutation(subkey, np.arange(y0s.shape[0]))
    y0s = y0s[permuted_idx, :]
    lamTs = lamTs[permuted_idx, :]

    nx = problem_params['nx']

    # data to evaluate the nn on
    eval_set, y0s = np.split(y0s, [256], axis=0)
    xs_eval, gradients_eval, v_eval = np.split(eval_set, [nx, 2*nx], axis=1)

    # x0s from test set.
    # confirmed they are actually the same as the ones used in the experiment.
    # (if seed=0 in the main function)
    x0s = jax.random.choice(x0key, xs_eval, shape=(algo_params['sim_N'],))


    print('making comparison to closed form LQR solution')
    # too lazy to properly extract these from the problem description...
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0, 1]]).T
    Q = np.eye(2)
    R = np.eye(1)
    invR = np.linalg.inv(R)
    N = np.zeros((2, 1))  # coupled state-input cost

    # define matrix ricatti diff. eq.
    def P_dot(t, P, args=None):
        return -( A.T @ P + P @ A - (P @ B + N) @ invR @ (B.T @ P + N.T) + Q )

    def get_P0(P_T, dt=1/128):
        term = diffrax.ODETerm(P_dot)
        solver = diffrax.Tsit5()

        # maybe easier to control the timing intervals like this?
        saveat = diffrax.SaveAt(steps=True)

        dt = -np.abs(dt)  # negative for backward integration.

        max_steps = int(problem_params['T'] / np.abs(dt))

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=problem_params['T'], t1=0., dt0=dt, y0=P_T,
                saveat=saveat, max_steps=max_steps,
        )

        return solution

    # even smaller dt, and large P_T
    P_sol = get_P0(10000 * np.eye(2), dt=2**(-18))

    P0 = P_sol.ys[-1]
    K0 = invR @ (B.T @ P0 + N.T)

    # compare with inf-horizon solution
    # from http://www.mwm.im/lqr-controllers-with-python/
    def lqr(A,B,Q,R):
        """Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """
        #ref Bertsekas, p.151
        #first, try to solve the ricatti equation
        X = onp.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        #compute the LQR gain
        K = onp.matrix(scipy.linalg.inv(R)*(B.T*X))
        eigVals, eigVecs = scipy.linalg.eig(A-B*K)
        return K, X, eigVals

    K0_inf, P0_inf, eigvals = lqr(A, B, Q, R)
    K0_inf = invR @ (B.T @ P0_inf + N.T)

    print(f' CARE finite-horizon controller: {K0}')
    print(f' scipy   inf-horizon controller: {K0_inf}')

    # they are basically the same, we are good.
    # so now we step to calculating the cost. just like in the experiment itself :)

    ustar_fct = lambda x: -K0 @ x
    all_sols = eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)

    # extract cost...
    cost_mean = all_sols.ys[:, -1, nx].mean()
    cost_std = all_sols.ys[:, -1, nx].std()

    mean_std = np.array([cost_mean, cost_std])
    print(f'mean cost: {cost_mean}')

    np.save(f'datasets/controlcost_lqr_meanstd_{sysname}.npy', mean_std)
    return

    # from here on this has become my debugging playground
    # but the corresponding error is found and fixed so hopefully we won't need this anymore

    # sanity check - compare w/ dataset itself
    # y0s = np.load('datasets/last_y0s_double_integrator_linear.npy')
    # y0s_smaller_dt = np.load('datasets/last_y0s_double_integrator_linear_pontryagintest.npy')
    x0s, lam0s, v0s      = np.split(y0s, [2, 4], axis=1)
    # x0s_smaller_dt, _, v0s_smaller_dt = np.split(y0s_smaller_dt, [2, 4], axis=1)

    # get_lam_v = jax.vmap(lambda x0: np.concatenate([(2 * x0.T @ P0).reshape(2,), (x0.T @ P0 @ x0).reshape(1,)]))
    # lqr_lam_v_base = get_lam_v(x0s)
    # lqr_lam_v_smalldt = get_lam_v(x0s_smaller_dt)


    # mean_errs_base = np.mean(((lqr_lam_v_base - y0s[:, 2:])/(1+np.abs(lqr_lam_v_base)))**2, axis=0)
    # mean_errs_smalldt = np.mean(((lqr_lam_v_smalldt - y0s_smaller_dt[:, 2:])/(1+np.abs(lqr_lam_v_smalldt)))**2, axis=0)
    # # mean_errs_smalldt = np.mean((lqr_lam_v_smalldt - y0s_smaller_dt[:, 2:])**2, axis=0)

    # print(' errors. cols = [λ0, λ1, v] ')
    # print(f'base    : {mean_errs_base}')
    # print(f'small dt: {mean_errs_smalldt}')

    # really weird: these errors do not seem to decrease at all or only veery slightly
    # is the problem posed the wrong way???

    # # should look like a line, but is a long thin ellipse kind of thing
    # pl.figure()
    # pl.scatter(lqr_lam_v_base[:, 0], y0s[:, 2])

    # fig = pl.figure()
    # ax3d = fig.add_subplot(131, projection='3d')
    # idx = jax.random.uniform(key, shape=(x0s.shape[0],)) < 0.1
    # sc = ax3d.scatter(x0s[idx, 0], x0s[idx, 1], lqr_lam_v_base[idx, 0], c='tab:blue', label='lqr')
    # sc = ax3d.scatter(x0s[idx, 0], x0s[idx, 1], y0s[idx, 2], c='tab:green', label='pmp')
    # # pl.colorbar(sc, label='λ0 error (LQR-PMP)')
    # pl.show()


    f = problem_params['f']
    l = problem_params['l']
    nx, nu = 2, 1
    U = problem_params['U_interval']

    ### linear fit for u* found by PMP.

    # u_star_simple = lambda x, λ: pontryagin_utils.u_star_costate(f, l, λ, 0, x, nx, nu, U)
    ustars = jax.vmap(pontryagin_utils.u_star_new, in_axes=(0, 0, None))(x0s, lam0s, problem_params)

    # we want to find K such that u = -K x
    K_pmp, residuals, rank, svals = np.linalg.lstsq(x0s, -ustars)

    print(' linear gain obtained by fitting lstsq to PMP u0')
    print(K_pmp.T)
    print(' elementwise K_pmp / K_lqr finite horizon')
    print(K_pmp.T / K0)
    # very good fit with max error norm 1e-15...



    ### compare trajectories found by PMP with value/costate info from LQR sol.

    solver = pontryagin_utils.make_single_pontryagin_solver(problem_params, algo_params)
    lamT = np.array([-.05, -.025])  # should be ok for linear and nonlinear example in report
    yT = np.concatenate([np.zeros(2), lamT, np.zeros(1)])

    sol, y0 = solver(yT, problem_params['T'], 0.)


    # V(x) = x.T P x, λ(x) = grad_x V(x) = 2 x.T P
    lqr_costate_fct = lambda P, x: 2 * x.T @ P
    lqr_V_fct = lambda P, x:  x.T @ P @ x

    # subsample P solution for lighter plotting.
    step = 1024
    P_sol_ts = P_sol.ts[::step]
    P_sol_ys = P_sol.ys[::step]

    ys_for_Psol = jax.vmap(sol.evaluate)(P_sol_ts)
    xs_for_Psol = ys_for_Psol[:, 0:2]

    lqr_costates_along_pmp_traj = jax.vmap(lqr_costate_fct, in_axes=(0, 0))(P_sol_ys, xs_for_Psol)
    lqr_values_along_pmp_traj = jax.vmap(lqr_V_fct, in_axes=(0, 0))(P_sol_ys, xs_for_Psol)


    def plot_var(ts, ys, comment):
        labels = ['x0', 'x1', 'λ0', 'λ1', 'v']
        for i in range(5):
            pl.subplot(5, 1, i+1)
            pl.plot(ts, ys[:, i], label=labels[i] + ' ' + comment)
            pl.legend()
            pl.subplots_adjust(hspace=0)

    plot_var(sol.ts, sol.ys, '(PMP solver)')

    ustar_fct = lambda x: -K0 @ x
    x0s = y0[None, 0:2]
    lqr_sol = eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)
    lqr_t_idx = lqr_sol.ts.squeeze() < 8
    lqr_ts = lqr_sol.ts[0, lqr_t_idx]

    pseudo_vs = lqr_sol.ys[0, lqr_t_idx, 2]
    lqr_ys = np.column_stack([
        lqr_sol.ys[0, lqr_t_idx, 0:2],
        np.zeros((lqr_ts.shape[0], 2)),
        pseudo_vs[-1] - pseudo_vs,
    ])

    plot_var(lqr_ts, lqr_ys, '(LQR closed loop)')

    ustar_fct = lambda x: -K_pmp.T @ x
    x0s = y0[None, 0:2]
    lqr_sol = eval_utils.closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s)
    lqr_t_idx = lqr_sol.ts.squeeze() < 8
    lqr_ts = lqr_sol.ts[0, lqr_t_idx]

    pseudo_vs = lqr_sol.ys[0, lqr_t_idx, 2]
    lqr_ys = np.column_stack([
        lqr_sol.ys[0, lqr_t_idx, 0:2],
        np.zeros((lqr_ts.shape[0], 2)),
        pseudo_vs[-1] - pseudo_vs,
    ])

    plot_var(lqr_ts, lqr_ys, '(K = lstsq fit of PMP u0s closed loop)')

    # pl.subplot(311)
    # pl.plot(sol.ts, sol.ys[:, 2], label='λ0 (PMP solver)')
    # pl.plot(P_sol_ts, lqr_costates_along_pmp_traj[:, 0], label='λ0 (from LQR sol, along PMP traj)')


    # pl.legend()

    # pl.subplot(312)
    # pl.plot(sol.ts, sol.ys[:, 3], label='λ1 (PMP solver)')
    # pl.plot(P_sol_ts, lqr_costates_along_pmp_traj[:, 1], label='λ1 (from LQR sol, along PMP traj)')

    # pl.legend()

    # pl.subplot(313)
    # pl.plot(sol.ts, sol.ys[:, 4], label='v (PMP solver)')
    # pl.plot(P_sol_ts, lqr_values_along_pmp_traj, label='λ1 (from LQR sol, along PMP traj)')

    # pl.legend()




    # next sanity check - see if the gradient of V w.r.t. x is really the costate...

    x0 = y0[0:2]

    # maps λT to y0 (whole vector!)
    pmp_fct = lambda λT: solver(np.concatenate([np.zeros(2), λT, np.zeros(1)]), problem_params['T'], 0.)[1]


    dydlam = jax.jacobian(pmp_fct)(lamT)

    dxdlam, dlam0dlam, dvdlam = np.split(dydlam, [2, 4], axis=0)

    # we want: dvdx, to see if it is equal to the costate
    # dvdx = dvdlam * dlamdx = dvdlam @

    # we want: dvdx, to see if it is equal to the costate
    # dvdx = dvdlam @ dlamdx = dvdlam @ (dxdlam)^-1



    dvdx = dvdlam @ np.linalg.inv(dxdlam)
    print(f'dv/dx calculated with autodiff:  {dvdx}')
    print(f'dv/dx according to costate λ(0): {y0[2:4]}')
    # but also: dvdx = lam0. that seems to not be the case at all...

    # then: is d/dt H(x, u, λ) = 0?

    H = lambda x, u, λ: l(0, x, u) + λ.T @ f(0, x, u)

    xs, λs, vs = np.split(sol.ys[:-1], [2, 4], axis=1)

    all_ustars = jax.vmap(pontryagin_utils.u_star_new, in_axes=(0, 0, None))(xs, λs, problem_params)
    all_H_vals = jax.vmap(H)(xs, all_ustars, λs)


    xtest = xs[4]
    lamtest = λs[4]

    def u_star_gd(xtest, lamtest, N=1000):
        # find ustar by gradient descent
        u = np.zeros(1)
        gradH = jax.jit(jax.grad(H, argnums=1))
        for i in range(N):
            u = u = u - 0.01 * gradH(xtest, u, lamtest)

        return u

    # it seems that consistently u_star_simple = 2*u_star_gd.
    print('u* closed form:')
    print(pontryagin_utils.u_star_new(xtest, lamtest, problem_params))
    print('u* gradient descent:')
    print(u_star_gd(xtest, lamtest))

    pl.show()
    ipdb.set_trace()






# simple control system. double integrator with friction term.
def f(t, x, u):
    # p' = v
    # v' = f
    # f =  u

    # v_cubed = (np.array([[0, 1]]) @ x)**3
    return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ (u)

def l(t, x, u):
    Q = np.eye(2)
    R = np.eye(1)
    return x.T @ Q @ x + u.T @ R @ u

def h(x):
    Qf = 1 * np.eye(2)
    return (x.T @ Qf @ x).reshape()


problem_params = {
        'system_name': 'double_integrator_linear_corrected',
        'f': f,
        'l': l,
        'h': h,
        'T': 8,
        'nx': 2,
        'nu': 1,
        'U_interval': [-np.inf, np.inf],
        'terminal_constraint': True,  # not tested with False for a long time
        'V_max': 16,
}

x_sample_scale = np.diag(np.array([1, 3]))
x_sample_cov = x_sample_scale @ x_sample_scale.T

# algo params copied from first resampling characteristics solvers
# -> so some of them might not be relevant
algo_params = {
        'pontryagin_solver_dt': 1/16,
        'pontryagin_solver_dense': False,

        # 'pontryagin_sampler_n_trajectories': 32,
        # 'pontryagin_sampler_n_iters': 8,
        # 'pontryagin_sampler_n_extrarounds': 2,
        # 'pontryagin_sampler_strategy': 'importance',
        # 'pontryagin_sampler_deterministic': False,
        # 'pontryagin_sampler_plot': False,  # plotting takes like 1000x longer than the computation
        # 'pontryagin_sampler_returns': 'functions',

        'sampler_dt': 1/64,
        'sampler_burn_in': 8,
        'sampler_N_chains': 64,  # with pmap this has to be 4
        'samper_samples_per_chain': 2**6,  # actual samples = N_chains * samples
        'sampler_steps_per_sample': 4,
        'sampler_plot': False,
        'sampler_tqdm': False,

        'x_sample_cov': x_sample_cov,
        'x_max_mahalanobis_dist': 2,

        'load_last': True,

        'nn_layersizes': [64, 64, 64],
        'nn_V_gradient_penalty': 50,
        'nn_batchsize': 128,
        'nn_N_max': 8192,
        'nn_N_epochs': 10,
        'nn_progressbar': True,
        'nn_testset_fraction': 0.1,
        'nn_ensemble_size': 16,
        'lr_staircase': False,
        'lr_staircase_steps': 4,
        'lr_init': 0.05,
        'lr_final': 0.005,

        'sim_T': 16,
        'sim_dt': 1/16,
        'sim_N': 32,
}

# the matrix used to define the relevant state space subset in the paper
#   sqrt(x.T @ Σ_inv @ x) - max_dist
# = max_dist * sqrt((x.T @ Σ_inv/max_dist**2 @ x) - 1)
# so we can set Q_S = Σ_inv/max_dist and just get a different scaling factor
algo_params['x_Q_S'] = np.linalg.inv(x_sample_cov) / algo_params['x_max_mahalanobis_dist']**2

# generate data
sample_uniform(problem_params, algo_params, key=jax.random.PRNGKey(0))

# fit NNs & evaluate controller performance for different amounts of data
key = jax.random.PRNGKey(0)
# experiment_controlcost_vs_traindata_lqr_comparison(problem_params, algo_params, key)
# experiment_controlcost_vs_traindata(problem_params, algo_params, key)

# ode_dt_sweep(problem_params, algo_params)
