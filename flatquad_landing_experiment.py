#!/usr/bin/env python

import jax
import jax.numpy as np
import diffrax

import rrt_sampler
import pontryagin_utils
import ddp_optimizer
import visualiser
import ct_basics

import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf

import ipdb
import time
import numpy as onp
import tqdm
from operator import itemgetter


def u_star_debugging(problem_params, algo_params):

    # found this failure case randomly during ddp testing. it does appear to 
    # show up right before things go south. not sure if it is the culprit. 

    # this is pretty ugly, and i'm not exactly sure why it happens. in general
    # there seems to be some numerical noise on the u_star, although generally 
    # only like 1e-6 or 1e-5

    x = np.array([4.538097,6.19019,-0.10679064,-2.4105127,-3.6202462,0.04513513])
    lam = np.array([17.425396  , 22.878305  ,  0.73656803, -2.9136074 , -6.116059  ,0.28396177])

    # for pdb
    print(pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=True))

    def ustar_fct(x, lam):
        return pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=True)

    def ustar_fct_nonsmooth(x, lam):
        return pontryagin_utils.u_star_2d(x, lam, problem_params, smooth=False)

    ustar = pontryagin_utils.u_star_2d(x, lam, problem_params)
    ustar_vmap = jax.vmap(ustar_fct, in_axes=(0, 0))
    ustar_vmap_nonsmooth = jax.vmap(ustar_fct_nonsmooth, in_axes=(0, 0))

    # go in random directions a bit and plot. 
    k = jax.random.PRNGKey(2)
    direction = jax.random.normal(k, shape=(12,))
    direction = direction / np.linalg.norm(direction)
    xdir, lamdir = np.split(direction, [6])
    xdir = 0 * xdir  # just to check if piecewise linear...

    alphas = np.linspace(0, .3, 10001)[:, None]

    xs = x + alphas * xdir
    lams = lam + alphas * lamdir


    def ustar_fct(x, lam): 
        return pontryagin_utils.u_star_2d(x, lam, problem_params, debug_oups=True)

    ustar_vmap = jax.vmap(ustar_fct, in_axes=(0, 0))

    (ustars, debug_oups), (grads, debug_oups_grads) = jax.vmap(lambda x, lam: jax.jvp(ustar_fct, (x, lam), (lam, lamdir)))(xs, lams)

    pl.subplot(211)
    pl.plot(ustars, label='u*')
    pl.legend()
    pl.subplot(212)
    pl.plot(grads, label='(du*/dx).T (x direction of sweep)')
    pl.legend()

    # ipdb.set_trace()
    # this works only for the modified function with debug outputs!
    # ustars_nonsmooth, oups = ustar_vmap_nonsmooth(xs, lams)


    # pl.plot(oups['all_Hs_adjusted'] - oups['all_Hs_adjusted'].min(axis=1)[:, None], alpha=.5, label=('H unconstrained - H*', 'H1 - H*', 'H2 - H*', 'H3 - H*', 'H4 - H*'))
    # pl.gca().set_prop_cycle(None)
    # pl.plot(oups['all_Hs_adjusted_new'] - oups['all_Hs_adjusted_new'].min(axis=1)[:, None], alpha=.5, linestyle='--', label=('Ht unconstrained - Ht*', 'Ht1 - Ht*', 'Ht2 - Ht*', 'Ht3 - Ht*', 'Ht4 - Ht*'))
    # pl.legend(); pl.show()
    # ipdb.set_trace()


    # plot the curve traced by u*
    pl.figure()
    pl.plot(ustars[:, 0], ustars[:, 1], color='red', label='u*')

    cands = debug_oups['all_candidates']

    for j in range(cands.shape[1]):
        pl.plot(cands[:, j, 0], cands[:, j, 1], alpha=.3)

    lowerbounds = problem_params['U_interval'][0]
    upperbounds = problem_params['U_interval'][1]
    pl.plot([lowerbounds[0], lowerbounds[0], upperbounds[0], upperbounds[0], lowerbounds[0]],
            [lowerbounds[1], upperbounds[1], upperbounds[1], lowerbounds[1], lowerbounds[1]], 
            color='black', label='constraints', alpha=0.2)
    

    pl.show()

    ipdb.set_trace()



def backward_with_hessian(problem_params, algo_params):

    # find terminal LQR controller and value function. 
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    reldiff = np.linalg.norm(Phalf @ Phalf - P_lqr) / np.linalg.norm(P_lqr)
    assert reldiff < 1e-6, 'sqrtm(P) calculation failed or inaccurate'

    key = jax.random.PRNGKey(0)

    # two unit norm points
    normal_pts = jax.random.normal(key, shape=(2, problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # interpolate between them
    alphas = np.linspace(0, 1, 200)[:, None]
    pts_interp = alphas * unitsphere_pts[0] + (1-alphas) * unitsphere_pts[1]

    # put back on unit sphere. 
    pt_norms = jax.vmap(np.linalg.norm)(pts_interp)[:, None]
    pts_interp = pts_interp / pt_norms

    # different strategy: purely random ass points. 
    normal_pts = jax.random.normal(key, shape=(200, problem_params['nx']))
    pts_interp = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]


    # map to boundary of Xf. 
    xfs = pts_interp @ np.linalg.inv(Phalf) * 0.01




    # test if it worked
    xf_to_Vf = lambda x: x @ P_lqr @ x.T
    vfs =  jax.vmap(xf_to_Vf)(xfs)
    vf = vfs[0]

    # define RHS of the backward system, including Vxx
    def f_extended(t, state, args=None):

        # state variables = x and quadratic taylor expansion of V.
        x       = state['x']
        v       = state['v']
        costate = state['vx']
        S       = state['vxx']

        H = lambda x, u, λ: l(t, x, u) + λ.T @ f(t, x, u)

        # RHS of the necessary conditions without the hessian. 
        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)
            nx = problem_params['nx']

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # its derivatives, for propagation of Vxx.
        fx, gx = jax.jacobian(pmp_rhs, argnums=0)(x, costate)
        flam, glam = jax.jacobian(pmp_rhs, argnums=1)(x, costate)

        # we calculate this here one extra time, could be optimised
        u_star = pontryagin_utils.u_star_2d(x, costate, problem_params)

        # calculate all the RHS terms
        v_dot = -problem_params['l'](t, x, u_star)
        x_dot, costate_dot = pmp_rhs(x, costate)
        S_dot = gx + glam @ S - S @ fx - S @ flam @ S 

        # and pack them in a nice dict for the state.
        state_dot = dict()
        state_dot['x'] = x_dot
        state_dot['t'] = 1.
        state_dot['v'] = v_dot
        state_dot['vx'] = costate_dot
        state_dot['vxx'] = S_dot

        # don't forget to adjust integration boundaries and dt0 when changing 
        # physical time
        # return state_dot
        # v/t rescaling
        return jax.tree_util.tree_map(lambda z: z / -np.sqrt(-v_dot), state_dot)

    def solve_backward(x_f):

        # another 'advantage' of including the hessian is that now the ODE solver
        # actually takes the trouble of stepping rather precisely to the active
        # set changes.

        # still unsure about these factors of 2 or 1/2 or 1
        # but results look OK, Vxx stays constant-ish in the linear-ish region.
        v_f = x_f.T @ P_lqr @ x_f
        vx_f = 2 * P_lqr @ x_f
        vxx_f = P_lqr 

        state_f = {
            'x': x_f,
            't': 0,
            'v': v_f,
            'vx': vx_f, 
            'vxx': vxx_f,
        }

        term = diffrax.ODETerm(f_extended)

        relax_factor = 1.
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
            dtmin = 0.005,  # just to avoid getting stuck completely
            dtmax = 0.5,
        )

        # try this, maybe it works better \o/
        # step_ctrl_fixed = diffrax.StepTo(prev_forward_sol.ts)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        T = 10.

        backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=v_f, t1=10., dt0=vf/10, y0=state_f,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
        )

        return backward_sol

    '''
    the grand idea here is about this: 
    - simulate forward with approx. optimal controller, collect N trajectories 
      ending in a region where optimal control is known precisely enough
    - from those states, do backward shooting, until (one of):
      - we leave some big state space set outside of which we really dont care
      - we are really confident that the solution is not globally optimal anymore
      - after a couple of the fastest time constants have passed
    - add to dataset, train BNN/NN ensemble/similar
    - repeat

    here as a first testing bed thing we just do the first step: LQR forward sim -> backward sim. 
    seems like many of the same issues pop up as during semester project obvs. 
    here is why I am thinkinig it *might* work suddenly: 
    - timescale separation can be addressed by smart proposals that prioritise making
      progress wrt the slow system first, instead of exploring the fast dynamics
    - scalability will not fundamentally improve. but assuming the value function is smooth/simple
      enough, maybe we can get away with a (way) underparameterised NN? which would make it faster
      to compare against approximate value, do forward sim, everything basically.
    - maybe we again revive the idea of "tracing" the value level sets. because if we are at some v_k
      the only other states that are of interest are the ones with same value. maybe like this we can 
      keep a constant amount of data instead of an ever growing one?
    - some sort of resampling<->NN fitting loop will be inevitable, especially with larger timescale
      separation. can we make some error bound to keep everything from compounding? intuitively: value
      hessian, funnel type sketch will look the same even if we make small error. but concretely? 

    it would be cool to have a "definitive" value function upper bound, so we can quit the 
    backward integration when we notice we are worse. initially take some stable suboptimal 
    controller and train a cost-to-go approximation? can LQR tell us anything like this? 
    '''


    # do a forward simulation with controller u(x) = u*(x, lambda(x))
    def forward_sim_lqr(x0):

        def forwardsim_rhs(t, x, args):
            lam_x = 2 * P_lqr @ x
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](t, x, u)

        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True) 

        # simulate for pretty damn long
        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=20., dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
        )

        return forward_sol


    max_x0 = np.array([2, 2, 0.5, 2, 2, 0.5]) / 4
    min_x0 = -max_x0
    x0s = jax.random.uniform(key+5, shape=(200, 6), minval=min_x0, maxval=max_x0)

    forward_sols = jax.vmap(forward_sim_lqr)(x0s)

    # find the ones that enter Xf.
    forward_xf = jax.vmap(lambda sol: sol.evaluate(sol.t1))(forward_sols)
    forward_vf = jax.vmap(xf_to_Vf)(forward_xf)
    enters_Xf = forward_vf <= vfs[0]  # from points before we constructed to lie on dXf

    forward_sols_that_enter_Xf = jax.tree_util.tree_map(lambda z: z[enters_Xf], forward_sols)
    visualiser.plot_trajectories_meshcat(forward_sols_that_enter_Xf)

    # use that to start the backward solutions. 
    xfs = forward_xf[enters_Xf]

    # alternative: for each trajectory find the FIRST node in Xf, if available.
    vs = jax.vmap(jax.vmap(xf_to_Vf))(forward_sols_that_enter_Xf.ys)
    is_in_Xf = vs < vf
    idx_of_first_state_in_Xf = np.argmax(is_in_Xf, axis=1)  # is this correct??
    xfs = forward_sols_that_enter_Xf.ys[np.arange(vs.shape[0]), idx_of_first_state_in_Xf]
 
    # to see if they are really in Xf
    # vs[np.arange(vs.shape[0]), idx_of_first_state_in_Xf]


    # backward_sol = solve_backward(xfs[0])
    sols = jax.vmap(solve_backward)(xfs)

    def plot_sol(sol):

        # adapted from plot_forward_backward in ddp_optimizer
        # this works regardless of v/t reparameterisation. 
        # all the x axes are physical times as stored in sol.ys['t']
        # all interpolations are done with ODE solver "t", so whatever independent
        # variable we happen to have
        
        interp_ts = np.linspace(sol.t0, sol.t1, 5001)

        # plot the state trajectory of the forward pass, interpolation & nodes. 
        ax1 = pl.subplot(411)

        pl.plot(sol.ys['t'], sol.ys['x'], marker='.', linestyle='', alpha=1, label=problem_params['state_names'])
        interp_ys = jax.vmap(sol.evaluate)(interp_ts)
        pl.gca().set_prop_cycle(None)
        pl.plot(interp_ys['t'], interp_ys['x'], alpha=0.5)
        pl.legend()


        pl.subplot(412, sharex=ax1)
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


        # plot the eigenvalues of S from the backward pass.
        pl.subplot(413, sharex=ax1)

        # eigenvalues at nodes. 
        sorted_eigs = lambda S: np.sort(np.linalg.eig(S)[0].real)

        S_eigenvalues = jax.vmap(sorted_eigs)(sol.ys['vxx'])
        eigv_label = ['S(t) eigenvalues'] + [None] * (problem_params['nx']-1)
        pl.semilogy(sol.ys['t'], S_eigenvalues, color='C0', marker='.', linestyle='', label=eigv_label)
        # also as line bc this line is more accurate than the "interpolated" one below if timesteps become very small
        pl.semilogy(sol.ys['t'], S_eigenvalues, color='C0')  

        # eigenvalues interpolated. though this is kind of dumb seeing how the backward
        # solver very closely steps to the non-differentiable points. 
        sorted_eigs_interp = jax.vmap(sorted_eigs)(interp_ys['vxx'])
        pl.semilogy(interp_ys['t'], sorted_eigs_interp, color='C0', linestyle='--', alpha=.5)
        pl.legend()

        pl.subplot(414, sharex=ax1)
        # and raw Vxx entries. 

        vxx_entries = interp_ys['vxx'].reshape(-1, problem_params['nx']**2)
        label = ['entries of Vxx(t)'] + [None] * (problem_params['nx']**2-1)
        pl.plot(interp_ys['t'], vxx_entries, label=label, color='green', alpha=.3)
        pl.legend()



    # plot_sol(backward_sol)
    # visualiser.plot_trajectories_meshcat(backward_sol)
    pl.rcParams['figure.figsize'] = (15, 8)

    '''
    for j, _ in tqdm.tqdm(enumerate(xfs)):
        plot_sol(jax.tree_util.tree_map(itemgetter(j), sols))
        pl.savefig(f'tmp/backward_{j:03d}.png')
        pl.close('all')
    '''

    # random backward trajectory + local feedback ?!?!??!??!!!?!?
    ipdb.set_trace()


def current_weird_experiment(problem_params, algo_params):

    # get initial "easy" solution with LQR for state close to goal. 
    x0 = jax.random.normal(jax.random.PRNGKey(0), shape=(problem_params['nx'],)) * 0.1
    print(x0)

    # this trajectory should be rather boring
    # visualiser.plot_trajectories_meshcat(init_sol)

    ddp_optimizer.ddp_main(problem_params, algo_params, x0)
    ipdb.set_trace()

    
    
    



if __name__ == '__main__':

    # classic 2D quad type thing. 6D state.

    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

    # remove time arguments sometime now that we're mostly treating 
    # infinite horizon, time-invariant problems? 
    def f(t, x, u):

        # unpack for easier names
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        xdot = np.array([
            vx,
            vy,
            omega,
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr-Fl) * r / I,
        ])

        return xdot

    def l(t, x, u):
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        # state_length_scales = np.array([1, 1, np.deg2rad(10), 1, 1, np.deg2rad(45)])
        state_length_scales = np.array([1, 1, np.deg2rad(30), .5, .5, np.deg2rad(120)])
        Q = np.diag(1/state_length_scales**2)
        state_cost = x.T @ Q @ x  

        # can we just set an input penalty that is zero at hover?
        # penalise x acc, y acc, angular acc here.
        # this here is basically a state-dependent linear map of the inputs, i.e. M(x) u with M(x) a 3x2 matrix.
        # the overall input cost will be acc.T M(x).T Q M(x) acc, so for each state it is still a nice quadratic in u.
        accelerations = np.array([
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr - Fl) * r / I,
        ])

        accelerations_lengthscale = np.array([1, 1, 1])

        input_cost = accelerations.T @ np.diag(1/accelerations_lengthscale**2) @ accelerations

        return state_cost + input_cost


    def h(x):
        # irrelevant if terminal constraint or infinite horizon
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    problem_params = {
        'system_name': 'flatquad',
        'f': f,
        'l': l,
        'h': h,
        'T': 10.,  # problems come up if not float
        'nx': 6,
        'state_names': ("x", "y", "Phi", "vx", "vy", "omega"),
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'terminal_constraint': True,
        'V_max': 1000.,
        'u_eq': np.ones(2) * m * g / 2,
        'x_eq': np.zeros(6),
    }


    algo_params = {
            # 'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            # 'pontryagin_solver_adaptive': True,  always adaptivee
            'pontryagin_solver_atol': 1e-5,
            'pontryagin_solver_rtol': 1e-5,
            'pontryagin_solver_maxsteps': 4096, # nice if it is not waaay too much
            # causes it not to quit when hitting maxsteps. probably still all subsequent 
            # results will be unusable due to evaluating solutions outside their domain giving NaN
            'throw': False,  
    }

        
        
    # backward_with_hessian(problem_params, algo_params)
    current_weird_experiment(problem_params, algo_params)
    # u_star_debugging(problem_params, algo_params)
    ipdb.set_trace()

    # all_sols = rrt_sampler.rrt_sample(problem_params, algo_params)

    # visualiser.plot_trajectories_meshcat(all_sols, colormap='viridis')

    # otherwise visualiser closes immediately

    


