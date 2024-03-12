import jax
import jax.numpy as np
import diffrax

import nn_utils
import plotting_utils
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


def rnd(a, b):
    # relative norm difference. useful for checking if matrices or vectors are close
    return np.linalg.norm(a - b) / np.maximum(np.linalg.norm(a), np.linalg.norm(b))

def main(problem_params, algo_params):

    # possibly cleaner implementation of this. 
    # idea: learn V(x) for some level set V(x) <= v_k.
    # once we have that, increase v_k. 
    
    # find terminal LQR controller and value function. 
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # find a matrix mapping from the unit circle to the value level set
    # previously done with eigendecomposition -> sqrt of eigenvalues.
    # with the cholesky decomp the trajectories look about the same
    # qualitatively. it is nicer so we'll keep that. 
    # cholesky decomposition says: P = L L.T but not L.T L 
    L_lqr = np.linalg.cholesky(P_lqr)

    assert rnd(L_lqr @ L_lqr.T, P_lqr) < 1e-6, 'cholesky decomposition wrong or inaccurate'

    key = jax.random.PRNGKey(1)

    # purely random ass points for initial batch of trajectories. 
    normal_pts = jax.random.normal(key, shape=(500, problem_params['nx']))
    unitsphere_pts = normal_pts / np.linalg.norm(normal_pts, axis=1)[:, None]
    xfs = unitsphere_pts @ np.linalg.inv(L_lqr) * np.sqrt(problem_params['V_f']) * np.sqrt(2)

    # test if it worked
    V_f = lambda x: 0.5 * x.T @ P_lqr @ x
    vfs = jax.vmap(V_f)(xfs)

    assert np.allclose(vfs, problem_params['V_f']), 'wrong terminal value...'

    # define RHS of the backward system, including Vxx
    def f_extended(t, state, args=None):

        # state variables = x and quadratic taylor expansion of V.
        x   = state['x']
        v   = state['v']
        vx  = state['vx']
        vxx = state['vxx']

        nx = problem_params['nx']

        H = lambda x, u, λ: problem_params['l'](x, u) + λ.T @ problem_params['f'](x, u)

        # RHS of the necessary conditions without the hessian. 
        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)
            nx = problem_params['nx']

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # its derivatives, for propagation of Vxx.
        # fx, gx = jax.jacobian(pmp_rhs, argnums=0)(x, vx)
        # flam, glam = jax.jacobian(pmp_rhs, argnums=1)(x, vx)

        # certified this is the same. maybe more efficient? 
        full_jacobian = jax.jacobian(pmp_rhs, argnums=(0, 1))(x, vx)
        (fx, flam), (gx, glam) = full_jacobian


        # we calculate this here one extra time, could be optimised
        u_star = pontryagin_utils.u_star_2d(x, vx, problem_params)

        # calculate all the RHS terms
        v_dot = -problem_params['l'](x, u_star)
        x_dot, vx_dot = pmp_rhs(x, vx)
        vxx_dot = gx + glam @ vxx - vxx @ fx - vxx @ flam @ vxx 


        # and pack them in a nice dict for the state.
        state_dot = dict()
        state_dot['x'] = x_dot
        state_dot['t'] = 1.
        state_dot['v'] = v_dot
        state_dot['vx'] = vx_dot
        state_dot['vxx'] = vxx_dot

        return state_dot

    def solve_backward(x_f):

        # another 'advantage' of including the hessian is that now the ODE solver
        # actually takes the trouble of stepping rather precisely to the active
        # set changes.

        # still unsure about these factors of 2 or 1/2 or 1
        # but results look OK, Vxx stays constant-ish in the linear-ish region.

        # this probably comes from the 1/n! in the taylor expansion, which will be:
        # v(x+dx) = v(x) + v_x(x) dx + 1/2 dx.T vxx(x) dx

        # but P_lqr gives the value direcly... or does it? at least here the usual 
        # continuous time ricatti solution is given like that: 
        #     https://stanford.edu/class/ee363/lectures/clqr.pdf
        
        # so if V_f(x) = x.T P_lqr x, then its hessian is 2 P_LQR! 

        # according to usual differentiation rules:
        v_f = 0.5 * x_f.T @ P_lqr @ x_f
        vx_f = P_lqr @ x_f
        vxx_f = P_lqr

        # summarise to maybe avoid confusion. P = P_lqr. 
        # the LQR value function is x.T P x, therefore its hessian is 2P. 
        # however we initialise this function with state['vxx'] = P, the half hessian.

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

        backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=-4., dt0=-0.1, y0=state_f,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
        )

        return backward_sol


    # do a forward simulation with controller u(x) = u*(x, lambda(x))
    def forward_sim_lqr(x0):

        def forwardsim_rhs(t, x, args):
            lam = P_lqr @ x
            u = pontryagin_utils.u_star_2d(x, lam, problem_params)
            return problem_params['f'](x, u)

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



    '''
    max_x0 = np.array([2, 2, 0.5, 2, 2, 0.5]) / 4
    min_x0 = -max_x0
    x0s = jax.random.uniform(key+5, shape=(200, 6), minval=min_x0, maxval=max_x0)
    forward_sols = jax.vmap(forward_sim_lqr)(x0s)

    # find the ones that enter Xf.
    forward_xf = jax.vmap(lambda sol: sol.evaluate(sol.t1))(forward_sols)
    forward_vf = jax.vmap(V_f)(forward_xf)
    enters_Xf = forward_vf <= vfs[0]  # from points before we constructed to lie on dXf

    forward_sols_that_enter_Xf = jax.tree_util.tree_map(lambda z: z[enters_Xf], forward_sols)
    # visualiser.plot_trajectories_meshcat(forward_sols_that_enter_Xf)

    # use that to start the backward solutions. 
    xfs = forward_xf[enters_Xf]

    test_rhs = True
    if test_rhs:
        x_f = xfs[0]
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

        oup = f_extended(0., state_f, None)


    # alternative: for each trajectory find the FIRST node in Xf, if available.
    vs = jax.vmap(jax.vmap(V_f))(forward_sols_that_enter_Xf.ys)
    is_in_Xf = vs < vf
    idx_of_first_state_in_Xf = np.argmax(is_in_Xf, axis=1)  # is this correct??
    xfs = forward_sols_that_enter_Xf.ys[np.arange(vs.shape[0]), idx_of_first_state_in_Xf]
 
    # to see if they are really in Xf
    # vs[np.arange(vs.shape[0]), idx_of_first_state_in_Xf]
    '''


    # backward_sol = solve_backward(xfs[0])
    # sols = jax.vmap(solve_backward)(xfs)

    # visualiser.plot_trajectories_meshcat(sols_orig, color=(.9, .7, .7))
    # visualiser.plot_trajectories_meshcat(sols, color=(.7, .7, .9))




    def plot_sol(sol):

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


    def plot_us(sols, rotate=True, c='C0'):

        # plot all the u trajectories of a vmapped solutions object.

        # we flatten them here -- the inf padding breaks up the plot nicely
        all_xs = sols.ys['x'].reshape(-1, 6)
        all_lams = sols.ys['vx'].reshape(-1, 6)
        us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(all_xs, all_lams, problem_params)

        diff_and_sum = np.array([[1, -1], [1, 1]]).T
        if rotate: 
            us = us @ diff_and_sum
            pl.xlabel('u0 - u1')
            pl.ylabel('u0 + u1')
        else:
            pl.xlabel('u0')
            pl.ylabel('u1')


        pl.plot(us[:, 0], us[:, 1], alpha=0.1, marker='.', c=c)



    def plot_taylor_sanitycheck(sol):

        # see if the lambda = vx and S = vxx are remotely plausible. 

        pl.figure(f'taylor sanity check')

        # does a part of both of the above functions plus the retrieval of the solutions.
        # plot v(x(t)) alone
        interp_ts = np.linspace(sol.t0, sol.t1, 1000)
        interp_ys = jax.vmap(sol.evaluate)(interp_ts)
        pl.plot(sol.ys['t'], sol.ys['v'], linestyle='', marker='.', color='C0', alpha=0.4)
        pl.plot(interp_ts, interp_ys['v'], linestyle='--', label='v(x(t))', color='C0', alpha=0.4)

        us = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(sol.ys['x'], sol.ys['vx'], problem_params)
        fs = jax.vmap(problem_params['f'], in_axes=(0, 0))(sol.ys['x'], us)

        # the total time derivative of v we're after
        v_ts = jax.vmap(np.dot)(sol.ys['vx'], fs)

        # for each of the derivatives, plot a small line. 

        def line_params(t, v, v_t):

            line_len = 0.1
            diffvec_unscaled = np.array([1, v_t])
            diffvec = line_len * diffvec_unscaled / np.linalg.norm(diffvec_unscaled)

            # nan in between to break up lines. 
            xs = np.array([t-diffvec[0], t+diffvec[0], np.nan])
            ys = np.array([v-diffvec[1], v+diffvec[1], np.nan])
            return xs, ys

        xs, ys = jax.vmap(line_params)(sol.ys['t'], sol.ys['v'], v_ts)

        pl.plot(xs.flatten(), ys.flatten(), label='d/dt v(x(t))', color='C1', alpha=0.5)


        # now the hessians, def the hardest part...

        def line_params_hessian(ode_state, f_value):

            t_len = 0.02
            n = 20

            dts = np.concatenate([np.linspace(-t_len, t_len, n), np.array([np.nan])])
            dxs = np.vstack([np.linspace(-f_value * t_len, +f_value * t_len, n), np.nan * np.ones(problem_params['nx'])])

            xs = ode_state['x'] + dxs 
            ts = ode_state['t'] + dts

            vs_taylor = jax.vmap(lambda dx: ode_state['v'] + ode_state['vx'] @ dx + 0.5 * dx.T @ ode_state['vxx'] @ dx)(dxs)

            return ts, vs_taylor

        def line_params_twice_hessian(ode_state, f_value):

            t_len = 0.02
            N = 20

            dts = np.concatenate([np.linspace(-t_len, t_len, N), np.array([np.nan])])
            dxs = np.vstack([np.linspace(-f_value * t_len, +f_value * t_len, N), np.nan * np.ones(problem_params['nx'])])

            xs = ode_state['x'] + dxs 
            ts = ode_state['t'] + dts

            vs_taylor = jax.vmap(lambda dx: ode_state['v'] + ode_state['vx'] @ dx + 0.5 * 2 * dx.T @ ode_state['vxx'] @ dx)(dxs)

            return ts, vs_taylor

        ts, vs = jax.vmap(line_params_hessian, in_axes=(0, 0))(sol.ys, fs)

        pl.plot(ts.flatten(), vs.flatten(), alpha=.5, color='C2', label='hessian')

        # also interesting: 
        pl.figure()
        idx = 20
        state_idx = jax.tree_util.tree_map(itemgetter(idx), sol.ys)
        ts, vs = line_params_hessian(state_idx, fs[idx])
        pl.plot(ts, vs, label='v taylor')
        ts, vstwice = line_params_twice_hessian(state_idx, fs[idx])
        pl.plot(ts, vstwice, label='v taylor but twice hessian')
        pl.scatter(state_idx['t'], state_idx['v'], color='C0')
        pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'], label='v sol')
        pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'] - vs, label='diff')
        pl.plot(ts, jax.vmap(sol.evaluate)(ts)['v'] - vstwice, label='diff with twice hessian')


        pl.legend()
        # ipdb.set_trace()

        '''
        pl.figure()

        # and another one. this time with the inputs. 
        # we know: u(x) = u*(x, lambda(x))
        # so, dudx = u*_x + u*_lambda lambda_x 

        def u_and_dudt(ode_state, f_value):

            u = pontryagin_utils.u_star_2d(ode_state['x'], ode_state['vx'], problem_params)
            # in one go. thanks jax :) 
            # insert linearisation of lambda(x) to get accurate total derivative. 
            # linearisation about ode_state['x'], evaluation at x. 
            du_dx = jax.jacobian(lambda x: pontryagin_utils.u_star_2d(x, ode_state['vx'] + ode_state['vxx'] @ (x - ode_state['x']), problem_params))(ode_state['x'])
            dx_dt = f_value

            du_dt = du_dx @ dx_dt 

            return u, du_dt

            # now for the plot x axisode_state['t'] + np.ones_like(dudt) * 0.1 not system state x
            # line_len_x = 0.05
            # xs = ode_state['t'] + line_len_x * np.array([-1, 1, np.nan])

            # ys = u + du_dt @ 


            ipdb.set_trace()


        # line_params(state_idx, fs[idx])
        us, dudts = jax.vmap(u_and_dudt, in_axes=(0, 0))(sol.ys, fs)

        us_interp = jax.vmap(pontryagin_utils.u_star_2d, in_axes=(0, 0, None))(interp_ys['x'], interp_ys['vx'], problem_params)
        fs_interp = jax.vmap(problem_params['f'], in_axes=(0, 0))(interp_ys['x'], us_interp)

        us_interp, dudts_interp = jax.vmap(u_and_dudt, in_axes=(0, 0))(interp_ys, fs_interp)
        ipdb.set_trace()


        pl.plot(interp_ts, us_interp, alpha=0.5, label=('u0', 'u1'))
        pl.gca().set_prop_cycle(None)
        pl.plot(interp_ts, dudts_interp, label=('du0/dt', 'du1/dt')) 
        '''

        # pl.plot(sol.ys['t'], us, alpha=0.5, label=('u0', 'u1'))
        # pl.gca().set_prop_cycle(None)
        # pl.plot(sol.ys['t'], dudts, label=('du0/dt', 'du1/dt')) 
        # ipdb.set_trace()


    sols_orig = jax.vmap(solve_backward)(xfs)

    '''
    for j in range(10):
        pl.figure(str(j))
        plot_sol(jax.tree_util.tree_map(itemgetter(j), sols_orig))
    pl.show()
    '''



    # ipdb.set_trace()

    # choose initial value level. 
    v_k = np.inf  # fullest gas
    v_k = 50  # full gas
    v_k = 1000 * problem_params['V_f']
    print(f'target value level: {v_k}')

    # extract corresponding data points for NN fitting. 
    # this is a multi dim bool index! indexing a multidim array with it effectively flattens it.
    bool_train_idx = sols_orig.ys['v'] < v_k
    train_ode_states = jax.tree_util.tree_map(lambda node: node[bool_train_idx], sols_orig.ys)

    print(f'dataset size: {bool_train_idx.sum()} (= {bool_train_idx.mean()*100:.2f}%)')

    v_nn = nn_utils.nn_wrapper(
        input_dim=problem_params['nx'],
        layer_dims=algo_params['nn_layerdims'],
        output_dim=1
    )


    normaliser = nn_utils.data_normaliser(train_ode_states)
    nn_xs_n, nn_ys_n = normaliser.normalise_all(train_ode_states)
    ys_n = normaliser.normalise_all_dict(train_ode_states)
    


    # nn_xs_normalised = (nn_xs - x_means)/x_stds
    '''
    params, oups = v_nn.init_and_train(
        key, nn_xs_n, nn_ys_n, algo_params
    )
    '''

    # new sobolev training method. 
    pl.rcParams['figure.figsize'] = (16, 9)
    vxx_weights = np.concatenate([np.zeros(1,), np.logspace(-2, 4, 1000)])
    hessian_rnds = np.zeros_like(vxx_weights)
    final_training_errs = np.zeros((vxx_weights.shape[0], 3))
    for i, vxx_weight in tqdm.tqdm(enumerate(vxx_weights)):

        algo_params['nn_sobolev_weights'] = algo_params['nn_sobolev_weights'].at[2].set(vxx_weight)

        # liek if u use same kei everytiem
        init_key, key = jax.random.split(key)
        params_sobolev = v_nn.nn.init(init_key, np.zeros(problem_params['nx']))

        train_key, key = jax.random.split(key)
        params_sobolev, oups_sobolev = v_nn.train_sobolev(train_key, ys_n, params_sobolev, algo_params)

        # the value function back in the "unnormalised" domain, ie. the actual state space. 
        v_nn_unnormalised = lambda params, x: normaliser.unnormalise_v(v_nn(params, normaliser.normalise_x(x)))
        # and its hessian at 0 just as a sanity check.
        hess_vnn_unnormalised = lambda params, x: jax.hessian(v_nn_unnormalised, argnums=1)(params, x).squeeze()
        H0 = hess_vnn_unnormalised(params_sobolev, np.zeros(6,))
        
        # compute some statistics :) 
        hess_rnd = rnd(H0, P_lqr)
        loss_means = oups_sobolev['loss_terms'][-100:, :].mean(axis=0)
        # print(hess_rnd)
        hessian_rnds = hessian_rnds.at[i].set(hess_rnd)
        final_training_errs = final_training_errs.at[i, :].set(loss_means)

        '''
        pl.figure(f'vxx weight: {vxx_weight:.8f}')
        pl.suptitle(f'vxx weight: {vxx_weight:.8f}')
        pl.loglog(oups_sobolev['loss_terms'], label=('v', 'vx', 'vxx'), alpha=.5)
        pl.grid('on')
        pl.ylim([1e-7, 1e1])
        pl.legend()
        figpath = f'./tmp/losses_{i:04d}_{vxx_weight:.3f}.png'
        pl.savefig(figpath)
        print(f'saved "{figpath}"')
        pl.close('all')
        '''

    pl.plot(vxx_weights, hessian_rnds, linestyle='', marker='.', label='hessian rel norm diff at x=0')
    pl.plot(vxx_weights, final_training_errs, linestyle='', marker='.', label=('v', 'vx', 'vxx'))
    pl.legend()
    pl.show()

    ipdb.set_trace()


    # need to adapt that one still. 
    # plotting_utils.plot_nn_train_outputs(oups)

    params = params_sobolev

    def plot_trajectory_vs_nn(idx):

        sol = jax.tree_util.tree_map(itemgetter(idx), sols_orig)

        pl.figure()
        ax = pl.subplot(211)

        interp_ts = np.linspace(sol.t0, sol.t1)

        xs = sol.ys['x']
        ts = sol.ys['t']
        vs = sol.ys['v']
        interp_ys = jax.vmap(sol.evaluate)(interp_ts)

        pl.plot(interp_ts, interp_ys['v'], alpha=.5, label='trajectory v(x(t))', c='C0')
        pl.plot(ts, vs, alpha=.5, linestyle='', marker='.', c='C0')
        lqr_vs = jax.vmap(V_f)(interp_ys['x'])
        pl.plot(interp_ts, lqr_vs, label='LQR V(x(t))', color='C1', alpha=.5)

        # same with the NN
        xn = jax.vmap(normaliser.normalise_x)(interp_ys['x'])
        nn_vs_n = jax.vmap(v_nn.nn.apply, in_axes=(None, 0))(params, xn)
        pl.plot(interp_ts, jax.vmap(normaliser.unnormalise_v)(nn_vs_n), c='C1', label='NN v(x(t))')
        pl.legend()

        # and now the same for vx
        pl.subplot(212, sharex=ax)

        vxs = sol.ys['vx']
        pl.plot(interp_ts, interp_ys['vx'], alpha=.5, label='trajectory vx(x(t))', c='C0')
        pl.plot(ts, vxs, alpha=.5, linestyle='', marker='.', c='C0')

        nn_vxs_n = jax.vmap(jax.jacobian(v_nn.nn.apply, argnums=1), in_axes=(None, 0))(params, xn).squeeze()
        nn_vxs = jax.vmap(normaliser.unnormalise_vx)(nn_vxs_n)
        pl.plot(interp_ts, nn_vxs, label='NN v_x(x(t))', c='C1')

        lqr_vxs = jax.vmap(jax.jacobian(V_f))(interp_ys['x'])
        pl.plot(interp_ts, lqr_vxs, label='lqr Vx', c='C2', alpha=.5)
        pl.legend()

    plot_trajectory_vs_nn(12)
    pl.show()

    # do a forward sim starting from 0 just to see what happens. 
    def forward_sim_nn(x0):

        def forwardsim_rhs(t, x, args):
            lam_x = jax.jacobian(v_nn_unnormalised, argnums=1)(params, x).squeeze()
            # lam_x = P_lqr @ x
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](x, u)

        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True) 

        # simulate for pretty damn long
        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=10., dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
        )

        return forward_sol


    x0s = jax.random.normal(jax.random.PRNGKey(0), shape=(200, 6)) * .1

    sols = jax.vmap(forward_sim_nn)(x0s)
    visualiser.plot_trajectories_meshcat(sols)

    ipdb.set_trace()

    for k in range(5):

        '''
        basic idea for value step scheduling:
         - set "target" v_k+1. intuitively: how far can we simulate backward without the sensitivity problem
           completely messing us up? something like:
               (small multiple of) fastest time constant * minimal dv/dt
           the fastest time constant we can assume we know, take from linearisation or estimate from previously generated data
           dv/dt = -l(x, u), and the bottleneck are the trajectories which stay on low l for some time. 
           probably we can just take the smallest l(x, u) encountered in the last "value band" V_k \\ V_k-1. 
           (if we don't exclude the sublevel set V_k-1 then the lowest will always be l=0 at equilibrium)
         - sample lots of states in the value band V_k+1 \\ V_k. This hinges on extrapolation! we should make sure that
           this set is bounded, maybe by "pushing up" the value function slightly so the extrapolated one is an overestimate? 
           or if we don't ensure it is bounded, make it work somehow otherwise
         - simulate forward. there should be an easy "upper bound" on the time duration of these trajectories until we enter V_k
           based on also the minimal dv/dt. simulate for that amount of time, if not in V_k, ignore that point. 
         - from remaining points where forward sim landed (maybe take the point where trajectory entered V_k? or lowest uncertainty?)
           do backward PMP. 
         - fit NN again with new data, maybe enlarging the training dataset in smaller value levelset steps. 
         - evaluate posterior uncertainty for many "test points" in V_k+1 \\ V_k. Estimate somehow the largest value step v_k+1
           we can accept without having large uncertainty in V_k+1. Simple way: just set V_k+1 = largest value for which no test
           point has too large uncertainty. Probably there is something nicer though based on statistics and shit. 
         - repeat. 

        generally we already have a dataset which goes beyond the current value level set. we might also try an initial step where
        we extend all current points up to v_k+1 and fit the NN with that incomplete data. then probably the forward simulations
        will be closer to optimal and we get better sampling distributions. 

        '''
        # main loop, first draft. 
        vj_goal = vj_prev * 1.5

        # obtain solutions. 
        sols_orig = jax.vmap(solve_backward)(xfs)

        


        # train NN by using that data, maybe including an actually continuous
        # sweep of v_k -> v_k+1 by adding data during training. 
        raise NotImplementedError

        # find out up to which value level the NN is actually accurate
        # (i.e. has low posterior uncertainty)
        raise NotImplementedError


        # vk = largest v for which we have low enough uncertainty in sublevel set


    # plot only the N_plot ones with lowest initial cost-to-go. 
    v0s = jax.vmap(lambda s: s.evaluate(s.t1)['v'])(sols_orig) 
    N_plot = 200

    # the highest v0 we're going to plot
    v0_cutoff = np.percentile(v0s, N_plot/v0s.shape[0] * 100)

    bool_plot_idx = v0s <= v0_cutoff

    sols_plot = jax.tree_util.tree_map(lambda node: node[bool_plot_idx], sols_orig)
    sols_notplot = jax.tree_util.tree_map(lambda node: node[~bool_plot_idx], sols_orig)

    visualiser.plot_trajectories_meshcat(sols_plot)

    pl.plot(sols_plot.ys['t'].flatten(), sols_plot.ys['v'].flatten(), alpha=.2, c='C0', label='plotted sols')
    pl.plot(sols_notplot.ys['t'].flatten(), sols_notplot.ys['v'].flatten(), alpha=.2, c='C1', label='not plotted sols')


    '''
    for j in range(5):
        pl.figure(str(j))
        plot_sol(jax.tree_util.tree_map(itemgetter(j), sols_orig))
    '''

    # plot_taylor_sanitycheck(jax.tree_util.tree_map(itemgetter(0), sols_orig))
    pl.show()
    ipdb.set_trace()



