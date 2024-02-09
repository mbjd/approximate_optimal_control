import jax
import jax.numpy as np
import diffrax

import matplotlib.pyplot as pl

import ipdb
import tqdm

import pontryagin_utils
import visualiser

# attept at implementing continuous-time DDP, a bit like: 
# - https://dl.acm.org/doi/pdf/10.1145/3592454 (Hutter) but without the parameterised stuff
# - https://arxiv.org/pdf/2107.04507.pdf (Sun) but without the opponent/disturbance
# - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10156422 
# ofc later we can add any of those components back maybe hopefully. 


def ddp_main(problem_params, algo_params, init_sol):

    # problem_params, algo_params: the usual things
    # init_sol: the initial guess trajectory. 
    
    # what OCP are we trying to solve here? 

    T = 20 # long-ish horizon to approximate infinity. later adaptively increase.

    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']


    # do this again here. should we put this in one of the params dicts? 
    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'

    # should work the same for any unit vector (xT is disregarded)
    unitvec_example = np.eye(nx)[0] 
    xf_example = np.linalg.solve(Phalf, unitvec_example) * 0.01
    v_f = xf_example.T @ P_lqr @ xf_example
    # with this we have defined Xf = {x: x.T P_lqr x <= vT}
    


    # the usual Hamiltonian. t argument irrelevant.
    H = lambda x, u, λ: l(0., x, u) + λ.T @ f(0., x, u)

    # the pre-optimised Hamiltonian. 
    def H_opt(x, λ): 
        # replace with relevant inner convex optimisation
        u_star = pontryagin_utils.u_star_2d(x, λ, problem_params)
        return H(x, u_star, λ)
            
    # do we have problems if this solution was calculated backwards? 
    # -> probably not, because evaluation works the same in any case.
    forward_sol = init_sol
    x0 = forward_sol.evaluate(0.)[0:nx]  # this stays the same in the entire function

    # this is to handle case where t0>t1 (if solution was obtained backwards)
    t0, tf = sorted([forward_sol.t0, forward_sol.t1])  

    # write the whole iteration in a scan-type loop? 
    # which pass should we do first in each loop iteration? 
    # backward then forward? 
    # + no special case for initial trajectory

    # main iterations 
    for i in range(10): 

        # backward pass: Find the local value function expansion around the trajectory.

        def backwardpass_rhs(t, state, args):

            # v = cost to go, lam = costate, S = value hessian

            v, lam, S = state  
            prev_forward_sol = args

            # current state and optimal input *according to backward-pass costate*
            xbar = prev_forward_sol.evaluate(t)[0:nx]
            u_star = pontryagin_utils.u_star_2d(xbar, lam, problem_params)

            # this partial derivative we can do by hand :) H is linear in lambda.
            H_lam_fct = jax.jacobian(H, argnums=2)
            H_lambda = f(t, xbar, u_star)

            # do NOT confuse these! I have the power
            # of multivariate analysis and LaTeX on my side!
            H_x_fct = jax.jacobian(H, argnums=0)
            H_x = H_x_fct(xbar, u_star, lam)

            # TODO do one of these: 
            # a) evaluate RHS that was used to generate the forward trajectory
            # b) approximate with d/dt of interpolated solution

            # this is the entirety of option b)
            dot_xbar = prev_forward_sol.derivative(t)[0:nx]

            option_A = False
            if option_A:
                ### # this is a rough draft of option a) 
                ### # would also be cool if we could make diffrax store an interpolation
                ### # of some "output" variables so we can put the input there, which would
                ### # be handy for all sorts of things.

                ### lam_prev = None # get this from last backward solution? 
                ### # maybe that could be done with the SaveAt object in diffrax? 
                ### # https://docs.kidger.site/diffrax/api/saveat/
                ### # i guess with that we can also just save the input as a function of 
                ### # state. But it will probably be evaluated twice...
                ### 
                ### # we could require that the forward solution be constructed with an approximate 
                ### # optimal input given by some function lambda(xbar). this would encompass the repeated
                ### # forward passes, and initialisation with some global approximate V_x function. 
                ### # then we can do this: https://github.com/patrick-kidger/diffrax/issues/60
                ### # and save the costate during the forward simulation. because it is smooth, at 
                ### # least along forward trajectories, we expect that it works well with interpolation. 
                ### # then we could just get it here as part of the prev_forward_sol object probably...

                ### # just for a small test
                # TODO take actual value
                lam_prev = lam * (1 + jax.random.normal(jax.random.PRNGKey(9130), shape=lam.shape) * 0.1)

                # this is the input from the previous forward simulation
                u_prev = pontryagin_utils.u_star_2d(xbar, lam_prev, problem_params)

                # to get previous RHS in this way...
                # beware of possible - for backward integration...
                dot_xbar_1  = f(0., xbar, u_prev)

            # according to my own derivations
            # am I now really the type of guy who finds working this out myself easier 
            # than copying formulas from existing papers? anyway...

            # the money shot, derivation & explanations in dump, section 3.5.*
            v_dot = -l(t, xbar, u_star) + lam.T @ (dot_xbar - H_lambda)
            # maybe ^ equivalent to -l(t, xbar, u_prev?)

            lam_dot = -H_x + S @ (dot_xbar - H_lambda)

            # other derivation in idea dump, second try, DOC inspired. 
            # all second partial derivatives of the pre-optimised hamiltonian. 
            H_opt_xx = jax.hessian(H_opt, argnums=0)(xbar, lam)
            H_opt_xlam = jax.jacobian(jax.jacobian(H_opt, argnums=0), argnums=1)(xbar, lam)
            # H_opt_lamx = jax.jacobian(jax.jacobian(H_opt, argnums=1), argnums=0)(xbar, lam)
            H_opt_lamlam = jax.hessian(H_opt, argnums=1)(xbar, lam)

            # purely along characteristic curve, no correction term here. 
            S_dot = -H_opt_xx - H_opt_xlam @ S - (H_opt_xlam @ S).T - S.T @ H_opt_lamlam @ S

            # and the other derivatives too just to sanity check the different derivations....
            H_xx_hess = jax.hessian(H, argnums=0)(xbar, u_star, lam)

            H_xx = jax.jacobian(H_x_fct, argnums=0)(xbar, u_star, lam)
            H_xu = jax.jacobian(H_x_fct, argnums=1)(xbar, u_star, lam)  
            H_xlam = jax.jacobian(H_x_fct, argnums=2)(xbar, u_star, lam)
            # H_ux == H_xu.T etc

            H_lamx = jax.jacobian(H_lam_fct, argnums=0)(xbar, u_star, lam)
            H_lamu = jax.jacobian(H_lam_fct, argnums=1)(xbar, u_star, lam)
            H_lamlam = jax.jacobian(H_lam_fct, argnums=2)(xbar, u_star, lam)
            
            u_star_x = jax.jacobian(pontryagin_utils.u_star_2d, argnums=0)(xbar, lam, problem_params)
            u_star_lam = jax.jacobian(pontryagin_utils.u_star_2d, argnums=1)(xbar, lam, problem_params)

            # the matrix governing the evolution of (\delta x, \delta \lambda) in the linearised 
            # pontryagin BVP (see notes in idea dump)
            Alin = np.block([[H_lamx, H_lamlam], [-H_xx, -H_xlam]]) + np.vstack([H_lamu, -H_xu]) @ np.hstack([u_star_x, u_star_lam])

            # because the derivation uses those to shorten notation...
            A11 = Alin[0:nx, 0:nx]
            A12 = Alin[0:nx, nx:]
            A21 = Alin[nx:, 0:nx]
            A22 = Alin[nx:, nx:]

            s = lam
            Sdotnew = A21 + A22 @ S - S @ A11 - S @ A12 @ S
            sdotnew = (-S @ A12 + A22) @ s


            # both look pretty close to each other, and also both symmetric up to relative error of about 1e-8 :) 
            S_dot_rel_asymmetry = np.linalg.norm(S_dot - S_dot.T) / np.linalg.norm(S_dot)
            Sdotnew_rel_asymmetry = np.linalg.norm(Sdotnew - Sdotnew.T) / np.linalg.norm(Sdotnew)
            Sdot_rel_diff = np.linalg.norm(Sdotnew - S_dot) / np.linalg.norm(Sdotnew)

            # however, sdotnew and lam_dot are markedly different. 
            # this is despite the two relevant directions (d/dt xbar and H_lambda) being almost the same (worst ratio .997)
            # are they even comparable? the early, characteristics type derivation only has a \lambda variable. 
            # in the new DOC derivation, we have lambda AND \delta lambda, the latter one being what we are finding in the linear BVP. 
            # still a bit of understanding left to be gained here. 

            # REALLY unsure whether I got this correctly from the paper. e.g. \hat H is supposed to be a matrix right? 
            # f_x = jax.jacobian(f, argnums=1)(0., xbar, u_star)
            # Sdot_DOC = H_opt_xx - f_x.T @ S - S @ f_x - S @ H_opt(xbar, lam) @ S
            # sdot_DOC = H_opt_x


            ipdb.set_trace()

            # S_dot slightly off of being symmetric unfortunately. 
            # so, artificially "symmetrize" them here? or baumgarte type stabilisation? 
            # let's do nothing for the moment and wait until it bites us in the ass :) 

            return (v_dot, lam_dot, S_dot)

        term = diffrax.ODETerm(backwardpass_rhs)
        
        # V_inf(x) = x.T @ P_lqr @ x 
        # Standard differentiation rules say:
        # V_inf_x(x) = 2 P_lqr @ x
        # V_inf_xx(x) = 2 P_lqr
        
        # terminal state from forward sim.
        xf = forward_sol.evaluate(tf)[0:nx]

        # terminal conditions given by taylor expansion of terminal value
        v_T = xf.T @ P_lqr @ xf
        lam_T = 2 * P_lqr @ xf
        S_T = 2 * P_lqr 

        # small additional tolerance for numerical inaccuracies
        if v_T >= v_f * 1.001: 
            # better to be a bit of a bünzli about this :) 
            raise ValueError('Terminal value function V_f undefined outside terminal region X_f')

        init_state = (v_T, lam_T, S_T)

        # test the RHS function. 
        out = backwardpass_rhs(tf, init_state, forward_sol)

        # TODO also get the specifics (adaptive controller, max steps, SaveAt) right. 

        # same tolerance parameters used in pure unguided backward integration of whole PMP
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
        # 10x relaxed tolerance
        step_ctrl = diffrax.PIDController(rtol=10*algo_params['pontryagin_solver_rtol'], atol=10*algo_params['pontryagin_solver_atol'])
        saveat = diffrax.SaveAt(steps=True, dense=True)

        backwardpass_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=tf, t1=t0, dt0=-0.1, y0=init_state,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = 1024,
            args = forward_sol
        )
        # WE GOT OUR FIRST RUN! and it actually produces numbers and not NaNs and the like. 
        # let us investigate if they make sense. 
        investigate = True
        if investigate:

            # compare values from backward pass and "forward" sol
            pl.figure()
            pl.plot(backwardpass_sol.ts, backwardpass_sol.ys[0], label='v from backward pass')
            pl.plot(forward_sol.ts, forward_sol.ys[:, -1], label='v from "forward" pass')
            pl.legend()

            # compare costates. 
            pl.figure()
            pl.plot(backwardpass_sol.ts, backwardpass_sol.ys[1], label='costate from backward pass')
            pl.gca().set_prop_cycle(None)
            pl.plot(forward_sol.ts, forward_sol.ys[:, nx:2*nx], label='costate from "forward" pass', linestyle='--')
            pl.legend()


            pl.figure()
            # now the interesting stuff, stats about the S matrix. 
            pl.subplot(131)
            pl.xlabel('S matrix - raw and interpolated entries')
            pl.plot(backwardpass_sol.ts, backwardpass_sol.ys[2].reshape(-1, nx*nx), marker='.', linestyle='')
            interp_ts = np.linspace(backwardpass_sol.t0, backwardpass_sol.t1, 5001) # continuous time go brrrr
            interp_ys = jax.vmap(backwardpass_sol.evaluate)(interp_ts)
            pl.gca().set_prop_cycle(None)
            pl.plot(interp_ts, interp_ys[2].reshape(-1, nx*nx))
            '''

            these S entries look kind of wild but then again so does the state
            trajectory. so maybe nothing to worry about. need to see other
            trajectories and do numerical sanity checks by using the local
            value function in feedback controller and looking at those
            trajectories. 

            '''

            pl.subplot(132)
            pl.xlabel('norm(S - S.T)')
            batch_matrix_asymmetry = jax.vmap(lambda mat: np.linalg.norm(mat - mat.T))
            pl.plot(backwardpass_sol.ts, batch_matrix_asymmetry(backwardpass_sol.ys[2]), marker='.', linestyle='')
            pl.gca().set_prop_cycle(None)
            pl.plot(interp_ts, batch_matrix_asymmetry(interp_ys[2]))

            '''
            
            the asymmetry does grow to about does grow to about .00018. I'd
            say its nothing to worry about right now. whenever we need a
            precisely symmetric matrix S we can just use (S + S.T)/2. If it
            becomes a problem for longer horizons let's call up our boi
            Baumgarte

            '''

            pl.subplot(133)
            pl.xlabel('eigs(S)')
            batch_real_eigvals = jax.vmap(lambda mat: np.sort(np.linalg.eig(mat)[0].real))
            pl.semilogy(backwardpass_sol.ts, batch_real_eigvals(backwardpass_sol.ys[2]), marker='.', linestyle='')
            pl.gca().set_prop_cycle(None)
            pl.semilogy(interp_ts, batch_real_eigvals(interp_ys[2]))

            '''
            mostly this seems to stay positive definite, although there was ONE case where one eigenvalue 
            dropped to like -0.1 or something, so clearly negative but not overly so, and only for like .1 or .2 seconds. 
            lost the prng seed responsible for it due to a crash :( maybe keep this in mind for future debugging. 
            '''


            pl.show()
            # ipdb.set_trace()


        # Then, first thing tomorrow: 
        # - work out the kinks above so it actually runs. (done)
        # - find some actually optimal trajectory by PMP backwards integration (done)
        # - give it to this as an initial guess (done)
        # - verify that the backward pass is consistent w/ this info
        #   (we should have the backward pass costate equal to the initial costate, 
        #   and if we forward sim again, obtain the same trajectory again)

        # looks good until now :) backward pass very nice. 
        # let us implement a forward pass here. 

        # this means basically just a closed loop simulation, with the controller 
        #    u(t, x) = argmin_H(t, x, lambda(t, x))
        # with lambda(t, x) given by the taylor approximation from the backward pass. 

        def forwardpass_rhs(t, state, args):

            # here we only have the actual system state for a change
            x = state  
            # need both backward and forward pass solutions. 
            prev_forward_sol, backwardpass_sol = args 

            # this stacked vector forward sol is not too elegant, should probably
            # introduce a tuple or even dict state to access solution more clearly
            xbar = prev_forward_sol.evaluate(t)[0:nx] 
            dx = x - xbar
            v_xbar, lam_xbar, S_xbar = backwardpass_sol.evaluate(t)
            # this defines a local quadratic value function: 
            # v(xbar + dx) = v + lam.T dx + 1/2 dx.T S dx
            # we need the first gradient of this quadratic value function. 
            # lambda(xbar + dx) = 0 + lam.T + dx.T S  (or its transpose whatever)
            lam_x = lam_xbar + S_xbar @ dx

            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            dot_x = f(t, x, u)

            return dot_x

        # setup the ODE solver call. 
        term = diffrax.ODETerm(forwardpass_rhs)

        # same tolerance parameters used in pure unguided backward integration of whole PMP
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
        saveat = diffrax.SaveAt(steps=True, dense=True) 

        forwardpass_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=t0, t1=tf, dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
            args = (forward_sol, backwardpass_sol)
        )

        if investigate:
            # we got a solution!!! plot it. 
            # specifically, compare the new forwardpass_sol with the previous, forward_sol. 

            # here plot just the forward pass solution (solid) and the initial guess (translucent)
            # they should match each other very closely. 
            interp_ts = np.linspace(t0, tf, 501)


            label = ['forward pass: ' + name for name in problem_params['state_names']]
            pl.plot(forwardpass_sol.ts, forwardpass_sol.ys, linestyle='', marker='.')
            pl.gca().set_prop_cycle(None)
            pl.plot(interp_ts, jax.vmap(forwardpass_sol.evaluate)(interp_ts), label=label)

            label = ['init guess (optimal already) ' + name for name in problem_params['state_names']]
            pl.gca().set_prop_cycle(None)
            pl.plot(forward_sol.ts, forward_sol.ys[:, 0:nx], linestyle='', marker='.', alpha=.5)
            pl.gca().set_prop_cycle(None)
            pl.plot(interp_ts, jax.vmap(forward_sol.evaluate)(interp_ts)[:, 0:nx], label=label, linestyle='--', alpha=.5)
            pl.legend()

            # here we plot N different forward passes from different, but close initial conditions. 
            # then we visualise with meshcat :) 
            def forward_sim(x0):
                forwardpass_sol = diffrax.diffeqsolve(
                    term, diffrax.Tsit5(), t0=t0, t1=tf, dt0=0.1, y0=x0,
                    stepsize_controller=step_ctrl, saveat=saveat,
                    max_steps = algo_params['pontryagin_solver_maxsteps'],
                    args = (forward_sol, backwardpass_sol)
                )
                return forwardpass_sol

            # N initial states disturbed w/ gaussian noise. 
            N_sim = 100
            k = jax.random.PRNGKey(0)
            x0s = x0[None, :] + jax.random.normal(k, shape=(N_sim, nx)) * 0.1

            sols = jax.vmap(forward_sim)(x0s)
            visualiser.plot_trajectories_meshcat(sols)
            # this looks like complete shit. did i change the state format or something???

            # now that we have this nice sols object, we might as well plot some stats with it. 
            def local_v(t, x):
                xbar = forward_sol.evaluate(t)[0:nx]
                v, lam, S = backwardpass_sol.evaluate(t)
                dx = x - xbar
                quad_v = dx.T @ S @ dx
                v_x = v + lam.T @ dx + 0.5 * quad_v
                return v_x, quad_v

            # cool, calm & collected double vmap.
            vs, vquads = jax.vmap(jax.vmap(local_v, in_axes=(0, 0)), in_axes=(1, 1))(sols.ts, sols.ys)
            pl.figure()
            pl.semilogy(sols.ts.T, vs, color='green', alpha=0.1)
            pl.semilogy(sols.ts.T, vquads, color='red', alpha=0.1)
            pl.show()

            ipdb.set_trace()



        
        # then, here: 
        # - do forward pass
        # - do some convergence check
        # - if not converged, give forward solution to next iteration. 

        # forward pass: simulate our system in forward time, with input as if the
        # quadratic value expansion the backward pass actual value fct. 
        # (what if this fails? line search?)

        # for later: convergence criterion? 

