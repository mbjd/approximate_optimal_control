import jax
import jax.numpy as np
import diffrax

import matplotlib.pyplot as pl

import ipdb

import pontryagin_utils

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


            pl.show()
            ipdb.set_trace()

        # Then, first thing tomorrow: 
        # - work out the kinks above so it actually runs. 
        # - find some actually optimal trajectory by PMP backwards integration
        # - give it to this as an initial guess
        # - verify that the backward pass is consistent w/ this info
        #   (we should have the backward pass costate equal to the initial costate, 
        #   and if we forward sim again, obtain the same trajectory again)
        
        # then, here: 
        # - do forward pass
        # - do some convergence check
        # - if not converged, give forward solution to next iteration. 

        # and does the intuition about information flow along trajectories
        # still even exist? in the linear BVP world it is easy: we
        # parameterise all costate trajectories by the matrix S such that λ = S
        # x, and solve for S instead. However here what is the significance of
        # S? Trajectories can go "off" our iterate??

        # forward pass: simulate our system in forward time, with input as if the
        # quadratic value expansion the backward pass actual value fct. 
        # (what if this fails? line search?)

        # concretely: over [0, T] solve x' = f(x, u(x, t)), 
        # with u(x, t) = V(t) + Vx(t) @ (x - xl(t)) + (1/2) (x - xl(t)).T Vxx(t) (x - xl(t)) 
        # where V, Vx, Vxx are the value function parameters obtained in the backward pass
        # and xl(.) is the state trajectory used in that backward pass (i.e. the last iterate)

        # is this correct? can we just "ignore" input constraints like this? 


        # for later: convergence criterion? 

