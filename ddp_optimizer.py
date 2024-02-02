import jax
import jax.numpy as np
import diffrax

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

            # the taylor expansion stored normally as pytree.
            v, lam, S = state  

            prev_sol = args

            # how to give forwardsol nicely as an argument? 
            # is this te time to use "args"?          
            x = prev_sol.evaluate(t)[0:nx]

            u_star = pontryagin_utils.u_star_2d(x, lam, problem_params)

            # this partial derivative we can do by hand :) H is linear in lambda.
            H_lambda = f(t, x, u_star)  

            # do NOT confuse these! I have the power
            # of multivariate analysis and LaTeX on my side!
            H_x_fct = jax.jacobian(H, argnums=0)
            H_x = H_x_fct(x, u_star, lam)

            # the expression we got for \dot S. 

            def Hx_as_function_of_only_x(xp):
                # taylor approx of lambda(x), bc this function is for differentating
                # taylor approx is done about x, we are evaluating at xp.
                # if somehow someone demands third gradients of V in the form of second 
                # gradients of lambda, this will just give 0, as S is considered constant here. 

                lam_x = lam + S @ (xp - x)
                u_star = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
                return H_x_fct(x, u_star, lam_x)


            # all in all this is kind of a jacobian of a jacobian but the inner one is partial 
            # and the outer one is total...? 
            weird_Hxx = jax.jacobian(Hx_as_function_of_only_x)(x)

            ipdb.set_trace()

            # TODO do one of these: 
            # a) evaluate RHS that was used to generate the forward trajectory
            # b) approximate with d/dt of interpolated solution

            # this is the entirety of option b)
            dot_xbar = prev_sol.derivative(t)[0:nx]

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
            ### # optimal input given by some function lambda(x). this would encompass the repeated
            ### # forward passes, and initialisation with some global approximate V_x function. 
            ### # then we can do this: https://github.com/patrick-kidger/diffrax/issues/60
            ### # and save the costate during the forward simulation. because it is smooth, at 
            ### # least along forward trajectories, we expect that it works well with interpolation. 
            ### # then we could just get it here as part of the prev_sol object probably...

            ### # just for a small test
            ### lam_prev = lam * (1 + jax.random.normal(jax.random.PRNGKey(9130), shape=lam.shape) * 0.1)

            # this is the input from the previous forward simulation
            u_prev = pontryagin_utils.u_star_2d(x, lam_prev, problem_params)

            # according to my own derivations
            # am I now really the type of guy who finds working this out myself easier 
            # than copying formulas from existing papers? anyway...

            # the money shot, derivation & explanations in dump, section 3.5.*
            v_dot = -l(t, x, u_star) + lam.T @ (dot_xbar - H_lambda)
            # maybe ^ equivalent to -l(t, x, u_prev?)

            lam_dot = -H_x + S @ (dot_xbar - H_lambda)

            S_dot = -weird_Hxx

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
        ipdb.set_trace()

        # TODO also get the specifics (adaptive controller, max steps, SaveAt) right. 
        backwardpass_sol = diffrax.diffeqsolve(term, Diffrax.Tsit5(), t0=T, t1=0, dt0=0.1, y0=init_state)

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

