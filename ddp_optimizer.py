import jax
import jax.numpy as np
import diffrax

import matplotlib.pyplot as pl

import ipdb
import tqdm
import time
from operator import itemgetter

import pontryagin_utils
import visualiser

# attept at implementing continuous-time DDP, a bit like: 
# - https://dl.acm.org/doi/pdf/10.1145/3592454 (Hutter) but without the parameterised stuff
# - https://arxiv.org/pdf/2107.04507.pdf (Sun) but without the opponent/disturbance
# - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10156422 
# ofc later we can add any of those components back maybe hopefully. 


def ddp_main(problem_params, algo_params, x0):

    # problem_params, algo_params: the usual things
    # x0: the initial state. 
    
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
    # forward_sol = init_sol
    # x0 = forward_sol.evaluate(0.)[0:nx]  # this stays the same in the entire function

    # this is to handle case where t0>t1 (if solution was obtained backwards)
    t0, tf = 0., problem_params['T']

    # write the whole iteration in a scan-type loop? 
    # which pass should we do first in each loop iteration? 
    # backward then forward? 
    # + no special case for initial trajectory



    def PMP_wrongness(t, state, args):

        # this is a measure of how far we're off form an exact PMP solution. 
        # basically we take the direction difference from the backward pass: 
        # if the forward trajectory comes from the same direction the backward
        # characteristics go in, then it is optimal. This returns the norm
        # of that direction difference (for a single time instant)

        # call this function with same arguments as backwardpass RHS, it is 
        # basically a dumbed down version of that. 

        v, lam, S = state  
        prev_forward_sol = args


        xbar = prev_forward_sol.evaluate(t)[0:nx]
        dot_xbar = prev_forward_sol.derivative(t)[0:nx]

        # optimal input *according to backward-pass costate*
        u_star = pontryagin_utils.u_star_2d(xbar, lam, problem_params)

        # this partial derivative we can do by hand :) H is linear in lambda.
        H_lambda = f(t, xbar, u_star)


        # the money shot, derivation & explanations in dump, section 3.5.*
        direction_diff = dot_xbar - H_lambda
        return np.linalg.norm(direction_diff)


    def backwardpass_rhs_init(t, state, args):

        # same as the other one, except with different source of the 
        # function lambda(x) used in the forward pass.

        v, lam, S = state  

        # these would be the arguments are for the non-init version. 
        # prev_forward_sol, prev_backward_sol, forward_sol = args

        # instead here we have: 
        forward_sol, lambda_fct = args

        xbar = forward_sol.evaluate(t)[0:nx]
        lambda_bar = lambda_fct(xbar)

        # this is the optimally controlled system, according to the costate currently
        # being solved for in the backward pass
        u_star = pontryagin_utils.u_star_2d(xbar, lam, problem_params)

        H_lambda = f(t, xbar, u_star)
        H_x = jax.jacobian(H, argnums=0)(xbar, u_star, lam)

        # this is the substitude for forwardpass_rhs in the other version
        # here we basically just evaluate the RHS used during the forward pass
        u_star_bar = pontryagin_utils.u_star_2d(xbar, lambda_bar, problem_params)
        dot_xbar = f(t, xbar, u_star_bar)


        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # confirmed that [fx flam; gx glam] == Alin from above
        # f and g are defined as the pmp rhs: x_dot = f(x, lam), lam_dot = g(x, lam)
        # this removes any confusion with already present partial derivatives of H
        fx, gx = jax.jacobian(pmp_rhs, argnums=0)(xbar, lam)
        flam, glam = jax.jacobian(pmp_rhs, argnums=1)(xbar, lam)

        v_dot = -l(t, xbar, u_star) + lam.T @ (dot_xbar - H_lambda)
        lam_dot = -H_x + S @ (dot_xbar - H_lambda)
        S_dot = gx + glam @ S - S @ fx - S @ flam @ S

        return (v_dot, lam_dot, S_dot)

    def backwardpass_rhs(t, state, args):

        # ALL these dynamics are being specified as if time was running forward.
        # flipping the integration boundaries and starting with dt < 0 in the 
        # diffrax solve then does the time reversal correctly.

        # v = cost to go, lam = costate = value gradient, S = value hessian
        v, lam, S = state  

        # different previous solutions are organised like this: 
        # iteration     | k-1                | k
        # forward sol   | prev_forward_sol   | forward_sol
        # backward sol  | prev_backward_sol  | backward_sol  (created here)
        prev_forward_sol, prev_backward_sol, forward_sol = args


        regularise = False
        if regularise:
            # redefine the stage cost and hamiltonian to include an additional
            # term penalising deviations from the previous trajectory. 

            # this is copied from get_termial_lqr so no check again here if it is an equilibrium.
            x_eq = problem_params['x_eq']
            u_eq = problem_params['u_eq']
            Q = jax.hessian(l, argnums=1)(0., x_eq, u_eq)  # we just take this as state cost weights. 
            R = jax.hessian(l, argnums=2)(0., x_eq, u_eq)

            def H(x, u, λ):
                # t is constant in this whole function which is "redefined" at every RHS evaluation
                x_deviation = x - prev_forward_sol.evaluate(t)[0:nx]
                u_deviation = None  # ignore this for now
                deviation_penalty = 1. * x_deviation.T @ Q @ x_deviation
                return l(0., x, u) + deviation_penalty + λ.T @ f(0., x, u)
        else:
            H = lambda x, u, λ: l(0., x, u) + λ.T @ f(0., x, u)

        # current state and optimal input 
        # *according to backward-pass costate currently being solved for*
        xbar = forward_sol.evaluate(t)[0:nx]
        u_star = pontryagin_utils.u_star_2d(xbar, lam, problem_params)

        # this partial derivative we can do by hand :) H is linear in lambda.
        H_lam_fct = jax.jacobian(H, argnums=2)

        H_lambda = f(t, xbar, u_star)

        H_x = jax.jacobian(H, argnums=0)(xbar, u_star, lam)

        # TODO do one of these: 
        # a) evaluate RHS that was used to generate the forward trajectory
        # b) approximate with d/dt of interpolated solution
        use_RHS = True

        # ~~~ option a) ~~~

        # on a high level: we want the costate/input used when generating the
        # forward trajectory, so that we can accurately tell which direction it 
        # came from, without differentiating the polynomial interpolation wrt time. 

        # for this we would need the previous backward pass, to evaluate at the 
        # previous forward trajectory here, essentially "recreating" the last 
        # forwrad rollout 1:1. which I'd argue is not really overkill, because we 
        # need to evaluate it at different time points anyway, therefore re-evaluating.

        # or we could find some way to store lambda(t) as a polynomial interpolation
        # too, which would remove the issues with d/dt of an interpolation too, and 
        # also avoid interpolating nondifferentiable points that can be in u(t)

        # maybe that could be done with the SaveAt object in diffrax? 
        # https://docs.kidger.site/diffrax/api/saveat/
        # there is the 'fn' option... default fn = lambda t, y, args: y
        # instead we could say fn = lambda t, y, args: y, lambda_prev_backwardpass(y)
        # this would probably incur additional evaluations of the backward solution
        # just like the "obvious" approach of passing directly the whole backward sol. 
        # advantage:    we don't pass the whole backwardsol, there is only one backward
        #               sol at a time and we keep bookkeeping to a minimum
        # disadvantage: we incur additinal approximation error due to forming an interpolation
        #               of the lambda trajectory, instead of re-evaluating the precise backwardpass
        #               at the state trajectory. not sure if this is significant though.

        # -> this idea is ditched, whatever fn returns is stored in ys but not in the interpolation
        # https://github.com/patrick-kidger/diffrax/issues/301
        
        # therefore we kind of need to go "all the way": evaluate the RHS of the previous
        # forward pass again. This is the reason we do the backward pass first, so we can 
        # work with info from current and last iteration and not yet another one before. 
        dot_xbar_rhs = forwardpass_rhs(t, xbar, (prev_forward_sol, prev_backward_sol))


        

        # here, somehow do this: 
        # if it is the initial step: 
        #     dot_xbar = system_dynamics(t, x, u*(x, lambda(x)))
        # otherwise:
        #     dot_xbar = (the thing above with the previous solutions)
        # with init as a static argument for jitting (how did that work again?)

        # maybe with jax.lax switch/cond? 

        # OR write an entirely separate version that is responsible for the init step
        # or another cheap hack: for the first iteration fall back to option b, the 
        # time-differentiation of the interpolated solution? 

        # ~~~ option b) ~~~
        dot_xbar_differentiation = forward_sol.derivative(t)[0:nx]

        if use_RHS:
            dot_xbar = dot_xbar_rhs
        else:
            dot_xbar = dot_xbar_differentiation

        # the money shot, derivation & explanations in dump, section 3.5.*
        v_dot = -l(t, xbar, u_star) + lam.T @ (dot_xbar - H_lambda)
        # maybe ^ equivalent to -l(t, xbar, u_prev?)

        lam_dot = -H_x + S @ (dot_xbar - H_lambda)

        # # other derivation in idea dump, second try, DOC inspired. 
        # # all second partial derivatives of the pre-optimised hamiltonian. 
        # H_opt_xx = jax.hessian(H_opt, argnums=0)(xbar, lam)
        # H_opt_xlam = jax.jacobian(jax.jacobian(H_opt, argnums=0), argnums=1)(xbar, lam)
        # # H_opt_lamx = jax.jacobian(jax.jacobian(H_opt, argnums=1), argnums=0)(xbar, lam)
        # H_opt_lamlam = jax.hessian(H_opt, argnums=1)(xbar, lam)

        # # purely along characteristic curve, no correction term here. 
        # # still unsure whether or not this makes it wrong (see idea dump)
        # # indeed it seems there are differences between this and the other one which are
        # # more than just numerical noise (about 1.5% at one particular point)
        # S_dot = -H_opt_xx - H_opt_xlam @ S - (H_opt_xlam @ S).T - S.T @ H_opt_lamlam @ S


        # these are basically manual differentiation rules applied to the function
        #  (x, lam)  ->  ( H_x(x, u*(x, lam), lam), (H_x(x, u*(x, lam), lam) )
        # and in the end we get its jacobian and call it A_lin, a 2*nx square matrix.
        # kind of dumb in retrospect -- hence below this I do it all with jax in one step.

        # H_xx = jax.jacobian(H_x_fct, argnums=0)(xbar, u_star, lam)
        # H_xu = jax.jacobian(H_x_fct, argnums=1)(xbar, u_star, lam)  
        # H_xlam = jax.jacobian(H_x_fct, argnums=2)(xbar, u_star, lam)
        # # H_ux == H_xu.T etc

        # H_lamx = jax.jacobian(H_lam_fct, argnums=0)(xbar, u_star, lam)
        # H_lamu = jax.jacobian(H_lam_fct, argnums=1)(xbar, u_star, lam)
        # H_lamlam = jax.jacobian(H_lam_fct, argnums=2)(xbar, u_star, lam)
        # 
        # u_star_x = jax.jacobian(pontryagin_utils.u_star_2d, argnums=0)(xbar, lam, problem_params)
        # u_star_lam = jax.jacobian(pontryagin_utils.u_star_2d, argnums=1)(xbar, lam, problem_params)

        # # the matrix governing the evolution of (\delta x, \delta \lambda) in the linearised 
        # # pontryagin BVP (see notes in idea dump)
        # Alin = np.block([[H_lamx, H_lamlam], [-H_xx, -H_xlam]]) + np.vstack([H_lamu, -H_xu]) @ np.hstack([u_star_x, u_star_lam])

        # # because the derivation uses those to shorten notation...
        # # also I have two different short notations for the same thing...
        # A11 = Alin[0:nx, 0:nx]
        # A12 = Alin[0:nx, nx:]
        # A21 = Alin[nx:, 0:nx]
        # A22 = Alin[nx:, nx:]

        # s = lam
        # Sdotnew = A21 + A22 @ S - S @ A11 - S @ A12 @ S
        # sdotnew = (-S @ A12 + A22) @ s  # this one appears to be wrong though. lam_dot makes better plots.


        # yet another way, maybe the nicest???
        # this is basically f_forward from pontryagin_utils but without the separate v variable. 
        # also the arguments are kept separate so we nicely get the four derivative combinations
        # instead of one big matrix. 
        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # confirmed that [fx flam; gx glam] == Alin from above
        # f and g are defined as the pmp rhs: x_dot = f(x, lam), lam_dot = g(x, lam)
        # this removes any confusion with already present partial derivatives of H
        fx, gx = jax.jacobian(pmp_rhs, argnums=0)(xbar, lam)
        flam, glam = jax.jacobian(pmp_rhs, argnums=1)(xbar, lam)

        # Sdotnew = Sdot_three at least. it is literally the same formula :) 
        # Sdot_three from 'characteristics' derivation with (finally I think) derivation of Sdot. 
        # Sdotnew is from the pontryagin BVP -> linearisation path. 
        S_dot_three = gx + glam @ S - S @ fx - S @ flam @ S

        # = IS.T @ A_lin_alt @ IS
        # with 
        IS = np.vstack([np.eye(nx), S])
        A_lin_alt = np.block([[gx, glam], [-fx, flam]])
        # what if, for regularisation, we just add a small diagonal term to flam? 

 

        # both look pretty close to each other, and also both symmetric up to relative error of about 1e-8 :) 
        # S_dot_rel_asymmetry = np.linalg.norm(S_dot - S_dot.T) / np.linalg.norm(S_dot)
        # Sdotnew_rel_asymmetry = np.linalg.norm(Sdotnew - Sdotnew.T) / np.linalg.norm(Sdotnew)
        # Sdot_rel_diff = np.linalg.norm(Sdotnew - S_dot) / np.linalg.norm(Sdotnew)

        # however, sdotnew and lam_dot are markedly different. 
        # this is despite the two relevant directions (d/dt xbar and H_lambda) being almost the same (worst ratio .997)
        # are they even comparable? the early, characteristics type derivation only has a \lambda variable. 
        # in the new DOC derivation, we have lambda AND \delta lambda, the latter one being what we are finding in the linear BVP. 
        # still a bit of understanding left to be gained here. 

        # S_dot slightly off of being symmetric unfortunately. 
        # so, artificially "symmetrize" them here? or baumgarte type stabilisation? 
        # let's do nothing for the moment and wait until it bites us in the ass :) 

        # let us try to do gradient descent on ||A-A'||^2. 
        # matrixcalculus.org tells us that the gradient of that expression wrt A is: 
        # 2 (A - A') - 2 (A' - A) = 2 A - 2 A' - 2 A' + 2 A = 4 (A - A').

        # somehow this makes things worse even after flipping the sign twice
        # asymmetry_gradient = 4 * (S - S.T)
        # baumgarte_timeconstant = 0.5  # seconds (or other time units)
        # S_dot_three = S_dot_three + asymmetry_gradient / baumgarte_timeconstant

        # ipdb.set_trace()

        return (v_dot, lam_dot, S_dot_three)



    def backwardpass_rhs_DOC(t, state, args):

        # same as the original one BUT everything linearised around forward trajectory. 
        # not working yet - still not sure where we are in the state space (x, \lambda)
        # and where we are in the tangent space (\delta x, \delta \lambda). 

        # v = cost to go, lam = costate = value gradient, S = value hessian
        v, lam, S = state  

        # different previous solutions are organised like this: 
        # iteration     | k-1                | k
        # forward sol   | prev_forward_sol   | forward_sol
        # backward sol  | prev_backward_sol  | backward_sol  (created here)
        prev_forward_sol, prev_backward_sol, forward_sol = args

        # the DOC solution linearises everything around the previous trajectory (x, lambda).
        # thus we retrieve it here which is easy now that we have all previous solutions. 
        # (other than that not implemented yet)
        x_forward = forward_sol.evaluate(t)[0:nx]

        # these two define our *previous* value expansion, which was used to create the forward pass
        x_prev = prev_forward_sol.evaluate(t)[0:nx] 
        v_prev, lam_prev, S_prev = prev_backward_sol.evaluate(t)

        # so we use that expansion again to find the costate lambda used in the forward sim. 
        dx = x_forward - x_prev
        lam_forward = lam_prev + S_prev @ dx

        # purely for calculating V - could probably be ditched. 
        u_forward = pontryagin_utils.u_star_2d(x_forward, lam_forward, problem_params)


        # then we linearise the WHOLE PMP RHS around that state. "Hidden" in here is the u* map. 
        def pmp_rhs(state, costate):

            u_star = pontryagin_utils.u_star_2d(state, costate, problem_params)

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot

        # confirmed that [fx flam; gx glam] == Alin from above
        #            ( = [A11 A12; A21 A22])
        # all evaluated at (x_forward, lam_forward) from the forward pass. 
        # so now we have properly linearised the whole system around the forward pass. 
        # like in the derivation we now perform the backward pass for the *linearised* system. 

        # f and g are defined as the pmp rhs: x_dot = f(x, lam), lam_dot = g(x, lam)
        # maybe ditch this naming scheme to not confuse with problem parameters f and g? 
        # this removes any confusion with already present partial derivatives of H
        fx, gx = jax.jacobian(pmp_rhs, argnums=0)(x_forward, lam_forward)
        flam, glam = jax.jacobian(pmp_rhs, argnums=1)(x_forward, lam_forward)

        # arguably this v would be better calculated in forward time and later shifted so it is 0 at tf
        # because as it stands obviously this v_dot is literally the same ODE as in forward time...

        # not sure if this is actually the case mathematically... 
        # well actually if lambda = S dx + s then indeed this is just the costate on our forward trajectory
        s = lam  

        '''
        so this is a bit fishy. we get wildly different numbers than with the old
        function for s_dot. This is even at the very end of a rather benign trajectory
        where we might expect the old rhs to give the same result. why not? 
    
        todo tomorrow: 
        - go over equations again in new derivation and search for mistakes
        - do some more numerical sanity checks.

        '''


        # taylor expansion correction to adjust for the fact that we don't want u_forward, lam_forward, but
        # u, lam currently being solved for. but in the taylor expansion around the forward trajectory...
        xdot_forward = f(0., x_forward, u_forward)

        # u_optimal_taylor = u_forward + jax.jacobian(pontryagin_utils.u_star_2d, argnums=1)(x_forward, lam_forward, problem_params) @ (lam - lam_forward)
        # xdot_optimal = f(0., x_forward, u_forward) + jax.jacobian(f, argnums=2)(0., x_forward, u_forward) @ (u_optimal_taylor - u_forward)

        # lam_dot = -jax.jacobian(H, argnums=0)(x_forward, u_forward, lam_forward) + S @ (xdot_forward - xdot_optimal)
        # lam_dot1 = -jax.jacobian(H, argnums=0)(x_forward, u_forward, lam_forward) - S @ jax.jacobian(f, argnums=2)(0., x_forward, u_forward) @ (u_optimal_taylor - u_forward)
        # lam_dot2 = -jax.jacobian(H, argnums=0)(x_forward, u_forward, lam_forward) - S @ (jax.jacobian(f, argnums=2)(0., x_forward, u_forward) @ (jax.jacobian(pontryagin_utils.u_star_2d, argnums=1)(x_forward, lam_forward, problem_params)) @ (lam - lam_forward)) 

        # df_du = jax.jacobian(f, argnums=2)(0., x_forward, u_forward)
        # du_dlam = jax.jacobian(pontryagin_utils.u_star_2d, argnums=1)(x_forward, lam_forward, problem_params)
        # lam_dot3 = -jax.jacobian(H, argnums=0)(x_forward, u_forward, lam_forward) - S @ (df_du) @ (du_dlam) @ (lam - lam_forward)

        df_dlam = jax.jacobian(lambda lam: f(0., x_forward, pontryagin_utils.u_star_2d(x_forward, lam, problem_params)))(lam_forward)
        lam_dot4 = -jax.jacobian(H, argnums=0)(x_forward, u_forward, lam_forward) + S @ (df_dlam) @ (lam - lam_forward)



        v_dot = -l(t, x_forward, u_forward)
        s_dot = lam_dot4
        # s_dot = -S @ flam @ s + glam @ (s - lam_forward)
        S_dot = gx + glam @ S - S @ fx - S @ flam @ S

        # old version to compare:
        # H_x = jax.jacobian(H, argnums=0)(x_forward, u_star, lam)
        # lam_dot = -H_x + S @ (dot_xbar - H_lambda)
        # ipdb.set_trace()

        return (v_dot, s_dot, S_dot)

    def forwardpass_rhs(t, state, args):

        # here we only have the actual system state for a change
        x = state  
        # need both backward and forward pass solutions. 
        prev_forward_sol, prev_backward_sol = args 

        # this stacked vector forward sol is not too elegant, should probably
        # introduce a tuple or even dict state to access solution more clearly
        xbar = prev_forward_sol.evaluate(t)[0:nx] 
        dx = x - xbar
        v_xbar, lam_xbar, S_xbar = prev_backward_sol.evaluate(t)
        # this defines a local quadratic value function: 
        # v(xbar + dx) = v + lam.T dx + 1/2 dx.T S dx
        # we need the first gradient of this quadratic value function. 
        # lambda(xbar + dx) = 0 + lam.T + dx.T S  (or its transpose whatever)
        lam_x = lam_xbar + S_xbar @ dx

        u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
        dot_x = f(t, x, u)

        return dot_x

    # same as forwardpass_rhs but returns the input. mostly for plotting.
    def forwardpass_u(t, state, args):

        # here we only have the actual system state for a change
        x = state  
        # need both backward and forward pass solutions. 
        prev_forward_sol, prev_backward_sol = args 

        # this stacked vector forward sol is not too elegant, should probably
        # introduce a tuple or even dict state to access solution more clearly
        xbar = prev_forward_sol.evaluate(t)[0:nx] 
        dx = x - xbar
        v_xbar, lam_xbar, S_xbar = prev_backward_sol.evaluate(t)
        # this defines a local quadratic value function: 
        # v(xbar + dx) = v + lam.T dx + 1/2 dx.T S dx
        # we need the first gradient of this quadratic value function. 
        # lambda(xbar + dx) = 0 + lam.T + dx.T S  (or its transpose whatever)
        lam_x = lam_xbar + S_xbar @ dx

        u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
        return u, lam_x  # breaking change, this also returns lambda now. 



    def ddp_backwardpass(prev_forward_sol, prev_backward_sol, forward_sol):

        # backward pass. (more comments in python for loop below.)

        # different previous solutions are organised like this: 
        # iteration     | k-1                | k
        # forward sol   | prev_forward_sol   | forward_sol
        # backward sol  | prev_backward_sol  | backward_sol  (created here)

        term = diffrax.ODETerm(backwardpass_rhs)

        # not working yet :(
        term = diffrax.ODETerm(backwardpass_rhs_DOC)

        # this 0:nx is strictly only needed if we have extended state y = [x, lambda, v].
        # we are overloading this function to handle both extended and pure state. 
        # (this means we have to re-jit when changing between them)
        xf = forward_sol.evaluate(tf)[0:nx]  

        # terminal conditions given by taylor expansion of terminal value
        # these are static w.r.t. jit compilation. 
        v_T = xf.T @ P_lqr @ xf
        lam_T = 2 * P_lqr @ xf
        S_T = P_lqr 

        # instead of doing something about it, just pass it to the output 
        # so at least we know about it.
        # instead of this we might also output the SDF of Xf, v_T(xf) - v_f
        # xf_outside_Xf = v_T >= v_f * 1.001
        # or not, because this is just a function of the forward solution
        # which can be calculated outside whenever we want. 

        init_state = (v_T, lam_T, S_T)

        # relaxed tolerance - otherwise the backward pass needs many more steps
        # maybe it also "needs" the extra accuracy though...? 
        relax_factor = 10.
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
            dtmin = 0.002,  # preposterously small to avoid getting stuck completely

            # additionally step to the nodes from the forward solution
            # does jump_ts do the same? --> yes, except for re-evaluation of the RHS for FSAL solvers
            # step_ts = forward_sol.ts,
        )

        # try this, maybe it works better \o/
        # step_ctrl_fixed = diffrax.StepTo(prev_forward_sol.ts)
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=tf, t1=t0, dt0=-0.1, y0=init_state,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            args = (prev_forward_sol, prev_backward_sol, forward_sol),
        )

        return backward_sol # , xf_outside_Xf

    def ddp_forwardpass(prev_forward_sol, prev_backward_sol, x0):

        # forward pass. 
        term = diffrax.ODETerm(forwardpass_rhs)

        # same tolerance parameters used in pure unguided backward integration of whole PMP
        step_ctrl = diffrax.PIDController(
            rtol=algo_params['pontryagin_solver_rtol'],
            atol=algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
            dtmin = 0.05  # relatively large - see what happens
        )

        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True) 

        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=t0, t1=tf, dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            args = (prev_forward_sol, prev_backward_sol),
        )

        return forward_sol


    def scan_fct(carry, inp):

        # by first doing forward and then backward sol (not other way) we make
        # this iteration "markovian" -- each iteration only depends on the last. 
        # (the local feedback controller responsible for the forward pass is given
        # by the previous backward AND forward solution, from which we form our 
        # V(x) taylor exapnsion.) 

        # also any step length/line search/convergence checks can be put here.
        # here is a pseudocode sketch of that: 
        '''
        forward_sol = forwardpass(prev_forward_sol, prev_backward_sol)
        cost = cost(forward_sol)
        # prev_cost from carry
        accept = check_descent_condition(prev_forward_sol, prev_cost, 
                                         forward_sol, cost)
        if not accept: 
            forward_sol = prev_forward_sol
            alpha = (next higher alpha in some fixed sequence)
        
        # but to redo the last backward pass we again need one backwardpass
        # more than we have here..

        # other option: already produce a wide range of stepsizes in the 
        # backward pass. effectively do N backward passes, with different 
        # 
        backward_sol = ddp_backwardpass(prev_forward_sol, )

        
            
        '''


        # what if...
        prevprev_iter, prev_iter = carry

        prev_forward_sol, prev_backward_sol = carry
        x0 = inp

        # the whole DDP loop is it not beautiful <3 
        forward_sol = ddp_forwardpass(prev_forward_sol, prev_backward_sol, x0)
        backward_sol = ddp_backwardpass(prev_forward_sol, prev_backward_sol, forward_sol)

        out = dict()
        out['backward_sol'] = backward_sol
        out['forward_sol'] = forward_sol

        new_carry = forward_sol, backward_sol

        return new_carry, out

    def initial_step(lambda_fct, x0):
        # the scan loop is rearranged mainly for the reason that it makes it 
        # markovian -- the current iteration only depends on the last iteration. 
        # thus, in the initial step, we simply make an initial forward solution
        # and an initial backward solution. 

        # for the forward solution, we supply a function lambda(x), which via the 
        # hamiltonian minimisation gives an input u*(x, lambda(x)). simple enough.

        # Then we perform the first backward pass, which is the main
        # "exception" in this step --  subsequent backward passes (k) need the
        # forward sol at k, and also the forward AND backward sol at (k-1),
        # the latter two giving the input which leads to forward solution k. 
        # the first backward pass is actually simpler because we have the function 
        # lambda(x) right here, instead of being represented as a taylor expansion
        # (backward pass) around a previous solution (forward pass)


        # do a forward simulation with controller u(x) = u*(x, lambda(x))
        def forwardsim_rhs(t, x, args):
            lam_x = 2 * P_lqr @ x
            u = pontryagin_utils.u_star_2d(x, lam_x, problem_params)
            return problem_params['f'](t, x, u)

        term = diffrax.ODETerm(forwardsim_rhs)
        step_ctrl = diffrax.PIDController(rtol=algo_params['pontryagin_solver_rtol'], atol=algo_params['pontryagin_solver_atol'])
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True) 

        init_forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0., t1=problem_params['T'], dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'],
        )

        
        # then perform the backward pass. this is copied & adapted from the ddp_backwardpass fct. 

        # different previous solutions are organised like this: 
        # iteration     | k-1                | k
        # forward sol   | prev_forward_sol   | forward_sol
        # backward sol  | prev_backward_sol  | backward_sol  (created here)

        term = diffrax.ODETerm(backwardpass_rhs_init)

        xf = init_forward_sol.evaluate(tf)[0:nx]  

        v_T = xf.T @ P_lqr @ xf
        lam_T = 2 * P_lqr @ xf
        S_T = P_lqr 

        init_state = (v_T, lam_T, S_T)

        relax_factor = 1.
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
        )
        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        init_backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=tf, t1=t0, dt0=-0.1, y0=init_state,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            args = (init_forward_sol, lambda_fct),
        )

        return init_forward_sol, init_backward_sol 


    N_x0s = 4


    # prepare initial step
    # damn, this worked on like the second try, i am so good
    init_carry = initial_step(lambda x: 2 * P_lqr @ x, x0)
    # new_carry, output = scan_fct(init_carry, x0)


    forward_sol = ddp_forwardpass(init_carry[0], init_carry[1], x0)

    xf = forward_sol.evaluate(tf)[0:nx]  
    v_T = xf.T @ P_lqr @ xf
    lam_T = 2 * P_lqr @ xf
    S_T = P_lqr 
    init_state = (v_T, lam_T, S_T)

    rhs_args = (init_carry[0], init_carry[1], forward_sol)

    # does the backward pass make any sense?
    test = backwardpass_rhs(tf, init_state, rhs_args)
    test1 = backwardpass_rhs_DOC(tf, init_state, rhs_args)
    ipdb.set_trace()
    # test_DOC = backwardpass_rhs_DOC(tf, init_state, rhs_args)

    # ipdb.set_trace()



    # sweep the Phi and omega states from x0 to 0. 
    x0_final = x0.at[np.array([2,5])].set(0)
    # go to weird position
    x0_final = x0 + 2*np.array([10, 0, 0.5, 10., 0, 10])
    x0_final = x0 + 2*np.array([10, -10, 0, 0, 0, 0])
    x0_final = x0 + 0.1*np.array([10, 0, 0, 0, 10, 0])

    # x0_final = x0 + 2*np.array([0, 10., 0, 10., 0, 0])
    # x0_final = x0 + np.array([0, 0, np.pi, 0, 0, 0])
    # don't sweep at all
    # x0_final = x0  
    alphas = np.linspace(0, 1,  N_x0s)

    # this is kind of ugly ikr
    # we stay at each x0 and run the ddp loop for $iters_per_x0 times. 
    # after that we modify the N_iters variable 
    iters_per_x0 = 15
    # max_alpha = 91/320
    # alphas = np.clip(alphas, 0, max_alpha)  # stop at this point & iterate at same x0 forever
    alphas = np.repeat(alphas, iters_per_x0)[:, None]  # to "simulate" several iterations per x0.
    N_iters = alphas.shape[0]
    iters_from_same_x0 = np.arange(N_iters) % iters_per_x0

    x0s = x0 * (1-alphas) + x0_final * alphas

    start_t = time.time()
    final_carry, outputs = jax.lax.scan(scan_fct, init_carry, x0s)
    print(f'first run (with jit): {time.time() - start_t}')

    # start_t = time.time()
    # final_ca1ry, outputs = jax.lax.scan(scan_fct, new_carry, x0s)
    # print(f'second run: {time.time() - start_t}')

    final_forwardsol, final_backwardsol = final_carry

    plot=True
    if plot:

        def plot_forwardpass(sol, ts, alpha=1., labels=True):

            # plot trajectory with solver steps and interpolation.
            pl.gca().set_prop_cycle(None)
            pl.plot(sol.ts, sol.ys, marker='.', linestyle='', alpha=alpha)

            interp_ys = jax.vmap(sol.evaluate)(ts)
            pl.gca().set_prop_cycle(None)
            pl.plot(interp_ts, interp_ys, label=problem_params['state_names'] if labels else None, alpha=alpha)
            if labels: pl.legend()

        def plot_backwardpass(sol, ts, alpha=1., labels=True):

            # TODO also plot the continuous time interpolation here? 

            ax0 = pl.subplot(221)
            pl.semilogy(sol.ts, sol.ys[0], label='v' if labels else None)
            pl.legend()

            pl.subplot(222, sharex=ax0)
            pl.plot(sol.ts, sol.ys[1], label=problem_params['state_names'] if labels else None)
            pl.ylabel('costates')
            pl.legend()

            pl.subplot(223, sharex=ax0)
            pl.ylabel('S(t) - raw entries')
            pl.plot(sol.ts, sol.ys[2].reshape(-1, nx*nx), color='black', alpha=0.2)

            pl.subplot(224, sharex=ax0)
            S_eigenvalues = jax.vmap(lambda S: np.sort(np.linalg.eig(S)[0].real))(sol.ys[2])
            eigv_label = ['S(t) eigenvalues'] + [None] * (nx-1)
            pl.semilogy(sol.ts, S_eigenvalues, color='C0', label=eigv_label)

            # relative matrix asymmetry: || (S - S.T)/S || = || (1 matrix) - S.T/S ||
            # S_asym_rel = jax.vmap(lambda S: np.linalg.norm(1 - S.T/S))(sol.ys[2])
            # pl.semilogy(sol.ts, S_asym_rel, color='C1', label='norm( (S-S.T)/S )')

            # other versioon: || S-S.T || / ||S||
            # hard to say which one of these is better to use...
            S_asym_rel = jax.vmap(lambda S: np.linalg.norm(S-S.T)/np.linalg.norm(S))(sol.ys[2])
            pl.semilogy(sol.ts, S_asym_rel, color='C1', label='norm(S-S.T)/norm(S)')
            pl.legend()
            
        def plot_forward_backward(i):

            pl.figure(f'forward and backward pass {i}')

            # does a part of both of the above functions plus the retrieval of the solutions.
            ts = np.linspace(t0, tf, 5001)

            fw = jax.tree_util.tree_map(itemgetter(i), outputs['forward_sol'])
            bw = jax.tree_util.tree_map(itemgetter(i), outputs['backward_sol'])

            # plot the state trajectory of the forward pass, interpolation & nodes. 
            ax1 = pl.subplot(221)

            pl.plot(fw.ts, fw.ys, marker='.', linestyle='', alpha=1, label=problem_params['state_names'])
            interp_ys = jax.vmap(fw.evaluate)(ts)
            pl.gca().set_prop_cycle(None)
            pl.plot(ts, interp_ys, alpha=0.5)
            pl.legend()

            # plot the input used for the forward pass. 
            pl.subplot(222, sharex=ax1)

            fw_prev = jax.tree_util.tree_map(itemgetter(i-1), outputs['forward_sol'])
            bw_prev = jax.tree_util.tree_map(itemgetter(i-1), outputs['backward_sol'])
            args = (fw_prev, bw_prev)

            def u_t(t):
                return forwardpass_u(t, fw.evaluate(t), args)[0]

            us_trajectory = jax.vmap(u_t)(fw.ts)
            us_interp = jax.vmap(u_t)(ts)

            sumdiff = np.array([[1, 1], [1, -1]])
            pl.plot(fw.ts, us_trajectory, marker='.', linestyle='')
            pl.gca().set_prop_cycle(None)
            pl.plot(ts, us_interp, label=('u0', 'u1'))
            pl.plot(ts, us_interp @ sumdiff.T, label=('usum', 'udiff'))
            pl.legend()

            # plot the eigenvalues of S from the backward pass.
            pl.subplot(223, sharex=ax1)

            # eigenvalues at nodes. 
            sorted_eigs = lambda S: np.sort(np.linalg.eig(S)[0].real)
            S_eigenvalues = jax.vmap(sorted_eigs)(bw.ys[2])
            eigv_label = ['S(t) eigenvalues'] + [None] * (nx-1)
            pl.semilogy(bw.ts, S_eigenvalues, color='C0', marker='.', linestyle='', label=eigv_label)
            # also as line bc this line is more accurate than the "interpolated" one below if timesteps become very small
            pl.semilogy(bw.ts, S_eigenvalues, color='C0')  

            # eigenvalues interpolated. though this is kind of dumb seeing how the backward
            # solver very closely steps to the non-differentiable points. 
            S_interp = jax.vmap(bw.evaluate)(ts)[2]
            S_eigenvalues_interp = jax.vmap(sorted_eigs)(S_interp)
            pl.semilogy(ts, S_eigenvalues_interp, color='C0', linestyle='--', alpha=.5)
            pl.legend()

            pl.subplot(224, sharex=ax1)
            # raw entries of S(t)
            St = bw.ys[2].reshape(-1, 6*6)
            pl.plot(bw.ts, St, marker='.', linestyle='--', alpha=.2, color='black')
            pl.plot(ts, S_interp.reshape(-1, 6*6), alpha=.2, color='black')


        plot_forward_backward(400)
        
        # try to isolate that one failure. 
        i = 630
        ta = 4.7
        tb = 4.8
        fw = jax.tree_util.tree_map(itemgetter(i), outputs['forward_sol'])
        bw_prev = jax.tree_util.tree_map(itemgetter(i-1), outputs['backward_sol'])
        fw_prev = jax.tree_util.tree_map(itemgetter(i-1), outputs['forward_sol'])
        args = (fw_prev, bw_prev)
        ts = np.linspace(ta, tb, 1001)
        xs = jax.vmap(fw.evaluate)(ts)
        us, lams = jax.vmap(forwardpass_u, in_axes=(None, 0, None))(0., xs, args)

        interp_ts = np.linspace(t0, tf, 512)
        # for j in range(100, 110): plot_forward_backward(j)

        # find the spot where it failed. 
        # except if it fails due to instabilit
        # take just state 0 (pos x) because the others are inf/NaN in same case.
        # take second state of each trajectory because first one is initial state
        # and always defined. 
        second_states = outputs['forward_sol'].ys[:, 1, 0]
        isnan = np.isnan(second_states)
        isfinite = np.isfinite(second_states)
        is_valid = np.logical_and(isfinite, np.logical_not(isnan)) 
        last_valid_idx = np.max(is_valid * np.arange(is_valid.shape[0]))

        # for j in np.arange(last_valid_idx - 3, last_valid_idx + 3):
        #     try:
        #         plot_forward_backward(j)
        #     except Error as e:
        #         print(f'oops index {j} probably out of bounds')
        #         print(e)

        # plot the final solution - ODE solver nodes...
        pl.figure('final solution')

        plot_forwardpass(final_forwardsol, interp_ts)


        # pl.figure('intermediate solutions')
        # for j in range(N_iters):

        #     # alpha ~ total iterations
        #     alpha = 1-j/N_iters
        #     # alpha ~ iterations from same initial state
        #     alpha = float((iters_from_same_x0[j] + 1) / iters_per_x0)

        #     sol_j = jax.tree_util.tree_map(lambda z: z[j], outputs['forward_sol'])

        #     plot_forwardpass(sol_j, interp_ts, alpha=alpha, labels=(j==N_iters-1))

        # pl.legend()


        pl.figure('states vs iterations stats')
        N_t = 10
        ts = np.linspace(t0, tf, N_t)

        def all_states_t(t):
            return jax.vmap(lambda sol: sol.evaluate(t))(outputs['forward_sol'])

        # this has shape (N_t, N_iters, nx)
        all_statevecs = jax.vmap(all_states_t)(ts)



        pl.subplot(311)
        # update norm
        # difference of state vectors between iterations, norm across both time and state axis. 
        update_norm = np.linalg.norm(np.diff(all_statevecs, axis=1), axis=(0, 2)) / N_t
        pl.semilogy(update_norm, label='update norm, averaged over t')
        pl.legend()
        pl.xlabel('iterations')

        pl.subplot(312)
        # just for one iteration. 
        def PMP_residuals_iter(k, Nt):
            backsol = jax.tree_util.tree_map(lambda x: x[k], outputs['backward_sol']) 
            fwdsol = jax.tree_util.tree_map(lambda x: x[k], outputs['forward_sol']) 
            fct = lambda t: PMP_wrongness(t, backsol.evaluate(t), fwdsol)

            ts_fine = np.linspace(t0, tf, Nt)

            PMP_residuals = jax.vmap(fct)(ts_fine)
            return PMP_residuals

        # we expect this residual to be less smooth as a function of t than 
        # the update norm, thus finer discretisation is used here.
        Nt_fine = 100
        # and vmap for all iterations. this is of shape (N_iters, Nt_fine)
        all_PMP_residuals = jax.vmap(PMP_residuals_iter, in_axes=(0, None))(np.arange(N_iters), Nt_fine)

        # averaged over time axis. 
        mean_PMP_residuals = np.linalg.norm(all_PMP_residuals, axis=1) / Nt_fine
        pl.semilogy(mean_PMP_residuals, label='PMP residual, averaged over t')
        pl.legend()


        pl.subplot(313)
        v0s = jax.vmap(lambda sol: sol.evaluate(0.)[0])(outputs['backward_sol'])
        pl.plot(v0s, label='v(0)')
        pl.xlabel('iterations')
        pl.legend()

        # same data, but in a parametric form. 
        # same plot twice: markers plus translucent line. 
        pl.figure('PMP error vs update norm')
        pl.loglog(mean_PMP_residuals[1:], update_norm, linestyle='', marker='.', c='C0')
        pl.loglog(mean_PMP_residuals[1:], update_norm, alpha=0.2, c='C0')
        pl.xlabel('mean PMP residual')
        pl.ylabel('update norm')


        # plot something like || d/dt (x(t), lambda(t)) - PMP_RHS(x(t), lambda(t)) || as function of time? 
        # if PMP fulfilled to low tolerance we are (probably) not having any problems

        # pl.figure('backward pass')

        # plot the (last? all?) backward pass just to see how it handles stepsize selection etc.
        # also, ensure that S > 0 and S-S.T is small enough.

        pl.figure('solver stats')
        pl.plot(outputs['forward_sol'].stats['num_accepted_steps'], linestyle='', marker='.', label='forward steps')
        pl.plot(outputs['backward_sol'].stats['num_accepted_steps'], linestyle='', marker='.', label='backward steps')
        pl.xlabel('iteration')
        pl.legend()

        # astonishingly this works without any weird stuff :o very cool
        # but only the solutions corresponding to the last iteration at each initial state. 
        idxs = iters_from_same_x0 == (iters_per_x0 - 1)

        sols_pruned = jax.tree_util.tree_map(lambda z: z[idxs], outputs['forward_sol'])
        visualiser.plot_trajectories_meshcat(sols_pruned)


        def visualise_negative_hessian_direction(idx):
            '''
            For the given iteration, this finds the negative eigenvalues of the 
            hessian S(0), perturbs initial states in that direction and plots the
            resulting batch of forward simulations with meshcat. 
            '''

            # get solutions
            fw = jax.tree_util.tree_map(itemgetter(idx), outputs['forward_sol'])
            bw = jax.tree_util.tree_map(itemgetter(idx), outputs['backward_sol'])

            # find value S(0) and its eigendecomposition. 
            S0 = bw.evaluate(0)[2]

            vals, vecs = np.linalg.eigh(S0)

            # of shape (nx, n_negative_hessian_eigenvalues)
            negvecs = vecs[:, vals < 0]

            # so we can loop over it more neatly.
            negvecs = negvecs.T
            print(f'Found {negvecs.shape[0]} negative hessian direction(s) for iteration {idx}')

            for negvec in negvecs:

                test = negvec.T @ S0 @ negvec
                assert test < 0, 'messed up finding negative hessian eigenvalues'

                print(f'plotting perturbations in direction: {negvec}')
                print(f'eigenvalue = {test}')

                alphas = np.linspace(-5, 5, 201)

                x0 = fw.evaluate(0)

                x0s = x0[None, :] + alphas[:, None] * negvec[None, :]
                # emulates the *next* forward pass from "adversarially" perturbed x0
                sols = jax.vmap(ddp_forwardpass, in_axes=(None, None, 0))(fw, bw, x0s)
                visualiser.plot_trajectories_meshcat(sols)
 

        # show negative S eigenvalues. 
        # this as an image would be much cooler, with interpolation...
        eigs = lambda S: np.linalg.eig(S)[0].real

        # bit of a mouthful
        S_interp = jax.vmap(lambda sol: jax.vmap(sol.evaluate)(interp_ts))(outputs['backward_sol'])[2]

        all_min_eigs = jax.vmap(jax.vmap(eigs))(S_interp).min(axis=2)
        # adjust max index here to where completely weird stuff starts to happen

        # clip it so we can observe the zero crossing
        eig_clip = 50
        pl.figure('smallest eigenvalue of S(t) (red = negative)')
        pl.imshow(np.clip(all_min_eigs.T, -eig_clip, eig_clip), cmap='seismic_r', vmin=-eig_clip, vmax=eig_clip)
        pl.colorbar(); pl.tight_layout()
        pl.xlabel('iteration')
        pl.ylabel('t')
 


        # latest debugging spree
        '''
        ts = np.linspace(t0, tf, 512)

        # returns the input time series for one particular solution.
        # instead of the solution we pass the index (first axis in each leaf of sols pytree)
        # so we can access the previous solution too, to replicate exactly the forward pass.
        # if we jit this the whole thing crashes and just says "Killed" -- are we using too much memory?
        def idx_to_us(idx):
            prev_forward_sol = jax.tree_util.tree_map(itemgetter(idx-1), outputs['forward_sol'])
            current_backward_sol = jax.tree_util.tree_map(itemgetter(idx), outputs['backward_sol'])
            current_forward_sol = jax.tree_util.tree_map(itemgetter(idx), outputs['forward_sol'])
            args = (prev_forward_sol, current_backward_sol)

            t=1.1
            
            def u_t(t):
                return forwardpass_u(t, current_forward_sol.evaluate(t), args)

            us = jax.vmap(u_t)(ts)
            return us


        # plot angular rates
        N=50
        for j in (15, 16):
            pl.figure()
            idx = 800 + j  

            # plot the forward trajectory.
            pl.subplot(411)
            pl.gca().set_prop_cycle(None)
            pl.plot(outputs['forward_sol'].ts[j], outputs['forward_sol'].ys[j, :, 2], alpha=j/N, label='Phi')
            pl.plot(outputs['forward_sol'].ts[j], outputs['forward_sol'].ys[j, :, 5], alpha=j/N, label='omega')
            pl.legend()

            pl.subplot(412)
            pl.plot(outputs['backward_sol'].ts[j], outputs['backward_sol'].ys[1][j], label=problem_params['state_names'])
            pl.ylabel('costates')

            # plot the input 
            pl.subplot(413)
            pl.gca().set_prop_cycle(None)
            us = idx_to_us(idx)
            pl.plot(us, alpha=j/N, label=('u1', 'u2'))

            pl.subplot(414)
            # plot the backwardpass data.
            pl.plot(outputs['backward_sol'].ts[j], outputs['backward_sol'].ys[2][j].reshape(256, -1), color='C0', alpha=.2)
            # todo replace hardcoded 256 with maxsteps


        # it goes wrong for the first time at index 816. 


            # us = 

        '''


        pl.show()
        ipdb.set_trace()        

