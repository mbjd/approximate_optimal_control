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
    x0_orig = x0

    # this is to handle case where t0>t1 (if solution was obtained backwards)
    t0, tf = sorted([forward_sol.t0, forward_sol.t1])  

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

        H_x = jax.jacobian(H, argnums=0)(xbar, u_star, lam)

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
        # still unsure whether or not this makes it wrong (see idea dump)
        # indeed it seems there are differences between this and the other one which are
        # more than just numerical noise (about 1.5% at one particular point)
        S_dot = -H_opt_xx - H_opt_xlam @ S - (H_opt_xlam @ S).T - S.T @ H_opt_lamlam @ S


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

        return (v_dot, lam_dot, S_dot_three)

        # this does NOT seem to work....
        # return (v_dot, sdotnew, S_dot_three)

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

    # same as forwardpass_rhs but returns the input. mostly for plotting.
    def forwardpass_u(t, state, args):

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
        return u



    def ddp_backwardpass(forward_sol):

        # backward pass. (more comments in python for loop below.)
        term = diffrax.ODETerm(backwardpass_rhs)

        # this 0:nx is strictly only needed if we have extended state y = [x, lambda, v].
        # we are overloading this function to handle both extended and pure state. 
        # (this means we have to re-jit when changing between them)
        xf = forward_sol.evaluate(tf)[0:nx]  

        # terminal conditions given by taylor expansion of terminal value
        # these are static w.r.t. jit compilation. 
        v_T = xf.T @ P_lqr @ xf
        lam_T = 2 * P_lqr @ xf
        S_T = 2 * P_lqr 

        # instead of doing something about it, just pass it to the output 
        # so at least we know about it.
        # instead of this we might also output the SDF of Xf, v_T(xf) - v_f
        # xf_outside_Xf = v_T >= v_f * 1.001
        # or not, because this is just a function of the forward solution
        # which can be calculated outside whenever we want. 

        init_state = (v_T, lam_T, S_T)

        # relaxed tolerance - otherwise the backward pass needs many more steps
        # maybe it also "needs" the extra accuracy though...? 
        relax_factor = 10
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
        )
        saveat = diffrax.SaveAt(steps=True, dense=True)

        backward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=tf, t1=t0, dt0=-0.1, y0=init_state,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            args = forward_sol,
        )

        return backward_sol # , xf_outside_Xf

    def ddp_forwardpass(prev_forward_sol, backward_sol, x0):

        # forward pass. 
        term = diffrax.ODETerm(forwardpass_rhs)

        # same tolerance parameters used in pure unguided backward integration of whole PMP
        step_ctrl = diffrax.PIDController(
            rtol=algo_params['pontryagin_solver_rtol'],
            atol=algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
        )

        saveat = diffrax.SaveAt(steps=True, dense=True) 

        forward_sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=t0, t1=tf, dt0=0.1, y0=x0,
            stepsize_controller=step_ctrl, saveat=saveat,
            max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            args = (prev_forward_sol, backward_sol)
        )

        return forward_sol


    def scan_fct(carry, inp):

        # what do we need to carry? 
        # atm I think we really only need the forward_sol. 

        # later we might want to also have the previous backward sol, to 
        # evaluate the derivative of the forward trajectory as RHS(x(t)) 
        # instead of d/dt x(t). 

        # but for that we need to initialise already with a backward sol....
        # maybe we just need a "special" first iteration which takes a feedback
        # controller given as lambda(x), then computes the forward sol, and first 
        # backward sol while replacing the previous backward sol (which would be
        # needed to calculate RHS(x(t)) ) with the feedback controller.

        # also any step length/line search/convergence checks can be put here.

        prev_forward_sol, = carry
        x0 = inp

        out = dict()

        # the whole DDP loop is it not beautiful <3 
        backward_sol = ddp_backwardpass(prev_forward_sol)
        forward_sol = ddp_forwardpass(prev_forward_sol, backward_sol, x0)

        out['backward_sol'] = backward_sol
        out['forward_sol'] = forward_sol

        new_carry = forward_sol,

        return new_carry, out

    N_x0s = 64

    xf = forward_sol.evaluate(tf)[0:nx]  
    v_T = xf.T @ P_lqr @ xf
    lam_T = 2 * P_lqr @ xf
    S_T = 2 * P_lqr 
    init_state = (v_T, lam_T, S_T)


    # to catch the ipdb inside
    _ = backwardpass_rhs(tf, init_state, forward_sol)
    # this scan will only work if init_sol is without costate and value info
    # so here we perform a first "iteration". 

    # this works because while scan_fct works for state = [x, lambda, v] and [x]
    # because it basically just uses sol.ys[:, 0:nx]. When jitting though, which 
    # scan does with scan_fct, then the shapes have to be constant. 
    
    init_carry = forward_sol,
    new_carry, output = scan_fct((forward_sol,), x0)

    # sweep the Phi and omega states from x0 to 0. 
    x0_final = x0.at[np.array([2,5])].set(0)
    # go to weird position
    # x0_final = x0 + np.array([10, 0, 0, 0, 10, 0])
    x0_final = x0 + np.array([0, 0, 2*np.pi, 0, 0, 0])
    # don't sweep at all
    # x0_final = x0  
    alphas = np.linspace(0, 1,  N_x0s)

    # this is kind of ugly ikr
    # we stay at each x0 and run the ddp loop for $iters_per_x0 times. 
    # after that we modify the N_iters variable 
    iters_per_x0 = 5
    alphas = np.repeat(alphas, iters_per_x0)[:, None]  # to "simulate" several iterations per x0.
    N_iters = alphas.shape[0]
    iters_from_same_x0 = np.arange(N_iters) % iters_per_x0

    x0s = x0 * (1-alphas) + x0_final * alphas

    start_t = time.time()
    final_carry, outputs = jax.lax.scan(scan_fct, new_carry, x0s)
    print(f'first run (with jit): {time.time() - start_t}')

    # start_t = time.time()
    # final_ca1ry, outputs = jax.lax.scan(scan_fct, new_carry, x0s)
    # print(f'second run: {time.time() - start_t}')

    final_forwardsol, = final_carry



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
            



        interp_ts = np.linspace(t0, tf, 512)

        pl.figure('some random ass backward pass')
        sol = jax.tree_util.tree_map(itemgetter(3), outputs['backward_sol'])
        plot_backwardpass(sol, None)
        

        # plot the final solution - ODE solver nodes...
        pl.figure('final solution')

        plot_forwardpass(final_forwardsol, interp_ts)


        pl.figure('intermediate solutions')
        for j in range(N_iters):

            # alpha ~ total iterations
            alpha = 1-j/N_iters
            # alpha ~ iterations from same initial state
            alpha = float((iters_from_same_x0[j] + 1) / iters_per_x0)

            sol_j = jax.tree_util.tree_map(lambda z: z[j], outputs['forward_sol'])

            plot_forwardpass(sol_j, interp_ts, alpha=alpha, labels=(j==N_iters-1))

        pl.legend()





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

