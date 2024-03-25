import jax
import jax.numpy as np
import ipdb

import diffrax

import numpy as onp
import scipy


def u_star_2d(x, costate, problem_params, smooth=False, debug_oups=False):

    # calculate:
    #     u* = argmin_u l(t, x, u) + λ.T @ f(t, x, u)
    # as before we replace (without loss of information, bc. l quadratic in u and f control-affine):
    #     l by its second order taylor series l(t, x, u) = l_const + grad_l_u @ u + u.T @ (1/2 hess_l_u) @ u
    #     f by its linear version (is control affine)
    #     -> actually no, new way just approximate the complete H(u) as H0 + H_u u + u.T .5 H_uu u
    # all linearisations etc are done about u=0 (arbitrary choice)
    # l_const we can neglect

    # should we specify a custom jacobian-vector product of this function? 
    # or manually make an implementation that returns value and jacobian? 
    # because it can pretty much be taken from the PWA solution map
    # but not sure if it's worth the effort

    if not problem_params['nu'] == 2:
        raise NotImplementedError('this ustar function is for 2d only!')

    # we have only time invariant problems. if not change this.
    t = 0.
    zero_u = np.zeros(problem_params['nu'])

    # represent H(u) with its second order taylor poly -- by assumption they are equal :)
    H_fct = lambda u: problem_params['l'](x, u) + costate.T @ problem_params['f'](x, u)
    H0 = H_fct(zero_u)  # never needed this...
    H_u = jax.jacobian(H_fct)(zero_u)
    H_uu = jax.hessian(H_fct)(zero_u)

    # solve linear system: 0 = dH/du = d/du (H0 + H_u u + u.T H_uu/2 u) = H_u + H_uu u
    u_star_unconstrained = np.linalg.solve(H_uu, -H_u)

    # now handle the constraints with the simple algo from notes (in overleaf idea dump).
    lowerbounds = problem_params['U_interval'][0]
    upperbounds = problem_params['U_interval'][1]

    # find the lowest cost function on each constraint boundary.
    # 2 basic ways:
    # a) parameterise the active constraint subspace & minimise over that (1d) parameterisation
    # b) solve the linear system: < constraint active, cost gradient in constraint direction 0 >
    # here we go with a), representing each section of the constraint boundary as a line segment between two points.

    def ustar_over_line_segment(a, b):
        # find minimum of H_fct over the line {s a + (1-s) b,  0<=a<=1}.

        # first, find symbolically the jacobian dH/ds.
        # dH/ds = dH/du (evaluated at u(s)) @ du/ds
        # dH/du just as above: H_u + H_uu u.
        # du/ds from definition of line: a - b.
        # finding 0 = dH/ds is a 1D linear system, solved by simple division :)

        # write down dH/ds = (H_u + H_uu (s a + (1-s) b))) @ (a - b)  ...but left side as row vec.
        # = (H_u.T + s a.T H_uu + (1-s) b.T H_uu ) (a-b)
        # = (H_u.T + s (a.T H_uu - b.T H_uu) + b.T H_uu) (a-b)
        # = (H_u.T + b.T H_uu + s (a.T-b.T) H_uu) (a-b)
        # = (H_u.T + b.T H_uu) (a-b) + s (a-b).T H_uu (a-b)

        # s (a-b).T H_uu (a-b) = - (H_u.T + b.T H_uu) (a-b)
        # s = - (H_u.T + b.T H_uu) (a-b) / (a-b).T H_uu (a-b)
        # finally! division by 0 occurs if H_uu not positive definite or a=b. both avoidable

        # spent about an hour chasing this missing -
        s = -(H_u.T + b.T @ H_uu) @ (a-b) / ((a-b).T @ H_uu @ (a-b))
        s_constr = np.clip(s, 0, 1)
        return s_constr * a + (1-s_constr) * b

    # convex hull of input constraint set.
    cvx_hull = np.array([
        [lowerbounds[0], lowerbounds[1]],
        [lowerbounds[0], upperbounds[1]],
        [upperbounds[0], upperbounds[1]],
        [upperbounds[0], lowerbounds[1]],
    ])

    # call with vmap for each pair of adjacent vertices.
    boundary_candidates = jax.vmap(ustar_over_line_segment, in_axes=(0, 0))(cvx_hull, np.roll(cvx_hull, 1, 0))

    all_candidates = np.row_stack([u_star_unconstrained, boundary_candidates])

    all_Hs = jax.vmap(H_fct)(all_candidates)

    plot = False  # never again this works now
    if plot:
        import matplotlib.pyplot as pl

        xs = np.linspace(lowerbounds[0] - 10, upperbounds[0] + 10, 51)
        ys = np.linspace(lowerbounds[1] - 10, upperbounds[1] + 10, 51)
        xx, yy = np.meshgrid(xs, ys)
        xxyy = np.concatenate([xx[:, :, None], yy[:, :, None]], axis=2).reshape(-1, 2)

        # zz = jax.vmap(cost_fct)(xxyy).reshape(51, 51)
        zz = jax.vmap(lambda u: problem_params['l'](x, u) + costate.T @ problem_params['f'](x, u))(xxyy).reshape(51, 51)

        pl.contourf(xx, yy, zz, levels=100)

        # plot constraints
        pl.plot([lowerbounds[0], lowerbounds[0], upperbounds[0], upperbounds[0], lowerbounds[0]],
                [lowerbounds[1], upperbounds[1], upperbounds[1], lowerbounds[1], lowerbounds[1]], color='black',
                label='constraints')

        pl.scatter(all_candidates[:, 0], all_candidates[:, 1], label='candidate solutions')
        pl.scatter(ustar_overall[0], ustar_overall[1], color='red', label='best sol apparently')

        pl.legend()
        name=f'./plots/explicit_opti/{int(.5 + costate[-1]):04d}.png'
        pl.xlim((xs[0], xs[-1]))
        pl.ylim((ys[0], ys[-1]))
        pl.savefig(fname=name, dpi=300)
        pl.clf(); pl.close('all')
        print(name)


    if not smooth:
        # this is the standard case, where we literally choose the best candidate solution

        # if the unconstrained solution is outside of the constraints, make its cost +Inf.
        # TODO for not axis-aligned constraints, change this. could probably make this automatically with vertex repr
        # is_inside = (lowerbounds <= u_star_unconstrained).all() and (u_star_unconstrained <= upperbounds).all()
        unconstrained_is_outside = np.maximum(np.max(lowerbounds - u_star_unconstrained), np.max(u_star_unconstrained - upperbounds)) > 0
        # is_outside = np.logical_not(is_inside)
        penalty = unconstrained_is_outside * np.inf  # this actually works. 0 if False, inf if True
        all_Hs_adjusted = all_Hs + np.array([penalty, 0, 0, 0, 0])  # what if not 4 constraints?

        # find the best candidate solution
        best_candidate_idx = np.argmin(all_Hs_adjusted)

        ustar_overall = all_candidates[best_candidate_idx]

        # what if we do essentially the same thing again but with the taylor
        # expansion centered around current u*?  this should alleviate the problem 
        # of comparing large floats, if we drop the constant term, which we can without
        # changing anything about the solution. 

        # taylor expansion at 0: .5 u.T H_uu u + H_u(u=0) u + c
        # to find H_u(u*), evaluate this at u* and differentiate, giving: 
        # H_u(u*) = H_uu u* + H_u(u=0)
        # thus the new taylor expansion is: 
        # H(u - u*) + c = .5 (u-u*).T H_uu (u-u*) + [H_uu u* + H_u(0)] (u-u*)

        # if many active sets can also do this with just the best k from the first round
        H_u_new = H_uu @ ustar_overall + H_u
        H_suboptimality = lambda u: 0.5 * (u - ustar_overall).T @ H_uu @ (u - ustar_overall) + H_u_new @ (u - ustar_overall)

        all_Hs_new = jax.vmap(H_suboptimality)(all_candidates)
        all_Hs_adjusted_new = all_Hs_new + np.array([penalty, 0, 0, 0, 0])

        best_candidate_idx_new = np.argmin(all_Hs_adjusted_new)
        ustar_overall = all_candidates[best_candidate_idx_new]


        # maybe we are better off using the KKT conditions directly? 
        # they are, for problem min_x f(x) s.t. g(x) <= 0:
        # 1. stationarity
        #   f_x(x*) + mu.T g_x(x*) = 0
        # basically gradient zero, but with constraints. 
        # mu is a lagrange multiplier. if equality constraints, additional one. 
        # 2. primal feasibility
        #   g(x*) <= 0.
        # 3. dual feasibility
        #   mu >= 0.
        # 4. complementary slackness.
        #   mu.T @ g(x*) = 0
        # or simply stated: for each constraint, either the lagrange multiplier
        # is 0, or the constraint function is 0 (<=> constraint active), but not both
        # <sketch with boundary of quadrant in g-mu space highlighted as fat line>

        # can we for each solution compute some "KKT violation" penalty? 
        # basically like 10000 * max violation to any of these equations. 
        # maybe that is numerically better than just lowest cost. 
        # at least any cost comparisons or evaluations are out, and only the gradient remains.

        # probably being the KKT solution for some active set already implies primal
        # feasibility so we might ditch checking that. 

        # primal feasibility rules out infeasible points obviously.
        # what is the intuitive purpose of the other two? 

        # mu <= 0 would mean that a constraint is not active but in the wrong
        # direction, i.e. violated. meaning, we have assumed a constraint is
        # active but  can improve the objective by deactivating it, without it
        # being violated. geometrically, the downward direction points to the
        # interior of the feasible set, removing the need for that particular
        # boundary.

        # finally complementary slackness. if a candidate solution violates it, 
        # then what? then on one hand, the constraint is satisfied thus inactive. 
        # on the other hand, mu>0 means that we still "tilt" the cost function 
        # in the direction of that constraint, making us solve a different problem
        # than what we should. is this also already implied by only looking
        # at active set KKT solutions?

        debug_oups = {
            'all_candidates': all_candidates,
            'all_Hs': all_Hs,
            'all_Hs_adjusted': all_Hs_adjusted,
            'all_Hs_adjusted_new': all_Hs_adjusted_new,
            'best_candidate_idx': best_candidate_idx,
        }

        # return ustar_overall_new, debug_oups
        if debug_oups:
            # return ustar_overall, debug_oups
            # even if this branch is not taken jit is bothered by it and gives a TypeError...
            return ustar_overall
        else:
            return ustar_overall
    else:

        # because of chattering issues with the other method though, here we
        # explore what happens if we try to smooth out the kinks in u* as a
        # function of (x, \lambda) 

        # HOWEVER the chattering persists. the root cause is different: we
        # tried to compare relatively large floats which are very small
        # together, basically just an oversight concerning the limits of float
        # arithmetic.  this is addressed in the other if case now: basically
        # we evaluate the objective - the optimal objective again and do the
        # comparisons based on those, which are very small and thus have great
        # float resolution.

        # this is perhaps a bit overkill with all the transcendental functions. 
        # maybe drop in squareplus(x) = 1/2 (x + sqrt(x^2 + b)) looks like softmax. 

        # the unconstrained solution can violate some constraints. calculate here how much. 
        constraint_violations = np.concatenate([lowerbounds - u_star_unconstrained, u_star_unconstrained - upperbounds])
        smoothing_size = 0.01
        # this is a smooth approximation to max(all_constraint_violations)
        softmax_violation = jax.scipy.special.logsumexp(constraint_violations/smoothing_size) * smoothing_size

        # we are only interested in this violation if it is above 0, so we "clip" that with another logsumexp(0, .)
        # large penalty to approximate constraint well enough still. 
        penalty = 1000 * jax.scipy.special.logsumexp(np.array([0., softmax_violation])/smoothing_size)*smoothing_size

        # this is now what we want to minimise among all solution candidates, 
        all_Hs_adjusted = all_Hs + np.array([penalty, 0, 0, 0, 0])

        # and with a similar smoothing operation we find a "smooth argmin" of this array. 
        # no need for * smoothing_size here -- softmax output always sums to 1.
        weights = jax.nn.softmax(-all_Hs_adjusted / smoothing_size)        

        # like this the output is always a convex combination of candidate solutions. the weights vary smoothly
        # wrt all parameters and approximate the "nonsmooth" solution very vell except close to active set changes. 
        ustar_overall = weights.T @ all_candidates
        return ustar_overall




def u_star_general_activeset(x, costate, problem_params):

    raise NotImplementedError('not finished')

    '''
    was I fundamentally mistaken before about the nature of this problem? 

        u* = argmin_u H(x, u, lambda) = argmin_u l(x, u) + lambda.T @ f(x, u)

    obviously the whole problem changes linearly with lambda. but does that 
    say anything about how the *solution* changes with lambda? not so sure.

    the whole thing is a quadratic in u: 

        H(x, u, lambda) = H + H_u u + .5 u.T H_uu u

    where the RHS is evaluated at (x, 0, lambda) -- quadratic taylor expansion
    in u about 0. l is quadratic in u (by construction) and f is affine in u
    (control affine!) so this I am pretty sure about. 

    But: H changes as a function of x in possibly weird ways! therefore if we 
    want to solve the problem with KKT matrices, we have to recalculate the
    KKT matrices every time from the nonlinear system :(

    from a quick numerical check, it looks like: 
     - H_u is not constant across varying (x, u, lambda)
     - but H_uu is :) 

    this first one is only true if l_uu is also constant, which in our case it
    is. should we put that as basic assumption? or be more lax and say just
    that we assume we can reliably calculate (and autodiff) u*(x, lambda)? 

    therefore our QP looks like this: 

        u* = argmin_u .5 u.T H_uu u + H_u(x, 0 lambda) u (+ H)
        s.t. G u <= l  

    H_uu is constant so no arguments. H_u is evaluated at u=0, but (x, lambda)
    change.  The constant term H(x, 0, lambda) is irrelevant.

    This does not look that bad after all. there is one specific place for our
    parameter 

        p := H_u(x, 0, lambda) 

    to enter, and otherwise we have a constant QP. Now, i *think* what
    confused  me is this: it may well be that the QP solution is piecewise
    linear *in p*. but p as a function of (x, lambda) is NOT piecewise linear,
    easily verified by  plotting it along some line in (x, lambda) space. This
    is not a bad thing as long as we use the chain rule or jax correctly, and 
    not confusing grad_p u* with grad_(x, lam) u*. 

    In fact, H_u = l_u(x, u) + lambda.T @ g(x), if f(x, u) = fstate(x) + g(x) u
    so we see neatly where the nonlinearity comes from. 
    '''

    # TODO
    # - adapt constraint description to standard A x <= l
    # - write brute force "enumeration of active set" solver
    # - maybe depending on each problem we can already "prune" some impossible active sets
    #   e.g. when constraints are triangular (and strictly feasible point exists), not all can be active at once.

    # NEW constraint description -- be sure to update all problem_params:
    #     G u <= l
    # the other u_star_2d only handles box constraints.

    '''
    some new-ish thoughts. 
    we can easily write down all KKT matrices for all active sets. we can then do 
    one of two things. 
    1) explicitly invert each KKT matrix, to find the linear map from parameter 
       to solution (and lagrange multipliers) as a nice matrix. 
    2) store all KKT systems and solve online. 

    2 is certainly better numerically (matrix inversion bad!!!) but we will
    have to store differently sized matrices in some pytree thing. OTOH, with 1
    we might incur some numerical error but have a nice collection of equally sized 
    matrices we can store in an array of shape (N_activesets, dim(u)+dim(mu), dim(p)).
    
    checking KKT conditions afterwards should be the same regardless. I am pretty 
    sure now that directly checking KKT conditions is miles better than taking 
    the lowest-cost solution inside constraints, because for two close solutions
    with distance d, the objective only differs like d^2, whereas the KKT residuals
    (i think) are all linear-ish. therefore much better numerically. for another day :)

    '''

    # convert old to new: lb <= x  <=> -I @ x <= lb
    # lowerbounds = problem_params['U_interval'][0]
    # upperbounds = problem_params['U_interval'][1]
    G = np.vstack([-np.eye(2), np.eye(2)])
    l = np.concatenate(problem_params['U_interval'])

    # we have only time invariant problems. if not change this.
    t = 0
    zero_u = np.zeros(problem_params['nu'])

    # represent H(u) with its second order taylor poly -- by assumption they are equal :)
    # does jax really "cache" this stuff when jitting everything?
    # or does it "stupidly" evaluate the jacobian and hessian every time?
    H_fct = lambda u: problem_params['l'](x, u) + costate.T @ problem_params['f'](x, u)
    H0 = H_fct(zero_u)  # never needed this...
    H_u = jax.jacobian(H_fct)(zero_u)
    H_uu = jax.hessian(H_fct)(zero_u)

    # so, H(u) - H(0) = 1/2 u^T H_uu u + H_u u, which is the function we try to minimise.

    # solve linear system: 0 = dH/du = d/du (H0 + H_u u + u.T H_uu/2 u) = H_u + H_uu u
    # u_star_unconstrained = np.linalg.solve(H_uu, -H_u)

    # enumerate all possible active sets.
    # this is thanks to chatgpt completely: make array with
    N_bits = l.shape[0]  # number of inequality constraints
    int_array = np.arange(2 ** N_bits)

    # this has shape (2**N_bits, N_bits) and contains the binary representation of each number in its rows :)
    # much better than crooked itertools
    active_sets = ((integers_array[:, None] & (1 << np.arange(num_bits))) > 0).astype(bool)
    n_constraints_active = active_sets.sum(axis=1)

    # maybe in the future here throw out some impossible active sets.
    # actually, don't we kind of have to do this, since for parallel constraints
    # in the same active sets the KKT system will be unsolvable?
    # also, any active set with more than nu constraints is nonsensical (overdetermined)

    def solve_kkt_system(active_set):
        # from here: https://www.numerical.rl.ac.uk/people/nimg/course/lectures/parts/part4.2.pdf

        # active_set a bool vector of shape (n_constraints,)
        # maybe instead of this, just multiply the corresponding rows by zero?
        # for nice vmapping later...
        n_active = np.sum(active_set)
        G_active = G[active_set]
        l_active = l[active_set]

        kkt_matrix = np.vstack([
            np.hstack([H_uu, G_active.T]),
            np.hstack([G_active, np.zeros((n_active, n_active))])
        ])

        kkt_rhs = np.concatenate([-H_u.T, l_active])

        kkt_sol = np.linalg.solve(kkt_matrix, kkt_rhs)
        u, neg_y = np.split(kkt_sol, [problem_params['nu']])

        return u

    # then:
    # - solve kkt system for all active sets (or only ones where G_active has full rank?)
    # - find the lowest-cost solution that does not violate any constraints up to some tol.
    # - return it.




def define_backward_solver(problem_params, algo_params):

    # define RHS of the optimally controlled system in backward time
    # this is basically the PMP plus an extra derivative to get vxx
    def f_extended(t, state, args=None):

        # state variables = x and quadratic taylor expansion of V.
        x   = state['x']
        v   = state['v']
        vx  = state['vx']

        if algo_params['pontryagin_solver_vxx']:
            vxx = state['vxx']

        nx = problem_params['nx']

        H = lambda x, u, λ: problem_params['l'](x, u) + λ.T @ problem_params['f'](x, u)

        # RHS of the necessary conditions without the hessian.
        def pmp_rhs(state, costate):

            u_star = u_star_2d(state, costate, problem_params)
            nx = problem_params['nx']

            state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
            costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)

            return state_dot, costate_dot



        # we calculate this here one extra time, could be optimised
        u_star = u_star_2d(x, vx, problem_params)

        # calculate all the RHS terms
        v_dot = -problem_params['l'](x, u_star)
        x_dot, vx_dot = pmp_rhs(x, vx)


        # and pack them in a nice dict for the state.
        state_dot = dict()
        state_dot['x'] = x_dot
        state_dot['t'] = 1.
        state_dot['v'] = v_dot
        state_dot['vx'] = vx_dot

        if algo_params['pontryagin_solver_vxx']:
            # do everything related to the hessian calculation only here.
            full_jacobian = jax.jacobian(pmp_rhs, argnums=(0, 1))(x, vx)
            (fx, flam), (gx, glam) = full_jacobian

            vxx_dot = gx + glam @ vxx - vxx @ fx - vxx @ flam @ vxx

            state_dot['vxx'] = vxx_dot

        if args is not None and args == 'debug':
            # hacky way to get debug output.
            # just be sure to have args=None within anything jitted.
            aux_output = {
                'fx': fx,
                'flam': flam,
                'gx': gx,
                'glam': glam,
                'u_star': u_star,
            }

            return state_dot, aux_output

        return state_dot

    # could just as well take the algo_params already here w/ lexical closure...
    # actually *should* do that probably since they are already baked into f_extended.
    # therefore it would be very confusing if we tried to change them outside, this 
    # function respects the change but f_extended does not. 
    def solve_backward(y_f, algo_params):

        state_f = y_f

        term = diffrax.ODETerm(f_extended)


        relax_factor = 1.
        step_ctrl = diffrax.PIDController(
            rtol=relax_factor*algo_params['pontryagin_solver_rtol'],
            atol=relax_factor*algo_params['pontryagin_solver_atol'],
            # dtmin=problem_params['T'] / algo_params['pontryagin_solver_maxsteps'],
            dtmin = 0.0005,  # just to avoid getting stuck completely
            dtmax = 0.5,
        )

        saveat = diffrax.SaveAt(steps=True, dense=True, t0=True, t1=True)

        T = algo_params['pontryagin_solver_T']

        if 'vxx_max_norm' in algo_params and algo_params['pontryagin_solver_vxx']:

            terminating_event = diffrax.DiscreteTerminatingEvent(
                cond_fn = lambda state, **kwargs: np.linalg.norm(state.y['vxx']) > algo_params['vxx_max_norm']
            )

            backward_sol = diffrax.diffeqsolve(
                term, diffrax.Tsit5(), t0=0., t1=-T, dt0=-0.1, y0=state_f,
                stepsize_controller=step_ctrl, saveat=saveat,
                max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
                discrete_terminating_event=terminating_event,
            )
        else:
            backward_sol = diffrax.diffeqsolve(
                term, diffrax.Tsit5(), t0=0., t1=-T, dt0=-0.1, y0=state_f,
                stepsize_controller=step_ctrl, saveat=saveat,
                max_steps = algo_params['pontryagin_solver_maxsteps'], throw=algo_params['throw'],
            )

        return backward_sol

    return solve_backward, f_extended


def u_star_new(x, costate, problem_params):

    assert problem_params['nu'] == 1

    # basically a rewrite of the above mess in a single function, that hopefully works
    #   u* = argmin_u l(x, u) + λ.T @ f(t, x, u)

    # because everything is quadratic in u, we can solve this quite easily
    #   u* = argmin_u u.T R u + λ.T @ g(x, u) @ u
    #   0  = grad_u (...) = (R + R') u + λ.T @ g(x, u)
    # we find R and g with autodiff because we are lazy like that.
    # WARNING this will silently fail when l is not quadratic or f is not linear in u

    # we have only time invariant problems. if not change this.
    t = 0
    zero_u = np.zeros(problem_params['nu'])

    # === get the relevant matrices ===

    # this was actually the mistake before.
    # the hessian of u' R u is R + R', not R (if symmetric = 2R)
    hess_u_l = jax.hessian(problem_params['l'], argnums=2)(t, x, zero_u)  # shape (nx, nx)
    R = 0.5 * hess_u_l

    # g(x, u) = grad_f_u
    grad_f_u = jax.jacobian(problem_params['f'], argnums=2)(t, x, zero_u) # shape (nx, nu)

    # === solve the linear system: grad_u H(x, u, λ) = 0 ===
    u_star_unconstrained = np.linalg.solve(R + R.T, -costate.T @ grad_f_u)
    u_star = np.clip(u_star_unconstrained, *problem_params['U_interval'])
    return u_star






def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller
    (without the weird onp and * matrix multiplication stuff)

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    
    this X here is apparently such that LQR value = 0.5 x.T X x.
    """
    # ref Bertsekas, p.151
    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)
    # compute the LQR gain
    K = np.linalg.inv(R) @ (B.T @ X)
    eigVals = np.linalg.eigvals(A - B @ K)

    if not (eigVals.real < 0).all():
        raise ValueError('LQR closed loop not stable...')


    print(' ~~~ LQR timescale info ~~~')
    ratio = eigVals.real.min() / eigVals.real.max()
    print(f'closed loop pole ratio: {ratio:.2f}' )

    p = eigVals.real.min()
    print(f'fastest pole: λ = {p:.2f} Hz, τ = {-1/p:.2f} s')
    p = eigVals.real.max()
    print(f'slowest pole: λ = {p:.2f} Hz, τ = {-1/p:.2f} s')

    return K, X, eigVals



def get_terminal_lqr(problem_params):
    '''
    a wrapper for the above lqr function that does some sanity checks 
    and extracts the local dynamics & cost from the full system 
    '''

    x_eq = problem_params['x_eq']
    u_eq = problem_params['u_eq']
    f = problem_params['f']
    l = problem_params['l']

    assert np.allclose(f(x_eq, u_eq), 0), '(x_eq, u_eq) does not seem to be an equilibrium'

    A = jax.jacobian(f, argnums=0)(x_eq, u_eq)
    B = jax.jacobian(f, argnums=1)(x_eq, u_eq).reshape((problem_params['nx'], problem_params['nu']))
    Q = jax.hessian(l, argnums=0)(x_eq, u_eq)
    R = jax.hessian(l, argnums=1)(x_eq, u_eq)

    # cheeky controllability test
    ctrb =  np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(problem_params['nx'])])
    if np.linalg.matrix_rank(ctrb) < problem_params['nx']:
        raise ValueError('linearisation not controllable aaaaah what did you do idiot')

    K_lqr, P_lqr, _ = lqr(A, B, Q, R)

    return K_lqr, P_lqr

