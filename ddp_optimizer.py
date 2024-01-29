import jax
import diffrax
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

    # the usual Hamiltonian. 
    H = lambda x, u, λ: l(t, x, u) + λ.T @ f(t, x, u)

    # the pre-optimised Hamiltonian. 
    def H_opt(x, λ): 
        u_star = pontryagin_utils.u_star_2d(x, λ, problem_params)
        return H(x, u_star, λ)
            





    Vf = lambda x: x.T @ x  # some sensible terminal value...

    forward_sol = init_sol


    # main iterations 
    for i in range(10): 

        # backward pass: Find the local value function expansion around the trajectory.
        # Hutter/Grandia says that the linearised pontryagin BVP (eq. 23) is solved by 
        # parameterising λ(t) = S(t) x(t) + s(t), and then solving (26):

        # S' = H_xx - f_x^T S - S f_x - S H S
        # s' = H_x - f_x^T s - S f - S H s

        # Here H and f are understood with u* substituted in already. but which x? δx? 
        # a bit confused about this, read paper again...
        # In this other paper (also from Hutter) https://arxiv.org/pdf/2101.06067.pdf
        # they write the parameterisation as λ(t) = S(t) δx + s(t) which looks more reasonable. 
        # then the δx is probably the deviation from the previous iterate at time t. 

        # then, for each t in our time interval, we have a local approximation
        # of the cost-to-go v(x) at some other state x: 

        # 0 ord.    |-- 1st order --|   |--------- 2nd order ---------|
        # v(x(t)) + s(t) @ (x - x(t)) + (1/2) (x - x(t))^T S (x - x(t))

        # the 0th order is obviously correct if v represents the actual
        # experienced cost-to-go on that trajectory.

        # the 1st and second order i am a bit confused about. In nice pure
        # optimal control theory the value gradient is obviously the costate.
        # but is a similar thing true about our local cost-to-go approximation
        # around this suboptimal trajectory? not so sure...

        # is this correct? this is basically a taylor polynomial right?  or, is
        # the resulting "V" not actually a local approximation of the cost to
        # go on the previous iterate trajectory, but rather the actual full
        # value function formed by "estimating" the optimal trajectory based on
        # a lin/quad taylor expansion at the current one? This would nicely
        # explain the newton-ish convergence. 

        # man why can I not just write the equations and implement without
        # worrying so much about intuitively understanding every single detail

        # a weekend passed. let us assume that eq. 26 in DOC (Hutter) is
        # describing everything for the "tangential" linear system, and the "x"
        # in (25) is actually a deviation w.r.t. the current iterate
        # trajectory, and we have
        #  λ_t(x) = S(t) (x - x_prev(t)) + s(t)
        # (and thus obviously ∇x λ = S)

        # therefore the full backward pass, with arguments, is this: 

        # S' = H_xx - f_x^T S - S f_x - S H S
        #      ^      ^           ^     ^ S(t) @ <pre-optimised hamiltonian at x(t), u(t) from forward pass> @ S
        #      ^      ^-          ^ df/dx(x(t), u(t)) (arguments from forward pass)
        #      ^ hessian w.r.t. x of pre-optimized hamiltonian at x(t), u(t)

        # s' = H_x - f_x^T s - S f - S H s

        def backwardpass_rhs(t, state, args):
            S, s = state  # pyTree magic :) 

            # how to give forwardsol nicely as an argument? 
            # is this te time to use "args"?          
            x = forward_sol.evaluate(t)  
                                        

            # we are integrating backwards along the "rolled out" trajectory. 
            # the tangential linear system gives rise to the costate λ = S (x - x_prev) + s
            # but because we are on the trajectory we have x = x_prev and λ = s.
            # is this reasoning correct? 
            λ = s
            H = H_opt(x, λ)

            # for this we need first and second derivatives of inner convex optimisation! 
            # so basically the only option is explicit convex opti. 

            H_x = jax.jacobian(H_opt, argnums=0)(x, λ)
            H_xx = jax.hessian(H_opt, argnums=0)(x, λ)
            
            f_x = jax.jacobian(f, argnums=0)(x, λ)  # be careful if t argument first...

            S_dot = H_xx - f_x.T @ S - S @ f_x - S @ H @ S
            s_dot = H_x - f_x.T @ s - S @ f(x, λ) - S @ H @ s

            return (S_dot, s_dot)

        term = diffrax.ODETerm(backwardpass_rhs)
        

        # TODO somehow pass in terminal LQR stuff, and make sure that the terminal 
        # state is close enough to the equilibrium.
        # V_inf(x) = x.T @ P_inf @ x 
        # Standard differentiation rules say:
        # V_inf_x(x) = 2 P_inf @ x
        # V_inf_xx(x) = 2 P_inf
        P_inf = np.eye(nx)
        
        # terminal conditions given by taylor expansion of terminal value (eq. 27)
        S_init = 2 * P_inf 
        s_init = 2 * P_inf @ forward_sol.evaluate(T)  # terminal state from forward sim

        init_state = (S_init, s_init)

        # TODO also get the specifics (adaptive controller, max steps, SaveAt) right. 
        backwardpass_sol = diffrax.diffeqsolve(term, Diffrax.Tsit5(), t0=T, t1=0, dt0=0.1, y0=init_state)

        # Then, first thing tomorrow: 
        # - work out the kinks above so it actually runs. 
        # - find some actually optimal trajectory by PMP backwards integration
        # - give it to this as an initial guess
        # - verify that the backward pass is consistent w/ this info
        #   (we should have the backward pass costate equal to the initial costate, 
        #   and if we forward sim again, obtain the same trajectory again)
        


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

