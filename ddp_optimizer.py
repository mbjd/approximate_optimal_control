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


    # main iterations 
    for i in range(10): 

        # backward pass: Find the local value function expansion around the trajectory.
        # Hutter/Grandia says that the linearised pontryagin BVP (eq. 23) is solved by 
        # parameterising λ(t) = S(t) x(t) + s(t), and then solving (26):

        # S' = H_xx - f_x^T S - S f_x - S H S
        # s' = H_x - f_x^T s - S f_x - S H s

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

