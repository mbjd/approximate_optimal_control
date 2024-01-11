import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import pontryagin_utils

import ipdb
import time
import numpy as onp
import tqdm


def rrt_sample(problem_params, algo_params): 

    '''
    rough idea: 
    - sample initial trajectories from any half decent distribution over boundary of Xf
    - iteratively:
      - sample points from state space where we would like to know the optimal control 
      - guess the terminal conditions where a trajectory passing close to those points would start
      - evaluate the new trajectories and add them to the data set. 

    no fancy trajectory splitting yet. basically just the more practical version of
    the semester project. hopefully \o/
    '''

    K_lqr, P_lqr = pontryagin_utils.get_terminal_lqr(problem_params)

    # compute sqrtm of P, the value function matrix.
    P_eigv, P_eigvec = np.linalg.eigh(P_lqr)

    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P_lqr).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'


    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.
    key = jax.random.PRNGKey(0)
    normal_pts = jax.random.normal(key, shape=(algo_params['sampling_N_trajectories'], problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01
    # test if it worked
    xf_to_Vf = lambda x: x @ P_lqr @ x.T
    vTs =  jax.vmap(xf_to_Vf)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    xf_to_lamf = lambda x: x @ P_lqr * 2
    lamTs = jax.vmap(xf_to_lamf)(xTs)

    yTs = np.hstack([xTs, lamTs, np.zeros((algo_params['sampling_N_trajectories'], 1))])

    pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)
    # pontryagin_solver = pontryagin_utils.make_pontryagin_solver(problem_params, algo_params)

    # sols = vmap(pontryagin_solver)(boundary conditions)
    # now with vT also vmapped :) 
    vmap_pontryagin_solver = jax.jit(jax.vmap(pontryagin_solver, in_axes=(0, 0, None)))
    sols = vmap_pontryagin_solver(yTs, vTs, problem_params['V_max'])

    def find_yfs(sols):

        # finds the (non-extended) state at the boundary of Xf. 
        # sols a vmapped solution object, coming from vmap_pontryagin_solver. 
        # sols.ys.shape == (N_trajectories, N_timesteps, 2*nx+1)
        # sols.ts.shape == (N_trajectories, N_timesteps)

        # TODO jit this and adapt assertions to some jax pendant? 
        # surely this is not the main performance bottleneck though

        vf_diffs = sols.ts[:, 0] - vT
        assert vf_diffs.max() <= 1e-8, 'ODEs started outside of Xf'

        # spent ages figuring this one out. found on https://github.com/patrick-kidger/diffrax/blob/0ee47c98efe6de80388cce50eae80f91736047d1/test/test_vmap.py

        # y0_interp can be NaN in some places, if vT is outside the domain of the sol.
        # this can happen if vT is slightly larger due to numerical errors. 
        # in these cases, we just take instead the state where the integration actually 
        # started, hoping the difference will be quite small. 
        yfs_interp = jax.vmap(lambda sol: sol.evaluate(vT))(sols)

        # NaN -> 0. even though we are masking these out by multiplying by False (=0),
        # we need to clean NaNs because NaN * 0 == NaN.
        yfs_interp_nonan = np.nan_to_num(yfs_interp)  # NaN -> 0
        yfs_actual = sols.ys[:, 0, :]

        take_actual = (vf_diffs > 0)[:, None]  # extra dim to broadcast correctly
        take_interp = np.logical_not(take_actual)

        yfs = yfs_interp_nonan * take_interp + yfs_actual * take_actual

        # 
        # somehow, yfs_interp are *all* firmly outside of our Xf :(
        # no clue why exactly. in general though, the current way is still kind of 
        # messy due to the difference between the linearised and nonlinear system
        # and its associated cost function, value, optimal control etc. 
        # maybe it is more pragmatic to: 
        # - start all trajectories *inside* the set
        # - afterwards find the spot where the interpolation leaves Xf, by 
        #   bisection/regula falsi etc. 
        # maybe we also just have too large numerical errors close to the equilibrium? 
        # very plausible since the reparameterisation introduces a singularity at eq. 
        
        assert not np.isnan(yfs).any(), 'NaNs slipped thorugh :/'

        return yfs



    all_sol_ys = sols.ys

    def propose_new_xf_softmax(all_sol_ys, new_state):

        # propose a new terminal state x_f which starts a 
        # trajectory hopefully coming close to our point. 

        # this basically just comes up with convex combination weights 
        # roughly inspired by the neg. distances to each trajectory. 

        # new_state.shape  == (nx,)
        # all_sol_ys.shape == (N_trajectories_total, N_timesteps, 2nx+1)

        k = 12
        # find distance to all other trajectories.
        # just w.r.t. euclidean distance and only at saved points, not interpolated.
        # first we find the closest point on each trajectory, then the k closest trajectories. 
        state_diffs = all_sol_ys[:, :, 0:6] - new_state   # (N_traj, N_timesteps, nx)
        state_dists = np.linalg.norm(state_diffs, axis=2) # (N_traj, N_timesteps)
        shortest_traj_dists = state_dists.min(axis=1)

        # idea for other distance-like function: integral over trajectory of something like 
        # exp(-||distance||^2) or something similar. 

        neg_dists, indices = jax.lax.top_k(-shortest_traj_dists, k)

        # softmax only outputs valid convex combination weights. (x>0, 1.T x = 1)
        # even more basic approach -- just 1/n (equal weights?)
        cvx_combination_weights = jax.nn.softmax(neg_dists)

        # this also produces a costate and value but taking it from the terminal V fct
        # is definitely better, so we discard the interpolated values here. 
        proposed_yf = cvx_combination_weights @ all_sol_ys[indices, 0, :]
        proposed_xf = proposed_yf[0:problem_params['nx']]

        # to ensure that e get *some* state actually on the boundary of Xf: 
        # (this is not a projection!)

        # Xf = {x: x' (P/vT) x <= 1}
        #    = {x: || sqrtm(P/vT) x ||_2^2 <= 1}
        #    = {x = (P/vT)^(-1/2) : || z ||_2^2 <= 1}     (z = sqrtm(P/vT) x)

        # so, to adjust the proposal, we can: 

        # transform linearly so Xf is unit ball.
        proposed_zf = (Phalf @ proposed_xf) / np.sqrt(vT)
        # normalise to get on its surface
        new_proposed_zf = proposed_zf / np.linalg.norm(proposed_zf)
        # transform back. 
        new_proposed_xf = (Phalf_inv @ new_proposed_zf) * np.sqrt(vT)

        # UPDATE: found out that this is equivalent to just interpolating in original 
        # space and then scaling to be on ∂Xf. More in handwritten notes from 2023-01-10
        
        return new_proposed_xf


    def propose_new_xf_lstsq(all_sol_ys, new_state):

        # propose a new terminal state x_f which starts a 
        # trajectory hopefully coming close to our point. 

        # this version forms a local quadratic interpolation of V to estimate
        # V(new_state). Then all close trajectories are evaluated at V(new_state) and we 
        # find the linear combination of the corresponding states coming closest to new_state. 
        # should work, except if we have different local optima. not sure what to do then. 

        # secondguessing: maybe doing this much work just to find the proposal is excessive? 
        # maybe we should go with the *really* simple methods like just roughly guessing something in the 
        # middle of the nearest trajectories, to be able to have more sim throughput? 
        # I guess the fundamental question is whether or not simulations are expensive. For our simple toy models
        # simulations are very cheap. But if the model is more complex the relative cost of searching the dataset 
        # and finding precise convex combinations of trajectories might decrease...

        # atm i am leaving this unfinished, due to the messiness of having to estimate that many parameters just for 
        # a quick interpolation.  

        # also, instead of building the quadratic approximation like this, would it not be easier to do it 
        # using ricatti eqs around each trajectory? i guess that would go into the other direction of the tradeoff
        # assuming cheap simulation to minimise the extra stuff we need to do. 

        # new_state.shape  == (nx,)
        # all_sol_ys.shape == (N_trajectories_total, N_timesteps, 2nx+1)

        k = 12
        # find distance to all other trajectories.
        # just w.r.t. euclidean distance and only at saved points, not interpolated.
        # first we find the closest point on each trajectory, then the k closest trajectories. 
        state_diffs = all_sol_ys[:, :, 0:6] - new_state   # (N_traj, N_timesteps, nx)
        state_dists = np.linalg.norm(state_diffs, axis=2) # (N_traj, N_timesteps)
        shortest_traj_dists = state_dists.min(axis=1)

        neg_dists, indices = jax.lax.top_k(-shortest_traj_dists, k)

        # among those trajectories, somehow find a group of closest points? 
        
        # to address local/global issue: so some sort of k-means/EM thing to identify different branches? 
        # also, what if different trajectories pass close to the new state at vastly different value levels? 
        # is this just also a sign of local/global discrepancy? probably yes... should we just ignore that for now? 

        '''
        # form a quadratic local approximation of V to estimate V(x_new). 
        # How do we do this neatly? 

        # V(x) = x.T A_v x + b_v x + c_v. 
        # this is linear in parameters, so we can write θ_v = vec(A_v, b_v, c_v)
        # and V(x) = Φ(x) θ_v for some regressor matrix line Φ(x). What would that be?  

        # for each point x, we have observation of V(x) in R and dV(x) = λ(x) in R^nx. 
        # the regressor line would need to contain:
        # - all quadratic terms x_i * x_j, each one associated with an entry of A_v
        # - all linear terms x_i, each one associated with an entry of b_v
        # - one constant 1 term, for c_v. 

        # can we not do some jax magic to avoid the ugly vec(.) and associated bookkeeping? 
        # for example, we define f(A_v, b_v, c_v) = the loss on the dataset and somehow 
        # extract the linear system defining optimal A_v, b_v, c_v? 

        # somehow, get the loss as a function of theta_v, with fixed dataset. 

        def loss(theta_v, xs, Vs, Vxs):

            def single_loss(theta_v, x, V, Vx): 

                A_v, b_v, c_v = theta_v 

                V_loss = x.T @ A_v @ x + b_v @ x + c_v - V
                Vx_loss = 2 * A_v @ x + b_v.T - Vx

                # age old question: how to balance the two? 
                return V_loss + 10 * Vx_loss

            dataset_loss = jax.vmap(single_loss, in_axes=(None, 0, 0, 0))
        '''

            
        
        # how about just a linear approximation of V(x)? probably worse but easier. 
        # at least we get a learning exercise. 

        # but probably it will be biased. imagine a line approximating a quadratic-ish function. 
        # in the middle it is too high, outside too low. 

        # for each V(x) observation we directly get a line of the linear system: 
        # V(x) = b_v x + c_v. = [x1, ..., xnx, 1] @ [b_v, c_v].T.

        # for each i-th *entry* of dV(x), we also get a line: 
        # dV(x)/dx_i = b_v[i] (independent of x!)

        # so for one data point, the equation looks like this: 
        # 
        # [ x1, ..., x_nx, 1 ]                 [  V(x)  ]
        # [ 1            , 0 ]     [     ]     [ λ_1(x) ]
        # [   1      0   , 0 ]     [ b_v ]     [ λ_2(x) ]
        # [     1        , 0 ]     [     ]  =  [   .    ]   
        # [       1      , 0 ]     [ --- ]     [   .    ]     
        # [  0      1    , 0 ]     [ c_v ]     [   .    ]
        # [            1 , 0 ]                 [ λnx(x) ]

        # and we have to repeat this N times! obvs the system will be hugely overdetermined. 
        # this kind of sucks tbh. and if this doesn't suck yet, then expecially the quadratic 
        # approximation would suck! the regressor matrix would grow from
        #  (N_datapts * (nx+1), nx+1)
        # to
        #  (N_datapts * (nx+1), nx**2+nx+1)
        # both shapes are basically (N_observations, N_parameters)


        return new_proposed_xf


    def propose_new_xf_cvx(all_sol_ys, new_state):

        # yet another go at this. 
        # - choose some v_est "probably close" to V(x_new), such as the linear extrapolation from closest datapoint. 
        # - find convex combination of closest trajectories such that at v_est, it is closest to x_new. 
        #   with some jax qp solver, like https://github.com/kevin-tracy/qpax or jaxopt.BoxOSQP
        # if needed:
        # - from that point, re-estimate v_est using linear extrapolation from there. 
        # - go to start. 
        # probably 

        # this first bit is the same as the other version.
        # new_state.shape  == (nx,)
        # all_sol_ys.shape == (N_trajectories_total, N_timesteps, 2nx+1)

        k = 12
        # find distance to all other trajectories.
        # just w.r.t. euclidean distance and only at saved points, not interpolated.
        # first we find the closest point on each trajectory, then the k closest trajectories. 
        state_diffs = all_sol_ys[:, :, 0:6] - new_state   # (N_traj, N_timesteps, nx)
        state_dists = np.linalg.norm(state_diffs, axis=2) # (N_traj, N_timesteps)
        shortest_traj_dists = state_dists.min(axis=1)

        neg_dists, indices = jax.lax.top_k(-shortest_traj_dists, k)
        
        close_trajs = all_sol_ys[indices, :, :]

        # find k (again k, could use different constant) closest *points*. 
        # only search over already established closest trajectories. is it
        # even possible that some closest points are on other trajectories?

        # surely we could do this without this extra reshaping???
        close_traj_pts = close_trajs.reshape(-1, 13)
        _, closest_traj_pts_idx = jax.lax.top_k(
                -np.linalg.norm(close_traj_pts[:, 0:6] - new_state, axis=1),
                k
        )

        closest_pts = close_traj_pts[closest_traj_pts_idx]
        # every one of these points (v0, (x0, λ0, t)) induces a supporting hyperplane*
        # to V at the point x0, namely: V(x) >= v0 + λ0.T x. 
        # problem, we do not have those V's handy as the last state is t in the reparemterisation. 
        # thus, we also need the V's here somehow. 

        # easiest option: make extended state (x, λ, v, t).
        # next easiest options: store ts array alongside solutions, but name it vs. 
        # most future proof option: pass whole sols object in here

        V_lowerbounds = 

        ipdb.set_trace()

    
    # propose_new_xfs = jax.vmap(propose_new_xf, in_axes=(None, 0))


    maxmaxsteps = 0
    for i in range(algo_params['sampling_N_iters']):

        print(f'starting iteration {i}')

        # first mockup of sampling procedure, always starting at boundary of Xf

        # a) sample a number of points which are interesting
        print('    suggesting new points')

        # multiply by scale matrix from right bc it's a matrix of stacked row state vectors
        scale = np.diag(np.array([5, 5, np.pi, 5, 5, np.pi]))  # just guessed sth..
        key = jax.random.PRNGKey(i) 
        normal_pts = jax.random.normal(key, shape=(algo_params['sampling_N_trajectories'], problem_params['nx'])) 
        new_states_desired = normal_pts @ scale  # "at scale" -> hackernews salivating

        # b) based on those points, propose new terminal conditions yf that might lead 
        #    to trajectories passing close to the desired states. 
        print('    finding associated xfs')
        xf = propose_new_xf_cvx(all_sol_ys, new_states_desired[0])
        # new_xfs = propose_new_xfs(all_sol_ys, new_states_desired)
        new_lamfs = jax.vmap(xf_to_lamf)(new_xfs)
        new_tfs = np.zeros((algo_params['sampling_N_trajectories'], 1))
        new_Vfs = jax.vmap(xf_to_Vf)(new_xfs)

        # this should *really* not happen now that propose_new_xf also scales them back
        assert (new_Vfs <= vT + 1e-8).all(), 'trying to start trajectories outside Xf'

        new_yfs = np.hstack([new_xfs, new_lamfs, new_tfs])

        # c) evaluate those trajectories. 
        print('    simulating')
        new_sols = vmap_pontryagin_solver(new_yfs, new_Vfs, problem_params['V_max'])

        # d) 'clean' the trajectories so that the first saved point is the one on 
        #    the boundary of Xf. Then add that to our data set. 
        # yfs = find_yfs(new_sols)
        # new_ys_clean = new_sols.ys.at[:, 0, :].set(yfs)
        # all_sol_ys = np.concatenate([all_sol_ys, new_ys_clean], axis=0)

        # this ^ is ditched. had problems where evaluating trajectories at vT would
        # not give point on boundary due to nonlinearity or numerical errors. 
        # instead we initialise exactly on boundary of Xf, see propose_new_xf.


        # so instead d) add it to the dataset directly :) 
        all_sol_ys = np.concatenate([all_sol_ys, new_sols.ys], axis=0)

        steps = new_sols.stats['num_steps'].max()
        if steps > maxmaxsteps: 
            maxmaxsteps = steps
        print(f'    observed max steps taken: {steps}') 

    print(f'total max steps: {maxmaxsteps}')

    return all_sol_ys

# here a long list of considerations about the sampling method
# we want: the k trajectories which come closest to our point
# for that, first find the closest point on each trajectory

# assuming the trajectories are already "kind of" close, there should exist a 
# linear (or convex?) combination of them that goes thorugh our point. how do we find it?
# is it simpler if we assume the trajectories have been sampled at the same value levels? 
# ultra basic method: 
# - find some sort of coefficients quantifying how close each trajectory is to the sampled pt
# - squash them somehow so that they are positive and sum to 1
# - just guess the next terminal state with that squashed convex combination. 
# this will make sure we actually have a convex combination of the trajectories at hand, 
# regardless of irregular sampling of the closest points. 

# softmax is a natural first guess

# ALSO this does not consider at all what happens when conflicting local solutions are 
# among the closest. then, the convex combination of terminal conditions is probably
# meaningless. instead, we might have to do one/several of those things:
# - select only the "best" trajectories from the cloesest ones to start the next one
# - perform some k-means thing on (x, λ) to find the different local solutions
# - come up with some other smart thing

# to some order j, and associated matrices {A^0, ..., A^j}. but still maybe this is dumb. 
# is this even what we want? probably only when the trajectories are already close enough
# to warrant thinking about the Xf boundary in terms of its tangent space anyway. 
# let's try it like this for now -- the cases where xfs are far apart is anyway not the
# most interesting one.

# can we just not care and start the PMP from the proposed xf? possible
# issues: 

# - different linear and nonlinear system, maybe already wrong info on
# boundary of Xf. should not be large thoug if linearisation good and Xf
# small. 
# - bookkeeping a bit more cumbersome, need to retroactively find
# terminal costate to be compatible with re-using the dataset. 
