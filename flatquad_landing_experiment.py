#!/usr/bin/env python

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import pontryagin_utils

import ipdb
import time
import numpy as onp
import tqdm

if __name__ == '__main__':

    # classic 2D quad type thing. 6D state.

    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

    def f(t, x, u):

        # unpack for easier names
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        xdot = np.array([
            vx,
            vy,
            omega,
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr-Fl) * r / I,
        ])

        return xdot

    def l(t, x, u):
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        state_length_scales = np.array([1, 1, np.deg2rad(10), 1, 1, np.deg2rad(45)])
        Q = np.diag(1/state_length_scales**2)
        state_cost = x.T @ Q @ x + np.maximum(-posy, 0)**2  # state penalty? 

        # can we just set an input penalty that is zero at hover?
        # penalise x acc, y acc, angular acc here.
        # this here is basically a state-dependent linear map of the inputs, i.e. M(x) u with M(x) a 3x2 matrix.
        # the overall input cost will be acc.T M(x).T Q M(x) acc, so for each state it is still a nice quadratic in u.
        accelerations = np.array([
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr - Fl) * r / I,
        ])

        accelerations_lengthscale = np.array([1, 1, 1])

        input_cost = accelerations.T @ np.diag(1/accelerations_lengthscale**2) @ accelerations

        return state_cost + input_cost


    def h(x):
        # irrelevant if terminal constraint
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    state_length_scales = np.array([2, 1, 2, 1, np.deg2rad(45), np.deg2rad(45)])
    x_sample_cov = np.diag(state_length_scales**2)

    problem_params = {
        'system_name': 'flatquad',
        'f': f,
        'l': l,
        'h': h,
        'T': 2,
        'nx': 6,
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'terminal_constraint': True,
        'V_max': 500,
    }


    algo_params = {
            
            'sampling_N_trajectories': 2**5,
            'sampling_N_iters': 32,
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 128,
            'pontryagin_solver_dense': False,
    }

    # similar to orbits example: find local lqr value function


    # important to linearise about equilibrium, not zero!
    # that linearisation would be uncontrollable in this case.
    # equilibrium in y direction:
    #   0 = np.cos(Phi) * (Fl + Fr) / m - g,
    #   0 =               (Fl + Fr) / m - g,
    #   Fl + Fr = mg
    # so, to also have no torque, we need: Fl = Fr = mg/2

    u_eq = np.ones(problem_params['nu']) * m * g / 2

    x_eq = np.zeros(problem_params['nx'])
    A = jax.jacobian(f, argnums=1)(0., x_eq, u_eq)
    B = jax.jacobian(f, argnums=2)(0., x_eq, u_eq).reshape((problem_params['nx'], problem_params['nu']))
    Q = jax.hessian(l, argnums=1)(0., x_eq, u_eq)
    R = jax.hessian(l, argnums=2)(0., x_eq, u_eq)

    # cheeky controllability test
    ctrb =  np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(problem_params['nx'])])
    if np.linalg.matrix_rank(ctrb) < problem_params['nx']:
        raise ValueError('linearisation not controllable aaaaah what did you do idiot')

    K0_inf, P0_inf, eigvals = pontryagin_utils.lqr(A, B, Q, R)

    # compute sqrtm of P, the value function matrix.
    # this is to later map points x from a small unit sphere to points Phalf x which are on a small value level set.
    P_eigv, P_eigvec = np.linalg.eigh(P0_inf)
    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T
    Phalf_inv = np.linalg.inv(Phalf)  # only once! 

    maxdiff = np.abs(Phalf @ Phalf - P0_inf).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'


    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.
    key = jax.random.PRNGKey(0)
    normal_pts = jax.random.normal(key, shape=(algo_params['sampling_N_trajectories'], problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01
    # test if it worked
    xf_to_Vf = lambda x: x @ P0_inf @ x.T
    vTs =  jax.vmap(xf_to_Vf)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    xf_to_lamf = lambda x: x @ P0_inf * 2
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

    def propose_new_xf(all_sol_ys, new_state):
        # propose a new terminal state xT which starts a trajectory hopefully coming close to our point. 
        # with new_state.shape == (nx,)

        k = 8
        # find distance to all other trajectories.
        # just w.r.t. euclidean distance and only at saved points, not interpolated.
        # first we find the closest point on each trajectory, then the k closest trajectories. 
        state_diffs = all_sol_ys[:, :, 0:6] - new_state   # (N_traj, N_timesteps, nx)
        state_dists = np.linalg.norm(state_diffs, axis=2) # (N_traj, N_timesteps)
        shortest_traj_dists = state_dists.min(axis=1)

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
        
        return new_proposed_xf
    
    propose_new_xfs = jax.vmap(propose_new_xf, in_axes=(None, 0))


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
        new_xfs = propose_new_xfs(all_sol_ys, new_states_desired)
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
        print(f'    observed max steps taken: {steps}') 
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
        
        # for now, here is the 'brute force' solution with exp(At) calculated by
        # numpy. although this is almost certainly a bad solution. 

        # i am thinking that the practical solution is this: 
        # - start at proposed xf, don't care about linearisation vs actual
        # system error, select small Xf to avoid problems. 
        # - after finding solution, replace the first state of the trajectory 
        # with the interpolation from diffrax for the correct value level. 
        
        # HOWEVER this will be slightly uglier with diffrax, as the vmapped solver at the moment
        # does not support having different times. is this a problem though? 

        # let's be precise here. the 'ts' (independent variable of the ode) currently is the 
        # value function (up to different better local solutions). the last variable of the
        # extended state (x, λ, t) is the physical time til reaching Xf (negative). 
        # so if we want to start trajectories at (slightly) different *value* levels
        # we do have a problem (with the reparameterised pontryagin solver). 

        # what if we ditch the reparameterisation? can't we just not swap time and value? 
        # in turn we can start the trajectories all at the same time and record the value easily.
        # after getting the solution we might shift the time so 0 is exactly where they enter 
        # Xf, and not where the trajectories end a tiny bit later. 
        # the value sublevel set is then not as nicely defined anymore (as in we don't *exactly*
        # integrate up to the boundary), but we can still have a diffrax terminating event 
        # to stop once we have gone outside, so we don't make a cheeky NaN dough. 


        


        



    # having lots of trouble using sols.evaluate or sols.interpolation.evaluate in any way. thus, just use sol.ts \o/
    def plot_trajectories(ts, ys, color='green', alpha='.1'):

        # plot trajectories.
        pl.plot(ys[:, :, 0].T, ys[:, :, 1].T, color='green', alpha=.1)

        # sols.ys.shape = (N_trajectories, N_ts, 2*nx+1)
        # plot attitude with quiver.
        arrow_x = ys[:, :, 0].reshape(-1)
        arrow_y = ys[:, :, 1].reshape(-1)
        attitudes = ys[:, :, 4].reshape(-1)
        arrow_len = 0.5
        u = np.sin(-attitudes) * arrow_len
        v = np.cos(attitudes) * arrow_len

        pl.quiver(arrow_x, arrow_y, u, v, color='green', alpha=0.1)

    def plot_trajectories_meshcat(sols_ys, vis=None, arrows=False, reparam=True):

        '''
        tiny first draft of meshcat visualisation :o

        this is with the 2D quad model so 3D is kind of pointless but cool to have later.
        to do:
         - find out why the time needs to be scaled (otw too fast)
         - find a sensible way to plot a 'swarm' of multiple trajectories at once
         - force arrows to show input
         - do some 3d thing :)
         - separate this functionality into its nice own file
        '''

        # we now have the whole sols object. what do we do with it? 

        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf

        vis = meshcat.Visualizer()

        # scale force cylinder length like this:
        # vis['box/cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(0.1, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))

        arrow_length = .25

        def make_quad(vis, basepath, color=None, opacity=1):
            box_width = 1
            box_aspect = .1
            box = g.Box([box_width*box_aspect, box_width, box_width*box_aspect**2])

            # show "quadrotor" as a flat-ish long box
            if color is None:
                vis[basepath].set_object(box, g.MeshLambertMaterial(opacity=opacity))
            else:
                vis[basepath].set_object(box, g.MeshLambertMaterial(color, opacity=opacity))


            # also have two lines representing the rotor forces.
            # here they are of fixed length, we will animate their length
            # by setting a correspondingly scaling transform.
            # verts_l = np.array([[0, box_width/2, 0], [0, box_width/2, 1]]).T
            # vis['box']['arrow_l'].set_object(g.Line(g.PointsGeometry(verts_l), g.MeshBasicMaterial(color=0xff0000)))
            # verts_r = np.array([[0, -box_width/2, 0], [0, -box_width/2, 1]]).T
            # vis['box']['arrow_r'].set_object(g.Line(g.PointsGeometry(verts_r), g.MeshBasicMaterial(color=0xff0000)))

            # or, do them as cylinders to get thicker.
            # the cylinder is in a "cylinder frame" which moves the cylinder accordingly.

            if arrows:
                vis[basepath]['cyl_left_frame/cyl_left'].set_object(g.Cylinder(arrow_length, .01), g.MeshLambertMaterial(color=0xff0000))
                vis[basepath]['cyl_left_frame/cyl_left'].set_transform(onp.eye(4))
                vis[basepath]['cyl_left_frame'].set_transform(tf.concatenate_matrices(
                    tf.translation_matrix([0, -box_width/2, arrow_length/2]),
                    tf.rotation_matrix(np.pi/2, [1, 0, 0]),
                ))

                vis[basepath]['cyl_right_frame/cyl_right'].set_object(g.Cylinder(arrow_length, .01), g.MeshLambertMaterial(color=0xff0000))
                vis[basepath]['cyl_right_frame/cyl_right'].set_transform(onp.eye(4))

                vis[basepath]['cyl_right_frame'].set_transform(tf.concatenate_matrices(
                    tf.translation_matrix([0, box_width/2, arrow_length/2]),
                    tf.rotation_matrix(np.pi/2, [1, 0, 0]),
                ))

        # somehow though even when we put it into its own function it works perfectly when 
        # applied to the original visualiser but the force "arrows" are wrong when using frame
        def move_quad(vis, basepath, y):
            # vis = instance of meshcat visualiser, or frame in case of animation
            # y = extended system state with state, costate, time. 
            # names hardcoded ugly i know
            t = y[-1]
            T = tf.translation_matrix([0,y[0], y[1]])
            R = tf.rotation_matrix(y[2], np.array([1, 0, 0]))

            transform = tf.concatenate_matrices(T, R)
            vis[basepath].set_transform(transform)

            if arrows:
                nx = problem_params['nx']
                ustar = pontryagin_utils.u_star_2d(y[0:nx], y[nx:2*nx], problem_params)
                vis[basepath]['cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(ustar[0]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))
                vis[basepath]['cyl_right_frame/cyl_right'].set_transform(tf.scale_matrix(ustar[1]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))

        
        N_sols = sols_ys.shape[0]

        

        anim = meshcat.animation.Animation()

        # visualise which trajectories are close and which far from our sampled point. 
        # opacity ~ -shortest_traj_dists
        # closest = 0, furthest = 1
        # inv_opacities = (shortest_traj_dists - shortest_traj_dists.min()) / (shortest_traj_dists.max() - shortest_traj_dists.min())
        # opacities = 1 - inv_opacities

        for sol_i in tqdm.tqdm(range(N_sols)):
            quad_name = f'quad_{sol_i}'
            make_quad(vis, quad_name)

            min_t = sols_ys[sol_i, :, -1].min()
            for y in sols_ys[sol_i]:
                
                # data is inf-padded by diffrax. 
                if np.any(y == np.inf):
                    break

                t = y[-1]

                anim_t = 25*float(t - min_t)

                with anim.at_frame(vis, anim_t) as frame:
                    move_quad(frame, quad_name, y)


        vis.set_animation(anim, repetitions=np.inf)

    plot_trajectories_meshcat(all_sol_ys)

    # otherwise visualiser closes immediately
    ipdb.set_trace()


