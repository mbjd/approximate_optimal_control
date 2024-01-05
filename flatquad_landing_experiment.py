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
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 1024,
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

    maxdiff = np.abs(Phalf @ Phalf - P0_inf).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'

    N_trajectories = 2**8

    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.
    # key = jax.random.PRNGKey(0)
    key = jax.random.PRNGKey(int(time.time() * 10000))
    normal_pts = jax.random.normal(key, shape=(N_trajectories, problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01
    # test if it worked
    vTs =  jax.vmap(lambda x: x @ P0_inf @ x.T)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    lamTs = jax.vmap(lambda x: x @ P0_inf * 2)(xTs)

    yTs = np.hstack([xTs, lamTs, np.zeros((N_trajectories, 1))])

    pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)

    # sols = jax.vmap(pontryagin_solver)
    sol = pontryagin_solver(yTs[0], vTs[0], 10)
    sols = jax.vmap(pontryagin_solver, in_axes=(0, None, None))(yTs, vTs[0], problem_params['V_max'])
    
    all_sol_ys = sols.ys

    # for i in range(10):
    if True:

        # first mockup of sampling procedure, always starting at boundary of Xf

        # a) sample a number of points which are interesting

        # multiply by scale matrix from right bc it's a matrix of stacked row state vectors
        scale = np.diag(np.array([5, 5, np.pi, 5, 5, np.pi]))  # just guessed sth..

        key = jax.random.PRNGKey(1) 
        new_states_desired = jax.random.normal(key, shape=(N_trajectories, problem_params['nx'])) @ scale
        
        # b) for each of those points, find the closest k trajectories.
        # here as a mockup just for the first interesting point. 
        # this has shape (N_trajectories, N_timesteps, nx)
        state_diffs = all_sol_ys[:, :, 0:6] - new_states_desired[1]

        # just euclidean norm here. trying other matrix norms = linearly transforming space. 
        # shaped (N_trajectories, N_timesteps)
        state_dists = np.linalg.norm(state_diffs, axis=2)

        k = 8
        # we want: the k trajectories which come closest to our point
        # for that, first find the closest point on each trajectory
        shortest_traj_dists = state_dists.min(axis=1)
        neg_dists, indices = jax.lax.top_k(-shortest_traj_dists, k)

        # propose a new terminal state xT which starts a trajectory hopefully coming close to our point. 
        # assuming the trajectories are already "kind of" close, there should exist a 
        # linear (or convex?) combination of them that goes thorugh our point. how do we find it?
        # is it simpler if we assume the trajectories have been sampled at the same value levels? 
        # (this is pretty easy with diffrax' interpolation)
        # basic approach: ??? no clue bruv --> thoughts on paper notes, very unfinished

        # could also do the good old move of 
        # - finding the closest point and two adjacent ones
        # - finding the parabola that maps trajectory parameter (value here) to distance exactly
        #   at those three points
        # - finding the minimum of the parabola in closed form
        # maybe this will be a good estimate of the actual closest distance (in ct. time)? 

        # ultra basic method: 
        # - find some sort of coefficients quantifying how close each trajectory is to the sampled pt
        # - squash them somehow so that they are positive and sum to 1
        # - just guess the next terminal state with that squashed convex combination. 
        # this will make sure we actually have a convex combination of the trajectories at hand, 
        # regardless of irregular sampling of the closest points. 

        # softmax is a natural first guess
        cvx_combination_weights = jax.nn.softmax(neg_dists)

        # 0 for initial condition. 
        proposed_yf = cvx_combination_weights @ all_sol_ys[indices, 0, :]


        # now, this yf will probably not be exactly on the boundary of Xf. 
        # I'm pretty sure there is some 'correct' way of projecting it onto Xf, along
        # the flow of the optimally controlled system, so it still represents the same trajectory. 

        # "cheap" way: locally form a linear approximation of the boundary of Xf and the flow, solve linear system. 
        # maybe better way: w/ explicit solution of the system, or small approximation of exp(At)? 
        # OR, we just don't care? but probably it will get worse and worse.
        # well, actually the terminal states only go *into* the set Xf, so maybe not too bad.
        
        # say we have the linear system x' = (A-BK) x. Say we know x(0) = x0. 
        # we want tau s.t. x(tau)' P0_inf x(tau) = vf. 

        # can we solve it in closed form? we have x(t) = exp((A-BK)t) x0, therefore:
        # x0' exp((A-BK)t)' P0_inf exp((A-BK)t) x0 = vf, solve for t. looks
        # hairy though. 

        # simple approach: just integrate numerically from our v to the desired v, with the
        # reparameterised linear dynamics. could just do one RK4 step for
        # example and call it a day. 

        # maybe a bisection-type thing is fastest, if we compute exp(At) with its taylor series up
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

        '''
        Acl = A - B @ K0_inf
        x0 = proposed_yf[0:6]

        def vtau(tau): 
            xtau = jax.scipy.linalg.expm(Acl * tau) @ x0
            vtau = xtau.T @ P0_inf @ xtau
            return vtau

        # we want: vtau(tau) - vT = 0. 
        # we know: solution tau < 0. (because we must integrate the optimally
        # controlled system backwards to go from inside Xf to its boundary)
        
        print('branching out...')
        vtau_guess = -1e-4
        while vtau(vtau_guess) - vT < 0: 
            vtau_guess = vtau_guess * 2
            print(vtau_guess)

        # now we know: vtau_guess <= tau <= vtau_guess/2
        lower = vtau_guess; upper = vtau_guess/2
        vlower = vtau(lower); vupper = vtau(upper)
        err = 1000
        print('honing in...')
        while err > 1e-10: 
            # propose a new point where the linear interpolation is 0. 
            # i.e. new point = a * lower + (1-a) * upper, with a such that: 
            # a * vlower + (1-a) * vupper = 0. 
            # a * vlower + vupper - a vupper = 0. 
            # a (vlower-vupper) + vupper = 0. 
            # a (vlower-vupper) = -vupper
            # a = -vupper/(vlower-vupper).
            # a = -vupper/(vlower-vupper)

            # it doesn't work with a from above, probably some error. 
            a = .5  # regular old bisection instead of regula falsi. 

            new_guess = a * lower + (1-a) * upper
            new_v = vtau(new_guess)

            print(f'new guess: {new_guess}')
            print(f'new v: {new_v}')

            if new_v - vT < 0: 
                upper = new_guess
                vupper = new_v
            else: 
                lower = new_guess
                vlower = new_v

            print(f'guesses = {(lower, upper)}')
            print(f'v-vT\'s = {(vlower-vT, vupper-vT)}')

            err = np.abs(new_v - vT)
            print(f'err mag = {err}')
            print('\n\n')


        plot=True

        if plot: 
            taus = np.linspace(vtau_guess, 0, 101)
            vs = [vtau(tau).item() for tau in taus]
            pl.plot(taus, vs)
            pl.scatter([lower], [vlower])
            pl.scatter([upper], [vupper])
            pl.show()
        '''

        ipdb.set_trace()
        


        



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

    def plot_trajectories_meshcat(sols, vis=None, arrows=False):

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

        def make_quad(vis, basepath, color=None, opacity=1/3):
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

        
        N_sols = sols.ys.shape[0]

        

        anim = meshcat.animation.Animation()

        # visualise which trajectories are close and which far from our sampled point. 
        # opacity ~ -shortest_traj_dists
        # closest = 0, furthest = 1
        inv_opacities = (shortest_traj_dists - shortest_traj_dists.min()) / (shortest_traj_dists.max() - shortest_traj_dists.min())
        
        opacities = 1 - inv_opacities

        for sol_i in tqdm.tqdm(range(N_sols)):
            quad_name = f'quad_{sol_i}'
            make_quad(vis, quad_name, opacity=float(opacities[sol_i]))

            min_t = sols.ys[sol_i, :, -1].min()
            for y in sols.ys[sol_i]:
                
                # inf = no more data 
                if np.any(y == np.inf):
                    break

                t = y[-1]
                anim_t = 25*float(t - min_t)

                with anim.at_frame(vis, anim_t) as frame:
                    move_quad(frame, quad_name, y)


        vis.set_animation(anim, repetitions=np.inf)

    # just one trajectory atm
    # ts = np.linspace

    plot_trajectories_meshcat(sols)

    # plot_trajectories(sols.ts, sols.ys)

    ipdb.set_trace()

