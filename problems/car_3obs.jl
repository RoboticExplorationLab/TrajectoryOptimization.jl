
function DubinsCar(scenario=:three_obstacles)
    if scenario == :three_obstacles
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-2,
            penalty_scaling=100.,
            penalty_initial=0.1
        )

        #  Car w/ obstacles
        model = Dynamics.DubinsCar()
        n,m = size(model)

        N = 101 # number of knot points
        tf = 5.0
        dt = tf/(N-1)

        x0 = @SVector [0., 0., 0.]
        xf = @SVector [1., 1., 0.]

        Q = (1.0)*Diagonal(@SVector ones(n))
        R = (1.0e-1)*Diagonal(@SVector ones(m))
        Qf = 100.0*Diagonal(@SVector ones(n))
        obj = LQRObjective(Q,R,Qf,xf,N)

        # create obstacle constraints
        r_circle_3obs = 0.1
        circles_3obs = ((0.25,0.25,r_circle_3obs),(0.5,0.5,r_circle_3obs),(0.75,0.75,r_circle_3obs))
        n_circles_3obs = length(circles_3obs)

        circle_x = @SVector [0.25, 0.5, 0.75]
        circle_y = @SVector [0.25, 0.5, 0.75]
        circle_r = @SVector fill(r_circle_3obs, 3)

        circle_con = CircleConstraint(n, circle_x, circle_y, circle_r)
        con_obs = ConstraintVals(circle_con, 2:N-1)

        bnd = BoundConstraint(n,m, u_min=[-1,-3],u_max=[2,3])
        con_bnd = ConstraintVals(bnd, 1:N-1)

        goal_con = GoalConstraint(xf)
        con_xf = ConstraintVals(goal_con, N:N)

        conSet = ConstraintSet(n,m,[con_obs, con_xf], N)

        # Create problem
        U = [@SVector fill(0.01,m) for k = 1:N-1]
        car_3obs_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0)
        initial_controls!(car_3obs_static, U)
        return car_3obs_static, opts

    elseif scenario==:parallel_park
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-3,
            active_set_tolerance=1e-4
        )

        # model
        model = TrajectoryOptimization.Dynamics.DubinsCar()
        n,m = size(model)
        N = 101
        tf = 3.

        # cost
        x0 = @SVector [0., 0., 0.]
        xf = @SVector [0., 1., 0.]
        Qf = 100.0*Diagonal(@SVector ones(n))
        Q = (1e-2)*Diagonal(@SVector ones(n))
        R = (1e-2)*Diagonal(@SVector ones(m))

        # constraints
        u_bnd = 2.
        x_min = @SVector [-0.25, -0.001, -Inf]
        x_max = @SVector [0.25, 1.001, Inf]
        bnd = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd)
        goal = GoalConstraint(xf)

        # Constraint vals
        con_bnd = ConstraintVals(bnd, 1:N-1)
        con_goal = ConstraintVals(goal, N:N)

        # problem
        U = [@SVector fill(0.1,m) for k = 1:N-1]
        obj = LQRObjective(Q,R,Qf,xf,N)

        conSet = ConstraintSet(n,m,[con_bnd, con_goal], N)

        prob = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, U0=U)

        return prob, opts

    elseif scenario==:escape
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-3,
            penalty_scaling=100.,
            penalty_initial=10.,
            active_set_tolerance=1e-4
        )

        # Static Car Escape
        T = Float64;

        # model
        model = Dynamics.DubinsCar()
        n,m = size(model)
        x0 = @SVector [2.5,2.5,0.]
        xf = @SVector [7.5,2.5,0.]
        N = 101
        tf = 3.0

        # cost
        Q = (1e-3)*Diagonal(@SVector ones(n))
        R = (1e-2)*Diagonal(@SVector ones(m))
        Qf = 100.0*Diagonal(@SVector ones(n))
        obj = LQRObjective(Q,R,Qf,xf,N)

        # constraints
        r = 0.5
        s1 = 30; s2 = 50; s3 = 15

        circles_escape = NTuple{3,Float64}[]

        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(0.,i,r))
        end
        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(5.,i,r))
        end
        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(10.,i,r))
        end
        for i in range(0,stop=10,length=s2)
            push!(circles_escape,(i,0.,r))
        end
        for i in range(0,stop=3,length=s3)
            push!(circles_escape,(i,5.,r))
        end
        for i in range(5,stop=8,length=s3)
            push!(circles_escape,(i,5.,r))
        end

        n_circles_escape = 3*s1 + s2 + 2*s3

        circles_escape
        x,y,r = collect(zip(circles_escape...))
        x = SVector{n_circles_escape}(x)
        y = SVector{n_circles_escape}(y)
        r = SVector{n_circles_escape}(r)

        obs = CircleConstraint(n,x,y,r)
        con_obs = ConstraintVals(obs, 2:N-1)

        bnd = BoundConstraint(n,m,u_min=-5.,u_max=5.)
        con_bnd = ConstraintVals(bnd, 1:N-1)

        goal = GoalConstraint(xf)
        con_xf = ConstraintVals(goal, N:N)

        conSet = ConstraintSet(n,m,[con_obs, con_bnd, con_xf], N)

        # Build problem
        U0 = [@SVector ones(m) for k = 1:N-1]

        car_escape_static = Problem(model, obj, xf, tf;
            constraints=conSet, x0=x0)
        initial_controls!(car_escape_static, U0);

        X_guess = [2.5 2.5 0.;
                   4. 5. .785;
                   5. 6.25 0.;
                   7.5 6.25 -.261;
                   9 5. -1.57;
                   7.5 2.5 0.]
        X0_escape = interp_rows(N,tf,Array(X_guess'))
        initial_states!(car_escape_static, X0_escape)

        return car_escape_static, opts
    end
end
