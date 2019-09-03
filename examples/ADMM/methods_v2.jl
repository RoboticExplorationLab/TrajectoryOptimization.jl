function solve_admm_1slack(prob_lift, prob_load, admm_type, opts, n_slack=3)
    N = prob_load.N; dt = prob_load.dt

    # Problem dimensions
    num_lift = length(prob_lift)
    n_lift = prob_lift[1].model.n
    m_lift = prob_lift[1].model.m
    n_load = prob_load.model.n
    m_load = prob_load.model.m

    # Calculate cable lengths based on initial configuration
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    for i = 1:num_lift
        solve!(prob_lift[i],opts)

    end
    solve!(prob_load,opts)
    # return prob_lift,prob_load,1,1

    # Generate cable constraints
    X_lift = [deepcopy(prob_lift[i].X) for i = 1:num_lift]
    U_lift = [deepcopy(prob_lift[i].U) for i = 1:num_lift]

    X_load = deepcopy(prob_load.X)
    U_load = deepcopy(prob_load.U)

    cable_lift = [gen_lift_cable_constraints_1slack(X_load,
                    U_load,
                    i,
                    n_lift,
                    m_lift,
                    d[i],
                    n_slack) for i = 1:num_lift]

    cable_load = gen_load_cable_constraints_1slack(X_lift,U_lift,n_load,m_load,d,n_slack)

    self_col = [gen_self_collision_constraints(X_lift,i,n_lift,m_lift,r_lift,n_slack) for i = 1:num_lift]

    # Add system constraints to problems
    for i = 1:num_lift
        for k = 1:N
            prob_lift[i].constraints[k] += cable_lift[i][k]
            prob_lift[i].constraints[k] += self_col[i][k]
        end
    end

    for k = 1:N
        prob_load.constraints[k] += cable_load[k]
    end

    # Create augmented Lagrangian problems, solvers
    solver_lift_al = []
    prob_lift_al = []
    for i = 1:num_lift

        solver = TO.AbstractSolver(prob_lift[i],opts)
        prob = AugmentedLagrangianProblem(prob_lift[i],solver)
        prob.model = gen_lift_model(X_load,N,dt)

        push!(solver_lift_al,solver)
        push!(prob_lift_al,prob)
    end


    solver_load_al = TO.AbstractSolver(prob_load,opts)
    prob_load_al = AugmentedLagrangianProblem(prob_load,solver_load_al)
    prob_load_al.model = gen_load_model(X_lift,N,dt)

    for ii = 1:opts.iterations
        # Solve lift agents
        for i = 1:num_lift

            TO.solve_aula!(prob_lift_al[i],solver_lift_al[i])

            # Update constraints (sequentially)
            if admm_type == :sequential
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
        end

        # Update constraints (parallel)
        if admm_type == :parallel
            for i = 1:num_lift
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
        end

        # Solve load
        # return prob_lift,prob_load,1,1

        prob_load_al.model = gen_load_model(X_lift,N,dt)
        TO.solve_aula!(prob_load_al,solver_load_al)

        # Update constraints
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U

        for i = 1:num_lift
            prob_lift_al[i].model = gen_lift_model(X_load,N,dt)
        end

        # Update lift constraints prior to evaluating convergence
        for i = 1:num_lift
            TO.update_constraints!(prob_lift_al[i].obj.C,prob_lift_al[i].obj.constraints, prob_lift_al[i].X, prob_lift_al[i].U)
            TO.update_active_set!(prob_lift_al[i].obj)
        end
        # TO.update_constraints!(prob_load_al.obj.C,prob_load_al.obj.constraints, prob_load_al.X, prob_load_al.U)
        # TO.update_active_set!(prob_load_al.obj)

        max_c = max([max_violation(solver_lift_al[i]) for i = 1:num_lift]...,max_violation(solver_load_al))
        println(max_c)

        if max_c < opts.constraint_tolerance
            @info "ADMM problem solved"
            break
        end
    end

    return prob_lift_al, prob_load_al, solver_lift_al, solver_load_al
end

function gen_lift_cable_constraints_1slack(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = []
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                c[1] = u[end] - U_load[k][(agent-1) + 1]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[2] = u[end] - U_load[k][(agent-1) + 1]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1,end] = 1.0
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[2,end] = 1.0
                end
            end
        end
        if k == 1
            p_con = 1
        else
            k < N ? p_con = 1+1 : p_con = 1
        end
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_lift)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_load_cable_constraints_1slack(X_lift,U_lift,n,m,d,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = []

    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    c[_shift + 1] = U_lift[i][k][end] - u[(i-1) + 1]
                    _shift += 1
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift + 1] = U_lift[i][k][end] - u[(i-1) + 1]
                        _shift += 1
                    end
                end
            end
        end

        function ∇con(C,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    u_idx = ((i-1) + 1)
                    C[_shift + 1,n + u_idx] = -1.0
                    _shift += 1
                end
            else
                for i = 1:num_lift
                    x_pos = X_lift[i][k][1:n_slack]
                    x_load_pos = x[1:n_slack]
                    dif = x_pos - x_load_pos
                    C[i,1:n_slack] = -2*dif
                end
                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        u_idx = ((i-1) + 1)
                        C[_shift + 1,n + u_idx] = -1.0
                        _shift += 1
                    end
                end
            end
        end
        if k == 1
            p_con = num_lift
        else
            k < N ? p_con = num_lift*(1 + 1) : p_con = num_lift
        end
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function gen_lift_model(X_load,N,dt)
      model = Model[]

      for k = 1:N-1
        function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params)
            q = normalize(Quaternion(view(x,4:7)))
            v = view(x,8:10)
            omega = view(x,11:13)

            # Parameters
            m = params[:m] # mass
            J = params[:J] # inertia matrix
            Jinv = params[:Jinv] # inverted inertia matrix
            g = params[:gravity] # gravity
            L = params[:motor_dist] # distance between motors

            w1 = u[1]
            w2 = u[2]
            w3 = u[3]
            w4 = u[4]

            kf = params[:kf]; # 6.11*10^-8;
            F1 = kf*w1;
            F2 = kf*w2;
            F3 = kf*w3;
            F4 = kf*w4;
            F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

            km = params[:km]
            M1 = km*w1;
            M2 = km*w2;
            M3 = km*w3;
            M4 = km*w4;
            tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

            ẋ[1:3] = v # velocity in world frame
            ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
            Δx = X_load[k][1:3] - x[1:3]
            dir = Δx/norm(Δx)
            ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
            ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
            return tau, omega, J, Jinv
        end
        mm = midpoint(Model(quadrotor_lift_dynamics!,13,5,quad_params),dt)
        mm.info[:mass] = mass_load
        mm.info[:radius] = 0.2
        push!(model,mm)
    end
    model
end

function gen_lift_model_initial(xload,xlift0)

    function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params)
        q = normalize(Quaternion(view(x,4:7)))
        v = view(x,8:10)
        omega = view(x,11:13)

        # Parameters
        m = params[:m] # mass
        J = params[:J] # inertia matrix
        Jinv = params[:Jinv] # inverted inertia matrix
        g = params[:gravity] # gravity
        L = params[:motor_dist] # distance between motors

        w1 = u[1]
        w2 = u[2]
        w3 = u[3]
        w4 = u[4]

        kf = params[:kf]; # 6.11*10^-8;
        F1 = kf*w1;
        F2 = kf*w2;
        F3 = kf*w3;
        F4 = kf*w4;
        F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

        km = params[:km]
        M1 = km*w1;
        M2 = km*w2;
        M3 = km*w3;
        M4 = km*w4;
        tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

        ẋ[1:3] = v # velocity in world frame
        ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
        Δx = xload0[1:3] - xlift0[1:3]
        dir = Δx/norm(Δx)
        ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
        ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
        return tau, omega, J, Jinv
    end
    mm = Model(quadrotor_lift_dynamics!,13,5,quad_params)
    mm.info[:mass] = 0.85
    mm.info[:radius] = 0.5
    mm
end

function gen_load_model(X_lift,N,dt)
      model = Model[]
      for k = 1:N-1
          function double_integrator_3D_dynamics_load!(ẋ,x,u)
              Δx1 = X_lift[1][k][1:3] - x[1:3]
              Δx2 = X_lift[2][k][1:3] - x[1:3]
              Δx3 = X_lift[3][k][1:3] - x[1:3]

              u_slack1 = u[1]*Δx1/norm(Δx1)
              u_slack2 = u[2]*Δx2/norm(Δx2)
              u_slack3 = u[3]*Δx3/norm(Δx3)
              Dynamics.double_integrator_3D_dynamics!(ẋ,x,(u_slack1+u_slack2+u_slack3)/0.35)
          end
          mm = Model(double_integrator_3D_dynamics_load!,6,3)
          mm.info[:mass] = mass_load
          mm.info[:radius] = 0.5
          push!(model,midpoint(mm,dt))
    end
    model
end

function gen_load_model_initial(xload0,xlift0)

      function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
          Δx1 = (xlift0[1][1:3] - xload0[1:3])
          Δx2 = (xlift0[2][1:3] - xload0[1:3])
          Δx3 = (xlift0[3][1:3] - xload0[1:3])
          u_slack1 = u[1]*Δx1/norm(Δx1)
          u_slack2 = u[2]*Δx2/norm(Δx2)
          u_slack3 = u[3]*Δx3/norm(Δx3)
          Dynamics.double_integrator_3D_dynamics!(ẋ,x,(u_slack1+u_slack2+u_slack3)/mass_load)
      end
      mm = Model(double_integrator_3D_dynamics_load!,6,3)
      mm.info[:mass] = mass_load
      mm
end



function output_traj(prob,idx=collect(1:6),filename=joinpath(pwd(),"examples/ADMM/traj_output.txt"))
    f = open(filename,"w")
    x0 = prob.x0
    for k = 1:prob.N
        x, y, z, vx, vy, vz = prob.X[k][idx]
        str = "$(x-x0[1]) $(y-x0[2]) $(z) $vx $vy $vz"
        if k != prob.N
            str *= " "
        end
        write(f,str)
    end

    close(f)
end

function gen_self_collision_constraints(X_lift,agent,n,m,r_lift,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    p_con = num_lift - 1

    self_col_con = []

    for k = 1:N
        function col_con(c,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    # c[p_shift] = (r_lift + r_lift)^2 - norm(x_pos - x_pos2)^2
                    c[p_shift] = circle_constraint(x_pos,x_pos2[1],x_pos2[2],2*r_lift)
                    p_shift += 1
                end
            end
        end

        function ∇col_con(C,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    # dif = x_pos - x_pos2
                    # C[p_shift,1:n_slack] = -2*dif
                    C[p_shift,1] = -2*(x_pos[1] - x_pos2[1])
                    C[p_shift,2] = -2*(x_pos[2] - x_pos2[2])
                    p_shift += 1
                end
            end
        end

        push!(self_col_con,Constraint{Inequality}(col_con,∇col_con,n,m,p_con,:self_col))
    end

    return self_col_con
end

function get_quad_locations(x_load::Vector, d::Real, α=π/4, num_lift=3;
        config=:default, r_cables=[zeros(3) for i = 1:num_lift], ϕ=0.0)
    if config == :default
        h = d*cos(α)
        r = d*sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r*cos(θ), x_load[2] + r*sin(θ)]
        θ = range(0,2π,length=num_lift+1) .+ ϕ
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            if num_lift == 2
                x_lift[i][1:2] = circle(θ[i] + pi/2)
            else
                x_lift[i][1:2] = circle(θ[i])
            end
            x_lift[i][3] = z
            x_lift[i] += r_cables[i]  # Shift by attachment location
        end
    elseif config == :doorway
        y = x_load[2]
        fan(θ) = [x_load[1] - d*sin(θ), y, x_load[3] + d*cos(θ)]
        θ = range(-α,α, length=num_lift)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            x_lift[i][1:3] = fan(θ[i])
        end
    end
    return x_lift
end

function quad_obstacles(door=:middle)
    r_cylinder = 0.1
    _cyl = []
    h = 3 - 0*1.8  # x-loc [-1.8,2.0]
    w = 0.5      # doorway width [0.1, inf)
    off = 0.0    # y-offset [0, 0.6]
    door_width = 1.0
    off += door_location(door)
    push!(_cyl,(h,  w+off, r_cylinder))
    push!(_cyl,(h, -w+off, r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder, 3r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder, 3r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder+3r_cylinder, 4r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder-3r_cylinder, 4r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder+9r_cylinder, 6r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder-9r_cylinder, 6r_cylinder))
    # push!(_cyl,(h, -w+off-3r_cylinder, 3r_cylinder))
    x_door = [h, off, 0]
    return _cyl, x_door
end

function door_location(door, door_width=1.0)
    if door == :left
        off = door_width
    elseif door == :middle
        off = 0.0
    elseif door == :right
        off = -door_width
    else
        error(string(door) * " not a defined door")
    end
    return off
end

function update_lift_problem(prob_lift, prob_load::Problem, X_cache, U_cache, agent::Int, num_lift=3, n_slack=3)

    X_load = X_cache[1]
    U_load = U_cache[1]

    X_lift = X_cache[2:(num_lift+1)]
    U_lift = U_cache[2:(num_lift+1)]

    d = norm(prob_lift.x0[1:n_slack] - prob_load.x0[1:n_slack])

    cable_lift = gen_lift_cable_constraints_1slack(X_load,
                    U_load,
                    agent,
                    prob_lift[agent].model.n,
                    prob_lift[agent].model.m,
                    d,
                    n_slack)

    self_col = gen_self_collision_constraints(X_lift,agent,prob_lift.model.n,prob_lift.model.m,prob_lift.model.info[:radius],n_slack)

    # Add system constraints to problems
        for k = 1:N
            prob_lift.constraints[k] += cable_lift[k]
            prob_lift.constraints[k] += self_col[k]
        end

end

function update_load_problem(prob_lift,prob_load, X_lift, U_lift)
    n_load = prob_load.model.n
    m_load = prob_load.model.m
    n_slack = 3
    N = prob_load.N

    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]
    # cable_load = gen_load_cable_constraints_1slack(X_lift,U_lift,prob_load.model.n,prob_load.model.m,d,n_slack)
    #
    # for k = 1:N
    #     prob_load.constraints[k] += cable_load[k]
    # end
end
