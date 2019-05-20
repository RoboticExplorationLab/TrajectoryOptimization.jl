using Ipopt

function gen_ipopt_functions(prob::Problem, solver::DIRCOLSolver)

    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    p_custom = sum(num_constraints(prob))
    p_total = p_colloc + p_custom

    # Create initial primals
    Z0 = Primals(prob,true)
    NN = length(Z0)
    X0,U0 = Z0.X, Z0.U

    # Get constraint jacobian sparsity structure
    jac_structure = spzeros(p_total, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    #################
    # COST FUNCTION #
    #################
    function eval_f(Z)
        Z = Primals(Z,X0,U0)
        cost(prob, Z)
    end

    ###########################
    # COLLOCATION CONSTRAINTS #
    ###########################
    function eval_g(Z, g)
        copyto!(solver.Z.Z, Z)
        Z = Primals(Z, X0, U0)
        dynamics!(prob, solver, Z)
        traj_points!(prob, solver, Z)
        update_constraints!(g, prob, solver, Z)
    end


    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
        copyto!(solver.Z.Z, Z)
        Z = Primals(Z, X0, U0)
        cost_gradient!(grad_f, prob, solver, Z)
    end

    #######################
    # CONSTRAINT JACOBIAN #
    #######################
    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows, r)
            copyto!(cols, c)
        else
            copyto!(solver0.Z.Z, Zsol.Z)
            Z = Primals(Z, X0, U0)
            dynamics!(prob, solver, Z)
            traj_points!(prob, solver, Z)
            calculate_jacobians!(prob, solver, Z)
            constraint_jacobian!(vals, prob, solver, Z)
        end
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function gen_ipopt_functions2(prob::Problem)

    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    p_custom = sum(num_constraints(prob))
    p_total = p_colloc + p_custom

    # Create initial primals
    Z0 = Primals(prob,true)
    NN = length(Z0)
    X0,U0 = Z0.X, Z0.U
    part_z = create_partition(n,m,N,N)

    # Get constraint jacobian sparsity structure
    jac_structure = spzeros(p_total, NN)
    # collocation_constraint_jacobian_sparsity!(jac_structure, prob)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    #################
    # COST FUNCTION #
    #################
    function eval_f(Z)
        X,U = unpackZ(Z, part_z)
        cost(prob.obj, X, U)
    end

    ###########################
    # COLLOCATION CONSTRAINTS #
    ###########################
    function eval_g(Z, g)
        X,U = unpackZ(Z, part_z)
        update_constraints!(g, prob, X, U)
    end


    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
        X,U = unpackZ(Z, part_z)
        cost_gradient!(grad_f, prob, X, U)
    end

    #######################
    # CONSTRAINT JACOBIAN #
    #######################
    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows, r)
            copyto!(cols, c)
        else
            X,U = unpackZ(Z, part_z)
            # collocation_constraint_jacobian!(vals, prob, X, U)
            constraint_jacobian!(vals, prob, X, U)
        end
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function remove_bounds!(prob::Problem)
    n,m,N = size(prob)
    bounds = [BoundConstraint(n,m) for k = 1:prob.N]

    # Initial Time step
    if :bound ∈ labels(prob.constraints[1])
        bnd_init = remove_bounds!(prob.constraints[1])[1]
    else
        bnd_init = bounds[1]
    end
    bounds[1] = BoundConstraint(n,m, x_min=prob.x0, u_min=bnd_init.u_min,
                                     x_max=prob.x0, u_max=bnd_init.u_max)

    # All time steps
    for k = 2:prob.N
        bnd = remove_bounds!(prob.constraints[k])
        if !isempty(bnd)
            bounds[k] = bnd[1]::BoundConstraint
        end
    end

    # Terminal time step
    if :goal ∈ labels(prob.constraints[N])
        goal = pop!(prob.constraints[N])
        xf = zeros(n)
        evaluate!(xf, goal, zero(xf))
        term_bound = BoundConstraint(n,m, x_min=-xf, u_min=bounds[N-1].u_min,
                                          x_max=-xf, u_max=bounds[N-1].u_max)
        bounds[N] = term_bound::BoundConstraint
    end
    return bounds
end

function remove_goal_constraint!(prob::Problem)
    xf = zero(prob.x0)
    goal = pop!(prob.constraints[prob.N], :goal)
    evaluate!(xf, goal, zero(xf))
    return -xf
end

function get_bounds(prob::Problem, bounds::Vector{<:BoundConstraint})
    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    Z = Primals(prob, true)

    Z.equal ? uN = N : uN = N-1
    x_U = [zeros(n) for k = 1:N]
    x_L = [zeros(m) for k = 1:N]
    u_U = [zeros(m) for k = 1:uN]
    u_L = [zeros(m) for k = 1:uN]
    for k = 1:uN
        x_U[k] = bounds[k].x_max
        x_L[k] = bounds[k].x_min
        u_U[k] = bounds[k].u_max
        u_L[k] = bounds[k].u_min
    end
    if !Z.equal
        x_U = bounds[N].x_max
        x_L = bounds[N].x_min
    end
    z_U = Primals(x_U,u_U)
    z_L = Primals(x_L,u_L)

    # Constraints
    p = num_constraints(prob)
    g_U = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    g_L = [PartedVector(prob.constraints[k]) for k = 1:N-1]
    push!(g_U, PartedVector(prob.constraints[N], :terminal))
    push!(g_L, PartedVector(prob.constraints[N], :terminal))
    for k = 1:N
        if p[k] > 0
            g_L[k].inequality .= -Inf
        end
    end
    g_U = vcat(zeros(p_colloc), g_U...)
    g_L = vcat(zeros(p_colloc), g_L...)

    convertInf!(z_U.Z)
    convertInf!(z_L.Z)
    convertInf!(g_U)
    convertInf!(g_L)
    return z_U.Z, z_L.Z, g_U, g_L
end

function solve!(prob::Problem, solver::DIRCOLSolver)

    prob0 = copy(prob)
    bnds = remove_bounds!(prob0)
    solver0 = DIRCOLSolver(prob0, solver.opts)
    # solver0 = solver
    n,m,N = size(prob0)
    eval_f, eval_g, eval_grad_f, eval_jac_g = gen_ipopt_functions2(prob0) #, solver0)

    Z0 = Primals(prob, true)

    NN = N*(n+m)
    p = num_constraints(prob0)
    p_colloc = num_colloc(prob0)
    p_custom = sum(p)
    P = p_colloc + p_custom
    nG = p_colloc*2(n+m) + sum(p[1:N-1])*(n+m) + p[N]*n
    nH = 0

    z_U, z_L, g_U, g_L = get_bounds(prob0, solver0, bnds)

    problem = Ipopt.createProblem(NN, z_L, z_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)
    problem.x = Z0.Z


    # Set options
    options=Dict{String,Any}()
    dir = root_dir()
    opt_file = joinpath(dir,"ipopt.opt")
    addOption(problem,"option_file_name",opt_file)
    for (opt,val) in pairs(options)
        addOption(problem,opt,val)
    end
    # if solver.opts.verbose == false
    #     addOption(problem,"print_level",0)
    # end

    problem
end

function gen_ipopt_prob(prob::Problem)
    prob = copy(prob)
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    P = p_colloc + sum(p) - n
    NN = N*(n+m)
    nG = p_colloc*2(n+m)
    nH = 0

    solver = DIRCOLSolver(prob)
    bnds = remove_bounds!(prob)
    z_U, z_L, g_U, g_L = get_bounds(prob, solver, bnds)
    @show P
    @show length(g_U)

    eval_f, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_ipopt_functions2(prob)

    problem = createProblem(NN, z_L, z_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)

    opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
    addOption(problem,"option_file_name",opt_file)

    return problem
end

function solve_ipopt(prob::Problem)
    n,m,N = size(prob)

    Z0 = Primals(prob, true)
    part_z = create_partition(n,m,N,N)

    problem = gen_ipopt_prob(prob)
    problem.x = copy(Z0.Z)
    solveProblem(problem)
    return Primals(problem.x, part_z)
end


function gen_ipopt_functions3(prob::Problem{T}) where T
    n,m,N = size(prob)
    NN = N*(n+m)
    p_colloc = num_colloc(prob)
    p = num_constraints(prob)
    P = p_colloc + sum(p)
    dt = prob.dt

    pcum = cumsum(p)
    part_z = create_partition(n,m,N,N)
    part_f = create_partition2(prob.model)
    constraints = prob.constraints
    ∇F         = [PartedMatrix(zeros(T,n,n+m),part_f)           for k = 1:N]
    ∇C         = [PartedMatrix(T,constraints[k],n,m,:stage) for k = 1:N-1]
    ∇C         = [∇C..., PartedMatrix(T,constraints[N],n,m,:terminal)]
    C          = [PartedVector(T,constraints[k],:stage)     for k = 1:N-1]
    C          = [C...,  PartedVector(T,constraints[N],:terminal)]

    jac_structure = spzeros(P, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    function eval_f(Z)
        X,U = unpack(Z,part_z)
        cost(prob.obj, X, U)
    end

    function eval_grad_f(Z, grad_f)
        X,U = unpack(Z, part_z)
        cost_gradient!(grad_f, prob, X, U)
    end

    function eval_g(Z, g)
        X,U = unpack(Z,part_z)
        g_colloc = reshape(view(g,1:p_colloc), n, N-1)
        g_custom = view(g,p_colloc+1:length(g))
        insert!(pcum, 1, 0)
        p = num_constraints(prob)
        pcum = [0; cumsum(p)]

        fVal = [zero(X[1]) for k = 1:N]
        Xm = [zero(X[1]) for k = 1:N-1]

        # Calculate midpoints
        for k = 1:N
            evaluate!(fVal[k], prob.model, X[k], U[k])
        end
        for k = 1:N-1
            Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
        end
        fValm = zero(X[1])
        for k = 1:N-1
            Um = (U[k] + U[k+1])*0.5
            evaluate!(fValm, prob.model, Xm[k], Um)
            g_colloc[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
        end

        for k = 1:N
            if p[k] > 0
                k == N ? part = :terminal : part = :stage
                part_c = create_partition(prob.constraints[k], part)
                inds = pcum[k] .+ (1:p[k])
                if k == N
                    evaluate!(PartedArray(view(g_custom, inds), part_c), prob.constraints[k], X[k])
                else
                    evaluate!(PartedArray(view(g_custom, inds), part_c), prob.constraints[k], X[k], U[k])
                end
            end
        end
    end

    # Calculate jacobian
    function calc_block!(vals::PartedMatrix, F1,F2,Fm,dt)
        In = Diagonal(I, n)
        Im = Diagonal(I, m)
        vals.x1 .= dt/6*(F1.xx + 4Fm.xx*( dt/8*F1.xx + In/2)) + In
        vals.u1 .= dt/6*(F1.xu + 4Fm.xx*( dt/8*F1.xu) + 4Fm.xu*(Im/2))
        vals.x2 .= dt/6*(F2.xx + 4Fm.xx*(-dt/8*F2.xx + In/2)) - In
        vals.u2 .= dt/6*(F2.xu + 4Fm.xx*(-dt/8*F2.xu) + 4Fm.xu*(Im/2))
        return nothing
    end

    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows,r)
            copyto!(cols,c)
        else
            X,U = unpack(Z, part_z)

            # Compute dynamics jacobians
            F = [PartedMatrix(zeros(n,n+m), part_f) for k = 1:N]
            for k = 1:N
                jacobian!(F[k], prob.model, X[k], U[k])
            end

            # Calculate midpoints
            fVal = [zeros(n) for k = 1:N]
            Xm = [zeros(n) for k = 1:N-1]

            for k = 1:N
                evaluate!(fVal[k], prob.model, X[k], U[k])
            end
            for k = 1:N-1
                Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
            end

            # Collocation jacobians
            Fm = zero(∇F[1])
            n_blk = 2(n+m)n
            off = 0
            In = Matrix(I,n,n)
            Im = Matrix(I,m,m)
            part = create_partition2((n,),(n,m,n,m), Val((:x1,:u1,:x2,:u2)))
            for k = 1:N-1
                block = PartedArray(reshape(view(vals, off .+ (1:n_blk)), n, 2(n+m)), part)
                Um = (U[k] + U[k+1])/2
                jacobian!(Fm, prob.model, Xm[k], Um)
                calc_block!(block, F[k], F[k+1], Fm, dt)
                off += n_blk
            end

            # General constraint jacobians
            p = num_constraints(prob)
            for k = 1:N
                if k == N
                    n_blk = p[k]*n
                    part_c = create_partition2(prob.constraints[k], n, m, :terminal)
                    block = PartedArray(reshape(view(vals, off .+ (1:n_blk)), p[k], n), part_c)
                    jacobian!(block, prob.constraints[k], X[k])
                else
                    n_blk = p[k]*(n+m)
                    part_c = create_partition2(prob.constraints[k], n, m)
                    block = PartedArray(reshape(view(vals, off .+ (1:n_blk)), p[k], n+m), part_c)
                    jacobian!(block, prob.constraints[k], X[k], U[k])
                end
                off += n_blk
            end
        end

        return nothing
    end
    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function solve_ipopt(prob::Problem)
    prob = copy(prob)
    bnds = remove_bounds!(prob)
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    P = p_colloc + sum(p)
    NN = N*(n+m)
    nG = num_colloc(prob)*2*(n + m) + sum(p[1:N-1])*(n+m) + p[N]*n
    nH = 0

    z_U, z_L, g_U, g_L = get_bounds(prob, bnds)

    eval_f, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_ipopt_functions3(prob)

    problem = createProblem(NN, z_L, z_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)

    opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
    addOption(problem,"option_file_name",opt_file)
    solveProblem(problem)
    sol = Primals(problem.x, n, m)
    return sol, problem
end

function gen_ipopt_functions3(prob::Problem{T}, solver::DIRCOLSolver) where T
    n,m,N = size(prob)
    NN = N*(n+m)
    p_colloc = num_colloc(prob)
    p = num_constraints(prob)
    P = p_colloc + sum(p)
    dt = prob.dt

    part_f = create_partition2(prob.model)
    part_z = create_partition(n,m,N,N)
    pcum = cumsum(p)

    jac_structure = spzeros(P, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    function eval_f(Z)
        X,U = unpack(Z,part_z)
        cost(prob.obj, X, U)
    end

    function eval_grad_f(Z, grad_f)
        X,U = unpack(Z, part_z)
        cost_gradient!(grad_f, prob, X, U)
    end

    function eval_g(Z, g)
        X,U = unpack(Z,part_z)
        g_colloc = reshape(view(g,1:p_colloc), n, N-1)
        g_custom = view(g,p_colloc+1:length(g))
        insert!(pcum, 1, 0)
        p = num_constraints(prob)
        pcum = [0; cumsum(p)]

        fVal = solver.fVal  #[zero(X[1]) for k = 1:N]
        Xm = solver.X_  #[zero(X[1]) for k = 1:N-1]

        # Calculate midpoints
        for k = 1:N
            evaluate!(fVal[k], prob.model, X[k], U[k])
        end
        for k = 1:N-1
            Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
        end
        fValm = zero(X[1])
        for k = 1:N-1
            Um = (U[k] + U[k+1])*0.5
            evaluate!(fValm, prob.model, Xm[k], Um)
            g_colloc[:,k] = -X[k+1] + X[k] + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
        end

        for k = 1:N
            if p[k] > 0
                k == N ? part = :terminal : part = :stage
                part_c = create_partition(prob.constraints[k], part)
                inds = pcum[k] .+ (1:p[k])
                if k == N
                    evaluate!(PartedArray(view(g_custom, inds), part_c), prob.constraints[k], X[k])
                else
                    evaluate!(PartedArray(view(g_custom, inds), part_c), prob.constraints[k], X[k], U[k])
                end
            end
        end
    end

    # Calculate jacobian
    function calc_block!(vals::PartedMatrix, F1,F2,Fm,dt)
        In = Diagonal(I, n)
        Im = Diagonal(I, m)
        vals.x1 .= dt/6*(F1.xx + 4Fm.xx*( dt/8*F1.xx + In/2)) + In
        vals.u1 .= dt/6*(F1.xu + 4Fm.xx*( dt/8*F1.xu) + 4Fm.xu*(Im/2))
        vals.x2 .= dt/6*(F2.xx + 4Fm.xx*(-dt/8*F2.xx + In/2)) - In
        vals.u2 .= dt/6*(F2.xu + 4Fm.xx*(-dt/8*F2.xu) + 4Fm.xu*(Im/2))
        return nothing
    end

    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows,r)
            copyto!(cols,c)
        else
            X,U = unpack(Z, part_z)

            # Compute dynamics jacobians
            F = [PartedMatrix(zeros(n,n+m), part_f) for k = 1:N]
            for k = 1:N
                jacobian!(F[k], prob.model, X[k], U[k])
            end

            # Calculate midpoints
            fVal = [zeros(n) for k = 1:N]
            Xm = [zeros(n) for k = 1:N-1]

            for k = 1:N
                evaluate!(fVal[k], prob.model, X[k], U[k])
            end
            for k = 1:N-1
                Xm[k] = (X[k] + X[k+1])/2 + dt/8*(fVal[k] - fVal[k+1])
            end

            # Collocation jacobians
            Fm = PartedMatrix(prob.model)
            n_blk = 2(n+m)n
            off = 0
            In = Matrix(I,n,n)
            Im = Matrix(I,m,m)
            part = create_partition2((n,),(n,m,n,m), Val((:x1,:u1,:x2,:u2)))
            for k = 1:N-1
                block = PartedArray(reshape(view(vals, off .+ (1:n_blk)), n, 2(n+m)), part)
                Um = (U[k] + U[k+1])/2
                jacobian!(Fm, prob.model, Xm[k], Um)
                @show F[k].parts
                @show F[k+1].parts
                @show Fm.parts
                calc_block!(block, F[k], F[k+1], Fm, dt)
                off += n_blk
            end

            # General constraint jacobians
            p = num_constraints(prob)
            for k = 1:N-1
                n_blk = p[k]*(n+m)
                part_c = create_partition2(prob.constraints[k], n, m)
                block = PartedArray(reshape(view(vals, off .+ (1:n_blk)), p[k], n+m), part_c)
                jacobian!(block, prob.constraints[k], X[k], U[k])
                off += n_blk
            end
        end

        return nothing
    end
    return eval_f, eval_g, eval_grad_f, eval_jac_g
end
