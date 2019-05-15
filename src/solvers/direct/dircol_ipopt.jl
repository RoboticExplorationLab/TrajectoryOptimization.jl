
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
        Z = Primals(Z, X0, U0)
        dynamics!(prob, solver, Z)
        traj_points!(prob, solver, Z)
        update_constraints!(g, prob, solver, Z)
    end


    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
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
            Z = Primals(Z, X0, U0)
            dynamics!(prob, solver, Z)
            traj_points!(prob, solver, Z)
            calculate_jacobians!(prob, solver, Z)
            constraint_jacobian!(vals, prob, solver, Z)
        end
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function remove_bounds!(prob::Problem)
    bounds = [BoundConstraint(n,m) for k = 1:prob.N]
    for k = 1:prob.N
        bnd = remove_bounds!(prob.constraints[k])
        if !isempty(bnd)
            bounds[k] = bnd[1]::BoundConstraint
        end
    end
    return bounds
end

function get_bounds(prob::Problem, solver::DirectSolver, bounds::Vector{<:BoundConstraint})
    Z = solver.Z
    n,m,N = size(Z)
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

    p = num_constraints(prob)
    g_U = [PartedArray(zeros(p[k]), solver.C[k].parts) for k = 1:N]
    g_L = [PartedArray(zeros(p[k]), solver.C[k].parts) for k = 1:N]
    for k = 1:N
        if p[k] > 0
            g_L[k].inequality .= -Inf
        end
    end
    return z_U, z_L, g_U, g_L
end

function solve!(prob::Problem, solver::DIRCOLSolver)

    prob0 = copy(prob)
    bnds = remove_bounds!(prob0)
    solver0 = DIRCOLSolver(prob0, solver.opts)
    eval_f, eval_g, eval_grad_f, eval_jac_g = gen_ipopt_functions(prob, solver)

    Z0 = Primals(prob, true)

    NN = N*(n+m)
    p = num_constraints(prob0)
    p_colloc = num_colloc(prob0)
    p_custom = sum(p)
    P = p_colloc + p_custom
    nG = p_colloc*2(n+m) + sum(p[1:N-1])*(n+m) + p[N]*n
    nH = 0

    z_U, z_L, g_U, g_L = get_bounds(prob0, dircol0, bnds)
    z_U = z_U.Z
    z_L = z_L.Z
    g_U = vcat(zeros(p_colloc), g_U...)
    g_L = vcat(zeros(p_colloc), g_L...)



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
