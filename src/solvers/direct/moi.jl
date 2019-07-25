using MathOptInterface
const MOI = MathOptInterface

struct DIRCOLProblem{T} <: MOI.AbstractNLPEvaluator
    prob::Problem{T,Continuous}
    cost::Function
    cost_gradient!::Function
    solver::DIRCOLSolver{T,HermiteSimpson}
    jac_struct::Vector{NTuple{2,Int}}
    part_z::NamedTuple{(:X,:U), NTuple{2,Matrix{Int}}}
    p::NTuple{2,Int}    # (total constraints, p_colloc)
    nG::NTuple{2,Int}   # (total constraint jacobian, nG_colloc)
    zL
    zU
    gL
    gU
end

function DIRCOLProblem(prob::Problem{T,Continuous}, solver::DIRCOLSolver{T,HermiteSimpson}, zL, zU, gL, gU) where T
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    P = p_colloc + sum(p)
    NN = N*(n+m)
    nG_colloc = p_colloc*2*(n + m)
    nG = nG_colloc + sum(p[1:N-1])*(n+m) + p[N]*n

    part_z = create_partition(n,m,N,N)

    jac_structure = spzeros(nG, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)
    jac_struct = collect(zip(r,c))
    num_con = (P,p_colloc)
    num_jac = (nG, nG_colloc)
    DIRCOLProblem(prob, gen_stage_cost(prob), gen_stage_cost_gradient(prob), solver, jac_struct, part_z, num_con, num_jac, zL, zU, gL, gU)
end

MOI.features_available(d::DIRCOLProblem) = [:Grad, :Jac]
MOI.initialize(d::DIRCOLProblem, features) = nothing

MOI.jacobian_structure(d::DIRCOLProblem) = d.jac_struct
MOI.hessian_lagrangian_structure(d::DIRCOLProblem) = []

function MOI.eval_objective(d::DIRCOLProblem, Z)
    X,U = unpack(Z, d.part_z)
    d.cost(X, U, get_dt_traj(d.prob))
end

function MOI.eval_objective_gradient(d::DIRCOLProblem, grad_f, Z)
    X,U = unpack(Z, d.part_z)
    d.cost_gradient!(grad_f,X,U,get_dt_traj(d.prob))
end

function MOI.eval_constraint(d::DIRCOLProblem, g, Z)
    X,U = unpack(Z, d.part_z)
    P,p_colloc = d.p
    g_colloc = view(g, 1:p_colloc)
    g_custom = view(g, (p_colloc+1):P)

    collocation_constraints!(g_colloc, d.prob, d.solver, X, U)
    update_constraints!(g_custom, d.prob, d.solver, X, U)

    # cache c_max
    push!(d.solver.stats[:iter_time],time() - d.solver.stats[:iter_time][1])
    push!(d.solver.stats[:c_max],max_violation_dircol(d,Z,g))
end

function MOI.eval_constraint_jacobian(d::DIRCOLProblem, jac, Z)
    X,U = unpack(Z, d.part_z)
    n,m = size(d.prob)
    P,p_colloc = d.p
    nG_colloc = p_colloc * 2(n+m)
    jac_colloc = view(jac, 1:nG_colloc)
    collocation_constraint_jacobian!(jac_colloc, d.prob, d.solver, X, U)

    jac_custom = view(jac, nG_colloc+1:length(jac))
    constraint_jacobian!(jac_custom, d.prob, d.solver, X, U)
end

MOI.eval_hessian_lagrangian(::DIRCOLProblem, H, x, σ, μ) = nothing

function solve_moi(prob::Problem, dircol::DIRCOLSolver)
    opts = dircol.opts

    prob = copy(prob)
    bnds = remove_bounds!(prob)
    z_U, z_L, g_U, g_L = get_bounds(prob, bnds)
    n,m,N = size(prob)
    NN = (n+m)*N

    # Get initial condition
    Z0 = Primals(prob, true)

    # Create NLP Block
    has_objective = true
    d = DIRCOLProblem(prob, dircol, z_L, z_U, g_L, g_U)
    nlp_bounds = MOI.NLPBoundsPair.(g_L, g_U)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    solver = eval(opts.nlp).Optimizer(;nlp_options(opts)...)
    Z = MOI.add_variables(solver, NN)

    # Add bound constraints
    for i = 1:NN
        zi = MOI.SingleVariable(Z[i])
        MOI.add_constraint(solver, zi, MOI.LessThan(z_U[i]))
        MOI.add_constraint(solver, zi, MOI.GreaterThan(z_L[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), Z[i], Z0.Z[i])
    end

    # Solve the problem
    @info "DIRCOL solve using " * String(opts.nlp)
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)


    # solve
    t0 = time()
    d.solver.stats[:iter_time] = [t0]
    MOI.optimize!(solver)
    d.solver.stats[:time] = time() - t0

    d.solver.stats[:iter_time] .-= d.solver.stats[:iter_time][2]
    deleteat!(d.solver.stats[:iter_time],1)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), Z)
    res = Primals(res, d.part_z)

    d.solver.Z = copy(res)
    d.solver.stats[:status] = MOI.get(solver, MOI.TerminationStatus())

    # Return the results
    return d.solver
end

function max_violation_dircol(d::DIRCOLProblem, Z, g)
    max_viol = 0.
    max_viol = max(max_viol,norm(max.(d.zL - Z,0),Inf))
    max_viol = max(max_viol,norm(max.(Z - d.zU,0),Inf))
    max_viol = max(max_viol,norm(max.(d.gL - g,0),Inf))
    max_viol = max(max_viol,norm(max.(g - d.gU,0),Inf))
    return max_viol
end

function solve!(prob::Problem{T,Continuous}, solver::DIRCOLSolver) where T<:AbstractFloat

    dircol = solve_moi(prob, solver)

    copyto!(prob.X,dircol.Z.X)
    prob.U = copy(dircol.Z.U)

    return dircol
end

function solve(prob::Problem{T,Discrete}, solver::DIRCOLSolver) where T<:AbstractFloat
    prob0 = continuous(prob)
    solver = solve!(prob0, solver)
    return prob0, solver
end

function solve(prob::Problem{T,Discrete}, opts::DIRCOLSolver) where T<:AbstractFloat
    prob0 = copy(prob)
    rollout!(prob0)
    prob_c = continuous(prob0)
    solver = AbstractSolver(prob_c, solver)
    solver = solve!(prob_c, solver)
    return prob_c, solver
end

function nlp_options(opts::DIRCOLSolverOptions)
    if opts.nlp == :Ipopt
        !opts.verbose ? opts.opts[:print_level] = 0 : nothing
        if opts.feasibility_tolerance > 0.
            opts.opts[:constr_viol_tol] = opts.feasibility_tolerance
            opts.opts[:tol] = opts.feasibility_tolerance
        end
    elseif opts.nlp == :SNOPT7
        if !opts.verbose
            opts.opts[:Major_print_level] = 0
            opts.opts[:Minor_print_level] = 0
        end
        if opts.feasibility_tolerance > 0.
            opts.opts[:Major_feasibility_tolerance] = opts.feasibility_tolerance
            opts.opts[:Minor_feasibility_tolerance] = opts.feasibility_tolerance
            opts.opts[:Major_optimality_tolerance] = opts.feasibility_tolerance
        end
    else
        error("Nonlinear solver not implemented")
    end

    return opts.opts
end
