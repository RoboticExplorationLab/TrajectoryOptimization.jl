using MathOptInterface
const MOI = MathOptInterface

struct DIRCOLProblemMT{T} <: MOI.AbstractNLPEvaluator
    prob::Problem{T,Continuous}
    cost::Function
    cost_gradient!::Function
    solver::DIRCOLSolverMT{T,HermiteSimpson}
    jac_struct
    part_z::NamedTuple{(:X,:U,:H), NTuple{3,Matrix{Int}}}
    p::NTuple{4,Int}    # (total constraints, p_colloc, p_custom, p_h)
    nG::NTuple{4,Int}   # (total constraint jacobian, nG_colloc, nG_custom, nG_h)
end

function DIRCOLProblemMT(prob::Problem{T,Continuous}, solver::DIRCOLSolverMT{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    p_custom = sum(p)
    p_h = N-2
    P = p_colloc + p_custom + p_h

    NN = N*(n+m) + (N-1)

    nG_colloc = p_colloc*2*(n + m) + p_colloc
    nG_custom = sum(p[1:N-1])*(n+m) + p[N]*n
    nG_h = (N-2)*2
    nG = nG_colloc + nG_custom + nG_h

    part_z = create_partition(n,m,1,N,N,N-1)

    jac_struct_colloc = collocation_constraint_jacobian_sparsityMT!(prob)
    jac_struct_custom = custom_constraint_jacobian_sparsityMT!(prob,p_colloc)
    jac_struct_h = h_eq_constraint_sparsityMT!(prob,p_colloc+p_custom)

    jac_struct = copy(jac_struct_colloc)
    append!(jac_struct,jac_struct_custom)
    append!(jac_struct,jac_struct_h)

    num_con = (P,p_colloc,p_custom,p_h)
    num_jac = (nG, nG_colloc, nG_custom, nG_h)
    DIRCOLProblemMT(prob, gen_stage_cost_min_time(prob,solver.opts.R_min_time), gen_stage_cost_gradient_min_time(prob,solver.opts.R_min_time), solver, jac_struct, part_z, num_con, num_jac)
end

MOI.features_available(d::DIRCOLProblemMT) = [:Grad, :Jac]
MOI.initialize(d::DIRCOLProblemMT, features) = nothing

MOI.jacobian_structure(d::DIRCOLProblemMT) = d.jac_struct
MOI.hessian_lagrangian_structure(d::DIRCOLProblemMT) = []

function MOI.eval_objective(d::DIRCOLProblemMT, Z)
    X,U,H = unpackMT(Z, d.part_z)
    d.cost(X, U, H)
end

function MOI.eval_objective_gradient(d::DIRCOLProblemMT, grad_f, Z)
    X,U,H = unpackMT(Z, d.part_z)
    d.cost_gradient!(grad_f,X,U,H)
end

function MOI.eval_constraint(d::DIRCOLProblemMT, g, Z)
    X,U,H = unpackMT(Z, d.part_z)
    P,p_colloc,p_custom,p_h = d.p
    g_colloc = view(g, 1:p_colloc)
    g_custom = view(g, p_colloc .+ (1:p_custom))
    g_h = view(g, (p_colloc+p_custom) .+ (1:p_h))

    collocation_constraints!(g_colloc, d.prob, d.solver, X, U, H)
    update_constraints!(g_custom, d.prob, d.solver, X, U)
    h_eq_constraints!(g_h,d.prob,d.solver,H)
end

function MOI.eval_constraint_jacobian(d::DIRCOLProblemMT, jac, Z)
    X,U,H = unpackMT(Z, d.part_z)
    n,m = size(d.prob)
    P, p_colloc, p_custom, p_h = d.p
    nG, nG_colloc, nG_custom, nG_h = d.nG
    jac_colloc = view(jac, 1:nG_colloc)
    jac_custom = view(jac, nG_colloc .+ (1:nG_custom))
    jac_h = view(jac, (nG_colloc+nG_custom) .+ (1:nG_h))

    collocation_constraint_jacobian!(jac_colloc, d.prob, d.solver, X, U, H)
    constraint_jacobian!(jac_custom, d.prob, d.solver, X, U)
    h_eq_constraint_jacobian!(jac_h,d.prob,d.solver,H)
end

MOI.eval_hessian_lagrangian(::DIRCOLProblemMT, H, x, σ, μ) = nothing

function solve_moi(prob::Problem, opts::DIRCOLSolverMTOptions)
    prob = copy(prob)
    bnds = remove_bounds!(prob)
    z_U, z_L, g_U, g_L = get_boundsMT(prob, bnds, opts.h_max, opts.h_min)
    n,m,N = size(prob)
    NN = (n+m)*N + (N-1)

    # Get initial condition
    Z0 = PrimalsMT(prob, true)

    # Create NLP Block
    has_objective = true
    dircol = DIRCOLSolverMT(prob, opts)
    d = DIRCOLProblemMT(prob, dircol)
    nlp_bounds = MOI.NLPBoundsPair.(g_L, g_U)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    # Create solver
    solver = typeof(opts.nlp)(;opts.opts...)
    Z = MOI.add_variables(solver, NN)

    # Add bound constraints
    for i = 1:NN
        zi = MOI.SingleVariable(Z[i])
        MOI.add_constraint(solver, zi, MOI.LessThan(z_U[i]))
        MOI.add_constraint(solver, zi, MOI.GreaterThan(z_L[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), Z[i], Z0.Z[i])
    end

    # Solve the problem
    @info "DIRCOL solve using " * String(nameof(parentmodule(typeof(solver))))
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), Z)
    res = PrimalsMT(res, d.part_z)

    d.solver.Z = res
    d.solver.stats[:status] = MOI.get(solver, MOI.TerminationStatus())

    # Return the results
    return d.solver
end

function solve!(prob::Problem,opts::DIRCOLSolverMTOptions)
    dircol = solve_moi(prob, opts)

    copyto!(prob.X,dircol.Z.X)
    prob.U = [k != prob.N ? [dircol.Z.U[k];sqrt(dircol.Z.H[k])] : dircol.Z.U[k] for k = 1:prob.N]

    return dircol
end
