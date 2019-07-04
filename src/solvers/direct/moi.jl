using MathOptInterface
const MOI = MathOptInterface

struct DIRCOLProblem{T} <: MOI.AbstractNLPEvaluator
    prob::Problem{T,Continuous}
    xm::Function
    cost_gradient!::Function
    solver::DIRCOLSolver{T,HermiteSimpson}
    jac_struct::Vector{NTuple{2,Int}}
    part_z::NamedTuple{(:X,:U), NTuple{2,Matrix{Int}}}
    p::NTuple{2,Int}    # (total constraints, p_colloc)
    nG::NTuple{2,Int}   # (total constraint jacobian, nG_colloc)
end

function DIRCOLProblem(prob::Problem{T,Continuous}, solver::DIRCOLSolver{T,HermiteSimpson}) where T
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
    DIRCOLProblem(prob, gen_xm(prob,solver), gen_stage_cost_gradient(prob), solver, jac_struct, part_z, num_con, num_jac)
end

function cost(solver::DIRCOLSolver{T,HermiteSimpson},obj,xm::Function,X,U,H) where T
    N = length(X)
    J = 0.0
    for k = 1:N-1
        Xm = xm(X[k+1],X[k],U[k+1],U[k],H[k])
        Um = 0.5*(U[k] + U[k+1])
        J += H[k]/6*(stage_cost(obj[k],X[k],U[k]) + 4*stage_cost(obj[k],Xm,Um) + stage_cost(obj[k],X[k+1],U[k+1]))
    end
    J += stage_cost(obj[N],X[N])
    return J
end

function cost(solver::DIRCOLSolver{T,Midpoint},obj,xm,X,U,H) where T
    N = length(X)
    J = 0.0
    for k = 1:N-1
        xm = xm(X[k+1],X[k],U[k+1],U[k],H[k])
        um = 0.5*(U[k] + U[k+1])
        J += stage_cost(obj[k],xm,um,H[k])
    end
    J += stage_cost(obj[N],X[N])
    return J
end

MOI.features_available(d::DIRCOLProblem) = [:Grad, :Jac]
MOI.initialize(d::DIRCOLProblem, features) = nothing

MOI.jacobian_structure(d::DIRCOLProblem) = d.jac_struct
MOI.hessian_lagrangian_structure(d::DIRCOLProblem) = []

function MOI.eval_objective(d::DIRCOLProblem, Z)
    X,U = unpack(Z, d.part_z)
    # cost(d.prob.obj, X, U, get_dt_traj(d.prob))
    cost(d.solver,d.prob.obj,d.xm, X, U, get_dt_traj(d.prob))
end

function MOI.eval_objective_gradient(d::DIRCOLProblem, grad_f, Z)
    X,U = unpack(Z, d.part_z)
    # cost_gradient!(grad_f, d.prob, X, U, get_dt_traj(d.prob))
    d.cost_gradient!(grad_f,d.prob,X,U,get_dt_traj(d.prob))
end

function MOI.eval_constraint(d::DIRCOLProblem, g, Z)
    X,U = unpack(Z, d.part_z)
    P,p_colloc = d.p
    g_colloc = view(g, 1:p_colloc)
    g_custom = view(g, p_colloc+1:P)

    collocation_constraints!(g_colloc, d.prob, d.solver, X, U)
    update_constraints!(g_custom, d.prob, d.solver, X, U)
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

function solve_moi(prob::Problem, opts::DIRCOLSolverOptions)
    prob = copy(prob)
    bnds = remove_bounds!(prob)
    z_U, z_L, g_U, g_L = get_bounds(prob, bnds)
    n,m,N = size(prob)
    NN = (n+m)*N

    # Get initial condition
    Z0 = Primals(prob, true)

    # Create NLP Block
    has_objective = true
    dircol = DIRCOLSolver(prob, opts)
    d = DIRCOLProblem(prob, dircol)
    nlp_bounds = MOI.NLPBoundsPair.(g_L, g_U)
    block_data = MOI.NLPBlockData(nlp_bounds, d, has_objective)

    # Create solver
    solver = eval(opts.nlp).Optimizer(;opts.opts...)
    Z = MOI.add_variables(solver, NN)

    # Add bound constraints
    for i = 1:NN
        zi = MOI.SingleVariable(Z[i])
        MOI.add_constraint(solver, zi, MOI.LessThan(z_U[i]))
        MOI.add_constraint(solver, zi, MOI.GreaterThan(z_L[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), Z[i], Z0.Z[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), Z)
    res = Primals(res, d.part_z)

    # Return the results
    return res, dircol
end

function solve!(prob::Problem,opts::DIRCOLSolverOptions)
    res, dircol = solve_moi(prob, opts)
    copyto!(prob.X,res.X)
    copyto!(prob.U,res.U[1:prob.N-1])

    return dircol
end
