export
    ProjectedNewtonSolverOptions,
    ProjectedNewtonSolver

@with_kw mutable struct ProjectedNewtonStats{T}
    iterations::Int = 0
    c_max::Vector{T} = zeros(5)
    cost::Vector{T} = zeros(5)
end


@with_kw mutable struct ProjectedNewtonSolverOptions{T} <: DirectSolverOptions{T}
    verbose::Bool = true
    n_steps::Int = 1
    solve_type::Symbol = :feasible
    active_set_tolerance::T = 1e-3
    feasibility_tolerance::T = 1e-6
    ρ::T = 1e-2
    r_threshold::T = 1.1
end


struct ProblemInfo{T,N}
    model::AbstractModel
    obj::Objective
    conSet::ConstraintSets{T}
    x0::SVector{N,T}
    xf::SVector{N,T}
end

function ProblemInfo(prob::Problem)
    n = size(prob)[1]
    ProblemInfo(prob.model, prob.obj, prob.constraints, SVector{n}(prob.x0), SVector{n}(prob.xf))
end


struct ProjectedNewtonSolver{T,N,M,NM} <: DirectSolver{T}
    # Problem Info
    prob::ProblemInfo{T,N}
    Z::Vector{KnotPoint{T,N,M,NM}}
    Z̄::Vector{KnotPoint{T,N,M,NM}}

    opts::ProjectedNewtonSolverOptions{T}
    stats::ProjectedNewtonStats{T}
    P::Primals{T,N,M}
    P̄::Primals{T,N,M}

    H::SparseMatrixCSC{T,Int}
    g::Vector{T}
    E::CostExpansion

    D::SparseMatrixCSC{T,Int}
    d::Vector{T}

    # fVal::Vector{SVector{N,T}}
    # ∇F::Vector{SMatrix{N,NM,T,NNM}}
    dyn_vals::DynamicsVals{T}
    active_set::Vector{Bool}

    dyn_inds::Vector{SVector{N,Int}}
    con_inds::Vector{<:Vector}
end

function ProjectedNewtonSolver(prob::Problem, opts=ProjectedNewtonSolverOptions())
    Z = prob.Z  # grab trajectory before copy to keep associativity
    prob = copy(prob)  # don't modify original problem

    n,m,N = size(prob)
    NN = n*N + m*(N-1)
    stats = ProjectedNewtonStats()

    # Add dynamics constraints
    add_dynamics_constraints!(prob)
    conSet = prob.constraints
    NP = sum(num_constraints(conSet))

    # Trajectory
    prob_info = ProblemInfo(prob)
    Z̄ = copy(prob.Z)

    # Create concatenated primal vars
    P = Primals(n,m,N)
    P̄ = Primals(n,m,N)

    # Allocate Cost Hessian & Gradient
    H = spzeros(NN,NN)
    g = zeros(NN)
    E = CostExpansion(n,m,N)

    D = spzeros(NP,NN)
    d = zeros(NP)

    fVal = [@SVector zeros(n) for k = 1:N]
    xMid = [@SVector zeros(n) for k = 1:N-1]
    ∇F = [@SMatrix zeros(n,n+m+1) for k = 1:N]
    dyn_vals = DynamicsVals(fVal, xMid, ∇F)
    active_set = zeros(Bool,NP)

    con_inds = gen_con_inds(conSet)

    # Set constant pieces of the Jacobian
    xinds,uinds = P.xinds, P.uinds

    dyn_inds = SVector{n,Int}[]
    ProjectedNewtonSolver(prob_info, Z, Z̄, opts, stats, P, P̄, H, g, E, D, d, dyn_vals, active_set, dyn_inds, con_inds)
end

Base.size(solver::ProjectedNewtonSolver{T,n,m}) where {T,n,m} = n,m,length(solver.Z)

primals(solver::ProjectedNewtonSolver) = solver.P.Z
primal_partition(solver::ProjectedNewtonSolver) = solver.P.xinds, solver.P.uinds
get_model(solver::ProjectedNewtonSolver) = solver.prob.model
get_constraints(solver::ProjectedNewtonSolver) = solver.prob.conSet
get_trajectory(solver::ProjectedNewtonSolver) = solver.Z
get_objective(solver::ProjectedNewtonSolver) = solver.prob.obj
get_active_set(solver::ProjectedNewtonSolver) = solver.active_set

function max_violation(solver::ProjectedNewtonSolver)
    conSet = get_constraints(solver)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end
