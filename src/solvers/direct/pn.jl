export
    ProjectedNewtonSolverOptions,
    ProjectedNewtonSolver

@with_kw mutable struct ProjectedNewtonStats{T}
    iterations::Int = 0
    c_max::Vector{T} = zeros(5)
    cost::Vector{T} = zeros(5)
end


"""$(TYPEDEF)
Solver options for the Projected Newton solver.
$(FIELDS)
"""
@with_kw mutable struct ProjectedNewtonSolverOptions{T} <: DirectSolverOptions{T}
    verbose::Bool = true
    n_steps::Int = 2
    solve_type::Symbol = :feasible
    active_set_tolerance::T = 1e-3
    constraint_tolerance::T = 1e-6
    ρ::T = 1e-2
    r_threshold::T = 1.1
end

function ProjectedNewtonSolverOptions(opts::SolverOptions)
    ProjectedNewtonSolverOptions(
        constraint_tolerance=opts.constraint_tolerance,
        active_set_tolerance=opts.active_set_tolerance,
        verbose=opts.verbose,
    )
end


struct ProblemInfo{T,N}
    model::AbstractModel
    obj::Objective
    conSet::ConstraintSet{T}
    x0::SVector{N,T}
    xf::SVector{N,T}
end

function ProblemInfo(prob::Problem)
    n = size(prob)[1]
    ProblemInfo(prob.model, prob.obj, prob.constraints, SVector{n}(prob.x0), SVector{n}(prob.xf))
end


"""
$(TYPEDEF)
Projected Newton Solver
Direct method developed by the REx Lab at Stanford University
Achieves machine-level constraint satisfaction by projecting onto the feasible subspace.
    It can also take a full Newton step by solving the KKT system.
This solver is to be used exlusively for solutions that are close to the optimal solution.
    It is intended to be used as a "solution polishing" method for augmented Lagrangian methods.
"""
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
    E::Vector{CostExpansion{T,N,N,M}}

    D::SparseMatrixCSC{T,Int}
    d::Vector{T}

    dyn_vals::DynamicsVals{T}
    active_set::Vector{Bool}

    dyn_inds::Vector{SVector{N,Int}}
    con_inds::Vector{<:Vector}
end

function ProjectedNewtonSolver(prob::Problem, opts=SolverOptions())
    Z = prob.Z  # grab trajectory before copy to keep associativity
    prob = copy(prob)  # don't modify original problem

    n,m,N = size(prob)
    NN = n*N + m*(N-1)
    stats = ProjectedNewtonStats()

    # Add dynamics constraints
    TrajOptCore.add_dynamics_constraints!(prob)
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
    E = [CostExpansion{Float64}(n,m) for k = 1:N]

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

    opts_pn = ProjectedNewtonSolverOptions(opts)
    dyn_inds = SVector{n,Int}[]
    ProjectedNewtonSolver(prob_info, Z, Z̄, opts_pn, stats,
        P, P̄, H, g, E, D, d, dyn_vals, active_set, dyn_inds, con_inds)
end


primals(solver::ProjectedNewtonSolver) = solver.P.Z
primal_partition(solver::ProjectedNewtonSolver) = solver.P.xinds, solver.P.uinds

# AbstractSolver interface
Base.size(solver::ProjectedNewtonSolver{T,n,m}) where {T,n,m} = n,m,length(solver.Z)
TrajOptCore.get_model(solver::ProjectedNewtonSolver) = solver.prob.model
TrajOptCore.get_constraints(solver::ProjectedNewtonSolver) = solver.prob.conSet
TrajOptCore.get_trajectory(solver::ProjectedNewtonSolver) = solver.Z
TrajOptCore.get_objective(solver::ProjectedNewtonSolver) = solver.prob.obj
get_active_set(solver::ProjectedNewtonSolver) = solver.active_set
