
@with_kw mutable struct StaticPNStats{T}
    iterations::Int = 0
    c_max::Vector{T} = zeros(1)
    cost::Vector{T} = zeros(1)
end


@with_kw mutable struct StaticPNSolverOptions{T} <: DirectSolverOptions{T}
    verbose::Bool = true
    n_steps::Int = 1
    solve_type::Symbol = :feasible
    active_set_tolerance::T = 1e-3
    feasibility_tolerance::T = 1e-6
end


struct ProblemInfo{T,N}
    model::AbstractModel
    obj::Objective
    conSet::ConstraintSets{T}
    x0::SVector{N,T}
    xf::SVector{N,T}
end

function ProblemInfo(prob::StaticProblem)
    n = size(prob)[1]
    ProblemInfo(prob.model, prob.obj, prob.constraints, SVector{n}(prob.x0), SVector{n}(prob.xf))
end

struct DynamicsVals{T}
    fVal::Vector{SVector{N,T}} where N
    ∇F::Vector{SMatrix{N,M,T,L}} where {N,M,L}
end

struct StaticPNSolver{T,N,M,NM} <: DirectSolver{T}
    # Problem Info
    prob::ProblemInfo{T,N}
    Z::Vector{KnotPoint{T,N,M,NM}}
    Z̄::Vector{KnotPoint{T,N,M,NM}}

    opts::StaticPNSolverOptions{T}
    stats::StaticPNStats{T}
    P::StaticPrimals{T,N,M}
    P̄::StaticPrimals{T,N,M}

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
    con_inds::Vector{Vector{SV} where SV}
end

function StaticPNSolver(prob::StaticProblem, opts=StaticPNSolverOptions())
    prob = copy(prob)  # don't modify original problem

    n,m,N = size(prob)
    NN = n*N + m*(N-1)
    stats = StaticPNStats()

    # Add dynamics constraints
    add_dynamics_constraints!(prob)
    conSet = prob.constraints
    NP = sum(num_constraints(conSet))

    # Trajectory
    prob_info = ProblemInfo(prob)
    Z = Traj(prob)
    Z̄ = Traj(prob)

    # Create concatenated primal vars
    P = StaticPrimals(n,m,N)
    P̄ = StaticPrimals(n,m,N)

    # Allocate Cost Hessian & Gradient
    H = spzeros(NN,NN)
    g = zeros(NN)
    E = CostExpansion(n,m,N)

    D = spzeros(NP,NN)
    d = zeros(NP)

    fVal = [@SVector zeros(n) for k = 1:N]
    ∇F = [@SMatrix zeros(n,n+m+1) for k = 1:N]
    dyn_vals = DynamicsVals(fVal, ∇F)
    active_set = zeros(Bool,NP)

    con_inds = gen_con_inds(conSet)

    # Set constant pieces of the Jacobian
    xinds,uinds = P.xinds, P.uinds

    dyn_inds = SVector{n,Int}[]
    StaticPNSolver(prob_info, Z, Z̄, opts, stats, P, P̄, H, g, E, D, d, dyn_vals, active_set, dyn_inds, con_inds)
end

Base.size(solver::StaticPNSolver{T,n,m}) where {T,n,m} = n,m,length(solver.Z)

primals(solver::StaticPNSolver) = solver.P.Z
primal_partition(solver::StaticPNSolver) = solver.P.xinds, solver.P.uinds
get_constraints(solver::StaticPNSolver) = solver.prob.conSet
get_trajectory(solver::StaticPNSolver) = solver.Z
get_objective(solver::StaticPNSolver) = solver.prob.obj

function max_violation(solver::StaticPNSolver)
    conSet = get_constraints(solver)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end
