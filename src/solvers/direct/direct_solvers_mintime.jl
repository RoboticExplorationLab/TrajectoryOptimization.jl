@with_kw mutable struct DIRCOLSolverMTOptions{T} <: DirectSolverOptions{T}
    "NLP Solver to use. See MathOptInterface for available NLP solvers"
    nlp::Symbol = :Ipopt

    "Options dictionary for the nlp solver"
    opts::Dict{Symbol,Any} = Dict{Symbol,Any}()

    "Print output to console"
    verbose::Bool = true

    "Minimum Time Options"
    R_min_time::T = 1.0

    h_max::T = 1.0
    h_min::T = 1.0e-3
end

"""
$(TYPEDEF)
Direct Collocation Solver.
Uses a commerical NLP solver to solve the Trajectory Optimization problem.
"""
mutable struct DIRCOLSolverMT{T,Q} <: DirectSolver{T}
    opts::DIRCOLSolverMTOptions{T}
    stats::Dict{Symbol,Any}
    Z::PrimalsMT{T}
    X_::VectorTrajectory{T}
    ∇F::PartedMatTrajectory{T}
    C::PartedVecTrajectory{T}
    ∇C::PartedMatTrajectory{T}
    fVal::VectorTrajectory{T}
    p::Vector{Int}
end

DIRCOLSolverMT(prob::Problem, opts::DIRCOLSolverMTOptions=DIRCOLSolverMTOptions{Float64}(),
    Z::PrimalsMT{T}=PrimalsMT(prob,true)) where {T,Q} = AbstractSolver(prob, opts, Z)

type(::PrimalsMT{T}) where T = T

function AbstractSolver(prob::Problem, opts::DIRCOLSolverMTOptions, Z::PrimalsMT{T}=PrimalsMT(prob, true)) where T
    n,m,N = size(prob)
    X_ = [zeros(T,n) for k = 1:N-1] # midpoints

    part_f = create_partition2(prob.model)
    constraints = prob.constraints
    p = num_stage_constraints(constraints)
    c_stage = [stage(constraints[k]) for k = 1:N-1]
    c_part = [create_partition(c_stage[k]) for k = 1:N-1]
    c_part2 = [create_partition2(c_stage[k],n,m) for k = 1:N-1]

    # Create Trajectories
    ∇F         = [PartedMatrix(zeros(T,n,length(prob.model)),part_f)         for k = 1:N]
    C          = [PartedVector(T,constraints[k],:stage)     for k = 1:N-1]
    ∇C         = [PartedMatrix(T,constraints[k],n,m,:stage) for k = 1:N-1]
    C          = [C...,  PartedVector(T,constraints[N],:terminal)]
    ∇C         = [∇C..., PartedMatrix(T,constraints[N],n,m,:terminal)]

    c_term = terminal(constraints[N])
    p_N = num_constraints(c_term)
    fVal = [zeros(T,n) for k = 1:N]
    p = num_constraints(prob)

    solver = DIRCOLSolverMT{T,HermiteSimpson}(opts, Dict{Symbol,Any}(), Z, X_, ∇F, C, ∇C, fVal, p)
    reset!(solver)
    return solver
end

function reset!(solver::DIRCOLSolverMT{T,Q}) where {T, Q<:QuadratureRule}
    state = Dict{Symbol,Any}(:iterations=>0, :c_max=>T[], :cost=>T[])
    solver.stats[:iterations] = 0
    solver.stats[:c_max] = T[]
    solver.stats[:cost] = T[]
end
