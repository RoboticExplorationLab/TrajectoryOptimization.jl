export
    DIRCOLSolver,
    DIRCOLSolverOptions

abstract type DirectSolver{T} <: AbstractSolver{T} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

abstract type QuadratureRule end
abstract type HermiteSimpson <: QuadratureRule end

include("primals.jl")

@with_kw mutable struct DIRCOLSolverOptions{T} <: DirectSolverOptions{T}
    "NLP Solver to use. Options are (:Ipopt) (more to be added in the future)"
    nlp::Symbol = :Ipopt

    "Options dictionary for the nlp solver"
    opts::Dict{String,Any} = Dict{String,Any}()

    "Quadrature rule"

end


"""
$(TYPEDEF)
Direct Collocation Solver.
Uses a commerical NLP solver to solve the Trajectory Optimization problem.
"""
struct DIRCOLSolver{T,Q} <: DirectSolver{T}
    opts::DIRCOLSolverOptions{T}
    stats::Dict{Symbol,Any}
    Z::Primals{T}
    X_::VectorTrajectory{T}
    ∇F::PartedMatTrajectory{T}
    C::PartedVecTrajectory{T}
    ∇C::PartedMatTrajectory{T}
    fVal::VectorTrajectory{T}
    p::Vector{Int}
end

DIRCOLSolver(prob::Problem, opts::DIRCOLSolverOptions=DIRCOLSolverOptions{Float64}(),
    Z::Primals{T}=Primals(prob,true)) where {T,Q} = AbstractSolver(prob, opts, Z)

type(::Primals{T}) where T = T

function AbstractSolver(prob::Problem, opts::DIRCOLSolverOptions, Z::Primals{T}=Primals(prob, true)) where T
    n,m,N = size(prob)
    X_ = [zeros(T,n) for k = 1:N-1] # midpoints

    part_f = create_partition2(prob.model)
    constraints = prob.constraints
    p = num_stage_constraints(constraints)
    c_stage = [stage(constraints[k]) for k = 1:N-1]
    c_part = [create_partition(c_stage[k]) for k = 1:N-1]
    c_part2 = [create_partition2(c_stage[k],n,m) for k = 1:N-1]

    # Create Trajectories
    ∇F         = [PartedMatrix(zeros(T,n,n+m),part_f)           for k = 1:N]
    C          = [PartedVector(T,constraints[k],:stage)     for k = 1:N-1]
    ∇C         = [PartedMatrix(T,constraints[k],n,m,:stage) for k = 1:N-1]
    C          = [C...,  PartedVector(T,constraints[N],:terminal)]
    ∇C         = [∇C..., PartedMatrix(T,constraints[N],n,m,:terminal)]

    c_term = terminal(constraints[N])
    p_N = num_constraints(c_term)
    fVal = [zeros(T,n) for k = 1:N]
    p = num_constraints(prob)

    solver = DIRCOLSolver{T,HermiteSimpson}(opts, Dict{Symbol,Any}(), Z, X_, ∇F, C, ∇C, fVal, p)
    reset!(solver)
    return solver
end

function reset!(solver::DIRCOLSolver{T,Q}) where {T, Q<:QuadratureRule}
    state = Dict{Symbol,Any}(:iterations=>0, :c_max=>T[], :cost=>T[])
    solver.stats[:iterations] = 0
    solver.stats[:c_max] = T[]
    solver.stats[:cost] = T[]
end
