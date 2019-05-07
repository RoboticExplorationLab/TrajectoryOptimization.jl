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
    C::PartedVecTrajectory{T}
    fVal::VectorTrajectory{T}

end

DIRCOLSolver(prob::Problem{T}, opts::DIRCOLSolverOptions{T}=DIRCOLSolverOptions{T}()) where {T,Q} = AbstractSolver(prob, opts)

function AbstractSolver(prob::Problem{T}, opts::DIRCOLSolverOptions{T}) where T
    n,m,N = size(prob)
    Z = Primals(prob, true)
    X_ = [zeros(n) for k = 1:N-1] # midpoints

    p = num_stage_constraints(prob.constraints)
    p_N = num_terminal_constraints(prob.constraints)

    c_stage = stage(prob.constraints)
    c_term = terminal(prob.constraints)
    c_part = create_partition(c_stage)
    c_part2 = create_partition2(c_stage,n,m)

    # Create Trajectories
    C = [PartedVector(zeros(T,p),c_part) for k = 1:N-1]
    C = [C...,PartedVector(T,c_term)]
    fVal = [zeros(n) for k = 1:N]
    solver = DIRCOLSolver{T,HermiteSimpson}(opts, Dict{Symbol,Any}(), Z, X_, C, fVal)
    reset!(solver)
    return solver
end

function reset!(solver::DIRCOLSolver{T,Q}) where {T, Q<:QuadratureRule}
    state = Dict{Symbol,Any}(:iterations=>0, :c_max=>T[], :cost=>T[])
    solver.stats[:iterations] = 0
    solver.stats[:c_max] = T[]
    solver.stats[:cost] = T[]
end
