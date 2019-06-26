export
    DIRCOLSolver,
    DIRCOLSolverOptions

abstract type DirectSolver{T} <: AbstractSolver{T} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

abstract type QuadratureRule end
abstract type HermiteSimpson <: QuadratureRule end

include("primals.jl")


@with_kw mutable struct ProjectedNewtonSolverOptions{T} <: DirectSolverOptions{T}
    "Print output to console"
    verbose::Bool = true

    "Tolerance for checking active inequality constraints. Positive values move the boundary further into the feasible region (i.e. negative)"
    active_set_tolerance = 1e-3

    "Tolerance for constraint feasibility during projection"
    feasibility_tolerance = 1e-6
end


"""
$(TYPEDEF)
Projected Newton Solver
Direct method developed by the Rex Lab at Stanford University
"""
struct ProjectedNewtonSolver{T} <: DirectSolver{T}
    opts::ProjectedNewtonSolverOptions{T}
    stats::Dict{Symbol,Any}
    V::PrimalDual{T}
    H::SparseMatrixCSC{T,Int}      # Cost Hessian
    g::Vector{T}                   # Cost gradient
    Y::SparseMatrixCSC{T,Int}      # Constraint Jacobian
    y::Vector{T}                   # Constraint Violations

    fVal::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    # ∇F::Vector{PartedArray{T,2,SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}, P}} where P
    ∇F::PartedMatTrajectory{T}
    C::Vector{PartedArray{T,1,SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}, P} where P}
    ∇C::Vector{PartedArray{T,2,SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}, P} where P}
    a::PartedVector{Bool,Vector{Bool},NamedTuple{(:primals,:duals,:ν,:λ),NTuple{4,UnitRange{Int}}}}
    active_set::Vector{PartedArray{Bool,1,SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}, P} where P}
    parts::NamedTuple{(:primals,:duals,:ν,:λ),NTuple{4,UnitRange{Int}}}
end

function AbstractSolver(prob::Problem{T,D}, opts::ProjectedNewtonSolverOptions{T}) where {T,D}
    n,m,N = size(prob)
    X_ = [zeros(T,n) for k = 1:N-1] # midpoints

    NN = N*n + (N-1)*m
    p = num_constraints(prob)
    pcum = insert!(cumsum(p),1,0)
    P = sum(p) + N*n

    V = PrimalDual(prob)

    part_f = create_partition2(prob.model)
    constraints = prob.constraints
    part_a = (primals=1:NN, duals=NN+1:NN+P, ν=NN .+ (1:N*n), λ=NN + N*n .+ (1:sum(p)))


    # Build Blocks
    H = spzeros(NN,NN)
    g = zeros(NN)
    Y = spzeros(P,NN)
    y = zeros(P)
    a = PartedVector(ones(Bool, NN+P), part_a)

    # Build views
    fVal = [view(y,(k-1)*n .+ (1:n)) for k = 1:N]
    C = [PartedArray(view(y, N*n + pcum[k] .+ (1:p[k])), create_partition(constraints[k], k==N ? :terminal : :stage))  for k = 1:N]
    active_set = [PartedArray(view(a.A, NN + N*n + pcum[k] .+ (1:p[k])), create_partition(constraints[k], k==N ? :terminal : :stage))  for k = 1:N]

    # ∇F = [PartedArray(view(Y, (k-1)*n .+ (1:n), (k-1)*(n+m+1) .+ (1:n+m*(k<N)+1)), part_f) for k = 1:N]
    ∇F = [PartedMatrix(zeros(n,n+m+1), part_f) for k = 1:N]
    ∇C = [begin
            if k == N
                d2 = n
                stage = :terminal
            else
                d2 = n+m
                stage = :stage
            end
            part = create_partition2(constraints[k], n, m, stage)
            PartedArray(view(Y, N*n + pcum[k] .+ (1:p[k]), (k-1)*(n+m) .+ (1:d2)), part)
        end for k = 1:N]

    # return C
    solver = ProjectedNewtonSolver{T}(opts, Dict{Symbol,Any}(), V, H, g, Y, y, fVal, ∇F, C, ∇C, a, active_set, part_a)
    reset!(solver)
    return solver


    # Create Trajectories
    Q          = [k < N ? Expansion(prob) : Expansion(prob,:x) for k = 1:N]
    ∇F         = [PartedMatrix(zeros(n,n+m+1),part_f)       for k = 1:N]
    C          = [PartedVector(T,constraints[k],:stage)     for k = 1:N-1]
    ∇C         = [PartedMatrix(T,constraints[k],n,m,:stage) for k = 1:N-1]
    a          = [PartedVector(Bool,constraints[k],:stage)     for k = 1:N-1]
    C          = [C...,  PartedVector(T,constraints[N],:terminal)]
    ∇C         = [∇C..., PartedMatrix(T,constraints[N],n,m,:terminal)]
    a          = [a...,  PartedVector(Bool,constraints[N],:terminal)]

    c_term = terminal(constraints[N])
    p_N = num_constraints(c_term)
    fVal = [zeros(T,n) for k = 1:N]
    p = num_constraints(prob)

    solver = ProjectedNewtonSolver{T}(opts, Dict{Symbol,Any}(), V, Q, fVal, ∇F, C, ∇C, a, p)
    reset!(solver)
    return solver
end

ProjectedNewtonSolver(prob::Problem,
    opts::ProjectedNewtonSolverOptions=ProjectedNewtonSolverOptions{Float64}()) = AbstractSolver(prob, opts)


function reset!(solver::ProjectedNewtonSolver{T}) where T
    solver.stats[:iterations] = 0
    solver.stats[:c_max] = T[]
    solver.stats[:cost] = T[]
end

function num_active_constraints(solver::ProjectedNewtonSolver)
    sum(solver.a.duals)
end


@with_kw mutable struct DIRCOLSolverOptions{T} <: DirectSolverOptions{T}
    "NLP Solver to use. Options are (:Ipopt) (more to be added in the future)"
    nlp::Symbol = :Ipopt

    "Options dictionary for the nlp solver"
    opts::Dict{String,Any} = Dict{String,Any}()

    "Print output to console"
    verbose::Bool = true
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
