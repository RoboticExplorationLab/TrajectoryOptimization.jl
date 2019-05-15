abstract type AbstractSolver{T} end

jacobian!(prob::Problem{T,Continuous}, solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U)
jacobian!(prob::Problem{T,Discrete},   solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U, prob.dt)


"$(TYPEDEF) Iterative LQR results"
struct iLQRSolver{T} <: AbstractSolver{T}
    opts::iLQRSolverOptions{T}
    stats::Dict{Symbol,Any}

    # Data variables
    X̄::VectorTrajectory{T} # states (n,N)
    Ū::VectorTrajectory{T} # controls (m,N-1)

    K::MatrixTrajectory{T}  # State feedback gains (m,n,N-1)
    d::VectorTrajectory{T}  # Feedforward gains (m,N-1)

    ∇F::PartedMatTrajectory{T} # discrete dynamics jacobian (block) (n,n+m+1,N)

    S::ExpansionTrajectory{T} # Optimal cost-to-go expansion trajectory
    Q::ExpansionTrajectory{T} # cost-to-go expansion trajectory

    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

end

function iLQRSolver(prob::Problem{T},opts=iLQRSolverOptions{T}()) where T
     AbstractSolver(prob, opts)
end

function AbstractSolver(prob::Problem{T}, opts::iLQRSolverOptions{T}) where T
    # Init solver statistics
    stats = Dict{Symbol,Any}()

    # Init solver results
    n = prob.model.n; m = prob.model.m; N = prob.N

    X̄  = [zeros(T,n)   for k = 1:N]
    Ū  = [zeros(T,m)   for k = 1:N-1]

    K  = [zeros(T,m,n) for k = 1:N-1]
    d  = [zeros(T,m)   for k = 1:N-1]

    part_f = create_partition2(prob.model)
    ∇F = [PartedMatrix(zeros(n,n+m+1),part_f) for k = 1:N-1]

    S  = [Expansion(prob,:x) for k = 1:N]
    Q = [k < N ? Expansion(prob) : Expansion(prob,:x) for k = 1:N]

    ρ = zeros(T,1)
    dρ = zeros(T,1)

    solver = iLQRSolver{T}(opts,stats,X̄,Ū,K,d,∇F,S,Q,ρ,dρ)

    reset!(solver)
    return solver
end

function reset!(solver::iLQRSolver{T}) where T
    solver.stats[:iterations]      = 0
    solver.stats[:cost]            = T[]
    solver.stats[:dJ]              = T[]
    solver.stats[:gradient]        = T[]
    solver.stats[:dJ_zero_counter] = 0
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
end

function copy(r::iLQRSolver{T}) where T
    iLQRSolver{T}(copy(r.opts),copy(r.stats),copy(r.X̄),copy(r.Ū),copy(r.K),copy(r.d),copy(r.∇F),copy(r.S),copy(r.Q),copy(r.ρ),copy(r.dρ))
end

get_sizes(solver::iLQRSolver) = length(solver.X̄[1]), length(solver.Ū[2]), length(solver.X̄)


"$(TYPEDEF) Augmented Lagrangian solver"
struct AugmentedLagrangianSolver{T} <: AbstractSolver{T}
    opts::AugmentedLagrangianSolverOptions{T}
    stats::Dict{Symbol,Any}
    stats_uncon::Vector{Dict{Symbol,Any}}

    # Data variables
    C::PartedVecTrajectory{T}      # Constraint values [(p,N-1) (p_N)]
    C_prev::PartedVecTrajectory{T} # Previous constraint values [(p,N-1) (p_N)]
    ∇C::PartedMatTrajectory{T}   # Constraint jacobians [(p,n+m,N-1) (p_N,n)]
    λ::PartedVecTrajectory{T}      # Lagrange multipliers [(p,N-1) (p_N)]
    μ::PartedVecTrajectory{T}     # Penalty matrix [(p,p,N-1) (p_N,p_N)]
    active_set::PartedVecTrajectory{Bool} # active set [(p,N-1) (p_N)]
end

AugmentedLagrangianSolver(prob::Problem{T},
    opts::AugmentedLagrangianSolverOptions{T}=AugmentedLagrangianSolverOptions{T}()) where T =
    AbstractSolver(prob,opts)

"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AbstractSolver(prob::Problem{T}, opts::AugmentedLagrangianSolverOptions{T}) where T
    # check for conflicting convergence criteria between unconstrained solver and AL: warn

    # Init solver statistics
    stats = Dict{Symbol,Any}(:iterations=>0,:iterations_total=>0,
        :iterations_inner=>Int[],:cost=>T[],:c_max=>T[])
    stats_uncon = Dict{Symbol,Any}[]

    # Init solver results
    n = prob.model.n; m = prob.model.m; N = prob.N

    C,∇C,λ,μ,active_set = init_constraint_trajectories(prob.constraints,n,m,N)

    AugmentedLagrangianSolver{T}(opts,stats,stats_uncon,C,copy(C),∇C,λ,μ,active_set)
end

# function init_constraint_trajectories(constraints::ConstraintSet,n::Int,m::Int,N::Int;
#         μ_init::T=1.,λ_init::T=0.) where T
#     p = num_stage_constraints(constraints)
#     p_N = num_terminal_constraints(constraints)
#
#     # Initialize the partitions
#     c_stage = stage(constraints)
#     c_term = terminal(constraints)
#     c_part = create_partition(c_stage)
#     c_part2 = create_partition2(c_stage,n,m)
#
#     # Create Trajectories
#     C          = [PartedVector(zeros(T,p),c_part)       for k = 1:N-1]
#     ∇C         = [PartedMatrix(zeros(T,p,n+m),c_part2)  for k = 1:N-1]
#     λ          = [PartedVector(ones(T,p),c_part) for k = 1:N-1]
#     μ          = [PartedVector(ones(T,p),c_part) for k = 1:N-1]
#     active_set = [PartedVector(ones(Bool,p),c_part)     for k = 1:N-1]
#
#     C = [C..., PartedVector(T,c_term)]
#     ∇C = [∇C..., PartedMatrix(T,c_term,n,0)]
#     λ = [λ..., PartedVector(T,c_term)]
#     μ = [μ..., PartedArray(ones(T,num_constraints(c_term)), create_partition(c_term))]
#     active_set = [active_set..., PartedVector(Bool,c_term)]
#
#     # Initialize dual and penality values
#     for k = 1:N
#         λ[k] .*= λ_init
#         μ[k] .*= μ_init
#     end
#
#     return C,∇C,λ,μ,active_set
# end

function init_constraint_trajectories(constraints::ProblemConstraints,n::Int,m::Int,N::Int;
        μ_init::T=1.,λ_init::T=0.) where T

    p = num_constraints(constraints)
    c_stage = [stage(constraints[k]) for k = 1:N-1]
    c_part = [create_partition(c_stage[k]) for k = 1:N-1]
    c_part2 = [create_partition2(c_stage[k],n,m) for k = 1:N-1]

    # Create Trajectories
    C          = [PartedVector(T,constraints[k],:stage)     for k = 1:N-1]
    ∇C         = [PartedMatrix(T,constraints[k],n,m,:stage) for k = 1:N-1]
    C          = [C...,  PartedVector(T,constraints[N],:terminal)]
    ∇C         = [∇C..., PartedMatrix(T,constraints[N],n,m,:terminal)]


    λ          = [PartedVector(ones(T,p[k]), C[k].parts)  for k = 1:N]
    μ          = [PartedVector(ones(T,p[k]), C[k].parts)  for k = 1:N]
    active_set = [PartedVector(ones(Bool,p[k]), C[k].parts)  for k = 1:N]

    # Initialize dual and penality values
    for k = 1:N
        λ[k] .*= λ_init
        μ[k] .*= μ_init
    end

    return C,∇C,λ,μ,active_set
end

function reset!(solver::AugmentedLagrangianSolver{T}) where T
    solver.stats[:iterations]       = 0
    solver.stats[:iterations_total] = 0
    solver.stats[:iterations_inner] = T[]
    solver.stats[:cost]             = T[]
    solver.stats[:c_max]            = T[]
    n,m,N = get_sizes(solver)
    for k = 1:N
        solver.λ[k] .*= 0
        solver.μ[k] .= solver.μ[k]*0 .+ solver.opts.penalty_initial
    end
end

function copy(r::AugmentedLagrangianSolver{T}) where T
    AugmentedLagrangianSolver{T}(deepcopy(r.C),deepcopy(r.C_prev),deepcopy(r.∇C),deepcopy(r.λ),deepcopy(r.μ),deepcopy(r.active_set))
end

get_sizes(solver::AugmentedLagrangianSolver{T}) where T = size(solver.∇C[1].x,2), size(solver.∇C[1].u,2), length(solver.λ)

#TODO
"$(TYPEDEF) ALTRO solver"
struct ALTROSolver{T} <: AbstractSolver{T}
    opts::ALTROSolverOptions{T}
end

function AbstractSolver(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    ALTROSolver{T}(opts)
end
