"$(TYPEDEF) Expansion"
struct Expansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
    ux::Matrix{T}
end

function Expansion(prob::Problem{T}) where T
    n = prob.model.n; m = prob.model.m
    Expansion(zeros(T,n),zeros(T,m),zeros(T,n,n),zeros(T,m,m),zeros(T,m,n))
end

function copy(e::Expansion{T}) where T
    Expansion{T}(copy(e.x),copy(e.u),copy(e.xx),copy(e.uu),copy(e.ux))
end

function reset!(e::Expansion)
    e.x .= zero(e.x)
    e.u .= zero(e.u)
    e.xx .= zero(e.xx)
    e.uu .= zero(e.uu)
    e.ux .= zero(e.ux)
    return nothing
end

ExpansionTrajectory{T} = Vector{Expansion{T}} where T <: AbstractFloat

function reset!(et::ExpansionTrajectory)
    for e in et
        reset!(e)
    end
end


struct BackwardPassNew{T<:AbstractFloat}
    Qx::VectorTrajectory{T}
    Qu::VectorTrajectory{T}
    Qxx::MatrixTrajectory{T}
    Qux::MatrixTrajectory{T}
    Quu::MatrixTrajectory{T}
    Qux_reg::MatrixTrajectory{T}
    Quu_reg::MatrixTrajectory{T}
end

function BackwardPassNew(p::Problem{T}) where T
    n = p.model.n; m = p.model.m; N = p.N

    Qx = [zeros(T,n) for i = 1:N-1]
    Qu = [zeros(T,m) for i = 1:N-1]
    Qxx = [zeros(T,n,n) for i = 1:N-1]
    Qux = [zeros(T,m,n) for i = 1:N-1]
    Quu = [zeros(T,m,m) for i = 1:N-1]

    Qux_reg = [zeros(T,m,n) for i = 1:N-1]
    Quu_reg = [zeros(T,m,m) for i = 1:N-1]

    BackwardPassNew{T}(Qx,Qu,Qxx,Qux,Quu,Qux_reg,Quu_reg)
end

function copy(bp::BackwardPassNew{T}) where T
    BackwardPassNew{T}(deepcopy(bp.Qx),deepcopy(bp.Qu),deepcopy(bp.Qxx),deepcopy(bp.Qux),deepcopy(bp.Quu),deepcopy(bp.Qux_reg),deepcopy(bp.Quu_reg))
end

function reset!(bp::BackwardPassNew)
    N = length(bp.Qx)
    for k = 1:N-1
        bp.Qx[k] = zero(bp.Qx[k]); bp.Qu[k] = zero(bp.Qu[k]); bp.Qxx[k] = zero(bp.Qxx[k]); bp.Quu[k] = zero(bp.Quu[k]); bp.Qux[k] = zero(bp.Qux[k])
        bp.Quu_reg[k] = zero(bp.Quu_reg[k]); bp.Qux_reg[k] = zero(bp.Qux_reg[k])
    end
end

abstract type AbstractSolver{T} end

"$(TYPEDEF) Iterative LQR results"
struct iLQRSolver{T} <: AbstractSolver{T}
    opts::iLQRSolverOptions{T}
    stats::Dict{Symbol,Any}

    # Data variables
    X̄::VectorTrajectory{T} # states (n,N)
    Ū::VectorTrajectory{T} # controls (m,N-1)

    K::MatrixTrajectory{T}  # State feedback gains (m,n,N-1)
    d::VectorTrajectory{T}  # Feedforward gains (m,N-1)

    S::MatrixTrajectory{T}  # Cost-to-go Hessian (n,n,N)
    s::VectorTrajectory{T}  # Cost-to-go gradient (n,N)

    ∇F::PartedMatTrajectory{T} # discrete dynamics jacobian (block) (n,n+m+1,N)

    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

    Q::ExpansionTrajectory{T} # cost-to-go expansion trajectory
end

function iLQRSolver(prob::Problem{T},opts=iLQRSolverOptions{T}()) where T
     AbstractSolver(prob, opts)
end

function AbstractSolver(prob::Problem{T}, opts::iLQRSolverOptions{T}) where T
    # Init solver statistics
    stats = Dict{Symbol,Any}()

    # Init solver results
    n = prob.model.n; m = prob.model.m; N = prob.N

    X̄  = [zeros(T,n)   for i = 1:N]
    Ū  = [zeros(T,m)   for i = 1:N-1]

    K  = [zeros(T,m,n) for i = 1:N-1]
    d  = [zeros(T,m)   for i = 1:N-1]

    S  = [zeros(T,n,n) for i = 1:N]
    s  = [zeros(T,n)   for i = 1:N]

    part_f = create_partition2(prob.model)
    ∇F = [BlockArray(zeros(n,n+m+1),part_f) for i = 1:N-1]

    ρ = zeros(T,1)
    dρ = zeros(T,1)

    Q = [Expansion(prob) for i = 1:N-1]

    solver = iLQRSolver{T}(opts,stats,X̄,Ū,K,d,S,s,∇F,ρ,dρ,Q)
    reset!(solver)
    return solver
end

function reset!(solver::iLQRSolver{T}) where T
    solver.stats[:iterations]      = 0
    solver.stats[:cost]            = T[]
    solver.stats[:dJ]              = T[]
    solver.stats[:gradient]        = T[]
    solver.stats[:dJ_zero_counter] = 0
    solver.ρ[1] = 0
    solver.dρ[1] = 0
end

function copy(r::iLQRSolver{T}) where T
    iLQRSolver{T}(copy(r.opts),copy(r.stats),copy(r.X̄),copy(r.Ū),copy(r.K),copy(r.d),copy(r.S),copy(r.s),copy(r.∇F),copy(r.ρ),copy(r.dρ),copy(r.Q))
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
    # Init solver statistics
    stats = Dict{Symbol,Any}(:iterations=>0,:iterations_total=>0,
        :iterations_inner=>Int[],:cost=>T[],:c_max=>T[])
    stats_uncon = Dict{Symbol,Any}[]

    # Init solver results
    n = prob.model.n; m = prob.model.m; N = prob.N
    p = num_stage_constraints(prob)

    C,∇C,λ,μ,active_set = init_constraint_trajectories(prob.constraints,n,m,N)

    AugmentedLagrangianSolver{T}(opts,stats,stats_uncon,C,copy(C),∇C,λ,μ,active_set)
end

function init_constraint_trajectories(constraints::AbstractConstraintSet,n::Int,m::Int,N::Int;
        μ_init::T=1.,λ_init::T=0.) where T
    p = num_stage_constraints(constraints)
    p_N = num_terminal_constraints(constraints)

    # Initialize the partitions
    c_stage = stage(constraints)
    c_term = terminal(constraints)
    c_part = create_partition(c_stage)
    c_part2 = create_partition2(c_stage,n,m)

    # Create Trajectories
    C          = [BlockArray(zeros(T,p),c_part)       for k = 1:N-1]
    ∇C         = [BlockArray(zeros(T,p,n+m),c_part2)  for k = 1:N-1]
    λ          = [BlockArray(ones(T,p),c_part) for k = 1:N-1]
    μ          = [BlockArray(ones(T,p),c_part) for k = 1:N-1]
    active_set = [BlockArray(ones(Bool,p),c_part)     for k = 1:N-1]
    push!(C,BlockVector(T,c_term))
    push!(∇C,BlockMatrix(T,c_term,n,0))
    push!(λ,BlockVector(T,c_term))
    push!(μ,BlockArray(ones(T,num_constraints(c_term)), create_partition(c_term)))
    push!(active_set,BlockVector(Bool,c_term))

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


"Second-order Taylor expansion of cost function at time step k"
function cost_expansion!(solver::iLQRSolver,cost::QuadraticCost, x::Vector{T},
        u::Vector{T}, k::Int) where T
    Q = solver.Q[k]
    Q.x .= cost.Q*x + cost.q
    Q.u .= cost.R*u
    Q.xx .= cost.Q
    Q.uu .= cost.R
    Q.ux .= cost.H
    return nothing
end

function cost_expansion!(solver::iLQRSolver{T},cost::QuadraticCost, xN::Vector{T}) where T
    solver.S[end] .= cost.Qf
    solver.s[end] .= cost.Qf*xN + cost.qf
    return nothing
end

function cost_expansion!(solver::iLQRSolver{T},cost::ALCost{T},
        x::AbstractVector{T},u::AbstractVector{T}, k::Int) where T
    Q = solver.Q[k]
    cost_expansion!(solver, cost.cost, x, u, k)
    c = cost.C[k]
    λ = cost.λ[k]
    μ = cost.μ[k]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    ∇c = cost.∇C[k]
    jacobian!(cost.cost,∇c,cost.constraints,x,u,k)
    cx = ∇c.x
    cu = ∇c.u

    # Second Order pieces
    Q.xx .+= cx'Iμ*cx
    Q.uu .+= cu'Iμ*cu
    Q.ux .+= cu'Iμ*cx

    # First order pieces
    g = (Iμ*c + λ)
    Q.x .+= cx'g
    Q.u .+= cu'g

    return nothing
end

function cost_expansion!(solver::iLQRSolver,cost::ALCost{T},x::AbstractVector{T}) where T

    cost_expansion!(solver,cost.cost,x)
    N = length(cost.μ)

    c = cost.C[N]
    λ = cost.λ[N]
    μ = cost.μ[N]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    cx = cost.∇C[N]

    jacobian!(cx,cost.constraints,x)

    # Second Order pieces
    solver.S[N] .+= cx'Iμ*cx

    # First order pieces
    solver.s[N] .+= cx'*(Iμ*c + λ)

    return nothing
end

function cost_expansion!(solver::iLQRSolver{T},cost::GenericCost, x::Vector{T},
        u::Vector{T}, k::Int) where T

    e = cost.expansion(x,u)
    Q = solver.Q[k]

    Q.x .= e[4]
    Q.u .= e[5]
    Q.xx .= e[1]
    Q.uu .= e[2]
    Q.ux .= e[3]
    return nothing
end

function cost_expansion!(solver::iLQRSolver{T},cost::GenericCost, xN::Vector{T}) where T
    Qf, qf = cost.expansion(xN)
    solver.S[end] .= Qf
    solver.s[end] .= qf
    return nothing
end

"$(TYPEDEF) ALTRO solver"
struct ALTROSolver{T} <: AbstractSolver{T}
    opts::ALTROSolverOptions{T}
    solver_al::AugmentedLagrangianSolver{T}
end

function AbstractSolver(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    solver_al = AbstractSolver(prob,opts.opts_con)
    ALTROSolver{T}(opts,solver_al)
end

"Second-order Taylor expansion of minimum time cost function at time step k"
function cost_expansion!(solver::iLQRSolver{T},cost::MinTimeCost{T}, x::Vector{T},
        u::Vector{T}, k::Int) where T

    @assert cost.cost isa QuadraticCost
    n,m = get_sizes(cost.cost)
    idx = (x=1:n,u=1:m)
    R_min_time = cost.R_min_time
    Q = solver.Q[k]
    τ = u[end]
    Qx = cost.cost.Q*x[idx.x] + cost.cost.q
    Qu = cost.cost.R*u[idx.u] + cost.cost.r
    Q.x[idx.x] .= Qx*τ
    Q.u[idx.u] .= Qu*τ
    Q.xx[idx.x,idx.x] .= cost.cost.Q*τ
    Q.uu[idx.u,idx.u] .= cost.cost.R*τ
    Q.ux[idx.u,idx.x] .= cost.cost.H*τ

    ℓ1 = stage_cost(cost.cost,x[idx.x],u[idx.u])
    tmp = 2.0*τ*Qu

    Q.u[end] = τ*(2.0*ℓ1 + R_min_time)
    Q.uu[idx.u,end] = tmp
    Q.uu[end,idx.u] = tmp'
    Q.uu[end,end] = (2.0*ℓ1 + R_min_time)
    Q.ux[end,idx.x] = 2.0*τ*Qx'

    Q.x[end] = R_min_time*x[end]
    Q.xx[end,end] = R_min_time

    return nothing
end

function cost_expansion!(solver::iLQRSolver,cost::MinTimeCost,xN::Vector{T}) where T
    n, = get_sizes(cost.cost)
    R_min_time = cost.R_min_time
    S = solver.S[end]
    s = solver.s[end]
    idx = 1:n
    S[idx,idx] = cost.cost.Qf
    s[idx] = cost.cost.Qf*xN[idx] + cost.cost.qf
    S[end,end] = R_min_time*xN[end]
    s[end] = R_min_time

    return nothing
end

jacobian!(cost::CostFunction,∇c,constraints::AbstractConstraintSet,x::Vector{T},u::Vector{T},k::Int) where T = jacobian!(∇c,constraints,x,u)
function jacobian!(cost::MinTimeCost{T},∇c,constraints::AbstractConstraintSet,x::Vector{T},u::Vector{T},k::Int) where T
    jacobian!(∇c,constraints,x,u)
    k == 1 ? ∇c[:min_time_eq][:] .= 0.0 : nothing
end
