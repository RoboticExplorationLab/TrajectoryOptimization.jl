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

function Expansion(prob::Problem{T},exp::Symbol) where T
    n = prob.model.n; m = prob.model.m
    if exp == :x
        return Expansion(zeros(T,n),zeros(T,0),zeros(T,n,n),zeros(T,0,0),zeros(T,0,0))
    elseif exp == :u
        return Expansion(zeros(T,0),zeros(T,m),zeros(T,0,0),zeros(T,m,m),zeros(T,0,0))
    else
        error("Invalid expansion components requested")
    end
end

import Base./, Base.*

function *(e::Expansion{T},a::T) where T
    e.x .*= a
    e.u .*= a
    e.xx .*= a
    e.uu .*= a
    e.ux .*= a
    return nothing
end

function /(e::Expansion{T},a::T) where T
    e.x ./= a
    e.u ./= a
    e.xx ./= a
    e.uu ./= a
    e.ux ./= a
    return nothing
end

function copy(e::Expansion{T}) where T
    Expansion{T}(copy(e.x),copy(e.u),copy(e.xx),copy(e.uu),copy(e.ux))
end

function reset!(e::Expansion)
    !isempty(e.x) ? e.x .= zero(e.x) : nothing
    !isempty(e.u) ? e.u .= zero(e.u) : nothing
    !isempty(e.xx) ? e.xx .= zero(e.xx) : nothing
    !isempty(e.uu) ? e.uu .= zero(e.uu) : nothing
    !isempty(e.ux) ? e.ux .= zero(e.ux) : nothing
    return nothing
end

ExpansionTrajectory{T} = Vector{Expansion{T}} where T <: AbstractFloat

function reset!(et::ExpansionTrajectory)
    for e in et
        reset!(e)
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

    X̄  = [zeros(T,n)   for i = 1:N]
    Ū  = [zeros(T,m)   for i = 1:N-1]

    K  = [zeros(T,m,n) for i = 1:N-1]
    d  = [zeros(T,m)   for i = 1:N-1]

    part_f = create_partition2(prob.model)
    ∇F = [BlockArray(zeros(n,n+m+1),part_f) for i = 1:N-1]

    S  = [Expansion(prob,:x) for i = 1:N]
    Q = [Expansion(prob) for i = 1:N-1]

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
    # p = num_stage_constraints(prob)

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

function init_constraint_trajectories(constraints::ProblemConstraints,n::Int,m::Int,N::Int;
        μ_init::T=1.,λ_init::T=0.) where T

    p = [num_stage_constraints(constraints[k]) for k = 1:N-1]
    c_stage = [stage(constraints[k]) for k = 1:N-1]
    c_part = [create_partition(c_stage[k]) for k = 1:N-1]
    c_part2 = [create_partition2(c_stage[k],n,m) for k = 1:N-1]

    # Create Trajectories
    C          = [BlockArray(zeros(T,p[k]),c_part[k])       for k = 1:N-1]
    ∇C         = [BlockArray(zeros(T,p[k],n+m),c_part2[k])  for k = 1:N-1]
    λ          = [BlockArray(ones(T,p[k]),c_part[k]) for k = 1:N-1]
    μ          = [BlockArray(ones(T,p[k]),c_part[k]) for k = 1:N-1]
    active_set = [BlockArray(ones(Bool,p[k]),c_part[k])     for k = 1:N-1]

    c_term = terminal(constraints[N])
    p_N = num_constraints(c_term)

    push!(C,BlockVector(T,c_term))
    push!(∇C,BlockMatrix(T,c_term,n,0))
    push!(λ,BlockVector(T,c_term))
    push!(μ,BlockArray(ones(T,p_N), create_partition(c_term)))
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

# "Second-order Taylor expansion of cost function at time step k"
# function cost_expansion!(Q::Expansion{T}, cost::QuadraticCost, x::Vector{T},
#         u::Vector{T}, k::Int) where T
#     Q.x .= cost.Q*x + cost.q
#     Q.u .= cost.R*u
#     Q.xx .= cost.Q
#     Q.uu .= cost.R
#     Q.ux .= cost.H
#     return nothing
# end
#
# function cost_expansion!(S::Expansion{T},cost::QuadraticCost, xN::Vector{T}) where T
#     S.xx .= cost.Qf
#     S.x .= cost.Qf*xN + cost.qf
#     return nothing
# end

function cost_expansion!(Q::ExpansionTrajectory{T},obj::ALObjectiveNew{T},
        X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X)

    cost_expansion!(Q, obj.cost, X, U)

    for k = 1:N-1
        c = obj.C[k]
        λ = obj.λ[k]
        μ = obj.μ[k]
        a = active_set(c,λ)
        Iμ = Diagonal(a .* μ)
        jacobian!(obj.∇C[k],obj.constraints[k],X[k],U[k])
        cx = obj.∇C[k].x
        cu = obj.∇C[k].u

        # Second Order pieces
        Q[k].xx .+= cx'Iμ*cx
        Q[k].uu .+= cu'Iμ*cu
        Q[k].ux .+= cu'Iμ*cx

        # First order pieces
        g = (Iμ*c + λ)
        Q[k].x .+= cx'g
        Q[k].u .+= cu'g
    end

    return nothing
end

function cost_expansion!(S::Expansion{T},obj::ObjectiveNew,x::AbstractVector{T}) where T
    cost_expansion!(S,obj.cost[end],x)
    return nothing
end

function cost_expansion!(S::Expansion{T},obj::ALObjectiveNew{T},x::AbstractVector{T}) where T
    N = length(obj.μ)
    cost_expansion!(S,obj[N],x)

    c = obj.C[N]
    λ = obj.λ[N]
    μ = obj.μ[N]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    cx = obj.∇C[N]

    jacobian!(cx,obj.constraints[N],x)

    # Second Order pieces
    S.xx .+= cx'Iμ*cx

    # First order pieces
    S.x .+= cx'*(Iμ*c + λ)

    return nothing
end

# function cost_expansion!(Q::Expansion{T},cost::GenericCost, x::Vector{T},
#         u::Vector{T}, k::Int) where T
#
#     e = cost.expansion(x,u)
#
#     Q.x .= e[4]
#     Q.u .= e[5]
#     Q.xx .= e[1]
#     Q.uu .= e[2]
#     Q.ux .= e[3]
#     return nothing
# end
#
# function cost_expansion!(S::Expansion{T},cost::GenericCost, xN::Vector{T}) where T
#     Qf, qf = cost.expansion(xN)
#     S.xx .= Qf
#     S.x .= qf
#     return nothing
# end

"$(TYPEDEF) ALTRO solver"
struct ALTROSolver{T} <: AbstractSolver{T}
    opts::ALTROSolverOptions{T}
    solver_al::AugmentedLagrangianSolver{T}
end

function AbstractSolver(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    solver_al = AbstractSolver(prob,opts.opts_al)
    ALTROSolver{T}(opts,solver_al)
end

# "Second-order Taylor expansion of minimum time cost function at time step k"
# function cost_expansion!(Q::Expansion{T},cost::MinTimeCost{T}, x::Vector{T},
#         u::Vector{T}) where T
#
#     @assert cost.cost isa QuadraticCost
#     n,m = get_sizes(cost.cost)
#     idx = (x=1:n,u=1:m)
#     R_min_time = cost.R_min_time
#     τ = u[end]
#
#     Qx = cost.cost.Q*x[idx.x] + cost.cost.q
#     Qu = cost.cost.R*u[idx.u] + cost.cost.r
#     Q.x[idx.x] .= Qx
#     Q.u[idx.u] .= Qu
#     Q.xx[idx.x,idx.x] .= cost.cost.Q
#     Q.uu[idx.u,idx.u] .= cost.cost.R
#     Q.ux[idx.u,idx.x] .= cost.cost.H
#
#
#     Q.u[end] = 2.0*τ*R_min_time
#     Q.uu[idx.u,end] = zeros(m)
#     Q.uu[end,idx.u] = zeros(m)
#     Q.uu[end,end] = 2.0*R_min_time
#     Q.ux[end,idx.x] = zeros(n)
#
#     return nothing
# end
#
# function cost_expansion!(S::Expansion{T},cost::MinTimeCost,xN::Vector{T}) where T
#     n, = get_sizes(cost.cost)
#     R_min_time = cost.R_min_time
#
#     idx = 1:n
#     S.xx[idx,idx] = cost.cost.Qf
#     S.x[idx] = cost.cost.Qf*xN[idx] + cost.cost.qf
#
#     return nothing
# end

# jacobian!(cost::CostFunction,∇c,constraints::AbstractConstraintSet,x::Vector{T},u::Vector{T},k::Int) where T = jacobian!(∇c,constraints,x,u)
# function jacobian!(cost::MinTimeCost{T},∇c,constraints::AbstractConstraintSet,x::Vector{T},u::Vector{T},k::Int) where T
#     jacobian!(∇c,constraints,x,u)
#     k == 1 ? ∇c[:min_time_eq][:] .= 0.0 : nothing
# end


## new objective stuff
"Calculate unconstrained cost for X and U trajectories"
function cost(obj::ObjectiveNew, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(obj[k],X[k],U[k])
    end
    J /= (N-1.0)
    J += stage_cost(obj[N],X[N])
    return J
end

function stage_cost(cost::QuadraticCost, x::Vector{T}, u::Vector{T}) where T
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c
end

function stage_cost(cost::QuadraticCost, xN::Vector{T}) where T
    0.5*xN'cost.Qf*xN + cost.qf'*xN + cost.cf
end

stage_cost(cost::GenericCost, x::Vector{T}, u::Vector{T}) where T = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::Vector{T}) where T = cost.ℓf(xN)

stage_cost(cost::MinTimeCost, x::Vector{T}, u::Vector{T}) where T = stage_cost(cost.cost,x[1:end-1],u[1:end-1],k) + cost.R_min_time*u[end]^2
stage_cost(cost::MinTimeCost, xN::Vector{T}) where T = stage_cost(cost.cost,xN[1:end-1])

function cost_expansion!(Q::ExpansionTrajectory{T},obj::ObjectiveNew,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    cost_expansion!(Q,obj.cost,X,U)
end
function cost_expansion!(Q::ExpansionTrajectory{T},c::CostTrajectory,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        cost_expansion!(Q[k],c[k],X[k],U[k])
    end
end

function cost_expansion!(Q::Expansion{T}, cost::QuadraticCost, x::Vector{T},
        u::Vector{T}) where T
    Q.x .= cost.Q*x + cost.q
    Q.u .= cost.R*u + cost.r
    Q.xx .= cost.Q
    Q.uu .= cost.R
    Q.ux .= cost.H
    return nothing
end

function cost_expansion!(S::Expansion{T}, cost::QuadraticCost, xN::Vector{T}) where T
    S.xx .= cost.Qf
    S.x .= cost.Qf*xN + cost.qf
    return nothing
end

function cost_expansion!(Q::Expansion{T}, cost::GenericCost, x::Vector{T},
        u::Vector{T}) where T

    e = cost.expansion(x,u)

    Q.x .= e[4]
    Q.u .= e[5]
    Q.xx .= e[1]
    Q.uu .= e[2]
    Q.ux .= e[3]
    return nothing
end

function cost_expansion!(S::Expansion{T}, cost::GenericCost, xN::Vector{T}) where T
    Qf, qf = cost.expansion(xN)
    S.xx .= Qf
    S.x .= qf
    return nothing
end

# Augmented Lagrangian
## new objective stuff
"Calculate unconstrained cost for X and U trajectories"
function cost(c::CostTrajectory, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(c[k],X[k],U[k])
    end
    J /= (N-1.0)
    J += stage_cost(c[N],X[N])
    return J
end

function cost(obj::AbstractObjective, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    cost(obj.cost,X,U)
end

## Augmented Lagrangian

# "Update constraints trajectories"
# function update_constraints!(C::PartedVecTrajectory{T},constraints::AbstractConstraintSet,
#         X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
#     N = length(X)
#     for k = 1:N-1
#         evaluate!(C[k],constraints,X[k],U[k])
#     end
#     evaluate!(C[N],constraints,X[N])
# end

"Update constraints trajectories"
function update_constraints!(C::PartedVecTrajectory{T},constraints::ProblemConstraints,
        X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        evaluate!(C[k],constraints[k],X[k],U[k])
    end
    evaluate!(C[N],constraints[N],X[N])
end
# function update_constraints!(C::PartedVecTrajectory{T},constraints::AbstractConstraintSet,
#         X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
#     N = length(X)
#     for k = 1:N-1
#         evaluate!(C[k],constraints,X[k],U[k])
#     end
#     C[1][:min_time_eq][1] = 0.0 # no constraint at timestep one
#     evaluate!(C[N],constraints,X[N][1:end-1])
# end

"Evaluate active set constraints for entire trajectory"
function update_active_set!(a::PartedVecTrajectory{Bool},c::PartedVecTrajectory{T},λ::PartedVecTrajectory{T},tol::T=0.0) where T
    N = length(c)
    for k = 1:N
        active_set!(a[k], c[k], λ[k], tol)
    end
end

function update_active_set!(obj::ALObjectiveNew{T},tol::T=0.0) where T
    update_active_set!(obj.active_set,obj.C,obj.λ,tol)
end

"Evaluate active set constraints for a single time step"
function active_set!(a::AbstractVector{Bool}, c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    # inequality_active!(a,c,λ,tol)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return nothing
end

function active_set(c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    a = BlockArray(trues(length(c)),c.parts)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return a
end

"Cost function terms for Lagrangian and quadratic penalty"
function aula_cost(a::AbstractVector{Bool},c::AbstractVector{T},λ::AbstractVector{T},μ::AbstractVector{T}) where T
    λ'c + 1/2*c'Diagonal(a .* μ)*c
end

function stage_constraint_cost(obj::ALObjectiveNew{T},x::AbstractVector{T},u::AbstractVector{T},k::Int) where T
    c = obj.C[k]
    λ = obj.λ[k]
    μ = obj.μ[k]
    a = obj.active_set[k]
    aula_cost(a,c,λ,μ)
end

function stage_constraint_cost(obj::ALObjectiveNew{T},x::AbstractVector{T}) where T
    c = obj.C[end]
    λ = obj.λ[end]
    μ = obj.μ[end]
    a = obj.active_set[end]
    aula_cost(a,c,λ,μ)
end

function stage_constraint_cost(c,λ,μ,
        a,x::AbstractVector{T},u::AbstractVector{T}) where T
    # c = obj.C[k]
    # λ = obj.λ[k]
    # μ = obj.μ[k]
    # a = obj.active_set[k]
    aula_cost(a,c,λ,μ)
end

function stage_constraint_cost(c,λ,μ,
        a,x::AbstractVector{T}) where T
    # c = obj.C[end]
    # λ = obj.λ[end]
    # μ = obj.μ[end]
    # a = obj.active_set[end]
    aula_cost(a,c,λ,μ)
end
#
# function stage_cost(alcost::ALCost{T}, x::AbstractVector{T}, u::AbstractVector{T}, k::Int) where T
#     J0 = stage_cost(alcost.cost,x,u,k)
#     J0 + stage_constraint_cost(alcost,x,u,k)
# end
#
# function stage_cost(alcost::ALCost{T}, x::AbstractVector{T}) where T
#     J0 = stage_cost(alcost.cost,x)
#     J0 + stage_constraint_cost(alcost,x)
# end


"Augmented Lagrangian cost for X and U trajectories"
function cost(obj::ALObjectiveNew{T},X::VectorTrajectory{T},U::VectorTrajectory{T}) where T <: AbstractFloat
    N = length(X)
    J = cost(obj.cost,X,U)

    update_constraints!(obj.C,obj.constraints, X, U)
    update_active_set!(obj)

    Jc = 0.0
    for k = 1:N-1
        Jc += stage_constraint_cost(obj.C[k], obj.λ[k], obj.μ[k],obj.active_set[k],X[k],U[k])
    end
    Jc /= (N-1.0)
    Jc += stage_constraint_cost(obj.C[N], obj.λ[N], obj.μ[N],obj.active_set[N],X[N])
    return J + Jc
end
