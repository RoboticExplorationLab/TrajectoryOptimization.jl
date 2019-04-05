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

abstract type AbstractSolver{T<:AbstractFloat} end

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
    stats_uncon::Vector{Dict{Symbol,Any}}  # Stash of unconstraint stats

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

function init_constraint_trajectories(constraints::ConstraintSet,n::Int,m::Int,N::Int;
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

get_sizes(solver::AugmentedLagrangianSolver) = size(solver.∇C[1].x,2), size(solver.∇C[1].u,2), length(solver.λ)


"Second-order Taylor expansion of cost function at time step k"
function cost_expansion!(e::ExpansionTrajectory,cost::QuadraticCost, x::Vector{T}, u::Vector{T}, k::Int) where T
    n,m = get_sizes(cost)
    e[k].x[1:n] .= cost.Q*x + cost.q
    e[k].u[1:m] .= cost.R*u[1:m]
    e[k].xx[1:n,1:n] .= cost.Q
    e[k].uu[1:m,1:m] .= cost.R
    e[k].ux[1:m,1:n] .= cost.H
    return nothing
end

function cost_expansion!(solver::iLQRSolver,cost::QuadraticCost, xN::Vector{T}) where T
    solver.S[end] .= cost.Qf
    solver.s[end] .= cost.Qf*xN + cost.qf
    return nothing
end

function cost_expansion!(e::ExpansionTrajectory,cost::AugmentedLagrangianCost{T},
        x::AbstractVector{T},u::AbstractVector{T}, k::Int) where T
    n,m = get_sizes(cost.cost)
    cost_expansion!(e,cost.cost, x, u, k)
    c = cost.C[k]
    λ = cost.λ[k]
    μ = cost.μ[k]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    ∇c = cost.∇C[k]
    jacobian!(∇c,cost.constraints,x,u)
    cx = ∇c.x
    cu = ∇c.u

    # Second Order pieces
    e[k].xx[1:n,1:n] .+= cx'Iμ*cx
    e[k].uu[1:m,1:m] .+= cu'Iμ*cu
    e[k].ux[1:m,1:n] .+= cu'Iμ*cx

    # First order pieces
    g = (Iμ*c + λ)
    e[k].x[1:n] .+= cx'g
    e[k].u[1:m] .+= cu'g

    return nothing
end

function cost_expansion!(solver::iLQRSolver,cost::AugmentedLagrangianCost{T},x::AbstractVector{T}) where T
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

#TODO change generic cost expansiont to perform in-place
function cost_expansion!(e::ExpansionTrajectory,cost::GenericCost, x::Vector{T}, u::Vector{T}, k::Int) where T
    n,m = get_sizes(cost)
    Q,R,H,q,r = cost.expansion(x,u)
    e[k].x[1:n] .= Q
    e[k].u[1:m] .= R
    e[k].xx[1:n,1:n] .= cost.Q
    e[k].uu[1:m,1:m] .= cost.R
    e[k].ux[1:m,1:n] .= cost.H
    return nothing
end

function cost_expansion!(solver::iLQRSolver,cost::GenericCost, xN::Vector{T}) where T
    Qf, qf = cost.expansion(xN)
    solver.S[end] .= Qf
    solver.s[end] .= qf
    return nothing
end

"$(TYPEDEF) ALTRO cost, potentially including infeasible start and minimum time costs"
struct ALTROCost{T} <: CostFunction
    cost::AugmentedLagrangianCost
    R_inf::T
    R_min_time::T
    n::Int # state dimension of original problem
    m::Int # input dimension of original problem
end

function ALTROCost(prob::Problem{T},cost::AugmentedLagrangianCost{T},R_inf::T,R_min_time::T) where T
    ALTROCost(cost,R_inf,R_min_time,prob.model.n,prob.model.m)
end

function get_sizes(cost::ALTROCost)
    n = cost.n
    m = cost.m

    if cost.R_min_time != NaN
        m̄ = m + 1
        n̄ = n + 1
    else
        m̄ = m
        n̄ = n
    end

    return n,m,n̄,m̄
end

#NOTE don't use these...
function stage_cost(cost::ALTROCost{T}, x::AbstractVector{T}, u::AbstractVector{T}, k::Int) where T
    n,m,n̄,m̄ = get_sizes(cost)

    J = 0.0

    # if cost.R_min_time != NaN
    #     J += cost.R_min_time*u[m̄]^2
    #     dt =
    # end
    #
    # if cost.R_inf!= NaN
    #     u_inf = u[m̄ .+(1:n)]
    #     J += 0.5*cost.R_inf*u_inf'*u_inf
    # end
    #
    # J = stage_cost(cost.cost.cost,x[1:n],u[1:m],k) # stage cost only for original x, u
    # J += stage_constraint_cost(cost.cost,x,u,k) # constraints consider


    J
end

function stage_cost(cost::ALTROCost{T}, x::AbstractVector{T}) where T
    # J0 = stage_cost(cost.cost.cost,x[1:cost.n])
    # J0 + stage_constraint_cost(cost.cost,x)
    0.0
end

"ALTRO cost for X and U trajectories"
function cost(cost::ALTROCost{T},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T) where T <: AbstractFloat
    N = length(X)
    n,m,n̄,m̄ = get_sizes(cost)

    update_constraints!(cost.cost.C,cost.cost.constraints,X,U)
    update_active_set!(cost.cost.active_set,cost.cost.C,cost.cost.λ)

    J = 0.0

    for k = 1:N-1
        # Minimum time stage cost
        if !isnan(R_min_time)
            dt = U[k][m̄]^2
            J += cost.R_min_time*dt
        end

        # Infeasible start stage cost
        if !isnan(R_inf)
            u_inf = u[m̄ .+(1:n)]
            J += 0.5*cost.R_inf*u_inf'*u_inf
        end

        J += cost(cost.cost.cost,X[k][1:n],U[k][1:m],k)*dt
        J += stage_constraint_cost(cost.cost,X[k],U[k],k)
    end

    J += cost(cost.cost.cost,X[N])
    J += stage_constraint_cost(cost.cost,X[N])

    return J
end

"Second-order expansion of ALTRO cost"
function cost_expansion!(Q::ExpansionTrajectory{T},cost::ALTROCost, x::Vector{T},
        u::Vector{T}, k::Int) where T

    n,m,n̄,m̄ = get_sizes(cost)
    R_min_time = cost.R_min_time
    R_inf = cost.R_inf
    cost_expansion!(Q,cost.cost,x,u,k)

    # Minimum time expansion components
    if !isnan(R_min_time)
        ℓ1 = stage_cost(cost.cost.cost,x,u)
        Qx, Qu = cost_expansion_gradients(cost.cost.cost,x,u,k)
        τ = u[m̄]
        tmp = 2.0*τ*Qu

        Q[k].u[m̄] = τ*(2.0*ℓ1 + R_min_time)
        Q[k].uu[1:m,m̄] = tmp
        Q[k].uu[m̄,1:m] = tmp'
        Q[k].uu[m̄,m̄] = (2.0*ℓ1 + R_min_time)
        Q[k].ux[m̄,1:n] = 2.0*τ*Qx'

        Q[k].x[n̄] = R_min_time*x[k][n̄]
        Q[k].xx[n̄,n̄] = R_min_time
    end

    # Infeasible expansion components
    if !isnan(R_inf)
        Q[k].u[m̄+1:mm] = R_inf*u[k][m̄+1:m̄+n]
        Q[k].uu[m̄+1:mm,m̄+1:mm] = R_inf
    end

    return nothing
end

function cost_expansion!(solver::iLQRSolver, cost::ALTROCost, xN::Vector{T}) where T
    cost_expansion!(solver,cost.cost,xN)
end
