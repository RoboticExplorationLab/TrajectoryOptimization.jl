export
	iLQRSolver2

struct iLQRSolver2{T,I<:QuadratureRule,L,O,n,m,L1} <: iLQRSolver{T}
    # Model + Objective
    model::L
    obj::O

    # Problem info
    x0::MVector{n,T}
    xf::MVector{n,T}
    tf::T
    N::Int

    opts::iLQRSolverOptions{T}
    stats::iLQRStats{T}

    # Primal Duals
    Z::Vector{KnotPoint{T,n,m,L1}}
    Z̄::Vector{KnotPoint{T,n,m,L1}}

    # Data variables
    # K::Vector{SMatrix{m,n̄,T,L2}}  # State feedback gains (m,n,N-1)
    K::Vector{Matrix{T}}  # State feedback gains (m,n,N-1)
    d::Vector{Vector{T}} # Feedforward gains (m,N-1)

    D::Vector{DynamicsExpansion{T}} # discrete dynamics jacobian (block) (n,n+m+1,N)
    G::Vector{Matrix{T}}  # state difference jacobian (n̄, n)

    S::Vector{Expansion{T}}  # Optimal cost-to-go expansion trajectory
    Q::Vector{Expansion{T}}  # cost-to-go expansion trajectory
	E::Expansion{T}          # error cost expansion

	Quu_reg::Matrix{T}
	Qux_reg::Matrix{T}
    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

    grad::Vector{T} # Gradient

    logger::SolverLogger

end

function iLQRSolver2(prob::Problem{QUAD,T}, opts=iLQRSolverOptions()) where {QUAD,T}

    # Init solver statistics
    stats = iLQRStats{T}() # = Dict{Symbol,Any}(:timer=>TimerOutput())

    # Init solver results
    n,m,N = size(prob)
    n̄ = state_diff_size(prob.model)

    x0 = SVector{n}(prob.x0)
    xf = SVector{n}(prob.xf)

    Z = prob.Z
    # Z̄ = Traj(n,m,Z[1].dt,N)
    Z̄ = copy(prob.Z)

	K = [zeros(T,m,n̄) for k = 1:N-1]
    d = [zeros(T,m)   for k = 1:N-1]

	D = [DynamicsExpansion{T}(n,n̄,m) for k = 1:N-1]
    G = [zeros(n,n̄) for k = 1:N]
	if state_diff_jacobian(prob.model, x0) isa UniformScaling
		G = [I(n) for k = 1:N]
	end

    S = [Expansion{T}(n̄,m)   for k = 1:N]
    Q = [Expansion{T}(n,m) for k = 1:N]
	E = Expansion{T}(n,n̄,m)

	Quu_reg = zeros(m,m)
	Qux_reg = zeros(m,n̄)
    ρ = zeros(T,1)
    dρ = zeros(T,1)

    grad = zeros(T,N-1)

    logger = default_logger(opts.verbose)
	L = typeof(prob.model)
	O = typeof(prob.obj)

    solver = iLQRSolver2{T,QUAD,L,O,n,m,n+m}(prob.model, prob.obj, x0, xf, prob.tf, N, opts, stats,
        Z, Z̄, K, d, D, G, S, Q, E, Quu_reg, Qux_reg, ρ, dρ, grad, logger)

    reset!(solver)
    return solver
end

function reset!(solver::iLQRSolver2{T}, reset_stats=true) where T
    if reset_stats
        reset!(solver.stats, solver.opts.iterations)
    end
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end

Base.size(solver::iLQRSolver2{T,I,L,O,n,m}) where {T,I,L,O,n,m} = n,m,solver.N
@inline get_trajectory(solver::iLQRSolver2) = solver.Z
@inline get_objective(solver::iLQRSolver2) = solver.obj
@inline get_model(solver::iLQRSolver2) = solver.model
@inline get_initial_state(solver::iLQRSolver2) = solver.x0

function cost(solver::iLQRSolver2, Z=solver.Z)
    cost!(solver.obj, Z)
    return sum(get_J(solver.obj))
end
