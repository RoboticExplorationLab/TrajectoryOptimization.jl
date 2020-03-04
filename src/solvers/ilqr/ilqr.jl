export
	iLQRSolver2

struct iLQRSolver2{T,I<:QuadratureRule,L,O,n,n̄,m,L1,GT} <: iLQRSolver{T}
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
    K::Vector{SizedMatrix{m,n̄,T,2}}  # State feedback gains (m,n,N-1)
    d::Vector{SizedVector{m,T,1}}  # Feedforward gains (m,N-1)

    D::Vector{SizedDynamicsExpansion{T,n,n̄,m}}  # discrete dynamics jacobian (block) (n,n+m+1,N)
    G::Vector{GT}                               # state difference jacobian (n̄, n)

    S::Vector{SizedExpansion{T,n,n̄,m}}      # Optimal cost-to-go expansion trajectory
    Q::Vector{SizedCostExpansion{T,n,n̄,m}}  # cost-to-go expansion trajectory
	E::SizedExpansion{T,n,n̄,m}

	Quu_reg::SizedMatrix{m,m,T,2}
	Qux_reg::SizedMatrix{m,n̄,T,2}
    ρ::Vector{T}   # Regularization
    dρ::Vector{T}  # Regularization rate of change

    grad::Vector{T}  # Gradient

    logger::SolverLogger

end

function iLQRSolver2(prob::Problem{QUAD,T}, opts=SolverOptions{T}()) where {QUAD,T}

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

	D = [SizedDynamicsExpansion{T}(n,n̄,m) for k = 1:N-1]
	if state_diff_jacobian(prob.model, x0) isa UniformScaling
		G = [I for k = 1:N]
	else
		G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N]
	end

    S = [SizedExpansion{T}(n,n̄,m) for k = 1:N]
	if prob.model isa RigidBody
		Q = [SizedCostExpansion{T}(n,n̄,m) for k = 1:N]
	else
		Q = [SizedCostExpansion{T}(n,m) for k = 1:N]
	end
	E = SizedExpansion{T}(n,n̄,m)
    # S = [GeneralExpansion{T}(SizedArray,n,n̄,m)   for k = 1:N]
    # Q = [GeneralExpansion{T}(SizedArray,n,n,m) for k = 1:N]
	# E = GeneralExpansion{T}(SizedArray,n,n̄,m)

	Quu_reg = SizedMatrix{m,m}(zeros(m,m))
	Qux_reg = SizedMatrix{m,n̄}(zeros(m,n̄))
    ρ = zeros(T,1)
    dρ = zeros(T,1)

    grad = zeros(T,N-1)

    logger = default_logger(opts.verbose)
	L = typeof(prob.model)
	O = typeof(prob.obj)
	ET = typeof(E)
	GT = eltype(G)

	opts_ilqr = iLQRSolverOptions(opts)
    solver = iLQRSolver2{T,QUAD,L,O,n,n̄,m,n+m,GT}(prob.model, prob.obj, x0, xf,
		prob.tf, N, opts_ilqr, stats,
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
