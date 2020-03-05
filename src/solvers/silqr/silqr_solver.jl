


"""$(TYPEDEF)
iLQR is an unconstrained indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller.
The main algorithm consists of two parts:
1) a backward pass that uses Differential Dynamic Programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, `K` and `d`, respectively, for an LQR tracking controller, and
2) a forward pass that uses the gains `K` and `d` to simulate forward the full nonlinear dynamics with feedback.
"""
struct StaticiLQRSolver{T,I<:QuadratureRule,L,O,n,m,L1,D,F,E1,E2,A} <: iLQRSolver{T}
    # Model + Objective
    model::L
    obj::O

    # Problem info
    x0::MVector{n,T}
    xf::SVector{n,T}
    tf::T
    N::Int

    opts::iLQRSolverOptions{T}
    stats::iLQRStats{T}

    # Primal Duals
    Z::Vector{KnotPoint{T,n,m,L1}}
    Z̄::Vector{KnotPoint{T,n,m,L1}}

    # Data variables
    # K::Vector{SMatrix{m,n̄,T,L2}}  # State feedback gains (m,n,N-1)
    K::Vector{A}  # State feedback gains (m,n,N-1)
    d::Vector{SVector{m,T}} # Feedforward gains (m,N-1)

    ∇F::Vector{D} # discrete dynamics jacobian (block) (n,n+m+1,N)
    G::Vector{F}  # state difference jacobian (n̄, n)

    S::E1  # Optimal cost-to-go expansion trajectory
    Q::E2  # cost-to-go expansion trajectory

    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

    grad::Vector{T} # Gradient

    logger::SolverLogger

    function StaticiLQRSolver{T,I}(model::L, obj::O, x0, xf, tf, N, opts, stats,
            Z::Vector{KnotPoint{T,n,m,L1}}, Z̄, K::Vector{A}, d,
            ∇F::Vector{D}, G::Vector{F}, S::E1, Q::E2, ρ, dρ, grad,
            logger) where {T,I,L,O,n,m,L1,D,F,E1,E2,A}
        new{T,I,L,O,n,m,L1,D,F,E1,E2,A}(model, obj, x0, xf, tf, N, opts, stats, Z, Z̄, K, d,
            ∇F, G, S, Q, ρ, dρ, grad, logger)
    end
end

function StaticiLQRSolver(prob::Problem{I,T}, opts=SolverOptions{T}()) where {I,T}

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

    if m*n̄ > MAX_ELEM
		K  = [zeros(T,m,n̄) for k = 1:N-1]
	else
		K  = [@SMatrix zeros(T,m,n̄) for k = 1:N-1]
	end
    d  = [@SVector zeros(T,m)   for k = 1:N-1]

	if n*(n+m+1) > MAX_ELEM
		∇F = [zeros(T,n,n+m+1) for k = 1:N-1]
	else
		∇F = [@SMatrix zeros(T,n,n+m+1) for k = 1:N-1]
	end
    ∇F = [@SMatrix zeros(T,n,n+m+1) for k = 1:N-1]
    G = [state_diff_jacobian(prob.model, x0) for k = 1:N]

    S = CostExpansion(n̄,m,N)
    Q = CostExpansion(n̄,m,N)

    ρ = zeros(T,1)
    dρ = zeros(T,1)

    grad = zeros(T,N-1)

    logger = default_logger(opts.verbose)

	opts_ilqr = iLQRSolverOptions(opts)

    solver = StaticiLQRSolver{T,I}(prob.model, prob.obj, x0, xf, prob.tf, N, opts_ilqr, stats,
        Z, Z̄, K, d, ∇F, G, S, Q, ρ, dρ, grad, logger)

    reset!(solver)
    return solver
end
