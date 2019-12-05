export
    StaticiLQRSolverOptions,
    StaticiLQRSolver


@with_kw mutable struct iLQRStats{T}
    iterations::Int = 0
    cost::Vector{T} = [0.]
    dJ::Vector{T} = [0.]
    gradient::Vector{T} = [0.]
    dJ_zero_counter::Int = 0
end

function reset!(stats::iLQRStats, N=0)
    stats.iterations = 0
    stats.cost = zeros(N)
    stats.dJ = zeros(N)
    stats.gradient = zeros(N)
    stats.dJ_zero_counter = 0
end


"""$(TYPEDEF)
Solver options for the iterative LQR (iLQR) solver.
$(FIELDS)
"""
@with_kw mutable struct StaticiLQRSolverOptions{T} <: AbstractSolverOptions{T}
    # Options

    "Print summary at each iteration."
    verbose::Bool=false

    "Live plotting."
    live_plotting::Symbol=:off # :state, :control

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve."
    cost_tolerance::T = 1.0e-4

    "gradient type: :todorov, :feedforward."
    gradient_type::Symbol = :todorov

    "gradient_norm < ϵ, gradient norm convergence criteria."
    gradient_norm_tolerance::T = 1.0e-5

    "iLQR iterations."
    iterations::Int = 300

    "restricts the total number of times a forward pass fails, resulting in regularization, before exiting."
    dJ_counter_limit::Int = 10

    "use square root method backward pass for numerical conditioning."
    square_root::Bool = false

    "forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound."
    line_search_lower_bound::T = 1.0e-8

    "forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < ∞."
    line_search_upper_bound::T = 10.0

    "maximum number of backtracking steps during forward pass line search."
    iterations_linesearch::Int = 20

    # Regularization
    "initial regularization."
    bp_reg_initial::T = 0.0

    "regularization scaling factor."
    bp_reg_increase_factor::T = 1.6

    "maximum regularization value."
    bp_reg_max::T = 1.0e8

    "minimum regularization value."
    bp_reg_min::T = 1.0e-8

    "type of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization."
    bp_reg_type::Symbol = :control

    "additive regularization when forward pass reaches max iterations."
    bp_reg_fp::T = 10.0

    # square root backward pass options:
    "type of matrix inversion for bp sqrt step."
    bp_sqrt_inv_type::Symbol = :pseudo

    "initial regularization for square root method."
    bp_reg_sqrt_initial::T = 1.0e-6

    "regularization scaling factor for square root method."
    bp_reg_sqrt_increase_factor::T = 10.0

    # Solver Numerical Limits
    "maximum cost value, if exceded solve will error."
    max_cost_value::T = 1.0e8

    "maximum state value, evaluated during rollout, if exceded solve will error."
    max_state_value::T = 1.0e8

    "maximum control value, evaluated during rollout, if exceded solve will error."
    max_control_value::T = 1.0e8
end


"""$(TYPEDEF)
iLQR is an unconstrained indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller.
The main algorithm consists of two parts:
1) a backward pass that uses Differential Dynamic Programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, `K` and `d`, respectively, for an LQR tracking controller, and
2) a forward pass that uses the gains `K` and `d` to simulate forward the full nonlinear dynamics with feedback.
"""
struct StaticiLQRSolver{T,N,M,NM,G,Q} <: AbstractSolver{T}
    opts::StaticiLQRSolverOptions{T}
    stats::iLQRStats{T}

    # Data variables
    X̄::Vector{N} # states (n,N)
    Ū::Vector{M} # controls (m,N-1)

    K::Vector{NM}  # State feedback gains (m,n,N-1)
    d::Vector{M} # Feedforward gains (m,N-1)

    ∇F::Vector{G}# discrete dynamics jacobian (block) (n,n+m+1,N)

    S::Q # Optimal cost-to-go expansion trajectory
    Q::Q # cost-to-go expansion trajectory

    ρ::Vector{T} # Regularization
    dρ::Vector{T} # Regularization rate of change

    grad::Vector{T} # Gradient
end

function StaticiLQRSolver(prob::StaticProblem, opts=StaticiLQRSolverOptions())
     AbstractSolver(prob, opts)
end

function AbstractSolver(prob::StaticProblem, opts::StaticiLQRSolverOptions{T}) where {T<:AbstractFloat,D<:DynamicsType}
    # Init solver statistics
    stats = iLQRStats{T}() # = Dict{Symbol,Any}(:timer=>TimerOutput())

    # Init solver results
    n,m,N = size(prob)

    X̄  = [@SVector zeros(T,n)   for k = 1:N]
    Ū  = [@SVector zeros(T,m)   for k = 1:N-1]

    K  = [@SMatrix zeros(T,m,n) for k = 1:N-1]
    d  = [@SVector zeros(T,m)   for k = 1:N-1]

    ∇F = [@SMatrix zeros(T,n,n+m+1) for k = 1:N-1]

    S = CostExpansion(n,m,N)
    Q = CostExpansion(n,m,N)

    ρ = zeros(T,1)
    dρ = zeros(T,1)

    grad = zeros(T,N-1)

    solver = StaticiLQRSolver(opts,stats,X̄,Ū,K,d,∇F,S,Q,ρ,dρ,grad)

    reset!(solver)
    return solver
end

function reset!(solver::StaticiLQRSolver{T}, reset_stats=true) where T
    if reset_stats
        reset!(solver.stats, solver.opts.iterations)
    end
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end

function copy(r::StaticiLQRSolver{T}) where T
    StaticiLQRSolver{T}(copy(r.opts),copy(r.stats),copy(r.X̄),copy(r.Ū),copy(r.K),copy(r.d),copy(r.∇F),copy(r.S),copy(r.Q),copy(r.ρ),copy(r.dρ))
end
