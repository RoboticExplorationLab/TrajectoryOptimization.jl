


"""$(TYPEDEF)
Solver options for the augmented Lagrangian solver.
$(FIELDS)
"""
@with_kw mutable struct AugmentedLagrangianSolverOptions{T} <: AbstractSolverOptions{T}
    "Print summary at each iteration."
    verbose::Bool=false

    "unconstrained solver options."
    opts_uncon::AbstractSolverOptions{T} = iLQRSolverOptions{T}()

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve."
    cost_tolerance::T = 1.0e-4

    "dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve."
    cost_tolerance_intermediate::T = 1.0e-3

    "gradient_norm < ϵ, gradient norm convergence criteria."
    gradient_norm_tolerance::T = 1.0e-5

    "gradient_norm_int < ϵ, gradient norm intermediate convergence criteria."
    gradient_norm_tolerance_intermediate::T = 1.0e-5

    "max(constraint) < ϵ, constraint convergence criteria."
    constraint_tolerance::T = 1.0e-3

    "max(constraint) < ϵ_int, intermediate constraint convergence criteria."
    constraint_tolerance_intermediate::T = 1.0e-3

    "maximum outerloop updates."
    iterations::Int = 30

    "minimum Lagrange multiplier."
    dual_min::T = -1.0e8

    "maximum Lagrange multiplier."
    dual_max::T = 1.0e8

    "maximum penalty term."
    penalty_max::T = 1.0e8

    "initial penalty term."
    penalty_initial::T = 1.0

    "penalty update multiplier; penalty_scaling > 0."
    penalty_scaling::T = 10.0

    "penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)."
    penalty_scaling_no::T = 1.0

    "ratio of current constraint to previous constraint violation; 0 < constraint_decrease_ratio < 1."
    constraint_decrease_ratio::T = 0.25

    "type of outer loop update (default, feedback)."
    outer_loop_update_type::Symbol = :default

    "numerical tolerance for constraint violation."
    active_constraint_tolerance::T = 0.0

    "terminal solve when maximum penalty is reached."
    kickout_max_penalty::Bool = false

end



@doc raw""" ```julia
struct AugmentedLagrangianSolver <: TrajectoryOptimization.AbstractSolver{T}
```
Augmented Lagrangian (AL) is a standard tool for constrained optimization. For a trajectory optimization problem of the form:
```math
\begin{equation*}
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
\end{equation*}
```
AL methods form the following augmented Lagrangian function:
```math
\begin{align*}
    \ell_f(x_N) + &λ_N^T c_N(x_N) + c_N(x_N)^T I_{\mu_N} c_N(x_N) \\
           & + \sum_{k=0}^{N-1} \ell_k(x_k,u_k,dt) + λ_k^T c_k(x_k,u_k) + c_k(x_k,u_k)^T I_{\mu_k} c_k(x_k,u_k)
\end{align*}
```

This function is then minimized with respect to the primal variables using any unconstrained minimization solver (e.g. iLQR).
    After a local minima is found, the AL method updates the Lagrange multipliers λ and the penalty terms μ and repeats the unconstrained minimization.
    AL methods have superlinear convergence as long as the penalty term μ is updated each iteration.
"""
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

    solver_uncon::AbstractSolver
end

AugmentedLagrangianSolver(prob::Problem{T},
    opts::AugmentedLagrangianSolverOptions{T}=AugmentedLagrangianSolverOptions{T}()) where T =
    AbstractSolver(prob,opts)

"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AbstractSolver(prob::Problem{T,D}, opts::AugmentedLagrangianSolverOptions{T}) where {T<:AbstractFloat,D<:DynamicsType}
    # check for conflicting convergence criteria between unconstrained solver and AL: warn

    # Init solver statistics
    to = TimerOutput()
    reset_timer!(to)
    stats = Dict{Symbol,Any}(:iterations=>0,:iterations_total=>0,
        :iterations_inner=>Int[],:cost=>T[],:c_max=>T[],:penalty_max=>[],
        :timer=>to)
    stats_uncon = Dict{Symbol,Any}[]

    # Init solver results
    n = prob.model.n; m = prob.model.m; N = prob.N

    C,∇C,λ,μ,active_set = init_constraint_trajectories(prob.constraints,n,m,N)

    solver_uncon = AbstractSolver(prob,opts.opts_uncon)
    solver_uncon.stats[:timer] = to

    AugmentedLagrangianSolver{T}(opts,stats,stats_uncon,C,copy(C),∇C,λ,μ,active_set,solver_uncon)
end


function init_constraint_trajectories(constraints::Constraints,n::Int,m::Int,N::Int;
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
AugmentedLagrangianSolver(prob::Problem, opts::AugmentedLagrangianSolverOptions) =
    AbstractSolver(prob, opts)

function reset!(solver::AugmentedLagrangianSolver{T}) where T
    solver.stats[:iterations]       = 0
    solver.stats[:iterations_total] = 0
    solver.stats[:iterations_inner] = T[]
    solver.stats[:cost]             = T[]
    solver.stats[:c_max]            = T[]
    solver.stats[:penalty_max]      = T[]
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
