export
    StaticALSolver,
    StaticALSolverOptions,
    get_constraints


@with_kw mutable struct ALStats{T}
    iterations::Int = 0
    iterations_total::Int = 0
    iterations_inner::Vector{Int} = zeros(Int,0)
    cost::Vector{T} = zeros(0)
    c_max::Vector{T} = zeros(0)
    penalty_max::Vector{T} = zeros(0)
end

function reset!(stats::ALStats, L=0)
    stats.iterations = 0
    stats.iterations_total = 0
    stats.iterations_inner = zeros(Int,L)
    stats.cost = zeros(L)*NaN
    stats.c_max = zeros(L)*NaN
    stats.penalty_max = zeros(L)*NaN
end


"""$(TYPEDEF)
Solver options for the augmented Lagrangian solver.
$(FIELDS)
"""
@with_kw mutable struct StaticALSolverOptions{T} <: AbstractSolverOptions{T}
    "Print summary at each iteration."
    verbose::Bool=false

    "unconstrained solver options."
    opts_uncon::AbstractSolverOptions{T} = StaticiLQRSolverOptions{T}()

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



struct StaticALSolver{T,S<:AbstractSolver} <: ConstrainedSolver{T}
    opts::StaticALSolverOptions{T}
    stats::ALStats{T}
    stats_uncon::Vector{STATS} where STATS
    solver_uncon::S
end

AbstractSolver(prob::StaticProblem{Q,T},
    opts::StaticALSolverOptions{T}=StaticALSolverOptions{T}()) where {Q,T} =
    StaticALSolver(prob,opts)

"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function StaticALSolver(prob::StaticProblem{Q,T}, opts::StaticALSolverOptions=StaticALSolverOptions{T}()) where {Q,T}
    # Init solver statistics
    stats = ALStats()
    stats_uncon = Vector{StaticiLQRSolverOptions{T}}()

    # Convert problem to AL problem
    alobj = StaticALObjective(prob.obj, prob.constraints)
    rollout!(prob)
    prob_al = StaticProblem(prob.model, alobj, ConstraintSets(size(prob)...),
        prob.x0, prob.xf, prob.Z, prob.N, prob.tf)

    solver_uncon = AbstractSolver(prob_al, opts.opts_uncon)

    solver = StaticALSolver(opts,stats,stats_uncon,solver_uncon)
    reset!(solver)
    return solver
end

function reset!(solver::StaticALSolver)
    reset!(solver.stats, solver.opts.iterations)
    reset!(solver.solver_uncon)
end

Base.size(solver::StaticALSolver) = size(solver.solver_uncon)
@inline cost(solver::StaticALSolver) = cost(solver.solver_uncon)
@inline get_trajectory(solver::StaticALSolver) = get_trajectory(solver.solver_uncon)
@inline get_objective(solver::StaticALSolver) = get_objective(solver.solver_uncon)
@inline get_model(solver::StaticALSolver) = get_model(solver.solver_uncon)
@inline get_initial_state(solver::StaticALSolver) = get_initial_state(solver.solver_uncon)



function get_constraints(solver::StaticALSolver{T}) where T
    obj = get_objective(solver)::StaticALObjective{T}
    obj.constraints
end




struct StaticALObjective{T,O<:Objective} <: AbstractObjective
    obj::O
    constraints::ConstraintSets{T}
end

get_J(obj::StaticALObjective) = obj.obj.J
Base.length(obj::StaticALObjective) = length(obj.obj)

# TrajectoryOptimization.num_constraints(prob::StaticProblem{Q,T,<:StaticALObjective}) where {T,Q} = prob.obj.constraints.p

function Base.copy(obj::StaticALObjective)
    StaticALObjective(obj.obj, ConstraintSets(copy(obj.constraints.constraints), length(obj.obj)))
end

function cost!(obj::StaticALObjective, Z::Traj)
    # Calculate unconstrained cost
    cost!(obj.obj, Z)

    # Calculate constrained cost
    evaluate!(obj.constraints, Z)
    update_active_set!(obj.constraints, Z, Val(0.0))
    for con in obj.constraints.constraints
        cost!(obj.obj.J, con, Z)
    end
end

function cost_expansion(E, obj::StaticALObjective, Z::Traj)
    # Update constraint jacobians
    jacobian!(obj.constraints, Z)

    ix, iu = Z[1]._x, Z[1]._u

    # Calculate expansion of original objective
    cost_expansion(E, obj.obj, Z)

    # Add in expansion of constraints
    for con in obj.constraints.constraints
        cost_expansion(E, con, Z)
    end
end


# StaticALProblem{Q,L,T} = StaticProblem{Q,L,<:StaticALObjective,T}
# function get_constraints(prob::StaticProblem)
#     if prob isa StaticALProblem
#         prob.obj.constraints
#     else
#         prob.constraints
#     end
# end
