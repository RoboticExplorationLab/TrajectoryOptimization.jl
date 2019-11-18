
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
    stats.cost = zeros(L)
    stats.c_max = zeros(L)
    stats.penalty_max = zeros(L)
end




struct StaticALSolver{T,S<:AbstractSolver} <: AbstractSolver{T}
    opts::AugmentedLagrangianSolverOptions{T}
    stats::ALStats{T}
    stats_uncon::Vector{STATS} where STATS
    solver_uncon::S
end

StaticALSolver(prob::StaticProblem{L,T,N,M,NM},
    opts::AugmentedLagrangianSolverOptions{T}=AugmentedLagrangianSolverOptions{T}()) where {L,T,N,M,NM} =
    AbstractSolver(prob,opts)

"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AbstractSolver(prob::StaticProblem{L,T}, opts::AugmentedLagrangianSolverOptions{T}) where {T<:AbstractFloat,L<:AbstractModel}
    # Init solver statistics
    stats = ALStats{T}()
    stats_uncon = Vector{StaticiLQRSolverOptions{T}}()

    solver_uncon = AbstractSolver(prob, opts.opts_uncon)
    solver = StaticALSolver(opts,stats,stats_uncon,solver_uncon)
    reset!(solver)
    return solver
end

function reset!(solver::StaticALSolver)
    reset!(solver.stats, solver.opts.iterations)
    reset!(solver.solver_uncon)
end


function convertProblem(prob::StaticProblem, solver::StaticALSolver)
    alobj = StaticALObjective(prob.obj, prob.constraints)
    rollout!(prob)
    StaticProblem(prob.model, alobj, ConstraintSets(prob.N),
        prob.x0, prob.xf, deepcopy(prob.Z), deepcopy(prob.Z̄), prob.N, prob.dt, prob.tf)
end




struct StaticALObjective{T} <: AbstractObjective
    obj::Objective
    constraints::ConstraintSets{T}
end

get_J(obj::StaticALObjective) = obj.obj.J

TrajectoryOptimization.num_constraints(prob::StaticProblem{L,T,<:StaticALObjective}) where {L,T} = prob.obj.constraints.p

function cost!(obj::StaticALObjective, Z::Traj)
    # Calculate unconstrained cost
    cost!(obj.obj, Z)

    # Calculate constrained cost
    evaluate(obj.constraints, Z)
    update_active_set!(obj.constraints, Z, Val(0.0))
    for con in obj.constraints.constraints
        cost!(obj.obj.J, con, Z)
    end
end

function cost_expansion(E, obj::StaticALObjective, Z::Traj)
    # Update constraint jacobians
    jacobian(obj.constraints, Z)

    ix, iu = Z[1]._x, Z[1]._u

    # Calculate expansion of original objective
    cost_expansion(E, obj.obj, Z)

    # Add in expansion of constraints
    for con in obj.constraints.constraints
        cost_expansion(E, con, Z)
    end
end

StaticALProblem{L,T,N,M,NM} = StaticProblem{L,T,<:StaticALObjective,N,M,NM}
function get_constraints(prob::StaticProblem)
    if prob isa StaticALProblem
        prob.obj.constraints
    else
        prob.constraints
    end
end
