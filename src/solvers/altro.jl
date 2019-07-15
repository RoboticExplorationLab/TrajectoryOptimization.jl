

"""$(TYPEDEF) Solver options for the ALTRO solver.
$(FIELDS)
"""
@with_kw mutable struct ALTROSolverOptions{T} <: AbstractSolverOptions{T}

    verbose::Bool=false

    "Augmented Lagrangian solver options."
    opts_al::AugmentedLagrangianSolverOptions=AugmentedLagrangianSolverOptions{T}()

    # Infeasible Start
    "infeasible control constraint tolerance."
    constraint_tolerance_infeasible::T = 1.0e-5

    "regularization term for infeasible controls."
    R_inf::T = 1.0

    "project infeasible results to feasible space using TVLQR."
    dynamically_feasible_projection::Bool = true

    "resolve feasible problem after infeasible solve."
    resolve_feasible_problem::Bool = true

    "initial penalty term for infeasible controls."
    penalty_initial_infeasible::T = 1.0

    "penalty update rate for infeasible controls."
    penalty_scaling_infeasible::T = 10.0

    # Minimum Time
    "regularization term for dt."
    R_minimum_time::T = 1.0

    "maximum allowable dt."
    dt_max::T = 1.0

    "minimum allowable dt."
    dt_min::T = 1.0e-3

    "initial penalty term for minimum time bounds constraints."
    penalty_initial_minimum_time_inequality::T = 1.0

    "initial penalty term for minimum time equality constraints."
    penalty_initial_minimum_time_equality::T = 1.0

    "penalty update rate for minimum time bounds constraints."
    penalty_scaling_minimum_time_inequality::T = 1.0

    "penalty update rate for minimum time equality constraints."
    penalty_scaling_minimum_time_equality::T = 1.0

    # Projected Newton
    "finish with a projecte newton solve."
    projected_newton::Bool = false

    "options for projected newton solver."
    opts_pn::ProjectedNewtonSolverOptions{T} = ProjectedNewtonSolverOptions{T}()

    "constraint satisfaction tolerance that triggers the projected newton solver.
    If set to a non-positive number it will kick out when the maximum penalty is reached."
    projected_newton_tolerance::T = 1.0e-3

end


"""$(TYPEDEF)
Augmented Lagrangian Trajectory Optimizer (ALTRO) is a solver developed by the Robotic Exploration Lab at Stanford University.
    The solver is special-cased to solve Markov Decision Processes by leveraging the internal problem structure.

ALTRO consists of two "phases":
1) AL-iLQR: iLQR is used with an Augmented Lagrangian framework to solve the problem quickly to rough constraint satisfaction
2) Projected Newton: A collocation-flavored active-set solver projects the solution from AL-iLQR onto the feasible subspace to achieve machine-precision constraint satisfaction.
"""
struct ALTROSolver{T} <: AbstractSolver{T}
    opts::ALTROSolverOptions{T}
    stats::Dict{Symbol,Any}
    solver_al::AugmentedLagrangianSolver{T}
    solver_pn::ProjectedNewtonSolver{T}
end

ALTROSolver(prob::Problem, opts::ALTROSolverOptions) = AbstractSolver(prob, opts)

function AbstractSolver(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    stats = Dict{Symbol,Any}()
    solver_al = AugmentedLagrangianSolver(prob, opts.opts_al)
    solver_pn = ProjectedNewtonSolver(prob, opts.opts_pn)
    ALTROSolver{T}(opts,stats,solver_al,solver_pn)
end
