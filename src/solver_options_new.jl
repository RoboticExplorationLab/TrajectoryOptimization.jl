using Parameters

abstract type AbstractSolverOptions{T<:Real} end

"""$(TYPEDEF)
Solver options for the iterative LQR (iLQR) solver. iLQR is an indirect, unconstrained solver.
$(FIELDS)
"""
@with_kw mutable struct iLQRSolverOptions{T} <: AbstractSolverOptions{T}
    # Options
    "Print summary at each iteration"
    verbose::Bool=false

    "Live plotting"
    live_plotting::Symbol=:off # :state, :control

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve"
    cost_tolerance::T = 1.0e-4

    "gradient type: :todorov, :feedforward"
    gradient_type::Symbol = :todorov

    "gradient_norm < ϵ, gradient norm convergence criteria"
    gradient_norm_tolerance::T = 1.0e-5

    "iLQR iterations"
    iterations::Int = 500

    "restricts the total number of times a forward pass fails, resulting in regularization, before exiting"
    dJ_counter_limit::Int = 10

    "use square root method backward pass for numerical conditioning"
    square_root::Bool = false

    "forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound"
    line_search_lower_bound::T = 1.0e-8

    "forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < ∞"
    line_search_upper_bound::T = 10.0

    "maximum number of backtracking steps during forward pass line search"
    iterations_linesearch::Int = 20

    # Regularization
    "initial regularization"
    bp_reg_initial::T = 0.0

    "regularization scaling factor"
    bp_reg_increase_factor::T = 1.6

    "maximum regularization value"
    bp_reg_max::T = 1.0e8

    "minimum regularization value"
    bp_reg_min::T = 1.0e-8

    "type of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    bp_reg_type::Symbol = :control

    "additive regularization when forward pass reaches max iterations"
    bp_reg_fp::T = 10.0

    # square root backward pass options:
    "type of matrix inversion for bp sqrt step"
    bp_sqrt_inv_type::Symbol = :pseudo

    "initial regularization for square root method"
    bp_reg_sqrt_initial::T = 1.0e-6

    "regularization scaling factor for square root method"
    bp_reg_sqrt_increase_factor::T = 10.0

    # Solver Numerical Limits
    "maximum cost value, if exceded solve will error"
    max_cost_value::T = 1.0e8

    "maximum state value, evaluated during rollout, if exceded solve will error"
    max_state_value::T = 1.0e8

    "maximum control value, evaluated during rollout, if exceded solve will error"
    max_control_value::T = 1.0e8
end

#TODO
function copy(r::iLQRSolverOptions)
    nothing
end

"""$(TYPEDEF)
Solver options for the augmented Lagrangian solver.
    Augmented Lagrangian is a general method for solving constrained problems by solving a sequence of unconstrained problems.
$(FIELDS)
"""
@with_kw mutable struct AugmentedLagrangianSolverOptions{T} <: AbstractSolverOptions{T}
    "Print summary at each iteration"
    verbose::Bool=false

    "unconstrained solver options"
    opts_uncon::AbstractSolverOptions{T} = iLQRSolverOptions{T}()

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve"
    cost_tolerance::T = 1.0e-4

    "dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve"
    cost_tolerance_intermediate::T = 1.0e-3

    "gradient_norm < ϵ, gradient norm convergence criteria"
    gradient_norm_tolerance::T = 1.0e-5

    "gradient_norm_int < ϵ, gradient norm intermediate convergence criteria"
    gradient_norm_tolerance_intermediate::T = 1.0e-5

    "max(constraint) < ϵ, constraint convergence criteria"
    constraint_tolerance::T = 1.0e-3

    "max(constraint) < ϵ_int, intermediate constraint convergence criteria"
    constraint_tolerance_intermediate::T = 1.0e-3

    "maximum outerloop updates"
    iterations::Int = 30

    "minimum Lagrange multiplier"
    dual_min::T = -1.0e8

    "maximum Lagrange multiplier"
    dual_max::T = 1.0e8

    "maximum penalty term"
    penalty_max::T = 1.0e8

    "initial penalty term"
    penalty_initial::T = 1.0

    "penalty update multiplier; penalty_scaling > 0"
    penalty_scaling::T = 10.0

    "penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)"
    penalty_scaling_no::T = 1.0

    "ratio of current constraint to previous constraint violation; 0 < constraint_decrease_ratio < 1"
    constraint_decrease_ratio::T = 0.25

    "type of outer loop update (default, momentum, individual, accelerated)"
    outer_loop_update_type::Symbol = :default

    "determines how many iterations should pass before the penalty is updated (1 is every iteration)"
    penalty_update_frequency::Int = 1

    "numerical tolerance for constraint violation"
    active_constraint_tolerance::T = 0.0

    "perform only penalty updates (no dual updates) until constraint_tolerance_intermediate < ϵ_int"
    use_penalty_burnin::Bool = false

end

function check_convergence_criteria(opts_uncon::AbstractSolverOptions{T},cost_tolerance::T,gradient_norm_tolerance::T) where T
    if opts_uncon.cost_tolerance != cost_tolerance
        @warn "Augmented Lagrangian cost tolerance overriding unconstrained solver option\n >>cost tolerance=$cost_tolerance"
    end

    if opts_uncon.gradient_norm_tolerance != gradient_norm_tolerance
        @warn "Augmented Lagrangian gradient norm tolerance overriding unconstrained solver option\n >>gradient norm tolerance=$gradient_norm_tolerance"
    end
    return nothing
end


@with_kw mutable struct ALTROSolverOptions{T} <: AbstractSolverOptions{T}

    verbose::Bool=false

    "Augmented Lagrangian solver options"
    opts_al::AugmentedLagrangianSolverOptions=AugmentedLagrangianSolverOptions{T}()

    # Infeasible Start
    "infeasible control constraint tolerance"
    constraint_tolerance_infeasible::T = 1.0e-5

    "regularization term for infeasible controls"
    R_inf::T = 1.0

    "resolve feasible problem after infeasible solve"
    resolve_feasible::Bool = true

    "project infeasible solution into feasible space w/ BP, rollout"
    feasible_projection::Bool = true

    "initial penalty term for infeasible controls"
    penalty_initial_infeasible::T = 1.0

    "penalty update rate for infeasible controls"
    penalty_scaling_infeasible::T = 10.0

    # Minimum Time
    "regularization term for dt"
    R_minimum_time::T = 1.0

    "maximum allowable dt"
    dt_max::T = 1.0

    "minimum allowable dt"
    dt_min::T = 1.0e-3

    "initial penalty term for minimum time bounds constraints"
    penalty_initial_minimum_time_inequality::T = 1.0

    "initial penalty term for minimum time equality constraints"
    penalty_initial_minimum_time_equality::T = 1.0

    "penalty update rate for minimum time bounds constraints"
    penalty_scaling_minimum_time_inequality::T = 1.0

    "penalty update rate for minimum time equality constraints"
    penalty_scaling_minimum_time_equality::T = 1.0
end
