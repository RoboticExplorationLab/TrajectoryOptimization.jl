abstract type SolverOptionsNew{T<:Real} end

mutable struct iLQRSolverOptions{T} <: SolverOptionsNew{T}
    # Options
    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve"
    cost_tolerance::T

    "dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve"
    cost_tolerance_intermediate::T

    "gradient type: :todorov, :feedforward"
    gradient_type::Symbol

    "gradient_norm < ϵ, gradient norm convergence criteria"
    gradient_norm_tolerance::T

    "gradient_norm_int < ϵ, gradient norm intermediate convergence criteria"
    gradient_norm_tolerance_intermediate::T

    "iLQR iterations"
    iterations::Int

    "restricts the total number of times a forward pass fails, resulting in regularization, before exiting"
    dJ_counter_limit::Int

    "use square root method backward pass for numerical conditioning"
    square_root::Bool

    "forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound"
    line_search_lower_bound::T

    "forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < ∞"
    line_search_upper_bound::T

    "maximum number of backtracking steps during forward pass line search"
    iterations_linesearch::Int

    # Regularization
    "initial regularization"
    bp_reg_initial::T

    "regularization scaling factor"
    bp_reg_increase_factor::T

    "maximum regularization value"
    bp_reg_max::T

    "minimum regularization value"
    bp_reg_min::T

    "type of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    bp_reg_type::Symbol

    "additive regularization when forward pass reaches max iterations"
    bp_reg_fp::T

    "type of matrix inversion for bp sqrt step"
    bp_sqrt_inv_type::Symbol

    "initial regularization for square root method"
    bp_reg_sqrt_initial::T

    "regularization scaling factor for square root method"
    bp_reg_sqrt_increase_factor::T


    # Solver Numerical Limits
    "maximum cost value, if exceded solve will error"
    max_cost_value::T

    "maximum state value, evaluated during rollout, if exceded solve will error"
    max_state_value::T

    "maximum control value, evaluated during rollout, if exceded solve will error"
    max_control_value::T

    function iLQRSolverOptions(T=Float64;
        cost_tolerance=1.0e-4,
        cost_tolerance_intermediate=1.0e-3,
        gradient_type=:todorov,
        gradient_norm_tolerance=1.0e-5,
        gradient_norm_tolerance_intermediate=1.0e-5,
        iterations=500,
        dJ_counter_limit=10,
        square_root=false,
        line_search_lower_bound=1.0e-8,
        line_search_upper_bound=10.0,
        iterations_linesearch=20,
        bp_reg_initial=0.0,
        bp_reg_increase_factor=1.6,
        bp_reg_max=1.0e8,
        bp_reg_min=1.0e-8,
        bp_reg_type=:control,
        bp_reg_fp=10.0,
        bp_sqrt_inv_type=:pseudo,
        bp_reg_sqrt_initial=1.0e-6,
        bp_reg_sqrt_increase_factor=10.0,
        max_cost_value=1.0e8,
        max_state_value=1.0e8,
        max_control_value=1.0e8
        )

        new{T}(cost_tolerance,
        cost_tolerance_intermediate,
        gradient_type,
        gradient_norm_tolerance,
        gradient_norm_tolerance_intermediate,
        iterations,
        dJ_counter_limit,
        square_root,
        line_search_lower_bound,
        line_search_upper_bound,
        iterations_linesearch,
        bp_reg_initial,
        bp_reg_increase_factor,
        bp_reg_max,
        bp_reg_min,
        bp_reg_type,
        bp_reg_fp,
        bp_sqrt_inv_type,
        bp_reg_sqrt_initial,
        bp_reg_sqrt_increase_factor,
        max_cost_value,
        max_state_value,
        max_control_value)
    end
end

mutable struct ALSolverOptions{T} <: SolverOptionsNew{T}
    "max(constraint) < ϵ, constraint convergence criteria"
    constraint_tolerance::T

    "max(constraint) < ϵ_int, intermediate constraint convergence criteria"
    constraint_tolerance_intermediate::T

    "maximum outerloop updates"
    iterations_outerloop::Int

    "minimum Lagrange multiplier"
    dual_min::T

    "maximum Lagrange multiplier"
    dual_max::T

    "maximum penalty term"
    penalty_max::T

    "initial penalty term"
    penalty_initial::T

    "penalty update multiplier; penalty_scaling > 0"
    penalty_scaling::T

    "penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)"
    penalty_scaling_no::T

    "ratio of current constraint to previous constraint violation; 0 < constraint_decrease_ratio < 1"
    constraint_decrease_ratio::T

    "type of outer loop update (default, momentum, individual, accelerated)"
    outer_loop_update_type::Symbol

    "determines how many iterations should pass before the penalty is updated (1 is every iteration)"
    penalty_update_frequency::Int

    "numerical tolerance for constraint violation"
    active_constraint_tolerance::T

    "perform only penalty updates (no dual updates) until constraint_tolerance_intermediate < ϵ_int"
    use_penalty_burnin::Bool

    function ALSolverOptions(T=Float64;
        constraint_tolerance=1.0e-3,
        constraint_tolerance_intermediate=1.0e-3,
        iterations_outerloop=30,
        dual_min=-1.0e8,
        dual_max=1.0e8,
        penalty_max=1.0e8,
        penalty_initial=1.0,
        penalty_scaling=10.0,
        penalty_scaling_no=1.0,
        constraint_decrease_ratio=0.25,
        outer_loop_update_type=:default,
        penalty_update_frequency=1,
        active_constraint_tolerance=0.0,
        use_penalty_burnin=false)

        new{T}(constraint_tolerance,
        constraint_tolerance_intermediate,
        iterations_outerloop,
        dual_min,
        dual_max,
        penalty_max,
        penalty_initial,
        penalty_scaling,
        penalty_scaling_no,
        constraint_decrease_ratio,
        outer_loop_update_type,
        penalty_update_frequency,
        active_constraint_tolerance,
        use_penalty_burnin)
    end
end

mutable struct ALTROSolverOptions{T} <: SolverOptionsNew{T}
    ## Infeasible Start
    "infeasible control constraint tolerance"
    constraint_tolerance_infeasible::T

    "regularization term for infeasible controls"
    R_infeasible::T

    "resolve feasible problem after infeasible solve"
    resolve_feasible::Bool

    "project infeasible solution into feasible space w/ BP, rollout"
    feasible_projection::Bool

    "initial penalty term for infeasible controls"
    penalty_initial_infeasible::T

    "penalty update rate for infeasible controls"
    penalty_scaling_infeasible::T

    # Minimum Time
    "regularization term for dt"
    R_minimum_time::T

    "maximum allowable dt"
    max_dt::T

    "minimum allowable dt"
    min_dt::T

    "initial guess for the length of the minimum time problem (in seconds)"
    minimum_time_tf_estimate::T

    "initial guess for dt of the minimum time problem (in seconds)"
    minimum_time_dt_estimate::T

    "initial penalty term for minimum time bounds constraints"
    penalty_initial_minimum_time_inequality::T

    "initial penalty term for minimum time equality constraints"
    penalty_initial_minimum_time_equality::T

    "penalty update rate for minimum time bounds constraints"
    penalty_scaling_minimum_time_inequality::T

    "penalty update rate for minimum time equality constraints"
    penalty_scaling_minimum_time_equality::T

    function ALTROSolverOptions(T=Float64;
        constraint_tolerance_infeasible=1.0e-5,
        R_infeasible=1.0,
        resolve_feasible=true,
        feasible_projection=true,
        penalty_initial_infeasible=1.0,
        penalty_scaling_infeasible=10.0,
        R_minimum_time=1.0,
        max_dt=1.0,
        min_dt=1.0e-3,
        minimum_time_tf_estimate=0.0,
        minimum_time_dt_estimate=0.0,
        penalty_initial_minimum_time_inequality=1.0,
        penalty_initial_minimum_time_equality=1.0,
        penalty_scaling_minimum_time_inequality=1.0,
        penalty_scaling_minimum_time_equality=1.0
        )

        new{T}(constraint_tolerance_infeasible,
        R_infeasible,
        resolve_feasible,
        feasible_projection,
        penalty_initial_infeasible,
        penalty_scaling_infeasible,
        R_minimum_time,
        max_dt,
        min_dt,
        minimum_time_tf_estimate,
        minimum_time_dt_estimate,
        penalty_initial_minimum_time_inequality,
        penalty_initial_minimum_time_equality,
        penalty_scaling_minimum_time_inequality,
        penalty_scaling_minimum_time_equality)
    end
end
