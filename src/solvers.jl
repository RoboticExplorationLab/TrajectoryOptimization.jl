abstract type Solver end

struct iLQRSolver <: Solver
    # Options
    cost_tolerance::Float64 # dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve
    cost_tolerance_intermediate::Float64 # dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve
    gradient_norm_tolerance::Float64 # gradient_norm < ϵ, gradient norm convergence criteria
    gradient_norm_tolerance_intermediate::Float64 # gradient_norm_int < ϵ, gradient norm intermediate convergence criteria

    iterations::Int64 # caps cumulative number of iLQR iterations (see iterations_innerloop, iterations_outerloop)
    dJ_counter_limit::Int64 # restricts the total number of times a forward pass fails, resulting in regularization, before entering an outerloop update

    square_root::Bool # use square root method backward pass for numerical conditioning
    line_search_lower_bound::Float64 # forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound
    line_search_upper_bound::Float64 # forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < Inf
    iterations_linesearch::Int64 # maximum number of backtracking steps during forward pass line search

    # Regularization
    bp_reg_initial::Float64 # initial regularization
    bp_reg_increase_factor::Float64 # regularization scaling factor
    bp_reg_max::Float64 # maximum regularization value
    bp_reg_min::Float64 # minimum regularization value
    bp_reg_type::Symbol # type of regularization- control: () + ρI, state: (S + ρI); see "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    bp_reg_fp::Float64 # additive regularization when forward pass reaches max iterations

    bp_sqrt_inv_type::Symbol # type of fix for potential inverse failure, :pseudo -pseudo inverse of Wxx, :reg - regularize Wxx to make invertible
    bp_reg_sqrt_initial::Float64 # initial regularization for square root method
    bp_reg_sqrt_increase_factor::Float64 # regularization scaling factor for square root method

end

struct ALSolver <: Solver
    constraint_tolerance::Float64 # max(constraint) < ϵ, constraint convergence criteria
    constraint_tolerance_intermediate::Float64 # max(constraint) < ϵ_int, intermediate constraint convergence criteria

    iterations_outerloop::Int64 # maximum outerloop updates
    dual_min::Float64 # minimum Lagrange multiplier
    dual_max::Float64 # maximum Lagrange multiplier
    penalty_max::Float64 # maximum penalty term
    penalty_initial::Float64 # initial penalty term
    penalty_scaling::Float64 # penalty update multiplier; penalty_scaling > 0
    penalty_scaling_no::Float64 # penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)
    constraint_decrease_ratio::Float64 # update term; 0 < constraint_decrease_ratio < 1
    outer_loop_update_type::Symbol # type of outer loop update (default, momentum, individual, accelerated)
    use_second_order_dual_update::Bool # second order update for Lagrange multipliers once sqrt(cost tolerance | gradient) < desired tolerance
    penalty_update_frequency::Int  # determines how many iterations should pass before the penalty is updated (1 is every iteration)
    constraint_tolerance_second_order_dual_update::Float64 # constraint tolerance for switching to second order dual update
    active_constraint_tolerance::Float64 # numerical tolerance for constraint violation
    use_nesterov::Bool # accelerated gradient descent for dual variables
    use_penalty_burnin::Bool # perform only penalty updates (no dual updates) until constraint_tolerance_intermediate < ϵ_int
    al_type::Symbol  # [:default, :algencan] pick which Augmented Lagrangian formulation to use. ALGENCAN uses a slightly different check for inactive constraints.

end

struct ALTROSolver <: Solver
    ######################
    ## Infeasible Start ##
    ######################
    constraint_tolerance_infeasible::Float64  # infeasible control constraint tolerance
    R_infeasible::Float64 # regularization term for infeasible controls
    resolve_feasible::Bool # resolve feasible problem after infeasible solve
    feasible_projection::Bool # project infeasible solution into feasible space w/ BP, rollout
    penalty_initial_infeasible::Float64 # initial penalty term for infeasible controls
    penalty_scaling_infeasible::Float64 # penalty update rate for infeasible controls

    ##################
    ## Minimum Time ##
    ##################
    R_minimum_time::Float64 # regularization term for dt
    max_dt::Float64 # maximum allowable dt
    min_dt::Float64 # minimum allowable dt
    minimum_time_tf_estimate::Float64 # initial guess for the length of the minimum time problem (in seconds)
    minimum_time_dt_estimate::Float64 # initial guess for dt of the minimum time problem (in seconds)
    penalty_initial_minimum_time_inequality::Float64 # initial penalty term for minimum time bounds constraints
    penalty_initial_minimum_time_equality::Float64 # initial penalty term for minimum time equality constraints
    penalty_scaling_minimum_time_inequality::Float64 # penalty update rate for minimum time bounds constraints
    penalty_scaling_minimum_time_equality::Float64 # penalty update rate for minimum time equality constraints

    #############################
    ## Solver Numerical Limits ##
    #############################
    max_cost_value::Float64 # maximum cost value, if exceded solve will error
    max_state_value::Float64 # maximum state value, evaluated during rollout, if exceded solve will error
    max_control_value::Float64 # maximum control value, evaluated during rollout, if exceded solve will error

end
