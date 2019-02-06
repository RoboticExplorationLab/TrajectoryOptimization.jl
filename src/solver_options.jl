import Base: show, copy

"""
$(TYPEDEF)
    options for Solver
"""
mutable struct SolverOptions
    ###################
    ## Functionality ##
    ###################
    verbose::Bool # Display solve statistics at each iteration
    benchmark::Bool # Run benchmarks on forward and backward passes
    live_plotting::Bool # Plot state and control trajectories during solve

    ##########################
    ## Convergence Criteria ##
    ##########################
    cost_tolerance::Float64 # dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve
    cost_tolerance_intermediate::Float64 # dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve
    cost_tolerance_infeasible::Float64  # Tolerance to kick into a "feasible" solve
    constraint_tolerance::Float64 # max(constraint) < ϵ, constraint convergence criteria
    constraint_tolerance_intermediate::Float64 # max(constraint) < ϵ_int, intermediate constraint convergence criteria
    gradient_norm_tolerance::Float64 # gradient_norm < ϵ, gradient norm convergence criteria
    gradient_norm_tolerance_intermediate::Float64 # gradient_norm_int < ϵ, gradient norm intermediate convergence criteria
    gradient_type::Symbol # type of gradient to evaluate: Augmented Lagrangian ∂L/∂u :AuLa, feedfoward Σ|d|^2 :feedforward, Todorov normalized feedforward :todorov

    ######################
    ## Solve Parameters ##
    ######################
    iterations::Int64 # caps cumulative number of iLQR iterations (see iterations_innerloop, iterations_outerloop)
    dJ_counter_limit::Int64 # restricts the total number of times a forward pass fails, resulting in regularization, before entering an outerloop update

    ###################
    ## Iterative LQR ##
    ###################
    square_root::Bool # use square root method backward pass for numerical conditioning
    iterations_innerloop::Int64 # maximum iterations for inner loop (iLQR)
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
    eigenvalue_scaling::Float64 # add this multiple of the magnitude of the most negative eigenvalue to Quu decomposition to make positive definite
    eigenvalue_threshold::Float64 # eigenvalues less than this threshold will be increased using the additive eigenvalue scaling
    bp_sqrt_inv_type::Symbol # type of fix for potential inverse failure, :pseudo -pseudo inverse of Wxx, :reg - regularize Wxx to make invertible
    bp_reg_sqrt_initial::Float64 # initial regularization for square root method
    bp_reg_sqrt_increase_factor::Float64 # regularization scaling factor for square root method

    ##########################
    ## Augmented Lagrangian ##
    ##########################
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

    ######################
    ## Infeasible Start ##
    ######################
    R_infeasible::Float64 # regularization term for infeasible controls
    resolve_feasible::Bool # resolve feasible problem after infeasible solve
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

    function SolverOptions(;verbose=false,benchmark=false,
        live_plotting=false,
        cost_tolerance=1.0e-4,
        cost_tolerance_intermediate=1.0e-3,
        cost_tolerance_infeasible=1e-3,
        constraint_tolerance=1.0e-3,
        constraint_tolerance_intermediate=sqrt(constraint_tolerance),
        gradient_norm_tolerance=1.0e-5,
        gradient_norm_tolerance_intermediate=1.0e-5,
        gradient_type=:todorov,
        iterations=500,
        dJ_counter_limit=10,
        square_root=false,
        iterations_innerloop=250,
        line_search_lower_bound=1.0e-8,
        line_search_upper_bound=10.0,
        iterations_linesearch=20,
        bp_reg_initial=0.0,
        bp_reg_increase_factor=1.6,
        bp_reg_max=1.0e8,
        bp_reg_min=1.0e-8,
        bp_reg_type=:control,
        bp_reg_fp=10.0,
        eigenvalue_scaling=2.0,
        eigenvalue_threshold=1e-8,
        bp_sqrt_inv_type=:pseudo,
        bp_reg_sqrt_initial=1.0e-6,
        bp_reg_sqrt_increase_factor=10.0,
        iterations_outerloop=30,
        dual_min=-1.0e8,
        dual_max=1.0e8,
        penalty_max=1.0e8,
        penalty_initial=1.0,
        penalty_scaling=10.0,
        penalty_scaling_no=1.0,
        constraint_decrease_ratio=0.25,
        outer_loop_update_type=:default,
        use_second_order_dual_update=false,
        penalty_update_frequency=1,
        constraint_tolerance_second_order_dual_update=sqrt(constraint_tolerance),
        active_constraint_tolerance=0.0,
        use_nesterov=false,
        use_penalty_burnin=false,
        al_type=:default,
        R_infeasible=1.0,
        resolve_feasible=true,
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
        penalty_scaling_minimum_time_equality=1.0,
        max_cost_value=1.0e8,
        max_state_value=1.0e8,
        max_control_value=1.0e8)

        new(verbose,
            benchmark,
            live_plotting,
            cost_tolerance,
            cost_tolerance_intermediate,
            cost_tolerance_infeasible,
            constraint_tolerance,
            constraint_tolerance_intermediate,
            gradient_norm_tolerance,
            gradient_norm_tolerance_intermediate,
            gradient_type,
            iterations,
            dJ_counter_limit,
            square_root,
            iterations_innerloop,
            line_search_lower_bound,
            line_search_upper_bound,
            iterations_linesearch,
            bp_reg_initial,
            bp_reg_increase_factor,
            bp_reg_max,
            bp_reg_min,
            bp_reg_type,
            bp_reg_fp,
            eigenvalue_scaling,
            eigenvalue_threshold,
            bp_sqrt_inv_type,
            bp_reg_sqrt_initial,
            bp_reg_sqrt_increase_factor,
            iterations_outerloop,
            dual_min,
            dual_max,
            penalty_max,
            penalty_initial,
            penalty_scaling,
            penalty_scaling_no,
            constraint_decrease_ratio,
            outer_loop_update_type,
            use_second_order_dual_update,
            penalty_update_frequency,
            constraint_tolerance_second_order_dual_update,
            active_constraint_tolerance,
            use_nesterov,
            use_penalty_burnin,
            al_type,
            R_infeasible,
            resolve_feasible,
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
            penalty_scaling_minimum_time_equality,
            max_cost_value,
            max_state_value,
            max_control_value)
    end

end

copy(opts::SolverOptions) = SolverOptions(;[name=>getfield(opts,name) for name in fieldnames(typeof(opts))]...)

function Base.:(==)(A::SolverOptions, B::SolverOptions)
    for name in fieldnames(A)
        if getfield(A,name) != getfield(B,name)
            return false
        end
    end
    return true
end
