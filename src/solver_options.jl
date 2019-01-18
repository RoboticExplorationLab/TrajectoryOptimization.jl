import Base: show, copy

"""
$(TYPEDEF)
    Specifies options for Solver.
"""
mutable struct SolverOptions
    "Use cholesky decomposition of the cost-to-go Hessian"
    square_root::Bool
    "Display statistics at each iteration"
    verbose::Bool

    "lower bound for forward pass line search, 0 < z_min < z_max"
    z_min::Float64
    "upper bound for forward pass line search, 0 < z_min < z_max < Inf"
    z_max::Float64

    "max cost value"
    max_cost_value::Float64
    "max state value"
    max_state_value::Float64
    "max control value"
    max_control_value::Float64
    "maximum allowable dt"
    max_dt::Float64
    "minimum allowable dt"
    min_dt::Float64
    "initial guess for the length of the minimum time problem (in seconds)"
    minimum_time_tf_estimate::Float64
    "initial guess for dt of the minimum time problem (in seconds)"
    minimum_time_dt_estimate::Float64

    "gradient exit criteria"
    gradient_tolerance::Float64

    "gradient Intermediate exit criteria"
    gradient_tolerance_intermediate::Float64

    "final cost convergence criteria"
    cost_tolerance::Float64
    "intermediate cost convergence criteria for outerloop of constrained solve"
    cost_tolerance_intermediate::Float64
    "maximum constraint violation termination criteria"
    constraint_tolerance::Float64
    constraint_tolerance_coarse::Float64
    "iterations (total)"
    iterations::Int64
    "iterations inner loop (iLQR loops)"
    iterations_innerloop::Int64
    "iterations for outer loop of constraint solve"
    iterations_outerloop::Int64
    "maximum number of backtracking steps during forward pass line search"
    iterations_linesearch::Int64

    "regularization term for augmented controls during infeasible start"
    R_infeasible::Float64
    "regularization term in R matrix for dt"
    R_minimum_time::Float64

    "Run benchmarks on forward and backward passes"
    benchmark::Bool

    "Pass infeasible trajectory solution to original problem"
    unconstrained_original_problem::Bool
    resolve_feasible::Bool # resolve feasible problem post infeasible solve

    "Augmented Lagrangian Method parameters" # terms defined in Practical Augmented Lagrangian Methods for Constrained Optimization
    dual_min::Float64 # minimum Lagrange multiplier
    dual_max::Float64 # maximum Lagrange multiplier
    penalty_max::Float64 # maximum penalty term
    penalty_initial::Float64 # initial penalty term
    penalty_initial_infeasible::Float64 # initial penalty term for infeasible controls
    penalty_initial_minimum_time_inequality::Float64 # initial penalty term for minimum time bounds constraints
    penalty_initial_minimum_time_equality::Float64 # initial penalty term for minimum time equality constraints
    penalty_scaling::Float64 # penalty update multiplier; penalty_scaling > 0
    penalty_scaling_infeasible::Float64 # penalty update rate for infeasible controls
    penalty_scaling_minimum_time_inequality::Float64 # penalty update rate for minimum time bounds constraints
    penalty_scaling_minimum_time_equality::Float64 # penalty update rate for minimum time equality constraints
    penalty_scaling_no::Float64 # penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)
    constraint_decrease_ratio::Float64 # update term; 0 < constraint_decrease_ratio < 1
    outer_loop_update_type::Symbol # type of outer loop update (default, momentum, individual, accelerated)
    use_second_order_dual_update::Bool # second order update for Lagrange multipliers once sqrt(cost tolerance | gradient) < desired tolerance
    penalty_update_frequency::Int  # determines how many iterations should pass before the penalty is updated (1 is every iteration)
    constraint_tolerance_second_order_dual_update::Float64 # constraint tolerance for switching to second order dual update
    use_nesterov::Bool
    use_penalty_burnin::Bool

    "Regularization parameters"
    bp_reg_initial::Float64 # initial regularization
    bp_reg_increase_factor::Float64 # scaling factor
    bp_reg_max::Float64 # maximum regularization value
    bp_reg_min::Float64 # minimum regularization value
    bp_reg_type::Symbol # type of regularization- control: () + ρI, state: (S + ρI); see "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    bp_reg_fp::Float64 # additive regularization when forward pass reaches max iterations

    eigenvalue_scaling::Float64 # add this multiple of the magnitude of the most negative eigenvalue to Quu decomposition to make positive definite
    eigenvalue_threshold::Float64 # eigenvalues less than this threshold will be increased using the additive eigenvalue scaling
    use_static::Bool

    live_plotting::Bool

    "Active set criteria"
    active_constraint_tolerance::Float64

    dJ_counter_limit::Int64

    gradient_type::Symbol # type of gradient to evaluate: Augmented Lagrangian ∂L/∂u :AuLa, feedfoward Σ|d|^2 :feedforward, Todorov normalized feedforward :todorov

    function SolverOptions(;square_root=false,verbose=false,
        z_min=1.0e-8,z_max=10.0,max_cost_value=1.0e8,max_state_value=1.0e8,max_control_value=1.0e8,max_dt=1.0,min_dt=1e-3,minimum_time_tf_estimate=0.0,minimum_time_dt_estimate=0.0,gradient_tolerance=1e-5,gradient_tolerance_intermediate=1e-5,cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-3,
        constraint_tolerance=1e-3, constraint_tolerance_coarse=sqrt(constraint_tolerance),
        iterations=500,iterations_innerloop=150,iterations_outerloop=50,
        iterations_linesearch=20,R_infeasible=1e3,R_minimum_time=1.0e3,
        benchmark=false,unconstrained_original_problem=false,resolve_feasible=true,dual_min=-1.0e8, dual_max=1.0e8,penalty_max=1.0e8,penalty_initial=1.0,penalty_initial_infeasible=1.0,penalty_initial_minimum_time_inequality=1.0,penalty_initial_minimum_time_equality=1.0,penalty_scaling=10.0,penalty_scaling_infeasible=10.0,penalty_scaling_minimum_time_inequality=10.0,penalty_scaling_minimum_time_equality=10.0,penalty_scaling_no=1.0,constraint_decrease_ratio=0.25,outer_loop_update_type=:default,use_second_order_dual_update=false,
        penalty_update_frequency=1,constraint_tolerance_second_order_dual_update=sqrt(constraint_tolerance), use_nesterov=false, use_penalty_burnin=false,
        bp_reg_initial=0.0,bp_reg_increase_factor=1.6,bp_reg_max=1.0e8,bp_reg_min=1e-6,bp_reg_type=:state,bp_reg_fp=10.0,eigenvalue_scaling=2.0,eigenvalue_threshold=1e-8,use_static=false,live_plotting=false,active_constraint_tolerance=1e-8,dJ_counter_limit=10,gradient_type=:todorov)

        new(square_root,verbose,z_min,z_max,max_cost_value,max_state_value,max_control_value,max_dt,min_dt,minimum_time_tf_estimate,minimum_time_dt_estimate,gradient_tolerance,gradient_tolerance_intermediate,cost_tolerance,cost_tolerance_intermediate,
        constraint_tolerance,constraint_tolerance_coarse,
        iterations,iterations_innerloop,iterations_outerloop,
        iterations_linesearch,R_infeasible,R_minimum_time,
        benchmark,unconstrained_original_problem,resolve_feasible,
        dual_min,dual_max,penalty_max,penalty_initial,penalty_initial_infeasible,penalty_initial_minimum_time_inequality,penalty_initial_minimum_time_equality,penalty_scaling,penalty_scaling_infeasible,penalty_scaling_minimum_time_inequality,penalty_scaling_minimum_time_equality,penalty_scaling_no,constraint_decrease_ratio,outer_loop_update_type,use_second_order_dual_update,
        penalty_update_frequency,constraint_tolerance_second_order_dual_update,use_nesterov,use_penalty_burnin,
        bp_reg_initial,bp_reg_increase_factor,bp_reg_max,bp_reg_min,bp_reg_type,bp_reg_fp,eigenvalue_scaling,eigenvalue_threshold,
        use_static,live_plotting,active_constraint_tolerance,dJ_counter_limit,gradient_type)
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
