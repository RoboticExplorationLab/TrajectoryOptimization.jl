import Base: show, copy

"""
$(TYPEDEF)
Specifies options for Solver.
"""
mutable struct SolverOptions
    constrained::Bool
    minimum_time::Bool
    infeasible::Bool

    "Use cholesky decomposition of the cost-to-go Hessian"
    square_root::Bool
    "Display statistics at each iteration"
    verbose::Bool

    "lower bound for forward pass line search, 0 < z_min < z_max"
    z_min::Float64
    "upper bound for forward pass line search, 0 < z_min < z_max < Inf"
    z_max::Float64

    "max cost value"
    max_cost::Float64
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
    gradient_intermediate_tolerance::Float64

    "final cost convergence criteria"
    cost_tolerance::Float64
    "intermediate cost convergence criteria for outerloop of constrained solve"
    cost_intermediate_tolerance::Float64
    "maximum constraint violation termination criteria"
    constraint_tolerance::Float64
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
    λ_min::Float64 # minimum Lagrange multiplier
    λ_max::Float64 # maximum Lagrange multiplier
    μ_max::Float64 # maximum penalty term
    μ_initial::Float64 # initial penalty term
    μ_initial_infeasible::Float64 # initial penalty term for infeasible controls
    μ_initial_minimum_time_inequality::Float64 # initial penalty term for minimum time bounds constraints
    μ_initial_minimum_time_equality::Float64 # initial penalty term for minimum time equality constraints
    γ::Float64 # penalty update multiplier; γ > 0
    γ_infeasible::Float64 # penalty update rate for infeasible controls
    γ_minimum_time_inequality::Float64 # penalty update rate for minimum time bounds constraints
    γ_minimum_time_equality::Float64 # penalty update rate for minimum time equality constraints
    γ_no::Float64 # penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)
    τ::Float64 # update term; 0 < τ < 1
    outer_loop_update::Symbol # type of outer loop update (default, individual)
    λ_second_order_update::Bool # second order update for Lagrange multipliers once sqrt(cost tolerance | gradient) < desired tolerance

    "Regularization parameters"
    ρ_initial::Float64 # initial regularization
    ρ_factor::Float64 # scaling factor
    ρ_max::Float64 # maximum regularization value
    ρ_min::Float64 # minimum regularization value
    regularization_type::Symbol # type of regularization- control: () + ρI, state: (S + ρI); see "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    ρ_forwardpass::Float64 # multiplicative regularization scaling when forward pass reaches max iterations
    eigenvalue_scaling::Float64 # add this multiple of the magnitude of the most negative eigenvalue to Quu decomposition to make positive definite
    eigenvalue_threshold::Float64 # eigenvalues less than this threshold will be increased using the additive eigenvalue scaling
    use_static::Bool

    live_plotting::Bool

    "Active Set Method parameters"
    active_set::Bool
    active_set_flag::Bool
    active_set_tolerance::Float64
    active_constraint_tolerance::Float64

    dJ_counter_limit::Int64


    function SolverOptions(;constrained=false,minimum_time=false,infeasible=false,square_root=false,verbose=false,
        z_min=1.0e-8,z_max=10.0,max_cost=1.0e8,max_state_value=1.0e8,max_control_value=1.0e8,max_dt=1.0,min_dt=1e-3,minimum_time_tf_estimate=0.0,minimum_time_dt_estimate=0.0,gradient_tolerance=1e-5,gradient_intermediate_tolerance=1e-5,cost_tolerance=1.0e-4,cost_intermediate_tolerance=1.0e-3,
        constraint_tolerance=1e-3,iterations=500,iterations_innerloop=150,iterations_outerloop=50,
        iterations_linesearch=15,R_infeasible=1e3,R_minimum_time=1.0e3,
        benchmark=false,unconstrained_original_problem=false,resolve_feasible=true,λ_min=-1.0e8,λ_max=1.0e8,μ_max=1.0e8,μ_initial=1.0,μ_initial_infeasible=1.0,μ_initial_minimum_time_inequality=1.0,μ_initial_minimum_time_equality=1.0,γ=10.0,γ_infeasible=10.0,γ_minimum_time_inequality=10.0,γ_minimum_time_equality=10.0,γ_no=1.0,τ=0.25,outer_loop_update=:default,λ_second_order_update=false,
        ρ_initial=0.0,ρ_factor=1.6,ρ_max=1.0e8,ρ_min=1e-6,regularization_type=:state,ρ_forwardpass=10.0,eigenvalue_scaling=2.0,eigenvalue_threshold=1e-8,use_static=false,live_plotting=false,active_set=false,active_set_flag=false,active_set_tolerance=1e-8,active_constraint_tolerance=1e-8,dJ_counter_limit=10)

        new(constrained,minimum_time,infeasible,square_root,verbose,z_min,z_max,max_cost,max_state_value,max_control_value,max_dt,min_dt,minimum_time_tf_estimate,minimum_time_dt_estimate,gradient_tolerance,gradient_intermediate_tolerance,cost_tolerance,cost_intermediate_tolerance,
        constraint_tolerance,iterations,iterations_innerloop,iterations_outerloop,
        iterations_linesearch,R_infeasible,R_minimum_time,
        benchmark,unconstrained_original_problem,resolve_feasible,
        λ_min,λ_max,μ_max,μ_initial,μ_initial_infeasible,μ_initial_minimum_time_inequality,μ_initial_minimum_time_equality,γ,γ_infeasible,γ_minimum_time_inequality,γ_minimum_time_equality,γ_no,τ,outer_loop_update,λ_second_order_update,ρ_initial,ρ_factor,ρ_max,ρ_min,regularization_type,ρ_forwardpass,eigenvalue_scaling,eigenvalue_threshold,
        use_static,live_plotting,active_set,active_set_flag,active_set_tolerance,active_constraint_tolerance,dJ_counter_limit)
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
