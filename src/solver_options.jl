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

    "max state value"
    max_state_value::Float64
    "max control value"
    max_control_value::Float64
    "maximum allowable dt"
    max_dt::Float64
    "minimum allowable dt"
    min_dt::Float64
    "initial guess for the length of the minimum time problem (in seconds)"
    min_time_init::Float64

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
    "iterations for iLQR solve"
    iterations::Int64
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
    use_static::Bool

    live_plotting::Bool

    function SolverOptions(;constrained=false,minimum_time=false,infeasible=false,square_root=false,verbose=false,
        z_min=1.0e-8,z_max=10.0,max_state_value=1.0e16,max_control_value=1.0e16,max_dt=1.0,min_dt=1e-4,min_time_init=0,gradient_tolerance=1e-5,gradient_intermediate_tolerance=1e-5,cost_tolerance=1.0e-4,cost_intermediate_tolerance=1.0e-4,
        constraint_tolerance=1e-3,iterations=250,iterations_outerloop=50,
        iterations_linesearch=15,R_infeasible=1e3,R_minimum_time=1.0,
        benchmark=false,unconstrained_original_problem=false,resolve_feasible=true,λ_min=-1.0e64,λ_max=1.0e64,μ_max=1.0e64,μ_initial=1.0,μ_initial_infeasible=1.0,μ_initial_minimum_time_inequality=1.0,μ_initial_minimum_time_equality=1.0,γ=10.0,γ_infeasible=10.0,γ_minimum_time_inequality=10.0,γ_minimum_time_equality=10.0,γ_no=1.0,τ=0.25,outer_loop_update=:default,λ_second_order_update=false,
        ρ_initial=0.0,ρ_factor=1.6,ρ_max=1.0e64,ρ_min=1e-6,regularization_type=:control,ρ_forwardpass=1.0,use_static=true,live_plotting=false)

        new(constrained,minimum_time,infeasible,square_root,verbose,z_min,z_max,max_state_value,max_control_value,max_dt,min_dt,min_time_init,gradient_tolerance,gradient_intermediate_tolerance,cost_tolerance,cost_intermediate_tolerance,
        constraint_tolerance,iterations,iterations_outerloop,
        iterations_linesearch,R_infeasible,R_minimum_time,
        benchmark,unconstrained_original_problem,resolve_feasible,
        λ_min,λ_max,μ_max,μ_initial,μ_initial_infeasible,μ_initial_minimum_time_inequality,μ_initial_minimum_time_equality,γ,γ_infeasible,γ_minimum_time_inequality,γ_minimum_time_equality,γ_no,τ,outer_loop_update,λ_second_order_update,ρ_initial,ρ_factor,ρ_max,ρ_min,regularization_type,ρ_forwardpass,
        use_static,live_plotting)
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

# function show(io::IO, opts::SolverOptions)
#     println(io, "SolverOptions:")
#     print(io,"  Use Square Root: $(opts.square_root)")
# end
