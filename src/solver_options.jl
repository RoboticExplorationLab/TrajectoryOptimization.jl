import Base: show, copy

"""
$(TYPEDEF)
Specifies options for Solver.
"""
mutable struct SolverOptions
    "Use cholesky decomposition of S, the Hessian of the cost-to-go"
    square_root::Bool
    "Display statistics at each iteration TODO: make this a symbol specifying level of output"
    verbose::Bool

    "lower bound for forward pass line search, 0 < c1 < c2"
    c1::Float64
    "upper bound for forward pass line search, 0 < c1 < c2 < Inf"
    c2::Float64

    "max state value"
    max_state_value::Float64
    "max control value"
    max_control_value::Float64

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
    infeasible_regularization::Float64
    "cache all intermediate state and control trajectories"
    cache::Bool

    "Run benchmarks on forward and backward passes"
    benchmark::Bool

    "Pass infeasible trajectory solution to original problem"
    solve_feasible::Bool
    infeasible::Bool
    unconstrained::Bool
    resolve_feasible::Bool # resolve feasible problem post infeasible solve

    "Augmented Lagrangian Method parameters" # terms defined in Practical Augmented Lagrangian Methods for Constrained Optimization
    λ_min::Float64 # minimum Lagrange multiplier
    λ_max::Float64 # maximum Lagrange multiplier
    μ_max::Float64 # maximum penalty term
    μ1::Float64 # initial penalty term
    γ::Float64 # penalty update multiplier; γ > 0
    γ_no::Float64 # penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)
    τ::Float64 # update term; 0 < τ < 1
    outer_loop_update::Symbol # type of outer loop update (default, uniform, uniform_time_step, individual)
    λ_second_order_update::Bool # second order update for Lagrange multipliers once sqrt(cost tolerance | gradient) < desired tolerance

    "Regularization parameters"
    ρ_initial::Float64 # initial regularization
    ρ_factor::Float64 # scaling factor
    ρ_max::Float64 # maximum regularization value
    ρ_min::Float64 # minimum regularization value

    use_static::Bool

    function SolverOptions(;square_root=false,verbose=false,
        c1=1.0e-8,c2=2.0,max_state_value=1.0e16,max_control_value=1.0e16,gradient_tolerance=1e-5,gradient_intermediate_tolerance=1e-5,cost_tolerance=1.0e-5,cost_intermediate_tolerance=1.0e-2,
        constraint_tolerance=1e-3,iterations=1000,iterations_outerloop=50,
        iterations_linesearch=10,infeasible_regularization=1e6,cache=false,
        benchmark=false,solve_feasible=true,infeasible=false,unconstrained=false,resolve_feasible=true,λ_min=-1.0e16,λ_max=1.0e16,μ_max=1.0e16,μ1=1.0,γ=10.0,γ_no=1.0,τ=0.25,outer_loop_update=:default,λ_second_order_update=false,
        ρ_initial=1.0,ρ_factor=1.6,ρ_max=1.0e10,ρ_min=1e-6,use_static=true)

        new(square_root,verbose,c1,c2,max_state_value,max_control_value,gradient_tolerance,gradient_intermediate_tolerance,cost_tolerance,cost_intermediate_tolerance,
        constraint_tolerance,iterations,iterations_outerloop,
        iterations_linesearch,infeasible_regularization,cache,
        benchmark,solve_feasible,infeasible,unconstrained,resolve_feasible,
        λ_min,λ_max,μ_max,μ1,γ,γ_no,τ,outer_loop_update,λ_second_order_update,ρ_initial,ρ_factor,ρ_max,ρ_min,
        use_static)
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
