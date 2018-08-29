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

    "final cost convergence criteria"
    eps::Float64
    "intermediate cost convergence criteria for outerloop of constrained solve"
    eps_intermediate::Float64
    "maximum constraint violation termination criteria"
    eps_constraint::Float64
    "iterations for iLQR solve"
    iterations::Int64
    "iterations for outer loop of constraint solve"
    iterations_outerloop::Int64
    "maximum number of backtracking steps during forward pass line search"
    iterations_linesearch::Int64
    "regularization term update"
    mu_reg_update::Float64
    "value increase mu_k by at each outer loop iteration"
    mu_al_update::Float64
    "regularization term for augmented controls during infeasible start"
    infeasible_regularization::Float64
    "cache all intermediate state and control trajectories"
    cache::Bool

    "Run benchmarks on forward and backward passes"
    benchmark::Bool

    infeasible::Bool
    unconstrained::Bool

    function SolverOptions(;square_root=false,verbose=false,
        c1=1e-4,c2=10.0,eps=1e-5,eps_intermediate=1e-2,
        eps_constraint=1e-3,iterations=1000,iterations_outerloop=50,
        iterations_linesearch=50,mu_reg_update=1e-3,mu_al_update=100.0,infeasible_regularization=1000.0,cache=false,
        benchmark=false,infeasible=false,unconstrained=false)

        new(square_root,verbose,c1,c2,eps,eps_intermediate,
        eps_constraint,iterations,iterations_outerloop,
        iterations_linesearch,mu_reg_update,mu_al_update,infeasible_regularization,cache,benchmark,infeasible,unconstrained)
    end
end

copy(opts::SolverOptions) = SolverOptions(;[name=>getfield(opts,name) for name in fieldnames(opts)]...)



# function show(io::IO, opts::SolverOptions)
#     println(io, "SolverOptions:")
#     print(io,"  Use Square Root: $(opts.square_root)")
# end
