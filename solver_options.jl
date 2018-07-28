import Base.show

mutable struct SolverOptions
    square_root::Bool
    augmented_lagrange::Bool
    verbose::Bool
    inplace_dynamics::Bool

    c1::Float64 # lower bound for forward pass line search, 0 < c1 < 1
    c2::Float64 # upper bound for forward pass line search, 0 < c1 < c2 < 1

    eps::Float64 # final cost convergence criteria
    eps_intermediate::Float64 # intermediate cost convergence criteria for outerloop of constrained solve
    eps_constraint::Float64 # maximum constraint violation termination criteria
    iterations::Int64 # iterations for iLQR solve
    iterations_outerloop::Int64 # iterations for outer loop of constraint solve
    iterations_linesearch::Int64 # maximum number of backtracking steps during forward pass line search
    mu_regularization::Float64 # termed add to Quu during backward pass to insure positive semidefiniteness

    function SolverOptions(;square_root=false,al=false,verbose=false,
        inplace_dynamics=false,c1=1e-4,c2=1.0,eps=1e-5,eps_intermediate=1e-2,
        eps_constraint=1e-2,iterations=100,iterations_outerloop=25,
        iterations_linesearch=25,mu_regularization=1.0)

        new(square_root,al,verbose,inplace_dynamics,c1,c2,eps,eps_intermediate,
        eps_constraint,iterations,iterations_outerloop,
        iterations_linesearch,mu_regularization)
    end
end

function show(io::IO, opts::SolverOptions)
    println(io, "SolverOptions:")
    print(io,"  Use Square Root: $(opts.square_root)")
end
