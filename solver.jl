include("solver_options.jl")

"""
$(SIGNATURES)
Determine if the dynamics in model are in place. i.e. the function call is of
the form `f!(xdot,x,u)`, where `xdot` is modified in place. Returns a boolean.
"""
function is_inplace_dynamics(model::Model)::Bool
    x = rand(model.n)
    u = rand(model.m)
    xdot = rand(model.n)
    try
        model.f(xdot,x,u)
    catch x
        if x isa MethodError
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Makes the dynamics function `f(x,u)` appear to operate as an inplace operation of the
form `f!(xdot,x,u)`.
"""
function wrap_inplace(f::Function)
    f!(xdot,x,u) = copy!(xdot, f(x,u))
end


"""
$(TYPEDEF)
Contains all information to solver a trajectory optimization problem.

The Solver type is immutable so is unique to the particular problem. However,
anything in `Solver.opts` can be changed dynamically.
"""
struct Solver
    model::Model         # Dynamics model
    obj::Objective       # Objective (cost function and constraints)
    opts::SolverOptions  # Solver options (iterations, method, convergence criteria, etc)
    dt::Float64          # Time step
    fd::Function         # Discrete in place dynamics function, `fd(ẋ,x,u)`
    F::Function          # Jacobian of discrete dynamics, `fx,fu = F(x,u)`
    N::Int               # Number of time steps

    function Solver(model::Model, obj::Objective; integration::Symbol=:rk4, dt=0.01, opts::SolverOptions=SolverOptions())
        N = Int(floor(obj.tf/dt));

        # Make dynamics inplace
        if is_inplace_dynamics(model)
            f! = model.f
        else
            f! = wrap_inplace(model.f)
        end

        # Get integration scheme
        if isdefined(iLQR,integration)
            discretizer = eval(integration)
        else
            throw(ArgumentError("$integration is not a defined integration scheme"))
        end

        # Generate discrete dynamics equations
        fd! = discretizer(f!, dt)
        f_aug! = f_augmented!(f!, model.n, model.m)
        fd_aug! = discretizer(f_aug!)
        F!(J,Sdot,S) = ForwardDiff.jacobian!(J,fd_aug!,Sdot,S)

        # Auto-diff discrete dynamics
        function Jacobians!(x,u)
            nm1 = model.n + model.m + 1
            J = zeros(nm1, nm1)
            S = zeros(nm1)
            S[1:model.n] = x
            S[model.n+1:end-1] = u
            S[end] = dt
            Sdot = zeros(S)
            F_aug = F!(J,Sdot,S)
            fx = F_aug[1:model.n,1:model.n]
            fu = F_aug[1:model.n,model.n+1:model.n+model.m]
            return fx, fu
        end
        new(model, obj, opts, dt, fd!, Jacobians!, N)

    end
end

"""
$(TYPEDEF)
Abstract type for the output of solving a trajectory optimization problem
"""
abstract type SolverResults end

"""
$(TYPEDEF)
Values computed for an unconstrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct UnconstrainedResults <: SolverResults
    X::Array{Float64,2}  # States (n,N)
    U::Array{Float64,2}  # Controls (m,N-1)
    K::Array{Float64,3}  # Feedback gain (m,n,N-1)
    d::Array{Float64,2}  # Feedforward gain (m,N-1)
    X_::Array{Float64,2} # Predicted states (n,N)
    U_::Array{Float64,2} # Predicted controls (m,N-1)

    function UnconstrainedResults(X,U,K,d,X_,U_)
        new(X,U,K,d,X_,U_)
    end
end

"""
$(SIGNATURES)
Construct results from sizes

# Arguments
* n: number of states
* m: number of controls
* N: number of time steps
"""
function UnconstrainedResults(n::Int,m::Int,N::Int)
    X = zeros(n,N)
    U = zeros(m,N-1)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    X_ = zeros(n,N)
    U_ = zeros(m,N-1)
    UnconstrainedResults(X,U,K,d,X_,U_)
end

"""
$(TYPEDEF)
Values computed for a constrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct ConstrainedResults <: SolverResults
    X::Array{Float64,2}  # States (n,N)
    U::Array{Float64,2}  # Controls (m,N-1)
    K::Array{Float64,3}  # Feedback gain (m,n,N-1)
    d::Array{Float64,2}  # Feedforward gain (m,N-1)
    X_::Array{Float64,2} # Predicted states (n,N)
    U_::Array{Float64,2} # Predicted controls (m,N-1)

    C::Array{Float64,2}      # Constraint values (p,N-1)
    Iμ::Array{Float64,3}     # Active constraint penalty matrix (p,p,N-1)
    LAMBDA::Array{Float64,2} # Lagrange multipliers (p,N-1)
    MU::Array{Float64,2}     # Penalty terms (p,N-1)

    CN::Array{Float64,1}     # Final constraint values (p_N,)
    IμN::Array{Float64,2}    # Final constraint penalty matrix (p_N,p_N)
    λN::Array{Float64,1}     # Final lagrange multipliers (p_N,)
    μN::Array{Float64,1}     # Final penalty terms (p_N,)

    function ConstrainedResults(X,U,K,d,X_,U_,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN)
        new(X,U,K,d,X_,U_,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN)
    end
end

"""
$(SIGNATURES)
Construct results from sizes

# Arguments
* n: number of states
* m: number of controls
* p: number of constraints
* N: number of time steps
* p_N (default=n): number of terminal constraints
"""
function ConstrainedResults(n::Int,m::Int,p::Int,N::Int,p_N::Int=n)
    X = zeros(n,N)
    U = zeros(m,N-1)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    X_ = zeros(n,N)
    U_ = zeros(m,N-1)

    # Stage Constraints
    C = zeros(p,N-1)
    Iμ = zeros(p,p,N-1)
    LAMBDA = zeros(p,N-1)
    MU = ones(p,N-1)

    # Terminal Constraints (make 2D so it works well with stage values)
    C_N = zeros(p_N)
    Iμ_N = zeros(p_N,p_N)
    λ_N = zeros(p_N)
    μ_N = ones(p_N)

    ConstrainedResults(X,U,K,d,X_,U_,
        C,Iμ,LAMBDA,MU,
        C_N,Iμ_N,λ_N,μ_N)

end

mutable struct ResultsCache <: SolverResults
    X::Array{Float64,2}
    U::Array{Float64,2}
    result::Array{SolverResults,1}
    cost::Array{Float64,1}
    time::Array{Float64,1}
    iter_type::Array{Int64,1}
    termination_index::Int64

    function ResultsCache(X, U, result, cost, time, iter_type, termination_index)
        new(X, U, result, cost, time, iter_type, termination_index)
    end
end

function ResultsCache(solver::Solver,max_steps::Int64)
    X = zeros(solver.model.n,solver.N)
    U = zeros(solver.model.m, solver.N-1)
    result = Array{SolverResults}(max_steps)
    cost = zeros(max_steps)
    time = zeros(max_steps)
    iter_type = zeros(max_steps)
    termination_index = 0
    ResultsCache(X, U, result, cost, time, iter_type, termination_index)
end

function merge_results_cache(r1::ResultsCache,r2::ResultsCache,solver::Solver)
    n1 = r1.termination_index
    n2 = r2.termination_index
    R = ResultsCache(solver,n1+n2)

    R.X = r2.X
    R.U = r2.U
    R.result[1:n1] = r1.result[1:n1]
    R.result[n1+1:end] = r2.result[1:n2]
    R.cost[1:n1] = r1.cost[1:n1]
    R.cost[n1+1:end] = r2.cost[1:n2]
    R.time[1:n1] = r1.time[1:n1]
    R.time[n1+1:end] = r2.time[1:n2]
    R.iter_type[1:n1] = r1.iter_type[1:n1]
    R.iter_type[n1+1:end] = r2.iter_type[1:n2]
    R.termination_index = n1+n2
    R
end
