include("solver_options.jl")
import Base: copy, length, size

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
        n,m = model.n, model.m

        # Make dynamics inplace
        if is_inplace_dynamics(model)
            f! = model.f
        else
            f! = wrap_inplace(model.f)
        end

        # Get integration scheme
        if isdefined(TrajectoryOptimization,integration)
            discretizer = eval(integration)
        else
            throw(ArgumentError("$integration is not a defined integration scheme"))
        end

        # Generate discrete dynamics equations
        fd! = discretizer(f!, dt)
        f_aug! = f_augmented!(f!, model.n, model.m)
        fd_aug! = discretizer(f_aug!)
        F!(J,Sdot,S) = ForwardDiff.jacobian!(J,fd_aug!,Sdot,S)

        fx = zeros(n,n)
        fu = zeros(n,m)

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
            fx .= F_aug[1:model.n,1:model.n]
            fu .= F_aug[1:model.n,model.n+1:model.n+model.m]
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
Abstract type for the output of a single iteration step
"""
abstract type SolverIterResults <: SolverResults end

"""
$(TYPEDEF)
Values computed for an unconstrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct UnconstrainedResults <: SolverIterResults
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

function copy(r::UnconstrainedResults)
    UnconstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_))
end

"""
$(TYPEDEF)
Values computed for a constrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct ConstrainedResults <: SolverIterResults
    X::Array{Float64,2}  # States (n,N)
    U::Array{Float64,2}  # Controls (m,N-1)
    K::Array{Float64,3}  # Feedback gain (m,n,N-1)
    d::Array{Float64,2}  # Feedforward gain (m,N-1)
    X_::Array{Float64,2} # Predicted states (n,N)
    U_::Array{Float64,2} # Predicted controls (m,N-1)

    fx::Array{Float64,3}
    fu::Array{Float64,3}

    C::Array{Float64,2}      # Constraint values (p,N-1)
    Iμ::Array{Float64,3}     # Active constraint penalty matrix (p,p,N-1)
    LAMBDA::Array{Float64,2} # Lagrange multipliers (p,N-1)
    MU::Array{Float64,2}     # Penalty terms (p,N-1)

    CN::Array{Float64,1}     # Final constraint values (p_N,)
    IμN::Array{Float64,2}    # Final constraint penalty matrix (p_N,p_N)
    λN::Array{Float64,1}     # Final lagrange multipliers (p_N,)
    μN::Array{Float64,1}     # Final penalty terms (p_N,)

    Cx::Array{Float64,3}
    Cu::Array{Float64,3}

    Cx_N::Array{Float64,2}

    function ConstrainedResults(X,U,K,d,X_,U_,fx,fu,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN,cx,cu,cxn)
        new(X,U,K,d,X_,U_,fx,fu,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN,cx,cu,cxn)
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

    fx = zeros(n,n,N-1)
    fu = zeros(n,m,N-1)

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

    cx = zeros(p,n,N-1)
    cu = zeros(p,m,N-1)
    cxn = zeros(p_N,n)

    ConstrainedResults(X,U,K,d,X_,U_,fx,fu,
        C,Iμ,LAMBDA,MU,
        C_N,Iμ_N,λ_N,μ_N,cx,cu,cxn)

end

function copy(r::ConstrainedResults)
    ConstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),
        copy(r.C),copy(r.Iμ),copy(r.LAMBDA),copy(r.MU),copy(r.CN),copy(r.IμN),copy(r.λN),copy(r.μN))
end

"""
$(TYPEDEF)
Values cached for each solve iteration
"""
mutable struct ResultsCache <: SolverResults #TODO look into making an immutable struct
    X::Array{Float64,2}            # Final state trajectory (n,N)
    U::Array{Float64,2}            # Final control trajectory (m,N-1)
    result::Array{SolverResults,1} # SolverResults at each solve iteration
    cost::Array{Float64,1}         # Objective cost at each solve iteration
    time::Array{Float64,1}         # iLQR inner loop evaluation time at each solve iteration
    iter_type::Array{Int64,1}      # Flag indicating final inner loop iteration before outer loop update (1), otherwise (0), for each solve iteration
    termination_index::Int64       # Iteration when solve terminates

    function ResultsCache(X, U, result, cost, time, iter_type, termination_index)
        new(X, U, result, cost, time, iter_type, termination_index)
    end
end

"""
$(SIGNATURES)
Construct ResultsCache from sizes with a pre-allocated size of `n_allocation`
"""
function ResultsCache(solver::Solver,n_allocation::Int64)
    n,m = solver.model.n, solver.model.m
    N = solver.N
    ResultsCache(n,m,N,n_allocation)
end

function ResultsCache(results::SolverIterResults,n_allocation::Int64)
    m,n,N = size(results.K)
    ResultsCache(n,m,N,n_allocation)
end

function ResultsCache(n::Int, m::Int, N::Int, n_allocation::Int64)
    X = zeros(n,N)
    U = zeros(m, N-1)
    result = Array{SolverResults}(n_allocation)
    cost = zeros(n_allocation)
    time = zeros(n_allocation)
    iter_type = zeros(n_allocation)
    termination_index = 0
    ResultsCache(X, U, result, cost, time, iter_type, termination_index)
end

" $(SIGNATURES) Get the number of iterations in the cache"
size(cache::ResultsCache) = length(cache.cost)

" $(SIGNATURES) Get number of completed iterations in the cache"
function length(cache::ResultsCache)
    for i = 1:size(cache)
        if !isassigned(cache.result,i)
            return i-1
        end
    end
    return size(cache)
end

"""
$(SIGNATURES)

Combine result caches r1 and r2 by stacking r1 on top of r2, such that r2
contains the most recent results.

Useful for making results caches dynamically larger. Removes any unused entries
so size(cache) == length(cache)
"""
function merge_results_cache(r1::ResultsCache,r2::ResultsCache)
    n1 = length(r1)      # number of results
    n2 = length(r2)      # number of results
    R = ResultsCache(r1.result[1],n1+n2) # initialize new ResultsCache that will contain both ResultsCache's

    R.X = r2.X # new ResultsCache will store most recent (ie, best) state trajectory
    R.U = r2.U # new ResultsCache will store most recent (ie, best) control trajectory
    R.result[1:n1] = r1.result[1:n1] # store all valid results
    R.result[n1+1:end] = r2.result[1:n2]
    R.cost[1:n1] = r1.cost[1:n1] # store all valid costs
    R.cost[n1+1:end] = r2.cost[1:n2]
    R.time[1:n1] = r1.time[1:n1] # store all valid times
    R.time[n1+1:end] = r2.time[1:n2]
    R.iter_type[1:n1] = r1.iter_type[1:n1] # store all valid iteration types
    R.iter_type[n1+1:end] = r2.iter_type[1:n2]
    R.termination_index = n1+n2 # update total number of iterations
    R
end

"""
$(SIGNATURES)
Add the result of an iteration to the cache
"""
function add_iter!(cache::ResultsCache, results::SolverIterResults, cost::Float64, time::Float64=0., iter::Int=length(cache)+1)::Void
    cache.result[iter] = copy(results)
    cache.cost[iter] = cost
    cache.time[iter] = time
    return nothing
end
