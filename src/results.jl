import Base: isempty,copy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Results types for storing arrays used during computation
#
#     TYPES                                        Tree
#        SolverResults                           ---------
#        SolverIterResults                      SolverResults
#        ConstrainedResults                       ↙     ↘
#        UnconstrainedResults          ResultsCache   SolverIterResults
#        ResultsCache                                     ↙     ↘
#                                      UnconstrainedResults    ConstrainedResults
#
#     METHODS
#         copy(UnconstrainedResults)
#         copy(ConstrainedResults)
#         size(ResultsCache): size of pre-allocated cache
#         length(ResultsCache): current number of stored iterations
#         merge_results_cache: merge two ResultsCache's
#         add_iter!: Add a SolverIterResults to ResultsCache
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    U::Array{Float64,2}  # Controls (m,N)
    K::Array{Float64,3}  # Feedback (state) gain (m,n,N)
    b::Array{Float64,3}  # Feedback (control) gain (m,m,N)
    d::Array{Float64,2}  # Feedforward gain (m,N)
    X_::Array{Float64,2} # Predicted states (n,N)
    U_::Array{Float64,2} # Predicted controls (m,N)
    S::Array{Float64,3}  # Cost-to-go hessian (n,n)
    s::Array{Float64,2}  # Cost-to-go gradient (n,1)
    fx::Array{Float64,3} # State jacobian (n,n,N)
    fu::Array{Float64,3} # Control (k) jacobian (n,m,N-1)
    fv::Array{Float64,3} # Control (k+1) jacobian (n,n,N-1)
    Ac::Array{Float64,3} # Continous dynamics state jacobian (n,n,N)
    Bc::Array{Float64,3} # Continuous dynamics control jacobian (n,n,N)

    mu_reg::Array{Float64,1}

    function UnconstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,mu_reg)
        new(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,mu_reg)
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
    U = zeros(m,N)
    K = zeros(m,n,N)
    b = zeros(m,m,N)
    d = zeros(m,N)
    X_ = zeros(n,N)
    U_ = zeros(m,N)
    S = zeros(n,n,N)
    s = zeros(n,N)
    fx = zeros(n,n,N-1)
    fu = zeros(n,m,N-1)
    fv = zeros(n,m,N-1) # gradient with respect to u_{k+1}
    Ac = zeros(n,n,N)
    Bc = zeros(n,m,N)
    mu_reg = zeros(1)
    UnconstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,mu_reg)
end

function copy(r::UnconstrainedResults)
    UnconstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fx),copy(r.fu),copy(r.fv),copy(r.Ac),copy(r.Bc),copy(r.mu_reg))
end

"""
$(TYPEDEF)
Values computed for a constrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct ConstrainedResults <: SolverIterResults
    X::Array{Float64,2}  # States (n,N)
    U::Array{Float64,2}  # Controls (m,N)
    K::Array{Float64,3}  # Feedback (state) gain (m,n,N)
    b::Array{Float64,3}  # Feedback (control) gain (m,m,N)
    d::Array{Float64,2}  # Feedforward gain (m,N)
    X_::Array{Float64,2} # Predicted states (n,N)
    U_::Array{Float64,2} # Predicted controls (m,N)
    S::Array{Float64,3}  # Cost-to-go hessian (n,n)
    s::Array{Float64,2}  # Cost-to-go gradient (n,1)

    fx::Array{Float64,3}
    fu::Array{Float64,3}
    fv::Array{Float64,3}

    Ac::Array{Float64,3}
    Bc::Array{Float64,3}

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

    mu_reg::Array{Float64,1}

    function ConstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN,cx,cu,cxn,mu_reg)
        new(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN,cx,cu,cxn,mu_reg)
    end
end

function ConstrainedResults()
    ConstrainedResults(0,0,0,0)
end

isempty(res::ConstrainedResults) = isempty(res.X) && isempty(res.U)

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
    U = zeros(m,N)
    K = zeros(m,n,N)
    b = zeros(m,m,N)
    d = zeros(m,N)
    X_ = zeros(n,N)
    U_ = zeros(m,N)
    S = zeros(n,n,N)
    s = zeros(n,N)

    fx = zeros(n,n,N-1)
    fu = zeros(n,m,N-1)
    fv = zeros(n,m,N-1)

    Ac = zeros(n,n,N)
    Bc = zeros(n,m,N)

    # Stage Constraints
    C = zeros(p,N)
    Iμ = zeros(p,p,N)
    LAMBDA = zeros(p,N)
    MU = ones(p,N)

    # Terminal Constraints (make 2D so it works well with stage values)
    C_N = zeros(p_N)
    Iμ_N = zeros(p_N,p_N)
    λ_N = zeros(p_N)
    μ_N = ones(p_N)

    cx = zeros(p,n,N)
    cu = zeros(p,m,N)
    cxn = zeros(p_N,n)

    mu_reg = zeros(1)

    ConstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,
        C,Iμ,LAMBDA,MU,
        C_N,Iμ_N,λ_N,μ_N,cx,cu,cxn,mu_reg)

end

function copy(r::ConstrainedResults)
    ConstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fx),copy(r.fu),copy(r.fv),copy(r.Ac),copy(r.Bc),
        copy(r.C),copy(r.Iμ),copy(r.LAMBDA),copy(r.MU),copy(r.CN),copy(r.IμN),copy(r.λN),copy(r.μN),
        copy(r.Cx),copy(r.Cu),copy(r.Cx_N),copy(r.mu_reg))
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
    n, m = solver.model.n, solver.model.m
    N = solver.N
    ResultsCache(n,m,N,n_allocation)
end

function ResultsCache(results::SolverIterResults,n_allocation::Int64)
    m,n,N = size(results.K)
    #N += 1 # K is (m,n,N-1) <- changed K to be (m,n,N)
    ResultsCache(n,m,N,n_allocation)
end

function ResultsCache(n::Int, m::Int, N::Int, n_allocation::Int64)
    X = zeros(n,N)
    U = zeros(m, N)
    result = Array{SolverResults}(n_allocation)
    cost = zeros(n_allocation)
    time = zeros(n_allocation)
    iter_type = zeros(n_allocation)
    termination_index = n_allocation
    ResultsCache(X, U, result, cost, time, iter_type, termination_index)
end

" $(SIGNATURES) Get the number of iterations in the cache"
size(cache::ResultsCache) = length(cache.cost)

" $(SIGNATURES) Get number of completed iterations in the cache"
function length(cache::ResultsCache) #TODO- I think something is wrong with this
    for i = 1:size(cache)
        if !isassigned(cache.result,i)
            return i-1
        end
    end
    return size(cache)
    #return cache.termination_index
end

"""
$(SIGNATURES)

Combine result caches r1 and r2 by stacking r1 on top of r2, such that r2
contains the most recent results.

Useful for making results caches dynamically larger. Removes any unused entries
so size(cache) == length(cache)
"""
function merge_results_cache(r1::ResultsCache,r2::ResultsCache;infeasible::Bool=true)
    n1 = r1.termination_index     # number of results
    n2 = r2.termination_index      # number of results

    R = ResultsCache(r2.result[1],n1+n2) # initialize new ResultsCache that will contain both ResultsCache's

    R.X = r2.X # new ResultsCache will store most recent (ie, best) state trajectory
    R.U = r2.U # new ResultsCache will store most recent (ie, best) control trajectory

    for i = 1:n1
        R.result[i] = copy(r1.result[i]) # store all valid results
    end
    for i = n1+1:n1+n2
        R.result[i] = copy(r2.result[i-n1]) # store all valid costs
    end

    R.cost[1:n1] = r1.cost[1:n1]
    R.cost[n1+1:end] = r2.cost[1:n2]
    R.time[1:n1] = r1.time[1:n1] # store all valid times
    R.time[n1+1:end] = r2.time[1:n2]
    R.iter_type[1:n1] = r1.iter_type[1:n1] # store all valid iteration types
    R.iter_type[n1+1:end] = r2.iter_type[1:n2]

    if infeasible
        R.iter_type[n1+1] = 2 # flag beginning of infeasible->feasible solve
    end

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

function add_iter_outerloop!(cache::ResultsCache, results::SolverIterResults, iter)::Void
    cache.result[iter] = copy(results)
    return nothing
end

function check_multipliers(results,solver)
    p,N = size(results.C)
    pI = solver.obj.pI
    for i = 1:N
        if (results.LAMBDA[1:pI,i]'*results.C[1:pI,i]) < 0.0
            println("multiplier problem @ $i")
            println("$(results.LAMBDA[1:pI,i].*results.C[1:pI,i] .< 0.0)")
            println("$(results.LAMBDA[1:pI,i])")
            println("$(results.C[1:pI,i])\n")
            break
        else
            nothing
            # println("no multiplier problems\n")
        end
    end

    return nothing
end
