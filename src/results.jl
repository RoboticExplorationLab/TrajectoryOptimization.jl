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

abstract type SolverIterResultsStatic <: SolverIterResults end

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

    xdot::Matrix # Continuous dynamics values (n,N)
    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    function UnconstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ)
        new(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ)
    end
end

struct UnconstrainedResultsStatic{N,M} <: SolverIterResultsStatic
    X::Vector{MVector{N,Float64}}  # States (n,N)
    U::Vector{MVector{M,Float64}}  # Controls (m,N)
    K::Vector{MMatrix{M,N,Float64}} # Feedback (state) gain (m,n,N)
    b::Vector{MMatrix{M,M,Float64}}  # Feedback (control) gain (m,m,N)
    d::Vector{MVector{M,Float64}}  # Feedforward gain (m,N)
    X_::Vector{MVector{N,Float64}} # Predicted states (n,N)
    U_::Vector{MVector{M,Float64}} # Predicted controls (m,N)
    S::Vector{MMatrix{N,N,Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{MVector{N,Float64}}  # Cost-to-go gradient (n,1)
    fx::Vector{MMatrix{N,N,Float64}} # State jacobian (n,n,N)
    fu::Vector{MMatrix{N,M,Float64}} # Control (k) jacobian (n,m,N-1)
    fv::Vector{MMatrix{N,M,Float64}} # Control (k+1) jacobian (n,m,N-1)
    Ac::Vector{MMatrix{N,N,Float64}} # Continous dynamics state jacobian (n,n,N)
    Bc::Vector{MMatrix{N,M,Float64}} # Continuous dynamics control jacobian (n,m,N)

    xdot::Vector{MVector{N,Float64}} # Continuous dynamics values (n,N)

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    function UnconstrainedResultsStatic(X::Vector{MVector{N,Float64}},U::Vector{MVector{M,Float64}},
            K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ) where {N,M}
        new{N,M}(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ)
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
    xdot = zeros(n,N)
    ρ = zeros(1)
    dρ = zeros(1)
    UnconstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ)
end

function UnconstrainedResults(res::UnconstrainedResultsStatic)
    UnconstrainedResults([convert(Array,getfield(res,name)) for name in fieldnames(typeof(res))]...)
end

function UnconstrainedResultsStatic(n::Int,m::Int,N::Int)
    X  = [@MVector zeros(n)   for i = 1:N]
    U  = [@MVector zeros(m)   for i = 1:N]
    K  = [@MMatrix zeros(m,n) for i = 1:N]
    b  = [@MMatrix zeros(m,m) for i = 1:N]
    d  = [@MVector zeros(m)   for i = 1:N]
    X_ = [@MVector zeros(n)   for i = 1:N]
    U_ = [@MVector zeros(m)   for i = 1:N]
    S  = [@MMatrix zeros(n,n) for i = 1:N]
    s  = [@MVector zeros(n)   for i = 1:N]
    fx = [@MMatrix zeros(n,n) for i = 1:N-1]
    fu = [@MMatrix zeros(n,m) for i = 1:N-1]
    fv = [@MMatrix zeros(n,m) for i = 1:N-1]
    Ac = [@MMatrix zeros(n,n) for i = 1:N]
    Bc = [@MMatrix zeros(n,m) for i = 1:N]
    xdot  = [@MVector zeros(n)   for i = 1:N]
    ρ = zeros(1)
    dρ = zeros(1)

    UnconstrainedResultsStatic(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,ρ,dρ)
end

function copy(r::UnconstrainedResults)
    UnconstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fx),copy(r.fu),copy(r.fv),copy(r.Ac),copy(r.Bc),copy(r.xdot),copy(r.ρ),copy(r.dρ))
end

"""
$(TYPEDEF)
Values computed for a constrained optimization problem

Time steps are always concatenated along the last dimension
"""
struct ConstrainedResultsStatic{N,M,P,PN} <: SolverIterResultsStatic
    X::Vector{MVector{N,Float64}}  # States (n,N)
    U::Vector{MVector{M,Float64}}  # Controls (m,N)
    K::Vector{MMatrix{M,N,Float64}} # Feedback (state) gain (m,n,N)
    b::Vector{MMatrix{M,M,Float64}}  # Feedback (control) gain (m,m,N)
    d::Vector{MVector{M,Float64}}  # Feedforward gain (m,N)
    X_::Vector{MVector{N,Float64}} # Predicted states (n,N)
    U_::Vector{MVector{M,Float64}} # Predicted controls (m,N)
    S::Vector{MMatrix{N,N,Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{MVector{N,Float64}}  # Cost-to-go gradient (n,1)
    fx::Vector{MMatrix{N,N,Float64}} # State jacobian (n,n,N)
    fu::Vector{MMatrix{N,M,Float64}} # Control (k) jacobian (n,m,N-1)
    fv::Vector{MMatrix{N,M,Float64}} # Control (k+1) jacobian (n,m,N-1)
    Ac::Vector{MMatrix{N,N,Float64}} # Continous dynamics state jacobian (n,n,N)
    Bc::Vector{MMatrix{N,M,Float64}} # Continuous dynamics control jacobian (n,m,N)

    xdot::Vector{MVector{N,Float64}} # Continuous dynamics values (n,N)

    C::Vector{MVector{P,Float64}}      # Constraint values (p,N)
    C_prev::Vector{MVector{P,Float64}} # Previous constraint values (p,N)
    Iμ::Array{Diagonal{Float64}}        # Active constraint penalty matrix (p,p,N)
    LAMBDA::Vector{MVector{P,Float64}} # Lagrange multipliers (p,N)
    MU::Vector{MVector{P,Float64}}     # Penalty terms (p,N)

    CN::MVector{PN,Float64}       # Final constraint values (p_N,)
    CN_prev::MVector{PN,Float64}  # Previous final constraint values (p_N,)
    IμN::Diagonal{Float64}        # Final constraint penalty matrix (p_N,p_N)
    λN::MVector{PN,Float64}       # Final lagrange multipliers (p_N,)
    μN::MVector{PN,Float64}       # Final penalty terms (p_N,)

    Cx::Vector{MMatrix{P,N,Float64}}
    Cu::Vector{MMatrix{P,M,Float64}}

    Cx_N::MMatrix{PN,N,Float64}

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    V_al_prev::Array{Float64,2} # Augmented Lagrangian Method update terms, see ALGENCAN notation
    V_al_current::Array{Float64,2} # Augmented Lagrangian Method update terms

    function ConstrainedResultsStatic(X::Vector{MVector{N,Float64}},U::Vector{MVector{M,Float64}},
            K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,
            C::Vector{MVector{P,Float64}},C_prev,Iμ,LAMBDA,MU,
            CN::MVector{PN,Float64},CN_prev,IμN,λN,μN,
            cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current) where {N,M,P,PN}
        # @show P
        # @show PN
        # @show typeof(cxn)
        new{N,M,P,PN}(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,C,C_prev,Iμ,LAMBDA,MU,CN,CN_prev,IμN,λN,μN,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)
    end
end

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

    xdot::Matrix             # Continous dynamics (n,N)

    C::Array{Float64,2}      # Constraint values (p,N)
    C_prev::Array{Float64,2} # Previous constraint values (p,N)
    Iμ::Array{Float64,3}     # Active constraint penalty matrix (p,p,N)
    LAMBDA::Array{Float64,2} # Lagrange multipliers (p,N)
    MU::Array{Float64,2}     # Penalty terms (p,N)

    CN::Array{Float64,1}      # Final constraint values (p_N,)
    CN_prev::Array{Float64,1} # Previous final constraint values (p_N,)
    IμN::Array{Float64,2}     # Final constraint penalty matrix (p_N,p_N)
    λN::Array{Float64,1}      # Final lagrange multipliers (p_N,)
    μN::Array{Float64,1}      # Final penalty terms (p_N,)

    Cx::Array{Float64,3}
    Cu::Array{Float64,3}

    Cx_N::Array{Float64,2}

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    V_al_prev::Array{Float64,2} # Augmented Lagrangian Method update terms, see ALGENCAN notation
    V_al_current::Array{Float64,2} # Augmented Lagrangian Method update terms

    function ConstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,C,C_prev,Iμ,LAMBDA,MU,CN,CN_prev,IμN,λN,μN,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)
        new(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,C,C_prev,Iμ,LAMBDA,MU,CN,CN_prev,IμN,λN,μN,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)
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

    xdot = zeros(n,N)

    # Stage Constraints
    C = zeros(p,N)
    C_prev = zeros(p,N)
    Iμ = zeros(p,p,N)
    LAMBDA = zeros(p,N)
    MU = ones(p,N)

    # Terminal Constraints (make 2D so it works well with stage values)
    C_N = zeros(p_N)
    C_N_prev = zeros(p_N)
    Iμ_N = zeros(p_N,p_N)
    λ_N = zeros(p_N)
    μ_N = ones(p_N)

    cx = zeros(p,n,N)
    cu = zeros(p,m,N)
    cxn = zeros(p_N,n)

    ρ = zeros(1)
    dρ = zeros(1)

    V_al_prev = zeros(p,N) #TODO preallocate only (pI,N)
    V_al_current = zeros(p,N)

    ConstrainedResults(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,
        C,C_prev,Iμ,LAMBDA,MU,
        C_N,C_N_prev,Iμ_N,λ_N,μ_N,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)

end

function ConstrainedResults(res::ConstrainedResultsStatic)
    ConstrainedResults([convert(Array,getfield(res,name)) for name in fieldnames(typeof(res))]...)
end



function ConstrainedResultsStatic(n::Int,m::Int,p::Int,N::Int,p_N::Int=n)
    X  = [@MVector zeros(n)   for i = 1:N]
    U  = [@MVector zeros(m)   for i = 1:N]
    K  = [@MMatrix zeros(m,n) for i = 1:N]
    b  = [@MMatrix zeros(m,m) for i = 1:N]
    d  = [@MVector zeros(m)   for i = 1:N]
    X_ = [@MVector zeros(n)   for i = 1:N]
    U_ = [@MVector zeros(m)   for i = 1:N]
    S  = [@MMatrix zeros(n,n) for i = 1:N]
    s  = [@MVector zeros(n)   for i = 1:N]
    fx = [@MMatrix zeros(n,n) for i = 1:N-1]
    fu = [@MMatrix zeros(n,m) for i = 1:N-1]
    fv = [@MMatrix zeros(n,m) for i = 1:N-1]
    Ac = [@MMatrix zeros(n,n) for i = 1:N]
    Bc = [@MMatrix zeros(n,m) for i = 1:N]
    xdot  = [@MVector zeros(n)   for i = 1:N]

    # Stage Constraints
    C      = [@MVector zeros(p)  for i = 1:N]
    C_prev = [@MVector zeros(p)  for i = 1:N]
    Iμ     = [Diagonal(zeros(p)) for i = 1:N]
    LAMBDA = [@MVector zeros(p)  for i = 1:N]
    MU     = [@MVector ones(p)   for i = 1:N]

    # Terminal Constraints (make 2D so it works well with stage values)
    C_N      = @MVector zeros(p_N)
    C_N_prev = @MVector zeros(p_N)
    Iμ_N     = Diagonal(zeros(p_N))
    λ_N      = @MVector zeros(p_N)
    μ_N      = @MVector ones(p_N)

    cx  = [@MMatrix zeros(p,n)   for i = 1:N]
    cu  = [@MMatrix zeros(p,m)   for i = 1:N]
    cxn = zeros(p_N,n)

    ρ = zeros(1)
    dρ = zeros(1)

    V_al_prev = zeros(p,N) #TODO preallocate only (pI,N)
    V_al_current = zeros(p,N)

    ConstrainedResultsStatic(X,U,K,b,d,X_,U_,S,s,fx,fu,fv,Ac,Bc,xdot,
        C,C_prev,Iμ,LAMBDA,MU,
        C_N,C_N_prev,Iμ_N,λ_N,μ_N,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)

end

function copy(r::ConstrainedResults)
    ConstrainedResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fx),copy(r.fu),copy(r.fv),copy(r.Ac),copy(r.Bc),copy(r.xdot),
        copy(r.C),copy(r.C_prev),copy(r.Iμ),copy(r.LAMBDA),copy(r.MU),copy(r.CN),copy(r.CN_prev),copy(r.IμN),copy(r.λN),copy(r.μN),
        copy(r.Cx),copy(r.Cu),copy(r.Cx_N),copy(r.ρ),copy(r.dρ),copy(r.V_al_prev),copy(r.V_al_current))
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
    result = Array{SolverResults}(undef,n_allocation)
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
function add_iter!(cache::ResultsCache, results::SolverIterResults, cost::Float64, time::Float64=0., iter::Int=length(cache)+1)::Nothing
    cache.result[iter] = copy(results)
    cache.cost[iter] = cost
    cache.time[iter] = time
    return nothing
end

function add_iter_outerloop!(cache::ResultsCache, results::SolverIterResults, iter)::Nothing
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

struct DircolVars
    Z::Vector{Float64}
    X::SubArray{Float64}
    U::SubArray{Float64}
end

function DircolVars(Z::Vector,n::Int,m::Int,N::Int)
    z = reshape(Z,n+m,N)
    X = view(z,1:n,:)
    U = view(z,n+1:n+m,:)
    DircolVars(Z,X,U)
end

function DircolVars(n::Int,m::Int,N::Int)
    Z = zeros((n+m)N)
    z = reshape(Z,n+m,N)
    X = view(z,1:n,:)
    U = view(z,n+1:n+m,:)
    DircolVars(Z,X,U)
end

function DircolVars(X::Matrix,U::Matrix)
    Z = packZ(X,U)
    n,m,N = get_sizes(X,U)
    DircolVars(Z,n,m,N)
end

DiffFloat = Union{Float64,ForwardDiff.Dual}
struct DircolResults <: SolverIterResults
    vars::DircolVars
    # vars_::DircolVars
    Z::Vector{Float64}
    X::SubArray{Float64}
    U::SubArray{Float64}
    fVal::Matrix{Float64}
    X_::Union{Matrix{Float64},SubArray{Float64}}
    U_::Union{Matrix{Float64},SubArray{Float64}}
    fVal_::Matrix{Float64}
    A::Array{Float64,3}
    B::Array{Float64,3}
    weights::Vector{Float64}
    # c_colloc::Matrix{Float64}
    # ceq::Matrix{Float64}
    # c::Matrix{Float64}
    # DircolResults(Z,X,U,fVal,X_,U_,fVal_,A,B,weights) =
    #     new(Z,X,U,fVal,X_,U_,fVal_,A,B,weights)
end
#
# abstract type DircolMethod end
#
# type HermiteSimpson <: DircolMethod end

function DircolResults(solver::Solver,method::Symbol)
    DircolResults(get_sizes(solver)...,method)
end

function DircolResults(n::Int,m::Int,N::Int,method::Symbol)
    N,N_ = get_N(N,method)
    Z = zeros(N*(n+m))
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X,vars.U
    fVal = zero(X)
    if method == :hermite_simpson
        X_ = zeros(n,N_)
        U_ = zeros(m,N_)
        fVal_ = zeros(n,N_)
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    elseif method == :midpoint
        X_ = zeros(n,N) # midpoints plus terminal
        U_ = U
        fVal_ = zero(X_)
        N_ = size(X_,2)
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    else
        X_ = X
        U_ = U
        fVal_ = fVal
        N_ = size(X_,2)
        A = zeros(n,n+m,N_) # These methods conveniently use the gradient of Z
        B = zeros(0,0,N_)
    end

    weights = get_weights(method,N_)
    DircolResults(vars,Z,X,U,fVal,X_,U_,fVal_,A,B,weights)
end
