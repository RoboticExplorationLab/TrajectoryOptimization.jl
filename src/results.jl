import Base: isempty,copy,getindex,setindex!,firstindex,lastindex,copyto!,length,*,+,IndexStyle,iterate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Results types for storing arrays used during computation
#
#     TYPES                                        Tree
#        SolverResults                           ---------
#        SolverIterResults                      SolverResults
#        ConstrainedResults                       ↙     ↘
#        UnconstrainedResults          ResultsCache   SolverIterResults
#                                                          ↙     ↘
#                                      UnconstrainedResults    ConstrainedResults
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



abstract type AbstractTrajectoryVariable   end

struct TrajectoryVariable{T <: AbstractArray} <: AbstractTrajectoryVariable
    x::Vector{T}
end

function TrajectoryVariable(N::Int,n::Int)
    x = [zeros(n) for k = 1:N]
    TrajectoryVariable(x)
end

function TrajectoryVariable(N::Int,sze::Vararg{Int,K}) where K
    x = [zeros(sze) for k = 1:N]
    TrajectoryVariable(x)
end

function TrajectoryVariable(N::Int,sze::Union{NTuple{K,Int} where K,Int}; size_N::Union{NTuple{K,Int} where K,Int})
    x = [k == N ? zeros(size_N) : zeros(sze) for k = 1:N]
    TrajectoryVariable(x)
end

function TrajectoryVariable(X::Matrix)
    x = [X[:,k] for k = 1:size(X,2)]
    TrajectoryVariable(x)
end

function size(x::TrajectoryVariable)
    return (size(x.x[1])...,length(x.x))
end

function getindex(x::TrajectoryVariable,ind::Int)
    x.x[ind]
end

function setindex!(x::TrajectoryVariable,value,ind::Int)
    x.x[ind] = value
end

firstindex(x::TrajectoryVariable) = 1
lastindex(x::TrajectoryVariable) = length(x.x)
length(x::TrajectoryVariable) = length(x.x)
*(x::TrajectoryVariable,c::Real) = TrajectoryVariable(x.x * c)

function copyto!(x::TrajectoryVariable,y::Matrix)
    for k = 1:length(x.x)
        x.x[k] = y[:,k]
    end
end

function copyto!(x::TrajectoryVariable,y::TrajectoryVariable)
    for k = 1:length(x.x)
        copyto!(x.x[k], y.x[k])
    end
end

function iterate(x::TrajectoryVariable)
    (x[1],1)
end
function iterate(x::TrajectoryVariable,state)
    if state < length(x.x)
        return (x[state+1],state+1)
    else
        return nothing
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

# abstract type SolverIterResultsStatic <: SolverIterResults end

abstract type SolverVectorResults <: SolverIterResults end
abstract type UnconstrainedIterResults <: SolverVectorResults end
abstract type ConstrainedIterResults <: SolverVectorResults end

################################################################################
#                                                                              #
#                   UNCONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct UnconstrainedVectorResults{TV,TM} <: UnconstrainedIterResults
    X::TV  # States (n,N)
    U::TV  # Controls (m,N)

    K::TM # Feedback (state) gain (m,n,N)
    d::TV  # Feedforward gain (m,N)

    X_::TV # Predicted states (n,N)
    U_::TV # Predicted controls (m,N)

    S::TM  # Cost-to-go hessian (n,n)
    s::TV  # Cost-to-go gradient (n,1)

    fdx::TM # Discrete dynamics state jacobian (n,n,N)
    fdu::TM # Discrete dynamics control jacobian (n,m,N-1)

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    function UnconstrainedVectorResults(X::TV,U::TV,
            K::TM,d,X_,U_,S,s,fdx,fdu,ρ,dρ) where {TV,TM}
        new{TV,TM}(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ)
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
function UnconstrainedVectorResults(n::Int,m::Int,N::Int)
    X  = [zeros(n)   for i = 1:N]
    U  = [zeros(m)   for i = 1:N-1]

    K  = [zeros(m,n) for i = 1:N-1]
    d  = [zeros(m)   for i = 1:N-1]

    X_ = [zeros(n)   for i = 1:N]
    U_ = [zeros(m)   for i = 1:N-1]


    S  = [zeros(n,n) for i = 1:N]
    s  = [zeros(n)   for i = 1:N]


    fdx = [zeros(n,n) for i = 1:N-1]
    fdu = [zeros(n,m) for i = 1:N-1]

    ρ = ones(1)
    dρ = ones(1)

    UnconstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ)
end

function UnconstrainedVectorResults(n::Int,m::Int,N::Int,T::Type)
    if T <: AbstractArray
        UnconstrainedVectorResults(n,m,N)
    else
        X = T(N,n)
        U = T(N-1,m)
        K = T(N,m,n)
        d = T(N,m)
        X_ = T(N,n)
        U_ = T(N-1,m)
        S = T(N,n,n)
        s = T(N,n)
        fdx = T(N,n,n)
        fdu = T(N,n,m)

        ρ = ones(1)
        dρ = ones(1)

        UnconstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ)
    end
end

function copy(r::UnconstrainedVectorResults)
    UnconstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.ρ),copy(r.dρ))
end

################################################################################
#                                                                              #
#                     CONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct ConstrainedVectorResults{TV,TM} <: ConstrainedIterResults
    X::TV  # States (n,N)
    U::TV  # Controls (m,N)

    K::TM # Feedback (state) gain (m,n,N)
    d::TV  # Feedforward gain (m,N)

    X_::TV # Predicted states (n,N)
    U_::TV # Predicted controls (m,N)

    S::TM  # Cost-to-go hessian (n,n)
    s::TV  # Cost-to-go gradient (n,1)

    fdx::TM # State jacobian (n,n,N)
    fdu::TM # Control (k) jacobian (n,m,N-1)

    C::TV      # Constraint values (p,N)
    C_prev::TV # Previous constraint values (p,N)
    Iμ::Vector{Diagonal{Float64,Vector{Float64}}}        # fcxtive constraint penalty matrix (p,p,N)
    λ::TV # Lagrange multipliers (p,N)
    μ::TV     # Penalty terms (p,N)

    Cx::TM # State jacobian (n,n,N)
    Cu::TM # Control (k) jacobian (n,m,N-1)

    active_set::Vector{Vector{T}} where T # active set of constraints

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    # function ConstrainedVectorResults(X::TV,U::TV,
    #         K::TM,d,X_,U_,S,s,fdx,fdu,
    #         C::TV,C_prev,Iμ,λ,μ,
    #         Cx,Cu,active_set,ρ,dρ) where {TV,TM}
    #     println("This constructor")
    #     new{TM,TV}(X,U,K,d,X_,U_,S,s,fdx,fdu,C,C_prev,Iμ,λ,μ,Cx,Cu,active_set,ρ,dρ)
    # end
end

isempty(res::SolverIterResults) = isempty(res.X) && isempty(res.U)

ConstrainedVectorResults() = ConstrainedVectorResults(0,0,0,0,0)


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
function ConstrainedVectorResults(n::Int,m::Int,p::Int,N::Int,p_N::Int)
    X  = [zeros(n)   for i = 1:N]
    U  = [zeros(m)   for i = 1:N-1]

    K  = [zeros(m,n) for i = 1:N-1]
    d  = [zeros(m)   for i = 1:N-1]

    X_ = [zeros(n)   for i = 1:N]
    U_ = [zeros(m)   for i = 1:N-1]

    S  = [zeros(n,n) for i = 1:N]
    s  = [zeros(n)   for i = 1:N]


    fdx = [zeros(n,n) for i = 1:N-1]
    fdu = [zeros(n,m) for i = 1:N-1]

    # Stage Constraints
    C      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    C_prev = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    λ      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]
    μ      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]

    Cx  = [i != N ? zeros(p,n) : zeros(p_N,n)  for i = 1:N]
    Cu  = [i != N ? zeros(p,m) : zeros(p_N,0)  for i = 1:N]

    active_set = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]

    ρ = ones(1)
    dρ = ones(1)

    ConstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,
        C,C_prev,Iμ,λ,μ,
        Cx,Cu,active_set,ρ,dρ)
end

function ConstrainedVectorResults(n::Int,m::Int,p::Int,N::Int,p_N::Int,T::Type)
    if T <: AbstractArray
        ConstrainedVectorResults(n,m,p,N,p_N)
    else
        X = T(N,n)
        U = T(N-1,m)
        K = T(N,m,n)
        d = T(N,m)
        X_ = T(N,n)
        U_ = T(N-1,m)
        S = T(N,n,n)
        s = T(N,n)
        fdx = T(N,n,n)
        fdu = T(N,n,m)

        C =      T(N,p, size_N=p_N)
        C_prev = T(N,p, size_N=p_N)
        Iμ     = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
        λ =      T(N,p, size_N=p_N)
        μ =      T([i != N ? ones(p) : ones(p_N)  for i = 1:N])

        Cx = T(N, (p,n), size_N=(p_N,n))
        Cu = T(N, (p,m), size_N=(p_N,0))

        active_set = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]

        ρ = ones(1)
        dρ = ones(1)

        ConstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,
            C,C_prev,Iμ,λ,μ,
            Cx,Cu,active_set,ρ,dρ)
    end
end


function copy(r::ConstrainedVectorResults)
    ConstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),
        copy(r.C),copy(r.C_prev),copy(r.Iμ),copy(r.λ),copy(r.μ),
        copy(r.Cx),copy(r.Cu),copy(r.active_set),copy(r.ρ),copy(r.dρ))
end

#############
# Utilities #
#############
function remove_infeasible_controls!(results::SolverIterResults,solver::Solver)
    # turn off infeasible functionality
    p_inf,pI_inf,pE_inf = get_num_constraints(solver)

    solver.state.infeasible = false

    # get sizes
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    if solver.state.minimum_time
        idx = 1:p_inf-n-1
        idx = [idx;p_inf]
    else
        idx = 1:p_inf-n
    end

    for k = 1:N-1
        results.U[k] = results.U[k][1:m̄]
        results.U_[k] = results.U_[k][1:m̄]
        results.K[k] = results.K[k][1:m̄,1:nn]
        results.d[k] = results.d[k][1:m̄]
        results.fdu[k] = results.fdu[k][1:nn,1:m̄]

        results.C[k] = results.C[k][idx]
        results.Cx[k] = results.Cx[k][idx,1:nn]
        results.Cu[k] = results.Cu[k][idx,1:m̄]
        results.λ[k] = results.λ[k][idx]
        results.μ[k] = results.μ[k][idx]
        results.Iμ[k] = Diagonal(Array(results.Iμ[k])[idx,idx]) # TODO there should be a more efficient way to do this
        results.active_set[k] = results.active_set[k][idx]
    end
    # Don't need to modify terminal results C,Cx,Cu,λ,μ,Iμ since they are uneffected by u_infeasible
    return nothing
end

function init_results(solver::Solver,X::AbstractArray,U::AbstractArray; λ=Array{Float64,2}(undef,0,0))
    n,m,N = get_sizes(solver)

    if !isempty(X)
        solver.state.infeasible = true
    end

    # Chop off last control if N controls are passed in
    if size(U,2) == N
        U = U[:,1:N-1]
    end

    # Generate initial trajectoy (tacking on infeasible and minimum time controls)
    X_init, U_init = get_initial_trajectory(solver, X, U)

    if solver.state.constrained
        # Get sizes
        m̄,mm = get_num_controls(solver)
        n̄,nn = get_num_states(solver)

        p,pI,pE = get_num_constraints(solver)
        p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

        m̄,mm = get_num_controls(solver)

        results = ConstrainedVectorResults(nn,mm,p,N,p_N,TrajectoryVariable)

        # Set initial penalty term values
        copyto!(results.μ, results.μ*solver.opts.μ_initial) # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float

        # Special penalty initializations
        if solver.state.minimum_time
            results.μ[1:N-1][p] .*= solver.opts.penalty_initial_minimum_time_equality
            results.μ[1:N-1][m̄] .*= solver.opts.penalty_initial_minimum_time_inequality
            results.μ[1:N-1][m̄+m̄] .*= solver.opts.penalty_initial_minimum_time_inequality
        end
        if solver.state.infeasible
            nothing #TODO
        end

        # Initial Lagrange multipliers (warm start)
        if ~isempty(λ)
            copy_λ!(solver, results, λ)
        end

        # Set initial regularization
        results.ρ[1] = solver.opts.bp_reg_initial

    else
        results = UnconstrainedVectorResults(n,m,N,TrajectoryVariable)
    end
    copyto!(results.X, X_init)
    copyto!(results.U, U_init)
    return results
end

function copy_λ!(solver, results, λ)
    N = solver.N
    p_new = length(λ[1])
    p_N_new = length(λ[end])

    p, = get_num_constraints(solver)
    p_N, = get_num_terminal_constraints(solver)

    if p_new == p  # all constraint λs passed in
        cid = trues(p)
    elseif p_new == solver.obj.p  # only "original" constraint λs passed
        cid = original_constraint_inds(solver)
    else
        err = ArgumentError("λ is not the correct dimension ($p_new). It must be either size $p or $(solver.obj.p)")
        throw(err)
    end
    for k = 1:N-1
        results.λ[k][cid] = λ[k]
    end
    results.λ[N] = λ[N]
end
