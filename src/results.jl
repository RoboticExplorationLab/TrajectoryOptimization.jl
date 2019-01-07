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
#                                                          ↙     ↘
#                                      UnconstrainedResults    ConstrainedResults
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

# abstract type SolverIterResultsStatic <: SolverIterResults end

abstract type SolverVectorResults <: SolverIterResults end
abstract type UnconstrainedIterResults <: SolverVectorResults end
abstract type ConstrainedIterResults <: SolverVectorResults end

################################################################################
#                                                                              #
#                   UNCONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct UnconstrainedVectorResults <: UnconstrainedIterResults
    X::Vector{Vector{Float64}}  # States (n,N)
    U::Vector{Vector{Float64}}  # Controls (m,N)

    K::Vector{Matrix{Float64}} # Feedback (state) gain (m,n,N)
    d::Vector{Vector{Float64}}  # Feedforward gain (m,N)

    X_::Vector{Vector{Float64}} # Predicted states (n,N)
    U_::Vector{Vector{Float64}} # Predicted controls (m,N)

    S::Vector{Matrix{Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{Vector{Float64}}  # Cost-to-go gradient (n,1)

    fdx::Vector{Matrix{Float64}} # Discrete dynamics state jacobian (n,n,N)
    fdu::Vector{Matrix{Float64}} # Discrete dynamics control jacobian (n,m,N-1)

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    function UnconstrainedVectorResults(X::Vector{Vector{Float64}},U::Vector{Vector{Float64}},
            K,d,X_,U_,S,s,fdx,fdu,ρ,dρ)
        new(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ)
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

function copy(r::UnconstrainedVectorResults)
    UnconstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.ρ),copy(r.dρ))
end

################################################################################
#                                                                              #
#                     CONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct ConstrainedVectorResults <: ConstrainedIterResults
    X::Vector{Vector{Float64}}  # States (n,N)
    U::Vector{Vector{Float64}}  # Controls (m,N)

    K::Vector{Matrix{Float64}} # Feedback (state) gain (m,n,N)
    d::Vector{Vector{Float64}}  # Feedforward gain (m,N)

    X_::Vector{Vector{Float64}} # Predicted states (n,N)
    U_::Vector{Vector{Float64}} # Predicted controls (m,N)

    S::Vector{Matrix{Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{Vector{Float64}}  # Cost-to-go gradient (n,1)

    fdx::Vector{Matrix{Float64}} # State jacobian (n,n,N)
    fdu::Vector{Matrix{Float64}} # Control (k) jacobian (n,m,N-1)

    C::Vector{Vector{Float64}}      # Constraint values (p,N)
    C_prev::Vector{Vector{Float64}} # Previous constraint values (p,N)
    Iμ::Vector{Diagonal{Float64,Vector{Float64}}}        # fcxtive constraint penalty matrix (p,p,N)
    λ::Vector{Vector{Float64}} # Lagrange multipliers (p,N)
    μ::Vector{Vector{Float64}}     # Penalty terms (p,N)

    Cx::Vector{Matrix{Float64}} # State jacobian (n,n,N)
    Cu::Vector{Matrix{Float64}} # Control (k) jacobian (n,m,N-1)

    active_set::Vector{Vector{Bool}} # active set of constraints

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    function ConstrainedVectorResults(X::Vector{Vector{Float64}},U::Vector{Vector{Float64}},
            K,d,X_,U_,S,s,fdx,fdu,
            C::Vector{Vector{Float64}},C_prev,Iμ,λ,μ,
            Cx,Cu,active_set,ρ,dρ)
        new(X,U,K,d,X_,U_,S,s,fdx,fdu,C,C_prev,Iμ,λ,μ,Cx,Cu,active_set,ρ,dρ)
    end
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


function copy(r::ConstrainedVectorResults)
    ConstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),
        copy(r.C),copy(r.C_prev),copy(r.Iμ),copy(r.λ),copy(r.μ),
        copy(r.Cx),copy(r.Cu),copy(r.active_set),copy(r.ρ),copy(r.dρ))
end

################################################################################
#                                                                              #
#                        DIRCOL VARIABLES STRUCTURE                            #
#                                                                              #
################################################################################

struct DircolVars{T}
    Z::Vector{T}
    X::SubArray{T}
    U::SubArray{T}
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

function DircolVars(res::SolverIterResults)
    DircolVars(to_array(res.X), to_array(res.U))
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

    if solver.state.minimum_time
        idx = 1:p_inf-n-1
        idx = [idx;p_inf]
    else
        idx = 1:p_inf-n
    end

    for k = 1:N-1
        results.U[k] = results.U[k][1:m̄]
        results.U_[k] = results.U_[k][1:m̄]
        results.K[k] = results.K[k][1:m̄,1:n]
        results.d[k] = results.d[k][1:m̄]
        results.fdu[k] = results.fdu[k][1:n,1:m̄]

        results.C[k] = results.C[k][idx]
        results.Cx[k] = results.Cx[k][idx,1:n]
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

    # Generate initial trajectoy (tacking on infeasible and minimum time controls)
    X_init, U_init = get_initial_trajectory(solver, X, U)

    if solver.state.constrained
        # Get sizes
        p,pI,pE = get_num_constraints(solver)
        p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

        m̄,mm = get_num_controls(solver)

        results = ConstrainedVectorResults(n,mm,p,N,p_N)

        # Set initial penalty term values
        results.μ .*= solver.opts.penalty_initial # TODO change to assign, not multiply: penalty_initial needs to be initialized as an array instead of float

        # Special penalty initializations
        if solver.state.minimum_time
            for k = 1:solver.N-1
                results.μ[k][p] = solver.opts.penalty_initial_minimum_time_equality
                results.μ[k][m̄] = solver.opts.penalty_initial_minimum_time_inequality
                results.μ[k][m̄+m̄] = solver.opts.penalty_initial_minimum_time_inequality
            end
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
        results = UnconstrainedVectorResults(n,m,N)
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
