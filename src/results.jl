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

# Trajectory Types
Trajectory = Vector{T} where T <: AbstractArray
TrajectoryVectors = Vector{Vector{T}} where T <: Real
TrajectoryMatrices = Vector{Matrix{T}} where T <: Real
TrajectoryDiagonals = Vector{Diagonal{Vector{T}}} where T <: Real

"""
$(TYPEDEF)
    Abstract type for the backward pass expansion terms
"""
abstract type AbstractBackwardPass end

struct BackwardPass <: AbstractBackwardPass
    Qx::Vector{Vector{Float64}}
    Qu::Vector{Vector{Float64}}
    Qxx::Vector{Matrix{Float64}}
    Qux::Vector{Matrix{Float64}}
    Quu::Vector{Matrix{Float64}}

    Qux_reg::Vector{Matrix{Float64}}
    Quu_reg::Vector{Matrix{Float64}}

    function BackwardPass(n::Int,m::Int,N::Int)
        Qx = [zeros(n) for i = 1:N-1]
        Qu = [zeros(m) for i = 1:N-1]
        Qxx = [zeros(n,n) for i = 1:N-1]
        Qux = [zeros(m,n) for i = 1:N-1]
        Quu = [zeros(m,m) for i = 1:N-1]

        Qux_reg = [zeros(m,n) for i = 1:N-1]
        Quu_reg = [zeros(m,m) for i = 1:N-1]

        new(Qx,Qu,Qxx,Qux,Quu,Qux_reg,Quu_reg)
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

struct UnconstrainedVectorResults <: UnconstrainedIterResults
    X::Trajectory  # States (n,N)
    U::Trajectory  # Controls (m,N)

    K::Trajectory # Feedback (state) gain (m,n,N)
    d::Trajectory  # Feedforward gain (m,N)

    X_::Trajectory # Predicted states (n,N)
    U_::Trajectory # Predicted controls (m,N)

    S::Trajectory  # Cost-to-go hessian (n,n)
    s::Trajectory  # Cost-to-go gradient (n,1)

    fdx::Trajectory # Discrete dynamics state jacobian (n,n,N)
    fdu::Trajectory # Discrete dynamics control jacobian (n,m,N-1)

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    bp::BackwardPass



    function UnconstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ,bp)
        new(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ,bp)
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

    ρ = zeros(1)
    dρ = zeros(1)

    bp = BackwardPass(n,m,N)

    UnconstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,ρ,dρ,bp)
end

function copy(r::UnconstrainedVectorResults)
    UnconstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.ρ),copy(r.dρ),copy(r.bp))
end

################################################################################
#                                                                              #
#                     CONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct ConstrainedVectorResults <: ConstrainedIterResults
    X::Trajectory  # States (n,N)
    U::Trajectory  # Controls (m,N)

    K::Trajectory # Feedback (state) gain (m,n,N)
    d::Trajectory  # Feedforward gain (m,N)

    X_::Trajectory # Predicted states (n,N)
    U_::Trajectory # Predicted controls (m,N)

    S::Trajectory  # Cost-to-go hessian (n,n)
    s::Trajectory  # Cost-to-go gradient (n,1)

    fdx::Trajectory # State jacobian (n,n,N)
    fdu::Trajectory # Control (k) jacobian (n,m,N-1)

    C::Trajectory      # Constraint values (p,N)
    C_prev::Trajectory # Previous constraint values (p,N)
    Iμ::Trajectory        # fcxtive constraint penalty matrix (p,p,N)
    λ::Trajectory # Lagrange multipliers (p,N)
    μ::Trajectory     # Penalty terms (p,N)

    Cx::Trajectory # State jacobian (n,n,N)
    Cu::Trajectory # Control (k) jacobian (n,m,N-1)

    t_prev::Trajectory
    λ_prev::Trajectory

    nesterov::Vector{Float64}

    active_set::Trajectory

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    bp::BackwardPass

    μ_prev::Trajectory
    H::Array{Float64,2}
    Iμ_prev::Trajectory
    y::Trajectory
    ss::Trajectory

    function ConstrainedVectorResults(
            X,U,
            K,d,
            X_,U_,
            S,s,
            fdx,fdu,
            C,C_prev,
            Iμ,λ,μ,
            Cx,Cu,
            t_prev,λ_prev,nesterov,
            active_set,
            ρ,dρ,bp,μ_prev,H,Iμ_prev,y,ss)
        new(X,U,K,d,X_,U_,S,s,fdx,fdu,C,C_prev,Iμ,λ,μ,Cx,Cu,t_prev,λ_prev,nesterov,active_set,ρ,dρ,bp,μ_prev,H,Iμ_prev,y,ss)
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

    t_prev      = [i != N ? ones(p) : ones(p_N)  for i = 1:N]
    λ_prev      = [i != N ? zeros(p) : zeros(p_N)  for i = 1:N]

    nesterov = [0.;1.]

    active_set = [i != N ? zeros(Bool,p) : zeros(Bool,p_N)  for i = 1:N]

    ρ = zeros(1)
    dρ = zeros(1)

    bp = BackwardPass(n,m,N)
    μ_prev = [i != N ? ones(p) : ones(p_N)  for i = 1:N]
    H = Array(Diagonal(ones(p*(N-1) + p_N)))
    Iμ_prev    = [i != N ? Diagonal(ones(p)) : Diagonal(ones(p_N)) for i = 1:N]
    y = [i != N ? ones(p) : ones(p_N)  for i = 1:N]
    ss = [i != N ? ones(p) : ones(p_N)  for i = 1:N]

    ConstrainedVectorResults(X,U,K,d,X_,U_,S,s,fdx,fdu,
        C,C_prev,Iμ,λ,μ,Cx,Cu,t_prev,λ_prev,nesterov,active_set,ρ,dρ,bp,μ_prev,H,Iμ_prev,y,ss)
end

function copy(r::ConstrainedVectorResults)
    ConstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),
        copy(r.C),copy(r.C_prev),copy(r.Iμ),copy(r.λ),copy(r.μ),
        copy(r.Cx),copy(r.Cu),copy(r.active_set),copy(r.ρ),copy(r.dρ),copy(r.bp),copy(r.μ_prev),copy(r.H),copy(r.Iμ_prev),copy(r.y),copy(r.ss))
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

        results.bp.Qu[k] = results.bp.Qu[k][1:m̄]
        results.bp.Quu[k] = results.bp.Quu[k][1:m̄,1:m̄]
        results.bp.Qux[k] = results.bp.Qux[k][1:m̄,1:nn]
        results.bp.Quu_reg[k] = results.bp.Quu_reg[k][1:m̄,1:m̄]
        results.bp.Qux_reg[k] = results.bp.Qux_reg[k][1:m̄,1:nn]
    end
    # Don't need to modify terminal results C,Cx,Cu,λ,μ,Iμ since they are uneffected by u_infeasible
    return nothing
end

function init_results(solver::Solver,X::AbstractArray,U::AbstractArray; λ=Array{Float64,2}(undef,0,0), μ=Array{Float64,2}(undef,0,0))
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

    # Get sizes
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    if solver.state.constrained
        p,pI,pE = get_num_constraints(solver)
        p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

        results = ConstrainedVectorResults(nn,mm,p,N,p_N)

        # Set initial penalty term values
        copyto!(results.μ, results.μ*solver.opts.penalty_initial) # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float

        # Special penalty initializations #TODO indexing needs to be fixed here to account for less control constraints than controls
        # if solver.state.minimum_time
        #     for k = 1:N-1
        #         k != 1 ? results.μ[k][p] = solver.opts.penalty_initial_minimum_time_equality : nothing
        #         results.μ[k][m̄] = solver.opts.penalty_initial_minimum_time_inequality
        #         results.μ[k][m̄+m̄] = solver.opts.penalty_initial_minimum_time_inequality
        #     end
        # end
        if solver.state.infeasible
            nothing #TODO
        end

        # Initial Lagrange multipliers (warm start)
        if ~isempty(λ)
            copy_λ!(solver, results, λ)
        end
        if ~isempty(μ)
            error("penalty warm start not implemented")
        end

        # Set initial regularization
        results.ρ[1] = solver.opts.bp_reg_initial

    else
        results = UnconstrainedVectorResults(n,m,N)
    end

    if solver.opts.square_root
        for k = 1:N
            results.S[k] = zeros(nn+mm,nn)
        end
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
