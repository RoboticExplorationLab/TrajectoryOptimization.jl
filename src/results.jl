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
#         add_iter!: fdxd a SolverIterResults to ResultsCache
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
    b::Vector{Matrix{Float64}}  # Feedback (control) gain (m,m,N)
    d::Vector{Vector{Float64}}  # Feedforward gain (m,N)

    X_::Vector{Vector{Float64}} # Predicted states (n,N)
    U_::Vector{Vector{Float64}} # Predicted controls (m,N)
    S::Vector{Matrix{Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{Vector{Float64}}  # Cost-to-go gradient (n,1)

    L::Array{Float64} # Stage costs
    Q::Array{Float64} # Action-value cost-to-go
    l::Vector{Float64}
    q::Vector{Float64}

    fdx::Vector{Matrix{Float64}} # Discrete dynamics state jacobian (n,n,N)
    fdu::Vector{Matrix{Float64}} # Discrete dynamics control jacobian (n,m,N-1)
    fdv::Vector{Matrix{Float64}} # Control (k+1) jacobian (n,m,N-1)
    fcx::Vector{Matrix{Float64}} # Continous dynamics state jacobian (n,n,N)
    fcu::Vector{Matrix{Float64}} # Continuous dynamics control jacobian (n,m,N)

    dx::Vector{Vector{Float64}} # Continuous dynamics values (n,N)
    xm::Vector{Vector{Float64}} # State midpoints (n,N) should be (n,N-1)
    um::Vector{Vector{Float64}}  # Control midpoints (m,N) should be (n,N-1)

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    function UnconstrainedVectorResults(X::Vector{Vector{Float64}},U::Vector{Vector{Float64}},
            K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ)
        new(X,U,K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ)
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
function UnconstrainedVectorResults(n::Int,m::Int,N::Int,ctrl_int::Symbol=:zoh)
    X  = [zeros(n)   for i = 1:N]
    U  = [zeros(m)   for i = 1:N]

    K  = [zeros(m,n) for i = 1:N]
    b  = [zeros(m,m) for i = 1:N]
    d  = [zeros(m)   for i = 1:N]

    X_ = [zeros(n)   for i = 1:N]
    U_ = [zeros(m)   for i = 1:N]

    if ctrl_int == :foh
        S  = [zeros(n+m,n+m) for i = 1:N]
        s  = [zeros(n+m)   for i = 1:N]
        L = zeros(2*(n+m),2*(n+m))
        Q = zeros(n+m+m,n+m+m)
        l = zeros(2*(n+m))
        q = zeros(n+m+m)
    else
        S  = [zeros(n,n) for i = 1:N]
        s  = [zeros(n)   for i = 1:N]
        L = zeros(n+m,n+m)
        Q = zeros(n+m,n+m)
        l = zeros(n+m)
        q = zeros(n+m)
    end

    fdx = [zeros(n,n) for i = 1:N-1]
    fdu = [zeros(n,m) for i = 1:N-1]
    fdv = [zeros(n,m) for i = 1:N-1]
    fcx = [zeros(n,n) for i = 1:N]
    fcu = [zeros(n,m) for i = 1:N]

    dx = [zeros(n) for i = 1:N]
    xm = [zeros(n) for i = 1:N]
    um  = [zeros(m)   for i = 1:N]

    ρ = ones(1)
    dρ = ones(1)

    UnconstrainedVectorResults(X,U,K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ)
end

function copy(r::UnconstrainedVectorResults)
    UnconstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.L),copy(r.Q),copy(r.l),copy(r.q),copy(r.fdx),copy(r.fdu),copy(r.fdv),copy(r.fcx),copy(r.fcu),copy(r.dx),copy(r.xm),copy(r.um),copy(r.ρ),copy(r.dρ))
end

################################################################################
#                                                                              #
#                     CONSTRAINED RESULTS STRUCTURE                            #
#                                                                              #
################################################################################

struct ConstrainedVectorResults <: ConstrainedIterResults
    X::Vector{Vector{Float64}}  # States (n,N)
    U::Vector{Vector{Float64}}  # Controls (m,N)

    K::Vector{Matrix{Float64}}  # Feedback (state) gain (m,n,N)
    b::Vector{Matrix{Float64}}  # Feedback (control) gain (m,m,N)
    d::Vector{Vector{Float64}}  # Feedforward gain (m,N)

    X_::Vector{Vector{Float64}} # Predicted states (n,N)
    U_::Vector{Vector{Float64}} # Predicted controls (m,N)

    S::Vector{Matrix{Float64}}  # Cost-to-go hessian (n,n)
    s::Vector{Vector{Float64}}  # Cost-to-go gradient (n,1)

    L::Array{Float64} # Stage costs
    Q::Array{Float64} # Action-value cost-to-go
    l::Vector{Float64}
    q::Vector{Float64}

    fdx::Vector{Matrix{Float64}} # State jacobian (n,n,N)
    fdu::Vector{Matrix{Float64}} # Control (k) jacobian (n,m,N-1)
    fdv::Vector{Matrix{Float64}} # Control (k+1) jacobian (n,m,N-1)
    fcx::Vector{Matrix{Float64}} # Continous dynamics state jacobian (n,n,N)
    fcu::Vector{Matrix{Float64}} # Continuous dynamics control jacobian (n,m,N)

    dx::Vector{Vector{Float64}} # Continuous dynamics values (n,N)
    xm::Vector{Vector{Float64}} # State midpoints (n,N) should be (n,N-1)
    um::Vector{Vector{Float64}}  # Control midpoints (m,N)

    gs::Vector{Vector{Float64}} # Constraint values (pIs,N)
    gc::Vector{Vector{Float64}} # Constraint values (pIc,N)
    hs::Vector{Vector{Float64}} # Constraint values (pEs,N) (note: Nth constraint is pEsN)
    hc::Vector{Vector{Float64}} # Constraint values (pEc,N)

    gs_prev::Vector{Vector{Float64}} # Previous constraint values (pIs,N)
    gc_prev::Vector{Vector{Float64}} # Previous constraint values (pIc,N)
    hs_prev::Vector{Vector{Float64}} # Previous constraint values (pEs,N) (note: Nth constraint is pEsN)
    hc_prev::Vector{Vector{Float64}} # Previous constraint values (pEc,N)

    λs::Vector{Vector{Float64}} # state multipliers (pIs,N)
    λc::Vector{Vector{Float64}} # control multipliers (pIc,N)
    κs::Vector{Vector{Float64}} # state multipliers (pEs,N) (note: Nth multiplier is pEsN)
    κc::Vector{Vector{Float64}} # control multipliers (pEc,N)

    μs::Vector{Vector{Float64}}     # state Penalty terms (pIs,N)
    μc::Vector{Vector{Float64}}     # control Penalty terms (pIc,N)
    νs::Vector{Vector{Float64}}     # state Penalty terms (pEs,N) (note: Nth multiplier is pEsN)
    νc::Vector{Vector{Float64}}     # control Penalty terms (pEc,N)

    # Penalty matrices
    Iμs::Vector{Diagonal{Float64,Vector{Float64}}}
    Iμc::Vector{Diagonal{Float64,Vector{Float64}}}
    Iνs::Vector{Diagonal{Float64,Vector{Float64}}}
    Iνc::Vector{Diagonal{Float64,Vector{Float64}}}

    # Constraint Jacobians
    gsx::Vector{Matrix{Float64}}
    gcu::Vector{Matrix{Float64}}
    hsx::Vector{Matrix{Float64}}
    hcu::Vector{Matrix{Float64}}

    gs_active_set::Vector{Vector{Bool}} # active set of state inequality constraints
    gc_active_set::Vector{Vector{Bool}} # active set of control inequality constraints

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    #########

    function ConstrainedVectorResults(X::Vector{Vector{Float64}},U::Vector{Vector{Float64}},
            K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,
            gs,gc,hs,hc,gs_prev,gc_prev,hs_prev,hc_prev,λs,λc,κs,κc,μs,μc,νs,νc,Iμs,Iμc,Iνs,Iνc,gsx,gcu,hsx,hcu,gs_active_set,gc_active_set,ρ,dρ)

        new(X,U,K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,gs,gc,hs,hc,gs_prev,gc_prev,hs_prev,hc_prev,λs,λc,κs,κc,μs,μc,νs,νc,Iμs,Iμc,Iνs,Iνc,gsx,gcu,hsx,hcu,gs_active_set,gc_active_set,ρ,dρ)
    end
end

isempty(res::SolverIterResults) = isempty(res.X) && isempty(res.U)

ConstrainedVectorResults() = ConstrainedVectorResults(0,0,0,0,0,0,0,0)

function ConstrainedVectorResults(n::Int,m::Int,N::Int,pIs::Int,pIc::Int,pEs::Int,pEsN::Int,pEc::Int,ctrl_int::Symbol=:zoh)
    X  = [zeros(n)   for k = 1:N]
    U  = [zeros(m)   for k = 1:N]

    K  = [zeros(m,n) for k = 1:N]
    b  = [zeros(m,m) for k = 1:N]
    d  = [zeros(m)   for k = 1:N]

    X_ = [zeros(n)   for k = 1:N]
    U_ = [zeros(m)   for k = 1:N]

    if ctrl_int == :foh
        S  = [zeros(n+m,n+m) for k = 1:N]
        s  = [zeros(n+m)   for k = 1:N]
        L = zeros(2*(n+m),2*(n+m))
        Q = zeros(n+m+m,n+m+m)
        l = zeros(2*(n+m))
        q = zeros(n+m+m)
    else
        S  = [zeros(n,n) for k = 1:N]
        s  = [zeros(n)   for k = 1:N]
        L = zeros(n+m,n+m)
        Q = zeros(n+m,n+m)
        l = zeros(n+m)
        q = zeros(n+m)
    end

    fdx = [zeros(n,n) for k = 1:N-1]
    fdu = [zeros(n,m) for k = 1:N-1]
    fdv = [zeros(n,m) for k = 1:N-1]
    fcx = [zeros(n,n) for k = 1:N]
    fcu = [zeros(n,m) for k = 1:N]
    dx = [zeros(n)   for k = 1:N]
    xm = [zeros(n)   for k = 1:N]
    um = [zeros(m)   for k = 1:N]

    gs = [zeros(pIs) for k = 1:N] # Constraint values (pIs,N)
    gc = [zeros(pIc) for k = 1:N] # Constraint values (pIc,N)
    hs = [k != N ? zeros(pEs) : zeros(pEsN) for k = 1:N] # Constraint values (pEs,N) (note: Nth constraint is pEsN)
    hc = [zeros(pEc) for k = 1:N] # Constraint values (pEc,N)

    gs_prev = [zeros(pIs) for k = 1:N] # Constraint values (pIs,N)
    gc_prev = [zeros(pIc) for k = 1:N] # Constraint values (pIc,N)
    hs_prev = [k != N ? zeros(pEs) : zeros(pEsN) for k = 1:N] # Constraint values (pEs,N) (note: Nth constraint is pEsN)
    hc_prev = [zeros(pEc) for k = 1:N] # Constraint values (pEc,N)

    λs = [zeros(pIs) for k = 1:N] # state multipliers (pIs,N)
    λc = [zeros(pIc) for k = 1:N] # control multipliers (pIc,N)
    κs = [k != N ? zeros(pEs) : zeros(pEsN) for k = 1:N] # state multipliers (pEs,N) (note: Nth multiplier is pEsN)
    κc = [zeros(pEc) for k = 1:N] # control multipliers (pEc,N)

    μs = [ones(pIs) for k = 1:N]    # state Penalty terms (pIs,N)
    μc = [ones(pIc) for k = 1:N]     # control Penalty terms (pIc,N)
    νs = [k != N ? ones(pEs) : ones(pEsN) for k = 1:N]     # state Penalty terms (pEs,N) (note: Nth multiplier is pEsN)
    νc = [ones(pEc) for k = 1:N]     # control Penalty terms (pEc,N)

    # Penalty matrices
    Iμs = [Diagonal(ones(pIs)) for k = 1:N]
    Iμc = [Diagonal(ones(pIc)) for k = 1:N]
    Iνs = [k != N ? Diagonal(ones(pEs)) : Diagonal(ones(pEsN)) for k = 1:N]
    Iνc = [Diagonal(ones(pEc)) for k = 1:N]

    # Constraint Jacobians
    gsx = [zeros(pIs,n) for k = 1:N]
    gcu = [zeros(pIc,m) for k = 1:N]
    hsx = [k != N ? zeros(pEs,n) : zeros(pEsN,n) for k = 1:N]
    hcu = [zeros(pEc,m) for k = 1:N]

    gs_active_set = [zeros(Bool,pIs) for k = 1:N] # active set of state inequality constraints
    gc_active_set = [zeros(Bool,pIc) for k = 1:N]# active set of control inequality constraints

    ρ = ones(1)
    dρ = ones(1)

    ConstrainedVectorResults(X,U,K,b,d,X_,U_,S,s,L,Q,l,q,fdx,fdu,fdv,fcx,fcu,dx,xm,um,
        gs,gc,hs,hc,gs_prev,gc_prev,hs_prev,hc_prev,λs,λc,κs,κc,μs,μc,νs,νc,Iμs,Iμc,Iνs,Iνc,gsx,gcu,hsx,hcu,gs_active_set,gc_active_set,ρ,dρ)
end


function copy(r::ConstrainedVectorResults)
    ConstrainedVectorResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.L),copy(r.Q),copy(r.l),copy(r.q),copy(r.fdx),copy(r.fdu),copy(r.fdv),copy(r.fcx),copy(r.fcu),copy(r.dx),copy(r.xm),copy(r.um),
    copy(r.gs),copy(r.gc),copy(r.hs),copy(r.hc),copy(r.gs_prev),copy(r.gc_prev),copy(r.hs_prev),copy(r.hc_prev),copy(r.λs),copy(r.λc),copy(r.κs),copy(r.κc),copy(r.μs),copy(r.μc),copy(r.νs),copy(r.νc),copy(r.Iμs),copy(r.Iμc),copy(r.Iνs),copy(r.Iνc),copy(r.gsx),copy(r.gcu),copy(r.hsx),copy(r.hcu),copy(r.gs_active_set),copy(r.gc_active_set),copy(r.ρ),copy(r.dρ))
end

##################
# STATIC RESULTS #
##################


"""
$(TYPEDEF)
Values computed for an unconstrained optimization problem
Time steps are always concatenated along the last dimension
"""
struct UnconstrainedStaticResults{N,M,NN,MM,NM} <: UnconstrainedIterResults
    X::Vector{MVector{N,Float64}}  # States (n,N)
    U::Vector{MVector{M,Float64}}  # Controls (m,N)

    K::Vector{MMatrix{M,N,Float64,NM}} # Feedback (state) gain (m,n,N)
    b::Vector{MMatrix{M,M,Float64,MM}}  # Feedback (control) gain (m,m,N)
    d::Vector{MVector{M,Float64}}  # Feedforward gain (m,N)

    X_::Vector{MVector{N,Float64}} # Predicted states (n,N)
    U_::Vector{MVector{M,Float64}} # Predicted controls (m,N)
    S::Vector{MMatrix{N,N,Float64,NN}}  # Cost-to-go hessian (n,n)
    s::Vector{MVector{N,Float64}}  # Cost-to-go gradient (n,1)

    fdx::Vector{MMatrix{N,N,Float64,NN}} # State jacobian (n,n,N)
    fdu::Vector{MMatrix{N,M,Float64,NM}} # Control (k) jacobian (n,m,N-1)
    fdv::Vector{MMatrix{N,M,Float64,NM}} # Control (k+1) jacobian (n,m,N-1)
    fcx::Vector{MMatrix{N,N,Float64,NN}} # Continous dynamics state jacobian (n,n,N)
    fcu::Vector{MMatrix{N,M,Float64,NM}} # Continuous dynamics control jacobian (n,m,N)

    dx::Vector{MVector{N,Float64}}
    xm::Vector{MVector{N,Float64}}
    um::Vector{MVector{M,Float64}}

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    function UnconstrainedStaticResults(X::Vector{MVector{N,Float64}},U::Vector{MVector{M,Float64}},
            K::Vector{MMatrix{M,N,Float64,NM}},b::Vector{MMatrix{M,M,Float64,MM}},d,X_,U_,S,s,fdx::Vector{MMatrix{N,N,Float64,NN}},fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ) where {N,M,NN,MM,NM}
        new{N,M,NN,MM,NM}(X,U,K,b,d,X_,U_,S,s,fdx,fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ)
    end
end

function UnconstrainedStaticResults(n::Int,m::Int,N::Int)
    X  = [@MVector zeros(n)   for i = 1:N]
    U  = [@MVector zeros(m)   for i = 1:N]

    K  = [@MMatrix zeros(m,n) for i = 1:N]
    b  = [@MMatrix zeros(m,m) for i = 1:N]
    d  = [@MVector zeros(m)   for i = 1:N]

    X_ = [@MVector zeros(n)   for i = 1:N]
    U_ = [@MVector zeros(m)   for i = 1:N]
    S  = [@MMatrix zeros(n,n) for i = 1:N]
    s  = [@MVector zeros(n)   for i = 1:N]

    fdx = [@MMatrix zeros(n,n) for i = 1:N-1]
    fdu = [@MMatrix zeros(n,m) for i = 1:N-1]
    fdv = [@MMatrix zeros(n,m) for i = 1:N-1]
    fcx = [@MMatrix zeros(n,n) for i = 1:N]
    fcu = [@MMatrix zeros(n,m) for i = 1:N]

    dx = [@MVector zeros(n)   for i = 1:N]
    xm = [@MVector zeros(n)   for i = 1:N]
    um  = [@MVector zeros(m)   for i = 1:N]

    ρ = ones(1)
    dρ = ones(1)

    UnconstrainedStaticResults(X,U,K,b,d,X_,U_,S,s,fdx,fdu,fdv,fcx,fcu,dx,xm,um,ρ,dρ)
end

function copy(r::UnconstrainedStaticResults)
    UnconstrainedStaticResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.fdv),copy(r.fcx),copy(r.fcu),copy(r.dx),copy(r.xm),(copy.um),copy(r.ρ),copy(r.dρ))
end

"""
$(TYPEDEF)
Values computed for a constrained optimization problem
Time steps are always concatenated along the last dimension
"""
struct ConstrainedStaticResults{N,M,P,PN,NM,NN,MM,PP,PPN,NP,MP,NPN} <: ConstrainedIterResults
    X::Vector{MVector{N,Float64}}  # States (n,N)
    U::Vector{MVector{M,Float64}}  # Controls (m,N)

    K::Vector{MMatrix{M,N,Float64,NM}} # Feedback (state) gain (m,n,N)
    b::Vector{MMatrix{M,M,Float64,MM}}  # Feedback (control) gain (m,m,N)
    d::Vector{MVector{M,Float64}}  # Feedforward gain (m,N)

    X_::Vector{MVector{N,Float64}} # Predicted states (n,N)
    U_::Vector{MVector{M,Float64}} # Predicted controls (m,N)
    S::Vector{MMatrix{N,N,Float64,NN}}  # Cost-to-go hessian (n,n)
    s::Vector{MVector{N,Float64}}  # Cost-to-go gradient (n,1)

    fdx::Vector{MMatrix{N,N,Float64,NN}} # State jacobian (n,n,N)
    fdu::Vector{MMatrix{N,M,Float64,NM}} # Control (k) jacobian (n,m,N-1)
    fdv::Vector{MMatrix{N,M,Float64,NM}} # Control (k+1) jacobian (n,m,N-1)
    fcx::Vector{MMatrix{N,N,Float64,NN}} # Continous dynamics state jacobian (n,n,N)
    fcu::Vector{MMatrix{N,M,Float64,NM}} # Continuous dynamics control jacobian (n,m,N)

    dx::Vector{MVector{N,Float64}}   # Continuous dynamics values (n,N)
    xm::Vector{MVector{N,Float64}}   # State midpoints (n,N)
    um::Vector{MVector{M,Float64}}     # Controls midpoints (m,N)

    C::Vector{MVector{P,Float64}}      # Constraint values (p,N)
    C_prev::Vector{MVector{P,Float64}} # Previous constraint values (p,N)
    Iμ::Vector{MMatrix{P,P,Float64,PP}}# fcxtive constraint penalty matrix (p,p,N)
    λ::Vector{MVector{P,Float64}} # Lagrange multipliers (p,N)
    μ::Vector{MVector{P,Float64}}     # Penalty terms (p,N)

    CN::MVector{PN,Float64}       # Final constraint values (p_N,)
    CN_prev::MVector{PN,Float64}  # Previous final constraint values (p_N,)
    IμN::MMatrix{PN,PN,Float64,PPN}        # Final constraint penalty matrix (p_N,p_N)
    λN::MVector{PN,Float64}       # Final lagrange multipliers (p_N,)
    μN::MVector{PN,Float64}       # Final penalty terms (p_N,)

    Cx::Vector{MMatrix{P,N,Float64,NP}}
    Cu::Vector{MMatrix{P,M,Float64,MP}}
    Cx_N::MMatrix{PN,N,Float64,NPN}

    ρ::Array{Float64,1}
    dρ::Array{Float64,1}

    V_al_prev::Array{Float64,2} # Augmented Lagrangian Method update terms, see ALGENCAN notation
    V_al_current::Array{Float64,2} # Augmented Lagrangian Method update terms

    function ConstrainedStaticResults(X::Vector{MVector{N,Float64}},U::Vector{MVector{M,Float64}},
            K::Vector{MMatrix{M,N,Float64,NM}},b::Vector{MMatrix{M,M,Float64,MM}},d,X_,U_,S::Vector{MMatrix{N,N,Float64,NN}},s,fdx,fdu,fdv,fcx,fcu,dx,xm,um,
            C::Vector{MVector{P,Float64}},C_prev,Iμ::Vector{MMatrix{P,P,Float64,PP}},λ,μ,
            CN::MVector{PN,Float64},CN_prev,IμN::MMatrix{PN,PN,Float64,PPN},λN,μN,
            cx::Vector{MMatrix{P,N,Float64,NP}},cu::Vector{MMatrix{P,M,Float64,MP}},cxn::MMatrix{PN,N,Float64,NPN},ρ,dρ,V_al_prev,V_al_current) where {N,M,P,PN,NM,NN,MM,PP,PPN,NP,MP,NPN}
        # @show P
        # @show PN
        # @show typeof(cxn)
        new{N,M,P,PN,NM,NN,MM,PP,PPN,NP,MP,NPN}(X,U,K,b,d,X_,U_,S,s,fdx,fdu,fdv,fcx,fcu,dx,xm,um,C,C_prev,Iμ,λ,μ,CN,CN_prev,IμN,λN,μN,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)
    end
end

function ConstrainedStaticResults(n::Int,m::Int,p::Int,N::Int,p_N::Int=n)
    X  = [@MVector zeros(n)   for i = 1:N]
    U  = [@MVector zeros(m)   for i = 1:N]

    K  = [@MMatrix zeros(m,n) for i = 1:N]
    b  = [@MMatrix zeros(m,m) for i = 1:N]
    d  = [@MVector zeros(m)   for i = 1:N]

    X_ = [@MVector zeros(n)   for i = 1:N]
    U_ = [@MVector zeros(m)   for i = 1:N]
    S  = [@MMatrix zeros(n,n) for i = 1:N]
    s  = [@MVector zeros(n)   for i = 1:N]

    fdx = [@MMatrix zeros(n,n) for i = 1:N-1]
    fdu= [@MMatrix zeros(n,m) for i = 1:N-1]
    fdv = [@MMatrix zeros(n,m) for i = 1:N-1]
    fcx = [@MMatrix zeros(n,n) for i = 1:N]
    fcu = [@MMatrix zeros(n,m) for i = 1:N]

    dx = [@MVector zeros(n)   for i = 1:N]
    xm = [@MVector zeros(n)   for i = 1:N]
    um = [@MVector zeros(m)   for i = 1:N]

    # Stage Constraints
    C      = [@MVector zeros(p)  for i = 1:N]
    C_prev = [@MVector zeros(p)  for i = 1:N]
    Iμ     = [@MMatrix zeros(p,p) for i = 1:N]
    λ = [@MVector zeros(p)  for i = 1:N]
    μ     = [@MVector ones(p)   for i = 1:N]

    # Terminal Constraints (make 2D so it works well with stage values)
    C_N      = @MVector zeros(p_N)
    C_N_prev = @MVector zeros(p_N)
    Iμ_N     = @MMatrix zeros(p_N,p_N)
    λ_N      = @MVector zeros(p_N)
    μ_N      = @MVector ones(p_N)

    cx  = [@MMatrix zeros(p,n)   for i = 1:N]
    cu  = [@MMatrix zeros(p,m)   for i = 1:N]
    cxn = @MMatrix zeros(p_N,n)

    ρ = ones(1)
    dρ = ones(1)

    V_al_prev = zeros(p,N) #TODO preallocate only (pI,N)
    V_al_current = zeros(p,N)

    ConstrainedStaticResults(X,U,K,b,d,X_,U_,S,s,fdx,fdu,fdv,fcx,fcu,dx,xm,um,
        C,C_prev,Iμ,λ,μ,
        C_N,C_N_prev,Iμ_N,λ_N,μ_N,cx,cu,cxn,ρ,dρ,V_al_prev,V_al_current)

end

function copy(r::ConstrainedStaticResults)
    ConstrainedStaticResults(copy(r.X),copy(r.U),copy(r.K),copy(r.b),copy(r.d),copy(r.X_),copy(r.U_),copy(r.S),copy(r.s),copy(r.fdx),copy(r.fdu),copy(r.fdv),copy(r.fcx),copy(r.fcu),copy(r.dx),copy(r.xm),copy(r.um),
        copy(r.C),copy(r.C_prev),copy(r.Iμ),copy(r.λ),copy(r.μ),copy(r.CN),copy(r.CN_prev),copy(r.IμN),copy(r.λN),copy(r.μN),
        copy(r.Cx),copy(r.Cu),copy(r.Cx_N),copy(r.ρ),copy(r.dρ),copy(r.V_al_prev),copy(r.V_al_current))
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

"""
$(SIGNATURES)
    For infeasible solve, return a constrained results from a (special) unconstrained results along with AuLa constrained results
"""
function unconstrained_to_constrained_results(r::SolverIterResults,solver::Solver)::ConstrainedIterResults
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    pIs, pIc, pEs, pEsN, pEc = get_num_constraints(solver)
    # if solver.opts.use_static
    #     results = ConstrainedStaticResults(n,m̄,p,N,p_N,solver.control_integration)
    # else
    results = ConstrainedVectorResults(n,m̄,N,pIs, pIc, pEs, pEsN, pEc,solver.control_integration)
    # end
    copyto!(results.X,r.X)
    copyto!(results.X_,r.X_)
    copyto!(results.dx,r.dx)
    copyto!(results.xm,r.xm)
    copyto!(results.fcx,r.fcx)
    copyto!(results.fdx,r.fdx)
    copyto!(results.λs,r.λs)
    copyto!(results.λc,r.λc)
    copyto!(results.κs,r.κs)

    for k = 1:N
        results.U[k] = r.U[k][1:m̄]
        results.U_[k] = r.U_[k][1:m̄]
        results.fcu[k][1:n,1:m] = r.fcu[k][1:n,1:m]
        results.κc[k] = r.κc[k][n+1:n+solver.opts.minimum_time+solver.obj.pEc] # retain multipliers from all but infeasible and minimum time equality
        k == N ? continue : nothing
        results.um[k][1:m̄] = r.um[k][1:m̄]
        results.fdu[k][1:n,1:m̄] = r.fdu[k][1:n,1:m̄]
        results.fdv[k][1:n,1:m̄] = r.fdv[k][1:n,1:m̄]
    end

    results
end

function init_results(solver::Solver,X::AbstractArray,U::AbstractArray; prevResults=ConstrainedVectorResults())
    n,m,N = get_sizes(solver)

    if !isempty(X)
        solver.opts.infeasible = true
    end

    # Generate initial trajectoy (tacking on infeasible and minimum time controls)
    X_init, U_init = get_initial_trajectory(solver, X, U)

    if solver.opts.constrained
        # Get sizes
        pIs, pIc, pEs,pEsN, pEc = get_num_constraints(solver)
        m̄,mm = get_num_controls(solver)

        # if solver.opts.use_static
        #     results = ConstrainedStaticResults(n,mm,p,N,n,solver.control_integration)
        # else
        results = ConstrainedVectorResults(n,mm,N,pIs,pIc,pEs,pEsN,pEc,solver.control_integration)
        # end

        # Set initial penalty term values
        results.μs .*= solver.opts.μ_initial # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float
        results.μc .*= solver.opts.μ_initial # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float
        results.νs .*= solver.opts.μ_initial # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float
        results.νc .*= solver.opts.μ_initial # TODO change to assign, not multiply: μ_initial needs to be initialized as an array instead of float
        # Special penalty initializations
        if solver.opts.minimum_time
            solver.opts.infeasible ? idx = n+1 : idx = 1
            for k = 1:solver.N
                results.νc[k][idx] = solver.opts.μ_initial_minimum_time_equality
                results.μc[k][m̄] = solver.opts.μ_initial_minimum_time_inequality
                results.μc[k][m̄+m̄] = solver.opts.μ_initial_minimum_time_inequality
            end
        end
        if solver.opts.infeasible
            nothing #TODO
        end

        # Initialize Lagrange multipliers (warm start)
        if ~isempty(prevResults)
            results.λs .= deepcopy(λs)
            results.λc .= deepcopy(λc)
            results.κs .= deepcopy(κs)

            # remove infeasible control multipliers
            for k = 1:N
                results.κc[k] = κc[n+1:n+solver.opts.minimum_time+solver.obj.pEc]
            end
        end

        # Set initial regularization
        results.ρ[1] = solver.opts.ρ_initial

    else
        # if solver.opts.use_static
        #     results = UnconstrainedStaticResults(n,m,N,solver.control_integration)
        # else
        results = UnconstrainedVectorResults(n,m,N,solver.control_integration)
        # end
    end
    copyto!(results.X, X_init)
    copyto!(results.U, U_init)
    return results
end

# function copy_λ!(solver, results, λ)
#     p_new = length(λ[1])
#     p, = get_num_constraints(solver)
#     if p_new == p  # all constraint λs passed in
#         cid = trues(p)
#     elseif p_new == solver.obj.p  # only "original" constraint λs passed
#         cid = original_constraint_inds(solver)
#     else
#         err = ArgumentError("λ is not the correct dimension ($p_new). It must be either size $p or $(solver.obj.p)")
#         throw(err)
#     end
#     for k = 1:N
#         results.λ[k][cid] = λ[k]
#     end
#     results.λN .= λ[N+1]
# end



"""
$(SIGNATURES)
    For infeasible solve, return an unconstrained results from a prior unconstrained or constrained results
        -removes infeasible controls and infeasible components in Jacobians
        -additionally, we need an unconstrained problem (temporarily) to project into the feasible space
"""
function remove_infeasible_controls_to_unconstrained_results(r::SolverIterResults,solver::Solver)::UnconstrainedIterResults
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # if solver.opts.use_static
    #     results = UnconstrainedStaticResults(n,m̄,N,solver.control_integration)
    # else
        results = UnconstrainedVectorResults(n,m̄,N,solver.control_integration)
    # end
    copyto!(results.X,r.X)
    copyto!(results.X_,r.X_)
    copyto!(results.dx,r.dx)
    copyto!(results.xm,r.xm)
    copyto!(results.fcx,r.fcx)
    copyto!(results.fdx,r.fdx)
    for k = 1:N
        results.U[k] = r.U[k][1:m̄]
        results.U_[k] = r.U_[k][1:m̄]
        results.fcu[k][1:n,1:m] = r.fcu[k][1:n,1:m]
        k == N ? continue : nothing
        results.um[k] = r.um[k][1:m̄]
        results.fdu[k][1:n,1:m̄] = r.fdu[k][1:n,1:m̄]
        results.fdv[k][1:n,1:m̄] = r.fdv[k][1:n,1:m̄]
    end
    results
end
