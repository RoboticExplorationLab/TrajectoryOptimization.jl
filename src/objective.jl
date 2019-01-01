
#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

"""
$(TYPEDEF)
Cost function of the form
    xₙᵀ Qf xₙ + qfᵀxₙ + ∫ ( xᵀQx + uᵀRu + q⁠ᵀx + rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct LinearQuadraticCost{TM,TH,TV} <: CostFunction
    Q::TM                 # Quadratic stage cost for states (n,n)
    R::TM                 # Quadratic stage cost for controls (m,m)
    H::TH                 # Quadratic Cross-coupling for state and controls (n,m)
    q::TV                 # Linear term on states (n,)
    r::TV                 # Lineqr term on controls (m,)
    Qf::TM                # Quadratic final cost for terminal state (n,n)
    qf::TV                # Linear term on terminal state (n,)
    function LinearQuadraticCost(Q::TM, R::TM, H::TH, q::TV, r::TV, Qf::TM, qf::TV) where {TM, TH, TV}
        if !isposdef(R)
            err = ArgumentError("R must be positive definite")
            throw(err)
        end
        if !ispossemidef(Q)
            err = ArgumentError("Q must be positive semi-definite")
            throw(err)
        end
        if !ispossemidef(Qf)
            err = ArgumentError("Qf must be positive semi-definite")
            throw(err)
        end
        new{TM,TH,TV}(Q,R,H,q,r,Qf,qf)
    end
end

function LQRCost(Q,R,Qf,xf)
    H = zeros(size(Q,1),size(R,1))
    q = -Q*xf
    r = zeros(size(R,1))
    qf = -Qf*xf
    return LinearQuadraticCost(Q, R, H, q, r, Qf, qf)
end

function taylor_expansion(cost::LinearQuadraticCost, x::Vector{Float64}, u::Vector{Float64})
    return cost.Q, cost.R, cost.H, cost.Q*x + cost.q, cost.R*u + cost.r
end

function taylor_expansion(cost::LinearQuadraticCost, xN::Vector{Float64})
    return cost.Qf, cost.Qf*xN + cost.qf
end

function stage_cost(cost::LinearQuadraticCost, x::Vector{Float64}, u::Vector{Float64})
    x'cost.Q*x + u'*cost.R*u + cost.q'x + cost.r'u
end

function stage_cost(cost::LinearQuadraticCost, xN::Vector{Float64})
    xN'cost.Qf*xN + cost.qf'*xN
end


"""
$(TYPEDEF)
Cost function of the form
    ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
"""
struct GenericCost <: CostFunction
    ℓ::Function
    ℓ_aug::Function
    ℓf::Function
    n::Int
    m::Int
    inds::NamedTuple
    hess::Matrix{Float64}  # Pre-allocated hessian
    grad::Vector{Float64}  # Pre-allocated gradient
    z::Vector{Float64}     # Pre-allocated augmented state and control

    function GenericCost(ℓ::Function, ℓf::Function, n::Int, m::Int)
        linds = LinearIndices(zeros(n+m,n+m))
        xinds = 1:n
        uinds = n .+(1:m)
        inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], xu=linds[xinds,uinds])
        function ℓ_aug(z)
            x = view(z,xinds)
            u = view(z,uinds)
            ℓ(x,u)
        end
        hess = zeros(n+m,n+m)
        grad = zeros(n+m)
        z = zeros(n+m)
        new(ℓ,ℓ_aug,ℓf,n,m,inds,hess,grad,z)
    end
end

function taylor_expansion(cost::GenericCost, x::Vector{Float64}, u::Vector{Float64})
    inds = cost.inds
    z = cost.z
    z[inds.x] = x
    z[inds.u] = u
    ForwardDiff.gradient!(cost.grad, cost.ℓ_aug, z)
    ForwardDiff.hessian!(cost.hess, cost.ℓ_aug, z)

    q = view(cost.grad,inds.x)
    r = view(cost.grad,inds.u)
    Q = view(cost.hess,inds.xx)
    R = view(cost.hess,inds.uu)
    H = view(cost.hess,inds.xu)
    return Q,R,H,q,r
end

function taylor_expansion(cost::GenericCost, xN::Vector{Float64})
    qf = ForwardDiff.gradient(cost.ℓf, xN)
    Qf = ForwardDiff.hessian(cost.ℓf, xN)
    return Qf,qf
end

stage_cost(cost::GenericCost, x::Vector{Float64}, u::Vector{Float64}) = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::Vector{Float64}) = cost.ℓf(xN)


"""
$(TYPEDEF)
Defines an objective for an unconstrained optimization problem.
xf does not have to specified. It is provided for convenience when used as part of the cost function (see LQRObjective function)
If tf = 0, the objective is assumed to be minimum-time.
"""
struct UnconstrainedObjectiveNew{C} <: Objective
    cost::C
    tf::Float64          # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Vector{Float64}  # Initial state (n,)
    xf::Vector{Float64}  # (optional) Final state (n,)

    function UnconstrainedObjectiveNew(cost::C,tf::Float64,x0,xf=Float64[]) where {C}
        if !isempty(xf) && length(xf) != length(x0)
            throw(ArgumentError("x0 and xf must be the same length"))
        end
        new{C}(cost,tf,x0,xf)
    end
end

"""
$(SIGNATURES)
Minimum time constructor for unconstrained objective
"""
function UnconstrainedObjectiveNew(cost::CostFunction,tf::Symbol,x0,xf)
    if tf == :min
        UnconstrainedObjectiveNew(cost,0.0,x0,xf)
    else
        err = ArgumentError(":min is the only recognized Symbol for the final time")
        throw(err)
    end
end





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                      CONSTRAINED OBJECTIVE                                   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

function is_inplace_constraint(f::Function,n)
    q = 100
    while true
        c = zeros(q)
        try
            f(c,rand(n))
            return true
        catch e
            if e isa MethodError
                return false
            elseif e isa BoundsError
                q *= 10
            else
                throw(e)
            end
        end
    end
end

function count_inplace_constraint(f::Function,n)
    q = 100
    while true
        c = zeros(q)*Inf
        try
            f(c,rand(n))
            return count(isfinite.(c))
        catch e
            if e isa BoundsError
                q *= 10
            else
                throw(e)
            end
        end
    end
end

struct ConstrainedObjectiveNew{C} <: Objective
    cost::C
    tf::Float64           # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)

    # Control Constraints
    u_min::Array{Float64,1}  # Lower control bounds (m,)
    u_max::Array{Float64,1}  # Upper control bounds (m,)

    # State Constraints
    x_min::Array{Float64,1}  # Lower state bounds (n,)
    x_max::Array{Float64,1}  # Upper state bounds (n,)

    # Custom constraints
    gs::Function    # inequality state constraint
    gc::Function    # inequality control constraint
    hs::Function    # equality state constraint
    hc::Function    # inequality control constraint

    # Terminal Constraints
    gs_N::Function  # terminal inequality state constraint
    use_terminal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false) # TODO I don't think this is used

    function ConstrainedObjectiveNew(cost::C,tf::Float64,x0,xf,
        u_min, u_max,
        x_min, x_max,
        gs, gc,
        hs, hc,
        gs_N, use_terminal_constraint) where {C}

        n = size(Q,1)
        m = size(R,1)

        # Make general inequality/equality constraints inplace
        if !is_inplace_constraint(gs,n)
            gs = wrap_inplace(gs)
            println("Custom state inequality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end

        if !is_inplace_constraint(gc,n)
            gc = wrap_inplace(gc)
            println("Custom control inequality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end

        if !is_inplace_constraint(hs,n)
            hs = wrap_inplace(hs)
            println("Custom state equality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end

        if !is_inplace_constraint(hc,n)
            hc = wrap_inplace(hc)
            println("Custom control equality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end


        new{C}(cost::C, tf,x0,xf, u_min, u_max, x_min, x_max, gs, gc, hs, hc, gs_N, use_terminal_constraint)
    end
end




"""
$(SIGNATURES)
Create unconstrained objective for a problem of the form:
    min (xₙ - xf)ᵀ Qf (xₙ - xf) + ∫ ( (x-xf)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
    s.t. x(0) = x0
         x(tf) = xf
"""
function LQRObjective(Q,R,Qf,tf,x0,xf)
    cost = LQRCost(Q,R,Qf,xf)
    UnconstrainedObjectiveNew(cost,tf,x0,xf)
end


"""
$(SIGNATURES)
Convenience method for getting the stage cost from any objective
"""
function stage_cost(obj::Objective,x::Vector{Float64},u::Vector{Float64})
    stage_cost(obj.cost,x,u)
end
