
#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

"""
$(TYPEDEF)
Cost function of the form
    (xₙ-xf)ᵀ Qf (xₙ-xf) + ∫ ( (x-xf)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct LQRCost{TQ,TR,TQf} <: CostFunction
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    Qf::TQf               # Quadratic final cost for terminal state (n,n)
    xf::Vector{Float64}

    function LQRCost(Q::TQ, R::TR, Qf::TQf, xf::Vector{Float64}) where {TQ,TR,TQf}
        if !isposdef(R)
            err = ArgumentError("R must be positive definite")
            throw(err)
        end
        if !ispossemidef(Q)
            err = ArgumentError("Q must be positive semi-definite")
            throw(err)
        end
        if !ispossemidef(Q)
            err = ArgumentError("Qf must be positive semi-definite")
            throw(err)
        end
        new{TQ,TR,TQf}(Q,R,Qf,xf)
    end
end

function taylor_expansion(cost::LQRCost, x::Vector{Float64}, u::Vector{Float64})
    Q = cost.Q
    R = cost.R
    H = zeros(size(Q,1),size(R,1))
    q = Q*(x-cost.xf)
    r = R*u
    return Q,R,H,q,r
end

function taylor_expansion(cost::LQRCost, xN::Vector{Float64})
    return Qf,qf
end

"""
$(TYPEDEF)
Cost function of the form
    (xₙ-xf)ᵀ Qf (xₙ-xf) + qfᵀxₙ + ∫ ( xᵀQx + uᵀRu + q⁠ᵀx + rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct LinearQuadraticCost{TM,TV} <: CostFunction
    Q::TM                 # Quadratic stage cost for states (n,n)
    R::TM                 # Quadratic stage cost for controls (m,m)
    H::TM                 # Quadratic Cross-coupling for state and controls (n,m)
    q::TV                 # Linear term on states (n,)
    r::TV                 # Lineqr term on controls (m,)
    Qf::TM                # Quadratic final cost for terminal state (n,n)
    qf::TV                # Linear term on terminal state (n,)
    function LinearQuadraticCost(Q::TM, R::TM, H::TM, q::TV, r::TV, Qf::TM, qf::TV) where {TM,TV}
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
        new{TM,TV}(Q,R,H,q,r,Qf,qf)
    end
end

struct GenericCost <: CostFunction
    ℓ::Function
    GenericCost(ℓ::Function) = new(ℓ)
end
