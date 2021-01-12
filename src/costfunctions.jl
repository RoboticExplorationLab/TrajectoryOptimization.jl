import Base: copy, +

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

"""
Abstract type that represents a scalar-valued function that accepts a state and control
at a single knot point.
"""
abstract type CostFunction end

import RobotDynamics: diffmethod, ForwardAD, FiniteDifference
diffmethod(::CostFunction) = ForwardAD()

# Automatic differentiation methods for generic costs
@inline function gradient!(E, cost::CostFunction, z::AbstractKnotPoint, cache=ExpansionCache(cost))
    _gradient!(diffmethod(cost), E, cost, z, cache)
end

@inline function hessian!(E, cost::CostFunction,  z::AbstractKnotPoint, cache=ExpansionCache(cost))
    _hessian!(diffmethod(cost), E, cost, z, cache) 
end

# ForwardDiff methods
function _gradient!(::ForwardAD, E, cost::CostFunction, z::AbstractKnotPoint, cache=nothing)
    if is_terminal(z)
        costfun_term(x) = stage_cost(cost, x)
        ForwardDiff.gradient!(E.x, costfun_term, state(z))
    else
        ix,iu = z._x, z._u
        costfun(z) = stage_cost(cost, z[ix], z[iu]) 
        ForwardDiff.gradient!(E.grad, costfun, z.z)
    end
    return false 
end

function _hessian!(::ForwardAD, E, cost::CostFunction, z::AbstractKnotPoint, cache=nothing)
    if is_terminal(z)
        costfun_term(x) = stage_cost(cost, x)
        ForwardDiff.hessian!(E.xx, costfun_term, state(z))
    else
        ix,iu = z._x, z._u
        costfun(z) = stage_cost(cost, z[ix], z[iu]) 
        ForwardDiff.hessian!(E.hess, costfun, z.z)
    end
    return false
end

# FiniteDiff methods
function _gradient!(::FiniteDifference, E, cost::CostFunction, z::AbstractKnotPoint, 
        cache=ExpansionCache(cost))
    if is_terminal(z)
        cache = cache[3]
        costfun_term(x) = stage_cost(cost, x)
        cache.c3 .= state(z) 
        FiniteDiff.finite_difference_gradient!(E.x.data, costfun_term, cache.c3, cache)
    else
        cache = cache[1]
        ix,iu = z._x, z._u
        costfun(z) = stage_cost(cost, z[ix], z[iu])
        cache.c3 .= z.z 
        FiniteDiff.finite_difference_gradient!(E.grad, costfun, cache.c3, cache)
    end
    return false 
end

function _hessian!(::FiniteDifference, E, cost::CostFunction, z::AbstractKnotPoint, 
        cache=ExpansionCache(cost))
    if is_terminal(z)
        cache = cache[4]
        costfun_term(x) = stage_cost(cost, x)
        x = state(z)
        cache.xmm .= x
        cache.xmp .= x
        cache.xpm .= x
        cache.xpp .= x
        FiniteDiff.finite_difference_hessian!(E.xx.data, costfun_term, cache.xmm, cache)
    else
        cache = cache[2]
        ix,iu = z._x, z._u
        costfun(z) = stage_cost(cost, z[ix], z[iu])
        cache.xmm .= z.z 
        cache.xmp .= z.z 
        cache.xpm .= z.z 
        cache.xpp .= z.z 
        FiniteDiff.finite_difference_hessian!(E.hess, costfun, cache.xmm, cache)
    end
    return false
end

"""
    ExpansionCache(cost)

Allocate the cache needed to evaluate the gradient and Hessian of the cost function.
Returns a 4-element vector with the following elements: `[grad, hess, grad_term, hess_term]`,
which are the caches need for the gradient and Hessian for the state and terminal cost
functions.
"""
@inline ExpansionCache(cost::CostFunction) = 
    ExpansionCache(diffmethod(cost), state_dim(cost), control_dim(cost))
@inline ExpansionCache(::ForwardAD, n::Int, m::Int) = (nothing,nothing,nothing,nothing)
function ExpansionCache(::FiniteDifference, n::Int, m::Int) 
    grad_cache0 = FiniteDiff.GradientCache(zeros(n), zeros(n), Val(:forward))
    hess_cache0 = FiniteDiff.HessianCache(zeros(n))
    grad_cache  = FiniteDiff.GradientCache(zeros(n+m), zeros(n+m), Val(:forward))
    hess_cache  = FiniteDiff.HessianCache(zeros(n+m))
    return (grad_cache, hess_cache, grad_cache0, hess_cache0)
end
