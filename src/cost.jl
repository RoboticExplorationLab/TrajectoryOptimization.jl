export
    cost

############################################################################################
#                              COST METHODS                                                #
############################################################################################

"$(TYPEDSIGNATURES) Evaluate the cost at a knot point, and automatically handle terminal
knot point, multiplying by dt as necessary."
function stage_cost(cost::CostFunction, z::KnotPoint)
    if is_terminal(z)
        stage_cost(cost, state(z))
    else
        stage_cost(cost, state(z), control(z))*z.dt
    end
end

# Default to no integration
cost(obj, dyn_con::DynamicsConstraint{Q}, Z) where Q<:QuadratureRule = cost(obj, Z)

"Evaluate the cost for a trajectory (non-allocating)"
@inline function cost!(obj::Objective, Z::Traj)
    map!(stage_cost, obj.J, obj.cost, Z)
end

# function cost_gradient(cost::CostFunction, model::AbstractModel, z::KnotPoint, G=I)
#     Qx,Qu = gradient(cost, state(z), control(z))
#     return G'Qx, Qu
# end
#
# function cost_hessian(cost::CostFunction, model::AbstractModel, z::KnotPoint, G=I)
#     Qxx,Quu,Qux = hessian(cost, state(z), control(z))
#     return G'Qxx*G
# end

function cost_expansion(cost::CostFunction, model::AbstractModel, z::KnotPoint, G=I)
    Qx,Qu = gradient(cost, state(z), control(z))
    Qxx,Quu,Qux = hessian(cost, state(z), control(z))
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
    Qx,Qu = Qx*dt_x, Qu*dt_u
    Qxx,Quu,Qux = Qxx*dt_x, Quu*dt_u, Qux*dt_u

    Qux = Qux*G
    Qx = G'Qx
    Qxx = G'Qxx*G + ∇²differential(model, state(z), Qx) #- Diagonal(idq)*(Qx'Diagonal(iq)*state(z))
    return Qxx, Quu, Qux, Qx, Qu
end

function cost_expansion!(E, G, obj::Objective, model::AbstractModel, Z::Traj)
    for k in eachindex(Z)
        z = Z[k]
        E.xx[k], E.uu[k], E.ux[k], E.x[k], E.u[k] =
            cost_expansion(obj.cost[k], model, z, G[k])
    end
end

# In-place cost-expansion
function cost_expansion!(E::AbstractExpansion, cost::CostFunction, z::KnotPoint)
    gradient!(E, cost, state(z), control(z))
    hessian!(E, cost, state(z), control(z))
    E.x0 .= E.x  # copy cost-only gradient
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
    E.xx .*= dt_x
    E.uu .*= dt_u
    E.ux .*= dt_u
    E.x  .*= dt_x
    E.u  .*= dt_u
    return nothing
end

function cost_expansion!(E::Vector{<:AbstractExpansion}, obj::Objective, Z::Traj)
    for k in eachindex(Z)
        z = Z[k]
        cost_expansion!(E[k], obj.cost[k], z)
    end
end

"```
Qx,Qu = cost_gradient(cost::CostFunction, z::KnotPoint)
```
Get Qx, Qu pieces of gradient of cost function, multiplied by dt"
function cost_gradient(cost::CostFunction, z::KnotPoint)
    Qx, Qu = gradient(cost, state(z), control(z))
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
    return Qx*dt_x, Qu*dt_u
end

"```
Qxx,Quu,Qux = cost_hessian(cost::CostFunction, z::KnotPoint)
```
Get Qxx, Quu, Qux pieces of Hessian of cost function, multiplied by dt"
function cost_hessian(cost::CostFunction, z::KnotPoint)
    Qxx, Quu, Qux = hessian(cost, state(z), control(z))
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
    return Qxx*dt_x, Quu*dt_u, Qux*dt_u
end

# "Calculate the 2nd order expansion of the cost at a knot point"
# cost_expansion(cost::CostFunction, z::KnotPoint) =
#     cost_gradient(cost, z)..., cost_hessian(cost, z)...


"""```
cost_gradient!(E::CostExpansion, obj::Objective, Z::Traj)
cost_gradient!(E::CostExpansion, obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::Traj)
```
Calculate the cost gradient for an entire trajectory. If a dynamics constraint is given,
    use the appropriate integration rule, if defined.
"""
function cost_gradient!(E, obj::Objective, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.x[k], E.u[k] = cost_gradient(obj[k], Z[k])
    end
end

# Default to no integration
cost_gradient!(E, obj, dyn_con::DynamicsConstraint{Q}, Z) where Q = cost_gradient!(E, obj, Z)

"```
cost_hessian!(E::CostExpansion, obj::Objective, Z::Traj)
```
 Calculate the cost Hessian for an entire trajectory"
function cost_hessian!(E, obj::Objective, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.xx[k], E.uu[k], E.ux[k] = cost_hessian(obj[k], Z[k])
    end
end

"$(TYPEDSIGNATURES) Expand cost for entire trajectory"
function cost_expansion!(E, obj::Objective, Z::Traj)
    cost_gradient!(E, obj, Z)
    cost_hessian!(E, obj, Z)
end


# "Calculate the error cost expansion"
# function error_expansion!(E::AbstractExpansion, Q::AbstractExpansion, model::AbstractModel, z::KnotPoint, G)
#     ∇²differential!(E.xx, model, state(z), Q.x)
#     # E.tmp .= G
#     return _error_expansion!(E, Q, G)
# end
#
# function _error_expansion!(E::AbstractExpansion, Q::AbstractExpansion, G)
#     E.u .= Q.u
#     E.uu .= Q.uu
#     mul!(E.ux, Q.ux, G)
#     mul!(E.x, Transpose(G), Q.x)
#     mul!(E.tmp, Q.xx, G)
#     mul!(E.xx, Transpose(G), E.tmp, 1.0, 1.0)
# end
# @inline _error_expansion!(E::AbstractExpansion, Q::AbstractExpansion, G::UniformScaling) = copyto!(E,Q)

"""
Assumes the cost expansion is already complete
"""
@inline error_expansion!(E::Vector{<:SizedCostExpansion}, model::AbstractModel, Z::Traj, G) = nothing
function error_expansion!(E::Vector{<:SizedCostExpansion}, model::RigidBody, Z::Traj, G)
    for k in eachindex(E)
        error_expansion!(E[k], model, Z[k], G[k])
    end
end

function error_expansion!(E::SizedCostExpansion{<:Any,N}, model::RigidBody, z, G) where N
    if N < 15
        G = SMatrix(G)
        E.u_  .= E.u
        E.uu_ .= E.uu
        E.ux_ .= SMatrix(E.ux)*G
        E.x_  .= G'*SVector(E.x)

        ∇²differential!(E.xx_, model, state(z), E.x0)
        E.xx_ .+= G'E.xx*G
    else
        E.u_ .= E.u
        E.uu_ .= E.uu
        mul!(E.ux_, E.ux, G)
        mul!(E.x_, Transpose(G), E.x)

        ∇²differential!(E.xx_, model, state(z), E.x0)
        mul!(E.tmp, E.xx, G)
        mul!(E.xx_, Transpose(G), E.tmp, 1.0, 1.0)
    end
end
