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

# In-place cost-expansion
function cost_expansion!(E::AbstractExpansion, cost::CostFunction, z::KnotPoint)
    cost_gradient!(E, cost, z)
    cost_hessian!(E, cost, z)
    E.x0 .= E.x  # copy cost-only gradient
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
function cost_gradient!(E, cost::CostFunction, z::KnotPoint)
    gradient!(E, cost, state(z), control(z))
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
	E.x .*= dt_x
	E.u .*= dt_u
    return nothing
end

"```
Qxx,Quu,Qux = cost_hessian(cost::CostFunction, z::KnotPoint)
```
Get Qxx, Quu, Qux pieces of Hessian of cost function, multiplied by dt"
function cost_hessian!(E, cost::CostFunction, z::KnotPoint)
    hessian!(E, cost, state(z), control(z))
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
    return nothing
end


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
        cost_gradient!(E, obj[k], Z[k])
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
        cost_hessian(E[k], obj[k], Z[k])
    end
end

"$(TYPEDSIGNATURES) Expand cost for entire trajectory"
function cost_expansion!(E, obj::Objective, Z::Traj)
    cost_gradient!(E, obj, Z)
    cost_hessian!(E, obj, Z)
end

"""
Compute the error expansion along the entire trajectory
	Assumes the cost expansion is already complete
"""
@inline error_expansion!(E::Vector{<:SizedCostExpansion}, model::AbstractModel, Z::Traj, G) = nothing
function error_expansion!(E::Vector{<:SizedCostExpansion}, model::RigidBody, Z::Traj, G)
    for k in eachindex(E)
        error_expansion!(E[k], model, Z[k], G[k])
    end
end

"Compute the error expansion at a single KnotPoint (only for state vectors not in Vector space)"
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

"Copy the error expansion to another expansion"
function error_expansion!(E::AbstractExpansion, Q::SizedCostExpansion)
	E.x  .= Q.x
	E.u  .= Q.u
	E.xx .= Q.xx
	E.uu .= Q.uu
	E.ux .= Q.ux
end

"Get the error expansion"
@inline function error_expansion(E::SizedCostExpansion, model::RigidBody)
	return StaticExpansion(E.x_, E.xx_, E.u_, E.uu_, E.ux_)
end

"Get the error expansion (same as cost expansion)"
@inline function error_expansion(E::SizedCostExpansion, model::AbstractModel)
	return StaticExpansion(E.x, E.xx, E.u, E.uu, E.ux)
end

"Get cost expansion"
@inline function cost_expansion(E::SizedCostExpansion{<:Any,N,N}) where N
	return StaticExpansion(E.x, E.xx, E.u, E.uu, E.ux)
end
