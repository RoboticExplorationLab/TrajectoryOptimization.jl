# export
#     cost

############################################################################################
#                              COST METHODS                                                #
############################################################################################

"$(TYPEDSIGNATURES) Evaluate the cost at a knot point, and automatically handle terminal
knot point, multiplying by dt as necessary."
function stage_cost(cost::CostFunction, z::AbstractKnotPoint)
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


############################################################################################
#                              COST EXPANSIONS                                             #
############################################################################################

function cost_gradient!(E::Objective, obj::Objective, Z::Traj, init::Bool=false)
    is_const = E.const_grad
    N = length(Z)
    for k in eachindex(Z)
        if init || !is_const[k]
			if is_terminal(Z[k])
            	is_const[k] = gradient!(E.cost[k], obj.cost[k], state(Z[k]))
			else
            	is_const[k] = gradient!(E.cost[k], obj.cost[k], state(Z[k]), control(Z[k]))
			end
            dt_x = k < N ? Z[k].dt :  one(Z[k].dt)
            dt_u = k < N ? Z[k].dt : zero(Z[k].dt)
            E[k].q .*= dt_x
            E[k].r .*= dt_u
        end
    end
end

function cost_hessian!(E::Objective, obj::Objective, Z::Traj, init::Bool=false)
    is_const = E.const_hess
    N = length(Z)
    for k in eachindex(Z)
        if init || !is_const[k]
			if is_terminal(Z[k])
            	is_const[k] = hessian!(E.cost[k], obj.cost[k], state(Z[k]))
			else
            	is_const[k] = hessian!(E.cost[k], obj.cost[k], state(Z[k]), control(Z[k]))
			end
            dt_x = k < N ? Z[k].dt :  one(Z[k].dt)
            dt_u = k < N ? Z[k].dt : zero(Z[k].dt)
            E[k].Q .*= dt_x
            E[k].R .*= dt_u
            E[k].H .*= dt_u
        end
    end
end

function cost_expansion!(E::Objective, obj::Objective, Z::Traj, init::Bool=false)
    cost_gradient!(E, obj, Z, init)
    cost_hessian!(E, obj, Z, init)
    return nothing
end

function error_expansion!(E::Objective, Jexp::Objective, model::AbstractModel, Z::Traj, G, tmp=G[end])
    @assert E === Jexp "E and Jexp should be the same object for AbstractModel"
    return nothing
end

function error_expansion!(E::Objective, Jexp::Objective, model::LieGroupModel, Z::Traj, G, tmp=G[end])
    for k in eachindex(E.cost)
        error_expansion!(E.cost[k], Jexp.cost[k], model, Z[k], G[k], tmp)
    end
end

function error_expansion!(E::QuadraticCost, cost::QuadraticCost, model, z::AbstractKnotPoint,
        G, tmp)
    RobotDynamics.∇²differential!(E.Q, model, state(z), cost.q)
    if size(model)[1] < 15
        G = SMatrix(G)
        E.H .= SMatrix(cost.H) * G
        E.q .= G'cost.q
        E.Q .+= G'cost.Q*G
    else
        mul!(E.H, cost.H, G)
        mul!(E.q, Transpose(G), cost.q)
        mul!(tmp, cost.Q, G)
        mul!(E.Q, Transpose(G), tmp, 1.0, 1.0)
    end
    return nothing
end

struct StaticExpansion{T,N,M,NN,MM,NM} <: AbstractExpansion{T}
	x::SVector{N,T}
	xx::SMatrix{N,N,T,NN}
	u::SVector{M,T}
	uu::SMatrix{M,M,T,MM}
	ux::SMatrix{M,N,T,NM}
end

function StaticExpansion(E::AbstractExpansion)
	StaticExpansion(SVector(E.x), SMatrix(E.xx),
		SVector(E.u), SMatrix(E.uu), SMatrix(E.ux))
end

function StaticExpansion(x,xx,u,uu,ux)
	StaticExpansion(SVector(x), SMatrix(xx), SVector(u), SMatrix(uu), SMatrix(ux))
end

function static_expansion(cost::QuadraticCost)
	StaticExpansion(cost.q, cost.Q, cost.r, cost.R, cost.H)
end

"""
	dgrad(E::QuadraticExpansion, dZ::Traj)
Calculate the derivative of the cost in the direction of `dZ`, where `E` is the current
quadratic expansion of the cost.
"""
function dgrad(E::QuadraticExpansion{n,m,T}, dZ::Traj)::T where {n,m,T}
	g = zero(T)
	N = length(E)
	for k = 1:N-1
		g += dot(E[k].q, state(dZ[k])) + dot(E[k].r, control(dZ[k]))
	end
	g += dot(E[N].q, state(dZ[N]))
	return g
end

"""
	dhess(E::QuadraticCost, dZ::Traj)
Calculate the scalar 0.5*dZ'G*dZ where G is the hessian of cost
"""
function dhess(E::QuadraticExpansion{n,m,T}, dZ::Traj)::T where {n,m,T}
	h = zero(T)
	N = length(E)
	for k = 1:N-1
		x = state(dZ[k])
		u = control(dZ[k])
		h += dot(x, E[k].Q, x) + dot(u, E[k].R, u)
	end
	x = state(dZ[N])
	h += dot(x, E[N].Q, x)
	return 0.5*h
end

"""
	norm_grad(E::QuadraticExpansion, p=2)
Norm of the cost gradient
"""
function norm_grad(E::QuadraticExpansion{n,m,T}, p=2)::T where {n,m,T}
	J = get_J(E)
	for (k,cost) in enumerate(E)
		J[k] = norm(cost.q, p)
		if !cost.terminal
			J[k] = norm(SA[norm(cost.r, p) J[k]], p)
		end
	end
	return norm(J, p)
end

# # In-place cost-expansion
# function cost_expansion!(E::AbstractExpansion, cost::CostFunction, z::KnotPoint)
#     cost_gradient!(E, cost, z)
#     cost_hessian!(E, cost, z)
#     E.x0 .= E.x  # copy cost-only gradient
#     return nothing
# end
#
# function cost_expansion!(E::Vector{<:AbstractExpansion}, obj::Objective, Z::Traj)
#     for k in eachindex(Z)
#         z = Z[k]
#         cost_expansion!(E[k], obj.cost[k], z)
#     end
# end
#
# "```
# Qx,Qu = cost_gradient(cost::CostFunction, z::KnotPoint)
# ```
# Get Qx, Qu pieces of gradient of cost function, multiplied by dt"
# function cost_gradient!(E, cost::CostFunction, z::KnotPoint)
#     gradient!(E, cost, state(z), control(z))
#     if is_terminal(z)
#         dt_x = 1.0
#         dt_u = 0.0
#     else
#         dt_x = z.dt
#         dt_u = z.dt
#     end
# 	E.x .*= dt_x
# 	E.u .*= dt_u
#     return nothing
# end
#
# "```
# Qxx,Quu,Qux = cost_hessian(cost::CostFunction, z::KnotPoint)
# ```
# Get Qxx, Quu, Qux pieces of Hessian of cost function, multiplied by dt"
# function cost_hessian!(E, cost::CostFunction, z::KnotPoint)
#     hessian!(E, cost, state(z), control(z))
#     if is_terminal(z)
#         dt_x = 1.0
#         dt_u = 0.0
#     else
#         dt_x = z.dt
#         dt_u = z.dt
#     end
# 	E.xx .*= dt_x
# 	E.uu .*= dt_u
# 	E.ux .*= dt_u
#     return nothing
# end
#
#
# """```
# cost_gradient!(E::CostExpansion, obj::Objective, Z::Traj)
# cost_gradient!(E::CostExpansion, obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::Traj)
# ```
# Calculate the cost gradient for an entire trajectory. If a dynamics constraint is given,
#     use the appropriate integration rule, if defined.
# """
# function cost_gradient!(E, obj::Objective, Z::Traj)
#     N = length(Z)
#     for k in eachindex(Z)
#         cost_gradient!(E, obj[k], Z[k])
#     end
# end
#
# # Default to no integration
# cost_gradient!(E, obj, dyn_con::DynamicsConstraint{Q}, Z) where Q = cost_gradient!(E, obj, Z)
#
# "```
# cost_hessian!(E::CostExpansion, obj::Objective, Z::Traj)
# ```
#  Calculate the cost Hessian for an entire trajectory"
# function cost_hessian!(E, obj::Objective, Z::Traj)
#     N = length(Z)
#     for k in eachindex(Z)
#         cost_hessian(E[k], obj[k], Z[k])
#     end
# end
#
# "$(TYPEDSIGNATURES) Expand cost for entire trajectory"
# function cost_expansion!(E, obj::Objective, Z::Traj)
#     cost_gradient!(E, obj, Z)
#     cost_hessian!(E, obj, Z)
# end
#
# """
# Compute the error expansion along the entire trajectory
# 	Assumes the cost expansion is already complete
# """
# @inline error_expansion!(E::Vector{<:CostExpansion}, model::AbstractModel, Z::Traj, G) = nothing
# function error_expansion!(E::Vector{<:CostExpansion}, model::RigidBody, Z::Traj, G)
#     for k in eachindex(E)
#         error_expansion!(E[k], model, Z[k], G[k])
#     end
# end
#
# "Compute the error expansion at a single KnotPoint (only for state vectors not in Vector space)"
# function error_expansion!(E::CostExpansion{<:Any,N}, model::RigidBody, z, G) where N
#     if N < 15
#         G = SMatrix(G)
#         E.u_  .= E.u
#         E.uu_ .= E.uu
#         E.ux_ .= SMatrix(E.ux)*G
#         E.x_  .= G'*SVector(E.x)
#
#         RobotDynamics.∇²differential!(E.xx_, model, state(z), E.x0)
#         E.xx_ .+= G'E.xx*G
#     else
#         E.u_ .= E.u
#         E.uu_ .= E.uu
#         mul!(E.ux_, E.ux, G)
#         mul!(E.x_, Transpose(G), E.x)
#
#         RobotDynamics.∇²differential!(E.xx_, model, state(z), E.x0)
#         mul!(E.tmp, E.xx, G)
#         mul!(E.xx_, Transpose(G), E.tmp, 1.0, 1.0)
#     end
# end
#
# "Copy the error expansion to another expansion"
# function error_expansion!(E::AbstractExpansion, Q::CostExpansion)
# 	E.x  .= Q.x
# 	E.u  .= Q.u
# 	E.xx .= Q.xx
# 	E.uu .= Q.uu
# 	E.ux .= Q.ux
# end
#
# "Get the error expansion"
# @inline function error_expansion(E::CostExpansion, model::RigidBody)
# 	return StaticExpansion(E.x_, E.xx_, E.u_, E.uu_, E.ux_)
# end
#
# "Get the error expansion (same as cost expansion)"
# @inline function error_expansion(E::CostExpansion, model::AbstractModel)
# 	return StaticExpansion(E.x, E.xx, E.u, E.uu, E.ux)
# end
#
# "Get cost expansion"
# @inline function cost_expansion(E::CostExpansion{<:Any,N,N}) where N
# 	return StaticExpansion(E.x, E.xx, E.u, E.uu, E.ux)
# end
