############################################################################################
#                              COST METHODS                                                #
############################################################################################

# """
# 	stage_cost(cost::CostFunction, z::AbstractKnotPoint)

# Evaluate the cost at a knot point, and automatically handle terminal
# knot point, multiplying by dt as necessary."""
# function stage_cost(cost::CostFunction, z::AbstractKnotPoint)
#     if is_terminal(z)
#         stage_cost(cost, state(z))
#     else
#         stage_cost(cost, state(z), control(z))*z.dt
#     end
# end

"""
	cost(obj::Objective, Z::SampledTrajectory)
	cost(obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::SampledTrajectory)

Evaluate the cost for a trajectory. If a dynamics constraint is given,
    use the appropriate integration rule, if defined.
"""
function cost(obj::Objective, Z::SampledTrajectory{<:Any,<:Any,<:AbstractFloat})
    cost!(obj, Z)
    J = get_J(obj)
    return sum(J)
end

# ForwardDiff-able method
function cost(obj::Objective, Z::SampledTrajectory{<:Any,<:Any,T}) where T
    J = zero(T)
    for k = 1:length(obj)
        J += RD.evaluate(obj[k], Z[k])
    end
    return J
end

"Evaluate the cost for a trajectory (non-allocating)"
@inline function cost!(obj::Objective, Z::SampledTrajectory)
    map!(RD.evaluate, obj.J, obj.cost, Z.data)
end
