

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DYNAMICS FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


"Propagate the dynamics forward, storing the result in the next knot point"
function propagate_dynamics(model::AbstractModel, z_::KnotPoint, z::KnotPoint)
    x_next = discrete_dynamics(model, z)
    z_.z = [x_next; control(z_)]
end


"Evaluate the discrete dynamics for all knot points"
function discrete_dynamics!(f, model, Z::Traj)
    for k in eachindex(Z)
        f[k] = discrete_dynamics(model, Z[k])
    end
end

"Evaluate the discrete dynamics Jacobian for all knot points"
function discrete_jacobian!(∇f, model, Z::Traj)
    for k in eachindex(∇f)
        ∇f[k] = discrete_jacobian(model, Z[k])
    end
end

############################################################################################
#                              COST FUNCTIONS                                              #
############################################################################################

"Evaluate the cost at a knot point"
function stage_cost(cost::CostFunction, z::KnotPoint)
    if is_terminal(z)
        stage_cost(cost, state(z))
    else
        stage_cost(cost, state(z), control(z), z.dt)
    end
end

"Evaluate the cost for a trajectory"
function cost(obj::Objective, Z::Traj)::Float64
    J::Float64 = 0.0
    for k in eachindex(Z)
        J += stage_cost(obj[k], Z[k])::Float64
    end
    return J
end

"Evaluate the cost for a trajectory (non-allocating)"
@inline function cost!(obj::Objective, Z::Traj)
    map!(stage_cost, obj.J, obj.cost, Z)
end



"Get Qx, Qu pieces of gradient of cost function, multiplied by dt"
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

"Get Qxx, Quu, Qux pieces of Hessian of cost function, multiplied by dt"
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EUCLIDEAN STATES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"Calculate the 2nd order expansion of the cost at a knot point"
cost_expansion(cost::CostFunction, model::AbstractModel, z::KnotPoint) =
    cost_gradient(cost, z)..., cost_hessian(cost, z)...


"Calculate the cost gradient for an entire trajectory"
function cost_gradient!(E, obj::Objective, model::AbstractModel, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.x[k], E.u[k] = cost_gradient(obj[k], Z[k])
    end
end

"Calculate the cost Hessian for an entire trajectory"
function cost_hessian!(E, obj::Objective, model::AbstractModel, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.xx[k], E.uu[k], E.ux[k] = cost_hessian(obj[k], Z[k])
    end
end

"Expand cost for entire trajectory"
function cost_expansion(E, obj::Objective, model::AbstractModel, Z::Traj)
    cost_gradient!(E, obj, model, Z)
    cost_hessian!(E, obj, model, Z)
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NON-EUCLIDEAN STATES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function cost_expansion(cost::CostFunction, model::FreeBodyModel, z::KnotPoint)
    Qx, Qu = cost_gradient(cost, z)
    Qxx, Quu, Qux = cost_hessian(cost, z)
    return G*Qx, Qu, G*Qxx'G, Quu,  Qux'G
end

"Calculate the cost gradient for an entire trajectory for a model with non-Euclidean states"
function cost_gradient!(E, obj::Objective, model::FreeBodyModel, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        G = state_diff_jacobian(model, state(Z[k]))
        Qx, Qu = cost_gradient(obj[k], Z[k])
        E.x[k], E.u[k] = G*Qx, Qu
    end
end

"Calculate the cost Hessian for an entire trajectory for a model with non-Euclidean states"
function cost_hessian!(E, obj::Objective, model::FreeBodyModel, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        G = state_diff_jacobian(model, state(Z[k]))
        Qxx, Quu, Qux = cost_hessian(obj[k], Z[k])
        E.xx[k], E.uu[k], E.ux[k] = G*Qxx*G', Quu, Qux*G'
    end
end

function cost_expansion(E, obj::Objective, model::FreeBodyModel, Z::Traj)
    for k in eachindex(Z)
        G = state_diff_jacobian(model, state(Z[k]))
        Qxx, Quu, Qux = cost_hessian(obj[k], Z[k])
        Qx, Qu = cost_gradient(obj[k], Z[k])
        E.xx[k], E.uu[k], E.ux[k] = G*Qxx*G', Quu, Qux*G'
        E.x[k], E.u[k] = G*Qx, Qu
    end
end
