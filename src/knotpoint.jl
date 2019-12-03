
"""
Stores the states and controls for a single knot point
"""
mutable struct KnotPoint{T,N,M,NM}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    _inds::SVector{NM,Bool}
    dt::T
end

function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Float64)
    n = length(x)
    m = length(u)
    xinds = ones(Bool, n+m)
    xinds[n+1:end] .= 0
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    _inds = SVector{n+m}(xinds)
    z = SVector{n+m}([x;u])
    KnotPoint(z, _x, _u, _inds, dt)
end

# Constructor for terminal time step
function KnotPoint(x::AbstractVector, m::Int)
    u = zeros(m)
    KnotPoint(x, u, 0.)
end

@inline state(z::KnotPoint) = z.z[z._x]
@inline control(z::KnotPoint) = z.z[z._u]
@inline is_terminal(z::KnotPoint) = z.dt == 0

const Traj = AbstractVector{<:KnotPoint}
traj_size(Z::Vector{<:KnotPoint{T,N,M}}) where {T,N,M} = N,M,length(Z)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DYNAMICS FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"Evaluate the continuous dynamics for a knot point"
@inline dynamics(model::AbstractModel, z::KnotPoint) = dynamics(model, state(z), control(z))

"Evaluate the discrete dynamics for a knot point"
@inline function discrete_dynamics(model::AbstractModel, z::KnotPoint)
    discrete_dynamics(model, state(z), control(z), z.dt)
end

"Propagate the dynamics forward, storing the result in the next knot point"
function propagate_dynamics(model::AbstractModel, z_::KnotPoint, z::KnotPoint)
    x_next = discrete_dynamics(model, z)
    z_.z = [x_next; control(z_)]
end

"Evaluate the discrete dynamics Jacobian at a knot point"
@inline function discrete_jacobian(model::AbstractModel, z::KnotPoint)
    discrete_jacobian(model, z.z, z.dt)
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COST FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

"Calculate the 2nd order expansion of the cost at a knot point"
cost_expansion(cost::CostFunction, z::KnotPoint) =
    cost_gradient(cost, z)..., cost_hessian(cost, z)...

function cost_gradient!(E, obj::Objective, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.x[k], E.u[k] = cost_gradient(obj[k], Z[k])
    end
end

function cost_hessian!(E, obj::Objective, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.xx[k], E.uu[k], E.ux[k] = cost_hessian(obj[k], Z[k])
    end
end

"Expand cost for entire trajectory"
function cost_expansion(E, obj::Objective, Z::Traj)
    cost_gradient!(E, obj, Z)
    cost_hessian!(E, obj, Z)
end
