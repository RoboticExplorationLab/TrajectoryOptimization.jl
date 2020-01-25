export
    AbstractModel,
    InfeasibleModel,
    RigidBody,
    dynamics,
    discrete_dynamics,
    jacobian,
    discrete_jacobian

export
    QuadratureRule,
    RK3,
    HermiteSimpson


""" $(TYPEDEF)
Abstraction of a model of a dynamical system of the form ẋ = f(x,u), where x is the n-dimensional state vector
and u is the m-dimensional control vector.

Any inherited type must define the following interface:
ẋ = dynamics(model, x, u)
n,m = size(model)
"""
abstract type AbstractModel end


""" $(TYPEDEF)
Abstraction of a dynamical system with free-body dynamics, with a 12 or 13-dimensional state
vector: `[p; q; v; ω]`
where `p` is the 3D position, `q` is the 3 or 4-dimension attitude representation, `v` is the
3D linear velocity, and `ω` is the 3D angular velocity.

# Interface
Any single-body system can leverage the `RigidBody` type by inheriting from it and defining the
following interface:
```julia
forces(::MyRigidBody, x, u)  # return the forces in the world frame
moments(::MyRigidBody, x, u) # return the moments in the body frame
inertia(::MyRigidBody, x, u) # return the 3x3 inertia matrix
inertia_inv(::MyRigidBody, x, u)  # return the 3x3 inverse of the inertia matrix
mass_matrix(::MyRigidBody, x, u)  # return the 3x3 mass matrix
```

# Rotation Parameterization
A `RigidBody` model must specify the rotational representation being used. Any of the following
can be used:
* [`UnitQuaternion`](@ref): Unit Quaternion. Note that this representation needs to be further parameterized.
* [`MRP`](@ref): Modified Rodrigues Parameters
* [`RPY`](@ref): Roll-Pitch-Yaw Euler angles
"""
abstract type RigidBody{R<:Rotation} <: AbstractModel end

"Integration rule for approximating the continuous integrals for the equations of motion"
abstract type QuadratureRule end
"Integration rules of the form x′ = f(x,u), where x′ is the next state"
abstract type Implicit <: QuadratureRule end
"Integration rules of the form x′ = f(x,u,x′,u′), where x′,u′ are the states and controls at the next time step."
abstract type Explicit <: QuadratureRule end
"Third-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK3 <: Implicit end
"Third-order Runge-Kutta method with first-order-hold on the controls"
abstract type HermiteSimpson <: Explicit end

"Default quadrature rule"
const DEFAULT_Q = RK3

#=
Convenient methods for creating state and control vectors directly from the model
=#
for method in [:rand, :zeros, :ones]
    @eval begin
        function Base.$(method)(model::AbstractModel)
            n,m = size(model)
            x = @SVector $(method)(n)
            u = @SVector $(method)(m)
            return x, u
        end
        function Base.$(method)(::Type{T}, model::AbstractModel) where T
            n,m = size(model)
            x = @SVector $(method)(T,n)
            u = @SVector $(method)(T,m)
            return x,u
        end
    end
end
function Base.fill(model::AbstractModel, val)
    n,m = size(model)
    x = @SVector fill(val,n)
    u = @SVector fill(val,m)
    return x, u
end

"""Default size method for model (assumes model has fields n and m)"""
@inline Base.size(model::AbstractModel) = model.n, model.m

############################################################################################
#                               CONTINUOUS TIME METHODS                                    #
############################################################################################
"""```
ẋ = dynamics(model, z::KnotPoint)
```
Compute the continuous dynamics of a dynamical system given a KnotPoint"""
@inline dynamics(model::AbstractModel, z::KnotPoint) = dynamics(model, state(z), control(z), z.t)

# Default to not passing in t
@inline dynamics(model::AbstractModel, x, u, t) = dynamics(model, x, u)

"""```
∇f = jacobian(model, z::KnotPoint)
∇f = jacobian(model, z::SVector)
```
Compute the Jacobian of the continuous-time dynamics using ForwardDiff. The input can be either
a static vector of the concatenated state and control, or a KnotPoint. They must be concatenated
to avoid unnecessary memory allocations.
"""
function jacobian(model::AbstractModel, z::KnotPoint)
    ix, iu = z._x, z._u
    f_aug(z) = dynamics(model, z[ix], z[iu])
    s = z.z
    ForwardDiff.jacobian(f_aug, s)
end

function jacobian(model::AbstractModel, z::SVector)
    n,m = size(model)
    ix,iu = 1:n, n .+ (1:m)
    f_aug(z) = dynamics(model, view(z,ix), view(z,iu))
    ForwardDiff.jacobian(f_aug, z)
end

############################################################################################
#                          IMPLICIT DISCRETE TIME METHODS                                  #
############################################################################################

# Set default integrator
@inline discrete_dynamics(model::AbstractModel, z::KnotPoint) =
    discrete_dynamics(DEFAULT_Q, model, z)

""" Compute the discretized dynamics of `model` using implicit integration scheme `Q<:QuadratureRule`.

Methods:
```
x′ = discrete_dynamics(model, model, z)  # uses $(DEFAULT_Q) as the default integration scheme
x′ = discrete_dynamics(Q, model, x, u, t, dt)
x′ = discrete_dynamics(Q, model, z::KnotPoint)
```

The default integration scheme is stored in `TrajectoryOptimization.DEFAULT_Q`
"""
@inline discrete_dynamics(::Type{Q}, model::AbstractModel, z::KnotPoint) where Q<:Implicit =
    discrete_dynamics(Q, model, state(z), control(z), z.t, z.dt)

""" Compute the discrete dynamics Jacobian of `model` using implicit integration scheme `Q<:QuadratureRule`

Methods:
```
∇f = discrete_dynamics(model, z::KnotPoint)  # uses $(DEFAULT_Q) as the default integration scheme
∇f = discrete_jacobian(Q, model, z::KnotPoint)
∇f = discrete_jacobian(Q, model, s::SVector{NM1}, t, ix::SVector{N}, iu::SVector{M})
```
where `s = [x; u; dt]`, `t` is the time, and `ix` and `iu` are the indices to extract the state and controls.
"""
@inline discrete_jacobian(model::AbstractModel, z::KnotPoint) =
    discrete_jacobian(DEFAULT_Q, model, z)

function discrete_jacobian(::Type{Q}, model::AbstractModel,
        z::KnotPoint{T,N,M,NM}) where {Q<:Implicit,T,N,M,NM}
    ix,iu,idt = z._x, z._u, NM+1
    t = z.t
    fd_aug(s) = discrete_dynamics(Q, model, s[ix], s[iu], t, s[idt])
    s = [z.z; @SVector [z.dt]]
    ForwardDiff.jacobian(fd_aug, s)
end

function discrete_jacobian(::Type{Q}, model::AbstractModel,
       s::SVector{NM1}, t::T, ix::SVector{N}, iu::SVector{M}) where {Q<:Implicit,T,N,M,NM1}
    idt = NM1
    fd_aug(s) = discrete_dynamics(Q, model, s[ix], s[iu], t, s[idt])
    ForwardDiff.jacobian(fd_aug, s)
end

function dynamics_expansion(∇f, G1, G2, model::AbstractModel, z::KnotPoint)
	ix,iu = z._x, z._u
	A = G2'∇f[ix,ix]*G1
	B = G2'∇f[ix,iu]
	return A,B
end


############################################################################################
#                               STATE DIFFERENTIALS                                        #
############################################################################################

@inline state_diff(model::AbstractModel, x, x0) = x - x0
# @inline state_diff_jacobian(model::AbstractModel, x::SVector{N,T}) where {N,T} = Diagonal(@SVector ones(T,N))
@inline state_diff_jacobian(model::AbstractModel, x::SVector{N,T}) where {N,T} = I
@inline state_diff_size(model::AbstractModel) = size(model)[1]

@inline state_diff_jacobian!(G, model::AbstractModel, Z::Traj) = nothing

function state_diff_jacobian!(G, model::RigidBody, Z::Traj)
    for k in eachindex(Z)
        G[k] = state_diff_jacobian(model, state(Z[k]))
    end
end

is_quat(model::AbstractModel, z::KnotPoint{T,N}) where {T,N} = @SVector zeros(N)

############################################################################################
#                               INFEASIBLE MODELS                                          #
############################################################################################
""" $(TYPEDEF)
An infeasible model is an augmented dynamics model that makes the system artifically fully
actuated by augmenting the control vector with `n` additional controls. The dynamics are
handled explicitly in discrete time:

``x_{k+1} = f(x_k,u_k,dt) + w_k``

where ``w_k`` are the additional `n`-dimensional controls. In practice, these are constrained
to be zero by the end of the solve.

# Constructors
```julia
InfeasibleModel(model::AbstractModel)
```
"""
struct InfeasibleModel{N,M,D<:AbstractModel} <: AbstractModel
    model::D
    _u::SVector{M,Int}  # inds to original controls
    _ui::SVector{N,Int} # inds to infeasible controls
end

function InfeasibleModel(model::AbstractModel)
    n,m = size(model)
    _u  = SVector{m}(1:m)
    _ui = SVector{n}((1:n) .+ m)
    InfeasibleModel(model, _u, _ui)
end

function Base.size(model::InfeasibleModel)
    n,m = size(model.model)
    return n, n+m
end

dynamics(::InfeasibleModel, x, u) =
    throw(ErrorException("Cannot evaluate continuous dynamics on an infeasible model"))

@generated function discrete_dynamics(::Type{Q}, model::InfeasibleModel{N,M},
        z::KnotPoint{T,N}) where {T,N,M,Q<:Implicit}
    _u = SVector{M}((1:M) .+ N)
    _ui = SVector{N}((1:N) .+ (N+M))
    quote
        x = state(z)
        dt = z.dt
        u0 = z.z[$_u]
        ui = z.z[$_ui]
        discrete_dynamics($Q, model.model, x, u0, z.t, dt) + ui
    end
end

@generated function discrete_jacobian(::Type{Q}, model::InfeasibleModel{N,M},
        z::KnotPoint{T,N,NM,L}) where {T,N,M,NM,L,Q<:Implicit}

    ∇ui = [(@SMatrix zeros(N,N+M)) Diagonal(@SVector ones(N)) @SVector zeros(N)]
    _x = SVector{N}(1:N)
    _u = SVector{M}((1:M) .+ N)
    _z = SVector{N+M}(1:N+M)
    _ui = SVector{N}((1:N) .+ (N+M))
    zi = [:(z.z[$i]) for i = 1:N+M]
    NM1 = N+M+1
    ∇u0 = @SMatrix zeros(N,N)

    quote
        # Build KnotPoint for original model
        s0 = SVector{$NM1}($(zi...), z.dt)

        u0 = z.z[$_u]
        ui = z.z[$_ui]
        ∇f = discrete_jacobian($Q, model.model, s0, z.t, $_x, $_u)::SMatrix{N,NM+1}
        ∇dt = ∇f[$_x, N+M+1]
        [∇f[$_x, $_z] $∇u0 ∇dt] + $∇ui
    end
end



"Calculate a dynamically feasible initial trajectory for an infeasible problem, given a
desired trajectory"
function infeasible_trajectory(model::InfeasibleModel{n,m}, Z0::Vector{<:KnotPoint{T,n,m}}) where {T,n,m}
    x,u = zeros(model)
    ui = @SVector zeros(n)
    Z = [KnotPoint(state(z), [control(z); ui], z.dt) for z in Z0]
    N = length(Z0)
    for k = 1:N-1
        propagate_dynamics(RK3, model, Z[k+1], Z[k])
        x′ = state(Z[k+1])
        u_slack = state(Z0[k+1]) - x′
        u = [control(Z0[k]); u_slack]
        set_control!(Z[k], u)
        set_state!(Z[k+1], x′ + u_slack)
    end
    return Z
end
