import Base.copy

abstract type Model end

"""
$(TYPEDEF)

Dynamics model

Holds all information required to uniquely describe a dynamic system, including
a general nonlinear dynamics function of the form `ẋ = f(x,u)`, where x ∈ ℜⁿ are
the states and u ∈ ℜᵐ are the controls.

Dynamics function `Model.f` should be in the following forms:
    'f!(ẋ,x,u)' and modify ẋ in place
"""
struct AnalyticalModel <:Model
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls

    # Construct a model from an explicit differential equation
    function AbstractModel(f::Function, n::Int64, m::Int64)
        # Make dynamics inplace
        if is_inplace_dynamics(f,n,m)
            f! = f
        else
            f! = wrap_inplace(f)
        end

        new(f!,n,m)
    end
end

""" $(SIGNATURES) Create a Model given an inplace analytical function for the continuous dynamics with n states and m controls"""
Model(f::Function, n::Int64, m::Int64) = AnalyticalModel(f,n,m)

"""
$(TYPEDEF)
RigidBodyDynamics model. Wrapper for a RigidBodyDynamics Mechanism
"""
struct RBDModel <: Model
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls
    mech::Mechanism  # RigidBodyDynamics Mechanism
end




# Construct model from a `Mechanism` type from `RigidBodyDynamics`
function Model(mech::Mechanism)
    m = length(joints(mech))  # subtract off joint to world
    Model(mech,ones(m))
end

"""
$(SIGNATURES)
Model(mech::Mechanism, torques::Array{Bool, 1}) Constructor for an underactuated mechanism, where torques is a binary array
that specifies whether a joint is actuated.
"""
function Model(mech::Mechanism, torques::Array)

    # construct a model using robot dynamics equation assembed from URDF file
    n = num_positions(mech) + num_velocities(mech) + num_additional_states(mech)
    num_joints = length(joints(mech))  # subtract off joint to world

    if length(torques) != num_joints
        error("Torque underactuation specified does not match mechanism dimensions")
    end

    m = convert(Int,sum(torques)) # number of actuated (ie, controllable) joints
    torque_matrix = 1.0*Matrix(I,num_joints,num_joints)[:,torques.== 1] # matrix to convert from control inputs to mechanism joints

    statecache = StateCache(mech)
    dynamicsresultscache = DynamicsResultCache(mech)

    function f(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
        state = statecache[T]
        dyn = dynamicsresultscache[T]
        dynamics!(view(ẋ,1:n), dyn, state, x, torque_matrix*u)
        return nothing
    end

    RBDModel(f, n, m, mech)
end

"$(SIGNATURES) Construct a fully actuated model from a string to a urdf file"
function Model(urdf::String)
    # construct model using string to urdf file
    mech = parse_urdf(urdf)
    Model(mech)
end

"$(SIGNATURES) Construct a partially actuated model from a string to a urdf file, where torques is a binary array that specifies whether a joint is actuated."
function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
    mech = parse_urdf(urdf)
    Model(mech,torques)
end

"""
$(SIGNATURES)
Determine if the dynamics in model are in place. i.e. the function call is of
the form `f!(xdot,x,u)`, where `xdot` is modified in place. Returns a boolean.
"""
function is_inplace_dynamics(model::Model)::Bool
    x = rand(model.n)
    u = rand(model.m)
    xdot = rand(model.n)
    try
        model.f(xdot,x,u)
    catch x
        if x isa MethodError
            return false
        end
    end
    return true
end

function is_inplace_dynamics(f::Function,n::Int64,m::Int64)::Bool
    x = rand(n)
    u = rand(m)
    xdot = rand(n)
    try
        f(xdot,x,u)
    catch x
        if x isa MethodError
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Makes the dynamics function `f(x,u)` appear to operate as an inplace operation of the
form `f!(xdot,x,u)`.
"""
function wrap_inplace(f::Function)
    f!(xdot,x,u) = copyto!(xdot, f(x,u))
    f!(xdot,x) = copyto!(xdot, f(x))
end
