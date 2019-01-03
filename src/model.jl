import Base.copy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Model and Objective Classes
#
#     TYPES                                             Tree
#         Model                                       --------
#         Objective                                     Model
#         UnconstrainedObjective
#         ConstrainedObjective                        Objective
#                                                     ↙       ↘
#                                 UnconstrainedObjective     ConstrainedObjective
#
#
#     METHODS
#         update_objective: Update ConstrainedObjective values, creating a new
#             Objective.
#         _validate_bounds: Check bounds on state and control bound inequalities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
$(TYPEDEF)

Dynamics model

Holds all information required to uniquely describe a dynamic system, including
a general nonlinear dynamics function of the form `ẋ = f(x,u)`, where x ∈ ℜⁿ are
the states and u ∈ ℜᵐ are the controls.

Dynamics function `Model.f` should be in the following forms:
    'f!(ẋ,x,u)' and modify ẋ in place
"""
struct Model
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls

    # Construct a model from an explicit differential equation
    function Model(f::Function, n::Int64, m::Int64)
        # Make dynamics inplace
        if is_inplace_dynamics(f,n,m)
            f! = f
        else
            f! = wrap_inplace(f)
        end

        new(f!,n,m)
    end

    # Construct model from a `Mechanism` type from `RigidBodyDynamics`
    function Model(mech::Mechanism)
        m = length(joints(mech))  # subtract off joint to world
        Model(mech,ones(m))
    end

    """
        Model(mech::Mechanism, torques::Array{Bool, 1})
    Constructor for an underactuated mechanism, where torques is a binary array
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

        new(f, n, m)
    end
end

"$(SIGNATURES) Construct a fully actuated model from a string to a urdf file"
function Model(urdf::String)
    # construct model using string to urdf file
    mech = parse_urdf(urdf)
    Model(mech)
end

"$(SIGNATURES) Construct a partially actuated model from a string to a urdf file"
function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
    mech = parse_urdf(urdf)
    Model(mech,torques)
end


"""
$(SIGNATURES)
    Determine if the constraints are inplace. Returns boolean and number of constraints
"""
function is_inplace_constraints(c::Function,n::Int64,m::Int64)
    x = rand(n)
    u = rand(m)
    q = 100
    iter = 1

    vals = NaN*(ones(q))
    try
        c(vals,x,u)
    catch e
        if e isa MethodError
            return false
        end
    end

    return true
end

function count_inplace_output(c::Function, n::Int, m::Int)
    x = rand(n)
    u = rand(m)
    q0 = 100
    iter = 1

    q = q0
    vals = NaN*(ones(q))
    while iter < 5
        try
            c(vals,x,u)
            break
        catch e
            q *= 10
            iter += 1
            vals = NaN*(ones(q))
        end
    end
    p = count(isfinite.(vals))

    q = q0
    vals = NaN*(ones(q))
    while iter < 5
        try
            c(vals,x)
            break
        catch e
            if e isa MethodError
                p_N = 0
                break
            else
                q *= 10
                iter += 1
                vals = NaN*(ones(q))
            end
        end
    end
    p_N = count(isfinite.(vals))

    return p, p_N
end

function count_inplace_output(c::Function, n::Int)
    x = rand(n)
    q = 100
    iter = 1
    vals = NaN*(ones(q))

    while iter < 5
        try
            c(vals,x,u)
            break
        catch e
            q *= 10
            iter += 1
            vals = NaN*(ones(q))
        end
    end
    return count(isfinite.(vals))
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
