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

Dynamics function `Model.f` should be in one of the following forms:
* `f(x,u)` and return ẋ
* 'f!(ẋ,x,u)' and modify ẋ in place (recommended)
"""
struct Model
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls

    # Construct a model from an explicit differential equation
    function Model(f::Function, n::Int64, m::Int64)
        # Make dynamics inplace
        if is_inplace_dynamics(f,n,m)
            f! =f
        else
            f! = wrap_inplace(f)
        end

        new(f!,n,m)
    end

    # Construct model from a `Mechanism` type from `RigidBodyDynamics`
    # Automatically assigns one control per joint
    function Model(mech::Mechanism)
        m = length(joints(mech))-1  # subtract off joint to world
        Model(mech,ones(m,1))
    end

    """
        Model(mech::Mechanism, torques::Array{Bool, 1})
    Constructor for an underactuated mechanism, where torques is a binary array
    that specifies whether a joint is actuated.
    """
    function Model(mech::Mechanism, torques::Array)

        # construct a model using robot dynamics equation assembed from URDF file
        n = num_positions(mech) + num_velocities(mech) + num_additional_states(mech)
        num_joints = length(joints(mech))-1  # subtract off joint to world
        m = num_joints # Default to number of joints

        function fc(xdot,x,u)
            state = MechanismState{eltype(x)}(mech)
            # set the state variables:
            q = x[1:num_joints]
            qd = x[1+num_joints:num_joints+num_joints]
            set_configuration!(state, q)
            set_velocity!(state, qd)
            xdot[1:num_joints] = qd
            xdot[num_joints+1:num_joints+num_joints] = Array(mass_matrix(state))\(torques.*u - Array(dynamics_bias(state)))
            return nothing
        end

        new(fc, n, convert(Int,sum(torques)))
    end
end

"$(SIGNATURES) Construct a fully actuated model from a string to a urdf file"
function Model(urdf::String)
    # construct modeling using string to urdf file
    mech = parse_urdf(Float64,urdf)
    Model(mech)
end

"$(SIGNATURES) Construct a partially actuated model from a string to a urdf file"
function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
    # construct modeling using string to urdf file
    mech = parse_urdf(Float64,urdf)
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
    xdot = NaN*ones(q)
    iter = 1

    while iter < 5
        try
            c(xdot,x,u)
        catch x
            if x isa MethodError
                return false, 0
            end
        end

        p = count(isfinite.(xdot))
        if p > 0
            return true, p
        else
            q *= 10
            iter += 1
        end
    end
    #println("Constraint function is inplace but does not modify output")
    return true, 0
end

function is_inplace_constraints(c::Function,n::Int64)
    x = rand(n)
    q = 100
    xdot = NaN*ones(q)
    iter = 1

    while iter < 5
        try
            c(xdot,x)
        catch x
            if x isa MethodError
                return false, 0
            end
        end

        p = count(isfinite.(xdot))
        if p > 0
            return true, p
        else
            q *= 10
            iter += 1
        end
    end
    #println("Constraint function is inplace but does not modify output")
    return true, 0
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
end

#*********************************#
#        OBJECTIVE CLASS          #
#*********************************#

"""
$(TYPEDEF)
Generic type for Objective functions, which are currently strictly Quadratic
"""
abstract type Objective end

"""
$(TYPEDEF)
Defines a quadratic objective for an unconstrained optimization problem
"""
mutable struct UnconstrainedObjective{TQ,TR,TQf} <: Objective
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    Qf::TQf               # Quadratic final cost for terminal state (n,n)
    tf::Float64           # Final time (sec)
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)
end

function copy(obj::UnconstrainedObjective)
    UnconstrainedObjective(copy(obj.Q),copy(obj.R),copy(obj.Qf),copy(obj.tf),copy(obj.x0),copy(obj.xf))
end

"""
$(TYPEDEF)
Define a quadratic objective for a constrained optimization problem.

# Constraint formulation
* Equality constraints: `f(x,u) = 0`
* Inequality constraints: `f(x,u) ≥ 0`

"""
mutable struct ConstrainedObjective{TQ<:AbstractArray,TR<:AbstractArray,TQf<:AbstractArray} <: Objective
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    Qf::TQf               # Quadratic final cost for terminal state (n,n)
    tf::Float64           # Final time (sec)
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)

    # Control Constraints
    u_min::Array{Float64,1}  # Lower control bounds (m,)
    u_max::Array{Float64,1}  # Upper control bounds (m,)

    # State Constraints
    x_min::Array{Float64,1}  # Lower state bounds (n,)
    x_max::Array{Float64,1}  # Upper state bounds (n,)

    # General Stage Constraints
    cI::Function  # inequality constraint function (inplace)
    cE::Function  # equality constraint function (inplace)

    # Terminal Constraints
    use_terminal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false)
    # Overload cI and cE with a single argument for terminal constraints

    # Constants
    p::Int   # Total number of stage constraints
    pI::Int  # Number of inequality constraints
    p_N::Int  # Number of terminal constraints
    pI_N::Int  # Number of terminal inequality constraints

    function ConstrainedObjective(Q::TQ,R::TR,Qf::TQf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint) where {TQ,TR,TQf}

        n = size(Q,1)
        m = size(R,1)

        # Make general inequality/equality constraints inplace
        flag_cI, pI_c = is_inplace_constraints(cI,n,m)
        if !flag_cI
            cI = wrap_inplace(cI)
            println("Custom inequality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end

        flag_cE, pE_c = is_inplace_constraints(cE,n,m)
        if !flag_cE
            cE = wrap_inplace(cE)
            println("Custom equality constraints are not inplace\n -converting to inplace\n -THIS IS SLOW")
        end

        # Validity Tests
        u_max, u_min = _validate_bounds(u_max,u_min,m)
        x_max, x_min = _validate_bounds(x_max,x_min,n)

        # Stage Constraints
        pI = 0
        pE = 0
        pI += count(isfinite, u_min)
        pI += count(isfinite, u_max)
        pI += count(isfinite, x_min)
        pI += count(isfinite, x_max)

        # u0 = zeros(m)
        # if ~isa(cI(x0,u0), Nothing)
        #     pI += size(cI(x0,u0),1)
        # end
        # if ~isa(cE(x0,u0), Nothing)
        #     pE += size(cE(x0,u0),1)
        # end
        pI += pI_c
        pE += pE_c

        p = pI + pE

        #TODO custom terminal constraints
        # Terminal Constraints
        pI_N = pE_N = 0

        # try cI(x0)
        #     pI_N = size(cI(x0),1)
        # catch
        #     pI_N = 0
        # end
        # try cE(x0)
        #     pE_N = size(cE(x0),1)
        # catch
        #     pE_N = 0
        # end
        if use_terminal_constraint
            pE_N += n
        end
        p_N = pI_N + pE_N

        new{TQ,TR,TQf}(Q,R,Qf,tf,x0,xf, u_min, u_max, x_min, x_max, cI, cE, use_terminal_constraint, p, pI, p_N, pI_N)
    end
end

"""
$(SIGNATURES)

Construct a ConstrainedObjective with defaults.

Create a ConstrainedObjective, specifying only the needed fields. All others
will be set to their default, constrained values.

# Constraint formulation
* Equality constraints: `f(x,u) = 0`
* Inequality constraints: `f(x,u) ≥ 0`

# Arguments
* u_min, u_max, x_min, x_max: Upper and lower bounds that can accept either a single scalar or
a vector of size (m,). A scalar will be copied to all states or controls. Values
can be ±Inf.
* cI, cE: Functions for inequality and equality constraints. Must be of the form
`c = f(x,u)`, where `c` is of size (pI_c,) or (pE_c,).
* cI_N, cE_N: Functions for terminal constraints. Must be of the from `c = f(x)`,
where `c` is of size (pI_c_N,) or (pE_c_N,).
"""
function ConstrainedObjective(Q,R,Qf,tf,x0,xf;
    u_min=-ones(size(R,1))*Inf, u_max=ones(size(R,1))*Inf,
    x_min=-ones(size(Q,1))*Inf, x_max=ones(size(Q,1))*Inf,
    cI=(c,x,u)->nothing, cE=(c,x,u)->nothing,
    use_terminal_constraint=true)

    ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint)
end

function copy(obj::ConstrainedObjective)
    ConstrainedObjective(copy(obj.Q),copy(obj.R),copy(obj.Qf),copy(obj.tf),copy(obj.x0),copy(obj.xf),
        u_min=copy(obj.u_min), u_max=copy(obj.u_max), x_min=copy(obj.x_min), x_max=copy(obj.x_max),
        cI=obj.cI, cE=obj.cE,
        use_terminal_constraint=obj.use_terminal_constraint)
end

"$(SIGNATURES) Construct a ConstrainedObjective from an UnconstrainedObjective"
function ConstrainedObjective(obj::UnconstrainedObjective; kwargs...)
    ConstrainedObjective(obj.Q, obj.R, obj.Qf, obj.tf, obj.x0, obj.xf; kwargs...)
end

"""
$(SIGNATURES)
Updates constrained objective values and returns a new objective.

Only updates the specified fields, all others are copied from the previous
Objective.
"""
function update_objective(obj::ConstrainedObjective;
    u_min=obj.u_min, u_max=obj.u_max, x_min=obj.x_min, x_max=obj.x_max,
    cI=obj.cI, cE=obj.cE,
    use_terminal_constraint=obj.use_terminal_constraint)

    ConstrainedObjective(obj.Q,obj.R,obj.Qf,obj.tf,obj.x0,obj.xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint)

end

"""
$(SIGNATURES)
Check max/min bounds for state and control.

Converts scalar bounds to vectors of appropriate size and checks that lengths
are equal and bounds do not result in an empty set (i.e. max > min).

# Arguments
* n: number of elements in the vector (n for states and m for controls)
"""
function _validate_bounds(max,min,n::Int)

    if min isa Real
        min = ones(n)*min
    end
    if max isa Real
        max = ones(n)*max
    end
    if length(max) != length(min)
        throw(DimensionMismatch("u_max and u_min must have equal length"))
    end
    if ~all(max .> min)
        throw(ArgumentError("u_max must be greater than u_min"))
    end
    if length(max) != n
        throw(DimensionMismatch("limit of length $(length(max)) doesn't match expected length of $n"))
    end
    return max, min
end


function to_static(obj::ConstrainedObjective)
    n = size(obj.Q,1)
    m = size(obj.R,2)
    ConstrainedObjective(SMatrix{n,n}(Array(obj.Q)),SMatrix{m,m}(Array(obj.R)), SMatrix{n,n}(Array(obj.Qf)),
        obj.tf, obj.x0, obj.xf,
        u_min=copy(obj.u_min), u_max=copy(obj.u_max), x_min=copy(obj.x_min), x_max=copy(obj.x_max),
        cI=obj.cI, cE=obj.cE,
        use_terminal_constraint=obj.use_terminal_constraint)
end

function to_static(obj::UnconstrainedObjective)
    n = size(obj.Q,1)
    m = size(obj.R,2)
    UnconstrainedObjective(SMatrix{n,n}(Array(obj.Q)),SMatrix{m,m}(Array(obj.R)), SMatrix{n,n}(Array(obj.Qf)),
        obj.tf, obj.x0, obj.xf)
end
