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
        new(f,n,m)
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

        function fc(x,u) # TODO: make this an in place operation
            state = MechanismState{eltype(x)}(mech)

            # set the state variables:
            q = x[1:num_joints]
            qd = x[(1:num_joints)+num_joints]
            set_configuration!(state, q)
            set_velocity!(state, qd)

            [qd; Array(mass_matrix(state))\(torques.*u) - Array(mass_matrix(state))\Array(dynamics_bias(state))]
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
$(TYPEDEF)
Generic type for Objective functions, which are currently strictly Quadratic
"""
abstract type Objective end

"""
$(TYPEDEF)
Defines a quadratic objective for an unconstrained optimization problem
"""
mutable struct UnconstrainedObjective <: Objective
    Q::Array{Float64,2}   # Quadratic stage cost for states (n,n)
    R::Array{Float64,2}   # Quadratic stage cost for controls (m,m)
    Qf::Array{Float64,2}  # Quadratic final cost for terminal state (n,n)
    tf::Float64           # Final time (sec)
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)
end

"""
$(TYPEDEF)
Define a quadratic objective for a constrained optimization problem.

# Constraint formulation
* Equality constraints: `f(x,u) = 0`
* Inequality constraints: `f(x,u) ≥ 0`

"""
mutable struct ConstrainedObjective <: Objective
    Q::Array{Float64,2}   # Quadratic stage cost for states (n,n)
    R::Array{Float64,2}   # Quadratic stage cost for controls (m,m)
    Qf::Array{Float64,2}  # Quadratic final cost for terminal state (n,n)
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
    cI::Function  # inequality constraint function
    cE::Function  # equality constraint function

    # Terminal Constraints
    use_terminal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false)
    cI_N::Function # terminal inequality constraint function
    cE_N::Function # terminal equality constraint function

    # Constants
    p::Int   # Total number of stage constraints
    pI::Int  # Number of inequality constraints
    p_N::Int  # Number of terminal constraints
    pI_N::Int  # Number of terminal inequality constraints

    function ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint,
        cI_N, cE_N)

        n = size(Q,1)
        m = size(R,1)

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

        u0 = zeros(m)
        if cI(x0,u0) != nothing
            pI += size(cI(x0,u0),1)
        end
        if cE(x0,u0) != nothing
            pE += size(cE(x0,u0),1)
        end
        p = pI + pE


        # Terminal Constraints
        pI_N = pE_N = 0
        if cI_N(x0) != nothing
            pI_N = size(cI_N(x0),1)
        end
        if cE_N(x0) != nothing
            pE_N = size(cE_N(x0),1)
        end
        if use_terminal_constraint
            pE_N += n
        end
        p_N = pI_N + pE_N

        new(Q,R,Qf,tf,x0,xf, u_min, u_max, x_min, x_max, cI, cE, use_terminal_constraint, cI_N, cE_N, p, pI, p_N, pI_N)
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
    cI=(x,u)->nothing, cE=(x,u)->nothing,
    use_terminal_constraint=true,
    cI_N=(x)->nothing, cE_N=(x)->nothing)

    ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint,
        cI_N, cE_N)
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
    use_terminal_constraint=obj.use_terminal_constraint,
    cI_N=obj.cI_N, cE_N=obj.cE_N)

    ConstrainedObjective(obj.Q,obj.R,obj.Qf,obj.tf,obj.x0,obj.xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint,
        cI_N, cE_N)

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
