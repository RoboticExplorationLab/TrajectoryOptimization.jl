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
        m = length(joints(mech))-1  # subtract off joint to world
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
        num_joints = length(joints(mech))-1  # subtract off joint to world

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
    mech = parse_urdf(Float64,urdf)
    Model(mech)
end

"$(SIGNATURES) Construct a partially actuated model from a string to a urdf file"
function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
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

"""
$(SIGNATURES)
    Determine if the constraints are inplace. Returns boolean and number of constraints
"""
function is_inplace_constraints(c::Function,n::Int64)
    x = rand(n)
    q = 100
    iter = 1

    vals = NaN*(ones(q))
    try
        c(vals,x)
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
            c(vals,x)
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
Defines a quadratic objective for an unconstrained optimization problem of the
    following form:
    J = 0.5(xₙ-xf)'Q(xₙ-xf) + Σ 0.5(xₖ-xf)'Q(xₖ-xf) + uₖ'Ruₖ
"""
mutable struct UnconstrainedObjective{TQ,TR,TQf} <: Objective
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    Qf::TQf               # Quadratic final cost for terminal state (n,n)
    c::Float64            # Constant stage cost (weight for minimum time problems)
    tf::Float64           # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)
    function UnconstrainedObjective(Q::TQ,R::TR,Qf::TQf,c::Float64,tf::Float64,x0,xf) where {TQ,TR,TQf}
        if !isposdef(R)
            err = ArgumentError("R must be positive definite")
            throw(err)
        end
        if tf < 0.
            err = ArgumentError("$tf is invalid input for final time. tf must be positive or zero (minimum time)")
            throw(err)
        end
        if c < 0.
            err = ArgumentError("$c is invalid input for constant stage cost. Must be positive")
            throw(err)
        end
        new{TQ,TR,TQf}(Q,R,Qf,c,tf,x0,xf)
    end
end

# Minimum time constructor (specified c)
function UnconstrainedObjective(Q,R,Qf,c::Float64,tf::Symbol,x0::Vector{Float64},xf::Vector{Float64})
    if tf == :min
        tf = 0.
        UnconstrainedObjective(Q,R,Qf,c,tf,x0,xf)
    else
        err = ArgumentError(":min is the only recognized Symbol for the final time")
        throw(err)
    end
end

# Minimum time constructor (set c to default)
function UnconstrainedObjective(Q,R,Qf,tf::Symbol,x0::Vector{Float64},xf::Vector{Float64})
    UnconstrainedObjective(Q,R,Qf,1.0,tf,x0,xf)
end

# No minimum time
function UnconstrainedObjective(Q,R,Qf,tf::Float64,x0::Vector{Float64},xf::Vector{Float64})
    UnconstrainedObjective(Q,R,Qf,0.0,tf,x0,xf)
end


function copy(obj::UnconstrainedObjective)
    UnconstrainedObjective(copy(obj.Q),copy(obj.R),copy(obj.Qf),copy(obj.c),copy(obj.tf),copy(obj.x0),copy(obj.xf))
end

"""
$(TYPEDEF)
    Define a quadratic objective and custom constraints for a constrained optimization problem.

    J = 1/2(xₙ-xf)'Qf(xₙ-xf) + Σ 1/2(xₖ-xf)'Q(xₖ-xf) + 1/2uₖ'Ruₖ

    Constraint formulation
    gs_custom(x) <= 0
    gc_custom(u) <= 0
    hs_custom(x) = 0
    hc_custom(u) = 0
"""
mutable struct ConstrainedObjective{TQ<:AbstractArray,TR<:AbstractArray,TQf<:AbstractArray} <: Objective
    Q::TQ             # Quadratic stage cost for states (n,n)
    R::TR              # Quadratic stage cost for controls (m,m)
    Qf::TQf          # Quadratic final cost for terminal state (n,n)
    tf::Float64           # Final time (sec). If tf = 0, the problem is set to minimum time
    x0::Array{Float64,1}  # Initial state (n,)
    xf::Array{Float64,1}  # Final state (n,)

    # Control Constraints
    u_min::Array{Float64,1}  # Lower control bounds (m,)
    u_max::Array{Float64,1}  # Upper control bounds (m,)

    # State Constraints
    x_min::Array{Float64,1}  # Lower state bounds (n,)
    x_max::Array{Float64,1}  # Upper state bounds (n,)

    # Custom Stage Constraints
    gs_custom::Function # inequality state constraints
    gc_custom::Function # inequality control constraints
    hs_custom::Function # equality state constraints
    hc_custom::Function # equality control constraints

    # Terminal Constraints
    use_terminal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false) # TODO I don't think this is used

    # Constants (these do not count infeasible or minimum time constraints)
    pIs::Int
    pIc::Int
    pEs::Int
    pEc::Int
    pEsN::Int # terminal state constraints (n)

    function ConstrainedObjective(Q::TQ,R::TR,Qf::TQf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        gs_custom, gc_custom, hs_custom, hc_custom,
        use_terminal_constraint) where {TQ,TR,TQf}

        n = size(Q,1)
        m = size(R,1)

        # Check that custom constraints are inplace
        if !is_inplace_constraints(gs_custom,n)
            error("Custom state inequality constraints are not inplace")
        end
        pIs_custom = count_inplace_output(gs_custom,n)

        if !is_inplace_constraints(gc_custom,m)
            error("Custom control inequality constraints are not inplace")
        end
        pIc_custom = count_inplace_output(gc_custom,m)

        if !is_inplace_constraints(hs_custom,n)
            error("Custom state equality constraints are not inplace")
        end
        pEs_custom = count_inplace_output(hs_custom,n)

        if !is_inplace_constraints(hc_custom,m)
            error("Custom control equality constraints are not inplace")
        end
        pEc_custom = count_inplace_output(hc_custom,m)

        # Validity Tests
        u_max, u_min = _validate_bounds(u_max,u_min,m)
        x_max, x_min = _validate_bounds(x_max,x_min,n)

        u_min_active = isfinite.(u_min)
        u_max_active = isfinite.(u_max)
        x_min_active = isfinite.(x_min)
        x_max_active = isfinite.(x_max)

        pIs = count(x_min_active) + count(x_max_active) + pIs_custom
        pIc = count(u_min_active) + count(u_max_active) + pIc_custom
        pEs = pEs_custom
        pEc = pEc_custom
        pEsN = pEs + n

        new{TQ,TR,TQf}(Q, R, Qf, tf, x0, xf, u_min, u_max, x_min, x_max, gs_custom, gc_custom, hs_custom, hc_custom, use_terminal_constraint, pIs, pIc, pEs, pEc, pEsN)
    end
end

"""
$(SIGNATURES)
    Construct a ConstrainedObjective with defaults.
"""
function ConstrainedObjective(Q,R,Qf,tf,x0,xf;
    u_min=-ones(size(R,1))*Inf, u_max=ones(size(R,1))*Inf,
    x_min=-ones(size(Q,1))*Inf, x_max=ones(size(Q,1))*Inf,
    gs_custom=(c,x)->nothing, gc_custom=(c,u)->nothing, hs_custom=(c,x)->nothing, hc_custom=(c,u)->nothing,
    use_terminal_constraint=true)

    ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min, u_max,
        x_min, x_max,
        gs_custom, gc_custom, hs_custom, hc_custom,
        use_terminal_constraint)
end

# Minimum time constructor
function ConstrainedObjective(Q,R,Qf,tf::Symbol,x0,xf; kwargs...)
    if tf == :min
        ConstrainedObjective(Q,R,Qf,0.,x0,xf; kwargs...)
    else
        throw(ArgumentError())
    end
end
#
function copy(obj::ConstrainedObjective)
    ConstrainedObjective(copy(obj.Q),copy(obj.R),copy(obj.Qf),copy(obj.tf),copy(obj.x0),copy(obj.xf),
        u_min=copy(obj.u_min), u_max=copy(obj.u_max), x_min=copy(obj.x_min), x_max=copy(obj.x_max),
        gs_custom=obj.gs_custom, gc_custom=obj.gc_custom, hs_custom=obj.hs_custom, hc_custom=obj.hc_custom,
        use_terminal_constraint=obj.use_terminal_constraint)
end

"""
$(SIGNATURES)
    Construct a ConstrainedObjective from an UnconstrainedObjective
"""
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
    Q=obj.Q, R=obj.R, Qf=obj.Qf, tf=obj.tf, x0=obj.x0, xf=obj.xf,
    u_min=obj.u_min, u_max=obj.u_max, x_min=obj.x_min, x_max=obj.x_max,
    gs_custom=obj.gs_custom, gc_custom=obj.gc_custom, hs_custom=obj.hs_custom, hc_custom=obj.hc_custom,
    use_terminal_constraint=obj.use_terminal_constraint)

    ConstrainedObjective(Q,R,Qf,tf,x0,xf,
        u_min=u_min, u_max=u_max,
        x_min=x_min, x_max=x_max,
        gs_custom=gs_custom, gc_custom=gc_custom, hs_custom=hs_custom, hc_custom=hc_custom,
        use_terminal_constraint=use_terminal_constraint)
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
