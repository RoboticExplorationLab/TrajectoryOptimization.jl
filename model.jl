using RigidBodyDynamics
using ForwardDiff

struct Model
    f::Function # continuous dynamics (ie, differential equation)
    n::Int # number of states
    m::Int # number of controls

    function Model(f::Function, n::Int64, m::Int64)
        # construct a model from an explicit differential equation
        new(f,n,m)
    end

    function Model(mech::Mechanism)
        # fully actuated
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

        function fc(x,u)
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

function Model(urdf::String)
    # construct modeling using string to urdf file
    mech = parse_urdf(Float64,urdf)
    Model(mech)
end

function Model(urdf::String,torques::Array{Float64,1})
    # underactuated system (potentially)
    # construct modeling using string to urdf file
    mech = parse_urdf(Float64,urdf)
    Model(mech,torques)
end

# cost function
abstract type Objective end

mutable struct UnconstrainedObjective <: Objective
    Q::Array{Float64,2}
    R::Array{Float64,2}
    Qf::Array{Float64,2}
    tf::Float64
    x0::Array{Float64,1}
    xf::Array{Float64,1}
end

mutable struct ConstrainedObjective <: Objective
    Q::Array{Float64,2}
    R::Array{Float64,2}
    Qf::Array{Float64,2}
    tf::Float64
    x0::Array{Float64,1}
    xf::Array{Float64,1}

    # Control Constraints
    u_min::Array{Float64,1}
    u_max::Array{Float64,1}

    # State Constraints
    x_min::Array{Float64,1}
    x_max::Array{Float64,1}

    # General Stage Constraints
    cI::Function  # inequality constraints
    cE::Function  # equality constraints

    # Terminal Constraints
    use_terminal_constraint::Bool  # Use terminal state constraint (true) or terminal cost (false)
    cI_N::Function # terminal inequality constraints
    cE_N::Function # terminal equality constraints

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
        u_max, u_min = validate_bounds(u_max,u_min,m)
        x_max, x_min = validate_bounds(x_max,x_min,n)

        # Stage Constraints
        pI = 0
        pE = 0
        pI += count(isfinite, u_min)
        pI += count(isfinite, u_max)
        pI += count(isfinite, x_min)
        pI += count(isfinite, x_max)

        u0 = zeros(m)
        if ~isa(cI(x0,u0), Void)
            pI += size(cI(x0,u0),1)
        end
        if ~isa(cE(x0,u0), Void)
            pE += size(cE(x0,u0),1)
        end
        p = pI + pE


        # Terminal Constraints
        pI_N = pE_N = 0
        if ~isa(cI_N(x0), Void)
            pI_N = size(cI_N(x0),1)
        end
        if ~isa(cE_N(x0), Void)
            pE_N = size(cE_N(x0),1)
        end
        if use_terminal_constraint
            pE_N += n
        end
        p_N = pI_N + pE_N

        new(Q,R,Qf,tf,x0,xf, u_min, u_max, x_min, x_max, cI, cE, use_terminal_constraint, cI_N, cE_N, p, pI, p_N, pI_N)
    end
end

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

function ConstrainedObjective(obj::UnconstrainedObjective; kwargs...)
    ConstrainedObjective(obj.Q, obj.R, obj.Qf, obj.tf, obj.x0, obj.xf; kwargs...)
end

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

# hack to keep struct immutable
function update_objective_infeasible(obj::ConstrainedObjective,R::Array{Float64,2};
    u_min=obj.u_min, u_max=obj.u_max, x_min=obj.x_min, x_max=obj.x_max,
    cI=obj.cI, cE=obj.cE,
    use_terminal_constraint=obj.use_terminal_constraint,
    cI_N=obj.cI_N, cE_N=obj.cE_N)

    ConstrainedObjective(obj.Q,R,obj.Qf,obj.tf,obj.x0,obj.xf,
        u_min, u_max,
        x_min, x_max,
        cI, cE,
        use_terminal_constraint,
        cI_N, cE_N)
end

function count_constraints(n,m,u_max,u_min,x_max,x_min,cI,cE,
    use_terminal_constraint, cI_N, cE_N)


end

function validate_bounds(max,min,n)

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
    # if length(max) != n
    #     throw(DimensionMismatch("limit of length $(length(max)) doesn't match expected length of $n"))
    # end
    return max, min
end
