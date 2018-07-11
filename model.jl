struct Model
    f::Function
    n::Int
    m::Int

    function Model(f::Function, n::Int64, m::Int64)
        new(f,n,m)
    end

    function Model(mech::Mechanism)
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

            [qd; Array(mass_matrix(state))\u - Array(mass_matrix(state))\Array(dynamics_bias(state))]
        end
        new(fc, n, m)
    end
end

function Model(urdf::String)
    mech = parse_urdf(Float64,urdf)
    Model(mech)
end

struct Objective
    Q::Array{Float64,2}
    R::Array{Float64,2}
    Qf::Array{Float64,2}
    tf::Float64
    x0::Array{Float64,1}
    xf::Array{Float64,1}
end