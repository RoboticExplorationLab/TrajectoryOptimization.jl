# Tests
# include("ilqr.jl")
# include("pendulum_simple.jl")
# include("pendulum_simple_v2.jl")
# using Pendulum
using BenchmarkTools
using RigidBodyDynamics
using ForwardDiff

x0 = [1;2]
u = 1
dt = 0.1
X = [x0;u;dt]

## Double Pendulum
q0 = [1.;2.]
qd0 = [0.;0.]
u = [1.;1.]
dt = 0.1
X = [q0;qd0;u;dt]


dir = Pkg.dir("DynamicWalking2018")
urdf = joinpath(dir,"notebooks","data","doublependulum.urdf")

mech = parse_urdf(Float64, urdf)
state = MechanismState(mech)

result = DynamicsResult(mech)
dynamics!(result, state, [0,1])

@show qdd = parent(result.v̇)
result.massmatrix.uplo

const q = configuration(state)
const v = velocity(state)
momentum(state)

function mom(v::AbstractVector{T}) where T
    # create a `MechanismState` that can handle the element type of `v` (which will be some `ForwardDiff.Dual`):
    state = MechanismState{T}(mech)

    # set the state variables:
    set_configuration!(state, q)
    set_velocity!(state, v)

    # return momentum converted to an `Array` (as this is the format that ForwardDiff expects)
    Array(momentum(state))
end

function dyn_wrapper(v::AbstractVector{T}) where T
    # q = X[1:2]
    # qd = X[3:4]
    # u = X[5:6]
    state = MechanismState{T}(mech)
    set_configuration!(state, q)
    set_velocity!(state, v)
    # dynamics!(result, state, u)
    # qdd = Array(result.v̇)

    Xd = zeros(size(X))
    # Xd[1:2] = qd
    # Xd[3:4] = qdd
    return qdd
end

qdd = mom(qd0)
ForwardDiff.jacobian(mom, qd0)
Xd = dyn_wrapper(qd0)
Df = ForwardDiff.jacobian(dyn_wrapper, qd0)


function myfun(x)
    function anotherfun(x)
        x+1
    end
    anotherfun(x)+1
end

myfun(2)
