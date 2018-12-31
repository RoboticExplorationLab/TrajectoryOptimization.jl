module Dynamics

using TrajectoryOptimization: Model, UnconstrainedObjective, ConstrainedObjective
using RigidBodyDynamics
using LinearAlgebra

export
    pendulum,
    doublependulum,
    cartpole,
    ballonbeam,
    acrobot

include("../dynamics/pendulum.jl")
include("../dynamics/doublependulum.jl")
include("../dynamics/acrobot.jl")
include("../dynamics/ballonbeam.jl")
include("../dynamics/cartpole.jl")
include("../dynamics/dubinscar.jl")
include("../dynamics/quadrotor.jl")
include("../dynamics/quadrotor_euler.jl")
include("../dynamics/kuka.jl")

end
