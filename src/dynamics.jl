module Dynamics

using TrajectoryOptimization: Model, UncertainModel, Trajectory
using RigidBodyDynamics
using LinearAlgebra

export
    pendulum_model,
    pendulum_model_uncertain,
    doublependulum_model,
    cartpole_model,
    cartpole_model_urdf,
    ballonbeam_model,
    acrobot_model,
    quadrotor_model,
    kuka_model,
    doubleintegrator_model,
    car_model

include("../dynamics/pendulum.jl")
include("../dynamics/doublependulum.jl")
include("../dynamics/acrobot.jl")
include("../dynamics/ballonbeam.jl")
include("../dynamics/cartpole.jl")
include("../dynamics/quadrotor.jl")
include("../dynamics/kuka.jl")
include("../dynamics/double_integrator.jl")
include("../dynamics/car.jl")

end
