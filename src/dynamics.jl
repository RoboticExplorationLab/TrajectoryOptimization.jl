module Dynamics

using TrajectoryOptimization: Model, UncertainModel, Trajectory, Constraint, Equality, Inequality, Problem
using TrajectoryOptimization
using RigidBodyDynamics
using LinearAlgebra
using DocStringExtensions
using StaticArrays
import TrajectoryOptimization: dynamics, AbstractModel 

export
    pendulum,
    doublependulum,
    cartpole,
    cartpole_urdf,
    ballonbeam,
    acrobot_model,
    quadrotor,
    quadrotor_euler,
    kuka,
    doubleintegrator,
    car

include("../dynamics/pendulum.jl")
include("../dynamics/doublependulum.jl")
include("../dynamics/acrobot.jl")
include("../dynamics/ballonbeam.jl")
include("../dynamics/cartpole.jl")
include("../dynamics/quadrotor.jl")
include("../dynamics/quadrotor_euler.jl")
include("../dynamics/kuka.jl")
include("../dynamics/double_integrator.jl")
include("../dynamics/car.jl")

end
