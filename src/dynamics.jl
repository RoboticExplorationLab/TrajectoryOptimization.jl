module Dynamics

# using TrajectoryOptimization: Model, UncertainModel, Trajectory, Constraint, Equality, Inequality, Problem
using TrajectoryOptimization: Equality, Inequality
using TrajectoryOptimization
# using RigidBodyDynamics
using LinearAlgebra
using DocStringExtensions
using StaticArrays
using Parameters
import TrajectoryOptimization: dynamics, AbstractModel, RigidBody, state_diff, state_diff_jacobian,
    state_diff_size, orientation

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

include("../dynamics/testrigidbody.jl")
include("../dynamics/pendulum.jl")
# include("../dynamics/doublependulum.jl")
# include("../dynamics/acrobot.jl")
# include("../dynamics/ballonbeam.jl")
include("../dynamics/cartpole.jl")
include("../dynamics/quadrotor.jl")
# include("../dynamics/quadrotor_euler.jl")
# include("../dynamics/kuka.jl")
include("../dynamics/double_integrator.jl")
include("../dynamics/car.jl")
include("../dynamics/rigidbody.jl")
include("../dynamics/satellite.jl")
include("../dynamics/yak_plane.jl")
include("../dynamics/noisy_rigidbody.jl")
include("../dynamics/freebody.jl")

end
