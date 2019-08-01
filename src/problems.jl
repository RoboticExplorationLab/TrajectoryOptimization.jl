"""
    Problems
        Collection of trajectory optimization problems
"""
module Problems

using TrajectoryOptimization
using RigidBodyDynamics
using LinearAlgebra
using ForwardDiff
using Plots
using Random

include("../problems/doubleintegrator.jl")
include("../problems/pendulum.jl")
include("../problems/parallel_park.jl")
include("../problems/cartpole.jl")
# include("../problems/doublependulum.jl")
# include("../problems/acrobot.jl")
include("../problems/car_escape.jl")
include("../problems/car_3obs.jl")
include("../problems/quadrotor.jl")
include("../problems/quadrotor_maze.jl")
# include("../problems/kuka_obstacles.jl")

export
    doubleintegrator,
    pendulum,
    parallel_park,
    cartpole,
    doublependulum,
    acrobot,
    car_escape,
    car_3obs,
    quadrotor,
    quadrotor_maze,
    kuka_obstacles

export
    plot_escape,
    plot_car_3obj,
    quadrotor_maze_objects,
    kuka_obstacles_objects

end
