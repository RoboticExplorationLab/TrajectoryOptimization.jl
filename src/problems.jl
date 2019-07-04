"""
    Problems
        Collection of trajectory optimization problems
"""
module Problems

using TrajectoryOptimization
using RigidBodyDynamics
using LinearAlgebra
using ForwardDiff

include("../problems/pendulum.jl")
include("../problems/parallel_park.jl")
include("../problems/car_escape.jl")
include("../problems/quadrotor_maze.jl")
include("../problems/kuka_obstacles.jl")

export
    pendulum_problem,
    box_parallel_park_problem,
    box_parallel_park_min_time_problem,
    car_escape_problem,
    quadrotor_maze_problem

end
