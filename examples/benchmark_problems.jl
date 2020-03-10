using TrajectoryOptimization
using BenchmarkTools
using TrajOptPlots
using Plots
using LinearAlgebra
using MeshCat

vis = Visualizer()
open(vis)

solver = DIRCOLSolver(Problems.Quadrotor()...,integration=HermiteSimpson)
solve!(solver)
cost(solver)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Double Integrator
solver = ALTROSolver(Problems.DoubleIntegrator()...)
benchmark_solve!(solver)
iterations(solver) # 8
plot(solver)

# Pendulum
solver = ALTROSolver(Problems.Pendulum()...)
benchmark_solve!(solver)
iterations(solver) # 19
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Cartpole
solver = ALTROSolver(Problems.Cartpole()...)
benchmark_solve!(solver)
iterations(solver) # 40
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Acrobot
solver = ALTROSolver(Problems.Acrobot()...)
benchmark_solve!(solver)
iterations(solver) # 50
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Parallel Park
solver = ALTROSolver(Problems.DubinsCar(:parallel_park)...)
benchmark_solve!(solver)
iterations(solver)  # 13
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Three Obstacles
solver = ALTROSolver(Problems.DubinsCar(:three_obstacles)...)
benchmark_solve!(solver)
iterations(solver) # 20
delete!(vis)
add_cylinders!(vis, solver, robot_radius=get_model(solver).radius)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Escape
solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
benchmark_solve!(solver, samples=1, evals=1)
iterations(solver) # 13
delete!(vis)
set_mesh!(vis, get_model(solver))
add_cylinders!(vis, solver, robot_radius=get_model(solver).model.radius, height=0.2)
visualize!(vis, solver)
A = rand(10,10)
typeof(view(A,1:0,1:0))

# Zig-zag
solver = ALTROSolver(Problems.Quadrotor(:zigzag)...)
benchmark_solve!(solver)
iterations(solver) # 15
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)

# Barrell Roll
solver = ALTROSolver(Problems.YakProblems()...)
benchmark_solve!(solver)
iterations(solver) # 17
max_violation(solver)
delete!(vis)
set_mesh!(vis, get_model(solver))
visualize!(vis, solver)
