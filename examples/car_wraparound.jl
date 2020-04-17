using TrajectoryOptimization
using TrajOptPlots
using RobotZoo
using LinearAlgebra
using StaticArrays
using MeshCat
using Plots

if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, RobotZoo.DubinsCar())
end

model = RobotZoo.DubinsCar()
n,m = size(model)

# Discretization
tf = 4.0  # sec
N = 101
dt = tf/(N-1)  # sec

# Initial and Final Conditions
x0 = SA[0,0,deg2rad(20)]
xf = SA[0,0,deg2rad(360-20)]

# Objective
Q = Diagonal(@SVector fill(1.0,n))
R = Diagonal(@SVector fill(1.0,m))
obj = LQRObjective(Q,R,100Q,xf,N)

# Solve
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
solve!(solver)
plot(states(solver), 3:3)
plot(solver)
visualize!(vis, solver)

J,E = TrajOptCore.build_cost_expansion(obj, model)
J[1]
