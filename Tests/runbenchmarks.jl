include("../iLQR.jl")
include("../dynamics.jl")
import iLQR, iLQR.solve
using BenchmarkTools
using Dynamics

# Unconstrained
model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1)
obj = Dynamics.pendulum[2]
obj = iLQR.ConstrainedObjective(obj, u_min=-2, u_max=2)


opts = iLQR.SolverOptions()
opts.inplace_dynamics = true
opts.benchmark = true
opts.verbose = false
solver = iLQR.Solver(model!,obj,dt=0.1,opts=opts)
U = ones(solver.model.m, solver.N-1)
@time results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3
