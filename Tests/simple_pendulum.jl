include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
using Base.Test

# Set up models and objective
model,obj = Dynamics.pendulum
model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = iLQR.ConstrainedObjective(obj, u_min=-2, u_max=2) # constrained objective

### UNCONSTRAINED ###
# rk4
solver = iLQR.Solver(model,obj,dt=0.1)
U = ones(solver.model.m, solver.N-1)
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


# midpoint
solver = iLQR.Solver(model,obj,integration=:midpoint,dt=0.1)
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


### CONSTRAINED ###
# rk4
solver = iLQR.Solver(model,obj_c,dt=0.1)
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2


# midpoint
solver = iLQR.Solver(model,obj_c,dt=0.1)
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2



### In-place dynamics ###
# Unconstrained
opts = iLQR.SolverOptions()
opts.inplace_dynamics = true
solver = iLQR.Solver(model!,obj,dt=0.1,opts=opts)
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@btime iLQR.solve(solver,U)

# Constrained
solver = iLQR.Solver(model!,obj_c,dt=0.1,opts=opts)
results = iLQR.solve(solver,U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
@btime iLQR.solve(solver,U)

# Constrained - midpoint
solver = iLQR.Solver(model!,obj_c, integration=:midpoint, dt=0.1, opts=opts)
results = iLQR.solve(solver,U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
@btime iLQR.solve(solver,U)
