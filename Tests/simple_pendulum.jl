using Dynamics
using Base.Test

# Unconstrained with square root
model,obj = Dynamics.pendulum
solver = iLQR.Solver(model,obj,dt=0.1)
U = ones(solver.model.m, solver.N-1)
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

# Unconstrained with square root
solver.opts.square_root = true
results = iLQR.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

# Constrained
obj_c = iLQR.ConstrainedObjective(obj_uncon, u_min=-2, u_max=2)
solver = iLQR.Solver(model,obj_c,dt=0.1)
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

# Constrained with Square Root
solver.opts.square_root = true
results_c = iLQR.solve(solver, U)
max_c = iLQR.max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
