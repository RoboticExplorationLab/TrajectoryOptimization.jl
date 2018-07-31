include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
using Plots
using BenchmarkTools


solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
solver.opts.verbose = true

U = ones(solver.model.m, solver.N-1)
@time x1,u1 = iLQR.solve(solver,U)
# @time xc,uc = iLQR.solve_al(solver,U)
# @btime xc,uc = iLQR.solve_al(solver,U)


# In place dynamics
opt = iLQR.SolverOptions()
opt.inplace_dynamics = true
opt.verbose = true
model! = iLQR.Model(Dynamics.pendulum_dynamics!,2,1)
obj_uncon = Dynamics.pendulum[2]


# Unconstrained problem
solver = iLQR.Solver(model!, obj_uncon,dt=0.1,opts=opt)
@time xc, uc = iLQR.solve(solver, U)
solver.opts.verbose = false
@btime iLQR.solve(solver, U)



# Constrained problem
obj = iLQR.ConstrainedObjective(obj_uncon, u_min=-2, u_max=2)
solver! = iLQR.Solver(model!,obj,dt=0.1,opts=opt)
# solver!.obj.Qf .*= 0
@time xc, uc = iLQR.solve_al(solver!,U)
@btime iLQR.solve_al(solver!,U)
@profiler xc, uc = iLQR.solve_al(solver!,U)

plot(xc',label=["pos (constrained)", "vel (constrained)"], color=[:red :blue])
plot!(x1',label=["pos (constrained)" "vel (constrained)"], color=[:red :blue], width=2)
plot(uc',label="constrained")
plot!(u1',label="unconstrained")
