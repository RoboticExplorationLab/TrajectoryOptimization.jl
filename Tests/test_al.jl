include("../iLQR.jl")
include("../dynamics.jl")
using iLQR
using Dynamics
using Plots

solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
solver.opts.verbose = true

U = ones(solver.model.m, solver.N-1)
@time x1,u1 = iLQR.solve(solver,U)

@time xc,uc = iLQR.solve_al(solver,U)
plot(xc',label=["pos (constrained)", "vel (constrained)"], color=[:red :blue])
plot!(x1',label=["pos (constrained)" "vel (constrained)"], color=[:red :blue], width=2)
plot(uc',label="constrained")
plot!(u1',label="unconstrained")
