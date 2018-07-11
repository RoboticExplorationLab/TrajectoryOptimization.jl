using iLQR
using BenchmarkTools

include("simple_pendulum.jl")
solver = iLQR.Solver(model, obj, dt=0.01)
@btime iLQR.solve(solver)