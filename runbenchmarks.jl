using iLQR
using BenchmarkTools

include("simple_pendulum.jl")
solver = Solver(model, obj, 0.01)
@btime iLQR.solve(solver)