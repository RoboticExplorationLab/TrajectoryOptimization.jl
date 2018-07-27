include("../iLQR.jl")
include("../dynamics.jl")
import iLQR, iLQR.solve
using BenchmarkTools
using Dynamics

solver = iLQR.Solver(Dynamics.pendulum...,dt=0.1)
# @btime iLQR.solve(solver)
# println(x[:,end])

solver = iLQR.Solver(Dynamics.doublependulum...,dt=0.1)
@time x,u = solve(solver)
