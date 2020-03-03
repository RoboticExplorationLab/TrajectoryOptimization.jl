using BenchmarkTools
using PkgBenchmark
using LinearAlgebra
using StaticArrays
using TrajectoryOptimization
using TrajOptPlots
using Plots
import TrajectoryOptimization.AbstractSolver
const TO = TrajectoryOptimization

const paramspath = joinpath(@__DIR__,"params.json")
const suite = BenchmarkGroup()

function benchmarkable_solve!(solver; samples=10, evals=10)
    Z0 = deepcopy(get_trajectory(solver))
    solver.opts.verbose = false
    b = @benchmarkable begin
        initial_trajectory!($solver,$Z0)
        solve!($solver)
    end samples=samples evals=evals
    return b
end

# ALTRO
const altro = BenchmarkGroup(["constrained"])
altro["double_int"]    = benchmarkable_solve!(ALTROSolver(Problems.DoubleIntegrator()...))
altro["pendulum"]      = benchmarkable_solve!(ALTROSolver(Problems.Pendulum()...))
altro["cartpole"]      = benchmarkable_solve!(ALTROSolver(Problems.Cartpole()...))
altro["parallel_park"] = benchmarkable_solve!(ALTROSolver(Problems.DubinsCar(:parallel_park)...))
altro["3obs"]          = benchmarkable_solve!(ALTROSolver(Problems.DubinsCar(:three_obstacles)...))
altro["escape"]        = benchmarkable_solve!(ALTROSolver(Problems.DubinsCar(:escape)...,
    infeasible=true, R_inf=0.1))
suite["ALTRO"] = altro

# iLQR
const ilqr = BenchmarkGroup(["unconstrained"])
ilqr["double_int"]    = benchmarkable_solve!(iLQRSolver(Problems.DoubleIntegrator()...))
ilqr["pendulum"]      = benchmarkable_solve!(iLQRSolver(Problems.Pendulum()...))
ilqr["cartpole"]      = benchmarkable_solve!(iLQRSolver(Problems.Cartpole()...))
ilqr["parallel_park"] = benchmarkable_solve!(iLQRSolver(Problems.DubinsCar(:parallel_park)...))
suite["iLQR"] = ilqr

SUITE = suite
