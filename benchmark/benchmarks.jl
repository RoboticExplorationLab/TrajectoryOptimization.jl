using TrajectoryOptimization
using BenchmarkTools
using Dates
using Logging
using LinearAlgebra
import TrajectoryOptimization: trim_entry
import BenchmarkTools: prettytime, prettymemory, prettydiff, prettypercent
using Formatting

const paramsfile = "benchmark/benchmark_params.json"
const histfile = "benchmark/benchmark_history.json"

include("benchmark_methods.jl")
# include("dubinscar_benchmarks.jl")
# include("pendulum_benchmarks.jl")
include("ilqr_benchmarks.jl")


loadparams!(SUITE, BenchmarkTools.load(paramsfile)[1])
# results = run(suite, verbose = true)
