using TrajectoryOptimization
using BenchmarkTools
using Dates
using Logging
using LinearAlgebra
import TrajectoryOptimization: trim_entry
import BenchmarkTools: prettytime, prettymemory, prettydiff, prettypercent
using Formatting

include("benchmark_methods.jl")
include("dubinscar_benchmarks.jl")
include("pendulum_benchmarks.jl")

const paramsfile = "benchmark/benchmark_params.json"
const histfile = "benchmark/benchmark_history.json"



date = today()
suite = BenchmarkGroup([date])
stats = BenchmarkGroup([date])

suite["dubinscar"], stats["dubinscar"] = dubinscar_benchmarks()
suite["pendulum"], stats["pendulum"] = pendulum_benchmarks()

loadparams!(suite, BenchmarkTools.load(paramsfile)[1])
# results = run(suite, verbose = true)

# Rename for PkgBenchmark.jl
SUITE = suite

# baseline_comparison(results, stats)
