using BenchmarkTools
using TrajectoryOptimization
using SNOPT7, Ipopt
const TO = TrajectoryOptimization

const suite = BenchmarkGroup()

suite["block"] = BenchmarkGroup()
suite["pendulum"] = BenchmarkGroup()
suite["acrobot"] = BenchmarkGroup()
suite["cartpole"] = BenchmarkGroup()
suite["car"] = BenchmarkGroup()


function run_benchmarks!(suite::BenchmarkGroup, prob::Problem, opts::Vector{<:TO.AbstractSolverOptions})
    for opt in opts
        suite[solver_name(opt)] = @benchmarkable solve($prob, $opt)
        # @time solve(prob, opt)
    end
end

benchmarks = filter!(readdir(@__DIR__)) do file
    endswith(file, "_benchmarks.jl")
end
for file in benchmarks
    include(file)
end

block_benchmarks!(suite["block"])
pendulum_benchmarks!(suite["pendulum"])
cartpole_benchmarks!(suite["cartpole"])
car_benchmarks!(suite["car"])
# quadrotor_benchmarks!(suite["quadrotor"])

paramspath = joinpath(dirname(@__FILE__), "params.json")


if isfile(paramspath)
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(suite)
    BenchmarkTools.save(paramspath, params(suite));
end

SUITE = suite
