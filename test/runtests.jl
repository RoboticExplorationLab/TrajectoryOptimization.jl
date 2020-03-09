using TrajectoryOptimization
using Distributions
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using Ipopt
using Logging
using ForwardDiff
using RobotZoo
using Dynamics
using DifferentialRotations
const TO = TrajectoryOptimization

@testset "Logging" begin
    include("logger_tests.jl")
end

@testset "Full Solves" begin
    include("car_tests.jl")
    TEST_TIME = false  # don't test timing results
    include("benchmark_solves.jl")
end

@testset "Solvers" begin
    include("solver_options.jl")
end

# @testset "Rotations" begin
#     include("rotations_tests.jl")
#     include("retraction_maps.jl")
# end

# @testset "Controllers" begin
#     include("controllers_test.jl")
# end
