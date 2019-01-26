using TrajectoryOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using SparseArrays
using ForwardDiff
using Logging

disable_logging(Logging.Info)

@testset "Simple Pendulum" begin
    include("simple_pendulum.jl")
end
@testset "Dubins Car" begin
    include("dubins_car_tests.jl")
end
@testset "Quadrotor" begin
    include("quadrotor_tests.jl")
end
@testset "Constrained Objective" begin
    include("objective_tests.jl")
end
@testset "Results" begin
    include("results_tests.jl")
end
@testset "Jacobians" begin
    include("jacobian_tests.jl")
end
@testset "Square Root Method" begin
    include("sqrt_method_tests.jl")
end
@testset "Infeasible Start" begin
    include("infeasible_start_tests.jl")
end
@testset "Minimum Time" begin
    include("minimum_time_tests.jl")
end
# @testset "Direct Collocation" begin
#     include("dircol_test.jl")
#     include("ipopt_test.jl")
# end

disable_logging(Logging.Debug)

"""
# NEEDED TESTS:
- constraint generator

- Attempt undefined integration scheme
- All dynamics
- Custom terminal constraints in objective
- Try/catch in solve_sqrt
- UnconstrainedResults constructor
- more advanced infeasible start
"""
