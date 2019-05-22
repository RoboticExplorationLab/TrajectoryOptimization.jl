using TrajectoryOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using SparseArrays
using ForwardDiff
using Logging

disable_logging(Logging.Debug)
@testset "Logging" begin
    include("logger_tests.jl")
end
disable_logging(Logging.Info)

@testset "Constraints" begin
    include("constraint_tests.jl")
end
@testset "Model" begin
    include("model_tests.jl")
end
@testset "Modified Model" begin
    include("modified_model_tests.jl")
end
@testset "Problems" begin
    include("problem_tests.jl")
end
@testset "Square Root Backward Pass" begin
    include("sqrt_bp_tests.jl")
end
@testset "Infeasible State Trajectory Initialization" begin
    include("infeasible_tests.jl")
end
@testset "Minimum Time" begin
    include("minimum_time_tests.jl")
end

# Systems
@testset "Pendulum" begin
    include("pendulum_tests.jl")
end
@testset "Car" begin
    include("car_tests.jl")
end
@testset "Quadrotor" begin
    include("quadrotor_tests.jl")
end

# Direct Methods
@testset "Ipopt" begin
    include("dircol_test.jl")
end
disable_logging(Logging.Debug)
