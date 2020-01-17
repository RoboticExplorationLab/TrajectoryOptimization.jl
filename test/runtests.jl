using TrajectoryOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using Ipopt
using Logging
using ForwardDiff
const TO = TrajectoryOptimization

@testset "Logging" begin
    include("logger_tests.jl")
end

@testset "Dynamics" begin
    include("models_test.jl")
    include("dynamics_constraints.jl")
end

@testset "Full Solves" begin
    include("car_tests.jl")
end

@testset "Solvers" begin
    include("solver_options.jl")
end

@testset "Constraints" begin
    include("constraint_tests.jl")
end

@testset "Rotations" begin
    include("rotations_tests.jl")
    include("retraction_maps.jl")
end

@testset "Costs" begin
    include("cost_tests.jl")
end
