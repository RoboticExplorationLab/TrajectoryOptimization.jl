using TrajectoryOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using Ipopt
const TO = TrajectoryOptimization

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
