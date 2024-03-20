using TrajectoryOptimization
using Test
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using ForwardDiff
using RobotDynamics
using BenchmarkTools
using FiniteDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics
Random.seed!(1)

include("test_models.jl")

const run_alloc_tests = !haskey(ENV, "CI")
##

@testset "Costs" begin
    include("cost_tests.jl")
    include("objective_tests.jl")
    include("nlcosts.jl")
end

@testset "Constraints" begin
    include("constraint_tests.jl")
    include("constraint_list.jl")
    include("cone_tests.jl")
end

@testset "Problems" begin
    include("problems_tests.jl")
    include("hybrid_dynamics_model.jl")
end

@testset "Examples" begin
    @testset "Quickstart" begin
        include(joinpath(@__DIR__, "..", "examples", "quickstart.jl"))
    end
end