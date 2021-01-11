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
Random.seed!(1)

include("test_models.jl")

@testset "Costs" begin
    include("cost_tests.jl")
    include("objective_tests.jl")
    include("nlcosts.jl")
end

@testset "Constraints" begin
    include("constraint_tests.jl")
    include("dynamics_constraints.jl")
    include("constraint_list.jl")
    include("constraint_sets.jl")
end

@testset "Problems" begin
    include("problems_tests.jl")
end

# @testset "Utils" begin
# end

@testset "NLP" begin
    include("nlp_tests.jl")
    include("moi_test.jl")
end

using NBInclude
@testset "Examples" begin
    @nbinclude(joinpath(@__DIR__, "..", "examples", "Internal API.ipynb"))
    # @test_nowarn include(joinpath(@__DIR__, "..", "examples", "quickstart.jl"))
    # @nbinclude(joinpath(@__DIR__, "..", "examples", "Cartpole.ipynb"); softscope=true)
    # @nbinclude(joinpath(@__DIR__, "..", "examples", "Quadrotor.ipynb"); softscope=true)
end