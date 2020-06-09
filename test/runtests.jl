using TrajectoryOptimization
using Test
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using ForwardDiff
const TO = TrajectoryOptimization


@testset "Costs" begin
    include("cost_tests.jl")
    include("objective_tests.jl")
end

@testset "Constraints" begin
    include("constraint_tests.jl")
end

# @testset "Rotations" begin
#     include("rotations_tests.jl")
#     include("retraction_maps.jl")
# end


@testset "Controllers" begin
    include("controllers_test.jl")
end
