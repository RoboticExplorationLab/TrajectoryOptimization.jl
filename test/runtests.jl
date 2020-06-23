using TrajectoryOptimization
using Test
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using ForwardDiff
using RobotDynamics
const TO = TrajectoryOptimization

include("test_models.jl")

@testset "Costs" begin
    include("cost_tests.jl")
    include("objective_tests.jl")
end

@testset "Constraints" begin
    include("constraint_tests.jl")
    include("dynamics_constraints.jl")
    include("constraint_list.jl")
    include("constraint_sets.jl")
end
