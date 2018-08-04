using TrajectoryOptimization
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# include("../iLQR.jl")
# include("../dynamics.jl")
using BenchmarkTools
# using iLQR


@testset "Simple Pendulum" begin
    include("simple_pendulum.jl")
end
@testset "Constrained Objective" begin
    include("objective_tests.jl")
end


"""
# NEEDED TESTS:
- integration schemes
- in place dynamics
- infeasible start
- constraint generator
- state inequality constraints
- asymmetric bounds (one bounded, another not)
- all solve methods
"""
