include("../iLQR.jl")
include("../dynamics.jl")
using Base.Test
using BenchmarkTools
using iLQR


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
