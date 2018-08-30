using TrajectoryOptimization
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using BenchmarkTools

@testset "Simple Pendulum" begin
    include("simple_pendulum.jl")
end
@testset "Constrained Objective" begin
    include("objective_tests.jl")
end
@testset "Results" begin
    include("results_tests.jl")
end
@testset "Square Root Method" begin
    include("sqrt_method_tests.jl")
end
# @testset "First Order Hold" begin
#     include("foh_tests.jl")
# end
@testset "Infeasible Start" begin
    include("infeasible_start_tests.jl")
end
@testset "Direct Collocation" begin
    include("dircol_test.jl")
end

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
