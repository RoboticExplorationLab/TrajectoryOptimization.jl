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
@testset "Costs" begin
    include("cost_tests.jl")
end
@testset "Utils" begin
    include("test_utils.jl")
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
@testset "DIRCOL" begin
    include("dircol_test.jl")
end
disable_logging(Logging.Debug)

#= Tests needed
* Generic Cost
* Expansion multiply and copy
* Quad Cost constructor exceptions
* AugmentedLagrangianObjective constructors
* Sqrt bp exceptions
* Dynamics module?
* iLQR not inplace solves
* iLQR live plotting
* feedforward gradient
* convergence criteria (max iters, dJ zero, max reg)
* infeasible mintime
* logger: trim entry
* logger: print header
* evaluate continuous dynamics
* simultaneous jacobian! function in model
* jacobian! on continuous dynamics for entire trajectory
* model from urdf string (normal and underactuated)
* wrap_inplace
* Objective from cost and terminal cost
* Objective from cost trajectory and terminal cost
* Pass in costfuntion directly to problem
* _validate_time
* not inplace rollout
* copy iLQR solver
=#
