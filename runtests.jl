include("iLQR.jl")
using Base.Test
using BenchmarkTools
using iLQR

@testset "iLQR" begin
    @testset "SimplePendulum" begin
    # initialization
    include("simple_pendulum.jl")
    iLQR.solve(solver)
    x,u = iLQR.solve(solver)
    println(x[:,end], xf)
    @test isapprox(x[:,end], xf, atol=1e-2)
    end

    @testset "Solver" begin
    include("simple_pendulum.jl")

    # initialization methods 
    @test_nowarn iLQR.Solver(model, obj)
    @test_nowarn iLQR.Solver(model, obj, dt=0.01)
    @test_nowarn iLQR.Solver(model, obj, iLQR.rk4)
    @test_nowarn iLQR.Solver(model, obj, iLQR.midpoint)
    @test_nowarn iLQR.Solver(model, obj, iLQR.midpoint, dt = 0.1)
    end
end
