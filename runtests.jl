using Base.Test
using BenchmarkTools

@testset "SimplePendulum" begin
     # initialization
    iLQR.solve(solver)
    x,u = iLQR.solve(solver)
    
    @test isapprox(x[:,end], xf, atol=1e-2)
    
end