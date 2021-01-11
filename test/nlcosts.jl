# using Test
# using StaticArrays
# using TrajectoryOptimization
# using RobotZoo: Cartpole
# using ForwardDiff
# using BenchmarkTools
# using RobotDynamics
# using DiffResults
# using FiniteDiff
# using LinearAlgebra
import TrajectoryOptimization: stage_cost, CostFunction, Expansion
# const TO = TrajectoryOptimization

##
struct CartpoleCost{T} <: TO.CostFunction 
    Q::Vector{T}
    R::Vector{T}
end
function TO.stage_cost(cost::CartpoleCost, x, u)
    return TO.stage_cost(cost, x) + 0.5 * cost.R[1] * u[1]^2 
end
function TO.stage_cost(cost::CartpoleCost, x)
    y = x[1]
    θ = x[2]
    ydot = x[3]
    θdot = x[4]
    J = cost.Q[2] * cos(θ/2)
    J += 0.5* (cost.Q[1] * y^2 + cost.Q[3] * ydot^2 + cost.Q[4] * θdot^2)
    return J
end
TO.state_dim(::CartpoleCost) = 4
TO.control_dim(::CartpoleCost) = 1

## Initialize Model
@testset "Nonlinear Costs" begin
    model = Cartpole()
    n,m = size(model)
    x = zeros(n)
    u = zeros(m)
    cst = CartpoleCost([1,2,3,4.], [2.])
    @test TO.stage_cost(cst, x, u) ≈ 2.0
    x[2] = π
    @test TO.stage_cost(cst, x) ≈ 0 atol = 10*eps()
    x,u = rand(model)
    z = KnotPoint(x,u,0.1)
    zterm = KnotPoint(x,u,0.0)

    ## Test ForwardDiff
    TO.diffmethod(::CartpoleCost) = RobotDynamics.ForwardAD()
    E = Expansion{Float64}(n,m)
    TO.gradient!(E, cst, zterm)
    TO.hessian!(E, cst, zterm)
    TO.gradient!(E, cst, z)
    TO.hessian!(E, cst, z)
    @test (@allocated TO.gradient!(E, cst, zterm)) == 0
    @test (@allocated TO.hessian!(E, cst, zterm)) == 0
    @test (@allocated TO.gradient!(E, cst, z)) == 0
    @test (@allocated TO.hessian!(E, cst, z)) == 0
    @test E.xx ≈ Diagonal([1, -0.5*cos(x[2]/2), 3, 4])
    @test E.uu ≈ Diagonal([2])
    @test E.x ≈ [1*x[1], -sin(x[2]/2), 3*x[3], 4*x[4]]
    @test E.u ≈ 2*u

    TO.diffmethod(::CartpoleCost) = RobotDynamics.FiniteDifference()
    E2 = Expansion{Float64}(n,m)
    TO.gradient!(E2, cst, zterm)
    TO.hessian!(E2, cst, zterm)
    TO.gradient!(E2, cst, z)
    TO.hessian!(E2, cst, z)

    cache = TO.ExpansionCache(cst)
    TO.gradient!(E2, cst, zterm, cache)
    TO.hessian!(E2, cst, zterm, cache)
    TO.gradient!(E2, cst, z, cache)
    TO.hessian!(E2, cst, z, cache)

    @test (@allocated TO.gradient!(E2, cst, zterm, cache)) == 0
    @test (@allocated TO.hessian!(E2, cst, zterm, cache)) == 0
    @test (@allocated TO.gradient!(E2, cst, z, cache)) == 0
    @test (@allocated TO.hessian!(E2, cst, z, cache)) == 0

    @test norm(E2.hess- E.hess) < 1e-6
    @test norm(E2.grad- E.grad) < 1e-6

    ## Test expansion of trajectory
    N = 11
    obj = Objective(cst, N)
    X = [rand(n) for k = 1:N]
    U = [rand(m) for k = 1:N-1]
    Z = Traj(X,U,fill(0.1,N))

    TO.diffmethod(::CartpoleCost) = RobotDynamics.ForwardAD()
    E0 = [Expansion{Float64}(n,m) for k = 1:N]
    cache = TO.ExpansionCache(obj)
    TO.cost_expansion!(E0, obj, Z, cache, init=true, rezero=true)
    # @test (@allocated TO.cost_expansion!(E0, obj, Z, cache, init=true, rezero=true)) == 0

    TO.diffmethod(::CartpoleCost) = RobotDynamics.FiniteDifference()
    E1 = [Expansion{Float64}(n,m) for k = 1:N]
    cache = TO.ExpansionCache(obj)
    TO.cost_expansion!(E1, obj, Z, cache, init=true, rezero=true)
    # @test (@allocated TO.cost_expansion!(E1, obj, Z, cache, init=true, rezero=true)) == 0
end