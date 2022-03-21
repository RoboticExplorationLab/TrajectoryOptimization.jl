import TrajectoryOptimization: stage_cost, CostFunction

##
RD.@autodiff struct CartpoleCost{T} <: TO.CostFunction 
    Q::Vector{T}
    R::Vector{T}
end
function RD.evaluate(cost::CartpoleCost, x, u)
    y = x[1]
    θ = x[2]
    ydot = x[3]
    θdot = x[4]
    J = cost.Q[2] * cos(θ/2)
    J += 0.5* (cost.Q[1] * y^2 + cost.Q[3] * ydot^2 + cost.Q[4] * θdot^2)
    J += 0.5 * cost.R[1] * u[1]^2 
    return J
end
TO.state_dim(::CartpoleCost) = 4
TO.control_dim(::CartpoleCost) = 1

## Initialize Model
@testset "Nonlinear Costs" begin
    model = Cartpole()
    n,m = RD.dims(model)
    x = zeros(n)
    u = zeros(m)
    cst = CartpoleCost{Float64}([1,2,3,4.], [2.])
    @test RD.evaluate(cst, x, u) ≈ 2.0
    x[2] = π
    x,u = rand(model)
    t,h = 1.1, 0.1
    z = KnotPoint(x,u,t,h)
    zterm = KnotPoint(x,u,t,0.0)
    ix = 1:n
    iu = n .+ (1:m)

    ## Test ForwardDiff
    grad = zeros(n + m)
    hess = zeros(n + m, n + m)
    RD.gradient!(RD.StaticReturn(), RD.ForwardAD(), cst, grad, z)
    RD.hessian!(RD.StaticReturn(), RD.ForwardAD(), cst, hess, z)
    @test hess[ix,ix] ≈ Diagonal([1, -0.5*cos(x[2]/2), 3, 4])
    @test hess[iu,iu] ≈ Diagonal([2])
    @test grad[ix] ≈ [1*x[1], -sin(x[2]/2), 3*x[3], 4*x[4]]
    @test grad[iu] ≈ 2*u
end