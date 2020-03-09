using Test, BenchmarkTools

function num_allocs(model)
    dt = 0.1
    x, u = rand(model)
    n,m = size(model)
    z = KnotPoint(x, u, dt)
    ∇c  = zeros(n,n+m)
    ∇cd = zeros(n,n+m+1)
    dynamics(model, x, u)
    jacobian!(∇c, model, z)
    discrete_dynamics(RK3, model, x, u, z.t, dt)
    discrete_jacobian!(RK3, ∇cd, model, z)
    allocs  = @allocated dynamics(model, x, u)
    allocs += @allocated jacobian!(∇c, model, z)
    allocs += @allocated discrete_dynamics(RK3, model, x, u, z.t, dt)
    allocs += @allocated discrete_jacobian!(RK3, ∇cd, model, z)
end

# Double Integrator
dim = 3
di = RobotZoo.DoubleIntegrator(dim)
n,m = size(di)
@test (n,m) == (6,3)
x,u = rand(di)
@test num_allocs(di) == 0

# Pendulum
pend = RobotZoo.Pendulum()
@test size(pend) == (2,1)
@test num_allocs(pend) == 0

# Car
car = RobotZoo.DubinsCar()
@test size(car) == (3,2)
@test num_allocs(car) == 0

# Cartpole
cartpole = RobotZoo.Cartpole()
@test size(cartpole) == (4,1)
@test num_allocs(cartpole) == 0

# Quadrotor
quad = RobotZoo.Quadrotor()
@test size(quad) == (13,4)
@test num_allocs(cartpole) == 0


# Infeasible
model = RobotZoo.DubinsCar()
inf = TO.InfeasibleModel(model)
@test inf._u == 1:2
@test inf._ui == 3:5
@test size(inf) == (3,5)
x = rand(inf)[1]
u0 = @SVector rand(2)
ui = @SVector rand(3)
u = [u0; ui]
dt = 0.1
@test length(rand(inf)[2]) == 5
@test_throws ErrorException dynamics(inf, x, u)
@test discrete_dynamics(RK3, inf, KnotPoint(x, u, dt)) ==
    (discrete_dynamics(RK3, model, x, u0, 0.0, dt) + ui)

function inf_allocs(inf)
    x,u = rand(inf)
    n,m = size(inf)
    dt = 0.1
    z = KnotPoint(x,u,0.1)
    # allocs = @allocated discrete_dynamics(RK3, inf, x, u, dt)
    ∇c = zeros(n,n+m+1)
    allocs = @allocated discrete_dynamics(RK3, inf, z)
    allocs += @allocated discrete_jacobian!(RK3, ∇c, inf, z)
end
@test inf_allocs(inf) == 0


# Test other functions
car = RobotZoo.DubinsCar()
n,m = size(car)
@test zeros(car) == (zeros(n), zeros(m))
@test zeros(Int,car)[1] isa SVector{n,Int}
@test fill(car,0.1) == (fill(0.1,n), fill(0.1,m))
@test ones(Float32,car)[2] isa SVector{m,Float32}

# Test default integrator
x,u = rand(car)
z = KnotPoint(x,u,dt)
@test discrete_dynamics(car, z) == discrete_dynamics(RK3, car, z)
