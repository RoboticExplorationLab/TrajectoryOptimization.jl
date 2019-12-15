using Test, BenchmarkTools

function num_allocs(model)
    dt = 0.1
    x, u = rand(model)
    z = KnotPoint(x, u, dt)
    dynamics(model, x, u)
    jacobian(model, z)
    discrete_dynamics(RK3, model, x, u, dt)
    discrete_jacobian(RK3, model, z)
    allocs  = @allocated dynamics(model, x, u)
    allocs += @allocated jacobian(model, z)
    allocs += @allocated discrete_dynamics(RK3, model, x, u, dt)
    allocs += @allocated discrete_jacobian(RK3, model, z)
end

# Double Integrator
dim = 3
di = Dynamics.DoubleIntegrator(dim)
n,m = size(di)
@test (n,m) == (6,3)
x,u = rand(di)
@test num_allocs(di) == 0

# Pendulum
pend = Dynamics.Pendulum()
@test size(pend) == (2,1)
@test num_allocs(pend) == 0

# Car
car = Dynamics.DubinsCar()
@test size(car) == (3,2)
@test num_allocs(car) == 0

# Cartpole
cartpole = Dynamics.Cartpole()
@test size(cartpole) == (4,1)
@test num_allocs(cartpole) == 0

# Quadrotor
quad = Dynamics.Quadrotor()
@test size(quad) == (13,4)
@test num_allocs(cartpole) == 0


# Infeasible
model = Dynamics.DubinsCar()
inf = InfeasibleModel(model)
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
@test discrete_dynamics(RK3, inf, KnotPoint(x, u, dt)) == (discrete_dynamics(RK3, model, x, u0, dt) + ui)

function inf_allocs(inf)
    x,u = rand(inf)
    dt = 0.1
    z = KnotPoint(x,u,0.1)
    # allocs = @allocated discrete_dynamics(RK3, inf, x, u, dt)
    allocs = @allocated discrete_dynamics(RK3, inf, z)
    allocs += @allocated discrete_jacobian(RK3, inf, z)
end
@test inf_allocs(inf) == 0
