using Test, BenchmarkTools
import TrajectoryOptimization: jacobian, discrete_dynamics, discrete_jacobian

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
dim = 2
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
